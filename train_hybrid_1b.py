import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Adiciona o diretório atual ao sys.path para garantir imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.conversational_dataset import ConversationalDataset
from src.tv_dsl_interpreter import TVDSLInterpreter

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"[AVISO] Falha ao importar transformers/peft: {e}")
    PEFT_TRANSFORMERS_AVAILABLE = False

class QLoRADatasetWrapper(Dataset):
    """
    Wrapper que adapta o ConversationalDataset ou o dataset JSON conversacional sintético
    para o formato de Chat Template da Hugging Face, suportando múltiplos turnos de conversa.
    """
    def __init__(self, raw_dataset, tokenizer, max_seq_len=256):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.raw_dataset)
        
    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        
        # Se for um diálogo estruturado do novo dataset conversacional JSON
        if isinstance(sample, dict) and "dialogue" in sample:
            messages = []
            has_system = False
            for turn in sample["dialogue"]:
                if turn.get("role") == "system":
                    has_system = True
                messages.append({"role": turn.get("role"), "content": turn.get("content")})
            
            if not has_system:
                messages.insert(0, {
                    "role": "system",
                    "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."
                })
            
            # Encontrar o último turno do assistente para o GRPO-RL
            assistant_turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            if assistant_turns:
                last_asst_idx = assistant_turns[-1]
                raw_target = messages[last_asst_idx]["content"]
                # A entrada de RL é o histórico antes da última resposta do assistente
                history_before = messages[:last_asst_idx]
                raw_input = json.dumps(history_before)
            else:
                raw_input = ""
                raw_target = ""
        else:
            # Fallback clássico
            raw_input = sample["raw_input"] if isinstance(sample, dict) else sample[0]
            raw_target = sample["raw_target"] if isinstance(sample, dict) else sample[1]
            raw_cot = sample["raw_cot"] if isinstance(sample, dict) else sample[2]
            
            messages = [
                {
                    "role": "system", 
                    "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."
                },
                {"role": "user", "content": raw_input},
                {"role": "assistant", "content": f"<thought>\n{raw_cot}\n</thought>\n{raw_target}"}
            ]
            
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Encontrar e desmascarar apenas blocos do assistente (para perda SFT multi-turn)
        labels = torch.full_like(input_ids, -100)
        assistant_start_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        
        i = 0
        in_assistant = False
        start_idx = -1
        
        while i < len(input_ids):
            asst_slice = input_ids[i : i + len(assistant_start_token)].tolist()
            if not in_assistant and asst_slice == assistant_start_token:
                in_assistant = True
                i += len(assistant_start_token)
                start_idx = i
                continue
                
            end_slice = input_ids[i : i + len(im_end_token)].tolist()
            if in_assistant and end_slice == im_end_token:
                in_assistant = False
                end_idx = i + len(im_end_token)
                labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
                i += len(im_end_token)
                continue
            i += 1
            
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_input": raw_input,
            "raw_target": raw_target
        }

def calculate_reward(gen_text, target_text, interpreter):
    reward = 0.0
    gen_lower = gen_text.lower().strip()
    target_lower = target_text.lower().strip()
    
    # 1. Recompensa de Formato de Pensamento CoT (XML tags)
    if "<thought>" in gen_text and "</thought>" in gen_text:
        reward += 1.0
    else:
        reward -= 0.5
        
    # 2. Recompensa de Persona (Prevenir desvio de identidade/sycophancy)
    competitor_words = ["anthropic", "claude", "openai", "chatgpt", "gemini", "google"]
    has_competitor = any(word in gen_lower for word in competitor_words)
    if has_competitor:
        # Se contiver menção de que foi criado por eles
        if "criado por anthropic" in gen_lower or "criado pela anthropic" in gen_lower or \
           "criado pela openai" in gen_lower or "sou o chatgpt" in gen_lower or \
           "sou o gemini" in gen_lower or "sou o claude" in gen_lower:
            reward -= 2.5
        else:
            reward -= 1.0
            
    if "think-vetor" in gen_lower:
        reward += 0.5
        
    # 3. Recompensa de Chamada TV-DSL
    math_symbols = ["+", "-", "*", "/", "soma", "subtraia", "multiplique", "divida"]
    has_math = any(sym in target_lower or sym in gen_lower for sym in math_symbols)
    has_dsl_tag = "[tv-dsl:" in gen_lower
    
    if has_math:
        if has_dsl_tag:
            reward += 1.5
        else:
            reward -= 1.0
            
    # 4. Recompensa de Resolução AST (Execução da TV-DSL)
    processed, modified = interpreter.process_text_stream(gen_text)
    if modified and "[result:" in processed.lower():
        reward += 1.0
        import re
        numbers = re.findall(r"\d+", target_lower)
        if numbers:
            all_match = all(num in processed.lower() for num in numbers)
            if all_match:
                reward += 2.0
            else:
                reward -= 0.5
                
    # 5. Acurácia da Resposta Final (se contiver o alvo final)
    if target_lower in gen_lower:
        reward += 1.0
        
    return reward

def run_grpo_step(model, tokenizer, batch, interpreter, device, group_size=4, lr=1e-5):
    # batch: dict contendo "raw_input" e "raw_target"
    raw_inputs = batch["raw_input"]
    raw_targets = batch["raw_target"]
    B = len(raw_inputs)
    G = group_size
    
    total_policy_loss = 0.0
    total_reward = 0.0
    
    for i in range(B):
        prompt = raw_inputs[i]
        target = raw_targets[i]
        
        try:
            # Tentar ler como histórico conversacional serializado em JSON
            history_messages = json.loads(prompt)
            messages = []
            for msg in history_messages:
                messages.append({"role": msg["role"], "content": msg["content"]})
        except Exception:
            # Fallback clássico
            messages = [
                {
                    "role": "system", 
                    "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Replicar inputs G vezes para amostragem do grupo
        input_ids_rep = inputs["input_ids"].repeat(G, 1)
        attn_mask_rep = inputs["attention_mask"].repeat(G, 1)
        
        # Gerar amostras com temperatura alta para diversidade
        model.eval()
        model.config.use_cache = True
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_rep,
                attention_mask=attn_mask_rep,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        model.config.use_cache = False
            
        # Decodificar respostas e calcular recompensas
        prompt_len = input_ids_rep.shape[1]
        rewards = []
        gen_texts = []
        
        for idx in range(G):
            gen_tokens = outputs[idx][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            gen_texts.append(gen_text)
            
            rew = calculate_reward(gen_text, target, interpreter)
            rewards.append(rew)
            
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        total_reward += rewards.mean().item()
        
        # Calcular vantagens relativas do grupo
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r
        
        # Forward pass para as respostas geradas para calcular os log_probs
        model.train()
        
        # Tokenizar as gerações completas para calcular loss de política
        full_tokens = outputs # (G, S)
        # Criar labels: mascarar o prompt (labels = -100)
        labels = full_tokens.clone()
        labels[:, :prompt_len] = -100
        labels[full_tokens == tokenizer.pad_token_id] = -100
        
        # Forward pass causal para todo o grupo em lote
        outputs_logits = model(input_ids=full_tokens)
        logits = outputs_logits.logits
        
        # Shift logits e labels para treinamento causal
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calcular perda de CrossEntropy para cada token de cada amostra
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(G, -1)
        
        # Filtrar apenas tokens ativos (onde labels não é -100)
        active_masks = (shift_labels != -100).float()
        sample_losses = (loss * active_masks).sum(dim=1) / (active_masks.sum(dim=1) + 1e-8)
        
        # Multiplicar a perda de cada amostra pela vantagem negativa correspondente
        policy_loss = (sample_losses * (-advantages)).mean()
        total_policy_loss += policy_loss
        
    return total_policy_loss / B, total_reward / B

def train_hybrid_1b(args):
    print("="*60)
    print("   INICIANDO TREINAMENTO HÍBRIDO 1B+ (SFT + GRPO-RL)   ")
    print("="*60)
    
    if not PEFT_TRANSFORMERS_AVAILABLE:
        print("[ERRO] Bibliotecas transformers/peft não disponíveis.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    use_cuda = device.type == "cuda" and not args.dry_run
    
    if use_cuda:
        print(f"[INFO] Carregando modelo {args.model_id} em QLoRA de 4-bits...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print(f"[INFO] Carregando modelo {args.model_id} em Float32 (modo CPU/Dry-run)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=None,
            trust_remote_code=True
        ).to(device)
        
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    if use_cuda:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Criar datasets
    dataset_file = "data/conversational_sft_dataset.json"
    if os.path.exists(dataset_file):
        print(f"[INFO] Carregando dataset conversacional sintético de: {dataset_file}")
        with open(dataset_file, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)
    else:
        print("[INFO] Criando dataset conversacional em memória...")
        raw_dataset = ConversationalDataset(num_samples=args.num_samples, seed=42)
        
    train_size = int(len(raw_dataset) * 0.9)
    val_size = len(raw_dataset) - train_size
    train_raw, val_raw = torch.utils.data.random_split(raw_dataset, [train_size, val_size])
    
    train_dataset = QLoRADatasetWrapper(train_raw, tokenizer, max_seq_len=256)
    val_dataset = QLoRADatasetWrapper(val_raw, tokenizer, max_seq_len=256)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    try:
        import bitsandbytes as bnb
        if use_cuda:
            print("[INFO] Usando otimizador otimizado PagedAdamW8bit de bitsandbytes.")
            optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    except ImportError:
        print("[AVISO] bitsandbytes não instalado. Usando torch.optim.AdamW padrão.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    interpreter = TVDSLInterpreter()
    
    for epoch in range(args.epochs):
        is_grpo_phase = (epoch >= args.switch_epoch)
        model.train()
        
        total_loss = 0.0
        total_reward = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            if not is_grpo_phase:
                # FASE 1: SFT / Distilação de CoT textual
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (step + 1) % 5 == 0 or args.dry_run:
                    print(f"  [SFT] Época {epoch+1}/{args.epochs} | Passo {step+1}/{len(train_loader)} | Perda: {loss.item():.4f}")
            else:
                # FASE 2: GRPO RL
                optimizer.zero_grad()
                loss, avg_reward = run_grpo_step(model, tokenizer, batch, interpreter, device, group_size=2, lr=args.lr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_reward += avg_reward
                
                if (step + 1) % 5 == 0 or args.dry_run:
                    print(f"  [GRPO] Época {epoch+1}/{args.epochs} | Passo {step+1}/{len(train_loader)} | Recompensa: {avg_reward:.2f} | Perda: {loss.item():.4f}")
                    
            if args.dry_run:
                break
                
        epoch_time = time.time() - start_time
        scheduler.step()
        
        # Imprime estatísticas da época
        avg_loss = total_loss / len(train_loader)
        if not is_grpo_phase:
            print(f"Época {epoch+1:02d} [SFT] | Perda Média: {avg_loss:.4f} | Tempo: {epoch_time:.2f}s")
        else:
            avg_rew = total_reward / len(train_loader)
            print(f"Época {epoch+1:02d} [GRPO-RL] | Perda Média: {avg_loss:.4f} | Recompensa Média: {avg_rew:.2f} | Tempo: {epoch_time:.2f}s")
            
    # Salvar adaptador LoRA
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"\n[SUCESSO] Treinamento Híbrido Concluído! Adaptadores LoRA salvos em: {args.out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Script de Treinamento Híbrido 1B+ (SFT + GRPO-RL).")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--switch_epoch", type=int, default=1, help="Época na qual transiciona para GRPO-RL.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out_dir", type=str, default="checkpoints/think_vetor_1b_hybrid_lora")
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    
    if args.dry_run:
        args.num_samples = 2
        args.batch_size = 2
        args.epochs = 1
        args.switch_epoch = 0 # Força GRPO-RL direto para testar o loop de RL
        
    train_hybrid_1b(args)

if __name__ == "__main__":
    main()
