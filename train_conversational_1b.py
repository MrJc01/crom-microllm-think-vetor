import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Adiciona o diretório atual ao sys.path para garantir imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.conversational_dataset import ConversationalDataset

# Tentar importar bibliotecas de PEFT e Transformers
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_TRANSFORMERS_AVAILABLE = True
except ImportError:
    PEFT_TRANSFORMERS_AVAILABLE = False

class QLoRADatasetWrapper(Dataset):
    """
    Wrapper que adapta o ConversationalDataset para o formato de Chat Template da Hugging Face.
    Aplica a tag XML <thought> para encapsular o CoT de planejamento latente.
    """
    def __init__(self, raw_dataset, tokenizer, max_seq_len=256):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.raw_dataset)
        
    def __getitem__(self, idx):
        # Acessa os dados brutos gerados proceduralmente
        sample = self.raw_dataset[idx]
        raw_input = sample["raw_input"]
        raw_target = sample["raw_target"]
        raw_cot = sample["raw_cot"]
        
        # Mapear para o formato do Chat Template do Qwen / LLMs modernas
        messages = [
            {
                "role": "system", 
                "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."
            },
            {
                "role": "user", 
                "content": raw_input
            },
            {
                "role": "assistant", 
                # Insere a cadeia de pensamento latente CoT dentro das tags <thought>
                "content": f"<thought>\n{raw_cot}\n</thought>\n{raw_target}"
            }
        ]
        
        # Aplicar chat template do tokenizer
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Tokenizar
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Criar Labels com Prompt Masking (mascarar system e user para não calcular perda neles)
        labels = input_ids.clone()
        
        # O Chat Template do Qwen/Llama usa tokens de controle como <|im_start|>assistant para delimitar a resposta
        # Vamos encontrar onde começa a resposta do assistente para mascarar tudo o que veio antes com -100
        assistant_start_token = self.tokenizer.encode("<|im_start|>assistant")
        
        # Caso o tokenizer use tokens específicos, procuramos por essa subsequência
        start_idx = -1
        # Busca linear simples da subsequência
        for i in range(len(input_ids) - len(assistant_start_token) + 1):
            if input_ids[i : i + len(assistant_start_token)].tolist() == assistant_start_token:
                # O assistente começa logo após o cabeçalho <|im_start|>assistant\n
                start_idx = i + len(assistant_start_token)
                break
                
        if start_idx != -1:
            # Mascarar todos os tokens antes do início da resposta do assistente
            labels[:start_idx] = -100
        else:
            # Fallback: mascarar pelo menos o input básico do usuário
            user_len = len(self.tokenizer.encode(raw_input))
            labels[:user_len] = -100
            
        # Também mascaramos os tokens de padding no final
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_conversational_1b(args):
    print("="*60)
    print("   INICIANDO PIPELINE DE TREINAMENTO DE MODELO DE 1B (QLoRA)   ")
    print("="*60)
    
    if not PEFT_TRANSFORMERS_AVAILABLE:
        print("[ERRO] Para rodar o fine-tuning de 1B, é necessário instalar as dependências de PEFT.")
        print("Execute localmente: pip install transformers peft bitsandbytes accelerate trl")
        print("Ou prepare-se para rodar no Google Colab.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de execução detectado: {device.type.upper()}")
    
    # 1. Carregar Tokenizer
    print(f"[INFO] Baixando tokenizer do modelo base: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 2. Carregar Model em 4-bits (ou float32 para CPU/Dry-run)
    use_cuda = device.type == "cuda" and not args.dry_run
    
    if use_cuda:
        print(f"[INFO] Carregando modelo base {args.model_id} em QLoRA de 4-bits na GPU...")
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
        # Modo CPU ou Dry-run leve para testes locais rápidos
        print(f"[AVISO/INFO] Modo CPU/Dry-run ativado. Carregando modelo leve {args.model_id} em Float32...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=None,
            trust_remote_code=True
        ).to(device)
        
    print("[SUCESSO] Modelo base carregado!")
    
    # 3. Configurar Adaptação LoRA
    print("[INFO] Injetando adaptadores LoRA (PEFT)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Adicionar suporte a checkpoint de gradientes para economizar VRAM extrema na GPU
    if use_cuda:
        model.gradient_checkpointing_enable()
        
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Criar Datasets
    print(f"[INFO] Gerando dataset conversacional sintético procedural ({args.num_samples} amostras)...")
    raw_dataset = ConversationalDataset(num_samples=args.num_samples, seed=42)
    
    train_size = int(len(raw_dataset) * 0.9)
    val_size = len(raw_dataset) - train_size
    train_raw, val_raw = torch.utils.data.random_split(raw_dataset, [train_size, val_size])
    
    train_dataset = QLoRADatasetWrapper(train_raw, tokenizer, max_seq_len=256)
    val_dataset = QLoRADatasetWrapper(val_raw, tokenizer, max_seq_len=256)
    
    print(f"[INFO] Amostras de Treinamento: {len(train_dataset)} | Validação: {len(val_dataset)}")
    
    # 5. Loop de Treinamento Híbrido SFT + QLoRA
    # Como queremos rodar isso em lote, usaremos o DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float("inf")
    
    print(f"=== INICIANDO AJUSTE FINO (PEFT) POR {args.epochs} ÉPOCAS ===")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            # Clip de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if (step + 1) % 10 == 0 or args.dry_run:
                print(f"  Época {epoch+1}/{args.epochs} | Passo {step+1}/{len(train_loader)} | Perda instantânea: {loss.item():.4f}")
                if args.dry_run:
                    break # Interrompe rápido no modo teste
                    
        epoch_time = time.time() - start_time
        scheduler.step()
        
        # Validar
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, val_batch in enumerate(val_loader):
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch["attention_mask"].to(device)
                val_labels = val_batch["labels"].to(device)
                
                val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                val_loss += val_outputs.loss.item()
                if args.dry_run:
                    break
                    
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d} Finalizada | Treino Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Tempo: {epoch_time:.2f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"[SUCESSO] Nova melhor perda na validação! Salvando adaptador LoRA em: {args.out_dir}...")
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            
            # Criar um arquivo config.json simples para documentar a compatibilidade
            import json
            custom_config = {
                "base_model": args.model_id,
                "vocab_size": tokenizer.vocab_size,
                "lora_r": 16,
                "lora_alpha": 32,
                "use_thought_tags": True,
                "is_conversational_1b": True
            }
            with open(os.path.join(args.out_dir, "think_vetor_config.json"), "w", encoding="utf-8") as f:
                json.dump(custom_config, f, indent=4)
                
    print(f"\n[SUCESSO] Ajuste Fino Concluído! Adaptadores LoRA salvos em: {args.out_dir}")
    print("Você pode integrar esses adaptadores ao modelo de base na inferência.")
    
def main():
    parser = argparse.ArgumentParser(description="Treinamento QLoRA da LLM de 1B+ (Think-Vetor Chat 1B)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                        help="ID do modelo na Hugging Face (sugerido: Qwen/Qwen2.5-1.5B-Instruct ou Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de ajuste fino.")
    parser.add_argument("--num_samples", type=int, default=1200, help="Quantidade de amostras multi-tarefa sintéticas.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size do treinamento.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Taxa de aprendizado para LoRA.")
    parser.add_argument("--out_dir", type=str, default="checkpoints/think_vetor_1b_lora", help="Pasta final para salvar o adaptador.")
    parser.add_argument("--dry_run", action="store_true", help="Rodar dry-run local rápido no CPU.")
    
    args = parser.parse_args()
    
    # Se for dry_run local, ajustamos amostras para teste instantâneo
    if args.dry_run:
        args.num_samples = 4
        args.batch_size = 2
        args.epochs = 1
        args.model_id = "Qwen/Qwen2.5-0.5B-Instruct" # Forçar modelo mais leve para dry-run
        
    train_conversational_1b(args)

if __name__ == "__main__":
    main()
