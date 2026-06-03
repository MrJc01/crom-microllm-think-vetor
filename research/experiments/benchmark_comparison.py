import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.logic_dataset import LogicCharTokenizer, LogicDataset
from src.arithmetic_word_dataset import ArithmeticWordDataset
from src.model import ThinkVetorModel
from src.traditional_gpt import TraditionalGPT
from src.distill_loss import DistillLoss
from train import compute_ponder_loss

def extract_final_answer(gen_str):
    """
    Extrai o resultado final previsto de uma cadeia de texto gerada.
    Útil para extrair a resposta da baseline GPT que gera o CoT discreto primeiro.
    Ex: "20>15 Alice" -> "Alice"
    """
    parts = gen_str.strip().split()
    if len(parts) == 0:
        return ""
    ans = parts[-1]
    # Limpa possíveis resíduos de pontuação ou operadores
    ans = ans.replace("=", "").replace(">", "").replace("<", "").strip()
    return ans

def prepare_gpt_batch(batch, tokenizer, device, max_len=250):
    """
    Prepara um lote de premissas e cadeias de pensamento para o TraditionalGPT.
    Mapeia: prompt + CoT + " " + target + " "
    Aplica máscara de atenção usando labels = -100 para ignorar a perda no prompt.
    """
    raw_inputs = batch["raw_input"]
    raw_cots = batch["raw_cot"]
    raw_targets = batch["raw_target"]
    
    B = len(raw_inputs)
    all_full_ids = []
    all_labels = []
    
    for i in range(B):
        in_ids = tokenizer.encode(raw_inputs[i])
        L_in = len(in_ids)
        
        # Sequência a ser prevista: CoT + Target + EOS (espaço)
        tgt_ids = tokenizer.encode(raw_cots[i] + " " + raw_targets[i] + " ")
        L_tgt = len(tgt_ids)
        
        full = in_ids + tgt_ids
        labels = [-100] * L_in + tgt_ids
        
        if len(full) > max_len:
            full = full[:max_len]
            labels = labels[:max_len]
            
        pad_len = max_len - len(full)
        full_padded = full + [tokenizer.pad_id] * pad_len
        labels_padded = labels + [-100] * pad_len
        
        all_full_ids.append(full_padded)
        all_labels.append(labels_padded)
        
    full_ids_tensor = torch.tensor(all_full_ids, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    return full_ids_tensor, labels_tensor

def evaluate_think_vetor(model, val_loader, tokenizer, device):
    """
    Avalia a acurácia exata (EM) e a eficiência de tokens do Think-Vetor.
    """
    model.eval()
    correct = 0
    total = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            raw_inputs = batch["raw_input"]
            raw_targets = batch["raw_target"]
            
            # Gera a resposta direta (apenas o resultado terminal)
            max_len = 15
            generated_ids = model.generate(input_ids, max_length=max_len, temperature=0.0)
            
            for idx, gen_seq in enumerate(generated_ids):
                pred_str = tokenizer.decode(gen_seq).strip()
                target_str = raw_targets[idx].strip()
                
                if pred_str == target_str:
                    correct += 1
                total += 1
                
                # Métricas de tokens
                prompt_len = len(tokenizer.encode(raw_inputs[idx]))
                gen_len = len(gen_seq)
                
                total_prompt_tokens += prompt_len
                total_generated_tokens += gen_len
                
    acc = (correct / total) * 100
    avg_prompt = total_prompt_tokens / total
    avg_gen = total_generated_tokens / total
    return acc, avg_prompt, avg_gen

def evaluate_traditional_gpt(model, val_loader, tokenizer, device):
    """
    Avalia a acurácia exata (EM) e a eficiência de tokens da baseline TraditionalGPT em lote (batch).
    """
    model.eval()
    correct = 0
    total = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            raw_inputs = batch["raw_input"]
            raw_targets = batch["raw_target"]
            
            # Geração em lote acelerada por hardware/paralelismo!
            gen_ids = model.generate(input_ids, max_length=40, temperature=0.0)
            
            # Extrai apenas a parte recém-gerada (após o tamanho fixo de input_ids)
            for idx in range(input_ids.shape[0]):
                gen_tokens = gen_ids[idx, input_ids.shape[1]:]
                gen_str = tokenizer.decode(gen_tokens).strip()
                
                pred_ans = extract_final_answer(gen_str)
                target_str = raw_targets[idx].strip()
                
                if pred_ans == target_str:
                    correct += 1
                total += 1
                
                # Métricas de tokens (removendo pads do prompt real)
                prompt_len = len(tokenizer.encode(raw_inputs[idx]))
                total_prompt_tokens += prompt_len
                total_generated_tokens += len(gen_tokens)
                
    acc = (correct / total) * 100
    avg_prompt = total_prompt_tokens / total
    avg_gen = total_generated_tokens / total
    return acc, avg_prompt, avg_gen

def create_mixed_dataset(num_samples, errata_prob, tokenizer, task="logic", difficulty="medium"):
    """
    Gera um dataset com taxa de contradição/errata misturada de forma controlada.
    """
    max_errata = 0.4 if task == "logic" else 0.3
    
    if errata_prob > max_errata:
        errata_prob = max_errata
        
    alpha = errata_prob / max_errata
    num_mutable = int(num_samples * alpha)
    num_clean = num_samples - num_mutable
    
    if task == "logic":
        clean_ds = LogicDataset(num_samples=num_clean, seed=101, tokenizer=tokenizer, mutable_context=False)
        mutable_ds = LogicDataset(num_samples=num_mutable, seed=102, tokenizer=tokenizer, mutable_context=True)
    else:
        clean_ds = ArithmeticWordDataset(num_samples=num_clean, seed=101, tokenizer=tokenizer, difficulty=difficulty, mutable_context=False)
        mutable_ds = ArithmeticWordDataset(num_samples=num_mutable, seed=102, tokenizer=tokenizer, difficulty=difficulty, mutable_context=True)
        
    combined_ds = LogicDataset(num_samples=1, tokenizer=tokenizer) if task == "logic" else ArithmeticWordDataset(num_samples=1, tokenizer=tokenizer)
    combined_ds.samples = clean_ds.samples + mutable_ds.samples
    random.shuffle(combined_ds.samples)
    return combined_ds

def main():
    parser = argparse.ArgumentParser(description="Script de Benchmarking Comparativo: Think-Vetor vs TraditionalGPT")
    parser.add_argument("--task", type=str, default="logic", choices=["logic", "arithmetic"], help="Tarefa a ser testada.")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"], help="Dificuldade dos word problems.")
    parser.add_argument("--epochs", type=int, default=10, help="Número total de épocas de treino.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do lote (batch size).")
    parser.add_argument("--num_samples", type=int, default=1000, help="Número de amostras de treino.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    print(f"=== INICIANDO BENCHMARK COMPARATIVO EM {device.type.upper()} ===")
    print(f"Tarefa: {args.task.upper()} | Amostras: {args.num_samples} | Épocas: {args.epochs}\n")
    
    tokenizer = LogicCharTokenizer()
    
    # 1. Preparar Datasets
    if args.task == "logic":
        # Usamos mutable_context=True no treino para que ambos os modelos aprendam a lidar com erratas
        train_ds = LogicDataset(num_samples=args.num_samples, seed=args.seed, tokenizer=tokenizer, mutable_context=True)
        val_ds = LogicDataset(num_samples=args.num_samples // 5, seed=args.seed + 1, tokenizer=tokenizer, mutable_context=False)
    else:
        train_ds = ArithmeticWordDataset(num_samples=args.num_samples, seed=args.seed, tokenizer=tokenizer, difficulty=args.difficulty, mutable_context=True)
        val_ds = ArithmeticWordDataset(num_samples=args.num_samples // 5, seed=args.seed + 1, tokenizer=tokenizer, difficulty=args.difficulty, mutable_context=False)
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Inicializar Modelos
    # Think-Vetor Model (~1.2M params)
    think_vetor = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_ponder_steps=6,
        num_memories=256,
        beta=8.0,
        use_pos_embedding=False,
        use_rope=True
    ).to(device)
    
    # TraditionalGPT matched parameter baseline (~1.2M params)
    traditional_gpt = TraditionalGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=8,
        num_layers=6,
        use_rope=True
    ).to(device)
    
    print(f"[INFO] Think-Vetor Parâmetros: {sum(p.numel() for p in think_vetor.parameters() if p.requires_grad):,}")
    print(f"[INFO] TraditionalGPT Parâmetros: {sum(p.numel() for p in traditional_gpt.parameters() if p.requires_grad):,}")
    
    # Otimizadores e schedulers
    opt_tv = optim.AdamW(think_vetor.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_gpt = optim.AdamW(traditional_gpt.parameters(), lr=1e-3, weight_decay=1e-4)
    
    sched_tv = optim.lr_scheduler.CosineAnnealingLR(opt_tv, T_max=args.epochs)
    sched_gpt = optim.lr_scheduler.CosineAnnealingLR(opt_gpt, T_max=args.epochs)
    
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    distill_criterion = DistillLoss(weight_cosine=1.0, weight_mse=0.1)
    
    scaler_tv = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))
    scaler_gpt = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))
    
    history_tv = []
    history_gpt = []
    
    # 3. Loop de Treino - Think-Vetor
    print("\n--- Treinando o Think-Vetor (SFT + Destilação Latente) ---")
    for epoch in range(args.epochs):
        think_vetor.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            cot_ids = batch["cot_ids"].to(device)
            
            opt_tv.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=(device_type == "cuda")):
                logits, halts, _, intermediate_states = think_vetor(input_ids, target_ids, return_details=True)
                
                # Cross-entropy no alvo final
                B, T, V = logits.shape
                ce_loss_raw = ce_criterion(logits.view(-1, V), target_ids.view(-1))
                ce_loss = (ce_loss_raw * target_mask.view(-1)).sum() / (target_mask.sum() + 1e-8)
                
                # Ponder Loss (PonderNet)
                p_loss = compute_ponder_loss(halts, prior_prob=0.3) if len(halts) > 0 else torch.tensor(0.0, device=device)
                
                # Distill Loss
                with torch.no_grad():
                    cot_embeddings = think_vetor.token_embeddings(cot_ids)
                seq_len_distill = len(intermediate_states)
                d_loss = distill_criterion(intermediate_states, cot_embeddings[:, :seq_len_distill, :])
                
                loss = ce_loss + 0.1 * p_loss + 0.5 * d_loss
                
            scaler_tv.scale(loss).backward()
            scaler_tv.unscale_(opt_tv)
            torch.nn.utils.clip_grad_norm_(think_vetor.parameters(), max_norm=1.0)
            scaler_tv.step(opt_tv)
            scaler_tv.update()
            
            total_loss += loss.item()
            
        sched_tv.step()
        epoch_time = time.time() - start_time
        val_acc, val_prompt, val_gen = evaluate_think_vetor(think_vetor, val_loader, tokenizer, device)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.2f}% | Tempo: {epoch_time:.2f}s")
        history_tv.append({"epoch": epoch+1, "val_acc": val_acc, "loss": total_loss/len(train_loader), "time": epoch_time})
        
    # 4. Loop de Treino - TraditionalGPT
    print("\n--- Treinando a Baseline TraditionalGPT (Causal Seq2Seq) ---")
    for epoch in range(args.epochs):
        traditional_gpt.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            opt_gpt.zero_grad()
            
            # Constrói o contexto e a máscara causal clássica
            full_ids, labels = prepare_gpt_batch(batch, tokenizer, device)
            
            with torch.amp.autocast("cuda", enabled=(device_type == "cuda")):
                logits = traditional_gpt(full_ids)
                
                # Computa perda ignorando os tokens com label -100
                B, S, V = logits.shape
                loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
                
            scaler_gpt.scale(loss).backward()
            scaler_gpt.unscale_(opt_gpt)
            torch.nn.utils.clip_grad_norm_(traditional_gpt.parameters(), max_norm=1.0)
            scaler_gpt.step(opt_gpt)
            scaler_gpt.update()
            
            total_loss += loss.item()
            
        sched_gpt.step()
        epoch_time = time.time() - start_time
        val_acc, val_prompt, val_gen = evaluate_traditional_gpt(traditional_gpt, val_loader, tokenizer, device)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.2f}% | Tempo: {epoch_time:.2f}s")
        history_gpt.append({"epoch": epoch+1, "val_acc": val_acc, "loss": total_loss/len(train_loader), "time": epoch_time})
        
    # 5. Avaliação Final de Eficiência de Tokens
    print("\n================ METRICAS DE EFICIENCIA DE TOKENS (VALIDACAO CLEAN) ================")
    acc_tv, prompt_tv, gen_tv = evaluate_think_vetor(think_vetor, val_loader, tokenizer, device)
    acc_gpt, prompt_gpt, gen_gpt = evaluate_traditional_gpt(traditional_gpt, val_loader, tokenizer, device)
    
    context_tv = prompt_tv + gen_tv
    context_gpt = prompt_gpt + gen_gpt
    compression_ratio = gen_gpt / gen_tv if gen_tv > 0 else 0.0
    
    print(f"Think-Vetor:    Acurácia EM: {acc_tv:.2f}% | Tokens Prompt Médios: {prompt_tv:.1f} | Tokens Gerados Médios: {gen_tv:.1f} | Contexto Máximo Médio: {context_tv:.1f}")
    print(f"TraditionalGPT: Acurácia EM: {acc_gpt:.2f}% | Tokens Prompt Médios: {prompt_gpt:.1f} | Tokens Gerados Médios: {gen_gpt:.1f} | Contexto Máximo Médio: {context_gpt:.1f}")
    print(f"--> Razão de Compressão de Tokens Gerados (Think-Vetor): {compression_ratio:.2f}x mais eficiente!")
    
    # 6. Teste de Resistência a Erratas e Ruído
    print("\n================ TESTE DE RESISTENCIA A ERRATAS E RUIDO (MUTABLE CONTEXT) ================")
    # Níveis de contradição de 0% a 40% (ou 30% para arithmetic)
    levels = [0.0, 0.1, 0.2, 0.3, 0.4] if args.task == "logic" else [0.0, 0.1, 0.2, 0.3]
    
    results_errata = []
    
    print("| Taxa de Errata | Acurácia Think-Vetor | Acurácia TraditionalGPT | Diferença (TV - GPT) |")
    print("|----------------|----------------------|------------------------|----------------------|")
    for lvl in levels:
        # Criar dataset de teste misturado correspondente (proporcional ao tamanho de validação)
        noise_ds = create_mixed_dataset(num_samples=max(50, len(val_ds)), errata_prob=lvl, tokenizer=tokenizer, task=args.task, difficulty=args.difficulty)
        noise_loader = DataLoader(noise_ds, batch_size=args.batch_size, shuffle=False)
        
        acc_tv_noise, _, _ = evaluate_think_vetor(think_vetor, noise_loader, tokenizer, device)
        acc_gpt_noise, _, _ = evaluate_traditional_gpt(traditional_gpt, noise_loader, tokenizer, device)
        
        diff = acc_tv_noise - acc_gpt_noise
        print(f"| {lvl*100:12.1f}% | {acc_tv_noise:19.2f}% | {acc_gpt_noise:21.2f}% | {diff:+19.2f}% |")
        
        results_errata.append({
            "errata_rate": lvl,
            "acc_think_vetor": acc_tv_noise,
            "acc_traditional_gpt": acc_gpt_noise
        })
        
    # Salvar resultados em checkpoints/benchmark_results.pt para persistência
    results_dict = {
        "task": args.task,
        "difficulty": args.difficulty,
        "history_tv": history_tv,
        "history_gpt": history_gpt,
        "efficiency": {
            "think_vetor": {"acc": acc_tv, "prompt": prompt_tv, "gen": gen_tv, "context": context_tv},
            "traditional_gpt": {"acc": acc_gpt, "prompt": prompt_gpt, "gen": gen_gpt, "context": context_gpt},
            "compression_ratio": compression_ratio
        },
        "errata_resilience": results_errata
    }
    torch.save(results_dict, "checkpoints/benchmark_results.pt")
    print(f"\n[INFO] Resultados do benchmark salvos com sucesso em checkpoints/benchmark_results.pt")
    
if __name__ == "__main__":
    main()
