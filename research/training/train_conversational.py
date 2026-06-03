import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.conversational_dataset import ConversationalDataset
from src.model import ThinkVetorModel
from src.distill_loss import DistillLoss
from train import compute_ponder_loss

def evaluate_conversational(model, val_loader, tokenizer, device, num_samples=3):
    """
    Demonstra previsões qualitativas e calcula acurácia exata (EM) no alvo final.
    """
    model.eval()
    correct = 0
    total = 0
    
    print("\n=== Demonstração Qualitativa (Conversação) ===")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            raw_inputs = batch["raw_input"]
            raw_targets = batch["raw_target"]
            raw_cots = batch["raw_cot"]
            
            # Geração autoregressiva da resposta terminal
            generated_ids = model.generate(input_ids, max_length=15, temperature=0.0)
            
            for idx, gen_seq in enumerate(generated_ids):
                pred_str = tokenizer.decode(gen_seq).strip()
                target_str = raw_targets[idx].strip()
                
                if pred_str == target_str:
                    correct += 1
                total += 1
                
                # Mostrar algumas amostras qualitativas na primeira iteração
                if i == 0 and idx < num_samples:
                    print(f"Amostra {idx+1}:")
                    print(f"  Prompt:      '{raw_inputs[idx]}'")
                    print(f"  Plano CoT:   '{raw_cots[idx]}'")
                    print(f"  Esperado:    '{target_str}'")
                    print(f"  Modelo:      '{pred_str}'")
                    print(f"  Resultado:   {'CORRETO' if pred_str == target_str else 'INCORRETO'}")
                    print("-" * 40)
                    
    acc = (correct / total) * 100
    return acc

def main():
    parser = argparse.ArgumentParser(description="Treinamento da Micro-LLM Conversacional Think-Vetor (SFT + Destilação)")
    parser.add_argument("--epochs", type=int, default=10, help="Número total de épocas de treino.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Número de amostras do dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do lote (batch size).")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Nome do tokenizer HuggingFace.")
    parser.add_argument("--out_dir", type=str, default="checkpoints/think_vetor_conversational_exported", 
                        help="Pasta de saída para o checkpoint exportado final.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    print(f"=== INICIANDO TREINAMENTO CONVERSACIONAL EM {device.type.upper()} ===")
    
    tokenizer = HFTokenizerWrapper(args.tokenizer_name)
    
    # 1. Carregar Datasets
    train_ds = ConversationalDataset(num_samples=args.num_samples, seed=args.seed, tokenizer=tokenizer)
    val_ds = ConversationalDataset(num_samples=args.num_samples // 5, seed=args.seed + 1, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"[INFO] Amostras de Treino: {len(train_ds)} | Validação: {len(val_ds)}")
    
    # 2. Configurar Modelo Expandido de Chat (~6.2M parâmetros)
    d_model = 256
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    max_ponder_steps = 4
    num_memories = 512
    use_rope = True
    
    model = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_ponder_steps=max_ponder_steps,
        num_memories=num_memories,
        beta=8.0,
        use_pos_embedding=False,
        use_rope=use_rope
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parâmetros Totais do Modelo Conversacional: {num_params:,}")
    
    # Otimizadores e Perdas
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    distill_criterion = DistillLoss(weight_cosine=1.0, weight_mse=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))
    
    best_acc = 0.0
    
    # 3. Loop de Treinamento
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_distill = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            cot_ids = batch["cot_ids"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=(device_type == "cuda")):
                logits, halts, _, intermediate_states = model(input_ids, target_ids, return_details=True)
                
                # Cross-entropy no texto de resposta
                B, T, V = logits.shape
                ce_loss_raw = ce_criterion(logits.view(-1, V), target_ids.view(-1))
                ce_loss = (ce_loss_raw * target_mask.view(-1)).sum() / (target_mask.sum() + 1e-8)
                
                # Ponder Loss
                p_loss = compute_ponder_loss(halts, prior_prob=0.3) if len(halts) > 0 else torch.tensor(0.0, device=device)
                
                # Distilação Latente do Plano Cognitivo BPE
                with torch.no_grad():
                    cot_embeddings = model.token_embeddings(cot_ids)
                seq_len_distill = len(intermediate_states)
                d_loss = distill_criterion(intermediate_states, cot_embeddings[:, :seq_len_distill, :])
                
                loss = ce_loss + 0.1 * p_loss + 0.5 * d_loss
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_distill += d_loss.item()
            
        scheduler.step()
        epoch_time = time.time() - start_time
        
        # Validar
        val_acc = evaluate_conversational(model, val_loader, tokenizer, device, num_samples=2)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"CE: {total_ce/len(train_loader):.4f} | "
              f"Distill: {total_distill/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.2f}% | Tempo: {epoch_time:.2f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Salvar checkpoint intermediário
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/think_vetor_conversational_best.pt")
            
    print(f"\n[INFO] Treinamento concluído! Melhor Acurácia de Validação: {best_acc:.2f}%")
    
    # 4. Exportação Automática do Modelo Conversacional Final
    print(f"\n[INFO] Exportando modelo conversacional empacotado para: {args.out_dir}...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "max_ponder_steps": max_ponder_steps,
        "num_memories": num_memories,
        "beta": 8.0,
        "use_pos_embedding": False,
        "use_rope": use_rope,
        "is_causal_coconut": False,
        "tokenizer_type": args.tokenizer_name
    }
    
    # Grava config.json
    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        
    # Grava weights.pt (copia o estado do melhor modelo se existir, senão salva o estado atual)
    weights_path = os.path.join(args.out_dir, "weights.pt")
    if os.path.exists("checkpoints/think_vetor_conversational_best.pt"):
        best_state = torch.load("checkpoints/think_vetor_conversational_best.pt", map_location="cpu")
    else:
        print("[INFO] Nenhum checkpoint intermediário encontrado (acurácia permaneceu em 0%). Exportando os pesos finais do modelo atual.")
        best_state = model.state_dict()
    torch.save(best_state, weights_path)
    print(f"[SUCESSO] Configuração e pesos salvos em {args.out_dir}!")
    
    # 5. Gatilho de Download Automático no Google Colab
    if 'google.colab' in sys.modules:
        try:
            print("\n[Colab] Detectado ambiente Google Colab! Disparando downloads automáticos de weights.pt e config.json...")
            from google.colab import files
            files.download(weights_path)
            files.download(config_path)
            print("[Colab] Downloads iniciados com sucesso! Verifique sua pasta de downloads do navegador.")
        except Exception as e:
            print(f"[WARNING] Falha ao acionar download automático do Colab: {e}")

if __name__ == "__main__":
    main()
