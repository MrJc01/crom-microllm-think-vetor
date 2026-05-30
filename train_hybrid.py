import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel
from src.distill_loss import DistillLoss
from src.grpo_agent import GRPOAgent
from train import compute_ponder_loss, evaluate_accuracy

def train_hybrid_model(model, dataloader, val_dataloader, tokenizer, total_epochs=60, switch_epoch=20, device="cpu"):
    print(f"\n=== Iniciando Treinamento Híbrido (DISTILL -> GRPO) ===")
    print(f"[INFO] Épocas de Destilação: 1 a {switch_epoch}")
    print(f"[INFO] Épocas de GRPO: {switch_epoch + 1} a {total_epochs}")
    model = model.to(device)
    
    # Otimizador com decaimento cosseno
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    
    # Critérios e agentes
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    distill_criterion = DistillLoss(weight_cosine=1.0, weight_mse=0.1)
    grpo_agent = GRPOAgent(model, optimizer, tokenizer, group_size=4)
    
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))
    
    best_acc = 0.0
    epochs_no_improve = 0
    stable_perfect_epochs = 0
    patience = 10
    
    for epoch in range(total_epochs):
        is_grpo_phase = (epoch >= switch_epoch)
        model.train()
        total_loss = 0.0
        
        # Monitoramento específico de cada fase
        total_ce = 0.0
        total_distill = 0.0
        total_policy = 0.0
        total_reward = 0.0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            
            if not is_grpo_phase:
                # ==========================================
                # FASE 1: Destilação Latente Supervisionada
                # ==========================================
                cot_ids = batch["cot_ids"].to(device)
                optimizer.zero_grad()
                
                with torch.amp.autocast("cuda", enabled=(device_type == "cuda")):
                    logits, halts, _, intermediate_states = model(input_ids, target_ids, return_details=True)
                    
                    # 1. Perda de Cross-Entropy (geração final de caracteres)
                    B, T, V = logits.shape
                    ce_loss_raw = ce_criterion(logits.view(-1, V), target_ids.view(-1))
                    ce_loss = (ce_loss_raw * target_mask.view(-1)).sum() / (target_mask.sum() + 1e-8)
                    
                    # 2. Perda de Ponderação (PonderNet)
                    if len(halts) > 0:
                        p_loss = compute_ponder_loss(halts, prior_prob=0.3)
                    else:
                        p_loss = torch.tensor(0.0, device=device)
                        
                    # 3. Perda de Destilação (Truncar para bater com os passos de reflexão)
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
            else:
                # ==========================================
                # FASE 2: Ajuste Fino por Reforço (GRPO)
                # ==========================================
                # O grpo_agent gerencia seu próprio backward/step
                loss_val, policy_val, avg_rew = grpo_agent.compute_grpo_loss(
                    input_ids, target_ids, target_mask, device, scaler=scaler
                )
                total_loss += loss_val
                total_policy += policy_val
                total_reward += avg_rew
                
        scheduler.step()
        
        # Validar
        val_acc, avg_steps = evaluate_accuracy(model, val_dataloader, tokenizer, device, use_ponder=True)
        
        # Log da época dependendo da fase ativa
        if not is_grpo_phase:
            print(f"Epoch {epoch+1:02d}/{total_epochs:02d} [DISTILL] | "
                  f"Loss: {total_loss/len(dataloader):.4f} | "
                  f"CE: {total_ce/len(dataloader):.4f} | "
                  f"Distill: {total_distill/len(dataloader):.4f} | "
                  f"Val Acc: {val_acc:.2f}% | Steps: {avg_steps:.2f}")
        else:
            print(f"Epoch {epoch+1:02d}/{total_epochs:02d} [GRPO-RL] | "
                  f"Loss: {total_loss/len(dataloader):.4f} | "
                  f"Policy: {total_policy/len(dataloader):.4f} | "
                  f"Reward: {total_reward/len(dataloader)*100:.2f}% | "
                  f"Val Acc: {val_acc:.2f}% | Steps: {avg_steps:.2f}")
            
        # Salvar o melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/think_vetor_hybrid_best.pt")
        else:
            epochs_no_improve += 1
            
        # Early Stopping por perfeição
        if val_acc >= 100.0:
            stable_perfect_epochs += 1
        else:
            stable_perfect_epochs = 0
            
        if stable_perfect_epochs >= 3:
            print(f"[Early Stopping] Acurácia de validação perfeita estável (100.00%) por 3 épocas. Parando.")
            break
            
        if epochs_no_improve >= patience:
            print(f"[Early Stopping] Sem melhorias na validação por {patience} épocas. Parando.")
            break
            
    print(f"Treinamento híbrido finalizado. Melhor Acurácia de Validação: {best_acc:.2f}%")
    return best_acc

def main():
    parser = argparse.ArgumentParser(description="Treinamento Híbrido (Destilação + GRPO) para o Think-Vetor.")
    parser.add_argument("--epochs", type=int, default=10, help="Número total de épocas.")
    parser.add_argument("--switch_epoch", type=int, default=5, help="Época para transicionar de Destilação para GRPO.")
    parser.add_argument("--align_operator", type=bool, default=True, help="Alinhamento de operador centralizado.")
    parser.add_argument("--num_digits", type=int, default=2, help="Número de dígitos das somas.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    
    # Dataset reduzido para teste rápido local
    train_ds = AdditionDataset(num_digits=args.num_digits, num_samples=100, seed=42, tokenizer=tokenizer, 
                               align_operator=args.align_operator, max_d_align=4)
    val_ds = AdditionDataset(num_digits=args.num_digits, num_samples=50, seed=43, tokenizer=tokenizer,
                             align_operator=args.align_operator, max_d_align=4)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_ponder_steps=6,
        num_memories=128,
        beta=8.0,
        use_pos_embedding=False,
        use_rope=True
    )
    
    train_hybrid_model(model, train_loader, val_loader, tokenizer, 
                       total_epochs=args.epochs, switch_epoch=args.switch_epoch, device=device)

if __name__ == "__main__":
    main()
