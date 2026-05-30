import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel
from src.distill_loss import DistillLoss
from train import compute_ponder_loss, evaluate_accuracy

def train_distill_model(model, dataloader, val_dataloader, tokenizer, epochs=10, device="cpu"):
    print(f"\n=== Iniciando Treinamento com Destilação Latente ===")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")
    distill_criterion = DistillLoss(weight_cosine=1.0, weight_mse=0.1)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_p_loss = 0.0
        total_d_loss = 0.0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            cot_ids = batch["cot_ids"].to(device) # (B, K)
            
            optimizer.zero_grad()
            
            # Forward pass com return_details=True para extrair hidden states intermediários
            logits, halts, pooled_states, intermediate_states = model(input_ids, target_ids, return_details=True)
            
            # 1. Perda de Cross-Entropy (geração de texto)
            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            targets_flat = target_ids.view(-1)
            mask_flat = target_mask.view(-1)
            
            ce_loss_raw = criterion(logits_flat, targets_flat)
            ce_loss = (ce_loss_raw * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            
            # 2. Perda de Ponderação (PonderNet)
            if len(halts) > 0:
                p_loss = compute_ponder_loss(halts, prior_prob=0.3)
            else:
                p_loss = torch.tensor(0.0, device=device)
                
            # 3. Perda de Destilação Latente
            # Embeddar os CoT IDs usando a própria tabela de embeddings do modelo
            with torch.no_grad():
                cot_embeddings = model.token_embeddings(cot_ids) # (B, K, d_model)
                
            d_loss = distill_criterion(intermediate_states, cot_embeddings)
            
            # Perda Híbrida Combinada
            loss = ce_loss + 0.1 * p_loss + 0.5 * d_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_p_loss += p_loss.item()
            total_d_loss += d_loss.item()
            
        # Avaliação de validação rápida (Acurácia Exata)
        val_acc, avg_steps = evaluate_accuracy(model, val_dataloader, tokenizer, device, use_ponder=True)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss Total: {total_loss/len(dataloader):.4f} | "
              f"CE: {total_ce_loss/len(dataloader):.4f} | "
              f"Ponder: {total_p_loss/len(dataloader):.4f} | "
              f"Distill: {total_d_loss/len(dataloader):.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Passos Médios: {avg_steps:.2f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/think_vetor_distill_best.pt")
            
    print(f"Treinamento finalizado. Melhor Acurácia de Validação (Destilado): {best_acc:.2f}%")
    return best_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    tokenizer = CharTokenizer()
    num_digits = 2
    
    # Gerar amostras disjuntas
    import random
    random.seed(42)
    max_val = 10**num_digits - 1
    all_pairs = []
    for a in range(max_val + 1):
        for b in range(max_val + 1):
            all_pairs.append((a, b))
    random.shuffle(all_pairs)
    
    train_pairs = all_pairs[:1000]
    val_pairs = all_pairs[1000:1200]
    
    def pairs_to_samples(pairs):
        return [f"{a}+{b}=" for a, b in pairs], [f"{a+b}" for a, b in pairs]
        
    train_inputs, train_targets = pairs_to_samples(train_pairs)
    val_inputs, val_targets = pairs_to_samples(val_pairs)
    
    train_samples = list(zip(train_inputs, train_targets))
    val_samples = list(zip(val_inputs, val_targets))
    
    train_ds = AdditionDataset(num_digits=num_digits, tokenizer=tokenizer, samples=train_samples)
    val_ds = AdditionDataset(num_digits=num_digits, tokenizer=tokenizer, samples=val_samples)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    # Instanciar o modelo com RoPE e max_ponder_steps=6 para bater com a largura do CoT ("014110")
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
    
    train_distill_model(model, train_loader, val_loader, tokenizer, epochs=5, device=device)

if __name__ == "__main__":
    main()
