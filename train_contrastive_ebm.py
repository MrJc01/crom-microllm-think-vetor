import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel
from train import compute_ponder_loss, evaluate_accuracy

def train_contrastive_ebm_model(model, dataloader, val_dataloader, tokenizer, epochs=10, device="cpu"):
    print(f"\n=== Iniciando Treinamento Contrastivo EBM ===")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")
    
    best_acc = 0.0
    margin = 1.0 # Margem hinge para perda contrastiva
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ebm_loss = 0.0
        total_entropy = 0.0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            
            optimizer.zero_grad()
            
            # 1. Trajetória Positiva (Gabarito correto)
            logits_pos, halts, pooled_states_pos, _ = model(input_ids, target_ids, return_details=True)
            
            # 2. Trajetória Negativa (Com targets corrompidos para simular alucinações/erros)
            # Embaralhar os targets para gerar um fluxo latente incorreto
            B, T = target_ids.shape
            shuffled_indices = torch.randperm(T, device=device)
            target_ids_neg = target_ids[:, shuffled_indices]
            
            _, _, pooled_states_neg, _ = model(input_ids, target_ids_neg, return_details=True)
            
            # 3. Calcular energias E(z) do Hopfield
            energy_pos = model.hopfield_ebm.compute_energy(pooled_states_pos)
            energy_neg = model.hopfield_ebm.compute_energy(pooled_states_neg)
            
            # Perda Contrastiva Hinge: ReLU(E_pos - E_neg + margin)
            ebm_loss = torch.relu(energy_pos - energy_neg + margin)
            
            # 4. Calcular entropia de ativação de memórias para evitar colapso
            _, entropy = model.hopfield_ebm(pooled_states_pos, temp=0.0, return_entropy=True)
            
            # 5. Perda de Cross Entropy padrão
            B_logits, T_logits, V = logits_pos.shape
            logits_flat = logits_pos.view(-1, V)
            targets_flat = target_ids.view(-1)
            mask_flat = target_mask.view(-1)
            
            ce_loss_raw = criterion(logits_flat, targets_flat)
            ce_loss = (ce_loss_raw * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            
            # Perda do PonderNet
            if len(halts) > 0:
                p_loss = compute_ponder_loss(halts, prior_prob=0.3)
            else:
                p_loss = torch.tensor(0.0, device=device)
                
            # Perda Híbrida Total (EBM + regularizações)
            # Maximizamos a entropia subtraindo-a (coeficiente negativo)
            loss = ce_loss + 0.1 * p_loss + 0.2 * ebm_loss - 0.05 * entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_ebm_loss += ebm_loss.item()
            total_entropy += entropy.item()
            
        # Avaliação de validação rápida (Acurácia Exata)
        val_acc, avg_steps = evaluate_accuracy(model, val_dataloader, tokenizer, device, use_ponder=True)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss Total: {total_loss/len(dataloader):.4f} | "
              f"CE: {total_ce_loss/len(dataloader):.4f} | "
              f"EBM: {total_ebm_loss/len(dataloader):.4f} | "
              f"Mem Entropy: {total_entropy/len(dataloader):.2f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Passos Médios: {avg_steps:.2f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/think_vetor_ebm_best.pt")
            
    print(f"Treinamento Contrastivo EBM finalizado. Melhor Acurácia de Validação: {best_acc:.2f}%")
    return best_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    tokenizer = CharTokenizer()
    num_digits = 2
    
    # Dataset reduzido para teste rápido
    train_ds = AdditionDataset(num_digits=num_digits, num_samples=100, seed=42, tokenizer=tokenizer)
    val_ds = AdditionDataset(num_digits=num_digits, num_samples=50, seed=43, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_ponder_steps=4,
        num_memories=128,
        beta=8.0,
        use_pos_embedding=False,
        use_rope=True
    )
    
    train_contrastive_ebm_model(model, train_loader, val_loader, tokenizer, epochs=5, device=device)

if __name__ == "__main__":
    main()
