import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel

def compute_ponder_loss(halting_probs, prior_prob=0.3):
    """
    Calcula a perda de ponderação baseada na KL-Divergência
    entre a distribuição de parada prevista (p) e uma distribuição geométrica prior (q).
    halting_probs: Lista de tensores de tamanho max_ponder_steps, cada um de formato (B, S, 1)
    """
    if len(halting_probs) == 0:
        return torch.tensor(0.0, device=halting_probs[0].device if len(halting_probs) > 0 else "cpu")
        
    # Stack: (B, S, max_ponder_steps)
    p = torch.cat(halting_probs, dim=-1)
    
    batch_size, seq_len, max_steps = p.shape
    device = p.device
    
    # Criar distribuição geométrica alvo q
    q = torch.zeros(max_steps, device=device)
    current_prob = prior_prob
    for k in range(max_steps - 1):
        q[k] = current_prob
        current_prob *= (1.0 - prior_prob)
    q[-1] = 1.0 - q[:-1].sum()
    
    # Expandir q para bater com as dimensões de p
    q = q.view(1, 1, -1).expand(batch_size, seq_len, -1)
    
    # KL-Divergence: P * log(P / Q)
    eps = 1e-8
    kl = p * torch.log((p + eps) / (q + eps))
    return kl.sum(dim=-1).mean()

def train_model(model_name, model, dataloader, val_dataloader, tokenizer, epochs=15, device="cpu", use_ponder=True):
    print(f"\n=== Iniciando Treinamento: {model_name} ===")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(reduction="none") # Usamos none para aplicar a máscara do padding
    
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))
    
    best_acc = 0.0
    epochs_no_improve = 0
    stable_perfect_epochs = 0
    patience = 8  # Interrompe se não melhorar a acurácia de validação por 8 épocas
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_p_loss = 0.0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass com autocast
            with torch.amp.autocast("cuda", enabled=(device_type == "cuda")):
                logits, halts = model(input_ids, target_ids)
                
                # 1. Perda de Cross-Entropy (somente nos tokens que não são padding)
                B, T, V = logits.shape
                logits_flat = logits.view(-1, V)
                targets_flat = target_ids.view(-1)
                mask_flat = target_mask.view(-1)
                
                ce_loss_raw = criterion(logits_flat, targets_flat)
                ce_loss = (ce_loss_raw * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                
                # 2. Perda de Ponderação
                if use_ponder and len(halts) > 0:
                    p_loss = compute_ponder_loss(halts, prior_prob=0.3)
                else:
                    p_loss = torch.tensor(0.0, device=device)
                    
                # Perda Híbrida Combinada
                loss = ce_loss + 0.1 * p_loss
            
            # Backward pass com GradScaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_p_loss += p_loss.item()
            
        scheduler.step()
        
        # Avaliação de validação rápida (Acurácia Exata)
        val_acc, avg_steps = evaluate_accuracy(model, val_dataloader, tokenizer, device, use_ponder)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss Total: {total_loss/len(dataloader):.4f} | "
              f"CE Loss: {total_ce_loss/len(dataloader):.4f} | "
              f"Ponder Loss: {total_p_loss/len(dataloader):.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Passos Médios: {avg_steps:.2f}")
        
        # Early stopping condicional
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{model_name.lower().replace(' ', '_')}_best.pt")
        else:
            epochs_no_improve += 1
            
        # Contagem de épocas com acurácia de 100%
        if val_acc >= 100.0:
            stable_perfect_epochs += 1
        else:
            stable_perfect_epochs = 0
            
        # Parada se bater acurácia perfeita por 3 épocas consecutivas
        if stable_perfect_epochs >= 3:
            print(f"[Early Stopping] Acurácia de validação perfeita estável (100.00%) por 3 épocas. Parando.")
            break
            
        # Parada se paciência esgotar
        if epochs_no_improve >= patience:
            print(f"[Early Stopping] Sem melhorias na validação por {patience} épocas. Parando.")
            break
            
    # Sempre salvar o modelo final para garantir a existência de checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    if best_acc == 0.0:
        torch.save(model.state_dict(), f"checkpoints/{model_name.lower().replace(' ', '_')}_best.pt")
            
    print(f"Treinamento finalizado. Melhor Acurácia de Validação: {best_acc:.2f}%")
    return best_acc

def evaluate_accuracy(model, dataloader, tokenizer, device, use_ponder=True):
    model.eval()
    correct = 0
    total = 0
    total_steps = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            raw_targets = batch["raw_target"]
            
            # Na inferência, geramos autoregressivamente
            # O target máximo tem comprimento de num_digits + 1
            max_len = input_ids.shape[1] // 2 + 1
            
            # Se for ThinkVetor, podemos rodar e também monitorar os passos
            # A função generate executa internamente a inferência sem ruído
            generated_ids = model.generate(input_ids, max_length=max_len, temperature=0.0)
            
            # Para monitorar passos médios, precisamos rodar o forward
            if use_ponder and model.max_ponder_steps > 0:
                _, halts = model(input_ids, input_ids[:, :1]) # forward simples para pegar halts
                # halts: lista de tensores (B, seq_len, 1)
                p = torch.cat(halts, dim=-1) # (B, seq_len, max_steps)
                # O passo em que parou é ponderado pela probabilidade de parada
                steps_per_token = (p * torch.arange(1, len(halts)+1, device=device).view(1, 1, -1)).sum(dim=-1)
                avg_steps_batch = steps_per_token.mean().item()
                total_steps += avg_steps_batch * input_ids.shape[0]
                total_samples += input_ids.shape[0]
            else:
                total_steps += 0.0
                total_samples += input_ids.shape[0]
                
            for idx, gen_seq in enumerate(generated_ids):
                pred_str = tokenizer.decode(gen_seq).strip()
                target_str = raw_targets[idx].strip()
                
                # Compara strings
                if pred_str == target_str:
                    correct += 1
                total += 1
                
    accuracy = (correct / total) * 100
    avg_steps = total_steps / total_samples if total_samples > 0 else 0.0
    return accuracy, avg_steps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # 1. Preparar Tokenizador e Datasets
    # Usaremos 2 dígitos para manter o treinamento rápido no protótipo
    num_digits = 2
    tokenizer = CharTokenizer()
    
    # Gerar todas as combinações únicas de 2 dígitos (10.000 pares no total)
    import random
    random.seed(42)
    max_val = 10**num_digits - 1
    all_pairs = []
    for a in range(max_val + 1):
        for b in range(max_val + 1):
            all_pairs.append((a, b))
    random.shuffle(all_pairs)
    
    # Divisão de 1.000 para treino e 200 para validação disjuntos
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
    
    # 2. Configurar Modelo Baseline (Sem Ponderação, com RoPE)
    baseline_model = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_ponder_steps=0, # Desativa ponderação latente
        use_pos_embedding=False,
        use_rope=True
    )
    
    # 3. Configurar Modelo Think-Vetor (Com Ponderação Latente, Langevin e RoPE)
    think_vetor = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_ponder_steps=6, # 6 passos de reflexão no máximo
        num_memories=128,
        beta=8.0,
        use_pos_embedding=False,
        use_rope=True
    )
    
    # 4. Treinar ambos os modelos
    epochs = 40
    train_model("Baseline Model", baseline_model, train_loader, val_loader, tokenizer, epochs=epochs, device=device, use_ponder=False)
    train_model("Think Vetor", think_vetor, train_loader, val_loader, tokenizer, epochs=epochs, device=device, use_ponder=True)

if __name__ == "__main__":
    main()
