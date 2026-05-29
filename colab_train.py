import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel
from train import compute_ponder_loss, evaluate_accuracy

def train_colab_model(model_name, model, dataloader, val_dataloader, tokenizer, epochs=80, device="cpu", use_ponder=True):
    print(f"\n=== Iniciando Treinamento no Colab: {model_name} ===")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")
    
    history = {"epoch": [], "loss": [], "ce_loss": [], "ponder_loss": [], "val_acc": [], "avg_steps": []}
    best_acc = 0.0
    
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
            
            logits, halts = model(input_ids, target_ids)
            
            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            targets_flat = target_ids.view(-1)
            mask_flat = target_mask.view(-1)
            
            ce_loss_raw = criterion(logits_flat, targets_flat)
            ce_loss = (ce_loss_raw * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            
            if use_ponder and len(halts) > 0:
                p_loss = compute_ponder_loss(halts, prior_prob=0.3)
            else:
                p_loss = torch.tensor(0.0, device=device)
                
            loss = ce_loss + 0.1 * p_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_p_loss += p_loss.item()
            
        val_acc, avg_steps = evaluate_accuracy(model, val_dataloader, tokenizer, device, use_ponder)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/len(dataloader):.4f} | Val Acc: {val_acc:.2f}% | Steps: {avg_steps:.2f}")
        
        # Salvar histórico
        history["epoch"].append(epoch + 1)
        history["loss"].append(total_loss / len(dataloader))
        history["ce_loss"].append(total_ce_loss / len(dataloader))
        history["ponder_loss"].append(total_p_loss / len(dataloader))
        history["val_acc"].append(val_acc)
        history["avg_steps"].append(avg_steps)
        
        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{model_name.lower().replace(' ', '_')}_best.pt")
            
    # Salvar modelo final
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{model_name.lower().replace(' ', '_')}_best.pt")
    
    return history

def plot_curves(baseline_history, tv_history):
    os.makedirs("checkpoints", exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    # 1. Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(baseline_history["epoch"], baseline_history["loss"], label="Baseline (Loss)", color="blue", linestyle="--")
    plt.plot(tv_history["epoch"], tv_history["loss"], label="Think-Vetor (Loss)", color="red")
    plt.title("Perda de Treinamento (Loss)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 2. Plot Val Acc
    plt.subplot(1, 3, 2)
    plt.plot(baseline_history["epoch"], baseline_history["val_acc"], label="Baseline (Val Acc)", color="blue", linestyle="--")
    plt.plot(tv_history["epoch"], tv_history["val_acc"], label="Think-Vetor (Val Acc)", color="red")
    plt.title("Acurácia de Validação (Exact Match)")
    plt.xlabel("Época")
    plt.ylabel("Acurácia (%)")
    plt.legend()
    plt.grid(True)
    
    # 3. Plot Ponder Steps
    plt.subplot(1, 3, 3)
    plt.plot(tv_history["epoch"], tv_history["avg_steps"], label="Passos Médios", color="green")
    plt.title("Passos de Pensamento (Think-Vetor)")
    plt.xlabel("Época")
    plt.ylabel("Passos")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("checkpoints/training_curves.png", dpi=150)
    print("\n[INFO] Gráfico de treinamento salvo em 'checkpoints/training_curves.png'")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando no Google Colab usando o dispositivo: {device.type.upper()}")
    
    # Hiperparâmetros de larga escala para T4 GPU
    num_digits = 2
    tokenizer = CharTokenizer()
    
    print("Gerando datasets em larga escala...")
    # 15.000 amostras cobre abundantemente o espaço de combinações de 2 dígitos
    train_ds = AdditionDataset(num_digits=num_digits, num_samples=15000, seed=42, tokenizer=tokenizer)
    val_ds = AdditionDataset(num_digits=num_digits, num_samples=1000, seed=99, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # Configurar modelos com maior capacidade (2 camadas, d_model=128)
    baseline_model = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_ponder_steps=0
    )
    
    think_vetor = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_ponder_steps=6,
        num_memories=512,
        beta=8.0
    )
    
    epochs = 80
    
    # Treinamento
    baseline_history = train_colab_model(
        "Baseline Model", baseline_model, train_loader, val_loader, tokenizer, epochs=epochs, device=device, use_ponder=False
    )
    
    tv_history = train_colab_model(
        "Think Vetor", think_vetor, train_loader, val_loader, tokenizer, epochs=epochs, device=device, use_ponder=True
    )
    
    # Plotar e salvar resultados
    try:
        plot_curves(baseline_history, tv_history)
    except Exception as e:
        print(f"Aviso: Não foi possível gerar os gráficos usando matplotlib: {e}")

if __name__ == "__main__":
    main()
