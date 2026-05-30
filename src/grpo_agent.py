import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel
from train import compute_ponder_loss, evaluate_accuracy

class GRPOAgent:
    """
    Agente de Aprendizado por Reforço que implementa Group Relative Policy Optimization (GRPO)
    para otimização de trajetórias contínuas e predição textual final.
    """
    def __init__(self, model, optimizer, tokenizer, group_size=4, kl_coef=0.01):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_grpo_loss(self, input_ids, target_ids, target_mask, device):
        # input_ids: (B, L)
        # target_ids: (B, T)
        # target_mask: (B, T)
        B, L = input_ids.shape
        G = self.group_size
        
        # 1. Replicar entradas G vezes para amostragem do grupo
        # input_ids_rep: (B * G, L)
        input_ids_rep = input_ids.repeat_interleave(G, dim=0)
        target_ids_rep = target_ids.repeat_interleave(G, dim=0)
        target_mask_rep = target_mask.repeat_interleave(G, dim=0)
        
        # 2. Gerar respostas usando inferência com temperatura
        self.model.eval()
        rewards = []
        with torch.no_grad():
            # max_length do target
            max_len = target_ids.shape[1]
            # Gerar com temperatura para obter diversidade no grupo
            gen_ids = self.model.generate(input_ids_rep, max_length=max_len, temperature=0.7)
            
            # Avaliar recompensa para cada geração (1.0 se for Exact Match com o alvo, 0.0 caso contrário)
            for idx, gen_seq in enumerate(gen_ids):
                pred_str = self.tokenizer.decode(gen_seq).strip()
                target_str = self.tokenizer.decode(target_ids_rep[idx]).strip()
                
                # Recompensa baseada no gabarito
                reward = 1.0 if pred_str == target_str else 0.0
                rewards.append(reward)
                
        self.model.train()
        rewards = torch.tensor(rewards, dtype=torch.float, device=device) # (B * G)
        
        # 3. Calcular vantagens relativas do grupo (GRPO)
        rewards_grouped = rewards.view(B, G)
        mean_r = rewards_grouped.mean(dim=-1, keepdim=True) # (B, 1)
        std_r = rewards_grouped.std(dim=-1, keepdim=True) + 1e-8 # (B, 1)
        
        # Normalização relativa dentro do grupo
        advantages = (rewards_grouped - mean_r) / std_r # (B, G)
        advantages = advantages.view(-1) # (B * G)
        
        # 4. Forward pass e cálculo do gradiente de política
        self.optimizer.zero_grad()
        
        logits, halts = self.model(input_ids_rep, target_ids_rep) # (B * G, T, V)
        
        # Perda de Cross Entropy por token
        B_G, T, V = logits.shape
        logits_flat = logits.view(-1, V)
        targets_flat = target_ids_rep.view(-1)
        mask_flat = target_mask_rep.view(-1)
        
        ce_loss_raw = self.criterion(logits_flat, targets_flat) # (B * G * T)
        ce_loss_per_sequence = (ce_loss_raw.view(B_G, T) * target_mask_rep).sum(dim=-1) / (target_mask_rep.sum(dim=-1) + 1e-8) # (B * G)
        
        # Perda de política: CE ponderada pela vantagem negativa
        policy_loss = (ce_loss_per_sequence * (-advantages)).mean()
        
        # Termo de Supervised Fine-Tuning (SFT) para guiar o início do treino (cold start)
        sft_loss = ce_loss_per_sequence.mean()
        
        # Adicionar perda do PonderNet se aplicável
        if len(halts) > 0:
            p_loss = compute_ponder_loss(halts, prior_prob=0.3)
        else:
            p_loss = torch.tensor(0.0, device=device)
            
        total_loss = policy_loss + 0.1 * p_loss + 0.5 * sft_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), rewards.mean().item()

def train_grpo(model, dataloader, val_dataloader, tokenizer, epochs=5, device="cpu"):
    print("\n=== Iniciando Treinamento GRPO ===")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    agent = GRPOAgent(model, optimizer, tokenizer, group_size=4)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_reward = 0.0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            
            loss, p_loss, avg_reward = agent.compute_grpo_loss(input_ids, target_ids, target_mask, device)
            total_loss += loss
            total_policy_loss += p_loss
            total_reward += avg_reward
            
        val_acc, avg_steps = evaluate_accuracy(model, val_dataloader, tokenizer, device, use_ponder=True)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss Total: {total_loss/len(dataloader):.4f} | "
              f"Policy Loss: {total_policy_loss/len(dataloader):.4f} | "
              f"Avg Group Reward: {total_reward/len(dataloader)*100:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Passos Médios: {avg_steps:.2f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/think_vetor_grpo_best.pt")
            
    print(f"Treinamento GRPO finalizado. Melhor Acurácia de Validação: {best_acc:.2f}%")
    return best_acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo para GRPO: {device}")
    
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
    
    train_grpo(model, train_loader, val_loader, tokenizer, epochs=3, device=device)
