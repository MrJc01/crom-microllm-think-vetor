import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    """
    Função de perda para alinhar os estados latentes do modelo (z_k)
    com as representações dos tokens correspondentes no CoT textual.
    """
    def __init__(self, use_cosine=True, use_mse=True, weight_cosine=1.0, weight_mse=0.1):
        super().__init__()
        self.use_cosine = use_cosine
        self.use_mse = use_mse
        self.weight_cosine = weight_cosine
        self.weight_mse = weight_mse

    def forward(self, intermediate_states, cot_embeddings):
        # intermediate_states: lista de K tensores de formato (B, d_model) ou tensor (B, K, d_model)
        # cot_embeddings: tensor (B, K, d_model)
        if isinstance(intermediate_states, list):
            # stack para (B, K, d_model)
            intermediate_states = torch.stack(intermediate_states, dim=1)
            
        B, K, D = intermediate_states.shape
        loss = torch.tensor(0.0, device=intermediate_states.device)
        
        # 1. Cosine similarity loss (1 - cos(z_k, cot_k))
        if self.use_cosine:
            # F.cosine_similarity retorna (B, K)
            cos_sim = F.cosine_similarity(intermediate_states, cot_embeddings, dim=-1)
            loss = loss + self.weight_cosine * (1.0 - cos_sim).mean()
            
        # 2. MSE loss (mean squared error)
        if self.use_mse:
            mse = F.mse_loss(intermediate_states, cot_embeddings)
            loss = loss + self.weight_mse * mse
            
        return loss
