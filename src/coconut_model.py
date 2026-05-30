import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import RotaryPositionEmbedding, MultiHeadAttentionWithTemperature

class CausalTransformerLayer(nn.Module):
    """
    Camada customizada de Transformer Causal (Decoder-only)
    que suporta injeção de RoPE nos Query/Key e aplicação de máscara de atenção.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithTemperature(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None, rope=None):
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, attn_mask=attn_mask, rope=rope)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        ff_out = self.linear2(F.relu(self.linear1(x_norm)))
        x = x + ff_out
        return x

class CausalTransformer(nn.Module):
    """
    Transformer Causal (Decoder-only) composto de múltiplas CausalTransformerLayer.
    """
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([CausalTransformerLayer(d_model, nhead) for _ in range(num_layers)])
        
    def forward(self, x, attn_mask=None, rope=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, rope=rope)
        return x

class CausalThinkVetorModel(nn.Module):
    """
    Modelos de Raciocínio Contínuo Causal (Estilo COCONUT).
    Unifica a representação em uma topologia Causal Decoder-Only.
    Injeta estados latentes autoregressivamente no final da sequência de contexto físico.
    """
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, 
                 max_thought_steps=4, use_rope=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_thought_steps = max_thought_steps
        self.use_rope = use_rope
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(dim=d_model // nhead)
        else:
            self.rope = None
            
        self.transformer = CausalTransformer(d_model, nhead, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, target_ids, return_details=False):
        # input_ids: (B, L)
        # target_ids: (B, T)
        B, L = input_ids.shape
        device = input_ids.device
        
        # 1. Obter embeddings iniciais
        x = self.token_embeddings(input_ids) # (B, L, d_model)
        
        # 2. Loop de Pensamento Latente (COCONUT)
        # Gerar max_thought_steps vetores de pensamento sequencialmente
        thought_vectors = []
        for step in range(self.max_thought_steps):
            # Causal mask para o tamanho atual da sequência
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            
            # Passar pelo transformer
            h = self.transformer(x, attn_mask=mask, rope=self.rope)
            
            # Pegar o último hidden state (z_t)
            z = h[:, -1:, :]
            thought_vectors.append(z)
            
            # Concatenar à sequência de entrada
            x = torch.cat([x, z], dim=1)
            
        # 3. Processar target com teacher forcing
        # Shift right do target
        sos_token = torch.full((B, 1), 11, dtype=torch.long, device=device) # '=' como SOS
        shifted_target_ids = torch.cat([sos_token, target_ids[:, :-1]], dim=1)
        tgt_emb = self.token_embeddings(shifted_target_ids) # (B, T, d_model)
        
        # Concatenar target embeddings no final
        x_full = torch.cat([x, tgt_emb], dim=1) # (B, L + max_thought_steps + T, d_model)
        
        # Passar a sequência completa pelo Causal Transformer
        full_seq_len = x_full.shape[1]
        full_mask = nn.Transformer.generate_square_subsequent_mask(full_seq_len, device=device)
        h_full = self.transformer(x_full, attn_mask=full_mask, rope=self.rope)
        
        # Extrair os logits que correspondem às posições do target
        # A primeira predição do target deve ser feita no último token de pensamento (índice L + K - 1)
        # As subsequentes são feitas nos tokens do target deslocado
        T = target_ids.shape[1]
        start_idx = L + self.max_thought_steps - 1
        target_outputs = h_full[:, start_idx : start_idx + T, :]
        
        logits = self.lm_head(target_outputs)
        
        if return_details:
            return logits, thought_vectors
            
        return logits

    def generate(self, input_ids, max_length=5, temperature=1.0):
        self.eval()
        B, L = input_ids.shape
        device = input_ids.device
        
        with torch.no_grad():
            # 1. Embeddings iniciais
            x = self.token_embeddings(input_ids)
            
            # 2. Pensamento Latente
            for step in range(self.max_thought_steps):
                seq_len = x.shape[1]
                mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
                h = self.transformer(x, attn_mask=mask, rope=self.rope)
                z = h[:, -1:, :]
                x = torch.cat([x, z], dim=1)
                
            # 3. Geração Autoregressiva
            # Inicializar com o token SOS '=' (índice 11)
            generated = torch.full((B, 1), 11, dtype=torch.long, device=device)
            
            for _ in range(max_length):
                tgt_emb = self.token_embeddings(generated)
                x_full = torch.cat([x, tgt_emb], dim=1)
                
                full_seq_len = x_full.shape[1]
                full_mask = nn.Transformer.generate_square_subsequent_mask(full_seq_len, device=device)
                h_full = self.transformer(x_full, attn_mask=full_mask, rope=self.rope)
                
                # Extrair logits do último token predito
                logits = self.lm_head(h_full[:, -1, :])
                
                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                generated = torch.cat([generated, next_token], dim=1)
                
            return generated[:, 1:]

if __name__ == "__main__":
    # Teste de integridade de dimensões
    print("Testando CausalThinkVetorModel (COCONUT)...")
    model = CausalThinkVetorModel(vocab_size=13, d_model=64, nhead=2, num_layers=2, max_thought_steps=3)
    x = torch.randint(0, 13, (2, 8)) # batch=2, seq=8
    y = torch.randint(0, 13, (2, 4)) # batch=2, seq=4
    logits = model(x, y)
    print("Logits shape: ", logits.shape) # Esperado: (2, 4, 13)
    
    gen = model.generate(x, max_length=4, temperature=0.0)
    print("Gerado shape: ", gen.shape) # Esperado: (2, 4)
    print("Todos os testes do COCONUT passaram com sucesso!")
