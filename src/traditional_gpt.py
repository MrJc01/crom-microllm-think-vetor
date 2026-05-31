import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import RotaryPositionEmbedding
from src.coconut_model import CausalTransformer

class TraditionalGPT(nn.Module):
    """
    Mini-GPT autoregressivo tradicional (Decoder-Only) para comparação.
    Não possui raciocínio contínuo latente. Consumirá mais tokens de contexto físico
    porque gera a cadeia de pensamento (CoT) textualmente no formato discreto.
    """
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, use_rope=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_rope = use_rope
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(dim=d_model // nhead)
        else:
            self.rope = None
            
        self.transformer = CausalTransformer(d_model, nhead, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, attn_mask=None):
        # x: (B, S)
        h = self.token_embeddings(x)
        S = x.shape[1]
        
        if attn_mask is None:
            # Gerar máscara causal
            attn_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
            
        # Passar pelo transformer causal
        h_out = self.transformer(h, attn_mask=attn_mask, rope=self.rope)
        
        logits = self.lm_head(h_out)
        return logits
        
    def generate(self, input_ids, max_length=50, temperature=1.0):
        """
        Geração autoregressiva tradicional.
        input_ids: (B, L)
        Retorna a sequência gerada (incluindo input_ids).
        """
        self.eval()
        device = input_ids.device
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)
                # Obter logits do último token
                next_token_logits = logits[:, -1, :]
                
                if temperature == 0.0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated

if __name__ == "__main__":
    print("=== Testando integridade estrutural do TraditionalGPT ===")
    vocab_size = 72
    model = TraditionalGPT(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=6, use_rope=True)
    
    # Contagem de parâmetros
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Número total de parâmetros: {num_params:,}")
    
    # Amostra de teste: Batch=2, Seq_len=16
    x = torch.randint(0, vocab_size, (2, 16))
    logits = model(x)
    print("Logits shape:   ", logits.shape)  # Esperado: (2, 16, 72)
    
    # Geração
    gen = model.generate(x, max_length=10, temperature=0.0)
    print("Gerado shape:   ", gen.shape)  # Esperado: (2, 26)
    print("Todos os testes de dimensão passaram com sucesso!")
