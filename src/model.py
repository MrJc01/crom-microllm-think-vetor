import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    """
    Injeta codificação posicional senoidal (Vaswani et al. 2017) que suporta
    extrapolação e generalização OOD inerente para sequências mais longas.
    """
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)].to(x.device)

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) que aplica rotações angulares em pares de coordenadas.
    Permite generalização e extrapolação para comprimentos de contexto arbitrários.
    """
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cached = None
        self.sin_cached = None

    def _update_cache(self, max_seq_len, device):
        t = torch.arange(max_seq_len, dtype=torch.float, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(1) # (1, 1, max_seq_len, head_dim)
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(1) # (1, 1, max_seq_len, head_dim)

    def _rotate_half(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        # x: (B, nhead, S, head_dim)
        S = x.shape[2]
        if self.cos_cached is None or S > self.cos_cached.shape[2] or self.cos_cached.device != x.device:
            self._update_cache(max(S, 2 * (self.cos_cached.shape[2] if self.cos_cached is not None else 128)), x.device)
            
        cos = self.cos_cached[:, :, :S, :].to(x.device)
        sin = self.sin_cached[:, :, :S, :].to(x.device)
        return (x * cos) + (self._rotate_half(x) * sin)

class LangevinHopfieldBlock(nn.Module):
    """
    Minimização de energia no espaço latente utilizando Dinâmica de Langevin.
    Aproxima o estado oculto em direção a atretores (memórias associativas).
    """
    def __init__(self, d_model, num_memories=512, beta=8.0):
        super().__init__()
        self.d_model = d_model
        self.beta = beta
        # Memórias armazenadas (padrões de atrator aprendíveis)
        self.memories = nn.Parameter(torch.randn(num_memories, d_model) * (d_model ** -0.5))

    def compute_energy(self, z):
        """
        Calcula a energia de Hopfield livre do estado z.
        E(z) = - (1/beta) * log(sum(exp(beta * X * z))) + 0.5 * ||z||^2
        """
        # z: (batch_size, seq_len, d_model)
        logits = torch.matmul(z, self.memories.T) * self.beta # (B, S, num_memories)
        
        # Log-Sum-Exp escalado
        lse = torch.logsumexp(logits, dim=-1) / self.beta # (B, S)
        
        # Regularização quadrática
        quadratic = 0.5 * torch.norm(z, p=2, dim=-1) ** 2 # (B, S)
        
        # Energia média por vetor latente
        energy = - lse + quadratic
        return energy.mean()

    def forward(self, z, temp=0.1, lr=0.1, return_entropy=False):
        """
        Executa uma única iteração de descida de gradiente de energia + ruído estocástico.
        z: (batch_size, seq_len, d_model)
        """
        # Gradiente da energia: dE/dz = z - X_T * Softmax(beta * X * z)
        # 1. Similaridade com os padrões armazenados (chaves de memória)
        logits = torch.matmul(z, self.memories.T) * self.beta
        attn_weights = F.softmax(logits, dim=-1)
        
        # 2. Estado recuperado
        retrieved = torch.matmul(attn_weights, self.memories)
        
        # 3. Gradiente da função de energia quadrática regularizada
        grad = z - retrieved
        
        # 4. Atualização via descida de gradiente
        z_next = z - lr * grad
        
        # 5. Adicionar ruído estocástico de Langevin (Simulated Annealing)
        if temp > 0.0:
            noise = torch.randn_like(z)
            # Desvio padrão do ruído térmico
            noise_scale = torch.sqrt(torch.tensor(2.0 * temp * lr, device=z.device))
            z_next = z_next + noise_scale * noise
            
        if return_entropy:
            # p_avg: distribuição média das ativações das memórias ao longo do lote e da sequência
            p_avg = attn_weights.mean(dim=(0, 1)) # (num_memories,)
            entropy = - (p_avg * torch.log(p_avg + 1e-8)).sum()
            return z_next, entropy
            
        return z_next

class MultiHeadAttentionWithTemperature(nn.Module):
    """
    Auto-Atenção Multi-Head que suporta fator de escala de temperatura variável,
    uso de máscara causal e injeção opcional de Rotary Position Embedding (RoPE).
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0, "d_model deve ser divisível por nhead"
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv=None, temp=1.0, attn_mask=None, rope=None):
        # x: (B, S, d_model)
        # kv: (B, S_kv, d_model) - opcional para cross-attention
        B, S, C = x.shape
        if kv is None:
            kv = x
        S_kv = kv.shape[1]
        
        # 1. Obter Q, K, V e remodelar para multi-head
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        
        # Aplicar RoPE se fornecido
        if rope is not None:
            q = rope(q)
            if kv is x:
                k = rope(k)
        
        # 2. Calcular scores de atenção: Q @ K^T / (sqrt(head_dim) * temp)
        scale = (self.head_dim ** -0.5) / max(temp, 1e-6)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Aplicar máscara se fornecida
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # 3. Aplicar Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. Combinar com os valores
        context = torch.matmul(attn_weights, v) # (B, nhead, S, head_dim)
        
        # 5. Transpor de volta e concatenar heads
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        
        return self.out_proj(context)

class CustomTransformerEncoderLayer(nn.Module):
    """
    Camada customizada de codificador que suporta injeção de RoPE nos Query/Key.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithTemperature(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, rope=None):
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, rope=rope)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        ff_out = self.linear2(F.relu(self.linear1(x_norm)))
        x = x + ff_out
        return x

class CustomTransformerEncoder(nn.Module):
    """
    Codificador customizado composto de múltiplas camadas CustomTransformerEncoderLayer.
    """
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        
    def forward(self, x, rope=None):
        for layer in self.layers:
            x = layer(x, rope=rope)
        return x

class CustomTransformerDecoderLayer(nn.Module):
    """
    Camada customizada de decodificador que suporta injeção de RoPE na auto-atenção.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithTemperature(d_model, nhead)
        self.multihead_attn = MultiHeadAttentionWithTemperature(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, rope=None):
        # Auto-atenção com máscara causal
        tgt_norm = self.norm1(tgt)
        attn_out = self.self_attn(tgt_norm, attn_mask=tgt_mask, rope=rope)
        tgt = tgt + attn_out
        
        # Atenção cruzada (Key/Value vêm da memória do codificador)
        tgt_norm = self.norm2(tgt)
        attn_out2 = self.multihead_attn(tgt_norm, kv=memory)
        tgt = tgt + attn_out2
        
        # FFN
        tgt_norm = self.norm3(tgt)
        ff_out = self.linear2(F.relu(self.linear1(tgt_norm)))
        tgt = tgt + ff_out
        return tgt

class CustomTransformerDecoder(nn.Module):
    """
    Decodificador customizado composto de múltiplas camadas CustomTransformerDecoderLayer.
    """
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([CustomTransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)])
        
    def forward(self, tgt, memory, tgt_mask=None, rope=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, rope=rope)
        return tgt

class RecurrentTransformerLayer(nn.Module):
    """
    Bloco Transformer customizado para o loop recorrente.
    Implementa self-attention com temperatura variável + FFN com conexões residuais e LayerNorm.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithTemperature(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, temp=1.0, rope=None):
        # Subcamada 1: Atenção com temperatura + residual
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, temp=temp, rope=rope)
        x = x + attn_out
        
        # Subcamada 2: FeedForward + residual
        x_norm = self.norm2(x)
        ff_out = self.linear2(F.relu(self.linear1(x_norm)))
        x = x + ff_out
        
        return x

class ThinkVetorModel(nn.Module):
    """
    Modelo Encoder-Decoder Think-Vetor.
    O Encoder processa o contexto inicial.
    O Bloco de Ponderação executa loops recorrentes latentes e Langevin-Hopfield.
    O Decoder gera autoregressivamente a resposta com base na representação ponderada.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2, 
                 num_decoder_layers=2, max_ponder_steps=6, num_memories=512, beta=8.0,
                 use_pos_embedding=False, use_rope=False):
        super().__init__()
        self.d_model = d_model
        self.max_ponder_steps = max_ponder_steps
        self.vocab_size = vocab_size
        self.use_pos_embedding = use_pos_embedding
        self.use_rope = use_rope
        
        # Camada de Embeddings compartilhado entre tokens de entrada e saída
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Injeção de Posição
        if self.use_pos_embedding:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model)
            self.pos_decoder = SinusoidalPositionalEncoding(d_model)
        
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(dim=d_model // nhead)
            # Codificador inicial customizado
            self.encoder = CustomTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_encoder_layers)
            # Decodificador final customizado
            self.decoder = CustomTransformerDecoder(d_model=d_model, nhead=nhead, num_layers=num_decoder_layers)
        else:
            self.rope = None
            # Codificador inicial padrão (compatibilidade com checkpoints existentes)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            
            # Decodificador final padrão
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Camada recorrente para loop de reflexão com suporte a temperatura
        self.recurrent_layer = RecurrentTransformerLayer(d_model=d_model, nhead=nhead)
        # Embeddings de passo de reflexão (para dar noção do tempo interno do loop)
        if max_ponder_steps > 0:
            self.step_embeddings = nn.Parameter(torch.randn(max_ponder_steps, d_model) * 0.02)
            # Bloco Baseado em Energia (Langevin-Hopfield)
            self.hopfield_ebm = LangevinHopfieldBlock(d_model=d_model, num_memories=num_memories, beta=beta)
            # Classificador para probabilidade de parada (PonderNet)
            self.halt_classifier = nn.Linear(d_model, 1)
        
        # Cabeçalho de projeção linear para vocabulário
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, target_ids, return_details=False, max_ponder_steps=None):
        """
        forward usado durante o treinamento (Teacher Forcing).
        input_ids: (batch_size, input_seq_len)
        target_ids: (batch_size, target_seq_len)
        """
        # 1. Embeddings e codificação inicial
        x_emb = self.token_embeddings(input_ids)
        
        # Adiciona embeddings de posição absolutos se configurado
        if self.use_pos_embedding:
            x_emb = self.pos_encoder(x_emb)
        
        if self.use_rope:
            x_encoded = self.encoder(x_emb, rope=self.rope)
        else:
            x_encoded = self.encoder(x_emb)
        
        batch_size, seq_len, d_model = x_encoded.shape
        device = x_encoded.device
        
        # Inicializar variáveis da Ponderação Latente
        halting_probabilities = []
        pooled_states = torch.zeros_like(x_encoded)
        intermediate_states = []
        
        actual_ponder_steps = max_ponder_steps if max_ponder_steps is not None else self.max_ponder_steps
        
        if actual_ponder_steps > 0:
            accumulated_remainders = torch.ones(batch_size, seq_len, 1, device=device)
            current_state = x_encoded
            init_temp = 0.5 # temperatura inicial do ruído
            
            for k in range(actual_ponder_steps):
                # Injeta a noção posicional do passo (proteção contra extrapolação temporal)
                step_idx = min(k, self.step_embeddings.shape[0] - 1) if hasattr(self, 'step_embeddings') else 0
                step_emb = self.step_embeddings[step_idx].view(1, 1, d_model) if hasattr(self, 'step_embeddings') else torch.zeros(1, 1, d_model, device=device)
                state_with_time = current_state + step_emb
                
                # Resfriamento térmico de atenção linear (Simulated Annealing de Atenção)
                if actual_ponder_steps > 1:
                    attn_temp = 2.0 - k * (2.0 - 0.2) / (actual_ponder_steps - 1)
                else:
                    attn_temp = 1.0
                attn_temp = max(attn_temp, 0.2) # Limitar inferiormente para evitar temp negativa
                    
                # Executa um ciclo da atenção recorrente com temperatura e RoPE
                if self.use_rope:
                    next_state = self.recurrent_layer(state_with_time, temp=attn_temp, rope=self.rope)
                else:
                    next_state = self.recurrent_layer(state_with_time, temp=attn_temp)
                
                # Aplica decaimento térmico de Langevin no atretor de Hopfield
                temp = init_temp * (0.5 ** k) # resfriamento
                next_state = self.hopfield_ebm(next_state, temp=temp, lr=0.1)
                
                # Calcula probabilidade de parada
                halt_prob = torch.sigmoid(self.halt_classifier(next_state))
                
                # PonderNet logic
                if k == actual_ponder_steps - 1:
                    step_halt_prob = accumulated_remainders
                else:
                    step_halt_prob = halt_prob * accumulated_remainders
                    accumulated_remainders = accumulated_remainders * (1.0 - halt_prob)
                
                # Acumula representação ponderada
                pooled_states = pooled_states + step_halt_prob * next_state
                halting_probabilities.append(step_halt_prob)
                intermediate_states.append(next_state[:, -1, :])
                
                current_state = next_state
        else:
            # Caso especial: modelo baseline sem loop de ponderação
            pooled_states = x_encoded
            # Força probabilidade 1.0 no passo único simulado
            halting_probabilities = [torch.ones(batch_size, seq_len, 1, device=device)]
            
        # 4. Processamento do Decoder (usando Teacher Forcing)
        # Shift right do target para treinamento autoregressivo
        # Prepend '=' (índice 11) como token <sos> e remove o último token
        sos_token = torch.full((batch_size, 1), 11, dtype=torch.long, device=device)
        shifted_target_ids = torch.cat([sos_token, target_ids[:, :-1]], dim=1)
        tgt_emb = self.token_embeddings(shifted_target_ids)
        
        # Adiciona embeddings de posição absolutos se configurado
        if self.use_pos_embedding:
            tgt_emb = self.pos_decoder(tgt_emb)
        
        # Máscara causal para o decoder não olhar para o futuro
        tgt_seq_len = target_ids.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
        
        # Decoder cruza atenção com os estados ocultos ponderados (pooled_states)
        if self.use_rope:
            decoded = self.decoder(tgt_emb, pooled_states, tgt_mask=causal_mask, rope=self.rope)
        else:
            decoded = self.decoder(tgt_emb, pooled_states, tgt_mask=causal_mask)
        
        # Mapeia de volta para o tamanho do vocabulário
        logits = self.lm_head(decoded)
        
        if return_details:
            return logits, halting_probabilities, pooled_states, intermediate_states
        
        return logits, halting_probabilities

    def generate(self, input_ids, max_length=5, temperature=1.0, max_ponder_steps=None):
        """
        Inspeciona e gera a resposta autoregressiva token por token.
        Usado em inferência.
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            # 1. Encodificação inicial
            x_emb = self.token_embeddings(input_ids)
            
            # Adiciona embeddings de posição absolutos se configurado
            if self.use_pos_embedding:
                x_emb = self.pos_encoder(x_emb)
            
            if self.use_rope:
                x_encoded = self.encoder(x_emb, rope=self.rope)
            else:
                x_encoded = self.encoder(x_emb)
            
            # 2. Ponderação Latente
            pooled_states = torch.zeros_like(x_encoded)
            actual_ponder_steps = max_ponder_steps if max_ponder_steps is not None else self.max_ponder_steps
            
            if actual_ponder_steps > 0:
                accumulated_remainders = torch.ones(batch_size, x_encoded.shape[1], 1, device=device)
                current_state = x_encoded
                init_temp = 0.0 # Sem ruído estocástico na inferência pura para manter determinismo
                
                for k in range(actual_ponder_steps):
                    step_idx = min(k, self.step_embeddings.shape[0] - 1) if hasattr(self, 'step_embeddings') else 0
                    step_emb = self.step_embeddings[step_idx].view(1, 1, self.d_model) if hasattr(self, 'step_embeddings') else torch.zeros(1, 1, self.d_model, device=device)
                    state_with_time = current_state + step_emb
                    
                    # Resfriamento térmico de atenção linear na inferência
                    if actual_ponder_steps > 1:
                        attn_temp = 2.0 - k * (2.0 - 0.2) / (actual_ponder_steps - 1)
                    else:
                        attn_temp = 1.0
                    attn_temp = max(attn_temp, 0.2) # Limitar inferiormente para evitar temp negativa
                        
                    if self.use_rope:
                        next_state = self.recurrent_layer(state_with_time, temp=attn_temp, rope=self.rope)
                    else:
                        next_state = self.recurrent_layer(state_with_time, temp=attn_temp)
                    next_state = self.hopfield_ebm(next_state, temp=init_temp, lr=0.1)
                    
                    halt_prob = torch.sigmoid(self.halt_classifier(next_state))
                    if k == actual_ponder_steps - 1:
                        step_halt_prob = accumulated_remainders
                    else:
                        step_halt_prob = halt_prob * accumulated_remainders
                        accumulated_remainders = accumulated_remainders * (1.0 - halt_prob)
                    
                    pooled_states = pooled_states + step_halt_prob * next_state
                    current_state = next_state
            else:
                pooled_states = x_encoded
                
            # 3. Geração Autoregressiva
            # Começa com o token de espaço (padding) como primeiro token gerado
            generated = torch.full((batch_size, 1), 11, dtype=torch.long, device=device) # '=' é o token de gatilho
            
            for _ in range(max_length):
                tgt_emb = self.token_embeddings(generated)
                
                # Adiciona embeddings de posição absolutos se configurado
                if self.use_pos_embedding:
                    tgt_emb = self.pos_decoder(tgt_emb)
                
                tgt_seq_len = generated.shape[1]
                causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
                
                if self.use_rope:
                    decoded = self.decoder(tgt_emb, pooled_states, tgt_mask=causal_mask, rope=self.rope)
                else:
                    decoded = self.decoder(tgt_emb, pooled_states, tgt_mask=causal_mask)
                logits = self.lm_head(decoded[:, -1, :])
                
                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                generated = torch.cat([generated, next_token], dim=1)
                
            # Retorna apenas a parte gerada (removendo o primeiro caractere que foi o gatilho '=')
            return generated[:, 1:]

if __name__ == "__main__":
    # Teste de integridade de dimensões (sem RoPE)
    print("Testando modelo padrão sem RoPE...")
    model_std = ThinkVetorModel(vocab_size=13, d_model=64, nhead=2, max_ponder_steps=4, use_pos_embedding=True)
    x = torch.randint(0, 13, (2, 8)) # batch=2, seq=8
    y = torch.randint(0, 13, (2, 4)) # batch=2, seq=4
    logits, halts = model_std(x, y)
    print("Sem RoPE -> Logits shape:   ", logits.shape) # Esperado: (2, 4, 13)
    print("Sem RoPE -> Número de halts:", len(halts))   # Esperado: 4 (max_ponder_steps)
    
    gen = model_std.generate(x, max_length=4, temperature=0.0)
    print("Sem RoPE -> Gerado shape:   ", gen.shape) # Esperado: (2, 4)
    
    # Teste de integridade de dimensões (com RoPE)
    print("\nTestando modelo com RoPE...")
    model_rope = ThinkVetorModel(vocab_size=13, d_model=64, nhead=2, max_ponder_steps=4, use_rope=True)
    logits, halts = model_rope(x, y)
    print("Com RoPE -> Logits shape:   ", logits.shape) # Esperado: (2, 4, 13)
    print("Com RoPE -> Número de halts:", len(halts))   # Esperado: 4
    
    gen = model_rope.generate(x, max_length=4, temperature=0.0)
    print("Com RoPE -> Gerado shape:   ", gen.shape) # Esperado: (2, 4)
    print("\nTodos os testes de dimensão passaram com sucesso!")
