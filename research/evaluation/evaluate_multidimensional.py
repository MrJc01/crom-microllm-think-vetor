import os
import random
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader

from src.logic_dataset import LogicCharTokenizer, LogicDataset
from src.logic_llm import MicroReasoningLLM

# Simple PyTorch linear probe model
class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def extract_probe_targets(extra_info):
    names_sorted = sorted(extra_info["rel1_order"])
    
    # Target 1: Is names_sorted[0] > names_sorted[-1] for Attribute 1?
    order1 = extra_info["rel1_order"]
    label1 = 1.0 if order1.index(names_sorted[0]) < order1.index(names_sorted[-1]) else 0.0
    
    # Target 2: Is names_sorted[0] > names_sorted[-1] for Attribute 2?
    order2 = extra_info.get("rel2_order")
    if order2 is not None:
        label2 = 1.0 if order2.index(names_sorted[0]) < order2.index(names_sorted[-1]) else 0.0
    else:
        label2 = 0.0
    
    return label1, label2

def compute_attention_entropy(model, input_ids, device):
    model.eval()
    with torch.no_grad():
        x_emb = model.token_embeddings(input_ids.to(device))
        
        # Adiciona embeddings de posição se configurado
        if model.use_pos_embedding:
            x_emb = model.pos_encoder(x_emb)
            
        if model.use_rope:
            x_encoded = model.encoder(x_emb, rope=model.rope)
        else:
            x_encoded = model.encoder(x_emb)
            
        batch_size, seq_len, d_model = x_encoded.shape
        current_state = x_encoded
        init_temp = 0.0 # determinístico
        
        entropy_per_step = []
        
        for k in range(model.max_ponder_steps):
            step_idx = min(k, model.step_embeddings.shape[0] - 1) if hasattr(model, 'step_embeddings') else 0
            step_emb = model.step_embeddings[step_idx].view(1, 1, d_model) if hasattr(model, 'step_embeddings') else torch.zeros(1, 1, d_model, device=device)
            state_with_time = current_state + step_emb
            
            # Recozimento térmico
            if model.max_ponder_steps > 1:
                attn_temp = 2.0 - k * (2.0 - 0.2) / (model.max_ponder_steps - 1)
            else:
                attn_temp = 1.0
            attn_temp = max(attn_temp, 0.2)
            
            if model.use_rope:
                next_state = model.recurrent_layer(state_with_time, temp=attn_temp, rope=model.rope)
            else:
                next_state = model.recurrent_layer(state_with_time, temp=attn_temp)
                
            # Extrair os pesos de atenção salvos
            attn_weights = model.recurrent_layer.self_attn.last_attn_weights
            
            # Shannon Entropy por cabeça: -sum(p * log(p + eps))
            entropy = - (attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1) # (B, nhead, S)
            mean_entropy = entropy.mean(dim=(0, 2)) # (nhead,)
            entropy_per_step.append(mean_entropy.cpu().numpy())
            
            next_state = model.hopfield_ebm(next_state, temp=init_temp, lr=0.1)
            current_state = next_state
            
        return np.array(entropy_per_step) # (max_ponder_steps, nhead)

def train_probe(X_train, y_train, X_val, y_val, input_dim, lr=0.01, epochs=100):
    probe = LinearProbe(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    probe.eval()
    with torch.no_grad():
        val_outputs = probe(X_val)
        preds = (torch.sigmoid(val_outputs) >= 0.5).float()
        acc = (preds == y_val).float().mean().item()
        
    return probe, acc

def main():
    parser = argparse.ArgumentParser(description="Script de avaliação científica para a Transitividade Multidimensional.")
    parser.add_argument("--model_path", type=str, default="checkpoints/think_vetor_hybrid_best.pt", help="Caminho do checkpoint do Think-Vetor.")
    parser.add_argument("--num_samples", type=int, default=200, help="Quantidade de amostras para avaliação de acurácia.")
    parser.add_argument("--probe_samples", type=int, default=500, help="Quantidade de amostras para o treinamento dos Probes.")
    parser.add_argument("--tokenizer_name", type=str, default="char", help="Tokenizer ('char' ou HuggingFace path).")
    parser.add_argument("--num_entities", type=int, default=3, help="Número de entidades no dataset lógico (3 a 6).")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AVALIAÇÃO CIENTÍFICA: TRANSITIVIDADE MULTIDIMENSIONAL ===")
    print(f"Dispositivo de avaliação: {device.type.upper()}")
    print(f"Número de entidades: {args.num_entities}")
    
    if args.tokenizer_name == "char":
        tokenizer = LogicCharTokenizer()
    else:
        from src.hf_tokenizer_wrapper import HFTokenizerWrapper
        tokenizer = HFTokenizerWrapper(args.tokenizer_name)
        
    # 1. Carregar modelo - detectar vocab_size do checkpoint
    checkpoint_vocab_size = tokenizer.vocab_size
    if os.path.exists(args.model_path):
        ckpt_state = torch.load(args.model_path, map_location=device, weights_only=True)
        if "token_embeddings.weight" in ckpt_state:
            checkpoint_vocab_size = ckpt_state["token_embeddings.weight"].shape[0]
            print(f"[INFO] Vocab size detectado no checkpoint: {checkpoint_vocab_size}")
        if checkpoint_vocab_size != tokenizer.vocab_size:
            print(f"[INFO] Usando LogicCharTokenizer (vocab={checkpoint_vocab_size}) para compatibilidade com checkpoint.")
            tokenizer = LogicCharTokenizer()
    
    model = MicroReasoningLLM(
        vocab_size=checkpoint_vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_ponder_steps=6,
        num_memories=256,
        beta=8.0,
        use_pos_embedding=False,
        use_rope=True
    ).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(ckpt_state)
        print(f"[INFO] Pesos do modelo carregados de: {args.model_path}")
    else:
        print(f"[AVISO] Checkpoint não encontrado em {args.model_path}. Usando pesos inicializados aleatoriamente.")
        
    # 2. Avaliação de Acurácia Exact Match (OOD Multidimensional)
    print("\n--- 1. Avaliando Acurácia Exata no Dataset Multidimensional ---")
    val_ds = LogicDataset(num_samples=args.num_samples, seed=888, tokenizer=tokenizer, multidimensional=(args.num_entities==3), num_entities=args.num_entities, mutable_context=False)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            target_str = item["raw_target"]
            
            gen_ids = model.generate(input_ids, max_length=15, temperature=0.0)
            pred_str = tokenizer.decode(gen_ids[0]).strip()
            
            if pred_str == target_str:
                correct += 1
                
    accuracy = (correct / len(val_ds)) * 100
    print(f"Acurácia Exact Match Multidimensional: {accuracy:.2f}% ({correct}/{len(val_ds)})")
    
    # 3. Treinamento de Probes Lineares (Desacoplamento de Embeddings)
    print("\n--- 2. Treinando Probes Lineares no Espaço de Embeddings ---")
    probe_ds = LogicDataset(num_samples=args.probe_samples, seed=999, tokenizer=tokenizer, multidimensional=(args.num_entities==3), num_entities=args.num_entities, mutable_context=False)
    
    embeddings = []
    labels1 = []
    labels2 = []
    
    with torch.no_grad():
        for i in range(len(probe_ds)):
            item = probe_ds[i]
            extra_info = probe_ds.samples[i][3]
            
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            # Rodar o ponderador para extrair z_thought (pooled_states da última posição)
            _, _, pooled_states, _ = model(input_ids, item["target_ids"].unsqueeze(0).to(device), return_details=True)
            z_thought = pooled_states[0, -1, :].cpu() # (d_model,)
            
            lbl1, lbl2 = extract_probe_targets(extra_info)
            
            embeddings.append(z_thought)
            labels1.append(lbl1)
            labels2.append(lbl2)
            
    X = torch.stack(embeddings) # (N, d_model)
    y1 = torch.tensor(labels1).unsqueeze(1) # (N, 1)
    y2 = torch.tensor(labels2).unsqueeze(1) # (N, 1)
    
    # Split em treino/val (80% / 20%)
    split_idx = int(0.8 * len(probe_ds))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y1_train, y1_val = y1[:split_idx], y1[split_idx:]
    y2_train, y2_val = y2[:split_idx], y2[split_idx:]
    
    print(f"Dataset de Probe: {len(X)} amostras total (Treino: {len(X_train)} | Validação: {len(X_val)})")
    
    probe1, acc1 = train_probe(X_train, y1_train, X_val, y1_val, input_dim=128)
    print(f"Acurácia do Probe 1 (Atributo 1 - Relação Lógica 1): {acc1 * 100:.2f}%")
    
    if args.num_entities == 3:
        probe2, acc2 = train_probe(X_train, y2_train, X_val, y2_val, input_dim=128)
        print(f"Acurácia do Probe 2 (Atributo 2 - Relação Lógica 2): {acc2 * 100:.2f}%")
        
        # Calcular cosseno e ortogonalidade
        w1 = probe1.linear.weight.data.squeeze()
        w2 = probe2.linear.weight.data.squeeze()
        cos_sim = torch.cosine_similarity(w1, w2, dim=0).item()
        angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
        
        print(f"Similaridade de Cosseno entre os Pesos dos Probes: {cos_sim:.4f}")
        print(f"Ângulo de Ortogonalidade dos Probes: {angle:.2f}°")
    else:
        print("[INFO] Ignorando Probe 2 / Ortogonalidade pois o dataset de num_entities > 3 é unidimensional.")
    
    # 4. Mapeamento de Entropia de Atenção
    print("\n--- 3. Mapeamento da Entropia de Atenção ao Longo da Reflexão ---")
    entropy_ds = LogicDataset(num_samples=20, seed=777, tokenizer=tokenizer, multidimensional=True, mutable_context=False)
    input_ids_batch = torch.stack([item["input_ids"] for item in entropy_ds]).to(device)
    
    entropies = compute_attention_entropy(model, input_ids_batch, device) # (max_ponder_steps, nhead)
    
    print(f"{'Passo k':<10} | " + " | ".join([f"Head {h}" for h in range(entropies.shape[1])]))
    print("-" * (12 + 10 * entropies.shape[1]))
    for k in range(entropies.shape[0]):
        head_strs = [f"{entropies[k, h]:.4f}" for h in range(entropies.shape[1])]
        print(f"Passo {k+1:02d}   | " + " | ".join(head_strs))
        
    print("\n*Nota: A queda da entropia indica que as cabeças de atenção estão se focalizando (esfriando) à medida que convergem.*")

if __name__ == "__main__":
    main()
