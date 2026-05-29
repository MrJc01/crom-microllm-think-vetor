import os
import torch
import numpy as np

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel

def analyze_trajectory(model, input_ids, device):
    """
    Analisa a trajetória dos vetores latentes durante o loop de ponderação.
    Calcula a distância euclidiana média e similaridade de cosseno entre passos consecutivos,
    demonstrando a convergência para um atretor estável (minimização de energia).
    """
    model.eval()
    with torch.no_grad():
        x_emb = model.token_embeddings(input_ids.to(device))
        x_encoded = model.encoder(x_emb)
        
        batch_size, seq_len, d_model = x_encoded.shape
        
        # Monitorar estados em cada passo
        states_per_step = [x_encoded]
        current_state = x_encoded
        init_temp = 0.0 # Inferência determinística
        
        for k in range(model.max_ponder_steps):
            step_emb = model.step_embeddings[k].view(1, 1, d_model)
            state_with_time = current_state + step_emb
            next_state = model.recurrent_layer(state_with_time)
            next_state = model.hopfield_ebm(next_state, temp=init_temp, lr=0.1)
            
            states_per_step.append(next_state)
            current_state = next_state
            
        # Calcular distâncias e similaridades consecutivas
        distances = []
        cos_similarities = []
        
        for i in range(1, len(states_per_step)):
            prev = states_per_step[i-1]
            curr = states_per_step[i]
            
            # Distância euclidiana média por vetor de token
            dist = torch.norm(curr - prev, dim=-1).mean().item()
            distances.append(dist)
            
            # Similaridade de cosseno média por vetor de token
            cos_sim = (F_cosine_sim(curr, prev)).mean().item()
            cos_similarities.append(cos_sim)
            
        return distances, cos_similarities

def F_cosine_sim(a, b):
    # a, b: (B, S, d)
    a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1)

def count_carries(a, b):
    """
    Calcula a quantidade de carries (transporte de dígitos) na soma de a e b.
    """
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        digit_a = a % 10
        digit_b = b % 10
        if digit_a + digit_b + carry > 9:
            carries += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return carries

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    
    # 1. Carregar modelos
    print("\n=== Instanciando Modelos ===")
    
    baseline = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_ponder_steps=0
    ).to(device)
    
    think_vetor = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_ponder_steps=6,
        num_memories=128,
        beta=8.0
    ).to(device)
    
    # Tentar carregar pesos salvos
    baseline_path = "checkpoints/baseline_model_best.pt"
    think_vetor_path = "checkpoints/think_vetor_best.pt"
    
    if os.path.exists(baseline_path):
        baseline.load_state_dict(torch.load(baseline_path, map_location=device))
        print("-> Pesos do Baseline carregados com sucesso.")
    else:
        print("-> AVISO: Pesos do Baseline não encontrados. Usando inicialização aleatória.")
        
    if os.path.exists(think_vetor_path):
        think_vetor.load_state_dict(torch.load(think_vetor_path, map_location=device))
        print("-> Pesos do Think-Vetor carregados com sucesso.")
    else:
        print("-> AVISO: Pesos do Think-Vetor não encontrados. Usando inicialização aleatória.")
        
    # 2. Criar Dataset de Teste (2 dígitos para in-distribution, 3 dígitos para out-of-distribution)
    print("\n=== Gerando Datasets de Teste ===")
    test_ds_2d = AdditionDataset(num_digits=2, num_samples=100, seed=123, tokenizer=tokenizer)
    test_ds_3d = AdditionDataset(num_digits=3, num_samples=100, seed=456, tokenizer=tokenizer)
    
    # 3. Teste de Acurácia e Carries
    print("\n=== Executando Testes ===")
    
    # Avaliar Baseline 2D
    baseline.eval()
    correct_b2, correct_b3 = 0, 0
    for item in test_ds_2d:
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        gen = baseline.generate(input_ids, max_length=3, temperature=0.0)
        pred = tokenizer.decode(gen[0]).strip()
        if pred == item["raw_target"]:
            correct_b2 += 1
            
    # Avaliar Baseline 3D (OOD)
    for item in test_ds_3d:
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        gen = baseline.generate(input_ids, max_length=4, temperature=0.0)
        pred = tokenizer.decode(gen[0]).strip()
        if pred == item["raw_target"]:
            correct_b3 += 1
            
    # Avaliar Think-Vetor 2D e 3D
    think_vetor.eval()
    correct_tv2, correct_tv3 = 0, 0
    
    # Coletar trajetórias de um subconjunto
    sample_trajectory_dists = []
    sample_trajectory_cos = []
    
    # Coletar estatísticas de passos
    steps_list = []
    
    # Grupos de Dificuldade baseados em carry
    difficulty_groups = {
        0: {"total": 0, "correct": 0, "steps": []}, # Fácil (0 carries)
        1: {"total": 0, "correct": 0, "steps": []}, # Médio (1 carry)
        2: {"total": 0, "correct": 0, "steps": []}  # Difícil (2 carries)
    }
    
    for idx, item in enumerate(test_ds_2d):
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        gen = think_vetor.generate(input_ids, max_length=3, temperature=0.0)
        pred = tokenizer.decode(gen[0]).strip()
        is_correct = (pred == item["raw_target"])
        if is_correct:
            correct_tv2 += 1
            
        # Calcular passos de parada ponderados
        _, halts = think_vetor(input_ids, input_ids[:, :1])
        p = torch.cat(halts, dim=-1)
        steps = (p * torch.arange(1, len(halts)+1, device=device).view(1, 1, -1)).sum(dim=-1).mean().item()
        steps_list.append((item["raw_input"], item["raw_target"], pred, steps))
        
        # Calcular o carry a partir do input raw (ex: "73+67=")
        raw_in = item["raw_input"].replace(" ", "").replace("=", "")
        try:
            parts = raw_in.split("+")
            a_val = int(parts[0])
            b_val = int(parts[1])
            carries = count_carries(a_val, b_val)
            group_key = min(carries, 2)
            
            difficulty_groups[group_key]["total"] += 1
            if is_correct:
                difficulty_groups[group_key]["correct"] += 1
            difficulty_groups[group_key]["steps"].append(steps)
        except Exception:
            pass
        
        # Analisar trajetória para os 5 primeiros exemplos
        if idx < 5:
            dists, cos_sims = analyze_trajectory(think_vetor, input_ids, device)
            sample_trajectory_dists.append(dists)
            sample_trajectory_cos.append(cos_sims)
            
    for item in test_ds_3d:
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        gen = think_vetor.generate(input_ids, max_length=4, temperature=0.0)
        pred = tokenizer.decode(gen[0]).strip()
        if pred == item["raw_target"]:
            correct_tv3 += 1
            
    print(f"\nAcurácia do Baseline (Sem Ponderação):")
    print(f"  - Em distribuição (2 dígitos): {correct_b2}%")
    print(f"  - Fora de distribuição (3 dígitos): {correct_b3}%")
    
    print(f"\nAcurácia do Think-Vetor (Com Ponderação):")
    print(f"  - Em distribuição (2 dígitos): {correct_tv2}%")
    print(f"  - Fora de distribuição (3 dígitos): {correct_tv3}%")
    
    # 4. Mostrar Tabela de Dificuldade vs. Tempo de Pensamento
    print("\n=== Análise Estatística: Dificuldade vs. Passos de Pensamento ===")
    print(f"{'Dificuldade (Carries)':<25} | {'Casos':<6} | {'Acurácia':<9} | {'Passos Médios de Pensamento'}")
    print("-" * 75)
    diff_names = {0: "Fácil (0 Carries)", 1: "Médio (1 Carry)", 2: "Difícil (2 Carries)"}
    for g_key in [0, 1, 2]:
        group = difficulty_groups[g_key]
        if group["total"] > 0:
            acc = (group["correct"] / group["total"]) * 100
            avg_steps_g = np.mean(group["steps"])
            print(f"{diff_names[g_key]:<25} | {group['total']:<6} | {acc:.1f}%    | {avg_steps_g:.2f}")
        else:
            print(f"{diff_names[g_key]:<25} | 0      | N/A       | N/A")
            
    # 5. Mostrar Tabela de Exemplos com Passos de Pensamento
    print("\n=== Amostra de Passos de Raciocínio (Think-Vetor) ===")
    print(f"{'Conta':<12} | {'Esperado':<8} | {'Predição':<8} | {'Passos de Pensamento'}")
    print("-" * 55)
    for raw_in, raw_tgt, pred, steps in steps_list[:15]:
        print(f"{raw_in:<12} | {raw_tgt:<8} | {pred:<8} | {steps:.2f}")
        
    # 6. Mostrar Análise de Trajetória Latente
    if len(sample_trajectory_dists) > 0:
        print("\n=== Análise de Dinâmica e Convergência Latente ===")
        avg_dists = np.mean(sample_trajectory_dists, axis=0)
        avg_cos = np.mean(sample_trajectory_cos, axis=0)
        
        print("Passo de Transição | Distância Euclidiana Média | Similaridade de Cosseno")
        print("-" * 65)
        for step_idx in range(len(avg_dists)):
            print(f"Passo {step_idx} -> {step_idx+1:<7} | {avg_dists[step_idx]:.5f}                    | {avg_cos[step_idx]:.5f}")
        
        print("\n*Nota: Conforme os passos avançam, a diminuição da distância euclidiana")
        print("e o aumento da similaridade de cosseno próxima a 1.0 indicam")
        print("que o vetor latente convergiu para o atretor estável do Hopfield.*")

if __name__ == "__main__":
    run_evaluation()
