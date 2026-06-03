import os
import json
import torch
import argparse
from torch.utils.data import DataLoader

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel

def evaluate_digits(model, num_digits, tokenizer, device, align_operator, max_d_align, num_samples=100):
    ds = AdditionDataset(
        num_digits=num_digits, 
        num_samples=num_samples, 
        seed=42 + num_digits, 
        tokenizer=tokenizer,
        align_operator=align_operator,
        max_d_align=max_d_align
    )
    
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            raw_targets = batch["raw_target"]
            
            # Na decodificação de soma, o comprimento máximo do target é num_digits + 1
            max_len = num_digits + 1
            
            generated_ids = model.generate(input_ids, max_length=max_len, temperature=0.0)
            
            for idx, gen_seq in enumerate(generated_ids):
                pred_str = tokenizer.decode(gen_seq).strip()
                target_str = raw_targets[idx].strip()
                if pred_str == target_str:
                    correct += 1
                total += 1
                
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Script de avaliação de Extrapolação OOD em Aritmética.")
    parser.add_argument("--use_rope", action="store_true", default=True, help="Usar Rotary Position Embeddings (RoPE).")
    parser.add_argument("--baseline_path", type=str, default="checkpoints/baseline_model_best.pt")
    parser.add_argument("--think_vetor_path", type=str, default="checkpoints/think_vetor_best.pt")
    parser.add_argument("--align_operator", action="store_true", default=True, help="Habilitar alinhamento centralizado do operador.")
    parser.add_argument("--max_d_align", type=int, default=4, help="Número máximo de dígitos para alinhamento na fita.")
    parser.add_argument("--d_model", type=int, default=128, help="Dimensão latente (d_model).")
    parser.add_argument("--nhead", type=int, default=8, help="Número de cabeças de atenção.")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Camadas do codificador.")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Camadas do decodificador.")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== AVALIAÇÃO DE EXTRAPOLAÇÃO OOD (ARITMÉTICA) ===")
    print(f"Dispositivo: {device}")
    print(f"Alinhamento do Operador: {args.align_operator} (Max dígitos alinhados: {args.max_d_align})")
    
    tokenizer = CharTokenizer()
    
    # 1. Instanciar Modelos
    baseline = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        max_ponder_steps=0,
        use_pos_embedding=not args.use_rope,
        use_rope=args.use_rope
    ).to(device)
    
    think_vetor = ThinkVetorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        max_ponder_steps=6,
        num_memories=256,
        beta=8.0,
        use_pos_embedding=not args.use_rope,
        use_rope=args.use_rope
    ).to(device)
    
    # Carregar pesos
    if os.path.exists(args.baseline_path):
        baseline.load_state_dict(torch.load(args.baseline_path, map_location=device))
        print(f"-> Pesos da Baseline carregados de {args.baseline_path}")
    else:
        print(f"-> AVISO: Baseline não encontrada em {args.baseline_path}. Usando pesos aleatórios.")
        
    if os.path.exists(args.think_vetor_path):
        think_vetor.load_state_dict(torch.load(args.think_vetor_path, map_location=device))
        print(f"-> Pesos do Think-Vetor carregados de {args.think_vetor_path}")
    else:
        print(f"-> AVISO: Think-Vetor não encontrado em {args.think_vetor_path}. Usando pesos aleatórios.")
        
    # 2. Avaliar em distribuição e OOD extrema (2 a 6 dígitos)
    digit_ranges = [2, 3, 4, 5, 6]
    results = {}
    
    for d in digit_ranges:
        print(f"\nAvaliando com {d} dígitos...")
        acc_base = evaluate_digits(baseline, d, tokenizer, device, args.align_operator, args.max_d_align)
        acc_tv = evaluate_digits(think_vetor, d, tokenizer, device, args.align_operator, args.max_d_align)
        
        results[str(d)] = {
            "baseline": acc_base,
            "think_vetor": acc_tv
        }
        
        print(f"  {d} dígitos - Baseline:    {acc_base:.2f}%")
        print(f"  {d} dígitos - Think-Vetor: {acc_tv:.2f}%")
        
    # 3. Exportar resultados para JSON
    os.makedirs("checkpoints", exist_ok=True)
    out_path = "checkpoints/ood_extrapolation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[SUCESSO] Resultados da extrapolação OOD salvos em: {out_path}")

if __name__ == "__main__":
    main()
