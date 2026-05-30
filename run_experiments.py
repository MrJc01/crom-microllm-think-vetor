import os
import argparse
import random
import torch
from torch.utils.data import DataLoader

from src.dataset import CharTokenizer, AdditionDataset
from src.model import ThinkVetorModel
from src.coconut_model import CausalThinkVetorModel
from train import train_model
from train_distill import train_distill_model
from src.grpo_agent import train_grpo
from train_contrastive_ebm import train_contrastive_ebm_model

def get_dataloaders(num_digits, batch_size, tokenizer, align_operator=False, max_d_align=None):
    # Gerar todas as combinações únicas de num_digits
    max_val = 10**num_digits - 1
    all_pairs = []
    for a in range(max_val + 1):
        for b in range(max_val + 1):
            all_pairs.append((a, b))
    random.seed(42)
    random.shuffle(all_pairs)
    
    # Divisão de 80% treino e 20% validação disjuntos
    split = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split]
    val_pairs = all_pairs[split:]
    
    def pairs_to_samples(pairs):
        return [f"{a}+{b}=" for a, b in pairs], [f"{a+b}" for a, b in pairs]
        
    train_inputs, train_targets = pairs_to_samples(train_pairs)
    val_inputs, val_targets = pairs_to_samples(val_pairs)
    
    train_samples = list(zip(train_inputs, train_targets))
    val_samples = list(zip(val_inputs, val_targets))
    
    train_ds = AdditionDataset(num_digits=num_digits, tokenizer=tokenizer, samples=train_samples, align_operator=align_operator, max_d_align=max_d_align)
    val_ds = AdditionDataset(num_digits=num_digits, tokenizer=tokenizer, samples=val_samples, align_operator=align_operator, max_d_align=max_d_align)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="Script mestre para rodar experimentos Think-Vetor no Google Colab ou Localmente.")
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["baseline", "think_vetor", "coconut", "distill", "grpo", "ebm"],
                        help="Topologia ou algoritmo de treino a ser executado.")
    parser.add_argument("--epochs", type=int, default=80, help="Número de épocas de treinamento.")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamanho do lote (batch size).")
    parser.add_argument("--num_digits", type=int, default=2, help="Número de dígitos para a tarefa de soma.")
    parser.add_argument("--d_model", type=int, default=128, help="Dimensão oculta do modelo.")
    parser.add_argument("--nhead", type=int, default=8, help="Número de cabeças de atenção.")
    parser.add_argument("--num_layers", type=int, default=2, help="Número de camadas de codificador/decodificador.")
    parser.add_argument("--use_rope", type=bool, default=True, help="Usar Rotary Position Embeddings.")
    parser.add_argument("--align_operator", type=bool, default=False, help="Usar padding centralizado no operador.")
    parser.add_argument("--max_d_align", type=int, default=4, help="Comprimento fixo de dígitos de alinhamento constante.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Executando experimento: {args.experiment.upper()}")
    print(f"[INFO] Dispositivo: {device.type.upper()}")
    
    tokenizer = CharTokenizer()
    train_loader, val_loader = get_dataloaders(args.num_digits, args.batch_size, tokenizer, 
                                               align_operator=args.align_operator, max_d_align=args.max_d_align)
    
    # Executar o experimento apropriado
    if args.experiment == "baseline":
        model = ThinkVetorModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            max_ponder_steps=0, # Sem ponderação
            use_pos_embedding=not args.use_rope,
            use_rope=args.use_rope
        )
        train_model("Baseline Model", model, train_loader, val_loader, tokenizer, 
                    epochs=args.epochs, device=device, use_ponder=False)
        
    elif args.experiment == "think_vetor":
        model = ThinkVetorModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            max_ponder_steps=6,
            num_memories=512,
            beta=8.0,
            use_pos_embedding=not args.use_rope,
            use_rope=args.use_rope
        )
        train_model("Think Vetor", model, train_loader, val_loader, tokenizer, 
                    epochs=args.epochs, device=device, use_ponder=True)
        
    elif args.experiment == "coconut":
        model = CausalThinkVetorModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers * 2, # Unificado, então dobramos camadas
            max_thought_steps=4,
            use_rope=args.use_rope
        )
        # O modelo causal requer um loop modificado (reutiliza train_model desativando ponderação clássica)
        print("[INFO] Iniciando treinamento Causal COCONUT...")
        train_model("Causal COCONUT", model, train_loader, val_loader, tokenizer, 
                    epochs=args.epochs, device=device, use_ponder=False)
        
    elif args.experiment == "distill":
        model = ThinkVetorModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            max_ponder_steps=6, # Deve bater com o CoT ("014110")
            num_memories=512,
            beta=8.0,
            use_pos_embedding=not args.use_rope,
            use_rope=args.use_rope
        )
        train_distill_model(model, train_loader, val_loader, tokenizer, 
                            epochs=args.epochs, device=device)
        
    elif args.experiment == "grpo":
        model = ThinkVetorModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            max_ponder_steps=6,
            num_memories=512,
            beta=8.0,
            use_pos_embedding=not args.use_rope,
            use_rope=args.use_rope
        )
        train_grpo(model, train_loader, val_loader, tokenizer, 
                   epochs=args.epochs, device=device)
        
    elif args.experiment == "ebm":
        model = ThinkVetorModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            max_ponder_steps=6,
            num_memories=512,
            beta=8.0,
            use_pos_embedding=not args.use_rope,
            use_rope=args.use_rope
        )
        train_contrastive_ebm_model(model, train_loader, val_loader, tokenizer, 
                                    epochs=args.epochs, device=device)

if __name__ == "__main__":
    main()
