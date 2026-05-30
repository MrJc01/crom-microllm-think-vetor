import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.logic_dataset import LogicCharTokenizer, LogicDataset
from src.logic_llm import MicroReasoningLLM
from train_hybrid import train_hybrid_model

def evaluate_qualitative(model, dataset, tokenizer, device, num_samples=5):
    """
    Demonstra a inferência qualitativa do modelo resolvendo problemas de lógica.
    """
    model.eval()
    print("\n=== Demonstração de Inferência Qualitativa ===")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Obter uma amostra aleatória
            idx = random.randint(0, len(dataset) - 1) if "random" in globals() else i
            item = dataset[idx]
            
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            raw_input = item["raw_input"]
            raw_target = item["raw_target"]
            raw_cot = item["raw_cot"]
            
            # Gerar a resposta autoregressiva
            max_len = 15
            generated_ids = model.generate(input_ids, max_length=max_len, temperature=0.0)
            pred_str = tokenizer.decode(generated_ids[0]).strip()
            
            print(f"Amostra {i+1}:")
            print(f"  Entrada:     {raw_input}")
            print(f"  CoT Latente: {raw_cot}")
            print(f"  Esperado:    {raw_target}")
            print(f"  Predição:    {pred_str}")
            print(f"  Resultado:   {'CORRETO' if pred_str == raw_target else 'INCORRETO'}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Treinamento da Micro-LLM de Raciocínio Lógico Contínuo.")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas total.")
    parser.add_argument("--switch_epoch", type=int, default=5, help="Época de transição para o GRPO.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do lote.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Número de amostras de treino.")
    parser.add_argument("--mutable_context", action="store_true", help="Ativar injeção estocástica de erratas e contradições lógicas.")
    parser.add_argument("--tokenizer_name", type=str, default="char", help="Nome do tokenizer ('char' ou nome/caminho do modelo HuggingFace).")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Treinando Micro-LLM de Raciocínio Lógico.")
    print(f"[INFO] Dispositivo: {device.type.upper()}")
    
    if args.tokenizer_name == "char":
        tokenizer = LogicCharTokenizer()
    else:
        from src.hf_tokenizer_wrapper import HFTokenizerWrapper
        tokenizer = HFTokenizerWrapper(args.tokenizer_name)
    
    # Criar Datasets (80% treino e 20% validação disjuntos)
    train_ds = LogicDataset(num_samples=args.num_samples, seed=42, tokenizer=tokenizer, mutable_context=args.mutable_context)
    val_ds = LogicDataset(num_samples=args.num_samples // 5, seed=43, tokenizer=tokenizer, mutable_context=args.mutable_context)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)
    
    # Instanciar a LLM com o vocabulário lógico
    model = MicroReasoningLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,          # Dimensões adequadas para lógica textual
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_ponder_steps=6,   # Capacidade para 6 passos de reflexão contínua
        num_memories=256,
        beta=8.0,
        use_pos_embedding=False,
        use_rope=True
    )
    
    # Treinar usando o pipeline híbrido
    train_hybrid_model(
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        total_epochs=args.epochs,
        switch_epoch=args.switch_epoch,
        device=device
    )
    
    # Avaliação final qualitativa
    import random
    random.seed(99)
    evaluate_qualitative(model, val_ds, tokenizer, device, num_samples=3)

if __name__ == "__main__":
    main()
