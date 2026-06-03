import os
import json
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Utilitário de Exportação de Checkpoints do Think-Vetor")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/think_vetor_hybrid_best.pt",
                        help="Caminho para o checkpoint .pt original.")
    parser.add_argument("--out_dir", type=str, default="checkpoints/think_vetor_logic_exported",
                        help="Diretório de saída para salvar o pacote exportado.")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERRO] Checkpoint de origem não encontrado: {args.checkpoint}")
        return
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Carrega e detecta os parâmetros
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    vocab_size, d_model = state_dict['token_embeddings.weight'].shape
    
    max_ponder_steps = 0
    if 'step_embeddings' in state_dict:
        max_ponder_steps = state_dict['step_embeddings'].shape[0]
        
    use_rope = False
    num_encoder_layers = 0
    while True:
        if f"encoder.layers.{num_encoder_layers}.linear1.weight" in state_dict or f"encoder.layers.{num_encoder_layers}.self_attn.in_proj_weight" in state_dict:
            if f"encoder.layers.{num_encoder_layers}.self_attn.q_proj.weight" in state_dict:
                use_rope = True
            num_encoder_layers += 1
        else:
            break
            
    num_decoder_layers = 0
    while True:
        if f"decoder.layers.{num_decoder_layers}.linear1.weight" in state_dict or f"decoder.layers.{num_decoder_layers}.self_attn.in_proj_weight" in state_dict:
            num_decoder_layers += 1
        else:
            break
            
    is_causal_coconut = num_encoder_layers == 0 and num_decoder_layers > 0
    
    if 'hopfield_ebm.memories' in state_dict:
        num_memories = state_dict['hopfield_ebm.memories'].shape[0]
    else:
        num_memories = 512 if d_model == 128 else 128
        
    tokenizer_type = "logic_char"
    if vocab_size > 50000:
        tokenizer_type = "gpt2"
    elif vocab_size == 13:
        tokenizer_type = "char"
        
    nhead = 8 if d_model == 128 else 4
    
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers if num_encoder_layers > 0 else 2,
        "num_decoder_layers": num_decoder_layers if num_decoder_layers > 0 else 2,
        "max_ponder_steps": max_ponder_steps,
        "num_memories": num_memories,
        "beta": 8.0,
        "use_pos_embedding": not use_rope,
        "use_rope": use_rope,
        "is_causal_coconut": is_causal_coconut,
        "tokenizer_type": tokenizer_type
    }
    
    # Salvar config.json
    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        
    # Salvar weights.pt
    weights_path = os.path.join(args.out_dir, "weights.pt")
    torch.save(state_dict, weights_path)
    
    print(f"\n[SUCESSO] Checkpoint exportado com sucesso para a pasta: {args.out_dir}")
    print(f"  - Configuração salva em: {config_path}")
    print(f"  - Pesos salvos em: {weights_path}")
    print("\nMetadados Exportados:")
    for k, v in config.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
