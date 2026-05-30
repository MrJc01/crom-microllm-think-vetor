import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import CharTokenizer
from src.logic_dataset import LogicCharTokenizer
from src.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.model import ThinkVetorModel
from src.coconut_model import CausalThinkVetorModel

def detect_architecture_and_tokenizer(checkpoint_path):
    print(f"\n[INFO] Carregando e analisando checkpoint: {checkpoint_path}")
    
    # Carregar state_dict de forma segura
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # 1. Vocab size e d_model
    vocab_size, d_model = state_dict['token_embeddings.weight'].shape
    print(f"  Vocab Size detectado: {vocab_size}")
    print(f"  d_model detectado: {d_model}")
    
    # 2. Tokenizer apropriado
    if vocab_size > 50000:
        print("  Tokenizer detectado: HuggingFace GPT-2 (Subwords BPE)")
        tokenizer = HFTokenizerWrapper("gpt2")
    elif vocab_size == 13:
        print("  Tokenizer detectado: CharTokenizer (Aritmética)")
        tokenizer = CharTokenizer()
    else:
        print("  Tokenizer detectado: LogicCharTokenizer (Lógica Clássica)")
        tokenizer = LogicCharTokenizer()
        
    # 3. Ponder Steps e Hopfield EBM
    max_ponder_steps = 0
    if 'step_embeddings' in state_dict:
        max_ponder_steps = state_dict['step_embeddings'].shape[0]
    print(f"  max_ponder_steps detectado: {max_ponder_steps}")
    
    # 4. Detectar número de camadas e RoPE
    use_rope = False
    num_encoder_layers = 0
    while True:
        # Verifica se existe camada no encoder
        if f"encoder.layers.{num_encoder_layers}.linear1.weight" in state_dict or f"encoder.layers.{num_encoder_layers}.self_attn.in_proj_weight" in state_dict:
            # Verifica se usa RoPE (CustomTransformerEncoderLayer usa self_attn.q_proj)
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
            
    # Se for causal coconut (decoder-only unificado, não tem encoder no state_dict)
    is_causal_coconut = "causal" in checkpoint_path or "coconut" in checkpoint_path or (num_encoder_layers == 0 and num_decoder_layers > 0)
    
    if is_causal_coconut:
        print("  Topologia detectada: Causal Decoder-Only (COCONUT)")
        # Causal coconut unificado
        num_layers = 0
        while f"decoder.layers.{num_layers}.linear1.weight" in state_dict:
            num_layers += 1
        print(f"  Camadas Decodificadoras: {num_layers}")
        
        nhead = 8 if d_model == 128 else 4
        model = CausalThinkVetorModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_thought_steps=4,
            use_rope=use_rope
        )
    else:
        print(f"  Topologia detectada: Think-Vetor (Encoder-Decoder)")
        print(f"  Camadas Encoder: {num_encoder_layers} | Camadas Decoder: {num_decoder_layers}")
        print(f"  Uso de RoPE: {use_rope}")
        
        nhead = 8 if d_model == 128 else 4
        
        # Obter o número real de memórias Hopfield a partir do state_dict para evitar incompatibilidade
        if 'hopfield_ebm.memories' in state_dict:
            num_memories = state_dict['hopfield_ebm.memories'].shape[0]
            print(f"  Número de memórias Hopfield detectado: {num_memories}")
        else:
            num_memories = 512 if d_model == 128 else 128
            
        model = ThinkVetorModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers if num_encoder_layers > 0 else 2,
            num_decoder_layers=num_decoder_layers if num_decoder_layers > 0 else 2,
            max_ponder_steps=max_ponder_steps,
            num_memories=num_memories,
            beta=8.0,
            use_pos_embedding=not use_rope,
            use_rope=use_rope
        )
        
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Checkpoint carregado com sucesso!\n")
    return model, tokenizer, is_causal_coconut

def run_inference(model, tokenizer, prompt, is_causal_coconut, device):
    model = model.to(device)
    
    # Processar entrada
    # Para bater com o padding do treino, dependendo do tokenizer, o input deve terminar em '='
    if not prompt.endswith("="):
        prompt += "="
        
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    print("\n" + "="*50)
    print(f"PROMPT ENVIADO: '{prompt}'")
    print("="*50)
    
    with torch.no_grad():
        if is_causal_coconut:
            # Inferência do Causal COCONUT
            # Gera pensamentos latentes intermediários e decodifica a resposta
            generated = model.generate(input_tensor, max_length=15, temperature=0.0)
            pred_str = tokenizer.decode(generated[0]).strip()
            print(f"SAÍDA DO MODELO: {pred_str}")
        else:
            # Inferência do Think-Vetor
            # 1. Codificação inicial e loop de ponderação PonderNet
            x_emb = model.token_embeddings(input_tensor)
            if model.use_pos_embedding:
                x_emb = model.pos_encoder(x_emb)
            
            if model.use_rope:
                x_encoded = model.encoder(x_emb, rope=model.rope)
            else:
                x_encoded = model.encoder(x_emb)
                
            batch_size, seq_len, d_model = x_encoded.shape
            
            # Executar o loop recorrente explicitamente para rastrear as probabilidades de halting
            accumulated_remainders = torch.ones(batch_size, seq_len, 1, device=device)
            pooled_latent_states = torch.zeros_like(x_encoded)
            current_state = x_encoded
            halting_probs = []
            
            init_temp = 0.5
            for k in range(model.max_ponder_steps):
                step_emb = model.step_embeddings[k].view(1, 1, d_model)
                state_temp = current_state + step_emb
                
                next_state = model.recurrent_layer(state_temp)
                current_temp = init_temp * (0.6 ** k)
                next_state = model.hopfield_ebm(next_state, temp=current_temp, lr=0.1)
                
                halt_prob = torch.sigmoid(model.halt_classifier(next_state))
                
                if k == model.max_ponder_steps - 1:
                    step_halt_prob = accumulated_remainders
                else:
                    step_halt_prob = halt_prob * accumulated_remainders
                    accumulated_remainders = accumulated_remainders * (1.0 - halt_prob)
                    
                pooled_latent_states = pooled_latent_states + step_halt_prob * next_state
                halting_probs.append(step_halt_prob.mean().item())
                current_state = next_state
            
            # Exibir análise do pensamento latente
            print("ANÁLISE DE PENSAMENTO LATENTE (PonderNet):")
            for step, p in enumerate(halting_probs):
                bar = "█" * int(p * 20)
                print(f"  Passo {step+1:02d}: {p*100:6.2f}% de Parada | {bar}")
            
            # Calcular o passo médio ponderado
            avg_steps = sum((step + 1) * p for step, p in enumerate(halting_probs))
            print(f"  Passo médio de reflexão: {avg_steps:.2f}")
            print("-" * 50)
            
            # 2. Decodificação autoregressiva a partir do estado de reflexão acumulado
            # Começa com '=' (ou o primeiro token do decoder)
            tgt_ids = [tokenizer.char_to_id["="]] if hasattr(tokenizer, "char_to_id") else [tokenizer.encode("=")[0]]
            
            for _ in range(15):
                tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
                tgt_emb = model.token_embeddings(tgt_tensor)
                
                if model.use_pos_embedding:
                    tgt_emb = model.pos_decoder(tgt_emb)
                    
                # O decodificador atende aos estados latentes ponderados
                x_decoded = model.decoder(tgt_emb, pooled_latent_states)
                logits = model.lm_head(x_decoded[:, -1, :])
                
                next_token = torch.argmax(logits, dim=-1).item()
                
                # Se o token predito for o pad_id (que age como EOS / stop token)
                if next_token == tokenizer.pad_id:
                    break
                    
                tgt_ids.append(next_token)
                
            # Decodificar e mostrar resposta
            # Remove o '=' inicial se presente na decodificação
            pred_str = tokenizer.decode(tgt_ids[1:]).strip()
            print(f"SAÍDA DO MODELO: {pred_str}")
            
    print("="*50 + "\n")

def main():
    print("="*60)
    print("       PLAYGROUND INTERATIVO DO MODELO THINK-VETOR      ")
    print("="*60)
    
    # 1. Localizar checkpoints
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"[ERRO] Pasta '{checkpoint_dir}' não encontrada. Certifique-se de executar na raiz do repositório.")
        sys.exit(1)
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        print(f"[ERRO] Nenhum arquivo de peso '.pt' encontrado na pasta '{checkpoint_dir}'.")
        sys.exit(1)
        
    print("Checkpoints disponíveis:")
    for idx, cp in enumerate(sorted(checkpoints)):
        print(f"  [{idx + 1}] {cp}")
        
    while True:
        try:
            choice = int(input("\nEscolha o número do checkpoint para carregar: "))
            if 1 <= choice <= len(checkpoints):
                selected_cp = sorted(checkpoints)[choice - 1]
                break
        except ValueError:
            pass
        print("[ERRO] Escolha inválida. Tente novamente.")
        
    checkpoint_path = os.path.join(checkpoint_dir, selected_cp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model, tokenizer, is_causal_coconut = detect_architecture_and_tokenizer(checkpoint_path)
    except Exception as e:
        print(f"[ERRO] Falha ao analisar e carregar o modelo: {e}")
        sys.exit(1)
        
    print("Modelo carregado com sucesso no dispositivo:", device.type.upper())
    print("\nDigite suas perguntas no terminal. Pressione Ctrl+C para sair.")
    print("Exemplos lógicos: 'Alice is taller than Bob. Bob is taller than Charlie. Who is taller, Alice or Charlie?='")
    print("Exemplos aritméticos: '45+18='")
    
    try:
        while True:
            prompt = input("\nPrompt > ").strip()
            if not prompt:
                continue
            run_inference(model, tokenizer, prompt, is_causal_coconut, device)
    except KeyboardInterrupt:
        print("\nSaindo do Playground. Até mais!")

if __name__ == "__main__":
    main()
