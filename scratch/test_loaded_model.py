import sys
import os
# Adiciona a raiz do projeto ao sys.path para garantir imports corretos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from interactive_playground import detect_architecture_and_tokenizer, run_inference

checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "think_vetor_conversational_exported", "weights.pt")

print("=== INICIANDO TESTE DO MODELO CONVERSACIONAL CARREGADO LOCALMENTE ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado: {device}")
print(f"Buscando checkpoint em: {checkpoint_path}")

try:
    model, tokenizer, is_causal_coconut = detect_architecture_and_tokenizer(checkpoint_path)
    print("\n[SUCESSO] Modelo e Tokenizer inicializados corretamente!")
    
    test_prompts = [
        "oi",
        "quem é você?",
        "o que você sabe fazer?",
        "Alice is younger than Bob. Charlie is younger than Alice. Who is younger, Bob or Charlie?"
    ]
    
    for prompt in test_prompts:
        run_inference(model, tokenizer, prompt, is_causal_coconut, device)
        
except Exception as e:
    print(f"\n[ERRO] Ocorreu uma falha no teste: {e}")
    import traceback
    traceback.print_exc()
