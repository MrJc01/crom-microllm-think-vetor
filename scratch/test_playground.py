import os
import torch
from interactive_playground import detect_architecture_and_tokenizer, run_inference

def test_local_model():
    checkpoint_path = "checkpoints/think_vetor_hybrid_best.pt"
    device = torch.device("cpu")  # CPU para teste local rápido
    
    print("=== Rodando Teste do Playground Local ===")
    model, tokenizer, is_causal_coconut = detect_architecture_and_tokenizer(checkpoint_path)
    
    prompts = [
        "Ivy is richer than Grace. Grace is richer than Frank. Who is poorer, Ivy or Frank?=",
        "Eve is shorter than Alice. Eve is taller than Charlie. Wait, Eve is shorter than Charlie. Who is taller, Alice or Charlie?="
    ]
    
    for prompt in prompts:
        run_inference(model, tokenizer, prompt, is_causal_coconut, device)

if __name__ == "__main__":
    test_local_model()
