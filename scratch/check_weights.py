import torch

checkpoint_path = "/home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/checkpoints/think_vetor_conversational_exported/weights.pt"
try:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print("Sucesso ao carregar state_dict!")
    for key, val in state_dict.items():
        if "rope" in key:
            print(f"{key}: {val.shape}")
except Exception as e:
    print(f"Erro ao carregar: {e}")
