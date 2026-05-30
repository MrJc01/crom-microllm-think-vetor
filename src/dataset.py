import random
import torch
from torch.utils.data import Dataset

class CharTokenizer:
    """
    Tokenizador simples de caracteres para operações de adição.
    Mapeia dígitos, operadores e tokens especiais para IDs numéricos.
    """
    def __init__(self):
        self.chars = "0123456789+= "
        self.char_to_id = {char: idx for idx, char in enumerate(self.chars)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.pad_id = self.char_to_id[" "]
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        decoded_chars = []
        for idx in ids:
            if idx == self.pad_id:
                break
            if idx in self.id_to_char:
                decoded_chars.append(self.id_to_char[idx])
        return "".join(decoded_chars)

class AdditionDataset(Dataset):
    """
    Dataset sintético de adição.
    Gera strings no formato 'A+B=' como entrada e 'C' como resposta,
    onde C = A + B.
    """
    def __init__(self, num_digits=3, num_samples=10000, seed=42, tokenizer=None, samples=None):
        super().__init__()
        self.num_digits = num_digits
        self.tokenizer = tokenizer or CharTokenizer()
        
        # Comprimento máximo da entrada e do alvo para fins de padding
        self.max_input_len = num_digits * 2 + 2 # ex: "99+99=" -> 6 chars
        self.max_target_len = num_digits + 1    # ex: "198" -> 3 chars
        
        if samples is not None:
            self.samples = samples
            self.num_samples = len(samples)
            return
            
        random.seed(seed)
        self.samples = []
        
        # Determinar limite superior dos números baseado em num_digits
        max_val = 10**num_digits - 1
        
        # Evitar loop infinito se num_samples for maior do que o espaço total de combinações únicas
        max_possible = (10**num_digits) ** 2
        if num_samples > max_possible:
            num_samples = max_possible
        self.num_samples = num_samples
        
        # Gerar amostras únicas
        seen = set()
        while len(self.samples) < num_samples:
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            if (a, b) not in seen:
                seen.add((a, b))
                
                # Ex: "123+456=" -> "579"
                input_str = f"{a}+{b}="
                target_str = f"{a+b}"
                
                self.samples.append((input_str, target_str))

        # Inicialização do comprimento máximo do padding concluída

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, target_str = self.samples[idx]
        
        # Codificar caracteres
        input_ids = self.tokenizer.encode(input_str)
        target_ids = self.tokenizer.encode(target_str)
        
        # Padding nas entradas (à esquerda para que o '=' fique alinhado antes do pensamento latente)
        input_pad_len = self.max_input_len - len(input_ids)
        input_ids = [self.tokenizer.pad_id] * input_pad_len + input_ids
        
        # Padding nos alvos (à direita)
        target_pad_len = self.max_target_len - len(target_ids)
        # Para treinamento autoregressivo, adicionamos padding ao target
        padded_target_ids = target_ids + [self.tokenizer.pad_id] * target_pad_len
        
        # Criar máscara para o loss incluir o primeiro padding (token EOS/stop)
        # O modelo precisa aprender a produzir o token de parada após os dígitos da resposta
        target_mask = [1] * min(len(target_ids) + 1, self.max_target_len)
        target_mask = target_mask + [0] * (self.max_target_len - len(target_mask))
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(padded_target_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.float),
            "raw_input": input_str,
            "raw_target": target_str
        }

if __name__ == "__main__":
    # Teste rápido
    tok = CharTokenizer()
    ds = AdditionDataset(num_digits=3, num_samples=5, tokenizer=tok)
    for i in range(len(ds)):
        item = ds[i]
        print(f"Sample {i}:")
        print("  Raw Input:  ", item["raw_input"])
        print("  Raw Target: ", item["raw_target"])
        print("  Encoded In: ", item["input_ids"])
        print("  Encoded Out:", item["target_ids"])
        print("  Mask:       ", item["target_mask"])
        print("  Decoded In: ", tok.decode(item["input_ids"]))
        print("  Decoded Out:", tok.decode(item["target_ids"]))
        print("-" * 30)
