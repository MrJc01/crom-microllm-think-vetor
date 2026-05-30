import random
import torch
from torch.utils.data import Dataset

class LogicCharTokenizer:
    """
    Tokenizador de caracteres completo para a Micro-LLM de raciocínio.
    Mapeia letras, números, pontuações e operadores lógicos para IDs.
    """
    def __init__(self):
        # Vocabulário abrangendo letras, pontuação de texto, operadores lógicos e padding
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+=>? <,-.!"
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
                # Na decodificação autorregressiva, o espaço age como stop token
                break
            if idx in self.id_to_char:
                decoded_chars.append(self.id_to_char[idx])
        return "".join(decoded_chars)

class LogicDataset(Dataset):
    """
    Dataset sintético de raciocínio lógico de transitividade.
    Ex: "Alice is taller than Bob. Bob is taller than Charlie. Who is taller, Alice or Charlie?=" -> "Alice"
    Cadeia de raciocínio contínua (CoT): "Alice>Bob Bob>Charlie Alice>Charlie"
    """
    def __init__(self, num_samples=1000, seed=42, tokenizer=None, max_input_len=90, max_target_len=15):
        super().__init__()
        self.tokenizer = tokenizer or LogicCharTokenizer()
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        
        random.seed(seed)
        self.samples = []
        
        names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
        
        # Tipos de relações de ordem
        relations = [
            {"positive": "older", "negative": "younger", "op": ">"},
            {"positive": "taller", "negative": "shorter", "op": ">"},
            {"positive": "richer", "negative": "poorer", "op": ">"}
        ]
        
        for _ in range(num_samples):
            # Escolher 3 nomes distintos: X, Y, Z (de forma que X > Y > Z na ordem lógica)
            selected_names = random.sample(names, 3)
            x, y, z = selected_names[0], selected_names[1], selected_names[2]
            
            # Relação lógica
            rel = random.choice(relations)
            
            # Gerar sentenças para X > Y e Y > Z com variação de voz
            # X > Y
            if random.random() > 0.5:
                s1 = f"{x} is {rel['positive']} than {y}."
            else:
                s1 = f"{y} is {rel['negative']} than {x}."
                
            # Y > Z
            if random.random() > 0.5:
                s2 = f"{y} is {rel['positive']} than {z}."
            else:
                s2 = f"{z} is {rel['negative']} than {y}."
                
            context = f"{s1} {s2}"
            
            # Decidir a pergunta sobre o relacionamento entre X e Z
            if random.random() > 0.5:
                # Pergunta sobre o maior (X)
                question = f"Who is {rel['positive']}, {x} or {z}?="
                answer = x
            else:
                # Pergunta sobre o menor (Z)
                question = f"Who is {rel['negative']}, {x} or {z}?="
                answer = z
                
            input_str = f"{context} {question}"
            target_str = answer
            
            # Cadeia de dedução intermediária (CoT)
            cot_str = f"{x}>{y} {y}>{z} {x}>{z}"
            
            self.samples.append((input_str, target_str, cot_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, target_str, cot_str = self.samples[idx]
        
        input_ids = self.tokenizer.encode(input_str)
        target_ids = self.tokenizer.encode(target_str)
        cot_ids = self.tokenizer.encode(cot_str)
        
        # Preencher input_ids à esquerda
        input_pad_len = self.max_input_len - len(input_ids)
        if input_pad_len > 0:
            input_ids = [self.tokenizer.pad_id] * input_pad_len + input_ids
        else:
            input_ids = input_ids[:self.max_input_len]
            
        # Preencher cot_ids à direita
        cot_pad_len = self.max_input_len - len(cot_ids)
        if cot_pad_len > 0:
            cot_ids = cot_ids + [self.tokenizer.pad_id] * cot_pad_len
        else:
            cot_ids = cot_ids[:self.max_input_len]
            
        # Preencher target_ids à direita
        target_pad_len = self.max_target_len - len(target_ids)
        padded_target_ids = target_ids + [self.tokenizer.pad_id] * target_pad_len
        
        # Máscara de perda incluindo o stop token
        target_mask = [1] * min(len(target_ids) + 1, self.max_target_len)
        target_mask = target_mask + [0] * (self.max_target_len - len(target_mask))
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(padded_target_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.float),
            "cot_ids": torch.tensor(cot_ids, dtype=torch.long),
            "raw_input": input_str,
            "raw_target": target_str,
            "raw_cot": cot_str
        }

if __name__ == "__main__":
    tok = LogicCharTokenizer()
    ds = LogicDataset(num_samples=3, tokenizer=tok)
    for i in range(len(ds)):
        item = ds[i]
        print(f"Sample {i}:")
        print("  Raw Input:  ", item["raw_input"])
        print("  Raw Target: ", item["raw_target"])
        print("  Raw CoT:    ", item["raw_cot"])
        print("  Encoded In: ", item["input_ids"])
        print("  Decoded In: '", tok.decode(item["input_ids"]), "'")
        print("  Decoded Out:'", tok.decode(item["target_ids"]), "'")
        print("  Decoded CoT:'", tok.decode(item["cot_ids"]), "'")
        print("  Mask:       ", item["target_mask"])
        print("-" * 30)
