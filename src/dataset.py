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
    Suporta Curriculum Learning / comprimentos mistos se num_digits for uma lista/lista de inteiros.
    """
    def __init__(self, num_digits=3, num_samples=10000, seed=42, tokenizer=None, samples=None, pad_left=True):
        super().__init__()
        self.tokenizer = tokenizer or CharTokenizer()
        self.pad_left = pad_left
        
        if isinstance(num_digits, (list, tuple)):
            self.num_digits = list(num_digits)
            max_d = max(self.num_digits)
        else:
            self.num_digits = [num_digits]
            max_d = num_digits
            
        # Comprimento máximo da entrada e do alvo para fins de padding
        self.max_input_len = max_d * 2 + 2 # ex: "99+99=" -> 6 chars
        self.max_target_len = max_d + 1    # ex: "198" -> 3 chars
        
        if samples is not None:
            self.samples = samples
            self.num_samples = len(samples)
            return
            
        random.seed(seed)
        self.samples = []
        
        # Evitar loop infinito se num_samples for maior do que o espaço total de combinações únicas
        max_possible = (10**max_d) ** 2
        if num_samples > max_possible:
            num_samples = max_possible
        self.num_samples = num_samples
        
        # Gerar amostras únicas
        seen = set()
        while len(self.samples) < num_samples:
            # Escolher aleatoriamente número de dígitos para cada operando
            d_a = random.choice(self.num_digits)
            d_b = random.choice(self.num_digits)
            
            a = random.randint(0, 10**d_a - 1)
            b = random.randint(0, 10**d_b - 1)
            
            if (a, b) not in seen:
                seen.add((a, b))
                
                # Ex: "123+456=" -> "579"
                input_str = f"{a}+{b}="
                target_str = f"{a+b}"
                
                self.samples.append((input_str, target_str))

        # Inicialização do comprimento máximo do padding concluída

    def get_cot_trace(self, a, b):
        """
        Gera a sequência passo a passo de soma e carry para alinhar com os passos
        do loop recorrente de reflexão (Fronteira 3).
        E.g. para 73+67, gera "014110" (soma_0, carry_0, soma_1, carry_1, soma_2, carry_2)
        """
        str_a = str(a)[::-1]
        str_b = str(b)[::-1]
        max_d = max(len(str_a), len(str_b))
        
        trace = []
        carry = 0
        for i in range(max_d):
            digit_a = int(str_a[i]) if i < len(str_a) else 0
            digit_b = int(str_b[i]) if i < len(str_b) else 0
            
            step_sum = digit_a + digit_b + carry
            out_digit = step_sum % 10
            carry = step_sum // 10
            
            trace.append(str(out_digit))
            trace.append(str(carry))
            
        # Adicionar o carry final
        trace.append(str(carry))
        trace.append("0") # final carry sem carry seguinte
        
        # Une a cadeia de rastreamento
        trace_str = "".join(trace)
        return trace_str

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, target_str = self.samples[idx]
        
        # Codificar caracteres
        input_ids = self.tokenizer.encode(input_str)
        target_ids = self.tokenizer.encode(target_str)
        
        # Extrair valores numéricos para gerar a cadeia de raciocínio passo a passo
        try:
            raw_in = input_str.replace(" ", "").replace("=", "")
            parts = raw_in.split("+")
            a_val = int(parts[0])
            b_val = int(parts[1])
            cot_str = self.get_cot_trace(a_val, b_val)
            cot_ids = self.tokenizer.encode(cot_str)
        except Exception:
            cot_ids = []
            
        # Garantir tamanho fixo igual a self.max_input_len para colação correta no DataLoader
        cot_pad_len = self.max_input_len - len(cot_ids)
        if cot_pad_len > 0:
            cot_ids = cot_ids + [self.tokenizer.pad_id] * cot_pad_len
        else:
            cot_ids = cot_ids[:self.max_input_len]
            
        # Padding nas entradas (à esquerda para que o '=' fique alinhado antes do pensamento latente ou à direita)
        input_pad_len = self.max_input_len - len(input_ids)
        if self.pad_left:
            input_ids = [self.tokenizer.pad_id] * input_pad_len + input_ids
        else:
            input_ids = input_ids + [self.tokenizer.pad_id] * input_pad_len
        
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
            "cot_ids": torch.tensor(cot_ids, dtype=torch.long),
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
        print("  CoT IDs:    ", item["cot_ids"])
        print("  Decoded CoT:", tok.decode(item["cot_ids"]))
        print("  Mask:       ", item["target_mask"])
        print("  Decoded In: ", tok.decode(item["input_ids"]))
        print("  Decoded Out:", tok.decode(item["target_ids"]))
        print("-" * 30)
