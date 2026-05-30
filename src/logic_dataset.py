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
    def __init__(self, num_samples=1000, seed=42, tokenizer=None, max_input_len=90, max_target_len=15, mutable_context=False, num_entities=3):
        super().__init__()
        if isinstance(tokenizer, str):
            from src.hf_tokenizer_wrapper import HFTokenizerWrapper
            self.tokenizer = HFTokenizerWrapper(tokenizer)
        else:
            self.tokenizer = tokenizer or LogicCharTokenizer()
        
        self.num_entities = num_entities
        if max_input_len == 90 and num_entities > 3:
            max_input_len = 30 * num_entities + 30
            
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.mutable_context = mutable_context
        
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
            if num_entities == 3:
                # Escolher 3 nomes distintos: X, Y, Z (na ordem lógica provisória X > Y > Z)
                selected_names = random.sample(names, 3)
                x, y, z = selected_names[0], selected_names[1], selected_names[2]
                
                # Relação lógica
                rel = random.choice(relations)
                
                # Sentenças provisórias estabelecendo X > Y e Y > Z
                if random.random() > 0.5:
                    s1 = f"{x} is {rel['positive']} than {y}."
                else:
                    s1 = f"{y} is {rel['negative']} than {x}."
                    
                if random.random() > 0.5:
                    s2 = f"{y} is {rel['positive']} than {z}."
                else:
                    s2 = f"{z} is {rel['negative']} than {y}."
                    
                context = f"{s1} {s2}"
                
                # Verificar se aplicamos errata de correção no contexto
                apply_errata = self.mutable_context and (random.random() < 0.4)
                
                if apply_errata:
                    errata_type = random.choice(["invert_first", "invert_second"])
                    if errata_type == "invert_first":
                        # Inverte X > Y para Y > X. Ordem final: Y > X > Z.
                        errata_phrase = f"Wait, {x} is {rel['negative']} than {y}."
                        context = f"{context} {errata_phrase}"
                        
                        # Pergunta sobre a nova ordem transitiva entre Y e Z
                        if random.random() > 0.5:
                            question = f"Who is {rel['positive']}, {y} or {z}?="
                            answer = y
                        else:
                            question = f"Who is {rel['negative']}, {y} or {z}?="
                            answer = z
                        cot_str = f"{y}>{x} {x}>{z} {y}>{z}"
                    else:
                        # Inverte Y > Z para Z > Y. Ordem final: X > Z > Y.
                        errata_phrase = f"Wait, {y} is {rel['negative']} than {z}."
                        context = f"{context} {errata_phrase}"
                        
                        # Pergunta sobre a nova ordem transitiva entre X e Z
                        if random.random() > 0.5:
                            question = f"Who is {rel['positive']}, {x} or {z}?="
                            answer = x
                        else:
                            question = f"Who is {rel['negative']}, {x} or {z}?="
                            answer = z
                        cot_str = f"{x}>{z} {z}>{y} {x}>{y}"
                else:
                    # Sem errata. Ordem clássica: X > Y > Z
                    if random.random() > 0.5:
                        question = f"Who is {rel['positive']}, {x} or {z}?="
                        answer = x
                    else:
                        question = f"Who is {rel['negative']}, {x} or {z}?="
                        answer = z
                    cot_str = f"{x}>{y} {y}>{z} {x}>{z}"
            else:
                # Caso geral de num_entities > 3 (sem errata por padrão para extrapolação linear OOD)
                selected_names = random.sample(names, num_entities)
                rel = random.choice(relations)
                
                # Gerar sentenças encadeadas: E_0 > E_1 > E_2 > ... > E_{n-1}
                sentences = []
                for i in range(num_entities - 1):
                    e_curr = selected_names[i]
                    e_next = selected_names[i+1]
                    if random.random() > 0.5:
                        sentences.append(f"{e_curr} is {rel['positive']} than {e_next}.")
                    else:
                        sentences.append(f"{e_next} is {rel['negative']} than {e_curr}.")
                
                context = " ".join(sentences)
                
                # Perguntar sobre o primeiro e o último (max_distance para exigir trânsito completo)
                first_ent = selected_names[0]
                last_ent = selected_names[-1]
                
                if random.random() > 0.5:
                    question = f"Who is {rel['positive']}, {first_ent} or {last_ent}?="
                    answer = first_ent
                else:
                    question = f"Who is {rel['negative']}, {first_ent} or {last_ent}?="
                    answer = last_ent
                
                # Construir CoT: E_0>E_1 E_1>E_2 ... E_0>E_n
                cot_parts = []
                for i in range(num_entities - 1):
                    cot_parts.append(f"{selected_names[i]}>{selected_names[i+1]}")
                cot_parts.append(f"{first_ent}>{last_ent}")
                cot_str = " ".join(cot_parts)
                
            input_str = f"{context} {question}"
            target_str = answer
            
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
    ds = LogicDataset(num_samples=20, tokenizer=tok, mutable_context=True)
    print("=== Testando Dataset de Transitividade Mutável (3 Entidades) ===")
    count = 0
    for i in range(len(ds)):
        item = ds[i]
        if "Wait," in item["raw_input"]:
            print(f"Amostra {count+1} (Com Errata/Correção):")
            print("  Raw Input:  ", item["raw_input"])
            print("  Raw Target: ", item["raw_target"])
            print("  Raw CoT:    ", item["raw_cot"])
            print("-" * 50)
            count += 1
            if count >= 2:
                break
                
    print("\n=== Testando Dataset de Transitividade Extrapolada (4 Entidades) ===")
    ds4 = LogicDataset(num_samples=5, tokenizer=tok, num_entities=4)
    for i in range(len(ds4)):
        item = ds4[i]
        print(f"Amostra {i+1}:")
        print("  Raw Input:  ", item["raw_input"])
        print("  Raw Target: ", item["raw_target"])
        print("  Raw CoT:    ", item["raw_cot"])
        print("-" * 50)
