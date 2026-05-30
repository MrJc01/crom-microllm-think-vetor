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
    def __init__(self, num_samples=1000, seed=42, tokenizer=None, max_input_len=90, max_target_len=15, mutable_context=False, num_entities=3, multidimensional=False):
        super().__init__()
        if isinstance(tokenizer, str):
            from src.hf_tokenizer_wrapper import HFTokenizerWrapper
            self.tokenizer = HFTokenizerWrapper(tokenizer)
        else:
            self.tokenizer = tokenizer or LogicCharTokenizer()
        
        self.num_entities = num_entities
        self.multidimensional = multidimensional
        if max_input_len == 90 and (num_entities > 3 or multidimensional):
            max_input_len = 150
            
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
                selected_names = random.sample(names, 3)
                
                if self.multidimensional:
                    # Relações multidimensionais (2 atributos lógicos distintos)
                    rel1, rel2 = random.sample(relations, 2)
                    
                    # Ordem do Atributo 1
                    order1 = random.sample(selected_names, 3)
                    x1, y1, z1 = order1[0], order1[1], order1[2]
                    
                    if random.random() > 0.5:
                        s1_1 = f"{x1} is {rel1['positive']} than {y1}."
                    else:
                        s1_1 = f"{y1} is {rel1['negative']} than {x1}."
                        
                    if random.random() > 0.5:
                        s1_2 = f"{y1} is {rel1['positive']} than {z1}."
                    else:
                        s1_2 = f"{z1} is {rel1['negative']} than {y1}."
                        
                    # Ordem do Atributo 2
                    order2 = random.sample(selected_names, 3)
                    x2, y2, z2 = order2[0], order2[1], order2[2]
                    
                    if random.random() > 0.5:
                        s2_1 = f"{x2} is {rel2['positive']} than {y2}."
                    else:
                        s2_1 = f"{y2} is {rel2['negative']} than {x2}."
                        
                    if random.random() > 0.5:
                        s2_2 = f"{y2} is {rel2['positive']} than {z2}."
                    else:
                        s2_2 = f"{z2} is {rel2['negative']} than {y2}."
                        
                    # Misturar todas as premissas
                    premises = [s1_1, s1_2, s2_1, s2_2]
                    random.shuffle(premises)
                    context = " ".join(premises)
                    
                    # Escolher o atributo alvo
                    if random.random() > 0.5:
                        target_rel = rel1
                        tx, ty, tz = x1, y1, z1
                        is_target_rel1 = True
                    else:
                        target_rel = rel2
                        tx, ty, tz = x2, y2, z2
                        is_target_rel1 = False
                        
                    final_order1 = list(order1)
                    final_order2 = list(order2)
                    
                    apply_errata = self.mutable_context and (random.random() < 0.4)
                    
                    if apply_errata:
                        errata_type = random.choice(["invert_first", "invert_second"])
                        if errata_type == "invert_first":
                            errata_phrase = f"Wait, {tx} is {target_rel['negative']} than {ty}."
                            context = f"{context} {errata_phrase}"
                            
                            if random.random() > 0.5:
                                question = f"Who is {target_rel['positive']}, {ty} or {tz}?="
                                answer = ty
                            else:
                                question = f"Who is {target_rel['negative']}, {ty} or {tz}?="
                                answer = tz
                            cot_str = f"{ty}>{tx} {tx}>{tz} {ty}>{tz}"
                            
                            if is_target_rel1:
                                final_order1 = [ty, tx, tz]
                            else:
                                final_order2 = [ty, tx, tz]
                        else:
                            errata_phrase = f"Wait, {ty} is {target_rel['negative']} than {tz}."
                            context = f"{context} {errata_phrase}"
                            
                            if random.random() > 0.5:
                                question = f"Who is {target_rel['positive']}, {tx} or {tz}?="
                                answer = tx
                            else:
                                question = f"Who is {target_rel['negative']}, {tx} or {tz}?="
                                answer = tz
                            cot_str = f"{tx}>{tz} {tz}>{ty} {tx}>{ty}"
                            
                            if is_target_rel1:
                                final_order1 = [tx, tz, ty]
                            else:
                                final_order2 = [tx, tz, ty]
                    else:
                        if random.random() > 0.5:
                            question = f"Who is {target_rel['positive']}, {tx} or {tz}?="
                            answer = tx
                        else:
                            question = f"Who is {target_rel['negative']}, {tx} or {tz}?="
                            answer = tz
                        cot_str = f"{tx}>{ty} {ty}>{tz} {tx}>{tz}"
                        
                    extra_info = {
                        "rel1_name": rel1["positive"],
                        "rel1_order": final_order1,
                        "rel2_name": rel2["positive"],
                        "rel2_order": final_order2,
                        "target_rel_name": target_rel["positive"]
                    }
                else:
                    # Clássico (1 atributo)
                    x, y, z = selected_names[0], selected_names[1], selected_names[2]
                    rel = random.choice(relations)
                    
                    if random.random() > 0.5:
                        s1 = f"{x} is {rel['positive']} than {y}."
                    else:
                        s1 = f"{y} is {rel['negative']} than {x}."
                        
                    if random.random() > 0.5:
                        s2 = f"{y} is {rel['positive']} than {z}."
                    else:
                        s2 = f"{z} is {rel['negative']} than {y}."
                        
                    context = f"{s1} {s2}"
                    final_order = list(selected_names)
                    apply_errata = self.mutable_context and (random.random() < 0.4)
                    
                    if apply_errata:
                        errata_type = random.choice(["invert_first", "invert_second"])
                        if errata_type == "invert_first":
                            errata_phrase = f"Wait, {x} is {rel['negative']} than {y}."
                            context = f"{context} {errata_phrase}"
                            if random.random() > 0.5:
                                question = f"Who is {rel['positive']}, {y} or {z}?="
                                answer = y
                            else:
                                question = f"Who is {rel['negative']}, {y} or {z}?="
                                answer = z
                            cot_str = f"{y}>{x} {x}>{z} {y}>{z}"
                            final_order = [y, x, z]
                        else:
                            errata_phrase = f"Wait, {y} is {rel['negative']} than {z}."
                            context = f"{context} {errata_phrase}"
                            if random.random() > 0.5:
                                question = f"Who is {rel['positive']}, {x} or {z}?="
                                answer = x
                            else:
                                question = f"Who is {rel['negative']}, {x} or {z}?="
                                answer = z
                            cot_str = f"{x}>{z} {z}>{y} {x}>{y}"
                            final_order = [x, z, y]
                    else:
                        if random.random() > 0.5:
                            question = f"Who is {rel['positive']}, {x} or {z}?="
                            answer = x
                        else:
                            question = f"Who is {rel['negative']}, {x} or {z}?="
                            answer = z
                        cot_str = f"{x}>{y} {y}>{z} {x}>{z}"
                        
                    extra_info = {
                        "rel1_name": rel["positive"],
                        "rel1_order": final_order,
                        "rel2_name": None,
                        "rel2_order": None,
                        "target_rel_name": rel["positive"]
                    }
            else:
                # Caso geral de num_entities > 3
                selected_names = random.sample(names, num_entities)
                rel = random.choice(relations)
                
                sentences = []
                for i in range(num_entities - 1):
                    e_curr = selected_names[i]
                    e_next = selected_names[i+1]
                    if random.random() > 0.5:
                        sentences.append(f"{e_curr} is {rel['positive']} than {e_next}.")
                    else:
                        sentences.append(f"{e_next} is {rel['negative']} than {e_curr}.")
                
                context = " ".join(sentences)
                first_ent = selected_names[0]
                last_ent = selected_names[-1]
                
                if random.random() > 0.5:
                    question = f"Who is {rel['positive']}, {first_ent} or {last_ent}?="
                    answer = first_ent
                else:
                    question = f"Who is {rel['negative']}, {first_ent} or {last_ent}?="
                    answer = last_ent
                
                cot_parts = []
                for i in range(num_entities - 1):
                    cot_parts.append(f"{selected_names[i]}>{selected_names[i+1]}")
                cot_parts.append(f"{first_ent}>{last_ent}")
                cot_str = " ".join(cot_parts)
                
                extra_info = {
                    "rel1_name": rel["positive"],
                    "rel1_order": list(selected_names),
                    "rel2_name": None,
                    "rel2_order": None,
                    "target_rel_name": rel["positive"]
                }
                
            input_str = f"{context} {question}"
            target_str = answer
            
            self.samples.append((input_str, target_str, cot_str, extra_info))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, target_str, cot_str, extra_info = self.samples[idx]
        
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
        
    print("\n=== Testando Dataset Multidimensional (3 Entidades, Múltiplos Atributos) ===")
    ds_multi = LogicDataset(num_samples=5, tokenizer=tok, multidimensional=True, mutable_context=True)
    for i in range(len(ds_multi)):
        item = ds_multi[i]
        sample_info = ds_multi.samples[i][3]
        print(f"Amostra {i+1}:")
        print("  Raw Input:  ", item["raw_input"])
        print("  Raw Target: ", item["raw_target"])
        print("  Raw CoT:    ", item["raw_cot"])
        print("  Extra Info: ", sample_info)
        print("-" * 50)
