import random
import torch
from torch.utils.data import Dataset
from src.logic_dataset import LogicCharTokenizer

class ArithmeticWordDataset(Dataset):
    """
    Dataset sintético de raciocínio aritmético textual (Word Problems).
    Combina premissas descritivas em texto natural com operações matemáticas.
    
    Exemplo Easy (Comparação):
        Input:  "Alice has 24 cards. Bob has 18 cards. Who has more?="
        Target: "Alice"
        CoT:    "24>18 Alice"
        
    Exemplo Medium (Transferência):
        Input:  "Alice has 35 cards. She gives 12 cards to Bob. Bob had 15 cards. Who has more?="
        Target: "Alice"
        CoT:    "35-12=23 15+12=27 23<27 Bob" (Wait, Bob actually has more here: Alice: 23, Bob: 27)
        
    Exemplo Hard (Multi-step):
        Input:  "Alice has 15 coins. Bob has 28 coins. Alice finds 19 coins. Who has more now?="
        Target: "Alice"
        CoT:    "15+19=34 34>28 Alice"
    """
    def __init__(self, num_samples=1000, seed=42, tokenizer=None, max_input_len=120, max_target_len=15, difficulty="easy", mutable_context=False):
        super().__init__()
        if isinstance(tokenizer, str):
            from src.hf_tokenizer_wrapper import HFTokenizerWrapper
            self.tokenizer = HFTokenizerWrapper(tokenizer)
        else:
            self.tokenizer = tokenizer or LogicCharTokenizer()
            
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.difficulty = difficulty
        self.mutable_context = mutable_context
        
        random.seed(seed)
        self.samples = []
        
        names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
        
        for _ in range(num_samples):
            # Escolhe nomes distintos
            p1, p2 = random.sample(names, 2)
            
            # Faixas numéricas controladas
            if difficulty == "easy":
                # Aritmética simples / comparação direta
                val1 = random.randint(1, 50)
                val2 = random.randint(1, 50)
                while val1 == val2:
                    val2 = random.randint(1, 50)
                    
                input_str = f"{p1} has {val1} cards. {p2} has {val2} cards. Who has more?="
                if val1 > val2:
                    target_str = p1
                    cot_str = f"{val1}>{val2} {p1}"
                else:
                    target_str = p2
                    cot_str = f"{val1}<{val2} {p2}"
                    
            elif difficulty == "medium":
                # Transferência (Soma e Subtração)
                val1 = random.randint(15, 50)
                give_val = random.randint(3, 12)
                val2 = random.randint(5, 30)
                
                input_str = f"{p1} has {val1} cards. {p1} gives {give_val} to {p2}. {p2} had {val2}. Who has more?="
                
                final_val1 = val1 - give_val
                final_val2 = val2 + give_val
                
                if final_val1 > final_val2:
                    target_str = p1
                    op_sym = ">"
                elif final_val1 < final_val2:
                    target_str = p2
                    op_sym = "<"
                else:
                    target_str = "tie"
                    op_sym = "="
                    
                cot_str = f"{val1}-{give_val}={final_val1} {val2}+{give_val}={final_val2} {final_val1}{op_sym}{final_val2} {target_str}"
                
            elif difficulty == "hard":
                # Multi-step
                val1 = random.randint(5, 40)
                val2 = random.randint(10, 60)
                find_val = random.randint(10, 40)
                
                input_str = f"{p1} has {val1} coins. {p2} has {val2} coins. {p1} finds {find_val} more. Who has more now?="
                
                final_val1 = val1 + find_val
                
                if final_val1 > val2:
                    target_str = p1
                    op_sym = ">"
                elif final_val1 < val2:
                    target_str = p2
                    op_sym = "<"
                else:
                    target_str = "tie"
                    op_sym = "="
                    
                cot_str = f"{val1}+{find_val}={final_val1} {final_val1}{op_sym}{val2} {target_str}"
                
            else:
                raise ValueError(f"Dificuldade desconhecida: {difficulty}")
                
            # Tratamento estocástico de errata/retificação (mutable_context)
            if self.mutable_context and random.random() < 0.3:
                # Modifica o final com errata de correção numérica
                if difficulty == "easy":
                    new_val1 = random.randint(1, 50)
                    while new_val1 == val2 or new_val1 == val1:
                        new_val1 = random.randint(1, 50)
                    errata_phrase = f" Wait, {p1} actually has {new_val1}."
                    input_str = input_str.replace("Who has more?=", f"Who has more?={errata_phrase}")
                    input_str = input_str.replace(f"{p1} has {val1}", f"{p1} has {new_val1}") # retroativa na entrada real ou apenas anexa e resolve no CoT?
                    # Para ser consistente com o CoT e processamento causal, anexamos a errata no prompt e mudamos o CoT
                    input_str = f"{p1} has {val1} cards. {p2} has {val2} cards. Wait, {p1} has {new_val1} cards instead. Who has more?="
                    if new_val1 > val2:
                        target_str = p1
                        cot_str = f"{new_val1}>{val2} {p1}"
                    else:
                        target_str = p2
                        cot_str = f"{new_val1}<{val2} {p2}"
                        
                elif difficulty == "medium":
                    new_give_val = random.randint(3, 12)
                    while new_give_val == give_val:
                        new_give_val = random.randint(3, 12)
                    input_str = f"{p1} has {val1} cards. {p1} gives {give_val} to {p2}. {p2} had {val2}. Wait, {p1} gave {new_give_val} instead. Who has more?="
                    
                    final_val1 = val1 - new_give_val
                    final_val2 = val2 + new_give_val
                    
                    if final_val1 > final_val2:
                        target_str = p1
                        op_sym = ">"
                    elif final_val1 < final_val2:
                        target_str = p2
                        op_sym = "<"
                    else:
                        target_str = "tie"
                        op_sym = "="
                        
                    cot_str = f"{val1}-{new_give_val}={final_val1} {val2}+{new_give_val}={final_val2} {final_val1}{op_sym}{final_val2} {target_str}"
                    
                elif difficulty == "hard":
                    new_find_val = random.randint(10, 40)
                    while new_find_val == find_val:
                        new_find_val = random.randint(10, 40)
                    input_str = f"{p1} has {val1} coins. {p2} has {val2} coins. {p1} finds {find_val} coins. Wait, {p1} found {new_find_val} instead. Who has more now?="
                    
                    final_val1 = val1 + new_find_val
                    
                    if final_val1 > val2:
                        target_str = p1
                        op_sym = ">"
                    elif final_val1 < val2:
                        target_str = p2
                        op_sym = "<"
                    else:
                        target_str = "tie"
                        op_sym = "="
                        
                    cot_str = f"{val1}+{new_find_val}={final_val1} {final_val1}{op_sym}{val2} {target_str}"
            
            extra_info = {
                "p1": p1,
                "p2": p2,
                "difficulty": difficulty
            }
            
            self.samples.append((input_str, target_str, cot_str, extra_info))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, target_str, cot_str, extra_info = self.samples[idx]
        
        input_ids = self.tokenizer.encode(input_str)
        target_ids = self.tokenizer.encode(target_str)
        cot_ids = self.tokenizer.encode(cot_str)
        
        # Padding esquerdo na entrada
        input_pad_len = self.max_input_len - len(input_ids)
        if input_pad_len > 0:
            input_ids = [self.tokenizer.pad_id] * input_pad_len + input_ids
        else:
            input_ids = input_ids[:self.max_input_len]
            
        # Padding direito no CoT
        cot_pad_len = self.max_input_len - len(cot_ids)
        if cot_pad_len > 0:
            cot_ids = cot_ids + [self.tokenizer.pad_id] * cot_pad_len
        else:
            cot_ids = cot_ids[:self.max_input_len]
            
        # Padding direito no alvo
        target_pad_len = self.max_target_len - len(target_ids)
        padded_target_ids = target_ids + [self.tokenizer.pad_id] * target_pad_len
        
        # Máscara do loss para incluir EOS/Stop token
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
    print("=== Testando ArithmeticWordDataset ===")
    for diff in ["easy", "medium", "hard"]:
        ds = ArithmeticWordDataset(num_samples=3, difficulty=diff, tokenizer=tok, mutable_context=True)
        print(f"\nDificuldade: {diff.upper()}")
        for i in range(len(ds)):
            item = ds[i]
            print(f"Amostra {i+1}:")
            print("  Input:  ", item["raw_input"])
            print("  Target: ", item["raw_target"])
            print("  CoT:    ", item["raw_cot"])
            print("-" * 40)
