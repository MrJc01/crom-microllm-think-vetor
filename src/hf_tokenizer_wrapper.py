import sys
from src.logic_dataset import LogicCharTokenizer

class HFTokenizerWrapper:
    """
    Wrapper que encapsula qualquer tokenizer da biblioteca HuggingFace transformers.
    Fornece fallback para LogicCharTokenizer se transformers não estiver instalado.
    """
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.using_hf = False
        
        try:
            # Import dinâmico da HuggingFace
            from transformers import AutoTokenizer
            print(f"[INFO] Carregando tokenizer HuggingFace '{model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Garantir existência de token de pad
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    
            self.pad_id = self.tokenizer.pad_token_id
            self.vocab_size = self.tokenizer.vocab_size
            self.using_hf = True
            
        except (ModuleNotFoundError, ImportError):
            print(f"[WARNING] Biblioteca 'transformers' não encontrada localmente. Usando fallback para LogicCharTokenizer.")
            self.tokenizer = LogicCharTokenizer()
            self.pad_id = self.tokenizer.pad_id
            self.vocab_size = self.tokenizer.vocab_size
            self.using_hf = False

    def encode(self, text):
        if self.using_hf:
            # Retorna lista simples de IDs (sem tokens de início/fim automáticos)
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.tokenizer.encode(text)

    def decode(self, ids):
        if isinstance(ids, list):
            pass
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
        else:
            ids = list(ids)
            
        # Parar no primeiro pad_id (EOS)
        decoded_ids = []
        for token_id in ids:
            if token_id == self.pad_id:
                break
            decoded_ids.append(token_id)
            
        if self.using_hf:
            return self.tokenizer.decode(decoded_ids)
        else:
            return self.tokenizer.decode(decoded_ids)

if __name__ == "__main__":
    # Teste rápido
    print("=== Testando HFTokenizerWrapper ===")
    wrapper = HFTokenizerWrapper("gpt2")
    
    text = "Alice is older than Bob."
    encoded = wrapper.encode(text)
    decoded = wrapper.decode(encoded)
    
    print(f"Texto original:  '{text}'")
    print(f"IDs Codificados: {encoded}")
    print(f"Texto Decodificado: '{decoded}'")
    print(f"Vocab size:      {wrapper.vocab_size}")
    print(f"Pad ID:          {wrapper.pad_id}")
    print(f"Using HF:        {wrapper.using_hf}")
