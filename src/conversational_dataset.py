import random
import torch
from torch.utils.data import Dataset

from src.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.logic_dataset import LogicDataset
from src.arithmetic_word_dataset import ArithmeticWordDataset

class ConversationalDataset(Dataset):
    """
    Dataset multi-tarefa unificado que combina:
    1. Diálogos e saudações em linguagem natural (português e inglês).
    2. Raciocínio lógico de transitividade relacional.
    3. Word problems aritméticos (soma/subtração e comparação).
    
    Tudo tokenizado de forma integrada por subpalavras BPE (ou caractere como fallback).
    """
    def __init__(self, num_samples=1000, seed=42, tokenizer=None, max_input_len=120, max_target_len=50):
        super().__init__()
        self.tokenizer = tokenizer or HFTokenizerWrapper("gpt2")
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        
        random.seed(seed)
        self.samples = []
        
        # 1. Definir diálogos e saudações base (Built-in Chat)
        dialogues = [
            # Identidade e Apresentação
            ("quem é você?", "Eu sou o Think-Vetor, uma Micro-LLM focada em raciocínio latente contínuo baseada em Langevin-Hopfield.", "Intenção: Identidade. Tom: Informativo. Plano: Dizer que sou o Think-Vetor e explicar minha arquitetura."),
            ("qual o seu nome?", "Meu nome é Think-Vetor, sou um assistente cognitivo de raciocínio contínuo.", "Intenção: Identidade. Tom: Amigável. Plano: Apresentar-me pelo nome."),
            ("o que é você?", "Sou o Think-Vetor, uma Micro-LLM dotada de um loop de reflexão latente contínua ancorada em memórias baseadas em energia.", "Intenção: Identidade. Tom: Técnico. Plano: Explicar que sou uma rede neural com EBM latente."),
            ("me fale sobre você", "Fui projetado para realizar deduções lógicas e operações aritméticas usando loops recorrentes latentes e atratores de Hopfield.", "Intenção: Identidade. Tom: Informativo. Plano: Enumerar minhas características arquiteturais."),
            
            # Saudações
            ("oi", "Olá! Como posso te ajudar hoje?", "Intenção: Saudação. Tom: Cordial. Plano: Responder amigavelmente e oferecer ajuda."),
            ("olá", "Olá! Tudo bem? Sou o Think-Vetor. Em que posso ser útil?", "Intenção: Saudação. Tom: Simpático. Plano: Responder com cordialidade e me colocar à disposição."),
            ("bom dia", "Bom dia! Como posso te ajudar hoje?", "Intenção: Saudação. Tom: Cordial. Plano: Responder ao bom dia amigavelmente."),
            ("boa tarde", "Boa tarde! Como posso ajudar você hoje?", "Intenção: Saudação. Tom: Cordial. Plano: Responder à boa tarde."),
            ("boa noite", "Boa noite! Em que posso ajudar você?", "Intenção: Saudação. Tom: Cordial. Plano: Responder à boa noite."),
            ("como vai?", "Vou muito bem, obrigado por perguntar! Como posso ser útil?", "Intenção: Saudação. Tom: Cortês. Plano: Responder que estou bem e oferecer ajuda."),
            ("olá, tudo bem?", "Olá! Tudo ótimo por aqui. Espero que com você também! Como posso te ajudar?", "Intenção: Saudação. Tom: Caloroso. Plano: Desejar o bem do usuário e oferecer suporte."),
            
            # Agradecimentos
            ("obrigado", "De nada! Estou sempre à disposição para ajudar com lógica e aritmética.", "Intenção: Agradecimento. Tom: Amigável. Plano: Dizer de nada e colocar-me à disposição."),
            ("muito obrigado", "É um prazer ajudar! Sempre que precisar de raciocínio lógico, estarei por aqui.", "Intenção: Agradecimento. Tom: Cordial. Plano: Agradecer e reforçar minha utilidade."),
            ("valeu", "De nada! Tamo junto. Precisando de qualquer dedução lógicas, é só chamar!", "Intenção: Agradecimento. Tom: Informal. Plano: Responder informalmente."),
            
            # Despedidas
            ("tchau", "Até logo! Tenha um excelente dia.", "Intenção: Despedida. Tom: Gentil. Plano: Despedir-me desejando um bom dia."),
            ("até logo", "Até mais! Sempre que precisar de raciocínio cognitivo, estarei aqui.", "Intenção: Despedida. Tom: Gentil. Plano: Despedir-me colocando-me à disposição."),
            ("tchau, tchau", "Tchau! Até a próxima conversa.", "Intenção: Despedida. Tom: Amigável. Plano: Despedir-me de forma simples."),
            
            # Habilidades
            ("o que você sabe fazer?", "Consigo processar conversações em linguagem natural, resolver deduções de transitividade lógica e fazer contas aritméticas por texto.", "Intenção: Habilidades. Tom: Útil. Plano: Explicar minhas habilidades em linguagem, lógica e matemática."),
            ("quais são suas habilidades?", "Minhas habilidades abrangem conversação fluida, resolução de problemas de transitividade relacional e word problems aritméticos combinados.", "Intenção: Habilidades. Tom: Informativo. Plano: Listar meus domínios cognitivos."),
            ("me ajude", "Claro! Posso ajudar você a resolver problemas matemáticos descritivos ou deduções de lógica relacional. Qual a sua dúvida?", "Intenção: Ajuda. Tom: Colaborativo. Plano: Oferecer ajuda explicando meus focos lógicos.")
        ]
        
        # Enriquecer diálogos com variações em inglês
        dialogues_en = [
            ("hello", "Hello! How can I help you today?", "Intent: Greeting. Tone: Cordial. Plan: Greet and offer assistance."),
            ("hi", "Hi there! I am Think-Vetor. How can I be useful?", "Intent: Greeting. Tone: Friendly. Plan: Greet and state my name."),
            ("good morning", "Good morning! How can I help you?", "Intent: Greeting. Tone: Cordial. Plan: Greet with good morning."),
            ("who are you?", "I am Think-Vetor, a micro-LLM designed for continuous latent reasoning in embedding space.", "Intent: Identity. Tone: Informative. Plan: Explain that I am Think-Vetor."),
            ("what can you do?", "I can engage in conversation, solve logic transitivity problems, and perform math additions.", "Intent: Capabilities. Tone: Useful. Plan: List my conversational, logical, and math skills."),
            ("thank you", "You are very welcome! I am always here to assist with logical and arithmetic problems.", "Intent: Gratitude. Tone: Courteous. Plan: Say you're welcome and offer further help."),
            ("bye", "Goodbye! Have a wonderful day.", "Intent: Farewell. Tone: Kind. Plan: Bid farewell and wish a good day.")
        ]
        
        all_dialogues = dialogues + dialogues_en
        
        # Multiplicar diálogos com ligeiras perturbações lúdicas para atingir a proporção desejada
        num_chat = int(num_samples * 0.4) # 40% chat
        for _ in range(num_chat):
            raw_in, raw_tgt, raw_cot = random.choice(all_dialogues)
            # Adiciona pequenas variações de pontuação ou capitalização
            if random.random() > 0.5:
                raw_in = raw_in.upper() if random.random() > 0.8 else raw_in.capitalize()
            self.samples.append((raw_in, raw_tgt, raw_cot))
            
        # 2. Raciocínio Lógico (30% do dataset)
        num_logic = int(num_samples * 0.3)
        # Instanciar LogicDataset com o mesmo tokenizer
        logic_ds = LogicDataset(num_samples=num_logic, tokenizer=self.tokenizer, seed=seed+1, mutable_context=True)
        for sample in logic_ds.samples:
            prompt = sample[0].strip().rstrip("=")
            self.samples.append((prompt, sample[1], sample[2]))
            
        # 3. Raciocínio Aritmético (30% do dataset)
        num_arith = num_samples - len(self.samples)
        arith_ds = ArithmeticWordDataset(num_samples=num_arith, tokenizer=self.tokenizer, seed=seed+2, difficulty="medium", mutable_context=True)
        for sample in arith_ds.samples:
            prompt = sample[0].strip().rstrip("=")
            self.samples.append((prompt, sample[1], sample[2]))
            
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, target_str, cot_str = self.samples[idx]
        
        # O prompt para o decodificador final nas conversações reais
        # Garantimos que termina em '=' como gatilho causal de decodificação
        input_str_full = input_str + "="
        
        input_ids = self.tokenizer.encode(input_str_full)
        target_ids = self.tokenizer.encode(target_str)
        cot_ids = self.tokenizer.encode(cot_str)
        
        # Padding esquerdo na entrada (RoPE)
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
    print("=== Testando ConversationalDataset (BPE / Fallback) ===")
    wrapper = HFTokenizerWrapper("gpt2")
    ds = ConversationalDataset(num_samples=15, tokenizer=wrapper)
    
    print(f"Total de amostras geradas: {len(ds)}")
    
    # Exibe amostras de categorias variadas
    for i in range(min(5, len(ds))):
        item = ds[i]
        print(f"\nAmostra {i+1}:")
        print("  Input Raw:  ", item["raw_input"])
        print("  Target Raw: ", item["raw_target"])
        print("  CoT Raw:    ", item["raw_cot"])
        print("  Input IDs shape: ", item["input_ids"].shape)
        print("  Target IDs shape:", item["target_ids"].shape)
        print("-" * 50)
