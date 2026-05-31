import os
import sys
import torch
import time
import multiprocessing

# Garantir imports corretos da raiz do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Otimização crucial de CPU para PyTorch local
# Limitar threads para o número de cores físicos (geralmente cpu_count // 2) evita thread thrashing e acelera CPU em até 3x-5x!
logical_cores = multiprocessing.cpu_count()
physical_cores = max(1, logical_cores // 2)
torch.set_num_threads(physical_cores)
print(f"[INFO] Otimização de CPU: configurado para usar {physical_cores} threads físicas (total de {logical_cores} lógicas).")

# Tentar importar dependências da Hugging Face
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def print_banner():
    print("="*70)
    print("       PLAYGROUND INTERATIVO COGNITIVO - THINK-VETOR CHAT 1.5B      ")
    print("="*70)
    print("  Fusão de Raciocínio Latente Contínuo e Conhecimento Geral de Mundo")
    print("  Baseado no Qwen2.5-1.5B-Instruct + Adaptadores LoRA Causal CoT")
    print("="*70)

def main():
    print_banner()
    
    if not PEFT_AVAILABLE:
        print("[ERRO] Para rodar a inferência local da LLM de 1.5B com adaptadores LoRA,")
        print("é necessário instalar as bibliotecas do ecossistema Hugging Face.")
        print("\nExecute no seu terminal local:")
        print("  .venv/bin/pip install transformers peft accelerate safetensors")
        print("\nEm seguida, execute este script novamente.")
        sys.exit(1)
        
    adapter_dir = "checkpoints/think_vetor_1b_lora"
    if not os.path.exists(adapter_dir):
        print(f"[ERRO] Diretório de adaptadores LoRA '{adapter_dir}' não encontrado.")
        print("Certifique-se de descompactar o arquivo think_vetor_1b_lora.zip na pasta checkpoints.")
        sys.exit(1)
        
    # 1. Carregar metadados do adaptador
    import json
    config_path = os.path.join(adapter_dir, "think_vetor_config.json")
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct" # Fallback padrão
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                custom_config = json.load(f)
            base_model_id = custom_config.get("base_model", base_model_id)
        except Exception as e:
            print(f"[AVISO] Erro ao carregar think_vetor_config.json: {e}")
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Carregando Modelo Base: {base_model_id}")
    print(f"[INFO] Dispositivo de Execução Selecionado: {device.type.upper()}")
    
    # 2. Inicializar Tokenizer
    print("[INFO] Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    
    # 3. Carregar Modelo Base
    print("[INFO] Carregando pesos do modelo base (isso pode levar de 30 a 60 segundos no CPU)...")
    if device.type == "cuda":
        # Carregamento otimizado na GPU em meia precisão
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Carregamento no CPU local em BFloat16 para economizar 50% de RAM e evitar travamentos
        try:
            print("[INFO] Carregando pesos no CPU em BFloat16 para economizar 50% de RAM...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"[AVISO] Falha ao alocar em BFloat16 ({e}). Usando Float32 padrão (cuidado com consumo de RAM)...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            ).to(device)
        
    # 4. Acoplar os Adaptadores LoRA do PEFT
    print(f"[INFO] Acoplando adaptadores LoRA de: {adapter_dir}...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print("\n[SUCESSO] Think-Vetor Chat 1.5B carregado com sucesso!\n")
    
    print("Digite suas interações e desfrute de raciocínio de alta fidelidade.")
    print("Para sair, pressione Ctrl+C ou digite 'sair'.")
    print("-" * 70)
    
    try:
        while True:
            prompt = input("\nPrompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["sair", "exit", "quit"]:
                print("\nSaindo do Playground. Até mais!")
                break
                
            # Formatar no chat template do modelo
            messages = [
                {
                    "role": "system", 
                    "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Gerar texto formatado
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenizar e enviar ao dispositivo
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            print("\nRefletindo no Espaço Latente...")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            # Decodificar resposta
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Separar a tag <thought> e a resposta final de forma estética
            thought_content = ""
            final_response = full_response
            
            if "<thought>" in full_response and "</thought>" in full_response:
                try:
                    parts = full_response.split("</thought>")
                    thought_content = parts[0].replace("<thought>", "").strip()
                    final_response = parts[1].strip()
                except Exception:
                    pass
            elif "<thought>" in full_response:
                parts = full_response.split("<thought>")
                final_response = parts[0].strip()
                thought_content = parts[1].strip()
                
            print("\n" + "="*50)
            if thought_content:
                print("🧠 [PENSAMENTO COGNITIVO LATENTE]")
                for line in thought_content.split("\n"):
                    print(f"  | {line}")
                print("-" * 50)
                
            print("💬 [RESPOSTA DO ASSISTENTE]")
            print(final_response)
            print("="*50 + "\n")
            
    except (KeyboardInterrupt, EOFError):
        print("\n\nSaindo do Playground. Até mais!")

if __name__ == "__main__":
    main()
