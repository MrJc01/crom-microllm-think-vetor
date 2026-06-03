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
        
    import argparse
    parser = argparse.ArgumentParser(description="Playground Interativo Think-Vetor")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Caminho para a pasta do adaptador LoRA")
    args = parser.parse_args()
    
    adapter_dir = args.adapter_dir
    if not adapter_dir:
        # Detectar de forma inteligente adaptadores locais em checkpoints/
        available_adapters = []
        checkpoints_root = "checkpoints"
        if os.path.exists(checkpoints_root):
            for item in sorted(os.listdir(checkpoints_root)):
                item_path = os.path.join(checkpoints_root, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "adapter_config.json")):
                    available_adapters.append(item_path)
                    
        if len(available_adapters) == 1:
            adapter_dir = available_adapters[0]
            print(f"[INFO] Adaptador LoRA detectado e carregado automaticamente: {adapter_dir}")
        elif len(available_adapters) > 1:
            print("\nMúltiplos adaptadores LoRA cognitivos detectados:")
            for idx, path in enumerate(available_adapters):
                print(f"  [{idx + 1}] {path}")
            try:
                choice = input("\nSelecione o número do modelo que deseja carregar > ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_adapters):
                    adapter_dir = available_adapters[choice_idx]
                else:
                    adapter_dir = available_adapters[0]
            except Exception:
                adapter_dir = available_adapters[0]
        else:
            adapter_dir = "checkpoints/think_vetor_1b_lora"
            
    if not os.path.exists(adapter_dir):
        print(f"[ERRO] Diretório de adaptadores LoRA '{adapter_dir}' não encontrado.")
        print("Certifique-se de extrair o zip de adaptadores na pasta checkpoints/.")
        sys.exit(1)
        
    # 1. Carregar metadados do adaptador
    import json
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct" # Fallback padrão
    
    # Priorizar a detecção a partir do adapter_config.json padrão do PEFT
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                peft_config = json.load(f)
            base_model_id = peft_config.get("base_model_name_or_path", base_model_id)
        except Exception as e:
            print(f"[AVISO] Erro ao carregar adapter_config.json: {e}")

    # Fallback para custom config
    config_path = os.path.join(adapter_dir, "think_vetor_config.json")
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
        # Seleção inteligente de dtype no CPU:
        # - Para 0.5B: Usamos Float32 nativo. Consome apenas ~2.0 GB de RAM e executa a velocidade máxima com vetorização AVX2/CPU!
        # - Para 1.5B: Mantemos BFloat16 para economizar 50% de RAM e proteger a memória de 12GB do usuário.
        is_05b = "0.5b" in base_model_id.lower() or "500m" in base_model_id.lower()
        
        if is_05b:
            print("[INFO] Modelo de 0.5B detectado! Forçando Float32 para máxima velocidade nativa no CPU (AVX2/AVX)...")
            dtype_to_use = torch.float32
        else:
            print("[INFO] Modelo maior detectado. Usando BFloat16 para economizar 50% de RAM no CPU...")
            dtype_to_use = torch.bfloat16
            
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=dtype_to_use,
                device_map=None,
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"[AVISO] Falha ao alocar com o dtype selecionado ({e}). Recuando para Float32 padrão...")
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
