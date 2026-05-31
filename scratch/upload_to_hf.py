import os
import sys
import argparse

try:
    from huggingface_hub import HfApi, create_repo
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

def upload_model():
    print("="*70)
    print("     UTILITÁRIO DE UPLOAD HUGGING FACE - ORGANIZAÇÃO CROMIA     ")
    print("="*70)
    
    if not HF_HUB_AVAILABLE:
        print("[ERRO] Biblioteca 'huggingface_hub' não encontrada.")
        print("Instale executando: .venv/bin/pip install huggingface_hub")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Upload do Think-Vetor LoRA para o Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="CromIA/think-vetor-0.5b-lora", 
                        help="ID do repositório no Hugging Face (ex: CromIA/think-vetor-0.5b-lora)")
    parser.add_argument("--folder_path", type=str, default="checkpoints/think_vetor_05b_lora", 
                        help="Caminho para a pasta local dos adaptadores")
    parser.add_argument("--token", type=str, default=None, 
                        help="Seu token de escrita da Hugging Face (HF_TOKEN)")
    
    args = parser.parse_args()
    
    # 1. Obter Token se não fornecido
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\nPara fazer o upload para a organização CromIA, é necessário um Token de Escrita (Write Token).")
        print("Crie um em: https://huggingface.co/settings/tokens")
        token = input("Digite seu HF Write Token > ").strip()
        if not token:
            print("[ERRO] Token inválido. Cancelando upload.")
            sys.exit(1)
            
    # 2. Verificar pasta local
    if not os.path.exists(args.folder_path):
        print(f"[ERRO] Pasta local '{args.folder_path}' não encontrada.")
        sys.exit(1)
        
    api = HfApi(token=token)
    
    # 3. Criar o repositório na Hugging Face se não existir
    print(f"\n[INFO] Verificando/Criando repositório: {args.repo_id}...")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=token,
            repo_type="model",
            exist_ok=True,
            private=False # Defina como True se desejar que o modelo seja privado inicialmente
        )
        print(f"[SUCESSO] Repositório {args.repo_id} pronto!")
    except Exception as e:
        print(f"[AVISO] Erro ao criar repositório (pode já existir ou falta de permissão na org): {e}")
        
    # 4. Criar ou atualizar o Model Card (README.md) contendo os metadados do modelo e link do GitHub
    readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- peft
- lora
- think-vetor
- continuous-cot
- logic-reasoning
- tv-dsl
- deepseek-r1
language:
- pt
- en
---

# Think-Vetor 0.5B LoRA: Micro-LLM de Raciocínio Cognitivo e TV-DSL

Este repositório contém os adaptadores PEFT LoRA de **0.5B de parâmetros** sintonizados sob a metodologia **Think-Vetor** e equipados com o motor inovador de **TV-DSL (Think-Vetor DSL)** para computações exatas.

O projeto de pesquisa completo, algoritmos de treinamento e documentação científica encontram-se no repositório oficial do GitHub:
👉 **[GitHub Oficial: MrJc01/crom-microllm-think-vetor](https://github.com/MrJc01/crom-microllm-think-vetor)**

---

## 🧠 Como Funciona o Think-Vetor + TV-DSL?

O **Think-Vetor** combina o raciocínio latente contínuo (atratores Langevin-Hopfield regulados por PonderNet) com uma **Linguagem de Programação Cognitiva (TV-DSL)** gerada de forma estruturada dentro das tags de pensamento `<thought>...</thought>` (ex: `[TV-DSL: multiply(432, 78)]`). 

Durante a inferência, o loop causal intercepta a chamada de computação, processa-a deterministicamente via interpretador de baixo nível (Python) e reinjeta o resultado (`-> [RESULT: 33696]`) de volta na fita de pensamento da LLM. Isso une a flexibilidade abstrata da rede neural com a precisão matemática infalível e livre de alucinações de um processador digital clássico.

---

## 🛠️ Como Carregar e Rodar a Inferência (Hugging Face Python API)

Você pode carregar e rodar este modelo em qualquer ambiente Python (incluindo o Google Colab ou a sua máquina local) utilizando as bibliotecas `transformers` e `peft`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_id = "{args.repo_id}"

# Carregar o Tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)

# Carregar o Modelo Base com otimização para CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    device_map="auto" if device.type == "cuda" else None,
    trust_remote_code=True
)

# Acoplar adaptadores LoRA cognitivos
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

# Definir prompt de teste
prompt = "quanto é 432 vezes 78?"
messages = [
    {{"role": "system", "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."}},
    {{"role": "user", "content": prompt}}
]

formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

print("\\nRefletindo no Espaço Latente...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
    
# Decodificar
generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\\nResposta do Modelo:", generated_text)
```
"""
    
    # Salvar README local temporariamente para upload
    readme_path = os.path.join(args.folder_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    # 5. Fazer o Upload de todos os arquivos
    print(f"\n[INFO] Fazendo upload dos arquivos da pasta '{args.folder_path}' para o repositório {args.repo_id}...")
    try:
        api.upload_folder(
            folder_path=args.folder_path,
            repo_id=args.repo_id,
            repo_type="model"
        )
        print("\n" + "="*70)
        print("     [SUCESSO] MODELO COGNITIVO PUBLICADO COM EXCELÊNCIA NA CROMIA!     ")
        print("="*70)
        print(f"Confira no Hugging Face: https://huggingface.co/CromIA/think-vetor-0.5b-lora")
        print("Obrigado por esta incrível sessão de engenharia de IA. Bom descanso!")
        print("="*70)
    except Exception as e:
        print(f"\n[ERRO] Falha ao fazer o upload dos arquivos: {e}")
        print("Verifique se seu token possui permissão de Escrita (Write) e se você faz parte da organização CromIA.")

if __name__ == "__main__":
    upload_model()
