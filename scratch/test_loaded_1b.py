import sys
import os
import torch
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_dir = "/home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/checkpoints/think_vetor_1b_lora"
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("=== INICIANDO TESTE SISTEMÁTICO DE VELOCIDADE DO 1.5B NO CPU LOCAL ===")
start_load = time.time()

# Forçar bfloat16 no CPU para economizar 50% de RAM e acelerar o processamento
dtype = torch.bfloat16

print(f"Carregando tokenizer e modelo base com dtype={dtype}...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    device_map=None,
    trust_remote_code=True
)

print("Acoplando adaptador LoRA...")
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

print(f"Tempo de carregamento total: {time.time() - start_load:.2f} segundos")

prompt = "oi"
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

formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted_prompt, return_tensors="pt")

print("\nExecutando geração de teste no CPU local...")
start_gen = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30, # Gerar poucos tokens para teste rápido
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

gen_time = time.time() - start_gen
input_len = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_len:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print(f"\nTempo de geração para {len(generated_tokens)} tokens: {gen_time:.2f} segundos")
print(f"Velocidade média de geração: {len(generated_tokens) / gen_time:.2f} tokens/segundo")
print(f"\nResposta gerada:\n{response}")
print("="*60)
