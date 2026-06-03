import os
import sys
import time
import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Otimização de CPU para PyTorch - Seguro contra travamento
torch.set_num_threads(8)
print("[INFO] Thread pool configurado para 8 threads físicas.")

# Adiciona o diretório raiz do projeto ao sys.path para garantir imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tv_dsl_interpreter import TVDSLInterpreter

class GSM8KBenchmark:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.interpreter = TVDSLInterpreter()
        
        # Suite de 15 problemas matemáticos realistas no estilo GSM8K
        self.problems = [
            {
                "prompt": "Weng earns 12 dollars an hour baby-sitting. Yesterday, she baby-sat for 5 hours. How much money did she earn in total?",
                "expected": "60"
            },
            {
                "prompt": "Angelo began school at age 6. He spent 12 years in elementary and high school, and then completed 4 years of college. How old will he be when he graduates college?",
                "expected": "22"
            },
            {
                "prompt": "A baker sells 10 loaves of bread for 2 dollars each, and 15 baguettes for 3 dollars each. How much money did the baker make in total?",
                "expected": "65"
            },
            {
                "prompt": "A library has 120 books. 45 books are checked out on Monday, and 18 are returned on Tuesday. How many books are in the library now?",
                "expected": "93"
            },
            {
                "prompt": "A car travel 60 miles per hour. How many miles does it travel in 3.5 hours?",
                "expected": "210"
            },
            {
                "prompt": "We have 8 boxes. Each box contains 12 packs of crayons. Each pack has 10 crayons. How many crayons are there in total?",
                "expected": "960"
            },
            {
                "prompt": "A notebook costs 4 dollars and a pen costs 1.5 dollars. If John buys 5 notebooks and 10 pens, how much money does he spend?",
                "expected": "35"
            },
            {
                "prompt": "A rectangular garden is 15 meters long and 8 meters wide. What is the area of the garden in square meters?",
                "expected": "120"
            },
            {
                "prompt": "A store has 400 apples. They sell 120 on Saturday and 150 on Sunday. How many apples are left?",
                "expected": "130"
            },
            {
                "prompt": "A student solved 16 math questions on Monday, 22 questions on Tuesday, and 14 questions on Wednesday. What is the total number of questions solved?",
                "expected": "52"
            },
            {
                "prompt": "If a dozen eggs cost 3 dollars, how much do 36 eggs cost?",
                "expected": "9"
            },
            {
                "prompt": "An auditorium has 20 rows of seats, and each row has 15 seats. How many seats are there in total?",
                "expected": "300"
            },
            {
                "prompt": "A worker makes 150 widgets a day. How many widgets can the worker make in 6 days?",
                "expected": "900"
            },
            {
                "prompt": "A bucket has 5 liters of water. We add 2500 milliliters. How many liters of water are in the bucket now?",
                "expected": "7.5"
            },
            {
                "prompt": "A farmer harvested 240 potatoes. He keeps 40 for himself and divides the rest equally among 5 neighbors. How many potatoes does each neighbor receive?",
                "expected": "40"
            }
        ]

    def run_inference(self, prompt, use_tv_dsl=True, max_new_tokens=150):
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
        
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        current_prompt = formatted_prompt
        
        full_generation = ""
        
        # Loop causal com TV-DSL se ativado
        max_iters = 3 if use_tv_dsl else 1
        for iteration in range(max_iters):
            inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if use_tv_dsl:
                processed_text, modified = self.interpreter.process_text_stream(generated_text)
                if modified:
                    current_prompt = formatted_prompt + processed_text + "\n"
                    max_new_tokens = max(10, max_new_tokens - len(generated_tokens))
                    full_generation = processed_text
                    continue
                else:
                    full_generation = generated_text
                    break
            else:
                full_generation = generated_text
                break
                
        # Separar resposta final das tags <thought> se existirem
        final_response = full_generation
        if "</thought>" in full_generation:
            try:
                final_response = full_generation.split("</thought>")[-1].strip()
            except Exception:
                pass
        return final_response

    def evaluate_suite(self, use_tv_dsl=True, num_samples=None):
        correct = 0
        total = 0
        
        problems_to_eval = self.problems
        if num_samples is not None and num_samples > 0:
            problems_to_eval = self.problems[:num_samples]
            
        print(f"\nAvaliando com TV-DSL = {use_tv_dsl} ({len(problems_to_eval)} problemas)...")
        for idx, item in enumerate(problems_to_eval):
            prompt = item["prompt"]
            expected = item["expected"]
            
            response = self.run_inference(prompt, use_tv_dsl=use_tv_dsl)
            
            # Checar se o valor esperado está na resposta
            is_correct = expected.lower() in response.lower()
            if is_correct:
                correct += 1
            total += 1
            
            print(f"  [{idx+1}/{len(problems_to_eval)}] Prompt: \"{prompt}\"")
            print(f"  Esperado: {expected} | Resposta: \"{response}\" | Correto? {is_correct}")
            print("-" * 50)
            
        acc = (correct / total) * 100 if total > 0 else 0.0
        return acc

def main():
    parser = argparse.ArgumentParser(description="Benchmark GSM8K com coprocessador TV-DSL.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Caminho para o adaptador LoRA.")
    parser.add_argument("--num_samples", type=int, default=15, help="Número de problemas do benchmark a avaliar.")
    parser.add_argument("--output_json", type=str, default=None, help="Caminho para salvar os resultados em JSON.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== BENCHMARK COGNITIVO ESTILO GSM8K ===")
    print(f"Dispositivo: {device}")
    print(f"Modelo Base: {args.model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Carregar em Float32 nativo para CPU
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, device_map=None, trust_remote_code=True).to(device)
        
    if args.adapter_path is not None and os.path.exists(args.adapter_path):
        print(f"-> Carregando adaptador LoRA de: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        
    benchmark = GSM8KBenchmark(model, tokenizer, device)
    
    # Rodar testes
    acc_no_dsl = benchmark.evaluate_suite(use_tv_dsl=False, num_samples=args.num_samples)
    acc_with_dsl = benchmark.evaluate_suite(use_tv_dsl=True, num_samples=args.num_samples)
    
    print("\n" + "="*50)
    print("=== RESULTADOS FINAIS BENCHMARK GSM8K ===")
    print(f"Sem TV-DSL (Autorregressivo Puro): {acc_no_dsl:.2f}% de Acurácia")
    print(f"Com TV-DSL (Coprocessamento AST):  {acc_with_dsl:.2f}% de Acurácia")
    print(f"-> Impacto Líquido da TV-DSL: {acc_with_dsl - acc_no_dsl:+.2f}%")
    print("="*50)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "model_id": args.model_id,
                "adapter_path": args.adapter_path,
                "num_samples": args.num_samples,
                "acc_no_dsl": acc_no_dsl,
                "acc_with_dsl": acc_with_dsl,
                "net_impact": acc_with_dsl - acc_no_dsl
            }, f, indent=2, ensure_ascii=False)
        print(f"Resultados do GSM8K salvos em: {args.output_json}")

if __name__ == "__main__":
    main()
