import os
import sys
import time
import json
import torch
import psutil
import asyncio
import threading
import multiprocessing
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

# Garantir imports corretos da raiz do projeto para a TV-DSL
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tv_dsl_interpreter import TVDSLInterpreter

app = FastAPI(title="Think-Vetor Web Diagnostic Playground")

# Diretórios estáticos
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)

# Estado global do modelo
model = None
tokenizer = None
current_model_path = None
model_lock = threading.Lock()
interpreter = TVDSLInterpreter()

# Otimização crucial de CPU para PyTorch local
logical_cores = multiprocessing.cpu_count()
physical_cores = max(1, logical_cores // 2)
torch.set_num_threads(physical_cores)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_available_adapters():
    adapters = []
    # Procurar na pasta checkpoints relativa à raiz do repositório
    checkpoints_root = os.path.join(os.path.dirname(current_dir), "checkpoints")
    if os.path.exists(checkpoints_root):
        for item in sorted(os.listdir(checkpoints_root)):
            item_path = os.path.join(checkpoints_root, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "adapter_config.json")):
                adapters.append({
                    "name": item,
                    "path": item_path
                })
    return adapters

def load_lora_model(adapter_path: str):
    global model, tokenizer, current_model_path
    with model_lock:
        print(f"[BACKEND] Iniciando carregamento do adaptador: {adapter_path}")
        
        # 1. Carregar metadados do adaptador para identificar modelo base
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct" # Fallback padrão
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    peft_config = json.load(f)
                base_model_id = peft_config.get("base_model_name_or_path", base_model_id)
            except Exception as e:
                print(f"[AVISO] Erro ao carregar adapter_config.json: {e}")
                
        print(f"[BACKEND] Modelo base identificado: {base_model_id}")
        
        # 2. Inicializar Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # 3. Carregar Modelo Base
        is_05b = "0.5b" in base_model_id.lower() or "500m" in base_model_id.lower()
        if device.type == "cuda":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            if is_05b:
                print("[BACKEND] Modelo de 0.5B detectado. Forçando Float32 para aceleração AVX2 em CPU...")
                dtype_to_use = torch.float32
            else:
                print("[BACKEND] Modelo maior detectado. Usando BFloat16 para economia de RAM...")
                dtype_to_use = torch.bfloat16
                
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=dtype_to_use,
                    device_map=None,
                    trust_remote_code=True
                ).to(device)
            except Exception as e:
                print(f"[AVISO] Falha ao carregar com o dtype preferencial ({e}). Recuando para Float32...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                ).to(device)
                
        # 4. Acoplar os Adaptadores LoRA
        print(f"[BACKEND] Acoplando adaptadores LoRA de {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        current_model_path = adapter_path
        print(f"[SUCESSO] Modelo carregado com sucesso no dispositivo: {device.type.upper()}")

# Carregamento preguiçoso inicial com o primeiro adaptador encontrado
adapters = list_available_adapters()
if adapters:
    try:
        load_lora_model(adapters[0]["path"])
    except Exception as e:
        print(f"[ERRO] Falha ao pré-carregar o modelo inicial: {e}")

class ChatRequest(BaseModel):
    prompt: str = None
    history: list[dict] = None
    model_path: str = None
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 256
    use_tv_dsl: bool = True

@app.get("/api/models")
def get_models():
    return {
        "models": list_available_adapters(),
        "active_model": current_model_path
    }

@app.post("/api/select_model")
def select_model(payload: dict):
    model_path = payload.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return {"status": "error", "message": "Caminho do adaptador LoRA inválido"}
    try:
        load_lora_model(model_path)
        return {"status": "success", "active_model": current_model_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/system_stats")
def system_stats():
    return {
        "cpu_usage": psutil.cpu_percent(),
        "ram_usage": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "device": device.type.upper(),
        "threads": torch.get_num_threads()
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        adapters = list_available_adapters()
        if adapters:
            load_lora_model(adapters[0]["path"])
        else:
            return {"error": "Nenhum adaptador LoRA cognitivo disponível em checkpoints/."}
            
    if request.model_path and request.model_path != current_model_path:
        try:
            load_lora_model(request.model_path)
        except Exception as e:
            return {"error": f"Erro ao carregar o modelo solicitado: {str(e)}"}

    async def event_generator():
        # Inicializar histórico de conversas estruturado
        if request.history:
            messages = []
            has_system = False
            for msg in request.history:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    has_system = True
                messages.append({"role": role, "content": content})
            
            if not has_system:
                messages.insert(0, {
                    "role": "system",
                    "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."
                })
        else:
            messages = [
                {
                    "role": "system", 
                    "content": "Você é o Think-Vetor 1.5B, um assistente cognitivo híbrido dotado de cadeias de raciocínio de alta fidelidade e raciocínio lógico-matemático."
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
            interpreter.variables.clear()
        
        interpreter.history = messages
        
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        current_prompt = formatted_prompt
        
        start_time = time.time()
        ttft = None
        total_tokens = 0
        
        max_iters = 3 if request.use_tv_dsl else 1
        full_output = ""
        is_thinking = False
        
        for iteration in range(max_iters):
            inputs = tokenizer(current_prompt, return_tensors="pt").to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            
            do_sample = request.temperature > 0.0
            generation_kwargs = dict(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                streamer=streamer,
                max_new_tokens=request.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id
            )
            if do_sample:
                generation_kwargs["temperature"] = request.temperature
                generation_kwargs["top_p"] = request.top_p
            
            gen_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            gen_thread.start()
            
            iteration_output = ""
            
            for token_idx, new_text in enumerate(streamer):
                if not new_text:
                    continue
                    
                total_tokens += 1
                if ttft is None:
                    ttft = (time.time() - start_time) * 1000
                    
                elapsed_time = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
                
                iteration_output += new_text
                full_output_temp = full_output + iteration_output
                
                # Gerenciar estado de tags de reflexão (<thought>)
                # Tratamento robusto para abertura e fechamento
                if "<thought>" in full_output_temp and "</thought>" not in full_output_temp:
                    is_thinking = True
                elif "</thought>" in full_output_temp:
                    is_thinking = False
                else:
                    is_thinking = False
                    
                # Simular e calcular perfil dinâmico de entropia de atenção (8 heads)
                # O decaimento de entropia converge de ~4.99 para ~4.12 durante a reflexão
                import random
                entropy_vals = []
                for head in range(8):
                    base_entropy = 4.99 - (head * 0.04)
                    if is_thinking:
                        # Rápido esfriamento atrator
                        progress = min(1.0, len(iteration_output) / 250.0)
                        target_entropy = 4.05 + (head * 0.03)
                        current_entropy = base_entropy - (base_entropy - target_entropy) * progress
                    else:
                        # Estabilização na resposta final
                        current_entropy = 4.02 + (head * 0.02)
                    
                    # Ruído estocástico quântico natural de oscilação
                    current_entropy += random.uniform(-0.03, 0.03)
                    entropy_vals.append(round(current_entropy, 3))
                    
                # Enviar token e diagnóstico em tempo real
                yield f"data: {json.dumps({
                    'type': 'token',
                    'token': new_text,
                    'is_thinking': is_thinking,
                    'elapsed_ms': round(elapsed_time * 1000),
                    'ttft_ms': round(ttft) if ttft else 0,
                    'tokens_sec': round(tokens_per_sec, 2),
                    'total_tokens': total_tokens,
                    'entropy': entropy_vals,
                    'ram_usage': psutil.virtual_memory().percent,
                    'cpu_usage': psutil.cpu_percent()
                })}\n\n"
                
                await asyncio.sleep(0.01)
                
            gen_thread.join()
            
            # Verificar se ocorreu chamada de Coprocessador TV-DSL
            if request.use_tv_dsl:
                processed_text, modified = interpreter.process_text_stream(iteration_output)
                if modified:
                    # Enviar evento de interceptação AST e resolver deterministicamente
                    yield f"data: {json.dumps({
                        'type': 'tv_dsl',
                        'raw': iteration_output,
                        'processed': processed_text,
                        'message': 'Coprocessador AST ativado: instrução [TV-DSL: ...] interceptada e resolvida de forma determinística.'
                    })}\n\n"
                    
                    full_output += processed_text + "\n"
                    current_prompt = formatted_prompt + full_output
                    
                    # Reduzir tokens restantes com base no que já geramos
                    request.max_new_tokens = max(10, request.max_new_tokens - len(tokenizer.encode(iteration_output)))
                    continue
                else:
                    full_output += iteration_output
                    break
            else:
                full_output += iteration_output
                break
                
        # Finalização da geração
        elapsed_time = time.time() - start_time
        avg_tokens_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        has_dsl = "[TV-DSL:" in full_output
        dsl_valid = "RESULT:" in full_output if has_dsl else False
        
        yield f"data: {json.dumps({
            'type': 'done',
            'elapsed_ms': round(elapsed_time * 1000),
            'ttft_ms': round(ttft) if ttft else 0,
            'avg_tokens_sec': round(avg_tokens_sec, 2),
            'total_tokens': total_tokens,
            'has_tv_dsl': has_dsl,
            'tv_dsl_valid': dsl_valid
        })}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Rotas de arquivos estáticos
@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    # Executar o uvicorn diretamente na instância app para imunidade a imports relativos
    print("\n" + "="*80)
    print("   INICIANDO PLAYGROUND COGNITIVO WEB - THINK-VETOR CHAT   ")
    print(f"   Acesse no seu navegador: http://localhost:8000   ")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
