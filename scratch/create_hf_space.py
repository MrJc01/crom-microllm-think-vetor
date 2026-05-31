import os
import sys
import argparse

try:
    from huggingface_hub import HfApi, create_repo
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

def create_space():
    print("="*70)
    print("     UTILITÁRIO DE CRIAÇÃO DE SPACE - HUGGING FACE CROMIA     ")
    print("="*70)
    
    if not HF_HUB_AVAILABLE:
        print("[ERRO] Biblioteca 'huggingface_hub' não encontrada.")
        print("Instale executando: .venv/bin/pip install huggingface_hub")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Publicação do Gradio Space para o Think-Vetor no Hugging Face")
    parser.add_argument("--space_id", type=str, default="CromIA/think-vetor-chat", 
                        help="ID do Space no Hugging Face (ex: CromIA/think-vetor-chat)")
    parser.add_argument("--token", type=str, default=None, 
                        help="Seu token de escrita da Hugging Face (HF_TOKEN)")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\nPara criar o Space na organização CromIA, é necessário seu Token de Escrita (Write Token).")
        token = input("Digite seu HF Write Token > ").strip()
        if not token:
            print("[ERRO] Token inválido. Cancelando criação do Space.")
            sys.exit(1)

    # 1. Definir e escrever o código do Gradio App (app.py) que rodará no Space
    # Inclui o interpretador TV-DSL de forma autônoma para robustez de imports
    app_code = """import os
import time
import ast
import operator as op
import re
import math
import torch
# Otimização crucial de CPU para contêineres Docker no Hugging Face Spaces!
# Impede o thread thrashing limitando as threads lógicas à quota real de 2 vCPUs do Space
torch.set_num_threads(2)
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 1. INTERPRETADOR DA THINK-VETOR DSL (TV-DSL)
# ============================================================
class TVDSLInterpreter:
    SAFE_OPERATORS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos
    }

    def __init__(self):
        self.functions = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "subtract": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "multiply": lambda a, b: a * b,
            "div": lambda a, b: a / b if b != 0 else "Error: Division by zero",
            "divide": lambda a, b: a / b if b != 0 else "Error: Division by zero",
            "pow": lambda a, b: a ** b,
            "power": lambda a, b: a ** b,
            "sqrt": lambda a: math.sqrt(a) if a >= 0 else "Error: Square root of negative number",
            "abs": lambda a: abs(a)
        }

    def safe_eval(self, expr_str: str):
        expr_str = expr_str.strip()
        expr_str = expr_str.replace('^', '**')
        try:
            tree = ast.parse(expr_str, mode='eval')
            return self._eval_node(tree.body)
        except Exception as e:
            return f"Error: Expression parse failure ({str(e)})"

    def _eval_node(self, node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(left, str) or isinstance(right, str):
                return "Error: Invalid operand in binary operation"
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                try:
                    return self.SAFE_OPERATORS[op_type](left, right)
                except ZeroDivisionError:
                    return "Error: Division by zero"
            return f"Error: Unsupported binary operator '{op_type.__name__}'"
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(operand, str):
                return operand
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                return self.SAFE_OPERATORS[op_type](operand)
            return f"Error: Unsupported unary operator '{op_type.__name__}'"
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.functions:
                args = [self._eval_node(arg) for arg in node.args]
                for arg in args:
                    if isinstance(arg, str) and arg.startswith("Error"):
                        return arg
                try:
                    return self.functions[func_name](*args)
                except TypeError:
                    return f"Error: Incorrect argument count"
            return f"Error: Function '{func_name}' is not registered"
        return "Error: AST node blocked"

    def process_text_stream(self, text: str) -> tuple[str, bool]:
        pattern = r"\[TV-DSL:\s*(.*?)\]"
        matches = list(re.finditer(pattern, text))
        if not matches:
            return text, False
        processed_text = text
        offset = 0
        for match in matches:
            expr = match.group(1)
            start, end = match.start() + offset, match.end() + offset
            val = self.safe_eval(expr)
            result_str = f"[TV-DSL: {expr}] -> [RESULT: {val}]"
            processed_text = processed_text[:start] + result_str + processed_text[end:]
            offset += len(result_str) - (end - start)
        return processed_text, True

# ============================================================
# 2. CARREGAMENTO E CONFIGURAÇÃO DO MODELO
# ============================================================
print("[INFO] Carregando pesos do modelo e adaptadores da CromIA no Space...")
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_id = "CromIA/think-vetor-0.5b-lora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CPU Basic do Hugging Face Spaces roda incrivelmente rápido e com vetorização AVX em Float32!
dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    device_map=None,
    trust_remote_code=True
).to(device)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

interpreter = TVDSLInterpreter()

# ============================================================
# 3. ROTINA DE INFERÊNCIA INTERATIVA TV-DSL
# ============================================================
def run_think_vetor_inference(prompt):
    start_time = time.time()
    
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
    current_prompt = formatted_prompt
    
    usou_dsl = False
    full_generation = ""
    max_new_tokens = 256
    
    # Suporte a loops iterativos se a DSL for disparada
    for iteration in range(3):
        inputs = tokenizer(current_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        processed_text, modified = interpreter.process_text_stream(generated_text)
        
        if modified:
            usou_dsl = True
            current_prompt = formatted_prompt + processed_text + "\\n"
            max_new_tokens = max(10, max_new_tokens - len(generated_tokens))
            full_generation = processed_text
            continue
        else:
            full_generation = generated_text
            break
            
    latency = time.time() - start_time
    
    # Separar o thought da resposta final
    thought_content = ""
    final_response = full_generation
    
    if "<thought>" in full_generation and "</thought>" in full_generation:
        try:
            parts = full_generation.split("</thought>")
            thought_content = parts[0].replace("<thought>", "").strip()
            final_response = parts[1].strip()
        except Exception:
            pass
    elif "<thought>" in full_generation:
        parts = full_generation.split("<thought>")
        final_response = parts[0].strip()
        thought_content = parts[1].strip()
        
    return thought_content, final_response, latency, usou_dsl

# ============================================================
# 4. INTERFACE GRÁFICA GRADIO PREMIUM (WOW-FACTOR)
# ============================================================
theme = gr.themes.Default(
    primary_hue="emerald",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
    block_title_text_color="*primary_400",
    input_background_fill="*neutral_900",
    button_primary_background_fill="linear-gradient(90deg, *primary_600, *secondary_600)",
    button_primary_text_color="*white"
)

css = \"\"\"
.cognitive-card {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
    backdrop-filter: blur(5px) !important;
}
.latent-title {
    color: #10b981 !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.chat-window {
    border: 1px solid rgba(6, 182, 212, 0.2) !important;
    border-radius: 12px !important;
}
\"\"\"

with gr.Blocks(title="Think-Vetor Chat - CromIA") as demo:
    gr.HTML(
        \"\"\"
        <div style="text-align: center; margin-bottom: 25px;">
            <h1 style="font-size: 2.2em; font-weight: bold; background: linear-gradient(90deg, #10b981, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🧠 Think-Vetor 0.5B: Playground Cognitivo
            </h1>
            <p style="color: #94a3b8; font-size: 1.1em; margin-top: 5px;">
                Fusão de Raciocínio Contínuo e Computação Determinística de Altíssima Fidelidade (TV-DSL)
            </p>
            <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
                <span style="background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(16, 185, 129, 0.2);">
                    Organization: CromIA
                </span>
                <span style="background: rgba(6, 182, 212, 0.1); color: #06b6d4; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(6, 182, 212, 0.2);">
                    Model Scale: 0.5B LoRA
                </span>
            </div>
        </div>
        \"\"\"
    )
    
    with gr.Row():
        # PAINEL ESQUERDO: Trajetória Cognitiva Latente e TV-DSL
        with gr.Column(scale=1, variant="panel", elem_classes=["cognitive-card"]):
            gr.HTML(
                \"\"\"
                <div class="latent-title">
                    <span>🧠</span> TRAJETÓRIA COGNITIVA LATENTE (SCRATCHPAD)
                </div>
                \"\"\"
            )
            thought_output = gr.Markdown(
                "*Aguardando prompt do usuário para refletir no espaço latente...*",
                label="Processamento do Pensamento"
            )
            gr.HTML("<hr style='border: 0; border-top: 1px solid #334155; margin: 15px 0;'>")
            
            # Painel de Telemetria
            gr.HTML("<div style='color: #06b6d4; font-weight: bold; font-size: 0.9em; margin-bottom: 5px;'>📟 TELEMETRIA FÍSICA</div>")
            with gr.Row():
                latency_box = gr.Textbox(label="Latência Total", placeholder="0.00s", interactive=False)
                dsl_status_box = gr.Textbox(label="Status da TV-DSL", placeholder="Inativo", interactive=False)
                
        # PAINEL DIREITO: Chat com o Assistente
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Think-Vetor Chatbot Window",
                elem_classes=["chat-window"],
                height=450
            )
            
            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False,
                    placeholder="Digite seu prompt de lógica, matemática ou conversação aqui...",
                    scale=4,
                    container=False
                )
                btn_send = gr.Button("Enviar", variant="primary", scale=1)
                
    # Sugestões de Prompt para Teste Rápido
    gr.Examples(
        examples=[
            ["quanto é 432 vezes 78?"],
            ["calcule (150 + 250) * 5"],
            ["Alice is taller than Bob. Bob is taller than Charlie. Who is taller, Alice or Charlie?"],
            ["quem é você?"]
        ],
        inputs=txt_input
    )
    
    # Evento de envio
    def chat_action(user_message, history):
        if not user_message.strip():
            return "", history, "", "", ""
            
        # Executar inferência cognitiva
        thought, response, latency, usou_dsl = run_think_vetor_inference(user_message)
        
        # Formatar a exibição do thought de forma visualmente rica
        formatted_thought = ""
        if thought:
            formatted_thought = f"### 🧠 Pensamento Estruturado:\\n"
            for line in thought.split("\\n"):
                formatted_thought += f"> **|** {line}\\n"
        else:
            formatted_thought = "*Esta resposta foi gerada diretamente sem a necessidade de múltiplos passos de relaxamento de atrator.*"
            
        latency_str = f"{latency:.2f} segundos"
        dsl_str = "🔥 Ativo (Cálculo Determinístico Executado)" if usou_dsl else "Inativo"
        
        # Atualizar histórico do chat no formato do Gradio 5/6 (role/content)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response})
        
        return "", history, formatted_thought, latency_str, dsl_str

    # Conectar botões e envios
    txt_input.submit(chat_action, [txt_input, chatbot], [txt_input, chatbot, thought_output, latency_box, dsl_status_box])
    btn_send.click(chat_action, [txt_input, chatbot], [txt_input, chatbot, thought_output, latency_box, dsl_status_box])

if __name__ == "__main__":
    demo.queue().launch(theme=theme, css=css)
"""

    # 2. Definir o arquivo requirements.txt do Space
    requirements_code = """transformers>=4.40.0
peft>=0.10.0
accelerate>=0.28.0
safetensors>=0.4.0
torch>=2.2.0
gradio>=4.0.0
jinja2
"""

    # 3. Salvar arquivos temporários na pasta do Space
    os.makedirs("space_temp", exist_ok=True)
    with open("space_temp/app.py", "w", encoding="utf-8") as f:
        f.write(app_code)
    with open("space_temp/requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_code)
        
    api = HfApi(token=token)
    
    # 4. Criar o Space na Hugging Face
    print(f"\n[INFO] Criando Space Gradio: {args.space_id}...")
    try:
        create_repo(
            repo_id=args.space_id,
            token=token,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False
        )
        print(f"[SUCESSO] Space {args.space_id} criado!")
    except Exception as e:
        print(f"[AVISO] Erro ao criar Space (pode já existir ou falta de permissão): {e}")
        
    # 5. Enviar os arquivos para o Space
    print(f"\n[INFO] Fazendo upload dos arquivos para o Space {args.space_id}...")
    try:
        api.upload_folder(
            folder_path="space_temp",
            repo_id=args.space_id,
            repo_type="space"
        )
        print("\n" + "="*70)
        print("     [SUCESSO] GRADIO SPACE PUBLICADO COM EXCELÊNCIA NA CROMIA!     ")
        print("="*70)
        print(f"Confira e interaja na Web: https://huggingface.co/spaces/CromIA/think-vetor-chat")
        print("Agora sim seu modelo possui um playground web deslumbrante!")
        print("="*70)
    except Exception as e:
        print(f"\n[ERRO] Falha ao fazer o upload dos arquivos: {e}")
        
    # Limpar pasta temporária
    try:
        import shutil
        shutil.rmtree("space_temp")
    except Exception:
        pass

if __name__ == "__main__":
    create_space()
