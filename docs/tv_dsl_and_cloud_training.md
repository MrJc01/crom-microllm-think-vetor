# Coprocessador TV-DSL Stateful e Treinamento em Nuvem (1.5B)

Este documento documenta os avanços e a consolidação do **Coprocessador TV-DSL (Think-Vetor DSL)** com suporte a estados dinâmicos (variáveis e memória de contexto) e a infraestrutura otimizada para o treinamento de modelos de **1.5B parâmetros (SFT + GRPO-RL)** na nuvem.

---

## 🧠 1. O Coprocessador TV-DSL Stateful (Em-Pensamento)

Diferente do *Function Calling* tradicional que interrompe a geração do modelo, o **Coprocessador TV-DSL** atua de forma síncrona interceptando chamadas computacionais estruturadas diretamente dentro das tags de pensamento latente (`<thought>...</thought>`) da LLM durante o processo de inferência.

### A. Persistência de Estado (Variáveis Locais)
Implementamos uma tabela de símbolos na memória de execução (`self.variables` dentro da classe `TVDSLInterpreter`), permitindo a persistência de variáveis e sua posterior utilização em equações aritméticas.
* **`set(nome, valor)`**: Armazena dinamicamente um valor numérico ou string no dicionário de variáveis.
  * *Exemplo*: `[TV-DSL: set(x, 150)]` $\rightarrow$ Retorna `Stored x = 150`
* **`get(nome)`**: Recupera o valor armazenado na variável.
  * *Exemplo*: `[TV-DSL: get(x)]` $\rightarrow$ Retorna `150`
* **Resolução AST de Variáveis**: O avaliador seguro de Árvores de Sintaxe Abstrata (AST) agora é capaz de ler nomes de variáveis de forma infixada e resolvê-las inline.
  * *Exemplo*: `[TV-DSL: x * 5]` $\rightarrow$ Executa a expressão aritmética resolvendo `x` como `150` e retorna `750`.
* **`clear_vars()`**: Limpa todas as variáveis da sessão.

### B. Resgate Exato do Contexto (`recall`)
Para evitar a deriva conversacional (*persona drift*) e alucinações factuais de contexto, implementamos a ferramenta de recuperação contextual de alta fidelidade:
* **`recall(query)`**: Faz uma varredura de trás para frente no histórico de mensagens da conversa, localizando e retornando o trecho do diálogo correspondente ao termo buscado.
  * *Exemplo*: `[TV-DSL: recall("altura")]` $\rightarrow$ Retorna a premissa exata correspondente e a reinjeta no pensamento latente do modelo.

### C. Integração por Sessão de Conversa
O ciclo de vida do estado do interpretador foi integrado ao pipeline de inferência multi-turn:
* **Playground Web (FastAPI)**: O arquivo `web_playground/app.py` limpa o estado das variáveis de sessão ao iniciar uma nova conversa e repassa o histórico ativo a cada turno, permitindo ao interpretador usar o método `recall` com contexto real.
* **Bateria de Testes (`run_multiturn_tests.py`)**: Limpa os atratores de variáveis e vincula o histórico conversacional a cada um dos 50 testes lógicos executados sequencialmente.

---

## ⚡ 2. Treinamento na Nuvem do Think-Vetor 1.5B (SFT + GRPO-RL)

O treinamento de modelos maiores (como o `Qwen2.5-1.5B-Instruct` ou `Qwen2.5-Math-1.5B-Instruct`) usando o pipeline **GRPO-RL** (Group Relative Policy Optimization) consome quantidades massivas de VRAM e CPU debaixo de amostragem de grupo ($G=4$ completions por prompt). Para possibilitar a execução estável no **Google Colab** (GPU T4) ou **Vast.ai** (GPUs RTX 3090/4090), implementamos otimizações profundas.

### A. Otimizações de Engenharia
1. **Otimizador `PagedAdamW8bit`**: Em vez do otimizador clássico de 32-bits que consome o dobro da VRAM para armazenar os estados de momentum do gradiente, integramos o otimizador paginado em 8-bits da biblioteca `bitsandbytes` automaticamente quando o CUDA está ativo.
2. **LoRA + Quantização de 4-bits (QLoRA)**: Carregamento do modelo base de 1.5B parâmetros em precisão ultra-eficiente `NF4-bits` com computação interna em `BFloat16` e gradientes checkpointing ativados. A pegada de memória do treino GRPO cai para apenas **~5.5 GB de VRAM**, cabendo em GPUs grátis.
3. **Inferencia Otimizada em BFloat16**: O carregador de testes (`run_multiturn_tests.py`) agora detecta a GPU e inicializa os pesos em `torch.bfloat16` nativo com `device_map="auto"`, eliminando gargalos de RAM do sistema na inicialização do PyTorch.

---

## ⏱️ 3. Perfil de Velocidade e Benchmark de Treinamento

Abaixo está o benchmark comparativo de desempenho de execução do treinamento com o dataset de **500 diálogos multi-turn** ao longo de **3 épocas** (SFT no início + GRPO-RL no final):

| Ambiente de Execução | Tempo Total Estimado | Custo Aproximado | Observações |
| :--- | :--- | :--- | :--- |
| **CPU Local** (Sem GPU) | *Inviável / PC Congela* | - | Estouro de memória RAM (OOM) |
| **Google Colab** (GPU T4 - 15GB VRAM) | **25 a 40 minutos** | Grátis / Baixo | A geração paralela do GRPO-RL consome mais ciclos na T4 |
| **Vast.ai** (GPU RTX 3090/4090 - 24GB VRAM) | **5 a 10 minutos** | ~\$0.05 a \$0.10 | Tensor Cores modernos de alta velocidade aceleram a decodificação |

---

## 🛠️ 4. Como Executar na Nuvem

Criamos o Jupyter Notebook unificado **`train_think_vetor_1b_grpo.ipynb`** na raiz do repositório contendo todas as células necessárias para automatizar o setup.

### Passo a Passo:
1. Faça o upload do notebook `train_think_vetor_1b_grpo.ipynb` para o Google Colab ou Vast.ai.
2. Certifique-se de ativar o ambiente de GPU acelerada.
3. Execute a primeira célula para instalar as dependências de fine-tuning:
   ```bash
   pip install -q transformers peft bitsandbytes accelerate trl safetensors
   ```
4. Dispare a geração do dataset sintético conversacional:
   ```bash
   python3 generate_synthetic_conversations.py
   ```
5. Inicie o treinamento de alinhamento com a chamada do script mestre:
   ```bash
   python3 train_hybrid_1b.py \
       --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
       --epochs 3 \
       --switch_epoch 1 \
       --batch_size 2 \
       --lr 2e-5 \
       --out_dir "checkpoints/think_vetor_1b_hybrid_lora"
   ```
6. Execute a bateria de 50 testes de robustez conversacional e baixe o adaptador LoRA compilado no arquivo `.zip` resultante.
