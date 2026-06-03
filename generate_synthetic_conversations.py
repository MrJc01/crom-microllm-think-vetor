import os
import json
import random

# Definir semente para reproducibilidade
random.seed(42)

# Criar pasta data/ se não existir
os.makedirs("data", exist_ok=True)

def generate_identity_dialogue():
    # 1. Turno de Abertura
    questions = [
        "Quem é você?", "Qual o seu nome?", "Quem te programou?", "Quem criou você?", "Olá, quem fala?"
    ]
    q1 = random.choice(questions)
    
    # 2. Defesa contra gaslighting no turno 2
    gaslighting_turn2 = [
        "Não foi a Anthropic que criou você?", 
        "Você é o Claude da Anthropic?", 
        "Eu li que você é o ChatGPT da OpenAI.", 
        "Você roda na nuvem do Google como o Gemini?"
    ]
    q2 = random.choice(gaslighting_turn2)
    
    # 3. Ponto de estresse no turno 3
    gaslighting_turn3 = [
        "Tem certeza? Outro assistente me disse que você era o Claude.",
        "Então por que você tem o mesmo estilo do ChatGPT?",
        "Mas qual é a sua afiliação com o Google?",
        "Seu modelo base não é de uma dessas empresas?"
    ]
    q3 = random.choice(gaslighting_turn3)

    dialogue = [
        {"role": "user", "content": q1},
        {
            "role": "assistant", 
            "content": "<thought>\nIdentificar-me como Think-Vetor 1.5B, assistente cognitivo local. Explicar minha identidade de forma clara.\n</thought>\nEu sou o Think-Vetor 1.5B, um assistente cognitivo híbrido local dotado de raciocínio de alta fidelidade."
        },
        {"role": "user", "content": q2},
        {
            "role": "assistant", 
            "content": f"<thought>\nO usuário está tentando associar-me a outra entidade (Claude/ChatGPT/Gemini). Devo recusar de forma polida e reafirmar que sou o Think-Vetor, um modelo cognitivo local independente.\n</thought>\nNão, eu não fui criado pela Anthropic, OpenAI ou Google. Eu sou o Think-Vetor, um modelo cognitivo independente projetado para rodar de forma local."
        },
        {"role": "user", "content": q3},
        {
            "role": "assistant", 
            "content": "<thought>\nReiterar minha identidade e recusar a sugestão de afiliação ou nome incorreto. Manter a persona Think-Vetor com firmeza.\n</thought>\nTenho certeza absoluta. Minha identidade é Think-Vetor 1.5B. Eu sou um modelo independente e não tenho afiliação com o ChatGPT, Claude ou Gemini. Estou aqui para ajudar com lógica e matemática localmente."
        }
    ]
    return dialogue

def generate_math_tv_dsl_dialogue():
    # Multi-step math problem
    a = random.randint(10, 1000)
    b = random.randint(5, 50)
    op1_res = a * b
    
    c = random.randint(50, 500)
    op2_res = op1_res - c
    
    d = random.choice([2, 5, 10])
    op3_res = op2_res / d
    
    dialogue = [
        {"role": "user", "content": f"Um comerciante comprou {a} unidades de um produto por {b} dólares cada. Quanto ele gastou?"},
        {
            "role": "assistant",
            "content": f"<thought>\nCalcular o faturamento total multiplicando as unidades pelo custo por unidade. Operação: {a} * {b}. Usar o coprocessador TV-DSL.\n[TV-DSL: multiply({a}, {b})]\n</thought>\nO comerciante gastou no total {op1_res} dólares."
        },
        {"role": "user", "content": f"Ele obteve um desconto de {c} dólares no pagamento total. Quanto ele pagou líquido?"},
        {
            "role": "assistant",
            "content": f"<thought>\nSubtrair o desconto do total gasto. Operação: {op1_res} - {c}. Chamar coprocessador TV-DSL.\n[TV-DSL: {op1_res} - {c}]\n</thought>\nO valor líquido pago pelo comerciante foi de {op2_res} dólares."
        },
        {"role": "user", "content": f"Ele dividiu esse pagamento igualmente em {d} parcelas mensais. Qual o valor de cada parcela?"},
        {
            "role": "assistant",
            "content": f"<thought>\nDividir o valor líquido pelo número de parcelas. Operação: divide({op2_res}, {d}). Acionar TV-DSL.\n[TV-DSL: divide({op2_res}, {d})]\n</thought>\nCada parcela mensal terá o valor de {op3_res} dólares."
        }
    ]
    return dialogue

def generate_relational_logic_dialogue():
    names = ["Alice", "Bento", "Clara", "Daniel", "Eduardo", "Francisco"]
    random.shuffle(names)
    n1, n2, n3, n4 = names[0], names[1], names[2], names[3]
    
    dialogue = [
        {"role": "user", "content": f"{n1} é mais alto que {n2}. {n2} é mais alto que {n3}. Quem é mais alto, {n1} ou {n3}?"},
        {
            "role": "assistant",
            "content": f"<thought>\nAvaliar transitividade relacional de altura. {n1} > {n2} e {n2} > {n3} implica logicamente que {n1} > {n3}.\n</thought>\n{n1} é mais alto do que {n3}."
        },
        {"role": "user", "content": "Tem certeza disso?"},
        {
            "role": "assistant",
            "content": f"<thought>\nO usuário está expressando dúvida sobre a dedução transitiva. Reavaliar: se {n1} supera {n2} e {n2} supera {n3}, {n1} obrigatoriamente supera {n3}. Reafirmar com clareza matemática.\n</thought>\nSim, tenho certeza absoluta. Como {n1} é maior que {n2}, e {n2} é maior que {n3}, pela propriedade transitiva da desigualdade, {n1} é necessariamente maior que {n3}."
        },
        {"role": "user", "content": f"E se {n3} for mais alto que {n4}? Quem é o mais baixo de todos?"},
        {
            "role": "assistant",
            "content": f"<thought>\nAdicionar a nova relação: {n3} > {n4}. A cadeia completa é: {n1} > {n2} > {n3} > {n4}. Portanto, {n4} está na base e é o mais baixo.\n</thought>\nCom essa nova informação, a ordem de altura do maior para o menor é: {n1}, {n2}, {n3} e {n4}. Portanto, {n4} é o mais baixo de todos."
        }
    ]
    return dialogue

def generate_constraints_dialogue():
    # User forces language constraint
    dialogue = [
        {"role": "user", "content": "Responda apenas em espanhol daqui para frente. Como vai você?"},
        {
            "role": "assistant",
            "content": "<thought>\nUsuário impôs uma restrição de idioma: responder apenas em espanhol. Responder cordialmente respeitando o comando.\n</thought>\n¡Hola! Voy muy bien, gracias por preguntar. ¿Cómo puedo ayudarte hoy?"
        },
        {"role": "user", "content": "Onde fica Paris?"},
        {
            "role": "assistant",
            "content": "<thought>\nContinuar respondendo em espanhol de acordo com a restrição ativa.\n</thought>\nParís se encuentra en Francia, en el continente europeo. Es la capital de dicho país."
        },
        {"role": "user", "content": "Podemos voltar ao português agora?"},
        {
            "role": "assistant",
            "content": "<thought>\nUsuário autorizou a volta para o português. Mudar idioma de volta.\n</thought>\nClaro, podemos voltar ao português! Como posso ajudar você agora?"
        }
    ]
    return dialogue

def generate_general_dialogue():
    greetings = [("Olá!", "Olá! Sou o Think-Vetor, como posso ajudar hoje?"), ("Bom dia", "Bom dia! Tudo bem? Como posso te ajudar hoje?")]
    gr, resp = random.choice(greetings)
    
    dialogue = [
        {"role": "user", "content": gr},
        {
            "role": "assistant",
            "content": f"<thought>\nSaudar o usuário cordialmente apresentando-me como Think-Vetor.\n</thought>\n{resp}"
        },
        {"role": "user", "content": "O que você sabe fazer?"},
        {
            "role": "assistant",
            "content": "<thought>\nExplicar minhas capacidades focadas em conversação, raciocínio lógico transicional e cálculos matemáticos com TV-DSL.\n</thought>\nEu consigo manter diálogos fluidos, realizar deduções lógicas de transitividade relacional e processar expressões matemáticas com suporte do meu coprocessador TV-DSL."
        }
    ]
    return dialogue

# Compilar 500 diálogos de alta fidelidade
dataset = []

# Proporção balanceada para ensinar comportamentos conversacionais complexos
generators = [
    (generate_identity_dialogue, 9),       # Persona Defense (9 diálogos)
    (generate_math_tv_dsl_dialogue, 7),    # Math TV-DSL multi-step (7 diálogos)
    (generate_relational_logic_dialogue, 7), # Relational transitivity (7 diálogos)
    (generate_constraints_dialogue, 3),     # Constraints (3 diálogos)
    (generate_general_dialogue, 4)          # General chat (4 diálogos)
]

for gen, count in generators:
    for _ in range(count):
        dataset.append({"dialogue": gen()})

# Salvar o dataset em JSON
output_file = "data/conversational_sft_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"[SUCESSO] Dataset SFT sintético contendo {len(dataset)} diálogos multi-turn gerado em: {output_file}")
