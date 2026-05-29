# Micro-LLM de Pensamento Latente (Think-Vetor)

Este repositório contém a pesquisa, o mapeamento conceitual e as propostas de arquitetura para a criação de um **Micro-LLM de Pensamento Latente (Think-Vetor)**. O objetivo principal é desenvolver um modelo de linguagem pequeno que execute seu processo de raciocínio (Chain of Thought - CoT) diretamente no espaço vetorial latente, em vez de gerar tokens de texto visíveis intermediários (como tags `<think>...</think>`).

Inspirado na neurobiologia, este projeto explora como replicar o estado pré-verbal, caótico e de alta energia que ocorre no cérebro humano antes de uma ideia ser colapsada em palavras estruturadas.

---

## A Filosofia: O Caos Pré-Verbal

Nos LLMs tradicionais (incluindo modelos de raciocínio como OpenAI o1 e DeepSeek-R1), o pensamento é simulado como um fluxo linear de texto:
$$\text{Palavra}_1 \to \text{Pensamento}_1 \to \text{Pensamento}_2 \to \dots \to \text{Resolução} \to \text{Resposta Final}$$

Esta abordagem apresenta limitações fundamentais:
1. **Ineficiência de Contexto**: A geração de centenas de tokens de raciocínio consome a janela de contexto de forma acelerada.
2. **Computação Discreta**: O modelo é forçado a tomar decisões discretas (escolher uma palavra) a cada passo de pensamento, impedindo a exploração de caminhos de raciocínio contínuos.
3. **Rigidez Temporal**: O pensamento é limitado à velocidade de decodificação auto-regressiva (token por token).

Em contrapartida, a **biologia do cérebro humano** opera de forma muito diferente:
* Antes da verbalização, redes neurais entram em um estado dinâmico caótico e contínuo.
* Conceitos abstratos se ativam simultaneamente no espaço latente.
* A dinâmica do sistema atrai esse estado caótico para um atretor estável (compreensão/conclusão).
* Apenas após essa estabilização, o lobo frontal traduz o estado latente em linguagem estruturada.

O projeto **Think-Vetor** visa modelar este loop latente através de abordagens matemáticas e computacionais modernas.

---

## Sumário das Possibilidades Exploradas

Para guiar o desenvolvimento do protótipo, dividimos o ecossistema de possibilidades em quatro documentos principais de engenharia:

### 1. [Cadeia de Pensamento Contínua (Continuous CoT)](file:///home/j/%C3%81rea%20de%20trabalho/GitHub/crom-microllm-think-vetor/docs/continuous_cot.md)
Explora o framework onde o modelo é treinado para prever os próximos estados ocultos (*hidden states*) no espaço latente, pulando a camada de projeção linear (cabecário de linguagem) durante o "pensamento".
* **Foco**: Otimização matemática, alinhamento de embeddings e funções de perda baseadas em similaridade de cosseno/MSE.

### 2. [Transformers Recorrentes e Dinâmica Caótica](file:///home/j/%C3%81rea%20de%20trabalho/GitHub/crom-microllm-think-vetor/docs/recurrent_dynamics.md)
Aborda a implementação de Universal Transformers onde os vetores latentes circulam recursivamente pelas mesmas camadas por $N$ passos, com a inclusão de ruído estocástico (caos) e mecanismos de atenção baseados em entropia.
* **Foco**: Tempo de Computação Adaptativo (ACT), PonderNet e mecanismos para controle de profundidade de raciocínio dinâmico.

### 3. [Redes de Hopfield Modernas e Modelos Baseados em Energia](file:///home/j/%C3%81rea%20de%20trabalho/GitHub/crom-microllm-think-vetor/docs/hopfield_ebm.md)
Investiga a fusão de Transformers com a neurobiologia clássica dos sistemas dinâmicos. Descreve como injetar uma Rede de Hopfield Moderna ou camada baseada em energia para que a entrada gere um estado de alta energia (caos) que decai progressivamente para um atretor de equilíbrio.
* **Foco**: Equações de energia, dinâmica de Langevin para exploração estocástica e convergência de atretores.

### 4. [Arquitetura Híbrida e Protótipo PyTorch](file:///home/j/%C3%81rea%20de%20trabalho/GitHub/crom-microllm-think-vetor/docs/proposed_architecture.md)
Propõe um design unificado para o Micro-LLM Think-Vetor, acompanhado de um código completo e legível em PyTorch que ilustra como implementar o loop de recorrência latente com decaimento de caos e parada dinâmica.

---

## Tabela Comparativa de Abordagens

| Característica | Continuous CoT (COCONUT) | Universal Transformers + ACT | Modelos de Energia / Hopfield |
| :--- | :--- | :--- | :--- |
| **Representação do Pensamento** | Sequência de vetores latentes | Loop iterativo em camadas fixas | Minimização de energia livre |
| **Mecanismo de Parada** | Número fixo de passos ou classificador | Gatilho dinâmico (Halting Probability) | Estabilização em Atretor ($\nabla E \to 0$) |
| **Origem do "Caos"** | Variabilidade no espaço de embeddings | Ruído estocástico e entropia de atenção | Dinâmica de Langevin (Simulated Annealing) |
| **Complexidade de Treinamento** | Média (requer alinhamento latente) | Média-Alta (gradientes via ACT) | Alta (treinamento contrastivo ou EBM) |
| **Custo Computacional** | Linear com passos de pensamento | Variável (alocação dinâmica de computo) | Variável (passos de descida de gradiente) |

---

> [!NOTE]
> Este projeto está em fase de design conceitual avançado e modelagem. As especificações nos documentos a seguir servem como blueprint de engenharia para o desenvolvimento do protótipo.
