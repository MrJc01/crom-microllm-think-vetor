# Cadeia de Pensamento Contínua (Continuous Chain of Thought)

A **Cadeia de Pensamento Contínua (Continuous CoT)** é um paradigma onde o raciocínio intermediário de um modelo de linguagem ocorre inteiramente no espaço de embeddings $\mathbb{R}^d$, ignorando a discretização em tokens de texto até a fase de resposta final. Este conceito baseia-se em trabalhos recentes de pesquisa, como o framework **COCONUT (Continuous Chain of Thought)**.

---

## O Fluxo de Dados Latente vs. Tradicional

Para entender a Continuous CoT, podemos comparar o fluxo de processamento de um modelo autoregressivo padrão com o modelo contínuo:

```mermaid
graph TD
    subgraph LLM Tradicional (CoT em Texto)
        A[Pergunta: 'Quanto é 2+2?'] --> B[Tokenizador]
        B --> C[Camadas do Transformer]
        C --> D[lm_head / Softmax]
        D --> E["Token: '<think>'"]
        E --> F["Próximo passo de inferência..."]
    end

    subgraph Continuous CoT (Pensamento Latente)
        A2[Pergunta: 'Quanto é 2+2?'] --> B2[Tokenizador]
        B2 --> C2[Camadas do Transformer]
        C2 --> D2{Modo?}
        D2 -- Pensamento Latente --> E2[Ignora lm_head / Extrai Hidden State z_t]
        E2 --> F2[Injeta z_t diretamente no Contexto de Entrada]
        F2 --> C2
        D2 -- Resposta Final --> G2[lm_head / Softmax]
        G2 --> H2["Token: '4'"]
    end
```

No modelo tradicional, cada token de pensamento gerado precisa ser amostrado do vocabulário (decisão discreta de tamanho $V$, geralmente $\approx 32k\text{-}128k$), convertido de volta em embedding e reprocessado. 

Na Continuous CoT, o vetor de estado oculto da última camada, $z_t \in \mathbb{R}^d$, é inserido diretamente na sequência de entrada como se fosse o embedding de um token que o modelo "visualizou", pulando a projeção linear da palavra.

---

## Formulação Matemática

Considere um Transformer com dimensão oculta $d$. A entrada consiste em embeddings $H^{(0)} = [h_1, h_2, \dots, h_n]$, onde $h_i \in \mathbb{R}^d$.

Durante a geração tradicional do próximo token no passo $t$:
$$h_{next} = \text{Transformer}(H^{(t-1)})$$
$$P(y_t) = \text{Softmax}(W_{vocab} \cdot h_{next})$$
$$y_t = \text{sample}(P(y_t))$$
$$h_{input\_next} = \text{Embedding}(y_t)$$

Durante a **Continuous CoT**, para $k$ passos de pensamento latente:
1. O modelo processa a sequência atual e gera o estado oculto final $z_t \in \mathbb{R}^d$.
2. Ignoramos a projeção de vocabulário $W_{vocab}$.
3. O próprio vetor $z_t$ é concatenado à sequência de embeddings:
   $$H^{(t)} = [H^{(t-1)}, z_t]$$
4. Este processo se repete por $k$ passos de pensamento latente.
5. No passo $t+k$, o modelo reativa o cabeçalho de linguagem para produzir a resposta textual:
   $$P(y_{final}) = \text{Softmax}(W_{vocab} \cdot \text{Transformer}(H^{(t+k-1)}))$$

---

## Como Treinar o Espaço Latente?

O maior desafio da Continuous CoT é que não há um "gabarito" natural de quais vetores no espaço latente representam um bom pensamento. Existem três métodos principais para contornar isso durante o treinamento:

### 1. Destilação Latente (Teacher-Forcing com Alinhamento)
Se tivermos um dataset que contém raciocínio textual (ex: perguntas com tags `<think> raciocínio </think> resposta`), podemos treinar o modelo a mapear os seus pensamentos latentes diretamente para os embeddings que o modelo geraria se estivesse escrevendo o texto de raciocínio.

A função de perda para a fase latente é uma combinação de erro quadrático médio (MSE) ou distância de cosseno:

$$\mathcal{L}_{latent} = \sum_{j=1}^{k} \left( 1 - \cos(z_j, \text{Target\_Embedding}_j) \right) + \lambda \| z_j - \text{Target\_Embedding}_j \|^2$$

### 2. Gradientes de Política (Reinforcement Learning)
Podemos treinar o modelo usando Aprendizado por Reforço (ex: **GRPO** ou **PPO**).
* **Ação**: A escolha de vetores contínuos $z_j \in \mathbb{R}^d$.
* **Recompensa**: Avaliada apenas no final, baseada na exatidão da resposta textual final gerada no passo $t+k$.
* **Otimização**: O modelo ajusta seus pesos para direcionar os embeddings intermediários para direções latentes que maximizam a probabilidade de colapsar na resposta correta.

---

## Protótipo de Algoritmo em PyTorch

Abaixo está um esboço de como interceptar os *hidden states* de um modelo e utilizá-los recursivamente para realizar passos de Continuous CoT:

```python
import torch
import torch.nn as nn

class ContinuousCoTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer Decoder para processar a sequência
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cabeçalho de projeção linear para mapear de volta ao vocabulário
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, input_ids, num_thought_steps=5):
        # 1. Obter embeddings iniciais para os tokens de entrada
        # input_embeddings: (batch_size, sequence_length, d_model)
        x = self.embedding(input_ids)
        
        # 2. Executar o Loop de Pensamento Latente (Continuous CoT)
        for step in range(num_thought_steps):
            # Passa a sequência atual pelo Transformer
            hidden_states = self.transformer(x)
            
            # Extrai o último vetor oculto correspondente ao último passo temporal
            # last_hidden: (batch_size, 1, d_model)
            last_hidden = hidden_states[:, -1:, :]
            
            # Concatena o próprio vetor latente na entrada (sem passar pelo lm_head)
            # Isso aumenta a sequência temporal no espaço latente
            x = torch.cat([x, last_hidden], dim=1)
            
        # 3. Gerar a Resposta Final
        # Após pensar de forma latente, processamos o estado final para prever a resposta textual
        final_hidden = self.transformer(x)
        logits = self.lm_head(final_hidden[:, -1, :])
        
        return logits

# Exemplo de uso:
# model = ContinuousCoTTransformer(vocab_size=32000, d_model=512, nhead=8, num_layers=6)
# input_tokens = torch.randint(0, 32000, (2, 10)) # batch_size=2, seq_len=10
# output_logits = model(input_tokens, num_thought_steps=4)
```

---

> [!TIP]
> A principal vantagem dessa abordagem é a **velocidade**. A computação latente é realizada em uma única passada de matriz, sem a necessidade de amostragem autoregressiva e decodificação de múltiplos tokens de texto na janela de contexto.
