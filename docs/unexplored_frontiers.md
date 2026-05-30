# Fronteiras Inexploradas: Próximos Passos na Pesquisa do Think-Vetor

Com o sucesso do nosso protótipo in-distribution (100% de acurácia em soma de 2 dígitos), estabelecemos uma fundação sólida para o raciocínio contínuo no espaço latente. No entanto, ao analisarmos toda a documentação conceitual (`continuous_cot.md`, `recurrent_dynamics.md`, `hopfield_ebm.md` e `proposed_architecture.md`), identificamos vários "lados" e direções de pesquisa que ainda permanecem inexplorados e oferecem grande potencial científico.

Este documento cataloga essas fronteiras inexploradas, seus benefícios teóricos e como implementá-las.

---

## 1. Expansão de Sequência Causal (Estilo COCONUT)

### O que é?
No nosso design atual, os passos de ponderação ocorrem em um loop recorrente de tamanho fixo em relação à entrada do encoder, e os estados resultantes são passados via memória de atenção cruzada (*cross-attention*) para o decodificador.
A abordagem **COCONUT (Continuous Chain of Thought)** propõe algo diferente: **estender a sequência temporal inserindo os vetores latentes diretamente no contexto causal**.

```
[Pergunta] -> [z_1] -> [z_2] -> [z_3] -> [Geração da Resposta]
```

### Como Explorar?
1. **Remoção da Atenção Cruzada**: Unificar a arquitetura em um modelo puramente causal (Decoder-Only).
2. **Concatenação de Ativações**: Durante o pensamento, pegar o último *hidden state* $z_t \in \mathbb{R}^d$ e concatená-lo aos embeddings de entrada, permitindo que a atenção causal atenda a pensamentos anteriores de forma autoregressiva contínua.
3. **Vantagem**: Permite um número dinâmico e flexível de passos de pensamento que estendem o comprimento físico da "memória de trabalho" do modelo.

---

## 2. Generalização Extrapolada (RoPE & ALiBi)

### O que é?
Nossa avaliação mostrou que a acurácia para equações fora de distribuição (OOD) de 3 dígitos foi de apenas `1%`. Isso ocorre porque utilizamos **embeddings posicionais aprendíveis absolutos**, que não possuem capacidade de extrapolação matemática para sequências maiores que 32 tokens.

### Como Explorar?
1. **Substituição por RoPE (Rotary Position Embeddings)**: Substituir os embeddings posicionais absolutos por rotações complexas nas matrizes de Query e Key (como feito no LLaMA).
2. **Implementação de ALiBi (Attention with Linear Biases)**: Adicionar um viés linear negativo baseado na distância entre tokens diretamente na matriz de atenção.
3. **Vantagem**: Permitirá que um modelo treinado apenas com somas de 2 dígitos generalize organicamente para somas de 3, 4 ou mais dígitos na inferência, extrapolando as posições numéricas.

---

## 3. Destilação e Alinhamento Latente (Latent Distillation)

### O que é?
Atualmente, o modelo é treinado de ponta a ponta (*end-to-end*) sob a perda da resposta final. O espaço latente é livre para se organizar como quiser, o que pode levar a caminhos caóticos de otimização difíceis de convergir em problemas complexos (ex: multiplicação).
A **Destilação Latente** propõe guiar o espaço latente forçando os vetores $z_k$ a se alinharem com os embeddings de uma cadeia de pensamento textual real (passo a passo).

### Como Explorar?
1. **Dataset de CoT Textual**: Criar um dataset contendo o passo a passo matemático (ex: `"3+8=11, leva 1"`).
2. **Função de Perda de Cosseno**: Adicionar um termo de erro quadrático médio (MSE) ou similaridade de cosseno entre os estados latentes $z_k$ e os embeddings dos tokens do CoT textual correspondente:
   $$\mathcal{L}_{latent} = \sum_{k} \left(1 - \cos(z_k, \text{Embedding}(\text{Token}_k))\right)$$
3. **Vantagem**: Reduz drasticamente o espaço de busca de otimização, guiando o "pensamento latente" com estrutura lógica conhecida antes de remover os tokens textuais.

---

## 4. Aprendizado por Reforço no Espaço Contínuo (RL em Latent Space)

### O que é?
Para cenários onde não há um CoT textual de gabarito para destilação, o modelo precisa descobrir sozinho quais trajetórias contínuas geram respostas certas. Isso pode ser feito usando algoritmos de Reinforcement Learning adaptados para escolhas contínuas.

### Como Explorar?
1. **Framework GRPO (Group Relative Policy Optimization)**: Amostrar múltiplos vetores contínuos de perturbação no espaço latente.
2. **Recompensa por Exatidão**: Avaliar quais perturbações latentes resultaram em respostas corretas do decodificador e aplicar gradiente de política para mover as trajetórias naquela direção.
3. **Vantagem**: Permite a auto-descoberta de algoritmos matemáticos internos otimizados pelo próprio modelo, sem viés humano.

---

## 5. Regularização Contraintuitiva de Energia (Controle de Atratores Espúrios)

### O que é?
Em Modelos Baseados em Energia (EBMs) e Redes de Hopfield, o sistema pode sofrer com **atretores espúrios** — estados estáveis de energia mínima que representam alucinações matemáticas coerentes, mas incorretas.

### Como Explorar?
1. **Treinamento Contrastivo**: Minimizar a energia das trajetórias latentes que levam à resposta correta e, simultaneamente, maximizar a energia de caminhos que levam a predições incorretas.
2. **Entropia de Ativação**: Adicionar uma perda de regularização baseada na entropia das memórias ativas de Hopfield para garantir que a rede explore memórias diversas em vez de colapsar sempre no mesmo vale de atração.
3. **Vantagem**: Torna o processo de Langevin-Hopfield muito mais robusto contra alucinações.

---

## Resumo dos Próximos Passos Práticos

Para continuar a evolução deste protótipo, a seguinte ordem de experimentos é recomendada:

```
[Etapa 1: Extrapolação] ──> Instalar RoPE no modelo para tentar subir a acurácia OOD (3 dígitos).
          │
[Etapa 2: Causalidade]  ──> Testar o modo causal Decoder-Only (estilo COCONUT).
          │
[Etapa 3: Alinhamento]  ──> Inserir perda de Destilação Latente em tarefas mais complexas (ex: subtração).
```
