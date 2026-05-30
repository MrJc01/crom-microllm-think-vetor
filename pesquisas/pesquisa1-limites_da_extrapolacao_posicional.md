# Pesquisa 1: Limites da Extrapolação Posicional em Aritmética Latente
**Data:** 29 de Maio de 2026  
**Investigadores:** MrJc01 & Antigravity  
**Objetivo:** Analisar a barreira dos 1% de acurácia em equações de 3 dígitos (fora de distribuição) mesmo após a adoção de Positional Encodings Senoidais e propor soluções de alinhamento.

---

## 1. O Fenômeno Observado

Após treinar o modelo corrigido (com Positional Encodings Senoidais) por 80 épocas no Google Colab, coletamos as seguintes métricas de acurácia:
* **Em distribuição (2 dígitos):** **`100.0%`** de acurácia exata (Exact Match).
* **Validação (Disjunta - 2 dígitos):** **`49.25%`**, mostrando forte generalização algorítmica para novos dados da mesma escala.
* **Fora de distribuição (3 dígitos / OOD):** **`1.0%`** no Think-Vetor e **`0.0%`** no Baseline.

Embora o modelo tenha aprendido perfeitamente a lógica de carry e adição para 2 dígitos, ele falhou quase que completamente ao se deparar com números de 3 dígitos (ex: $123+456=$).

---

## 2. Análise Teórica: Por que a Extrapolação Falhou?

A falha na extrapolação, mesmo usando senoides (que teoricamente estendem posições matemáticas), decorre de três fatores geométricos e de alinhamento:

### A. Deslocamento Absoluto de Operadores e Gatilhos
O modelo foi treinado exclusivamente com sequências onde a largura máxima do input era 6 (`"99+99="`). O token de gatilho de geração (`=`) estava sempre posicionado no índice 5 (ou menor, devido ao padding à esquerda).
Ao testar com 3 dígitos (ex: `"123+456="`), o input tem comprimento 8, posicionando o `=` no índice 7.
* **O Efeito:** A projeção de atenção do decodificador nunca viu a chave do atretor latente associada a um gatilho `=` no índice posicional 7. O decodificador simplesmente "não sabe o que fazer" nessa posição física, falhando em ativar as previsões corretas.

### B. Variação da Distância Relativa entre Dígitos Alinhados
Na soma, os dígitos precisam ser alinhados da direita para a esquerda (unidade com unidade, dezena com dezena, centena com centena).
* No treino (2 dígitos):
  * Em `"12+34="` (preenchido com padding à esquerda para 6 tokens): `[pad, 1, 2, +, 3, 4, =]`
  * O dígito da unidade de A (`2`, índice 2) e o de B (`4`, índice 5) estão separados por uma **distância de 3 tokens**.
* No teste OOD (3 dígitos):
  * Em `"123+456="` (8 tokens): `[1, 2, 3, +, 4, 5, 6, =]`
  * A unidade de A (`3`, índice 2) e o de B (`6`, índice 6) estão separados por uma **distância de 4 tokens**.
* **O Efeito:** Mesmo que a atenção use posições relativas, as cabeças de atenção aprenderam a buscar o dígito correspondente da soma a uma distância relativa fixa de 3 tokens. Ao mudar a distância para 4 tokens, a projeção de atenção falha em alinhar os dígitos correspondentes.

```
Treino (2D):   [1]  [2]  [+]  [3]  [4]  [=]
                └─── 3 tokens ───┘

Teste (3D):    [1]  [2]  [3]  [+]  [4]  [5]  [6]  [=]
                └───── 4 tokens ───────────┘
```

### C. Alinhamento de Padding à Esquerda
O padding à esquerda empurra os dígitos menores para a direita para manter o `=` na última posição. Isso altera a posição absoluta de inputs mais curtos de forma inconsistente, prejudicando a constância do aprendizado posicional do Transformer.

---

## 3. Soluções Propostas para Generalização OOD Aritmética

Para que o Think-Vetor extrapole para somas de 3, 4 ou mais dígitos, propomos as seguintes investigações empíricas:

### Solução 1: Alinhamento Posicional Invariante a Comprimento
Em vez de alinhar o texto pela esquerda, podemos alinhar os dígitos a partir dos operadores, de modo que a distância relativa entre as unidades de A e B e o operador `+` seja sempre **constante**, independentemente do número de dígitos.
* Exemplo de entrada: usar padding ou formatação onde a unidade esteja sempre a uma distância fixa do `+`.

### Solução 2: Treinamento com Comprimento Misto (Mixed-Length Curriculum)
O modelo falha em 3 dígitos porque nunca viu nada diferente de 2 dígitos.
* **Abordagem:** Treinar o modelo com uma distribuição mista: 70% de somas de 1 e 2 dígitos, e 30% de somas de 3 dígitos. Isso força a rede a aprender a invariância de escala (a regra geral do carry recursivo) em vez de uma regra geométrica fixa para 2 dezenas.

### Solução 3: Causal Decoder-Only com RoPE (Rotary Position Embeddings)
Migrar a arquitetura para Decoder-Only, onde o modelo processa toda a sequência de forma causal e usa **RoPE**. Como as rotações do RoPE operam diretamente sobre ângulos relativos de Query e Key, o modelo consegue lidar melhor com distâncias ligeiramente maiores se treinado com uma pequena variação de comprimentos.
