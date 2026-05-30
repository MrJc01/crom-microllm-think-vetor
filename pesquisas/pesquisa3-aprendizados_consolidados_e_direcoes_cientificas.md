# Pesquisa 3: Compêndio de Aprendizados Consolidados, Diagnósticos e Direções Científicas (Think-Vetor)
**Data:** 30 de Maio de 2026  
**Investigadores:** MrJc01 & Antigravity  
**Objetivo:** Consolidar em um único documento de referência todas as descobertas de engenharia, bugs estruturais resolvidos, intuições geométricas e diretrizes científicas coletadas ao longo do projeto.

---

## 1. Fundamentos Geométricos do Raciocínio Contínuo

O projeto **Think-Vetor** baseia-se na premissa de que o raciocínio de grandes modelos de linguagem não precisa necessariamente ser discretizado em tokens de texto (como em *Chain-of-Thought* tradicional). Em vez disso, ele pode ocorrer em um espaço latente contínuo, onde o modelo executa passos de computação interna refinando um vetor de pensamento intermediário antes de projetar a resposta final.

### A. Dinâmica de Langevin e Atratores Hopfield
* **Conceito:** O modelo utiliza um bloco customizado de Langevin (`LangevinHopfieldBlock`) que atua como uma rede de memórias associativas Hopfield contínuas. A cada passo de reflexão, o pensamento latente $z_t$ é atualizado seguindo o gradiente de uma função de energia livre $E(z)$ mais uma perturbação estocástica controlada.
* **O que aprendemos:** A dinâmica de Langevin força o vetor de pensamento a convergir para "poços de atração" geométricos estáveis que correspondem a conceitos lógicos estruturados (por exemplo, "carry ativo"). Nas épocas finais de treinamento correto, a similaridade de cosseno entre estados de pensamentos consecutivos alcançou **`0.9955`**, provando que o espaço latente de fato converge para pontos fixos de atração matemática estáveis.

---

## 2. Grandes Lições de Engenharia e Arquitetura

Ao longo do desenvolvimento, diagnosticamos e resolvemos falhas sutis, mas fatais, que costumam passar despercebidas na criação de micro-LLMs de raciocínio:

### A. Transformers são "Bag-of-Words" por Padrão
* **O Bug:** Inicialmente, o modelo treinou com sucesso (loss $\approx 0.006$), mas sua acurácia de validação ficou travada em **0.00%**.
* **O Diagnóstico:** Sem embeddings de posição, a auto-atenção é totalmente simétrica a permutações de ordem. Para o Transformer, a string `"6+3="` e `"3+6="` eram idênticas. O modelo decorava as equações do treino devido ao overlap (memória associativa direta), mas não conseguia generalizar para novas somas.
* **A Solução:** A introdução de Positional Encodings Senoidais e de **Rotary Position Embeddings (RoPE)** restabeleceu imediatamente a noção de ordem física, permitindo que a rede diferenciasse as dezenas das unidades.

### B. O Perigo de Mascarar o Token EOS (Stop Token)
* **O Bug:** Mesmo com posição, o modelo continuava zerando a acurácia de validação porque gerava dígitos infinitos (ex: `"1095"` em vez de `"109 "`).
* **O Diagnóstico:** A máscara de perda do decodificador ignorava o caractere de espaço (padding) após a resposta numérica. Como consequência, a rede nunca era penalizada por não aprender a parar a geração.
* **A Solução:** Incluir o primeiro token de padding na máscara de perda (`loss mask`) treinou ativamente o modelo a emitir o stop token na hora certa, permitindo que a inferência autoregressiva decodifique respostas limpas e atinja acurácia perfeita.

### C. Alinhamento Temporal em Modelos Causais (Decoder-Only)
* **O Bug:** O Causal COCONUT obteve acurácia nula (0.30%) no início do treino no Colab.
* **O Diagnóstico:** Em modelos causais baseados em COCONUT, os pensamentos contínuos intermediários $z_1, z_2, \dots, z_K$ são injetados na sequência de entrada. O modelo realiza a primeira previsão de saída ($Y_0$) no último token de pensamento ($z_K$), mas as previsões subsequentes ($Y_t$) dependem do token de gatilho SOS (`=`) e de $Y_{t-1}$.
* **A Solução:** O loop de decodificação foi reestruturado (commit `b019697`) para separar a predição inicial ($Y_0$, tirada de $z_K$) da geração autoregressiva posterior (iniciada pelo acoplamento de $z_K$ com `=`). Essa correção cirúrgica fez a acurácia do COCONUT saltar para **84.15%**.

---

## 3. Avaliação Científica das 5 Fronteiras Avançadas

A execução integrada dos experimentos nos permitiu mapear os prós e contras de cada metodologia:

### 1. Destilação e SFT Latente (DISTILL)
* **Status:** Vencedor absoluto em estabilidade e velocidade de convergência (**100% de Acurácia** estável da Época 32 até 80).
* **Conclusão:** Forçar o alinhamento latente direto dos passos contínuos com representações de um raciocínio lógico estruturado (Chain-of-Thought) ensina a geometria correta de maneira extremamente rápida. É o melhor ponto de partida para qualquer modelo de raciocínio.

### 2. Regularização Contrastiva de Energia (EBM)
* **Status:** Excelente generalização lógica (**92.50% de Acurácia**).
* **Conclusão:** Minimizar a energia livre de Hopfield de trajetórias de pensamento corretas e maximizar a de incorretas cria barreiras geométricas que impedem a rede de desviar para pensamentos latentes sem sentido. Útil para refinar e "limpar" o ruído no raciocínio.

### 3. Aprendizado por Reforço no Espaço Contínuo (GRPO)
* **Status:** Generalização forte baseada em recompensa pura (**73.70% de Acurácia**).
* **Conclusão:** Permite calibrar a política de reflexão (parando em média com **2.94 passos**) sem precisar de dados de CoT textual rotulados. A necessidade de um termo híbrido de SFT de $0.5$ se mostrou crucial para o "cold start" (inicialização fria), evitando gradientes nulos no começo do treino de RL.

### 4. Causal COCONUT
* **Status:** Eficaz em decodificadores causalmente consistentes (**84.15% de Acurácia**).
* **Conclusão:** Unificar codificação, reflexão e decodificação na mesma fita causal é altamente eficiente, mas o uso de passos de pensamento fixos ($max\_thoughts=4$) limita a flexibilidade em tarefas de complexidade variável.

### 5. Extrapolação Posicional (Limites do OOD)
* **Status:** Barreira dos 1% em 3 dígitos.
* **Conclusão:** A extrapolação falha principalmente devido a deslocamentos absolutos de operadores em comprimentos maiores de string e à variação da distância relativa entre dígitos alinhados (2 dígitos versus 3 dígitos).

---

## 4. Diretrizes Científicas e Próximos Passos

Para as futuras iterações do Think-Vetor, o roadmap científico recomendado consiste em:

1. **Hibridização (Treinamento em Duas Fases):**
   * *Fase 1 (SFT Latente):* Treinar o modelo via **Destilação Latente** por 30 épocas para estruturar rapidamente a representação geométrica dos carries.
   * *Fase 2 (RL/GRPO):* Ajustar o modelo com **GRPO** para otimizar diretamente a recompensa de acerto de resposta exata e calibrar o número de passos de parada ideais para inputs de qualquer tamanho.
2. **Decaimento Dinâmico de Learning Rate no RL/EBM:**
   * Implementar um scheduler de taxa de aprendizado cosseno (cosine decay) nas épocas finais para amortecer as oscilações observadas no GRPO e no EBM (evitando as quedas de desempenho nas últimas 15 épocas).
3. **Invariância de Operadores com Padding Centralizado:**
   * Reformular a entrada do dataset para centralizar o alinhamento das unidades de soma em relação ao caractere `+`, tornando a distância geométrica das cabeças de atenção constante, independentemente do comprimento do dígito (resolvendo o problema de extrapolação OOD de 3 dígitos ou mais).
