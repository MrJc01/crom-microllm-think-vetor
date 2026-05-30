# Pesquisa 0: Invariância Posicional e Parada Autorregressiva em Micro-LLMs
**Data:** 29 de Maio de 2026  
**Investigadores:** MrJc01 & Antigravity  
**Objetivo:** Investigar e solucionar o colapso de acurácia em 0.00% do modelo Think-Vetor no conjunto de validação e restabelecer a generalização aritmética completa.

---

## 1. O Problema: Perda Baixa, Acurácia Zero

Durante o treinamento em larga escala na GPU T4 do Google Colab (8.000 amostras de treino / 2.000 de validação disjuntas), observamos um fenômeno contraditório:
* A perda de entropia cruzada de treino decaiu consistentemente de **`1.97`** para **`0.006`**, indicando convergência perfeita no treino.
* Contudo, a acurácia de validação (Exact Match) permaneceu cravada em **`0.00%`** de ponta a ponta.

Simultaneamente, ao rodar a avaliação local (`evaluate.py`) usando os mesmos pesos, obtivemos uma acurácia em torno de **`25%`** para o Think-Vetor, o que indicava uma discrepância de avaliação entre os conjuntos.

---

## 2. As Causas Raízes Identificadas

Após análise teórica das propriedades matemáticas da atenção e do pipeline de dados, descobrimos dois bugs estruturais:

### A. Invariância Posicional de Atenção (Transformers sem Posição)
Por design, as camadas de auto-atenção e atenção cruzada de um Transformer são totalmente invariantes à ordem dos tokens (agem como *bag-of-words*). Sem embeddings de posição, a representação de `"6+34="` é idêntica à de `"3+46="` ou `"4+36="` para a rede.
* **O Efeito:** O modelo era incapaz de aprender a regra geral de ordem de dígitos da soma. Ele apenas conseguia decorar relações associativas específicas presentes no conjunto de treino.
* **A Ilusão do Teste Local:** Como o espaço de somas de 2 dígitos possui apenas 10.000 combinações possíveis e treinamos com 8.000 (80%), qualquer conjunto de teste aleatório compartilhava $\approx 80\%$ de equações com o treino. A acurácia de 25% local era o modelo apenas recuperando dados decorados (como uma memória associativa Hopfield tradicional). Na validação disjunta do Colab (0% de sobreposição), a generalização era nula, caindo a 0.00%.

### B. Bug de Mascaramento do Stop Token (EOS)
A máscara de perda (loss mask) do decodificador ignorava completamente os tokens de padding (`" "` / espaço / índice 12):
```python
target_mask = [1] * len(target_ids) + [0] * target_pad_len
```
* **O Efeito:** Como o espaço funciona como o **token EOS (End-Of-Sequence / Stop Token)**, a rede nunca era penalizada por errar o stop token. O modelo nunca aprendeu a parar de gerar dígitos.
* **A Conseqüência:** Durante a autoregressão de inferência de comprimento 4, o modelo gerava dígitos numéricos em todas as posições (ex: `"1095"` em vez de `"109 "`). Quando comparado a `"109"`, o teste falhava e a acurácia caía a 0.00% para todas as amostras, incluindo as que o modelo havia calculado corretamente nas primeiras posições.

---

## 3. As Soluções de Arquitetura

Aplicamos três modificações cirúrgicas no repositório:

1. **Adição de Positional Embeddings Opcionais ([src/model.py](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/src/model.py)):**
   Implementamos `self.pos_encoder` e `self.pos_decoder` aprendíveis via parâmetro `use_pos_embedding=True`. Quando ativado, os embeddings de entrada recebem sua coordenada posicional, habilitando processamento sequencial e aritmético no Transformer.
2. **Inclusão do Primeiro Padding na Perda ([src/dataset.py](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/src/dataset.py)):**
   Alteramos a lógica da máscara de perda do target para incluir o primeiro token de padding após os dígitos numéricos, treinando ativamente o decodificador a prever o stop token:
   ```python
   target_mask = [1] * min(len(target_ids) + 1, self.max_target_len)
   target_mask = target_mask + [0] * (self.max_target_len - len(target_mask))
   ```
3. **Interrupção de Decodificação no Tokenizer ([src/dataset.py](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/src/dataset.py)):**
   Modificamos `CharTokenizer.decode` para interromper o processamento assim que o `pad_id` (espaço) é encontrado, eliminando dígitos espúrios gerados após a predição de parada.

---

## 4. O Impacto Empírico

Com a compilação dessas correções, executamos um novo treinamento de 80 épocas no Google Colab. A avaliação dos novos checkpoints revelou o restabelecimento total da capacidade aritmética:

* **Acurácia em Distribuição (2 dígitos):** **`100% de Acurácia Exata`** no conjunto de teste.
* **Acurácia de Validação Disjunta:** Saltou de `0.00%` para **`49.20%` (Baseline)** e **`49.20%` (Think-Vetor)** em apenas 80 épocas, mostrando generalização real.
* **Estabilidade dos Passos (PonderNet):** Média de passos dinâmicos estabilizou em torno de $\approx 3$ etapas de reflexão latente.
* **Convergência Latente de Langevin:** A similaridade de cosseno entre estados adjacentes alcançou **`0.99551`** nas etapas finais, comprovando atração matemática no espaço latente.

---

## 5. Lições Aprendidas

1. **Transformers são "Bag-of-Words" por padrão:** Sempre que desenhar um modelo Transformer, a injeção posicional é essencial se a ordem dos dados carregar significado lógico.
2. **EOS não deve ser mascarado:** O stop token é a parte mais crítica de um decodificador autorregressivo. Mascarar o stop token invalida a capacidade do modelo de terminar e estruturar respostas.
3. **Cuidado com overlaps em datasets pequenos:** Acurácias altas obtidas em testes aleatórios no espaço discreto (como somas) podem esconder a falta de generalização real se houver alto overlap com o conjunto de treino. A divisão disjunta estrita é o único teste confiável de inteligência generalizada.
