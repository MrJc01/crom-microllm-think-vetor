# Registro de Pesquisas e Aprendizados (Think-Vetor)

Este diretório armazena investigações empíricas, relatórios de erros estruturais resolvidos e descobertas científicas coletadas ao longo do desenvolvimento do modelo **Think-Vetor**.

## Lista de Pesquisas

* **[pesquisa0-invariancia_posicional_e_parada_aritmetica.md](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/pesquisas/pesquisa0-invariancia_posicional_e_parada_aritmetica.md)**: Investigação do colapso de acurácia em 0.00% no Google Colab. Explicação teórica sobre invariância por permutação (falta de positional encodings), o bug de mascaramento do stop token (EOS), a ilusão do overlap de dados local, e a solução de acurácia exata a 100%.
* **[pesquisa1-limites_da_extrapolacao_posicional.md](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/pesquisas/pesquisa1-limites_da_extrapolacao_posicional.md)**: Análise detalhada da barreira dos 1% de acurácia OOD em 3 dígitos. Mapeamento matemático sobre deslocamento absoluto de operadores, variação de distância relativa entre dígitos correspondentes e propostas de alinhamento e curriculum learning.
* **[pesquisa2-analise_experimentos_avancados.md](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/pesquisas/pesquisa2-analise_experimentos_avancados.md)**: Análise dos resultados práticos obtidos no Google Colab. Diagnóstico técnico do comportamento e baixa acurácia do Causal COCONUT (0.30%) devido a uma dessincronização de índices e repositório remoto desatualizado, e o sucesso notável do GRPO Contínuo com 73.70% de acurácia.
* **[pesquisa3-aprendizados_consolidados_e_direcoes_cientificas.md](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/pesquisas/pesquisa3-aprendizados_consolidados_e_direcoes_cientificas.md)**: O compêndio definitivo do projeto. Reúne toda a teoria de atratores contínuos de Hopfield, diagnósticos práticos de engenharia (como invariância posicional, mascaramento de EOS e alinhamento causal do COCONUT) e o roadmap de hibridização científica (SFT Latente + RL).
* **[pesquisa4-micro_llm_de_raciocinio_logico_e_early_stopping.md](file:///home/j/Área de trabalho/GitHub/crom-microllm-think-vetor/pesquisas/pesquisa4-micro_llm_de_raciocinio_logico_e_early_stopping.md)**: Documentação dos resultados consolidados da Micro-LLM de texto. Explora o sucesso notável de 100.00% de acurácia em apenas 3 épocas no dataset lógico e a melhora absoluta de +17.40% na validação disjunta da aritmética devido ao padding centralizado no operador.


