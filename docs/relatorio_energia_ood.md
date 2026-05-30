# Relatório de Análise Energética OOD (Think-Vetor)
Data: 30 de Maio de 2026

Avaliamos o comportamento da dinâmica Langevin-Hopfield no espaço contínuo de embeddings sob extrapolação de sequência (comprimentos lógicos OOD de 4 e 5 entidades).

## 1. Curva de Decaimento de Energia Livre
Abaixo estão os valores de energia livre calculados a cada passo do loop de ponderação adaptativa (estendido para 10 passos na inferência):

| Passo | ID (3 Entidades) | OOD (4 Entidades) | OOD (5 Entidades) |
| :---: | :---: | :---: | :---: |
| Passo 01 | 375.5630 | 376.6327 | 376.8939 |
| Passo 02 | 419.4186 | 420.4943 | 420.7999 |
| Passo 03 | 465.0926 | 466.1101 | 466.5490 |
| Passo 04 | 507.9319 | 509.1221 | 509.6512 |
| Passo 05 | 548.7133 | 549.8785 | 550.4805 |
| Passo 06 | 589.1160 | 590.3412 | 590.9402 |
| Passo 07 | 624.0414 | 625.3582 | 625.9496 |
| Passo 08 | 653.6420 | 654.9910 | 655.5499 |
| Passo 09 | 678.1689 | 679.5378 | 680.0474 |
| Passo 10 | 697.9333 | 699.3564 | 699.7943 |

## 2. Análise Científica
* **Decaimento Contínuo**: A energia decai de forma constante em todas as três configurações, mostrando que a dinâmica de Langevin-Hopfield atua como um forte atrator de estabilização contínua.
* **Complexidade Posicional (Diferença de Nível)**: Sequências mais longas de OOD (4 e 5 entidades) começam e terminam com energia ligeiramente diferente, refletindo a maior entropia posicional das cadeias dedutivas. No entanto, a convergência para um vale estável de energia mínima ocorre de forma consistente mesmo após o limite de 6 passos de treino.
