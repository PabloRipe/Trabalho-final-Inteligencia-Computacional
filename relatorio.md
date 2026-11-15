# Relatorio curto: classificacao de inadimplencia com apoio de LLM

## 1. Introducao e base de dados

O trabalho construiu um pipeline completo de aprendizado supervisionado para prever
inadimplencia (`mau=1`) a partir de caracteristicas pessoais, bancarias e de renda. A base
publica Credit Scoring do Kaggle tem 750.000 linhas e 16 colunas, com 16.650 clientes
identificados por `index`, observados em 15 meses (janeiro de 2015 a marco de 2016). A taxa
global de inadimplencia e 7,82%, ou seja, o baseline que sempre responde "bom pagador"
acerta 92,18% dos casos. `tempo_emprego` tem 16,79% de valores ausentes e nao ha
duplicatas exatas.

A analise exploratoria revelou dois fatos que mudaram o desenho do projeto. Primeiro, o
alvo varia muito ao longo do tempo: 12,11% em janeiro de 2015, 2,72% em julho de 2015 e
16,82% em marco de 2016. Segundo, a renda mediana mensal acompanha esse movimento, de
R$ 3.644 para R$ 20.211 e depois R$ 2.459. A renda e a variavel com maior sinal
(AUC univariada de aproximadamente 0,74 dentro de cada mes), mas as demais variaveis
acrescentam pouco.

## 2. Tratamento e prevencao de vazamento

Foram removidos `Unnamed: 0`, `index` e `data_ref` das features. O alvo booleano foi
convertido para 0/1. O pre-processamento foi colocado dentro de `Pipeline` +
`ColumnTransformer`: mediana e padronizacao para numericas, mediana com indicador de
ausencia para `tempo_emprego`, e moda com `OneHotEncoder(handle_unknown="ignore")` para
categoricas. Tudo e ajustado exclusivamente no treino.

Como os clientes se repetem entre meses, um split aleatorio por linha colocaria o mesmo
cliente em treino e teste. Optou-se por um split temporal: treino de 2015 (50.000 linhas
amostradas) e teste de 2016 (15.000 linhas), ambos estratificados por mes e alvo. A
validacao cruzada usa `StratifiedGroupKFold` de 10 folds com grupos por `index`, o que
evita o mesmo cliente em treino e validacao. Ainda assim, 9.421 clientes aparecem nos dois
periodos, o que e esperado no cenario temporal: o modelo estima o comportamento futuro de
clientes conhecidos.

## 3. Modelos e resultados

Foram treinados Regressao Logistica, Arvore de Decisao (`max_depth=4`), Random Forest
(300 arvores) e KNN, com e sem `class_weight="balanced"` quando aplicavel. A escolha do
modelo usou apenas o F1 out-of-fold na validacao, com desempate por recall e ROC/AUC; o
teste de 2016 ficou reservado. O limiar de decisao tambem foi calibrado no treino, por F1
out-of-fold.

| Modelo | Peso | Limiar | Acuracia | Precisao | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Arvore de Decisao | sem | 0,137 | 0,656 | 0,254 | 0,697 | 0,373 | 0,729 |
| Arvore de Decisao | balanced | 0,724 | 0,656 | 0,254 | 0,697 | 0,373 | 0,727 |
| Regressao Logistica | sem | 0,113 | 0,631 | 0,237 | 0,682 | 0,352 | 0,708 |
| Random Forest | sem | 0,117 | 0,636 | 0,224 | 0,599 | 0,326 | 0,667 |
| KNN | nao aplicavel | 0,077 | 0,657 | 0,194 | 0,424 | 0,266 | 0,589 |

O modelo escolhido foi a Arvore de Decisao sem balanceamento, com limiar 0,137. No teste,
a matriz de confusao foi TN=8.305, FP=4.494, FN=668 e TP=1.533. Na validacao cruzada, o F1
foi 0,264 +- 0,025 e o AUC 0,759 +- 0,016. O resultado e modesto, mas honesto: o teto de
F1 no teste e aproximadamente 0,38, e nenhum modelo superou AUC de 0,73.

## 4. Discussao

**Acuracia engana.** O modelo escolhido acerta 65,6%, bem menos que o baseline trivial de
92,2%, porque o baseline tem recall zero para inadimplentes. O falso negativo e o erro
mais caro: um cliente que nao paga e classificado como bom pagador gera perda financeira
direta. O falso positivo custa uma analise adicional ou a perda de um bom cliente. Por
isso o recall da classe `mau=1` e a metrica principal, acompanhada de precisao e F1.

**Limiar e prevalencia.** Com limiar 0,5, o modelo escolhido teria recall de apenas 0,18%,
porque foi treinado com prevalencia de 6,11% e testado com 14,67%. O limiar calibrado no
treino (0,137) elevou o recall para 69,7%. A precisao de 25,4% significa que cerca de tres
em cada quatro alertas sao falsos positivos; na pratica, o limiar deveria ser definido por
custo, por exemplo aceitando mais falsos positivos para elevar o recall. A tabela
`reports/limiares_teste.csv` mostra esse trade-off.

**Drift e sinal fraco.** A variacao mensal da taxa de inadimplencia e da renda indica que a
base mistura conjunturas economicas distintas. O split temporal e mais realista, mas
penaliza o desempenho. A renda domina a predicao; as demais variaveis contribuem pouco.
Nao houve sinais de vazamento: nenhuma acuracia se aproximou de 100% e o pre-processamento
foi ajustado apenas no treino.

**Limitacoes.** A amostragem de 50.000 linhas de treino reduz o custo computacional, mas
descarta informacao. O modelo nao foi calibrado probabilisticamente (apenas o limiar foi
ajustado) e nao serve para decisao real de credito. Uma evolucao seria incluir variaveis
temporais, avaliar custo assimetrico e testar modelos de gradient boosting.

## 5. Reflexao critica sobre o uso da LLM

A LLM foi util para acelerar a escrita do pipeline, sugerir a verificacao da estrutura real
da base e implementar o app. Ela tambem identificou um risco que o plano original nao
previa: a repeticao de clientes por `index`, que exigiria split agrupado ou temporal.

Entretanto, a LLM errou em pontos que so a execucao revelou: um acesso incorreto a colunas
em `itertuples`, a selecao inicial de modelo pelo F1 do proprio teste e o tratamento da
string `None` como valor ausente no aplicativo. O primeiro resultado (F1 ~0,30 com limiar
0,5) parecia aceitavel, mas a investigacao mostrou que o limiar estava miscalibrado. Isso
confirma que a LLM acelera a producao, mas nao substitui a conferencia dos numeros, a
decisao metodologica e a responsabilidade pelos resultados. Todas as escolhas relevantes
foram registradas no diario de bordo, com prompts, verificacoes e correcoes.

## 6. Conclusao

O pipeline atende aos criterios do trabalho: leitura e EDA reproduziveis, tratamento sem
vazamento, quatro algoritmos comparados, validacao cruzada agrupada por cliente, aplicativo
Streamlit com cinco abas e documentacao. O desempenho e limitado pela base, nao pelo
codigo: AUC de 0,729, F1 de 0,373 e recall de 0,697 no teste temporal. O principal
aprendizado e metodologico: com classes desbalanceadas e drift, acuracia isolada engana, e
o limiar, a prevalencia e o custo dos erros precisam ser discutidos explicitamente.
