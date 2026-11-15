# Diario de bordo

Projeto: Credit Scoring com aprendizado de maquina e apoio de LLM
Ferramenta: opencode (agente no terminal) com modelo DeepSeek
Periodo: 15/11/2026

## Resumo das interacoes

| # | Data/hora | Etapa | Prompt usado | Resposta resumida | O que foi conferido | Correcoes | Decisao humana | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 15/11 11:30 | Leitura | "leia o arquivo plano e vamos discutir a implementacao" | Leu `PLANO.md`, conferiu o CSV real e apontou divergencias | Shape 750.000x16, taxa 7,82%, 16.650 clientes, 16,79% ausentes, drift mensal | Nenhuma | Prosseguir com discussao | OK |
| 2 | 15/11 11:40 | Planejamento | Perguntas sobre split, amostra, persistencia e balanceamento | Recomendou split temporal, 50k/15k, joblib e comparar com/sem balanced | Alternativas e riscos de cada escolha | Nenhuma | Aprovar as quatro recomendacoes | OK |
| 3 | 15/11 11:45 | Setup | "pode manter tudo em train_eval, pode iniciar" | Criou pastas, `.venv`, `requirements.txt`, `config.py` e `train_eval.py` | Versoes instaladas, moveu CSV para `data/` | Nenhuma | Manter tudo em `train_eval.py` | OK |
| 4 | 15/11 11:52 | Leitura/EDA | Execucao de `python src/train_eval.py` | Erro `TypeError` no dicionario de drift | Stack trace apontou `itertuples` e acesso por nome | Trocado por `zip` das colunas | Corrigir e rodar de novo | OK |
| 5 | 15/11 11:57 | Aprendizado | Continuacao da execucao | 7 experimentos treinados; melhor F1 0,299 com balanced | Metricas, matriz, tempos | Nenhuma | Nao aceitar resultado sem investigar | OK |
| 6 | 15/11 12:00 | Verificacao | Diagnostico dos resultados baixos | AUC ~0,73 e teto de F1 ~0,38; renda domina o sinal; limiar 0,5 inadequado | AUC univariada, AUC por mes, F1 por limiar, drift de renda | Nenhuma | Investigar antes de entregar | OK |
| 7 | 15/11 12:02 | Verificacao | Calibrar limiar na validacao | F1 out-of-fold como criterio; limiar escolhido sem usar o teste | Threshold OOF 0,137; teste F1 0,373 | Reestruturado CV, escolha e metricas | Aceita para revisao do grupo | OK |
| 8 | 15/11 12:04 | Aplicativo | Escrever `app.py` com 5 abas | App le artefatos, 5 abas, teste de caso ao vivo | `AppTest` retornou excecao no Aprendizado | Corrigido `None` lido como NaN | Usar modelo persistido | OK |
| 9 | 15/11 12:09 | Revisao | Testar interacoes do app | Abas, filtros e formulario funcionam; renda altera a probabilidade | Sem excecoes; probabilidade 24,86% com renda 1.000 e 0,57% com 500.000 | Nenhuma | Manter limiar calibrado no app | OK |
| 10 | 15/11 12:15 | Documentacao | Fechar README, relatorio e diario | Documentos com numeros reais e limitacoes | Tabelas conferidas com `reports/*.csv` | Atualizar `PLANO.md` | Revisar no grupo | OK |

## Decisoes humanas registradas

### D1. Divisao temporal (2015 treino, 2016 teste)

- **Mantida porque:** 9.421 clientes aparecem em ambos os periodos e o split aleatorio
  colocaria o mesmo cliente nos dois lados. O split temporal tambem mede o drift.
- **Contra:** perde a comparacao com o split 70/30 do roteiro e a taxa de teste (14,67%)
  e bem maior que a de treino (6,11%).
- **Justificativa:** reflete o uso realista (prever comportamento futuro) e evita
  vazamento por cliente.

### D2. Amostra de 50.000 treino e 15.000 teste

- **Mantida porque:** KNN e Random Forest com 10 folds ficariam inviaveis na base completa.
- **Contra:** perde parte da variabilidade de 750.000 linhas.
- **Justificativa:** amostragem estratificada por mes e alvo preserva as taxas mensais
  (2,71% a 12,12% no treino; 12,42% a 16,82% no teste).

### D3. Modelo persistido em `models/modelo.joblib`

- **Mantida porque:** o app abre rapido e nao duplica a logica de treino.
- **Contra:** exige rodar `train_eval.py` antes de abrir o app.
- **Justificativa:** reprodutibilidade e separacao entre treino e apresentacao.

### D4. Comparacao com e sem `class_weight="balanced"`

- **Mantida porque:** mostra o trade-off precisao/recall.
- **Contra:** dobra o numero de experimentos.
- **Justificativa:** o relatorio precisa justificar a escolha por recall/F1, nao por acuracia.

### D5. Calibracao do limiar por F1 out-of-fold

- **Mantida porque:** no limiar 0,5 o modelo escolhido teria recall de apenas 0,18%,
  porque a prevalencia muda de 6,11% (treino) para 14,67% (teste).
- **Contra:** acrescenta uma etapa metodologica nao prevista no plano original.
- **Justificativa:** o limiar foi escolhido apenas com dados de treino (out-of-fold),
  sem consultar o teste. No teste, recall sobe para 69,7% e F1 para 0,373.
- **Status:** registrada para validacao do grupo.

### D6. Decisao sobre duplicatas e ausentes

- Duplicatas exatas: 0. A decisao do roteiro sobre 236 duplicatas nao se aplica.
- `tempo_emprego`: mantida como feature, imputacao pela mediana com indicador de ausencia.
- `Unnamed: 0`, `index` e `data_ref`: removidos das features.

## Verificacoes numericas

- Shape original: 750.000 x 16.
- Alvo: 58.643 `mau=1` (7,82%); baseline de acuracia 92,18%.
- Clientes distintos: 16.650; `index` se repete em ate 15 meses.
- Ausentes: 125.957 em `tempo_emprego` (16,79%).
- Drift mensal: 12,11% (jan/2015) -> 2,72% (jul/2015) -> 16,82% (mar/2016).
- AUC univariada da renda dentro do mes: ~0,74 (relacao inversa).
- Teto observado de F1 no teste para o modelo escolhido: ~0,38.
- Modelo final: Arvore de Decisao, limiar 0,1374.
  Teste: acuracia 0,6559; precisao 0,2544; recall 0,6965; F1 0,3726; AUC 0,7285.
- Matriz de confusao no teste: TN=8.305; FP=4.494; FN=668; TP=1.533.
- CV 10-fold agrupada: F1 0,264 +- 0,025; recall 0,340 +- 0,041; AUC 0,759 +- 0,016.

## Correcoes tecnicas feitas durante o projeto

1. `by_month.itertuples()` com acesso `row["mean"]` falhava; substituido por `zip` das colunas.
2. O modelo escolhido inicialmente por F1 no teste usava limiar 0,5 e ficava inutilizavel
   (recall 0,18%); passou a usar limiar out-of-fold e escolha por F1 de validacao.
3. A coluna `peso_classes` contem a string `None`, que o `pandas.read_csv` converte em NaN;
   o app passou a normalizar com `fillna("None")` antes de ordenar e filtrar.
4. Modelos individuais nao sao mais salvos em disco (os Random Forests somavam ~550 MB);
   apenas o modelo final e persistido.
