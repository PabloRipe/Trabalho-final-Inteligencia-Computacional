# Credit Scoring com Aprendizado de Maquina e apoio de LLM

Aplicativo didatico de classificacao binaria de inadimplencia construido com a base
publica **Credit Scoring** do Kaggle. O projeto cobre leitura, analise exploratoria,
tratamento sem vazamento, treino e comparacao de quatro algoritmos, validacao
cruzada agrupada por cliente, aplicativo Streamlit e documentacao.

O codigo foi escrito com apoio do agente **opencode** rodando o modelo DeepSeek.
As decisoes, verificacoes e correcoes estao registradas em `diario_de_bordo.md`.

## Guias de reproducao com outras LLMs

- [Passo a passo com Claude Code](docs/GUIA_CLAUDE_CODE.md)
- [Passo a passo com Codex (OpenAI)](docs/GUIA_CODEX_OPENAI.md)

## Base de dados

| Item | Valor |
|---|---|
| Pagina no Kaggle | https://www.kaggle.com/datasets/caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia |
| Download via CLI | `kaggle datasets download -d caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia` |
| Arquivo esperado | `data/credit_scoring.csv` |
| Dimensoes | 750.000 linhas x 16 colunas |
| Alvo | `mau` (1 = inadimplente, 0 = bom pagador) |
| Taxa de inadimplencia | 7,82% (58.643 casos) |
| Identificador de cliente | `index` (16.650 clientes, repetidos em ate 15 meses) |
| Valores ausentes | `tempo_emprego` com 16,79% (125.957) |
| Duplicatas exatas | 0 |

O download pelo navegador exige login no Kaggle. A extracao do CSV deve deixar o arquivo
em `data/credit_scoring.csv`. O codigo tambem aceita o arquivo na raiz do projeto.

## Instalacao

Requer Python 3.10 ou superior (testado com 3.13.5).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Execucao

```powershell
python src/train_eval.py
streamlit run app.py
```

O primeiro comando gera todos os artefatos em `reports/` e o modelo final em
`models/modelo.joblib`. O aplicativo apenas le esses arquivos; ele nao retreina nada.
Se os artefatos estiverem ausentes, o app mostra o comando que deve ser executado.

O treino completo leva cerca de 9 a 10 minutos nesta configuracao (Random Forest e a
validacao cruzada de 10 folds concentram o tempo).

## Estrutura

```
formiga/
├─ data/credit_scoring.csv          # base (nao versionar)
├─ src/config.py                    # caminhos, colunas, hiperparametros
├─ src/train_eval.py                # EDA, split, treino, metricas e artefatos
├─ models/modelo.joblib             # pipeline final (preprocessamento + modelo)
├─ reports/
│  ├─ metricas.csv                  # 7 experimentos no teste de 2016
│  ├─ metricas_cv.csv               # validacao cruzada agrupada por cliente
│  ├─ limiares_teste.csv            # trade-off precisao/recall por limiar
│  ├─ matriz_confusao.png
│  ├─ roc.png
│  ├─ precisao_recall.png
│  ├─ eda_*.png                     # graficos da analise exploratoria
│  ├─ dados_amostra.csv             # 5.000 linhas de treino para o app
│  ├─ opcoes_formulario.json
│  ├─ metadata.json                 # resumo completo da execucao
│  └─ decisao_modelo.txt
├─ app.py
├─ requirements.txt
├─ PLANO.md
├─ diario_de_bordo.md
└─ relatorio.md
```

## Decisoes de tratamento

- **Identificadores e data removidos das features:** `Unnamed: 0`, `index` e `data_ref`.
  O `index` e mantido apenas como grupo na validacao cruzada.
- **Alvo:** `mau` convertido de booleano para 0/1.
- **`tempo_emprego`:** imputacao pela mediana com `add_indicator=True`, gerando um
  indicador de ausencia. Ajuste feito somente no treino.
- **Categoricas:** imputacao pela moda e `OneHotEncoder(handle_unknown="ignore")`.
- **Numericas:** `StandardScaler`.
- **Sem vazamento:** todo o preprocessamento esta dentro de `Pipeline` +
  `ColumnTransformer`; nada e ajustado no conjunto de teste.
- **Divisao temporal:** treino de janeiro a dezembro de 2015 (50.000 linhas amostradas)
  e teste de janeiro a marco de 2016 (15.000 linhas), ambos estratificados por mes e alvo.
- **Por que nao split aleatorio por linha:** 9.421 clientes aparecem nos dois periodos e o
  mesmo cliente em treino e teste inflaria as metricas. O split temporal avalia
  comportamento futuro, que e o uso realista de um modelo de credito.
- **Validacao cruzada:** `StratifiedGroupKFold` de 10 folds com grupos por cliente,
  evitando o mesmo cliente em treino e validacao.
- **Balanceamento:** os quatro algoritmos foram comparados com e sem
  `class_weight="balanced"` (KNN nao suporta esse parametro).
- **Limiar:** calibrado por F1 out-of-fold no treino. O limiar 0,5 assume prevalencia de
  6,11% (treino), mas o teste tem 14,67%; usar 0,5 derrubaria o recall a 0,18%.

## Resultados no teste (2016)

Metricas da classe `mau=1`, com limiar calibrado no treino:

| Modelo | Peso | Limiar | Acuracia | Precisao | Recall | F1 | ROC/AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| **Arvore de Decisao** | sem | 0,137 | 0,656 | 0,254 | **0,697** | **0,373** | **0,729** |
| Arvore de Decisao | balanced | 0,724 | 0,656 | 0,254 | 0,697 | 0,373 | 0,727 |
| Regressao Logistica | sem | 0,113 | 0,631 | 0,237 | 0,682 | 0,352 | 0,708 |
| Regressao Logistica | balanced | 0,630 | 0,617 | 0,231 | 0,692 | 0,346 | 0,703 |
| Random Forest | sem | 0,117 | 0,636 | 0,224 | 0,599 | 0,326 | 0,667 |
| Random Forest | balanced | 0,197 | 0,583 | 0,205 | 0,639 | 0,310 | 0,650 |
| KNN | nao aplicavel | 0,077 | 0,657 | 0,194 | 0,424 | 0,266 | 0,589 |

Modelo escolhido: **Arvore de Decisao sem `class_weight`**, limiar 0,137, por maior F1
out-of-fold (0,265; desvio 0,025) na validacao agrupada. Matriz de confusao no teste:
TN=8.305, FP=4.494, FN=668, TP=1.533.

Baseline trivial (sempre "bom pagador"): acuracia 92,18%, porem recall 0 e F1 0.
Acuracia isolada engana: o modelo escolhido tem acuracia menor que o baseline porque
aceita muitos falsos positivos para capturar 69,7% dos inadimplentes.

Validacao cruzada (media +- desvio em 10 folds, no treino): F1 0,264 +- 0,025,
recall 0,340 +- 0,041, ROC/AUC 0,759 +- 0,016.

## Limitacoes

- **Sinal fraco:** o melhor AUC e ~0,73. A renda concentra quase todo o sinal
  (AUC univariada de ~0,74 dentro de cada mes). As demais variaveis acrescentam pouco.
- **Drift temporal:** a taxa de inadimplencia varia de 2,72% a 16,82% entre os meses.
  A distribuicao de renda tambem muda muito, o que dificulta a generalizacao para 2016.
- **Precisao baixa:** com recall de 69,7%, mais de dois tercos dos clientes sinalizados
  sao falsos positivos. Na pratica isso exigiria analise humana ou um limiar orientado a custo.
- **Amostragem:** o treino usa 50.000 das 600.000 linhas de 2015 para manter o tempo de
  execucao viavel, sobretudo para KNN e Random Forest.
- **Uso didatico:** o modelo nao deve decidir credito real.

## Reproducibilidade

`random_state=42` em amostragem, split, modelos e folds. As versoes usadas estao em
`requirements.txt` e em `reports/metadata.json`:

- Python 3.13.5
- pandas 3.1.1
- numpy 2.5.4
- scikit-learn 1.9.1
- streamlit 1.67.0
- matplotlib 3.11.2
- seaborn 0.13.2

## Diferencas em relacao ao PLANO.md original

- A taxa de inadimplencia real e 7,82%, nao ~8,6%.
- `mau` vem como booleano, nao como 0/1.
- O identificador `index` se repete entre meses (16.650 clientes); o split e agrupado ou
  temporal, nao aleatorio por linha.
- Existe drift mensal forte no alvo e na renda, que o plano nao previa.
- A base nao tem duplicatas exatas, entao a decisao do roteiro sobre 236 duplicatas nao se aplica.
- Foi incluida calibracao de limiar e validacao cruzada agrupada por cliente.
- Os valores de referencia do roteiro (~95,7% etc.) nao valem para esta base.
