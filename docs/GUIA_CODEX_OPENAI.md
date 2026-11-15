# Guia passo a passo com Codex (OpenAI)

Este guia ensina como reproduzir o projeto **Credit Scoring com aprendizado de maquina e
apoio de LLM** usando o **Codex CLI da OpenAI** como agente de programacao. O resultado
esperado e o mesmo deste repositorio: pipeline de treino/avaliacao em Python, artefatos em
`reports/`, modelo persistido em `models/modelo.joblib` e aplicativo Streamlit com cinco
abas.

O guia assume que voce vai construir o projeto do zero. O codigo deste repositorio pode ser
usado como referencia, mas rode o pipeline e confira os numeros.

---

## 1. O que sera construido

| Item | Conteudo |
|---|---|
| Base | Credit Scoring (Kaggle), 750.000 linhas x 16 colunas |
| Alvo | `mau` (1 = inadimplente, 0 = bom pagador) |
| Modelos | Regressao Logistica, Arvore de Decisao, Random Forest e KNN |
| Balanceamento | com e sem `class_weight="balanced"` quando aplicavel (7 experimentos) |
| Validacao | split temporal 2015/2016 + `StratifiedGroupKFold` 10-fold por cliente |
| Metrica principal | recall/F1 da classe `mau=1`, com limiar calibrado por F1 out-of-fold |
| Aplicativo | Streamlit com abas Dados, Tratamento, Aprendizado, Verificacao e Testar um caso |
| Entregaveis | `app.py`, `src/train_eval.py`, `README.md`, `diario_de_bordo.md`, `relatorio.md` |

### Estrutura final esperada

```text
credit_scoring_llm/
├─ data/
│  └─ credit_scoring.csv
├─ src/
│  ├─ config.py
│  └─ train_eval.py
├─ models/
│  └─ modelo.joblib
├─ reports/
│  ├─ metricas.csv
│  ├─ metricas_cv.csv
│  ├─ limiares_teste.csv
│  ├─ matriz_confusao.png
│  ├─ roc.png
│  ├─ precisao_recall.png
│  ├─ eda_*.png
│  ├─ dados_amostra.csv
│  ├─ opcoes_formulario.json
│  ├─ metadata.json
│  └─ decisao_modelo.txt
├─ app.py
├─ requirements.txt
├─ PLANO.md
├─ README.md
├─ diario_de_bordo.md
└─ relatorio.md
```

---

## 2. Pre-requisitos

- Python 3.10 ou superior (o projeto foi testado com Python 3.13.5).
- Git.
- Conta no Kaggle e, preferencialmente, a CLI do Kaggle configurada.
- Codex CLI com login via ChatGPT ou chave de API.
- Terminal PowerShell no Windows, ou Bash/Zsh no macOS/Linux.

### Instalar o Codex CLI

No Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
codex --version
```

Alternativas:

```powershell
npm install -g @openai/codex
```

```bash
# macOS/Linux
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

```bash
# macOS via Homebrew
brew install --cask codex
```

Na primeira execucao, rode `codex` e escolha **Sign in with ChatGPT** ou outro metodo de
autenticacao disponivel.

---

## 3. Preparar o ambiente do projeto

No PowerShell:

```powershell
mkdir credit_scoring_llm
cd credit_scoring_llm
git init
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

No macOS/Linux, troque a ativacao por:

```bash
source .venv/bin/activate
```

Crie o `requirements.txt` com as versoes usadas como referencia:

```text
pandas==3.1.1
numpy==2.5.4
scikit-learn==1.9.1
streamlit==1.67.0
matplotlib==3.11.2
seaborn==0.13.2
joblib==1.6.0
```

Instale:

```powershell
python -m pip install -r requirements.txt
```

### Baixar a base

Pagina do Kaggle:

```text
https://www.kaggle.com/datasets/caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia
```

Com a CLI configurada:

```powershell
kaggle datasets download -d caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia
Expand-Archive -Path credit_scoring.zip -DestinationPath .
```

Mova o CSV para `data/credit_scoring.csv`:

```powershell
New-Item -ItemType Directory -Path data -Force
Move-Item -Path credit_scoring.csv -Destination data/credit_scoring.csv
```

Se preferir, baixe pelo navegador e salve diretamente em `data/credit_scoring.csv`.

### Verificacao rapida da base

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/credit_scoring.csv', low_memory=False); print(df.shape); print(df.columns.tolist()); print(df['mau'].value_counts(dropna=False))"
```

Valores de referencia da base real:

| Item | Valor |
|---|---|
| Dimensoes | 750.000 x 16 |
| Alvo `mau=1` | 58.643 casos (7,82%) |
| Baseline trivial | 92,18% de acuracia, recall 0 |
| Clientes distintos | 16.650 em `index` |
| `tempo_emprego` ausente | 125.957 (16,79%) |
| Duplicatas exatas | 0 |
| Periodo | janeiro/2015 a marco/2016 |
| Drift mensal | 2,72% a 16,82% |

---

## 4. Criar o arquivo de contexto do Codex

O Codex le automaticamente um arquivo `AGENTS.md` na raiz do projeto. Ele funciona como
memoria persistente do repositorio. Voce pode gerar um esboco com `/init` dentro do Codex e
depois substituir pelo conteudo abaixo.

```markdown
# AGENTS.md - Credit Scoring com apoio de LLM

## Objetivo
Construir um pipeline completo de classificacao binaria de inadimplencia e um app
Streamlit didatico, sem vazamento de dados.

## Base
- CSV: data/credit_scoring.csv (750.000 x 16).
- Alvo: mau (1 = inadimplente, 0 = bom pagador). No CSV original vem como booleano.
- Identificadores/data que NAO entram como features: Unnamed: 0, index, data_ref.
- index identifica 16.650 clientes repetidos em ate 15 meses.
- tempo_emprego tem 16,79% de ausentes.
- Nao ha duplicatas exatas.

## Regras obrigatorias
1. Todo pre-processamento deve estar dentro de Pipeline + ColumnTransformer e ser ajustado
   somente no treino.
2. Nunca usar index, data_ref, Unnamed: 0 ou o alvo como feature.
3. Split temporal: treino em 2015 (50.000 linhas) e teste em 2016 (15.000 linhas),
   amostrados de forma estratificada por mes e alvo.
4. Validacao cruzada StratifiedGroupKFold de 10 folds, com grupos por index.
5. Escolher o modelo pelo maior F1 out-of-fold. Desempate por recall e depois ROC/AUC.
   O teste de 2016 nao pode ser usado na escolha.
6. Calibrar o limiar de decisao por F1 out-of-fold no treino. Nunca usar o teste para isso.
7. Persistir apenas o pipeline final em models/modelo.joblib com joblib.
8. O app Streamlit deve apenas ler os artefatos; nao deve retreinar.
9. Usar random_state=42 em amostragem, split, modelos e folds.
10. Nao inventar metricas. Rodar o codigo e registrar os numeros reais.

## Estilo
- Python 3.10+, type hints quando ajudarem, funcoes pequenas e nomes claros.
- Comentarios apenas quando explicarem uma decisao nao obvia.
- Encoding UTF-8.
- PowerShell no Windows: nao usar &&; usar ; ou comandos separados.
- Antes de grandes mudancas, apresentar um plano curto e esperar aprovacao.
- Depois de cada etapa, rodar uma verificacao e mostrar a saida relevante.
- Nao executar comandos destrutivos de git sem pedido explicito.
```

### Dica de fluxo

1. Abra o Codex na raiz do projeto:

   ```powershell
   codex
   ```

2. Use `/plan` para pedir um plano antes de permitir implementacao.
3. Use `/permissions` para escolher o nivel de autonomia. Para trabalho local, prefira
   `workspace-write` e aprovacao sob demanda; evite desabilitar o sandbox sem necessidade.
4. Use `/diff` para revisar as mudancas antes de aceitar.
5. Use `/review` ou `codex review --uncommitted` para uma revisao dedicada.
6. Use `/export session.md` para salvar o transcript da sessao e anexar ao diario.
7. Use `/clear` entre etapas grandes para reduzir o contexto, mantendo o `AGENTS.md`.
8. Use `codex resume` para continuar uma sessao anterior.

---

## 5. Roteiro por etapas

### Etapa 0 - Planejamento

**Prompt para o Codex:**

```text
Leia o AGENTS.md e proponha um plano de implementacao em etapas para este projeto.
Nao escreva codigo ainda. Para cada etapa, liste os arquivos que serao criados ou alterados,
os comandos de verificacao e os riscos de vazamento de dados. Considere:
- base de 750.000 linhas, alvo desbalanceado (7,82%);
- clientes repetidos em index;
- tempo_emprego com 16,79% de ausentes;
- drift temporal forte;
- treino 2015 e teste 2016;
- comparacao de 4 algoritmos com e sem class_weight;
- escolha por F1 out-of-fold e limiar calibrado no treino.
```

**O que conferir:**

- O plano respeita o split temporal e a validacao agrupada por cliente.
- O teste de 2016 fica reservado para a avaliacao final.
- O plano inclui limiar calibrado por F1 out-of-fold.
- O plano nao propoe usar acuracia como criterio unico de escolha.

---

### Etapa 1 - Leitura e validacao da base

**Prompt:**

```text
Implemente a leitura e a validacao da base em src/train_eval.py, usando src/config.py para
caminhos e listas de colunas. O script deve:
1. aceitar data/credit_scoring.csv e, como fallback, credit_scoring.csv na raiz;
2. validar a presenca de todas as colunas obrigatorias;
3. converter o alvo booleano mau para inteiro 0/1;
4. garantir que Unnamed: 0, index e data_ref nao entram nas features;
5. registrar no console: shape, clientes distintos, duplicatas, taxa de mau=1 e
   percentual de ausentes em tempo_emprego.
Nao faca EDA grafica nem treino ainda.
```

**O que conferir:**

- O shape e 750.000 x 16.
- A taxa de `mau=1` e aproximadamente 7,82%.
- Ha 16.650 clientes distintos em `index`.
- `tempo_emprego` tem aproximadamente 16,79% de ausentes.
- Nenhuma coluna de ID/data aparece em `FEATURES`.

---

### Etapa 2 - Analise exploratoria

**Prompt:**

```text
Adicione a EDA ao pipeline, salvando em reports/:
- eda_target_mes.png: taxa de mau=1 por data_ref;
- eda_faltantes.png: percentual de ausentes por coluna;
- eda_numericas.png: histogramas das numericas (use log10 para renda);
- eda_categoricas.png: taxa de mau=1 por categoria, com linha da media geral;
- eda_correlacao.png: heatmap de correlacao entre numericas e alvo.

Use matplotlib com backend Agg e seaborn. Tambem registre em um dicionario os fatos:
taxa do alvo, baseline, ausentes, drift minimo e maximo, numero de clientes e duplicatas.
Nao remova outliers automaticamente.
```

**O que conferir:**

- Os cinco PNGs existem e abrem.
- O drift mensal mostra variacao forte (2,72% a 16,82%).
- A renda aparece como a variavel com maior sinal.
- O grafico de faltantes mostra apenas `tempo_emprego` com ausencia relevante.

---

### Etapa 3 - Split temporal e pre-processamento

**Prompt:**

```text
Implemente o split e o pre-processamento com prevencao de vazamento:
1. Treino: data_ref < 2016-01-01. Teste: data_ref >= 2016-01-01.
2. Amostre 50.000 linhas de treino e 15.000 de teste, estratificando pela combinacao
   data_ref + mau, com random_state=42.
3. Monte um ColumnTransformer:
   - numericas: SimpleImputer(strategy="median", add_indicator=True) + StandardScaler;
   - categoricas: SimpleImputer(strategy="most_frequent") +
     OneHotEncoder(handle_unknown="ignore", sparse_output=False).
4. Coloque o pre-processamento dentro de um Pipeline junto com o modelo.
5. Registre no console as taxas de mau=1 no treino e no teste e quantos clientes aparecem
   nos dois periodos.
Nao ajuste nada no conjunto de teste.
```

**O que conferir:**

- O treino tem 50.000 linhas e o teste tem 15.000.
- As taxas mensais sao preservadas pela amostragem estratificada.
- O treino tem cerca de 6,11% de `mau=1` e o teste cerca de 14,67%.
- Cerca de 9.421 clientes aparecem nos dois periodos, o que e esperado no split temporal.
- O imputador, o encoder e o scaler so sao ajustados no treino.

---

### Etapa 4 - Treinar os sete experimentos

**Prompt:**

```text
Implemente os experimentos no mesmo split:
- Regressao Logistica: max_iter=2000;
- Arvore de Decisao: max_depth=4;
- Random Forest: n_estimators=300, n_jobs=-1;
- KNN: n_neighbors=25, weights="distance", n_jobs=-1.
Para Regressao Logistica, Arvore e Random Forest, treine com class_weight=None e
class_weight="balanced". Para KNN, registre "nao_aplicavel".
Use random_state=42.
Nao salve cada modelo em disco: Random Forests ocupam muito espaco. Guarde apenas em
memoria para a avaliacao.
```

**O que conferir:**

- Sao exatamente 7 experimentos.
- Cada pipeline inclui o pre-processamento.
- O treino nao usa o teste.
- O tempo total fica em torno de 9 a 10 minutos na configuracao de referencia.

---

### Etapa 5 - Validacao cruzada, limiar e escolha do modelo

**Prompt:**

```text
Implemente a avaliacao com prevencao de vazamento:
1. Para cada experimento, rode StratifiedGroupKFold(n_splits=10, shuffle=True,
   random_state=42) com grupos por index.
2. Gere probabilidades out-of-fold no treino.
3. Escolha o limiar por F1 out-of-fold usando precision_recall_curve.
4. Calcule no teste, com esse limiar: acuracia, precisao, recall, F1, ROC/AUC,
   TN, FP, FN e TP.
5. Escolha o modelo pelo maior F1 out-of-fold; desempate por recall e depois ROC/AUC.
6. Salve reports/metricas.csv, reports/metricas_cv.csv e reports/limiares_teste.csv.
7. Gere matriz_confusao.png, roc.png e precisao_recall.png do modelo escolhido.
8. Persista apenas o pipeline escolhido em models/modelo.joblib.
9. Gere reports/metadata.json, reports/decisao_modelo.txt,
   reports/dados_amostra.csv e reports/opcoes_formulario.json.
Inclua as metricas com limiar 0,5 apenas para comparacao.
```

**O que conferir:**

- O modelo e escolhido apenas pelo F1 out-of-fold, nunca pelo F1 do teste.
- O limiar do modelo escolhido e menor que 0,5 (referencia: 0,137).
- No limiar 0,5, o recall cai drasticamente (referencia: 0,18%).
- A matriz de confusao do teste e TN=8.305, FP=4.494, FN=668, TP=1.533 na referencia.
- As metricas de referencia no teste sao: acuracia 0,656, precisao 0,254, recall 0,697,
  F1 0,373 e AUC 0,729.
- O baseline trivial tem acuracia 0,922, mas recall 0 e F1 0.

Se os numeros forem muito diferentes, investigue antes de aceitar. Nao ajuste nada usando o
teste para "melhorar" o resultado.

---

### Etapa 6 - Aplicativo Streamlit

**Prompt:**

```text
Crie app.py com Streamlit, lendo apenas os artefatos de reports/ e models/modelo.joblib.
Requisitos:
1. Sidebar com nome do modelo, peso das classes, limiar, versoes e aviso de uso didatico.
2. Abas: Dados, Tratamento, Aprendizado, Verificacao e Testar um caso.
3. Aba Dados: metricas gerais, grafico de drift, amostra filtravel e graficos da EDA.
4. Aba Tratamento: decisoes de tratamento e prevencao de vazamento.
5. Aba Aprendizado: tabela comparativa, grafico de barras e tabela da validacao cruzada.
6. Aba Verificacao: cartoes de metricas, matriz de confusao, ROC, precisao x recall e
   tabela de limiares.
7. Aba Testar um caso: formulario com selects, sliders e number inputs coerentes com a base.
   Use um botao "Analisar crédito" com st.button (nao use st.form_submit_button) para que
   a tecla Enter nao dispare a analise. Guarde o resultado em st.session_state para permitir
   novos testes sem recarregar a pagina. Inclua um botao "Limpar campos e resultado".
   Mostre probabilidade, classificacao com o limiar calibrado, barra de progresso e a linha
   enviada ao modelo.
8. Use st.cache_data para JSON/CSV e st.cache_resource para o modelo.
9. Normalize a coluna peso_classes com fillna("None"), porque o pandas le a string "None"
   como NaN ao carregar metricas.csv.
10. Se artefatos estiverem ausentes, mostre o comando `python src/train_eval.py`.
```

**O que conferir:**

- O app abre com `streamlit run app.py`.
- As cinco abas funcionam sem excecao.
- Clicar em "Analisar crédito" gera o resultado.
- Alterar os campos e clicar novamente atualiza o resultado sem recarregar a pagina.
- A tecla Enter nos campos nao executa a analise.
- O botao de limpar reseta campos e resultado.
- O app nao retreina e nao usa o teste.

Teste automatizado opcional:

```powershell
python -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('app.py', default_timeout=60); at.run(); print('excecoes:', at.exception)"
```

---

### Etapa 7 - Documentacao

**Prompt:**

```text
Atualize a documentacao com os numeros reais do pipeline:
1. README.md: base, instalacao, execucao, estrutura, decisoes, resultados e limitacoes.
2. diario_de_bordo.md: prompts usados, respostas resumidas, verificacoes, correcoes e
   decisoes humanas. Inclua as decisoes sobre split temporal, amostragem, balanceamento,
   limiar e duplicatas/ausentes.
3. relatorio.md: relatorio curto de 1 a 2 paginas com introducao, tratamento, resultados,
   discussao e reflexao critica sobre o uso do Codex.
Nao invente numeros: use reports/*.csv e reports/metadata.json.
```

**O que conferir:**

- README e relatorio citam os numeros reais.
- O diario registra pelo menos uma correcao feita pelo agente e como foi detectada.
- As limitacoes (AUC ~0,73, precisao baixa, drift, amostragem, uso didatico) estao claras.

---

### Etapa 8 - Revisao final

**Prompt:**

```text
Revise o projeto como se fosse uma entrega final:
1. Rode python -m py_compile em todos os arquivos Python.
2. Rode o pipeline completo e confira os artefatos gerados.
3. Abra o app e teste as cinco abas, incluindo um caso com renda baixa e outro com renda alta.
4. Verifique se ha vazamento de dados ou uso indevido do teste.
5. Verifique se o README, o diario e o relatorio estao consistentes com os CSVs.
6. Liste os problemas encontrados e corrija apenas o que for necessario.
```

**Comandos:**

```powershell
python -m py_compile src/config.py src/train_eval.py app.py
python src/train_eval.py
streamlit run app.py
```

**Revisao dedicada do Codex:**

```powershell
codex review --uncommitted
```

**Criterios de aceite:**

- [ ] `python src/train_eval.py` roda em ambiente limpo e gera todos os artefatos.
- [ ] `streamlit run app.py` abre com as cinco abas.
- [ ] Nenhum identificador, data ou alvo entra nas features.
- [ ] Imputador, encoder e scaler sao ajustados apenas no treino.
- [ ] Split temporal e CV agrupada por cliente foram usados.
- [ ] Modelo escolhido por F1 out-of-fold, com teste reservado.
- [ ] Limiar calibrado no treino, nao no teste.
- [ ] Metricas incluem acuracia, precisao, recall, F1, matriz de confusao, ROC/AUC e CV.
- [ ] App permite novo teste pelo botao, sem recarregar a pagina.
- [ ] README, diario e relatorio consistentes com os resultados.
- [ ] Nenhuma acuracia proxima de 100% sem investigacao de vazamento.

---

## 6. Como conduzir o Codex com seguranca

- Comece com `/plan` e revise o plano antes de permitir edicoes.
- Use `/permissions` para manter aprovacao sob demanda. Para trabalho local, use
  `--sandbox workspace-write`; nao use `--dangerously-bypass-approvals-and-sandbox` fora de
  uma VM isolada.
- Revise com `/diff` antes de aceitar mudancas grandes.
- Rode `codex review --uncommitted` antes de cada commit importante.
- Faca commits pequenos ao final de cada etapa aprovada.
- Se o agente propuser usar o teste para escolher modelo ou limiar, recuse.
- Se aparecer acuracia de 100% ou muito acima do baseline, investigue vazamento antes de
  comemorar.
- Use `codex exec` para verificacoes repetiveis e registre a saida.
- Use `/export session.md` para guardar o transcript e citar prompts no diario.

### Exemplo de sequencia de comandos

```powershell
codex
# dentro do Codex:
# /init
# /plan
# cole o prompt da Etapa 0
# aprove o plano
# execute as etapas 1 a 8, revisando /diff e rodando /review
# /export diario_sessao.md
```

### Automacao opcional

Para uma verificacao nao interativa:

```powershell
codex exec "Rode python -m py_compile em src/config.py, src/train_eval.py e app.py; se falhar, explique o erro e nao altere arquivos."
```

Para revisar as mudancas:

```powershell
codex review --uncommitted
```

---

## 7. Registro no diario de bordo

Registre cada interacao relevante. Um formato simples:

| # | Data/hora | Etapa | Prompt usado | Resposta resumida | O que foi conferido | Correcoes | Decisao humana | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 15/11 10:00 | Leitura | "valide a base" | Confirmou shape e taxa | CSV real | Nenhuma | Prosseguir | OK |
| 2 | 15/11 10:20 | Split | "implemente split temporal" | Propôs 50k/15k | Taxas por mes | Nenhuma | Aprovar | OK |
| 3 | 15/11 11:00 | Avaliacao | "calibre o limiar" | Propôs F1 OOF | Sem uso do teste | Ajustou criterio | Aprovar | OK |

Inclua tambem:

- decisoes humanas com prós e contras;
- correcoes feitas depois de erros do agente;
- verificacoes numericas (shape, taxa, clientes, ausentes, metricas);
- limitacoes conhecidas.

O comando `/export session.md` salva o transcript completo em Markdown. Anexe ou resuma as
partes relevantes no `diario_de_bordo.md`.

---

## 8. Problemas comuns

| Problema | Causa provavel | Solucao |
|---|---|---|
| `FileNotFoundError` | CSV fora de `data/` | Coloque em `data/credit_scoring.csv` ou na raiz |
| `mau` com tipo inesperado | CSV traz booleano | Converter para `int64` apos validar dominio |
| Metricas altas demais | Vazamento de `index`, `data_ref` ou alvo | Revisar `FEATURES` e o Pipeline |
| Recall baixo no teste | Limiar 0,5 com prevalencia diferente | Calibrar limiar por F1 out-of-fold |
| `None` virou `NaN` | pandas le a string `None` como ausente | `fillna("None")` antes de filtrar/mapear |
| Random Forest lento ou pesado | 300 arvores x 10 folds | Reduzir amostra/arvores e registrar a decisao |
| App nao atualiza o resultado | Analise dentro de `st.form` | Usar `st.button` + `st.session_state` |
| Enter dispara a analise | `st.form_submit_button` | Trocar por `st.button` |
| Graficos nao aparecem | Backend interativo | Usar `matplotlib.use("Agg")` |
| `&&` falha no PowerShell | Sintaxe de outro shell | Usar `;` ou comandos separados |
| Codex pede permissao demais | Sandbox restritivo | Ajustar com `/permissions` ou `--sandbox workspace-write` |
| Contexto ficou longo | Sessao acumulada | `/compact` ou `/clear`, mantendo `AGENTS.md` |

---

## 9. Referencias

- Documentacao do Codex CLI: https://developers.openai.com/codex/cli
- Comandos do Codex CLI: https://developers.openai.com/codex/developer-commands
- AGENTS.md: https://developers.openai.com/codex/agent-configuration/agents-md
- Base no Kaggle: https://www.kaggle.com/datasets/caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia
- Repositorio de referencia: https://github.com/PabloRipe/Trabalho-final-Inteligencia-Computacional

> Aviso: este projeto e didatico. As metricas sao modestas e o modelo nao deve ser usado para
> conceder ou negar credito real.
