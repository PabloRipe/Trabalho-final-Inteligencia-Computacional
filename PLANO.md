# Plano do Projeto — Aprendizado de Máquina com apoio de LLM

**Trabalho:** aplicativo de ML com apoio de LLM (classificação binária)
**Ferramenta:** opencode (agente de código no terminal, fluxo equivalente ao Claude Code)
**Modelo:** DeepSeek (opencode)
**Stack:** Python 3.10+, pandas, scikit-learn, Streamlit, matplotlib/seaborn

---

## 1. Base de dados e link para download

| Item | Definição |
|---|---|
| Base | Credit Scoring — características pessoais e bancárias / inadimplência |
| Página no Kaggle | https://www.kaggle.com/datasets/caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia |
| Download via CLI | `kaggle datasets download -d caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia` |
| Arquivo | `credit_scoring.csv` |
| Tamanho | ≈78 MB |
| Dimensões | ≈750.000 linhas × 16 colunas |
| Target | `mau` — `1` = inadimplente, `0` = bom pagador (no CSV vem como booleano) |
| Contexto | Base de crédito usada em cursos brasileiros (estilo Nubank/EBAC), com colunas em português |
| Clientes | 16.650 clientes distintos em `index`, repetidos em até 15 meses |
| Inadimplência | 7,82% (58.643 casos) — baseline trivial de acurácia = 92,18% |
| Drift | taxa mensal varia de 2,72% (jul/2015) a 16,82% (mar/2016) |

> **Verificação pós-execução:** os números acima foram conferidos no CSV real. O PLANO
> original estimava 8,6% de inadimplentes e não previa a repetição de clientes nem o drift.

> O download pelo navegador exige login no Kaggle. O download via CLI exige o arquivo `kaggle.json` configurado. Após baixar, extrair o CSV para `data/credit_scoring.csv`.

### Estrutura observada na base

- **Numéricas:** `qtd_filhos`, `idade`, `tempo_emprego`, `qt_pessoas_residencia`, `renda`.
- **Categóricas:** `sexo`, `posse_de_veiculo`, `posse_de_imovel`, `tipo_renda`, `educacao`, `estado_civil`, `tipo_residencia`.
- **Identificadores/data (remover):** `Unnamed: 0`, `index`, `data_ref`.
- **Target:** `mau`.
- **Faltantes:** `tempo_emprego` tem 16,79% de valores ausentes (125.957 na base completa).
- **Duplicatas:** 0 linhas duplicadas exatas na base completa.
- **Desbalanceamento:** 7,82% de inadimplentes na base completa.
- **Clientes repetidos:** `index` identifica 16.650 clientes ao longo de 15 meses; split
  aleatório por linha vazaria informação do mesmo cliente entre treino e teste.

---

## 2. Objetivo

Construir o pipeline completo — leitura, exploração, tratamento, aprendizado e verificação — e transformá-lo em um aplicativo Streamlit funcional, com métricas reais, matriz de confusão, curva ROC, validação cruzada, README, diário de bordo e relatório curto.

A LLM escreve o código; o grupo decide, confere e assume a responsabilidade pelos resultados.

---

## 3. Entregáveis

| # | Entregável | Conteúdo |
|---|---|---|
| 1 | **Aplicativo funcionando** | `app.py`, `requirements.txt` e instruções de execução no README |
| 2 | **Diário de bordo** | Prompts usados, respostas resumidas, verificações, correções e decisões humanas |
| 3 | **Relatório curto (1–2 páginas)** | Decisões de tratamento, comparação dos modelos e reflexão crítica sobre o uso da LLM |
| Apoio | Script de treino/avaliação | `src/train_eval.py` com métricas, matriz de confusão, ROC e validação cruzada reais |
| Apoio | `PLANO.md` | Este documento |
| Apoio | `README.md` | Instalar, rodar, etapas, decisões, resultados e limitações |
| Apoio | `diario_de_bordo.md` | Modelo preenchido ao longo do projeto |
| Apoio | `relatorio.md` | Rascunho do relatório de 1–2 páginas |
| Apoio | `reports/` | Tabelas e gráficos gerados |

---

## 4. Premissas obrigatórias

### 4.1 Vazamento de dados (data leakage)

- Imputador, encoder e padronizador serão ajustados **somente no treino**, dentro de `Pipeline` + `ColumnTransformer`.
- `Unnamed: 0`, `index` e `data_ref` serão removidos para não vazar informação.
- **Split temporal:** treino em 2015, teste nos três primeiros meses de 2016 (estratificado por mês e alvo). O split aleatório 70/30 foi descartado porque 9.421 clientes aparecem nos dois períodos.
- Validação cruzada `StratifiedGroupKFold` 10-fold com grupos por `index`.
- Critério de alerta: acurácia de 100% ou muito acima do baseline deve ser investigada, não comemorada.

### 4.2 A métrica que engana

- Com 7,82% de inadimplentes, um modelo que sempre responde “bom pagador” já acerta 92,18%. Portanto, **acurácia sozinha não basta**.
- A métrica principal será o **recall da classe `mau=1`** (evitar falso negativo: classificar inadimplente como bom pagador), acompanhada de precisão, F1, matriz de confusão e ROC/AUC.
- O limiar de decisão será calibrado no treino por F1 out-of-fold, porque a prevalência muda de 6,11% (treino) para 14,67% (teste).
- O relatório deverá explicar qual erro é mais grave neste contexto e por quê.

### 4.3 Decisão humana sobre duplicatas e dados faltantes

A base completa **não possui duplicatas exatas**. A decisão sobre as 236 duplicatas do roteiro original não se aplica literalmente. As decisões humanas equivalentes que precisarão ser justificadas no diário de bordo são:

1. Manter ou remover `tempo_emprego` com ≈17% de faltantes;
2. Imputar pela mediana, pela moda ou criar indicador de ausência;
3. Remover `Unnamed: 0`, `index` e `data_ref`;
4. Usar a base completa ou uma amostra estratificada;
5. Usar ou não `class_weight="balanced"`;
6. Usar split aleatório, agrupado por cliente ou temporal (decidido: temporal);
7. Calibrar ou não o limiar de decisão (decidido: sim, por F1 out-of-fold).

Cada escolha será registrada com prós, contras e justificativa.

> **Atenção:** os números do roteiro original (699×10, 458×241, 236 duplicatas, 9 sliders de 1 a 10, 95,7% de acurácia) são referências da base de câncer de mama. Nesta base, eles mudam. A estrutura do pipeline permanece a mesma.

---

## 5. Ferramenta escolhida

**opencode** (equivalente ao Claude Code).

| Ferramenta | Por que não foi escolhida |
|---|---|
| ChatGPT | Exige copiar/colar e executar manualmente; mais erros de ambiente |
| Lovable | Foco em front-end; não treina modelo; exigiria exportar coeficientes em JSON |

**Fluxo:** o opencode cria a pasta, escreve os arquivos, instala as bibliotecas, executa o pipeline, lê os erros e corrige. O grupo confere cada saída, valida os números e registra as decisões no diário.

---

## 6. Etapas de implementação

| Etapa | O que fazer | Produto |
|---|---|---|
| **0. Setup** | Criar a pasta do projeto, `venv`, instalar dependências e colocar o CSV em `data/` | Ambiente pronto |
| **1. Leitura** | Ler `credit_scoring.csv`; conferir 750k×16, tipos, `tempo_emprego` numérico com faltantes e colunas de ID/data | Confirmação da estrutura |
| **2. Exploração** | Distribuição de `mau` (~8,6%), faltantes (~17% em `tempo_emprego`), duplicatas (0), estatísticas, correlação das numéricas e distribuição das categóricas | Gráficos + 3–4 frases |
| **3. Tratamento** | Remover IDs/data; `mau` → 0/1; split temporal 2015/2016 com amostragem estratificada por mês e alvo; `ColumnTransformer`: numéricas com mediana + indicador + padronização, categóricas com moda + OneHotEncoder; **fit só no treino** | Dados prontos, sem vazamento |
| **4. Aprendizado** | Treinar Regressão Logística, Árvore (`max_depth=4`), Random Forest e KNN na mesma divisão, com e sem `class_weight="balanced"` | 7 experimentos treinados |
| **5. Verificação** | Acurácia, precisão, recall, F1 da classe `mau=1`, matriz de confusão, ROC/AUC e validação cruzada agrupada 10-fold; calibrar o limiar por F1 out-of-fold; comparar com o baseline trivial (92,18%) | Tabela comparativa + gráficos |
| **6. Aplicativo** | Streamlit com abas Dados, Tratamento, Aprendizado, Verificação e “Testar um caso”; formulário adaptado (sliders e selects) + aviso didático | `app.py` + `requirements.txt` |
| **7. Documentação** | README, diário de bordo e relatório curto | Documentação final |
| **8. Revisão** | Rodar em ambiente limpo, testar casos extremos e conferir critérios de aceite | Entrega validada |

### Valores de referência

Os valores do roteiro (Regressão Logística e Random Forest ≈95,7%, Árvore ≈93,8%, KNN ≈95,2%) **não se aplicam** a esta base. Será construída uma tabela de referência própria, com o baseline trivial e os resultados reais obtidos.

---

## 7. Estrutura de arquivos prevista

```
credit_scoring_ml/
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
├─ README.md
├─ PLANO.md
├─ diario_de_bordo.md
└─ relatorio.md
```

---

## 8. Modelo de diário de bordo

| Campo | O que registrar |
|---|---|
| Data/hora | Momento da interação |
| Etapa | Leitura, exploração, tratamento, aprendizado, verificação ou app |
| Prompt usado | Texto integral enviado à LLM |
| Resposta resumida | O que a LLM propôs |
| O que foi conferido | Números, código, saída real |
| Correções feitas | Ajustes pedidos ou feitos manualmente |
| Decisão humana | Escolha e justificativa |
| Status | OK / refazer |

---

## 9. Critérios de aceite

- `python src/train_eval.py` roda em ambiente limpo e gera métricas e gráficos.
- `streamlit run app.py` abre com as 4 abas + “Testar um caso” + aviso didático.
- Nenhum vazamento: imputador, encoder e scaler ajustados só no treino.
- Resultados incluem acurácia, precisão, recall, F1, matriz de confusão, ROC/AUC e CV 10-fold (média e desvio).
- Modelo escolhido justificado por recall/F1, não apenas acurácia.
- README com instalação, execução, decisões, resultados e limitações.
- Diário de bordo completo e relatório de 1–2 páginas.
- Se aparecer 100% de acurácia, o resultado será investigado antes de ser aceito.

---

## 10. Riscos e decisões que exigem justificativa

| Risco | Mitigação |
|---|---|
| 750k linhas podem deixar o treino lento | Amostrar 30k–50k linhas estratificadas e registrar a decisão |
| Desbalanceamento (7,82%) | Comparar com/sem `class_weight="balanced"`, split estratificado, foco em recall/F1 |
| `index` repete o mesmo cliente entre meses | Split temporal e CV agrupada por `index` |
| Drift mensal (2,72% a 16,82%) | Split temporal, limiar calibrado no treino, avaliação por mês |
| `tempo_emprego` com ~17% de faltantes | Mediana + indicador de ausência; comparar cenários |
| Categóricas exigem encoding | `OneHotEncoder(handle_unknown="ignore")` dentro do pipeline |
| Sem duplicatas exatas | Registrar que a decisão do roteiro não se aplica e justificar as decisões equivalentes |
| Valores de referência do roteiro não valem | Construir tabela de referência própria desta base |
| Formulário do app não terá 9 sliders de 1 a 10 | Adaptar para sliders/selects coerentes com crédito |
| Reprodutibilidade | `random_state=42`, versões no `requirements.txt`, prompts registrados |

---

## 11. Próximos passos

1. Gravar a estrutura de pastas e o `requirements.txt`.
2. Colocar o `credit_scoring.csv` em `data/`.
3. Implementar `src/train_eval.py` e rodar a primeira versão.
4. Conferir métricas, matriz de confusão e ROC; ajustar se necessário.
5. Implementar `app.py` e testar as abas.
6. Escrever README, diário de bordo e relatório.
7. Revisão final em ambiente limpo.

## 12. Status de execução (15/11/2026)

Todos os passos foram executados. O pipeline roda com `python src/train_eval.py`
(~9–10 min) e o app com `streamlit run app.py` (testado via `AppTest`, sem exceções).

| Resultado | Valor |
|---|---|
| Modelo escolhido | Árvore de Decisão, sem `class_weight`, limiar 0,137 |
| Critério | maior F1 out-of-fold na CV agrupada por cliente (teste reservado) |
| Teste 2016 | acurácia 0,656; precisão 0,254; recall 0,697; F1 0,373; AUC 0,729 |
| Matriz de confusão | TN=8.305; FP=4.494; FN=668; TP=1.533 |
| Baseline trivial | acurácia 0,922; recall 0 e F1 0 |
| CV 10-fold | F1 0,264 ± 0,025; recall 0,340 ± 0,041; AUC 0,759 ± 0,016 |

> A acurácia do modelo é menor que a do baseline de propósito: o modelo aceita falsos
> positivos para capturar inadimplentes (recall 0,697 contra 0 do baseline). A
> discriminação é limitada pela base (AUC ~0,73, com renda concentrando o sinal), e os
> resultados não devem ser usados para decisão real de crédito.
