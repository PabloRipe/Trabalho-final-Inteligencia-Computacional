"""Aplicativo Streamlit do projeto de credit scoring.

Le apenas os artefatos gerados por src/train_eval.py; não retreina o modelo.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as cfg


st.set_page_config(
    page_title="Credit Scoring - ML com LLM",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def pct(value: float) -> str:
    return f"{value:.2%}"


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def peso_label(value) -> str:
    labels = {
        "None": "sem class_weight",
        "nan": "sem class_weight",
        "balanced": "balanced",
        "nao_aplicavel": "não aplicável",
    }
    return labels.get(str(value), str(value))


required = [
    cfg.METADATA_PATH,
    cfg.METRICS_PATH,
    cfg.CV_METRICS_PATH,
    cfg.THRESHOLD_TABLE_PATH,
    cfg.SAMPLE_PATH,
    cfg.BEST_MODEL_PATH,
]
missing = [path for path in required if not path.exists()]

if missing:
    st.title("Credit Scoring - ML com apoio de LLM")
    st.error("Artefatos ausentes. Gere o modelo e os relatorios antes de abrir o app.")
    st.code("python src/train_eval.py", language="powershell")
    st.write("Arquivos faltando:")
    for path in missing:
        st.write(f"- `{path}`")
    st.stop()

metadata = load_json(str(cfg.METADATA_PATH))
metrics = load_table(str(cfg.METRICS_PATH))
cv_metrics = load_table(str(cfg.CV_METRICS_PATH))
thresholds = load_table(str(cfg.THRESHOLD_TABLE_PATH))
sample = load_table(str(cfg.SAMPLE_PATH))
for _frame in (metrics, cv_metrics):
    _frame["peso_classes"] = (
        _frame["peso_classes"].fillna("None").astype(str).map(peso_label)
    )
model = load_model(str(cfg.BEST_MODEL_PATH))

chosen = metadata["modelo_escolhido"]
chosen_metrics = chosen["metricas"]
threshold = float(chosen["limiar"])
split = metadata["split"]
baseline = float(metadata["baseline_accuracy"])

with st.sidebar:
    st.header("Sobre o projeto")
    st.markdown(
        "Classificação binaria de inadimplencia com a base publica **Credit Scoring** "
        "(Kaggle: `caiodouglas/caracteristicas-pessoais-e-bancarios-inadimplencia`)."
    )
    st.markdown(
        f"**Modelo em uso:** {chosen['nome']}  \n"
        f"**Peso das classes:** {peso_label(chosen['peso_classes'])}  \n"
        f"**Limiar de decisão:** {fmt(threshold, 3)}"
    )
    st.divider()
    st.caption("Gerado em " + metadata["gerado_em"])
    st.caption(
        "Versoes: Python {python}, pandas {pandas}, scikit-learn {scikit_learn}".format(
            **metadata["versoes"]
        )
    )
    st.divider()
    st.warning(
        "Uso didatico. As métricas sao modestas e o modelo não deve ser usado "
        "para conceder ou negar credito real."
    )

st.title("Credit Scoring - aprendizado de maquina com apoio de LLM")
st.caption(
    f"Base com {metadata['shape_original'][0]:,} linhas e "
    f"{metadata['shape_original'][1]} colunas; "
    f"treino em {split['treino_periodo'][0]} a {split['treino_periodo'][1]} e "
    f"teste em {split['teste_periodo'][0]} a {split['teste_periodo'][1]}."
)

tab_dados, tab_tratamento, tab_aprendizado, tab_verificacao, tab_caso = st.tabs(
    ["Dados", "Tratamento", "Aprendizado", "Verificação", "Testar um caso"]
)

with tab_dados:
    st.subheader("Visão geral da base")
    colunas = st.columns(5)
    colunas[0].metric("Linhas", f"{metadata['shape_original'][0]:,}")
    colunas[1].metric("Colunas", metadata["shape_original"][1])
    colunas[2].metric("Clientes distintos", f"{metadata['n_clientes']:,}")
    colunas[3].metric("Inadimplentes (mau=1)", pct(metadata["taxa_target"]))
    colunas[4].metric("Baseline trivial", pct(baseline))

    st.markdown(
        f"A base tem **{metadata['duplicatas']} duplicatas exatas** e "
        f"**{pct(metadata['faltantes_tempo_emprego']['taxa'])} de ausencia em "
        "`tempo_emprego`**. O alvo e `mau` (1 = inadimplente). "
        f"O cliente e identificado por `index`, com {metadata['n_clientes']:,} clientes "
        "repetidos ao longo de 15 meses de referencia."
    )

    st.image(
        str(cfg.EDA_TARGET_MONTH_PATH),
        caption="Taxa de inadimplencia por mes: forte variacao temporal (drift).",
    )
    st.markdown(
        "A taxa mensal vai de "
        f"{pct(metadata['drift_mensal'][split['treino_periodo'][0]])} em "
        f"{split['treino_periodo'][0]} a "
        f"{pct(metadata['drift_mensal'][split['teste_periodo'][1]])} em "
        f"{split['teste_periodo'][1]}. Por isso a avaliacao usa um periodo futuro "
        "como teste e o limiar de decisão e calibrado no treino."
    )

    with st.expander("Amostra de treino com ate 5.000 linhas"):
        meses = sorted(sample["data_ref"].unique())
        selecionados = st.multiselect("Filtrar meses de referencia", meses, default=meses)
        alvo = st.radio(
            "Filtrar alvo",
            ["Todos", "Somente mau=0", "Somente mau=1"],
            horizontal=True,
        )
        visao = sample[sample["data_ref"].isin(selecionados)]
        if alvo == "Somente mau=0":
            visao = visao[visao[cfg.TARGET] == 0]
        elif alvo == "Somente mau=1":
            visao = visao[visao[cfg.TARGET] == 1]
        st.write(f"{len(visao):,} linhas na selecao")
        st.dataframe(visao.head(500), hide_index=True)

    with st.expander("Gráficos da análise exploratoria"):
        st.image(str(cfg.EDA_NUMERIC_PATH), caption="Distribuicoes das numericas.")
        st.image(str(cfg.EDA_CATEGORICAL_PATH), caption="Taxa de mau=1 por categoria.")
        st.image(str(cfg.EDA_CORR_PATH), caption="Correlacao entre numericas e alvo.")

with tab_tratamento:
    st.subheader("Decisões de tratamento")
    st.markdown(
        """
- Remocao de identificadores e data: `Unnamed: 0`, `index` e `data_ref` não entram como features.
- `mau` convertido de booleano para 0/1.
- `tempo_emprego` (16,79% ausente): imputacao pela mediana e indicador de ausencia,
  ambos ajustados apenas no treino.
- Categoricas: imputacao pela moda e `OneHotEncoder(handle_unknown="ignore")`.
- Numericas: padronizacao com `StandardScaler` dentro do `Pipeline`.
- Divisao temporal: 2015 para treino e 2016 para teste, com amostragem de 50.000 e 15.000 linhas
  estratificada por mes e alvo.
- Sem duplicatas exatas: a decisão sobre as 236 duplicatas do roteiro original não se aplica.
- Balanceamento: os quatro algoritmos foram comparados com e sem `class_weight="balanced"`.
- Limiar: calibrado no treino por F1 out-of-fold, porque a prevalencia muda de 6,1% para 14,7% entre treino e teste.
"""
    )

    st.image(
        str(cfg.EDA_MISSING_PATH),
        caption="Unica coluna com valores ausentes: tempo_emprego.",
    )

    st.markdown("### Prevencao de vazamento")
    st.markdown(
        """
- Imputador, encoder e scaler sao ajustados dentro de `Pipeline` + `ColumnTransformer`,
  exclusivamente no treino.
- Nenhum identificador ou o alvo entra na matriz de features.
- A validação cruzada e agrupada por cliente (`StratifiedGroupKFold`), evitando o mesmo
  cliente em treino e validação.
- A selecao do modelo usa apenas F1 out-of-fold; o teste de 2016 fica reservado.
- Acuriacias muito altas seriam tratadas como suspeita de vazamento, não como sucesso.
"""
    )

with tab_aprendizado:
    st.subheader("Comparação dos modelos")
    st.markdown(
        "Métricas no teste de 2016 com limiar calibrado no treino. "
        "O experimento escolhido foi definido pelo maior F1 out-of-fold na validação agrupada "
        "por cliente, com desempate por recall e AUC."
    )

    pesos = sorted(metrics["peso_classes"].unique())
    selecionados = st.multiselect("Filtrar por peso das classes", pesos, default=pesos)
    visao = metrics[metrics["peso_classes"].isin(selecionados)]

    colunas_metricas = [
        "modelo",
        "peso_classes",
        "limiar",
        "acuracia",
        "precisao",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    st.dataframe(
        visao[colunas_metricas],
        hide_index=True,
        column_config={
            "limiar": st.column_config.NumberColumn("limiar", format="%.3f"),
            **{
                coluna: st.column_config.NumberColumn(coluna, format="%.3f")
                for coluna in ["acuracia", "precisao", "recall", "f1", "roc_auc"]
            },
        },
    )

    grafico = visao.assign(
        experimento=visao["modelo"] + " / " + visao["peso_classes"]
    ).set_index("experimento")[["precisao", "recall", "f1"]]
    st.bar_chart(grafico)

    st.markdown("### Validação cruzada agrupada por cliente")
    st.dataframe(
        cv_metrics[
            [
                "modelo",
                "peso_classes",
                "limiar",
                "f1_oof",
                "recall_oof",
                "roc_auc_oof",
                "f1_media",
                "f1_desvio",
                "recall_media",
                "recall_desvio",
                "roc_auc_media",
                "roc_auc_desvio",
            ]
        ],
        hide_index=True,
        column_config={
            "limiar": st.column_config.NumberColumn("limiar", format="%.3f"),
            **{
                coluna: st.column_config.NumberColumn(coluna, format="%.3f")
                for coluna in [
                    "f1_oof",
                    "recall_oof",
                    "roc_auc_oof",
                    "f1_media",
                    "f1_desvio",
                    "recall_media",
                    "recall_desvio",
                    "roc_auc_media",
                    "roc_auc_desvio",
                ]
            },
        },
    )
    st.caption(
        f"Media e desvio de {cfg.N_SPLITS} folds. O limiar e o mesmo em todos os folds "
        "e foi escolhido pelo F1 out-of-fold."
    )

    with st.expander("Por que não usar limiar 0,5?"):
        st.markdown(
            "O limiar 0,5 supoe que a prevalencia no teste e parecida com a do treino. "
            f"Aqui a taxa de mau=1 e {pct(split['taxa_treino'])} no treino e "
            f"{pct(split['taxa_teste'])} no teste. No limiar 0,5, o modelo escolhido teria "
            f"recall {pct(chosen['metricas_limiar_0_5']['recall_limiar_0_5'])} e "
            f"F1 {fmt(chosen['metricas_limiar_0_5']['f1_limiar_0_5'])}; "
            f"com o limiar {fmt(threshold, 2)}, o recall sobe para "
            f"{pct(chosen_metrics['recall'])} e o F1 para {fmt(chosen_metrics['f1'])}."
        )

with tab_verificacao:
    st.subheader(
        f"Desempenho do modelo escolhido: {chosen['nome']} com {peso_label(chosen['peso_classes'])}"
    )
    cartoes = st.columns(6)
    cartoes[0].metric("Recall (mau=1)", pct(chosen_metrics["recall"]))
    cartoes[1].metric("Precisão", pct(chosen_metrics["precisao"]))
    cartoes[2].metric("F1", fmt(chosen_metrics["f1"]))
    cartoes[3].metric("ROC/AUC", fmt(chosen_metrics["roc_auc"]))
    cartoes[4].metric("Acurácia", pct(chosen_metrics["acuracia"]))
    cartoes[5].metric("Baseline (acurácia)", pct(baseline))

    st.markdown(
        f"O baseline que sempre responde bom pagador acerta {pct(baseline)} dos casos, "
        f"mas tem recall zero para `mau=1`. O modelo escolhido acerta "
        f"{pct(chosen_metrics['acuracia'])} e captura {pct(chosen_metrics['recall'])} "
        "dos inadimplentes, ao custo de muitos falsos positivos. "
        "Neste contexto, o falso negativo (inadimplente classificado como bom pagador) "
        "custa mais caro do que o falso positivo."
    )

    colunas_imagens = st.columns(2)
    colunas_imagens[0].image(
        str(cfg.CONFUSION_PATH),
        caption=(
            f"Matriz de confusão no teste: TN={int(chosen_metrics['tn'])}, "
            f"FP={int(chosen_metrics['fp'])}, FN={int(chosen_metrics['fn'])}, "
            f"TP={int(chosen_metrics['tp'])}."
        ),
    )
    colunas_imagens[1].image(str(cfg.ROC_PATH), caption="Curvas ROC de todos os experimentos.")

    st.image(
        str(cfg.PR_PATH),
        caption="Precisão x recall do modelo escolhido; o ponto marca o limiar calibrado.",
    )

    with st.expander("Tabela de limiares no teste"):
        proximo = thresholds.iloc[
            (thresholds["limiar"] - threshold).abs().argsort()[:1]
        ]
        st.write("Linha mais próxima do limiar escolhido:")
        st.dataframe(proximo, hide_index=True)
        st.dataframe(
            thresholds,
            hide_index=True,
            column_config={
                coluna: st.column_config.NumberColumn(coluna, format="%.3f")
                for coluna in ["limiar", "precisao", "recall", "f1", "acuracia", "positivos_previstos"]
            },
        )
        st.caption(
            "A tabela é descritiva. O limiar em uso foi escolhido apenas com dados de treino, "
            "por F1 out-of-fold."
        )

with tab_caso:
    st.subheader("Simular um cliente")
    st.markdown(
        "Preencha as caracteristicas e veja a probabilidade estimada de inadimplencia. "
        "O resultado usa o modelo e o limiar carregados dos artefatos de treino."
    )

    opcoes = metadata["opcoes_formulario"]
    categorias = opcoes["categorias"]
    numericos = opcoes["numericos"]

    coluna_1, coluna_2, coluna_3 = st.columns(3)

    sexo = coluna_1.selectbox("Sexo", categorias["sexo"])
    posse_veiculo = coluna_2.selectbox("Possui veículo", categorias["posse_de_veiculo"])
    posse_imovel = coluna_3.selectbox("Possui imóvel", categorias["posse_de_imovel"])

    qtd_filhos = coluna_1.slider(
        "Quantidade de filhos",
        int(numericos["qtd_filhos"]["min"]),
        int(numericos["qtd_filhos"]["max"]),
        int(numericos["qtd_filhos"]["mediana"]),
    )
    idade = coluna_2.slider(
        "Idade",
        int(numericos["idade"]["min"]),
        int(numericos["idade"]["max"]),
        int(numericos["idade"]["mediana"]),
    )
    qt_pessoas = coluna_3.slider(
        "Pessoas na residência",
        int(numericos["qt_pessoas_residencia"]["min"]),
        int(numericos["qt_pessoas_residencia"]["max"]),
        int(numericos["qt_pessoas_residencia"]["mediana"]),
    )

    tipo_renda = coluna_1.selectbox("Tipo de renda", categorias["tipo_renda"])
    educacao = coluna_2.selectbox("Escolaridade", categorias["educacao"])
    estado_civil = coluna_3.selectbox("Estado civil", categorias["estado_civil"])
    tipo_residencia = coluna_1.selectbox(
        "Tipo de residência", categorias["tipo_residencia"]
    )

    tempo_desconhecido = coluna_2.checkbox(
        "Tempo de emprego desconhecido", value=False
    )
    tempo_emprego = coluna_2.number_input(
        "Tempo de emprego (anos)",
        min_value=0.0,
        max_value=60.0,
        value=round(float(numericos["tempo_emprego"]["mediana"]), 2),
        step=0.5,
        disabled=tempo_desconhecido,
        help="Se marcado como desconhecido, entra como ausente e o pipeline aplica a mediana.",
    )
    renda = coluna_3.number_input(
        "Renda mensal (R$)",
        min_value=100.0,
        max_value=3_000_000.0,
        value=round(float(numericos["renda"]["mediana"]), 2),
        step=100.0,
    )

    caso = pd.DataFrame(
        [
            {
                "qtd_filhos": qtd_filhos,
                "idade": idade,
                "tempo_emprego": np.nan if tempo_desconhecido else tempo_emprego,
                "qt_pessoas_residencia": qt_pessoas,
                "renda": renda,
                "sexo": sexo,
                "posse_de_veiculo": posse_veiculo,
                "posse_de_imovel": posse_imovel,
                "tipo_renda": tipo_renda,
                "educacao": educacao,
                "estado_civil": estado_civil,
                "tipo_residencia": tipo_residencia,
            }
        ],
        columns=metadata["features"],
    )

    probabilidade = float(model.predict_proba(caso)[0, 1])
    previsao = 1 if probabilidade >= threshold else 0

    st.divider()
    resultado_1, resultado_2, resultado_3 = st.columns(3)
    resultado_1.metric("Probabilidade estimada de mau=1", pct(probabilidade))
    resultado_2.metric(
        "Classificação com o limiar atual",
        "Inadimplente (1)" if previsao == 1 else "Bom pagador (0)",
    )
    resultado_3.metric("Limiar de decisão", fmt(threshold, 3))

    st.progress(min(max(probabilidade, 0.0), 1.0))
    st.caption(
        "A barra mostra a probabilidade estimada; a classificação usa o limiar calibrado, "
        "não necessariamente 0,5."
    )

    st.info(
        "Resultado didatico. A base tem poder preditivo limitado "
        f"(AUC ~{fmt(chosen_metrics['roc_auc'], 2)}) e o modelo não substitui uma análise de credito."
    )

    with st.expander("Ver a linha enviada ao modelo"):
        st.dataframe(caso, hide_index=True)
