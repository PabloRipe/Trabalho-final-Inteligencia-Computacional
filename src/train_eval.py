"""Pipeline de ponta a ponta: leitura, EDA, tratamento, treino, avaliacao e artefatos.

Uso:
    python src/train_eval.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
sns.set_theme(style="whitegrid")
RNG = np.random.default_rng(cfg.RANDOM_STATE)


def log(msg: str) -> None:
    print(msg, flush=True)


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().replace(" ", "_")


def load_data() -> pd.DataFrame:
    path = cfg.DATA_PATH if cfg.DATA_PATH.exists() else cfg.DATA_PATH_FALLBACK
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado. Coloque o CSV em {cfg.DATA_PATH} "
            f"(ou em {cfg.DATA_PATH_FALLBACK})."
        )
    log(f"[leitura] {path}")
    started = time.perf_counter()
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    log(f"[leitura] {df.shape[0]:,} linhas x {df.shape[1]} colunas em {time.perf_counter() - started:.1f}s")
    return df


def validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    required = set(cfg.ID_COLS + cfg.NUMERICAS + cfg.CATEGORICAS + [cfg.TARGET])
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing_cols)}")

    if df[cfg.TARGET].isna().any():
        raise ValueError("Target com valores nulos.")
    if not set(df[cfg.TARGET].unique()) <= {0, 1, False, True}:
        raise ValueError("Target fora do dominio binario esperado.")

    df = df.copy()
    df[cfg.TARGET] = df[cfg.TARGET].astype("int64")
    df["qt_pessoas_residencia"] = df["qt_pessoas_residencia"].astype("int64")

    overlap = set(cfg.FEATURES) & set(cfg.ID_COLS)
    if overlap:
        raise ValueError(f"Vazamento estrutural: IDs usados como feature: {sorted(overlap)}")
    if not set(cfg.FEATURES).isdisjoint({cfg.GROUP_COL, cfg.TARGET}):
        raise ValueError("Vazamento estrutural: grupo ou target usados como feature.")

    log("[validacao] estrutura, tipos e target OK")
    log(f"[validacao] clientes distintos ({cfg.GROUP_COL}): {df[cfg.GROUP_COL].nunique():,}")
    log(f"[validacao] duplicatas exatas: {int(df.duplicated().sum()):,}")
    log(f"[validacao] faltantes em tempo_emprego: {int(df['tempo_emprego'].isna().sum()):,} "
        f"({df['tempo_emprego'].isna().mean():.2%})")
    log(f"[validacao] taxa de {cfg.TARGET}=1: {df[cfg.TARGET].mean():.4%} "
        f"(baseline trivial de acuracia: {1 - df[cfg.TARGET].mean():.4%})")
    return df


def run_eda(df: pd.DataFrame) -> dict:
    log("\n[eda] gerando graficos e estatisticas")
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    facts: dict = {}

    target_rate = float(df[cfg.TARGET].mean())
    missing_rate = float(df["tempo_emprego"].isna().mean())
    by_month = df.groupby("data_ref")[cfg.TARGET].agg(["mean", "size"]).reset_index()
    drift = {
        str(month): float(rate)
        for month, rate in zip(by_month["data_ref"], by_month["mean"])
    }

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(by_month["data_ref"].astype(str), by_month["mean"] * 100, marker="o", color="#1f77b4")
    ax.axhline(target_rate * 100, linestyle="--", color="#d62728", label=f"media geral {target_rate:.2%}")
    ax.set_title("Taxa de inadimplencia (mau=1) por mes de referencia")
    ax.set_xlabel("data_ref")
    ax.set_ylabel("% de mau=1")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.EDA_TARGET_MONTH_PATH, dpi=130)
    plt.close(fig)

    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(miss.index, miss.values / len(df) * 100, color="#ff7f0e")
    ax.set_title("Percentual de valores ausentes por coluna")
    ax.set_xlabel("% ausente")
    for i, value in enumerate(miss.values / len(df) * 100):
        ax.text(value, i, f" {value:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(cfg.EDA_MISSING_PATH, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.ravel()
    for ax, col in zip(axes, cfg.NUMERICAS):
        data = df[col].dropna()
        if col == "renda":
            data = np.log10(data.clip(lower=1))
            ax.set_xlabel("log10(renda)")
        else:
            ax.set_xlabel(col)
        ax.hist(data, bins=40, color="#2ca02c", alpha=0.8)
        ax.set_title(col)
        ax.set_ylabel("frequencia")
    axes[-1].axis("off")
    fig.suptitle("Distribuicao das variaveis numericas")
    fig.tight_layout()
    fig.savefig(cfg.EDA_NUMERIC_PATH, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    axes = axes.ravel()
    for ax, col in zip(axes, cfg.CATEGORICAS):
        rates = df.groupby(col)[cfg.TARGET].mean().sort_values()
        ax.barh(rates.index.astype(str), rates.values * 100, color="#9467bd")
        ax.axvline(target_rate * 100, linestyle="--", color="#d62728", linewidth=1)
        ax.set_title(col)
        ax.set_xlabel("% de mau=1")
    axes[-1].axis("off")
    fig.suptitle("Taxa de mau=1 por categoria (linha tracejada = media geral)")
    fig.tight_layout()
    fig.savefig(cfg.EDA_CATEGORICAL_PATH, dpi=130)
    plt.close(fig)

    corr = df[cfg.NUMERICAS + [cfg.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlacao (Pearson) entre numericas e target")
    fig.tight_layout()
    fig.savefig(cfg.EDA_CORR_PATH, dpi=130)
    plt.close(fig)

    facts["target_rate"] = target_rate
    facts["baseline_accuracy"] = 1 - target_rate
    facts["missing_tempo_emprego_rate"] = missing_rate
    facts["drift_by_month"] = drift
    facts["drift_min"] = min(drift.values())
    facts["drift_max"] = max(drift.values())
    facts["n_clientes"] = int(df[cfg.GROUP_COL].nunique())
    facts["renda_mediana"] = float(df["renda"].median())
    facts["idade_min"] = int(df["idade"].min())
    facts["idade_max"] = int(df["idade"].max())
    facts["duplicatas"] = int(df.duplicated().sum())

    log(f"[eda] taxa de mau=1: {target_rate:.2%}; baseline trivial: {1 - target_rate:.2%}")
    log(f"[eda] drift mensal: de {facts['drift_min']:.2%} a {facts['drift_max']:.2%}")
    log(f"[eda] tempo_emprego ausente: {missing_rate:.2%}")
    log(f"[eda] {facts['n_clientes']:,} clientes distintos em {df['data_ref'].nunique()} meses")
    return facts


def stratified_month_sample(df: pd.DataFrame, n_total: int, label: str) -> pd.DataFrame:
    strat_key = df["data_ref"].astype(str) + "|" + df[cfg.TARGET].astype(str)
    sample, _ = train_test_split(
        df,
        train_size=min(n_total, len(df)),
        stratify=strat_key,
        random_state=cfg.RANDOM_STATE,
    )
    check = sample.groupby("data_ref")[cfg.TARGET].mean()
    log(f"[split] {label}: {len(sample):,} linhas | taxas por mes preservadas "
        f"(min {check.min():.2%}, max {check.max():.2%})")
    return sample.sort_values(["data_ref", cfg.GROUP_COL]).reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("\n[split] divisao temporal + amostragem estratificada por mes")
    train_full = df[df["data_ref"].astype(str) < cfg.TEST_START]
    test_full = df[df["data_ref"].astype(str) >= cfg.TEST_START]
    if train_full.empty or test_full.empty:
        raise ValueError("Split temporal vazio; revise TEST_START.")

    train = stratified_month_sample(train_full, cfg.TRAIN_SAMPLE_SIZE, "treino")
    test = stratified_month_sample(test_full, cfg.TEST_SAMPLE_SIZE, "teste")

    overlap_clients = set(train[cfg.GROUP_COL]) & set(test[cfg.GROUP_COL])
    log(f"[split] treino={len(train):,} ({train['data_ref'].min()} a {train['data_ref'].max()}) | "
        f"teste={len(test):,} ({test['data_ref'].min()} a {test['data_ref'].max()})")
    log(f"[split] clientes presentes nos dois periodos: {len(overlap_clients):,} "
        "(esperado no cenario temporal: o modelo avalia comportamento futuro de clientes conhecidos)")
    log(f"[split] mau=1 treino {train[cfg.TARGET].mean():.2%} | teste {test[cfg.TARGET].mean():.2%}")
    return train, test


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, cfg.NUMERICAS),
            ("cat", categorical_pipe, cfg.CATEGORICAS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_experiments() -> list[dict]:
    specs = [
        (
            "Regressao Logistica",
            lambda cw: LogisticRegression(max_iter=2000, class_weight=cw, random_state=cfg.RANDOM_STATE),
        ),
        (
            "Arvore de Decisao",
            lambda cw: DecisionTreeClassifier(max_depth=4, class_weight=cw, random_state=cfg.RANDOM_STATE),
        ),
        (
            "Random Forest",
            lambda cw: RandomForestClassifier(
                n_estimators=cfg.N_ESTIMATORS,
                class_weight=cw,
                n_jobs=cfg.RF_JOBS,
                random_state=cfg.RANDOM_STATE,
            ),
        ),
        (
            "KNN",
            lambda cw: KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=cfg.RF_JOBS),
        ),
    ]
    experiments = []
    for name, factory in specs:
        weight_options = [None, "balanced"] if name != "KNN" else ["nao_aplicavel"]
        for weight in weight_options:
            label = "class_weight=balanced" if weight == "balanced" else (
                "sem class_weight" if weight is None else "sem class_weight (nao suportado)"
            )
            experiments.append(
                {
                    "nome": name,
                    "peso": weight,
                    "rotulo_peso": label,
                    "pipeline": Pipeline(
                        steps=[
                            ("preprocessamento", build_preprocessor()),
                            ("modelo", factory(weight)),
                        ]
                    ),
                }
            )
    return experiments


def evaluate(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def pick_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    denom = precision + recall
    f1_values = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1_values))
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def cross_validate_experiment(exp: dict, train: pd.DataFrame) -> tuple[dict, np.ndarray]:
    X = train[cfg.FEATURES]
    y = train[cfg.TARGET].astype(int)
    groups = train[cfg.GROUP_COL]
    cv = StratifiedGroupKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)
    splits = list(cv.split(X, y, groups))

    oof = np.full(len(y), np.nan)
    for train_idx, valid_idx in splits:
        fold_pipeline = clone(exp["pipeline"])
        fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = fold_pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("Predicoes out-of-fold incompletas.")

    threshold, f1_oof = pick_threshold(y, oof)
    oof_metrics = evaluate(y.reset_index(drop=True), oof, threshold)

    fold_rows = []
    for _, valid_idx in splits:
        fold_metrics = evaluate(
            y.iloc[valid_idx].reset_index(drop=True),
            oof[valid_idx],
            threshold,
        )
        fold_rows.append(fold_metrics)

    row = {
        "modelo": exp["nome"],
        "peso_classes": str(exp["peso"]),
        "limiar": threshold,
        "f1_oof": f1_oof,
        "recall_oof": oof_metrics["recall"],
        "precisao_oof": oof_metrics["precisao"],
        "acuracia_oof": oof_metrics["acuracia"],
        "roc_auc_oof": oof_metrics["roc_auc"],
    }
    for metric in ["acuracia", "precisao", "recall", "f1", "roc_auc"]:
        values = np.array([fold[metric] for fold in fold_rows])
        row[f"{metric}_media"] = float(np.mean(values))
        row[f"{metric}_desvio"] = float(np.std(values))
    return row, oof


def choose_model(cv_df: pd.DataFrame) -> pd.Series:
    ranked = cv_df.sort_values(
        ["f1_oof", "recall_oof", "roc_auc_oof"], ascending=False
    ).reset_index(drop=True)
    return ranked.iloc[0]


def plot_confusion(matrix: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["previsto 0", "previsto 1"],
        yticklabels=["real 0", "real 1"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(cfg.CONFUSION_PATH, dpi=130)
    plt.close(fig)


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, linewidth=1.4, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="aleatorio")
    ax.set_xlabel("Falso positivo (1 - especificidade)")
    ax.set_ylabel("Recall (sensibilidade)")
    ax.set_title("Curvas ROC no conjunto de teste (2016)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(cfg.ROC_PATH, dpi=130)
    plt.close(fig)


def plot_precision_recall(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float, title: str
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    point = evaluate(y_true, y_proba, threshold)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, linewidth=1.6, color="#1f77b4")
    ax.axhline(y_true.mean(), linestyle="--", color="gray", linewidth=1,
               label=f"prevalencia no teste {y_true.mean():.2%}")
    ax.scatter(
        [point["recall"]],
        [point["precisao"]],
        color="#d62728",
        zorder=3,
        label=f"limiar {threshold:.2f}: P={point['precisao']:.2f}, R={point['recall']:.2f}, F1={point['f1']:.2f}",
    )
    ax.set_xlabel("Recall (classe mau=1)")
    ax.set_ylabel("Precisao (classe mau=1)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(cfg.PR_PATH, dpi=130)
    plt.close(fig)


def build_form_options(df: pd.DataFrame) -> dict:
    categorical = {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in cfg.CATEGORICAS}
    numeric = {}
    for col in cfg.NUMERICAS:
        series = df[col].dropna()
        numeric[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mediana": float(series.median()),
        }
    return {"categorias": categorical, "numericos": numeric}


def main() -> None:
    started = time.perf_counter()
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = validate_and_normalize(df)
    eda = run_eda(df)
    train, test = split_data(df)

    X_train = train[cfg.FEATURES]
    y_train = train[cfg.TARGET].astype(int)
    X_test = test[cfg.FEATURES]
    y_test = test[cfg.TARGET].astype(int)

    experiments = build_experiments()
    log(f"\n[treino] {len(experiments)} experimentos (4 algoritmos x cenarios de balanceamento)")
    metrics_rows: list[dict] = []
    cv_rows: list[dict] = []
    roc_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    fitted: dict[str, Pipeline] = {}

    for exp in experiments:
        tag = slugify(exp["nome"]) + "__" + slugify(str(exp["peso"]))

        t0 = time.perf_counter()
        cv_row, _ = cross_validate_experiment(exp, train)
        cv_seconds = time.perf_counter() - t0
        cv_rows.append(cv_row)

        t0 = time.perf_counter()
        exp["pipeline"].fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        fitted[tag] = exp["pipeline"]

        y_proba = exp["pipeline"].predict_proba(X_test)[:, 1]
        default_metrics = evaluate(y_test.reset_index(drop=True), y_proba, 0.5)
        row = {
            "modelo": exp["nome"],
            "peso_classes": str(exp["peso"]),
            "limiar": float(cv_row["limiar"]),
        }
        row.update(evaluate(y_test.reset_index(drop=True), y_proba, row["limiar"]))
        row["acuracia_limiar_0_5"] = default_metrics["acuracia"]
        row["precisao_limiar_0_5"] = default_metrics["precisao"]
        row["recall_limiar_0_5"] = default_metrics["recall"]
        row["f1_limiar_0_5"] = default_metrics["f1"]
        row["tempo_treino_s"] = round(fit_seconds, 2)
        row["tempo_cv_s"] = round(cv_seconds, 2)
        metrics_rows.append(row)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((f"{exp['nome']} ({exp['rotulo_peso']})", fpr, tpr, row["roc_auc"]))

        log(
            f"  {exp['nome']:<20} {exp['rotulo_peso']:<32} "
            f"limiar={row['limiar']:.2f} acuracia={row['acuracia']:.3f} "
            f"recall={row['recall']:.3f} precisao={row['precisao']:.3f} "
            f"f1={row['f1']:.3f} auc={row['roc_auc']:.3f} "
            f"| OOF f1={cv_row['f1_oof']:.3f} | {fit_seconds:.1f}s+{cv_seconds:.1f}s"
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["f1", "recall", "roc_auc"], ascending=False
    )
    cv_df = pd.DataFrame(cv_rows).sort_values(
        ["f1_oof", "recall_oof", "roc_auc_oof"], ascending=False
    )
    metrics_df.to_csv(cfg.METRICS_PATH, index=False, float_format="%.6f")
    cv_df.to_csv(cfg.CV_METRICS_PATH, index=False, float_format="%.6f")

    best = choose_model(cv_df)
    match = metrics_df[
        (metrics_df["modelo"] == best["modelo"])
        & (metrics_df["peso_classes"] == str(best["peso_classes"]))
    ]
    best_row = match.iloc[0]
    best_tag = slugify(best["modelo"]) + "__" + slugify(str(best["peso_classes"]))
    best_pipeline = fitted[best_tag]
    best_threshold = float(best["limiar"])
    joblib.dump(best_pipeline, cfg.BEST_MODEL_PATH)

    best_proba = best_pipeline.predict_proba(X_test)[:, 1]
    best_cm = confusion_matrix(
        y_test, (best_proba >= best_threshold).astype(int), labels=[0, 1]
    )
    plot_confusion(
        best_cm,
        f"Matriz de confusao - {best['modelo']} ({best['peso_classes']}, limiar {best_threshold:.2f})",
    )
    plot_roc(roc_curves)
    plot_precision_recall(
        y_test.reset_index(drop=True),
        best_proba,
        best_threshold,
        f"Precisao x recall no teste - {best['modelo']} ({best['peso_classes']})",
    )

    sweep_rows = []
    for threshold in np.round(np.arange(0.05, 1.0, 0.05), 2):
        sweep = evaluate(y_test.reset_index(drop=True), best_proba, float(threshold))
        sweep_rows.append(
            {
                "limiar": float(threshold),
                "precisao": sweep["precisao"],
                "recall": sweep["recall"],
                "f1": sweep["f1"],
                "acuracia": sweep["acuracia"],
                "positivos_previstos": float((best_proba >= threshold).mean()),
            }
        )
    pd.DataFrame(sweep_rows).to_csv(
        cfg.THRESHOLD_TABLE_PATH, index=False, float_format="%.6f"
    )

    sample = train.sample(min(cfg.ARCHIVE_SAMPLE_SIZE, len(train)), random_state=cfg.RANDOM_STATE)
    sample.to_csv(cfg.SAMPLE_PATH, index=False)
    options = build_form_options(train)
    cfg.FORM_OPTIONS_PATH.write_text(json.dumps(options, ensure_ascii=False, indent=2), encoding="utf-8")

    decision = (
        f"Modelo escolhido: {best['modelo']} (peso_classes={best['peso_classes']}, "
        f"limiar={best_threshold:.4f})\n"
        f"Criterio: maior F1 out-of-fold na validacao cruzada agrupada por cliente "
        f"(desempate por recall e depois ROC/AUC); o conjunto de teste 2016 nao foi usado na escolha.\n"
        f"Teste 2016: F1={best_row['f1']:.4f} | recall={best_row['recall']:.4f} | "
        f"precisao={best_row['precisao']:.4f} | AUC={best_row['roc_auc']:.4f} | "
        f"acuracia={best_row['acuracia']:.4f}\n"
        f"No limiar padrao 0.5 o F1 cairia para {best_row['f1_limiar_0_5']:.4f} e o recall para "
        f"{best_row['recall_limiar_0_5']:.4f}: o limiar 0.5 assume prevalencia de treino "
        f"({y_train.mean():.2%}) e nao a do teste ({y_test.mean():.2%}).\n"
        f"Baseline trivial (sempre bom pagador): acuracia ~{eda['baseline_accuracy']:.4f}, recall=0.\n"
        f"Matriz de confusao: TN={best_row['tn']}, FP={best_row['fp']}, "
        f"FN={best_row['fn']}, TP={best_row['tp']}.\n"
        f"Observacao: split temporal (2015 -> 2016) e taxa de inadimplencia crescente no tempo; "
        f"o recall prioriza reduzir falsos negativos (inadimplente classificado como bom pagador).\n"
        f"Limitacao: a discriminacao e modesta (AUC ~0.73); renda concentra quase todo o sinal "
        f"(AUC univariada ~0.74 dentro do mes). Nao usar como decisao real de credito.\n"
    )
    cfg.DECISION_PATH.write_text(decision, encoding="utf-8")

    if best_row["acuracia"] >= 0.99:
        log("\n[AVISO] Acuracia >= 99%: investigar possivel vazamento antes de aceitar o resultado.")

    metadata = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shape_original": [int(df.shape[0]), int(df.shape[1])],
        "colunas_originais": list(df.columns),
        "features": cfg.FEATURES,
        "numericas": cfg.NUMERICAS,
        "categoricas": cfg.CATEGORICAS,
        "ids_removidos": cfg.ID_COLS,
        "target": cfg.TARGET,
        "taxa_target": eda["target_rate"],
        "baseline_accuracy": eda["baseline_accuracy"],
        "drift_mensal": eda["drift_by_month"],
        "n_clientes": eda["n_clientes"],
        "duplicatas": eda["duplicatas"],
        "faltantes_tempo_emprego": {
            "taxa": eda["missing_tempo_emprego_rate"],
            "tratamento": "mediana + indicador de ausencia dentro do Pipeline (fit apenas no treino)",
        },
        "split": {
            "estrategia": "temporal",
            "test_start": cfg.TEST_START,
            "treino_periodo": [str(train["data_ref"].min()), str(train["data_ref"].max())],
            "teste_periodo": [str(test["data_ref"].min()), str(test["data_ref"].max())],
            "treino_linhas": int(len(train)),
            "teste_linhas": int(len(test)),
            "taxa_treino": float(y_train.mean()),
            "taxa_teste": float(y_test.mean()),
            "amostragem": "estratificada por mes x target",
            "random_state": cfg.RANDOM_STATE,
            "clientes_em_ambos_periodos": int(
                len(set(train[cfg.GROUP_COL]) & set(test[cfg.GROUP_COL]))
            ),
        },
        "modelo_escolhido": {
            "nome": best["modelo"],
            "peso_classes": str(best["peso_classes"]),
            "limiar": best_threshold,
            "criterio": (
                "maior F1 out-of-fold (10-fold agrupado por cliente); "
                "desempate por recall e ROC/AUC; teste 2016 reservado"
            ),
            "metricas": {
                key: float(best_row[key])
                for key in ["acuracia", "precisao", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
            },
            "metricas_limiar_0_5": {
                key: float(best_row[key])
                for key in ["acuracia_limiar_0_5", "precisao_limiar_0_5", "recall_limiar_0_5", "f1_limiar_0_5"]
            },
            "arquivo": cfg.BEST_MODEL_PATH.name,
        },
        "cv": {
            "estrategia": f"StratifiedGroupKFold {cfg.N_SPLITS}-fold (grupos = {cfg.GROUP_COL})",
            "amostra": int(len(train)),
            "limiar_calibrado_por": "F1 out-of-fold na validacao (sem uso do teste)",
        },
        "opcoes_formulario": options,
        "versoes": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "tempo_total_s": round(time.perf_counter() - started, 2),
        "artefatos": {
            "metricas": cfg.METRICS_PATH.name,
            "metricas_cv": cfg.CV_METRICS_PATH.name,
            "amostra": cfg.SAMPLE_PATH.name,
            "matriz_confusao": cfg.CONFUSION_PATH.name,
            "roc": cfg.ROC_PATH.name,
            "precisao_recall": cfg.PR_PATH.name,
            "limiares_teste": cfg.THRESHOLD_TABLE_PATH.name,
            "decisao": cfg.DECISION_PATH.name,
        },
    }
    cfg.METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n[resultado] tabela de metricas no teste (limiar calibrado no treino):")
    log(metrics_df.to_string(index=False))
    log("\n[resultado] validacao cruzada agrupada por cliente (criterio de escolha):")
    log(cv_df.to_string(index=False))
    log(f"\n[decisao] {best['modelo']} (peso={best['peso_classes']}, limiar={best_threshold:.2f})")
    log(f"[artefatos] {cfg.REPORTS_DIR} e {cfg.MODELS_DIR}")
    log(f"[tempo] total {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
