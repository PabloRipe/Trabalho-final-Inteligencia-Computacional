"""Configuracoes compartilhadas entre o treino e o aplicativo Streamlit."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "credit_scoring.csv"
DATA_PATH_FALLBACK = BASE_DIR / "credit_scoring.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42

ID_COLS = ["Unnamed: 0", "index", "data_ref"]
TARGET = "mau"
GROUP_COL = "index"

NUMERICAS = [
    "qtd_filhos",
    "idade",
    "tempo_emprego",
    "qt_pessoas_residencia",
    "renda",
]

CATEGORICAS = [
    "sexo",
    "posse_de_veiculo",
    "posse_de_imovel",
    "tipo_renda",
    "educacao",
    "estado_civil",
    "tipo_residencia",
]

FEATURES = NUMERICAS + CATEGORICAS

TEST_START = "2016-01-01"
TRAIN_SAMPLE_SIZE = 50_000
TEST_SAMPLE_SIZE = 15_000

N_SPLITS = 10
N_ESTIMATORS = 300
RF_JOBS = -1

BASELINE_ACCURACY = 0.9218

ARCHIVE_SAMPLE_SIZE = 5_000

METRICS_PATH = REPORTS_DIR / "metricas.csv"
CV_METRICS_PATH = REPORTS_DIR / "metricas_cv.csv"
METADATA_PATH = REPORTS_DIR / "metadata.json"
FORM_OPTIONS_PATH = REPORTS_DIR / "opcoes_formulario.json"
SAMPLE_PATH = REPORTS_DIR / "dados_amostra.csv"
DECISION_PATH = REPORTS_DIR / "decisao_modelo.txt"
THRESHOLD_TABLE_PATH = REPORTS_DIR / "limiares_teste.csv"
CONFUSION_PATH = REPORTS_DIR / "matriz_confusao.png"
ROC_PATH = REPORTS_DIR / "roc.png"
PR_PATH = REPORTS_DIR / "precisao_recall.png"
EDA_TARGET_MONTH_PATH = REPORTS_DIR / "eda_target_mes.png"
EDA_MISSING_PATH = REPORTS_DIR / "eda_faltantes.png"
EDA_NUMERIC_PATH = REPORTS_DIR / "eda_numericas.png"
EDA_CATEGORICAL_PATH = REPORTS_DIR / "eda_categoricas.png"
EDA_CORR_PATH = REPORTS_DIR / "eda_correlacao.png"

BEST_MODEL_PATH = MODELS_DIR / "modelo.joblib"
