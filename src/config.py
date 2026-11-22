from pathlib import Path

# Diretórios base
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Caminhos dos arquivos Medallion
BRONZE_CSV = BRONZE_DIR / "credit_risk_dataset.csv"
SILVER_PARQUET = SILVER_DIR / "silver_credit_risk.parquet"
GOLD_PARQUET = GOLD_DIR / "gold_credit_risk.parquet"

# Garantir existência das pastas
for folder in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, ARTIFACTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Parâmetros globais e de reprodutibilidade
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "loan_status"

# Definição de colunas
CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

ENGINEERED_COLS = [
    "comprometimento_renda",
    "score_estabilidade",
    "valor_por_ano_credito",
]

# Grades de hiperparâmetros (Fase 5)
KNN_K_VALUES = [3, 5, 7, 9]
TREE_MAX_DEPTHS = [3, 5, 7, None]
