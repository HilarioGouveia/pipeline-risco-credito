import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    CATEGORICAL_COLS,
    GOLD_PARQUET,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from src.utils.io import load_parquet


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformador stateful sem data leakage (fit apenas no Treino):
    1. Imputa nulos numéricos com a Mediana do Treino ('person_emp_length', 'loan_int_rate').
    2. Aplica IQR Clipping na 'person_income' usando limites calculados apenas no Treino.
    3. Recalcula as features derivadas ('comprometimento_renda', 'score_estabilidade', 'valor_por_ano_credito')
       após a imputação/clipping para garantir zero NaNs ou infinitos.
    4. Codifica variáveis categóricas com OneHotEncoder(handle_unknown='ignore').
    """

    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = iqr_factor

    def fit(self, X: pd.DataFrame, y=None):
        self.num_imputer_ = SimpleImputer(strategy="median")
        self.num_imputer_.fit(X[NUMERIC_COLS])

        q1 = X["person_income"].quantile(0.25)
        q3 = X["person_income"].quantile(0.75)
        iqr = q3 - q1
        self.income_lower_ = max(1.0, float(q1 - self.iqr_factor * iqr))
        self.income_upper_ = float(q3 + self.iqr_factor * iqr)

        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe_.fit(X[CATEGORICAL_COLS])

        self.cat_feature_names_ = list(self.ohe_.get_feature_names_out(CATEGORICAL_COLS))
        self.continuous_feature_names_ = NUMERIC_COLS + [
            "comprometimento_renda",
            "score_estabilidade",
            "valor_por_ano_credito",
        ]
        self.all_feature_names_ = self.continuous_feature_names_ + self.cat_feature_names_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Imputação pela mediana aprendida no treino
        df[NUMERIC_COLS] = self.num_imputer_.transform(df[NUMERIC_COLS])

        # 2. IQR Clipping na renda (limites aprendidos no treino)
        df["person_income"] = df["person_income"].clip(
            lower=self.income_lower_, upper=self.income_upper_
        )

        # 3. Engenharia de atributos pós-imputação (sem NaNs ou infinitos)
        df["comprometimento_renda"] = (df["loan_amnt"] / df["person_income"]) * 100.0
        df["score_estabilidade"] = df["person_emp_length"] * df["cb_person_cred_hist_length"]
        df["valor_por_ano_credito"] = df["loan_amnt"] / (df["cb_person_cred_hist_length"] + 1.0)

        # 4. One-Hot Encoding
        cat_encoded = self.ohe_.transform(df[CATEGORICAL_COLS])
        cat_df = pd.DataFrame(cat_encoded, columns=self.cat_feature_names_, index=df.index)

        out = pd.concat([df[self.continuous_feature_names_], cat_df], axis=1)
        return out


def prepare_modeling_datasets(df_gold: pd.DataFrame | None = None) -> dict:
    """
    Fase 4: Split estratificado (80/20), Pré-processamento sem leakage,
    Balanceamento (SMOTE apenas no Treino) e StandardScaler (apenas para o KNN).
    """
    if df_gold is None:
        df_gold = load_parquet(GOLD_PARQUET)

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df_gold[feature_cols].copy()
    y = df_gold[TARGET_COL].astype(int).copy()

    # 1. Split estratificado antes de qualquer transformação estatística
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 2. Fit do pré-processador apenas no Treino
    preprocessor = CreditRiskPreprocessor(iqr_factor=1.5)
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # 3. Balanceamento SMOTE exclusivamente no conjunto de Treino
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
    X_train_bal = pd.DataFrame(X_train_bal, columns=preprocessor.all_feature_names_)

    # 4. StandardScaler exclusivo para o KNN (apenas nas variáveis contínuas)
    cont_cols = preprocessor.continuous_feature_names_
    scaler = StandardScaler()

    X_train_knn = X_train_bal.copy()
    X_test_knn = X_test_proc.copy()

    X_train_knn[cont_cols] = scaler.fit_transform(X_train_bal[cont_cols])
    X_test_knn[cont_cols] = scaler.transform(X_test_proc[cont_cols])

    # Árvore de Decisão usa os dados balanceados SEM escalonamento
    X_train_tree = X_train_bal.copy()
    X_test_tree = X_test_proc.copy()

    return {
        "preprocessor": preprocessor,
        "scaler": scaler,
        "y_train_before_smote": y_train,
        "y_train_bal": pd.Series(y_train_bal, name=TARGET_COL),
        "y_test": y_test,
        "X_train_knn": X_train_knn,
        "X_test_knn": X_test_knn,
        "X_train_tree": X_train_tree,
        "X_test_tree": X_test_tree,
        "income_clip_bounds": (preprocessor.income_lower_, preprocessor.income_upper_),
        "n_features_after_encoding": X_train_bal.shape[1],
    }
