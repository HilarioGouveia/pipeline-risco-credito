import numpy as np
import pandas as pd
from src.config import GOLD_PARQUET, SILVER_PARQUET
from src.utils.io import load_parquet, save_parquet


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis lógicas de domínio de crédito de forma determinística linha a linha:
    1. comprometimento_renda = (loan_amnt / person_income) * 100 (obrigatória pelo desafio)
    2. score_estabilidade = person_emp_length * cb_person_cred_hist_length
    3. valor_por_ano_credito = loan_amnt / (cb_person_cred_hist_length + 1)
    """
    out = df.copy()

    # Garantia de denominador positivo seguro
    safe_income = out["person_income"].clip(lower=1.0)
    out["comprometimento_renda"] = (out["loan_amnt"] / safe_income) * 100.0

    # Obs.: caso person_emp_length ainda tenha NaN antes da imputação no split,
    # score_estabilidade preserva NaN e é recalculado após a imputação segura no pipeline.
    out["score_estabilidade"] = out["person_emp_length"] * out["cb_person_cred_hist_length"]
    out["valor_por_ano_credito"] = out["loan_amnt"] / (out["cb_person_cred_hist_length"] + 1.0)

    return out


def run_feature_engineering(df_silver: pd.DataFrame | None = None, persist: bool = True) -> dict:
    """
    Fase 3 (Silver -> Gold): Gera as features de negócio e persiste o dataset Gold.
    """
    if df_silver is None:
        df_silver = load_parquet(SILVER_PARQUET)

    df_gold = engineer_domain_features(df_silver)

    if persist:
        save_parquet(df_gold, GOLD_PARQUET)

    new_cols = ["comprometimento_renda", "score_estabilidade", "valor_por_ano_credito"]
    summary_stats = df_gold[new_cols].describe().T.round(4)

    corr_new = (
        df_gold[new_cols + ["loan_status"]]
        .corr()["loan_status"]
        .drop("loan_status")
        .round(4)
        .to_dict()
    )

    return {
        "df_gold": df_gold,
        "new_columns": new_cols,
        "summary_stats": summary_stats,
        "corr_with_target": corr_new,
    }
