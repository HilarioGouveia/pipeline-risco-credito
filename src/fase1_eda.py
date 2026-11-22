import pandas as pd
import numpy as np
from src.config import BRONZE_CSV, TARGET_COL
from src.utils.io import ensure_bronze_dataset, load_csv


def run_eda(df: pd.DataFrame | None = None) -> dict:
    """
    Fase 1: Análise Exploratória de Dados (Camada Bronze).
    Retorna dimensões, tipos, sumário estatístico, diagnósticos de qualidade e correlações.
    """
    if df is None:
        csv_path = ensure_bronze_dataset()
        df = load_csv(csv_path)

    n_rows, n_cols = df.shape
    dtypes_df = pd.DataFrame({
        "Coluna": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Nulos": df.isnull().sum().values,
        "% Nulos": (df.isnull().mean().values * 100).round(2),
        "Valores Únicos": df.nunique().values,
    })

    describe_num = df.describe().T.round(4)
    describe_cat = df.describe(include=["object"]).T

    n_duplicates = int(df.duplicated().sum())
    target_counts = df[TARGET_COL].value_counts().to_dict()
    target_pct = (df[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict()

    anomalies = {
        "idades_acima_80": int((df["person_age"] > 80).sum()),
        "valores_idades_acima_80": sorted(df.loc[df["person_age"] > 80, "person_age"].unique().tolist()),
        "tempo_emprego_inconsistente": int((df["person_emp_length"] > (df["person_age"] - 14)).sum()),
        "tempo_emprego_max": float(df["person_emp_length"].max()),
        "renda_maxima": float(df["person_income"].max()),
    }

    corr_with_target = (
        df.select_dtypes(include=[np.number])
        .corr()[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(ascending=False)
        .round(4)
    )

    executive_summary = (
        f"A base bruta (Camada Bronze) possui {n_rows:,} registros e {n_cols} variáveis. "
        f"Observa-se desbalanceamento moderado na variável alvo ({target_pct.get(0, 0):.2f}% adimplentes vs. "
        f"{target_pct.get(1, 0):.2f}% inadimplentes), {n_duplicates} linhas duplicadas, valores ausentes em "
        f"'loan_int_rate' ({df['loan_int_rate'].isnull().sum():,}) e 'person_emp_length' ({df['person_emp_length'].isnull().sum()}), "
        f"além de anomalias severas (idades até {int(df['person_age'].max())} anos, tempo de emprego de "
        f"{int(df['person_emp_length'].max())} anos e renda altamente assimétrica até US$ {df['person_income'].max():,.0f}). "
        f"As variáveis numéricas com maior associação linear à inadimplência são 'loan_percent_income' (r={corr_with_target['loan_percent_income']:.2f}) "
        f"e 'loan_int_rate' (r={corr_with_target['loan_int_rate']:.2f}). Esses achados exigem deduplicação, tratamento de inconsistências, "
        f"imputação/clipping ajustados sem vazamento de dados, balanceamento exclusivo no treino e escalonamento exclusivo para o KNN."
    )

    return {
        "df": df,
        "shape": (n_rows, n_cols),
        "dtypes_df": dtypes_df,
        "describe_num": describe_num,
        "describe_cat": describe_cat,
        "n_duplicates": n_duplicates,
        "target_counts": target_counts,
        "target_pct": target_pct,
        "anomalies": anomalies,
        "corr_with_target": corr_with_target,
        "executive_summary": executive_summary,
    }
