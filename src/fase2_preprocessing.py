import numpy as np
import pandas as pd
from src.config import BRONZE_CSV, SILVER_PARQUET
from src.utils.io import ensure_bronze_dataset, load_csv, save_parquet


def run_structural_cleaning(df: pd.DataFrame | None = None, persist: bool = True) -> dict:
    """
    Fase 2 (Bronze -> Silver): Limpeza determinística e sanitização estrutural sem data leakage.
    Nota metodológica:
    - Regras puramente determinísticas (remoção de duplicatas exatas, filtro de idade > 80 anos e
      anulação de tempo de emprego fisicamente impossível > idade - 14) são aplicadas antes do split.
    - Estatísticas aprendidas dos dados (medianas para imputação de nulos e limites IQR para clipping de renda)
      são ajustadas exclusivamente sobre o conjunto de Treino na Fase 4 para impedir data leakage.
    """
    if df is None:
        csv_path = ensure_bronze_dataset()
        df = load_csv(csv_path)

    initial_rows = len(df)

    # 1. Remoção de duplicadas
    duplicates_removed = int(df.duplicated().sum())
    df_clean = df.drop_duplicates().copy()

    # 2. Remoção de idades biologicamente inconsistentes (> 80 anos)
    invalid_age_removed = int((df_clean["person_age"] > 80).sum())
    df_clean = df_clean[df_clean["person_age"] <= 80].copy()

    # 3. Sanitização de tempo de emprego impossível (person_emp_length > person_age - 14)
    inconsistent_emp_mask = df_clean["person_emp_length"] > (df_clean["person_age"] - 14)
    inconsistent_emp_nulled = int(inconsistent_emp_mask.sum())
    df_clean.loc[inconsistent_emp_mask, "person_emp_length"] = np.nan

    # Padronização de tipos
    df_clean["person_age"] = df_clean["person_age"].astype(int)
    df_clean["person_income"] = df_clean["person_income"].astype(float)
    df_clean["loan_amnt"] = df_clean["loan_amnt"].astype(float)
    df_clean["loan_status"] = df_clean["loan_status"].astype(int)

    df_clean.reset_index(drop=True, inplace=True)

    if persist:
        save_parquet(df_clean, SILVER_PARQUET)

    report = {
        "df_silver": df_clean,
        "initial_rows": initial_rows,
        "final_rows": len(df_clean),
        "duplicates_removed": duplicates_removed,
        "invalid_age_removed": invalid_age_removed,
        "inconsistent_emp_nulled": inconsistent_emp_nulled,
        "nulls_after_silver": df_clean.isnull().sum().to_dict(),
        "justifications": {
            "duplicatas": f"Removidas {duplicates_removed} linhas duplicadas exatas para evitar viés de ponderação na modelagem.",
            "idade": f"Removidos {invalid_age_removed} registros com person_age > 80 anos (ex.: 84, 94, 123 e 144 anos), erros evidentes de cadastro.",
            "tempo_emprego": (
                f"Convertidos {inconsistent_emp_nulled} registros onde person_emp_length > (person_age - 14) para NaN, "
                "a serem imputados pela Mediana ajustada apenas no Treino (robusta a assimetria e outliers)."
            ),
            "taxa_juros_e_renda": (
                "Os nulos de 'loan_int_rate' serão imputados pela Mediana do Treino (distribuição assimétrica) e "
                "os extremos de 'person_income' sofrerão IQR Clipping ajustado apenas no Treino."
            ),
        },
    }
    return report
