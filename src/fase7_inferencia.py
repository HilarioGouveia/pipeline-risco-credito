from pathlib import Path
import joblib
import pandas as pd
from src.config import ARTIFACTS_DIR, CATEGORICAL_COLS, NUMERIC_COLS

BUNDLE_PATH = ARTIFACTS_DIR / "inference_bundle.joblib"


def load_inference_bundle(bundle_path: Path = BUNDLE_PATH) -> dict:
    """Carrega o bundle persistido contendo pré-processador, scaler, melhor KNN e melhor Árvore."""
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle de inferência não encontrado em {bundle_path}. Execute a Fase 6 ou o pipeline primeiro."
        )
    return joblib.load(bundle_path)


def predict_single_applicant(applicant_data: dict, bundle: dict) -> dict:
    """
    Executa inferência ponta a ponta para um novo solicitante usando:
    - CreditRiskPreprocessor (imputação, IQR clipping, features derivadas e OneHotEncoder)
    - Árvore de Decisão (sem escalonamento)
    - KNN (com StandardScaler nas variáveis contínuas)
    """
    preprocessor = bundle["preprocessor"]
    scaler = bundle["scaler"]
    best_tree = bundle["best_tree_model"]
    best_knn = bundle["best_knn_model"]

    if preprocessor is None or scaler is None:
        raise ValueError("Pré-processador ou StandardScaler ausentes no bundle persistido.")

    # Garante consistência determinística entre loan_percent_income e renda/valor
    person_income = float(applicant_data["person_income"])
    loan_amnt = float(applicant_data["loan_amnt"])
    loan_percent_income = round(loan_amnt / max(person_income, 1.0), 4)

    raw_row = {
        "person_age": int(applicant_data["person_age"]),
        "person_income": person_income,
        "person_home_ownership": str(applicant_data["person_home_ownership"]),
        "person_emp_length": float(applicant_data["person_emp_length"]),
        "loan_intent": str(applicant_data["loan_intent"]),
        "loan_grade": str(applicant_data["loan_grade"]),
        "loan_amnt": loan_amnt,
        "loan_int_rate": float(applicant_data["loan_int_rate"]),
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": str(applicant_data["cb_person_default_on_file"]),
        "cb_person_cred_hist_length": int(applicant_data["cb_person_cred_hist_length"]),
    }

    df_input = pd.DataFrame([raw_row])[NUMERIC_COLS + CATEGORICAL_COLS]

    # Transformação para Árvore (sem StandardScaler)
    X_tree = preprocessor.transform(df_input)

    # Transformação para KNN (com StandardScaler nas contínuas)
    X_knn = X_tree.copy()
    cont_cols = preprocessor.continuous_feature_names_
    X_knn[cont_cols] = scaler.transform(X_knn[cont_cols])

    # Predições Árvore de Decisão
    tree_pred = int(best_tree.predict(X_tree)[0])
    tree_prob_default = (
        float(best_tree.predict_proba(X_tree)[0][1])
        if hasattr(best_tree, "predict_proba")
        else float(tree_pred)
    )

    # Predições KNN
    knn_pred = int(best_knn.predict(X_knn)[0])
    knn_prob_default = (
        float(best_knn.predict_proba(X_knn)[0][1])
        if hasattr(best_knn, "predict_proba")
        else float(knn_pred)
    )

    # Extrai features engenheiradas calculadas no transform
    engineered_values = {
        "comprometimento_renda_pct": round(float(X_tree["comprometimento_renda"].iloc[0]), 2),
        "score_estabilidade": round(float(X_tree["score_estabilidade"].iloc[0]), 2),
        "valor_por_ano_credito": round(float(X_tree["valor_por_ano_credito"].iloc[0]), 2),
        "renda_apos_clipping": round(float(X_tree["person_income"].iloc[0]), 2),
        "sofreu_clipping_renda": float(X_tree["person_income"].iloc[0]) != person_income,
    }

    return {
        "tree": {
            "config": bundle["best_tree_config"],
            "prediction": tree_pred,
            "label": "Risco de Inadimplência (Calote)" if tree_pred == 1 else "Crédito Seguro (Adimplente)",
            "prob_default_pct": round(tree_prob_default * 100, 2),
        },
        "knn": {
            "config": bundle["best_knn_config"],
            "prediction": knn_pred,
            "label": "Risco de Inadimplência (Calote)" if knn_pred == 1 else "Crédito Seguro (Adimplente)",
            "prob_default_pct": round(knn_prob_default * 100, 2),
        },
        "recommended_key": bundle["recommended_model_key"],
        "recommended_name": bundle["recommended_model_name"],
        "engineered_values": engineered_values,
    }
