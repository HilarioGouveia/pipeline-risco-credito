import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def evaluate_binary_classifier(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    param_label: str,
) -> dict:
    """Calcula métricas completas no Treino e no Teste para diagnóstico de overfitting."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, pos_label=1)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)

    train_rec = recall_score(y_train, y_train_pred, pos_label=1)
    test_rec = recall_score(y_test, y_test_pred, pos_label=1)

    test_prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    roc_auc = np.nan
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_prob)

    return {
        "Modelo": model_name,
        "Configuracao": param_label,
        "Acc_Treino": round(train_acc, 4),
        "Acc_Teste": round(test_acc, 4),
        "Gap_Acc": round(train_acc - test_acc, 4),
        "Recall_Treino_C1": round(train_rec, 4),
        "Recall_Teste_C1": round(test_rec, 4),
        "Precision_Teste_C1": round(test_prec, 4),
        "F1_Treino_C1": round(train_f1, 4),
        "F1_Teste_C1": round(test_f1, 4),
        "Gap_F1": round(train_f1 - test_f1, 4),
        "ROC_AUC_Teste": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "y_test_pred": y_test_pred,
        "report_dict": classification_report(y_test, y_test_pred, output_dict=True),
        "report_text": classification_report(y_test, y_test_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred),
    }


def simulate_financial_impact(
    cm: np.ndarray,
    avg_loan: float = 9500.0,
    lgd: float = 0.65,
    net_interest_margin: float = 0.14,
) -> dict:
    """
    Estima o impacto financeiro a partir da Matriz de Confusão:
    - FN (Falso Negativo): cliente inadimplente aprovado -> perda direta = avg_loan * lgd
    - FP (Falso Positivo): bom pagador rejeitado -> custo de oportunidade = avg_loan * net_interest_margin
    """
    tn, fp, fn, tp = cm.ravel()
    perda_fn = fn * avg_loan * lgd
    custo_oportunidade_fp = fp * avg_loan * net_interest_margin
    custo_total = perda_fn + custo_oportunidade_fp

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Perda_Direta_FN": round(perda_fn, 2),
        "Custo_Oportunidade_FP": round(custo_oportunidade_fp, 2),
        "Impacto_Financeiro_Total": round(custo_total, 2),
    }
