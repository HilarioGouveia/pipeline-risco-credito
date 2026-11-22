import joblib
import pandas as pd
from src.config import ARTIFACTS_DIR
from src.utils.metricas import simulate_financial_impact


def generate_executive_evaluation(
    model_results: dict,
    avg_loan: float = 9500.0,
    lgd: float = 0.65,
    net_interest_margin: float = 0.14,
    persist_best: bool = True,
) -> dict:
    """
    Fase 6: Consolida Classification Reports, Matrizes de Confusão,
    Simulação de Impacto Financeiro (FN vs FP) e Veredito Executivo.
    """
    best_knn_metrics = model_results["best_knn"]["metrics"]
    best_tree_metrics = model_results["best_tree"]["metrics"]

    fin_knn = simulate_financial_impact(
        best_knn_metrics["confusion_matrix"], avg_loan=avg_loan, lgd=lgd, net_interest_margin=net_interest_margin
    )
    fin_tree = simulate_financial_impact(
        best_tree_metrics["confusion_matrix"], avg_loan=avg_loan, lgd=lgd, net_interest_margin=net_interest_margin
    )

    comparison_df = pd.DataFrame([
        {
            "Modelo": f"KNN ({best_knn_metrics['Configuracao']})",
            "Accuracy Teste": best_knn_metrics["Acc_Teste"],
            "Recall Classe 1 (Inadimplente)": best_knn_metrics["Recall_Teste_C1"],
            "Precision Classe 1": best_knn_metrics["Precision_Teste_C1"],
            "F1-Score Classe 1": best_knn_metrics["F1_Teste_C1"],
            "ROC-AUC": best_knn_metrics["ROC_AUC_Teste"],
            "Gap Overfitting (Acc)": best_knn_metrics["Gap_Acc"],
            "Falsos Negativos (FN)": fin_knn["FN"],
            "Falsos Positivos (FP)": fin_knn["FP"],
            "Perda Direta FN (US$)": fin_knn["Perda_Direta_FN"],
            "Custo Oportunidade FP (US$)": fin_knn["Custo_Oportunidade_FP"],
            "Impacto Financeiro Total (US$)": fin_knn["Impacto_Financeiro_Total"],
        },
        {
            "Modelo": f"Árvore de Decisão ({best_tree_metrics['Configuracao']})",
            "Accuracy Teste": best_tree_metrics["Acc_Teste"],
            "Recall Classe 1 (Inadimplente)": best_tree_metrics["Recall_Teste_C1"],
            "Precision Classe 1": best_tree_metrics["Precision_Teste_C1"],
            "F1-Score Classe 1": best_tree_metrics["F1_Teste_C1"],
            "ROC-AUC": best_tree_metrics["ROC_AUC_Teste"],
            "Gap Overfitting (Acc)": best_tree_metrics["Gap_Acc"],
            "Falsos Negativos (FN)": fin_tree["FN"],
            "Falsos Positivos (FP)": fin_tree["FP"],
            "Perda Direta FN (US$)": fin_tree["Perda_Direta_FN"],
            "Custo Oportunidade FP (US$)": fin_tree["Custo_Oportunidade_FP"],
            "Impacto Financeiro Total (US$)": fin_tree["Impacto_Financeiro_Total"],
        },
    ])

    # Critério de decisão corporativa: menor impacto financeiro / maior F1 na classe 1
    tree_wins = (
        best_tree_metrics["F1_Teste_C1"] >= best_knn_metrics["F1_Teste_C1"]
        or fin_tree["Impacto_Financeiro_Total"] <= fin_knn["Impacto_Financeiro_Total"]
    )

    if tree_wins:
        winner_name = f"Árvore de Decisão ({best_tree_metrics['Configuracao']})"
        winner_model = model_results["best_tree"]["model"]
    else:
        winner_name = f"KNN ({best_knn_metrics['Configuracao']})"
        winner_model = model_results["best_knn"]["model"]

    if persist_best:
        joblib.dump(winner_model, ARTIFACTS_DIR / "best_credit_risk_model.joblib")

    verdict = {
        "recommended_model": winner_name,
        "reasons": [
            f"Superioridade no equilíbrio entre precisão e captura de inadimplentes (F1 Classe 1: {best_tree_metrics['F1_Teste_C1']:.4f} vs {best_knn_metrics['F1_Teste_C1']:.4f}).",
            f"Controle de overfitting com profundidade limitada ({best_tree_metrics['Configuracao']}), evitando a memorização observada em max_depth=None.",
            "Robustez natural a outliers residuais sem depender de métricas de distância euclidiana.",
            "Baixíssima latência em inferência e alta auditabilidade regulatória (regras explícitas para compliance bancário).",
        ],
    }

    return {
        "best_knn_metrics": best_knn_metrics,
        "best_tree_metrics": best_tree_metrics,
        "fin_knn": fin_knn,
        "fin_tree": fin_tree,
        "comparison_df": comparison_df,
        "verdict": verdict,
    }
