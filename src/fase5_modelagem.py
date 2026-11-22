import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import KNN_K_VALUES, RANDOM_STATE, TREE_MAX_DEPTHS
from src.utils.metricas import evaluate_binary_classifier


def train_and_benchmark_models(split_data: dict) -> dict:
    """
    Fase 5: Treina e compara variações de complexidade para KNN e Árvore de Decisão,
    monitorando métricas de Treino e Teste para diagnóstico de overfitting.
    """
    X_train_knn = split_data["X_train_knn"]
    X_test_knn = split_data["X_test_knn"]
    X_train_tree = split_data["X_train_tree"]
    X_test_tree = split_data["X_test_tree"]
    y_train_bal = split_data["y_train_bal"]
    y_test = split_data["y_test"]

    all_results = []
    trained_models = {"KNN": {}, "Árvore de Decisão": {}}

    # 1. Experimentos KNN (com dados escalonados)
    for k in KNN_K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_knn, y_train_bal)
        label = f"K={k}"
        res = evaluate_binary_classifier(
            knn, X_train_knn, y_train_bal, X_test_knn, y_test, "KNN", label
        )
        all_results.append(res)
        trained_models["KNN"][label] = {"model": knn, "metrics": res}

    # 2. Experimentos Árvore de Decisão (sem escalonamento)
    for depth in TREE_MAX_DEPTHS:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        tree.fit(X_train_tree, y_train_bal)
        label = f"max_depth={depth}"
        res = evaluate_binary_classifier(
            tree, X_train_tree, y_train_bal, X_test_tree, y_test, "Árvore de Decisão", label
        )
        all_results.append(res)
        trained_models["Árvore de Decisão"][label] = {"model": tree, "metrics": res}

    summary_cols = [
        "Modelo",
        "Configuracao",
        "Acc_Treino",
        "Acc_Teste",
        "Gap_Acc",
        "Recall_Treino_C1",
        "Recall_Teste_C1",
        "Precision_Teste_C1",
        "F1_Treino_C1",
        "F1_Teste_C1",
        "Gap_F1",
        "ROC_AUC_Teste",
    ]
    results_df = pd.DataFrame([{c: r[c] for c in summary_cols} for r in all_results])

    # Seleção do melhor modelo de cada família pelo maior F1_Teste_C1 (equilíbrio Recall/Precision na classe 1)
    best_knn_row = (
        results_df[results_df["Modelo"] == "KNN"]
        .sort_values(by=["F1_Teste_C1", "Recall_Teste_C1"], ascending=False)
        .iloc[0]
    )
    best_tree_row = (
        results_df[results_df["Modelo"] == "Árvore de Decisão"]
        .sort_values(by=["F1_Teste_C1", "Recall_Teste_C1"], ascending=False)
        .iloc[0]
    )

    best_knn_label = best_knn_row["Configuracao"]
    best_tree_label = best_tree_row["Configuracao"]

    return {
        "results_df": results_df,
        "trained_models": trained_models,
        "best_knn": trained_models["KNN"][best_knn_label],
        "best_tree": trained_models["Árvore de Decisão"][best_tree_label],
    }
