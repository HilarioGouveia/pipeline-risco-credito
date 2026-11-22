import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", palette="muted")


def plot_target_distribution(df: pd.DataFrame, target_col: str = "loan_status"):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[target_col].value_counts().sort_index()
    pcts = df[target_col].value_counts(normalize=True).sort_index() * 100

    bars = ax.bar(
        ["0 - Adimplente (Pago)", "1 - Inadimplente (Calote)"],
        counts.values,
        color=["#2E6F95", "#C84B31"],
        width=0.55,
    )
    ax.set_title("Distribuição da Variável Alvo (loan_status)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Quantidade de Clientes")
    ax.set_ylim(0, counts.max() * 1.18)

    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.02,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    return fig


def plot_numeric_histogram(df: pd.DataFrame, col: str, title: str, clip_upper: float | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    series = df[col].dropna()
    if clip_upper is not None:
        series = series[series <= clip_upper]

    sns.histplot(series, bins=40, kde=True, color="#2E6F95", ax=ax)
    ax.axvline(series.median(), color="#C84B31", linestyle="--", linewidth=1.8, label=f"Mediana: {series.median():,.2f}")
    ax.axvline(series.mean(), color="#2D6A4F", linestyle=":", linewidth=1.8, label=f"Média: {series.mean():,.2f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequência")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Mapa de Calor — Correlação de Pearson", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_boxplots_outliers(df: pd.DataFrame, cols: list[str]):
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#4A90A4")
        ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_smote_comparison(y_before: pd.Series, y_after: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, y_data, title in zip(
        axes,
        [y_before, y_after],
        ["Treino Antes do SMOTE", "Treino Depois do SMOTE"],
    ):
        counts = pd.Series(y_data).value_counts().sort_index()
        pcts = pd.Series(y_data).value_counts(normalize=True).sort_index() * 100
        bars = ax.bar(["Classe 0", "Classe 1"], counts.values, color=["#2E6F95", "#C84B31"], width=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.22)
        for bar, count, pct in zip(bars, counts.values, pcts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.02,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
    fig.tight_layout()
    return fig


def plot_overfitting_curve(results_df: pd.DataFrame, model_type: str):
    subset = results_df[results_df["Modelo"] == model_type].copy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(subset))

    ax.plot(x, subset["Acc_Treino"], marker="o", linewidth=2, label="Accuracy Treino", color="#1D3557")
    ax.plot(x, subset["Acc_Teste"], marker="s", linewidth=2, label="Accuracy Teste", color="#457B9D")
    ax.plot(x, subset["F1_Treino_C1"], marker="^", linestyle="--", linewidth=1.8, label="F1 (Classe 1) Treino", color="#E63946")
    ax.plot(x, subset["F1_Teste_C1"], marker="d", linestyle="--", linewidth=1.8, label="F1 (Classe 1) Teste", color="#F4A261")

    ax.set_xticks(x)
    ax.set_xticklabels(subset["Configuracao"])
    ax.set_title(f"Diagnóstico de Overfitting — {model_type}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Complexidade / Hiperparâmetro")
    ax.set_ylabel("Score")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_figure(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Previsto: 0 (Seguro)", "Previsto: 1 (Calote)"],
        yticklabels=["Real: 0 (Seguro)", "Real: 1 (Calote)"],
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig
