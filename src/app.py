import streamlit as st
import pandas as pd

from src.fase1_eda import run_eda
from src.fase2_preprocessing import run_structural_cleaning
from src.fase3_feature_engineering import run_feature_engineering
from src.fase4_data_split import prepare_modeling_datasets
from src.fase5_modelagem import train_and_benchmark_models
from src.fase6_avaliacao import generate_executive_evaluation
from src.utils.graficos import (
    plot_boxplots_outliers,
    plot_confusion_matrix_figure,
    plot_correlation_heatmap,
    plot_numeric_histogram,
    plot_overfitting_curve,
    plot_smote_comparison,
    plot_target_distribution,
)

st.set_page_config(
    page_title="Risco de Crédito — Pipeline Preditivo",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource(show_spinner="Executando pipeline Medallion (Bronze -> Silver -> Gold -> ML)...")
def load_pipeline_artifacts():
    eda = run_eda()
    silver = run_structural_cleaning(eda["df"], persist=True)
    gold = run_feature_engineering(silver["df_silver"], persist=True)
    split_data = prepare_modeling_datasets(gold["df_gold"])
    modeling = train_and_benchmark_models(split_data)
    return eda, silver, gold, split_data, modeling


st.sidebar.title("🏦 Risco de Crédito")
st.sidebar.caption("Arquitetura Medallion • Scikit-Learn • UV")

fase = st.sidebar.radio(
    "Navegue pelas Fases do Projeto:",
    [
        "Fase 1 - EDA (Bronze)",
        "Fase 2 - Data Prep (Silver)",
        "Fase 3 - Feature Engineering (Gold)",
        "Fase 4 - Split, SMOTE e Escala",
        "Fase 5 - Modelagem e Overfitting",
        "Fase 6 - Avaliação e Veredito",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Fluxo Anti-Leakage:**\n"
    "1. Sanitização estrutural (Silver)\n"
    "2. Split estratificado 80/20\n"
    "3. Mediana, IQR Clipping & SMOTE apenas no Treino\n"
    "4. StandardScaler exclusivo para KNN"
)

try:
    eda, silver, gold, split_data, modeling = load_pipeline_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ==============================================================================
# FASE 1: EDA
# ==============================================================================
if fase == "Fase 1 - EDA (Bronze)":
    st.title("Fase 1: Análise Exploratória de Dados (Camada Bronze)")
    tab1, tab2, tab3, tab4 = st.tabs(["Dimensões e Tipos", "Sumário Estatístico", "Gráficos Exploratórios", "Diagnóstico Analítico"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de Registros", f"{eda['shape'][0]:,}")
        c2.metric("Variáveis (Colunas)", eda["shape"][1])
        c3.metric("Linhas Duplicadas", eda["n_duplicates"])
        c4.metric("Taxa Inadimplência", f"{eda['target_pct'][1]:.2f}%")
        st.subheader("Metadados e Valores Ausentes")
        st.dataframe(eda["dtypes_df"], use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Estatística Descritiva — Variáveis Numéricas")
        st.dataframe(eda["describe_num"], use_container_width=True)
        st.subheader("Estatística Descritiva — Variáveis Categóricas")
        st.dataframe(eda["describe_cat"], use_container_width=True)

    with tab3:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.pyplot(plot_target_distribution(eda["df"]))
        with col_g2:
            st.pyplot(plot_numeric_histogram(eda["df"], "person_income", "Distribuição de Renda (até US$ 250k)", clip_upper=250000))

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            st.pyplot(plot_numeric_histogram(eda["df"], "person_age", "Distribuição de Idade dos Solicitantes"))
        with col_g4:
            st.pyplot(plot_correlation_heatmap(eda["df"]))

    with tab4:
        st.subheader("Síntese Executiva da Exploração")
        st.success(eda["executive_summary"])
        st.subheader("Anomalias Identificadas na Base Bruta")
        st.json(eda["anomalies"])

# ==============================================================================
# FASE 2: DATA PREP
# ==============================================================================
elif fase == "Fase 2 - Data Prep (Silver)":
    st.title("Fase 2: Tratamento, Limpeza e Sanitização (Camada Silver)")
    tab1, tab2, tab3, tab4 = st.tabs(["Duplicados e Inconsistências", "Nulos e Justificativas", "Outliers (Boxplots)", "Dataset Silver"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros Bronze", f"{silver['initial_rows']:,}")
        c2.metric("Duplicados Removidos", f"-{silver['duplicates_removed']}")
        c3.metric("Idades > 80 Removidas", f"-{silver['invalid_age_removed']}")
        c4.metric("Registros Silver", f"{silver['final_rows']:,}")

        for titulo, texto in silver["justifications"].items():
            st.markdown(f"**{titulo.replace('_', ' ').title()}:** {texto}")

    with tab2:
        st.subheader("Contagem de Nulos na Camada Silver (Pré-Split)")
        nulls_df = pd.DataFrame(
            list(silver["nulls_after_silver"].items()),
            columns=["Coluna", "Nulos Remanescentes"],
        )
        st.dataframe(nulls_df, use_container_width=True, hide_index=True)
        st.info(
            "Para evitar Data Leakage, a imputação pela Mediana ('person_emp_length' e 'loan_int_rate') "
            "e o IQR Clipping de 'person_income' são ajustados exclusivamente sobre a partição de Treino na Fase 4."
        )

    with tab3:
        st.subheader("Inspeção Visual de Outliers")
        st.pyplot(plot_boxplots_outliers(silver["df_silver"], ["person_age", "person_emp_length", "person_income", "loan_amnt"]))

    with tab4:
        st.subheader("Amostra da Camada Silver (`data/silver/silver_credit_risk.parquet`)")
        st.dataframe(silver["df_silver"].head(50), use_container_width=True)

# ==============================================================================
# FASE 3: FEATURE ENGINEERING
# ==============================================================================
elif fase == "Fase 3 - Feature Engineering (Gold)":
    st.title("Fase 3: Engenharia de Atributos (Camada Gold)")
    tab1, tab2, tab3 = st.tabs(["Variáveis Criadas", "Correlação com Alvo", "Dataset Gold"])

    with tab1:
        st.markdown(
            "- **`comprometimento_renda` (Obrigatória):** `(loan_amnt / person_income) * 100` — percentual da renda comprometido pelo empréstimo.\n"
            "- **`score_estabilidade`:** `person_emp_length * cb_person_cred_hist_length` — indicador combinado de estabilidade profissional e histórico de crédito.\n"
            "- **`valor_por_ano_credito`:** `loan_amnt / (cb_person_cred_hist_length + 1)` — intensidade do valor solicitado em relação à maturidade creditícia."
        )
        st.dataframe(gold["summary_stats"], use_container_width=True)

    with tab2:
        corr_df = pd.DataFrame(
            list(gold["corr_with_target"].items()),
            columns=["Feature Criada", "Correlação com loan_status"],
        )
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Amostra da Camada Gold (`data/gold/gold_credit_risk.parquet`)")
        st.dataframe(gold["df_gold"].head(50), use_container_width=True)

# ==============================================================================
# FASE 4: SPLIT, BALANCEAMENTO E ESCALONAMENTO
# ==============================================================================
elif fase == "Fase 4 - Split, SMOTE e Escala":
    st.title("Fase 4: Separação Estratificada, SMOTE e Escalonamento Seguro")
    tab1, tab2, tab3 = st.tabs(["Split e Encoding", "Balanceamento (SMOTE)", "Escalonamento (KNN vs Árvore)"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Treino (Antes SMOTE)", f"{len(split_data['y_train_before_smote']):,} linhas")
        c2.metric("Teste (Intacto)", f"{len(split_data['y_test']):,} linhas")
        c3.metric("Features após One-Hot", split_data["n_features_after_encoding"])
        low, high = split_data["income_clip_bounds"]
        st.write(f"**Limites IQR Clipping de Renda (aprendidos no Treino):** `US$ {low:,.2f}` a `US$ {high:,.2f}`")

    with tab2:
        st.pyplot(plot_smote_comparison(split_data["y_train_before_smote"], split_data["y_train_bal"]))
        st.success("O SMOTE foi aplicado estritamente na base de Treino, mantendo a base de Teste com a distribuição real (~21,8% inadimplentes).")

    with tab3:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("Entrada KNN (Com StandardScaler)")
            st.dataframe(split_data["X_train_knn"].head(10), use_container_width=True)
        with col_s2:
            st.subheader("Entrada Árvore (Escala Original)")
            st.dataframe(split_data["X_train_tree"].head(10), use_container_width=True)

# ==============================================================================
# FASE 5: MODELAGEM E OVERFITTING
# ==============================================================================
elif fase == "Fase 5 - Modelagem e Overfitting":
    st.title("Fase 5: Experimentação e Diagnóstico de Overfitting")
    results_df = modeling["results_df"]

    tab1, tab2, tab3 = st.tabs(["Tabela Comparativa Completa", "Curva KNN", "Curva Árvore de Decisão"])

    with tab1:
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.info(
            "**Leitura do Overfitting:** Observe que na Árvore de Decisão com `max_depth=None`, o modelo atinge ~100% no treino "
            "com queda expressiva no teste (alto `Gap_Acc` e `Gap_F1`), caracterizando sobreajuste. Profundidades controladas (ex.: `max_depth=5` ou `7`) "
            "reduzem o gap e preservam a capacidade de generalização."
        )

    with tab2:
        st.pyplot(plot_overfitting_curve(results_df, "KNN"))

    with tab3:
        st.pyplot(plot_overfitting_curve(results_df, "Árvore de Decisão"))

# ==============================================================================
# FASE 6: AVALIAÇÃO E VEREDITO DE NEGÓCIO
# ==============================================================================
elif fase == "Fase 6 - Avaliação e Veredito":
    st.title("Fase 6: Avaliação Final, Impacto Financeiro e Veredito Executivo")

    st.sidebar.markdown("### Parâmetros Financeiros (Simulador)")
    avg_loan = st.sidebar.number_input("Ticket Médio do Empréstimo (US$)", min_value=1000.0, max_value=50000.0, value=9500.0, step=500.0)
    lgd = st.sidebar.slider("Perda Dado o Calote — LGD (%)", min_value=10, max_value=100, value=65, step=5) / 100.0
    nim = st.sidebar.slider("Margem Líquida de Juros — NIM (%)", min_value=5, max_value=40, value=14, step=1) / 100.0

    eval_data = generate_executive_evaluation(
        modeling, avg_loan=avg_loan, lgd=lgd, net_interest_margin=nim, persist_best=True
    )

    tab1, tab2, tab3 = st.tabs(["Matrizes e Reports", "Simulação Financeira (FN vs FP)", "Veredito Executivo"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            knn_m = eval_data["best_knn_metrics"]
            st.subheader(f"Melhor KNN ({knn_m['Configuracao']})")
            st.pyplot(plot_confusion_matrix_figure(knn_m["confusion_matrix"], f"Matriz de Confusão — KNN ({knn_m['Configuracao']})"))
            st.text(knn_m["report_text"])

        with c2:
            tree_m = eval_data["best_tree_metrics"]
            st.subheader(f"Melhor Árvore ({tree_m['Configuracao']})")
            st.pyplot(plot_confusion_matrix_figure(tree_m["confusion_matrix"], f"Matriz de Confusão — Árvore ({tree_m['Configuracao']})"))
            st.text(tree_m["report_text"])

    with tab2:
        st.subheader("Comparativo Técnico e Financeiro na Base de Teste")
        st.dataframe(eval_data["comparison_df"], use_container_width=True, hide_index=True)
        st.markdown(
            "- **Falso Negativo (FN):** Cliente inadimplente classificado como seguro. Gera perda direta do capital principal ajustada pela LGD.\n"
            "- **Falso Positivo (FP):** Bom pagador classificado como risco de calote. Gera custo de oportunidade pela margem líquida não realizada."
        )

    with tab3:
        verdict = eval_data["verdict"]
        st.success(f"### MODELO RECOMENDADO PARA PRODUÇÃO: {verdict['recommended_model']}")
        for reason in verdict["reasons"]:
            st.markdown(f"- {reason}")
