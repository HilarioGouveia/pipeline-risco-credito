# SDD — Spec-Driven Development: Pipeline Preditivo de Risco de Crédito

**Versão:** 1.0.0  
**Status:** Aprovado para Implementação  
**Stack Principal:** Python 3.11+, `uv` (Package & Project Manager), `scikit-learn`, `imbalanced-learn`, `pandas`, `pyarrow`, `plotly`/`seaborn`, `Streamlit`

---

## 1. Visão Geral e Objetivo de Negócio

### 1.1 Problema
Construir um pipeline corporativo ponta a ponta de Ciência de Dados e Machine Learning para prever a inadimplência de clientes bancários (`loan_status = 1` para inadimplente; `loan_status = 0` para adimplente), comparando os algoritmos **K-Nearest Neighbors (KNN)** e **Árvore de Decisão (Decision Tree)**, com diagnóstico rigoroso de *overfitting* e recomendação fundamentada para produção.

### 1.2 Contexto Financeiro e Matriz de Custo
* **Falso Negativo (FN — Modelo prevê `0` [Seguro], Real é `1` [Inadimplente]):** Aprovação de crédito para mau pagador. Gera **perda direta do principal emprestado**, aumento de provisões (PDD) e deterioração da carteira. **Maior impacto financeiro.**
* **Falso Positivo (FP — Modelo prevê `1` [Risco], Real é `0` [Adimplente]):** Rejeição de bom pagador. Gera **custo de oportunidade** (perda de receita de juros e churn), mas preserva o capital principal.
* **Diretriz de Otimização:** O critério de seleção do modelo campeão priorizará **Recall da Classe 1 (Inadimplentes)** e **F1-Score da Classe 1**, condicionados ao controle de *overfitting* (baixo *gap* entre treino e teste).

---

## 2. Ambiente e Gerenciamento de Dependências (`uv`)

O projeto utilizará **`uv`** para isolamento determinístico de ambiente, resolução rápida de dependências e execução reproduzível.

### 2.1 Comandos de Bootstrap
```bash
# Inicializar projeto com Python 3.11+
uv init risco-credito --python 3.11
cd risco-credito

# Adicionar dependências de produção e análise
uv add pandas numpy scikit-learn imbalanced-learn pyarrow streamlit plotly matplotlib seaborn

# Adicionar ferramentas de qualidade e testes
uv add --dev pytest ruff
```

### 2.2 Especificação do `pyproject.toml`
```toml
[project]
name = "risco-credito"
version = "0.1.0"
description = "Pipeline Preditivo de Risco de Crédito com Arquitetura Medallion e Streamlit"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "imbalanced-learn>=0.12.0",
    "matplotlib>=3.8.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "plotly>=5.20.0",
    "pyarrow>=15.0.0",
    "scikit-learn>=1.4.0",
    "seaborn>=0.13.0",
    "streamlit>=1.32.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

### 2.3 Execução da Aplicação
```bash
# Executar pipeline batch completo (Bronze -> Silver -> Gold -> Treino/Avaliação)
uv run python -m src.pipeline_runner

# Iniciar dashboard interativo
uv run streamlit run src/app.py
```

---

## 3. Arquitetura de Dados (Medallion) e Estrutura de Diretórios

Para garantir rastreabilidade, reprodutibilidade e separação de responsabilidades, o sistema adota a **Arquitetura Medallion** persistida em formato colunar **Parquet** (exceto a ingestão bruta em CSV).

```text
risco-credito/
├── pyproject.toml
├── uv.lock
├── README.md
├── data/
│   ├── bronze/
│   │   └── credit_risk_dataset.csv          # Fonte única da verdade (imutável)
│   ├── silver/
│   │   └── silver_credit_risk.parquet       # Dados limpos, deduplicados e consistentes
│   └── gold/
│       └── gold_credit_risk.parquet         # Dataset enriquecido com Feature Engineering
├── artifacts/
│   ├── models/                              # Modelos serializados (.joblib) e scalers
│   └── reports/                             # Métricas JSON, matrizes de confusão e tabelas
└── src/
    ├── __init__.py
    ├── config.py                            # Constantes, caminhos, hiperparâmetros e schemas
    ├── pipeline_runner.py                   # Orquestrador CLI end-to-end
    ├── fase1_eda.py                         # Estatística descritiva, diagnósticos e gráficos
    ├── fase2_preprocessing.py               # Limpeza determinística (Bronze -> Silver)
    ├── fase3_feature_engineering.py         # Variáveis derivadas (Silver -> Gold)
    ├── fase4_data_split.py                  # Encoding, Split Estratificado, SMOTE e Scaler
    ├── fase5_modelagem.py                   # Grid/Varredura KNN vs Árvore e diagnóstico Overfitting
    ├── fase6_avaliacao.py                   # Classification Report, Matriz de Confusão e Veredito
    ├── utils/
    │   ├── io.py                            # Leitura/escrita segura CSV/Parquet/JSON
    │   ├── graficos.py                      # Fábrica de gráficos padronizados (Plotly/Matplotlib)
    │   └── metricas.py                      # Cálculo de métricas, gaps de overfitting e custo
    └── app.py                               # Interface interativa Streamlit (6 Fases)
```

---

## 4. Contrato de Dados e Regras de Negócio por Camada

### 4.1 Camada Bronze (`data/bronze/credit_risk_dataset.csv`)
* **Dimensão Esperada:** 32.581 linhas × 12 colunas.
* **Schema de Entrada:**

| Coluna | Tipo | Descrição | Valores Ausentes (Bruto) |
| :--- | :--- | :--- | :--- |
| `person_age` | `int64` | Idade do solicitante (anos) | 0 |
| `person_income` | `int64` | Renda anual declarada (US$) | 0 |
| `person_home_ownership` | `object` | Situação habitacional (`RENT`, `OWN`, `MORTGAGE`, `OTHER`) | 0 |
| `person_emp_length` | `float64` | Tempo de emprego (anos) | 895 (~2,75%) |
| `loan_intent` | `object` | Finalidade do empréstimo (`PERSONAL`, `EDUCATION`, etc.) | 0 |
| `loan_grade` | `object` | Grau de risco atribuído (`A` a `G`) | 0 |
| `loan_amnt` | `int64` | Valor solicitado do empréstimo (US$) | 0 |
| `loan_int_rate` | `float64` | Taxa de juros anual (%) | 3.116 (~9,56%) |
| `loan_status` | `int64` | **Alvo:** `0` = Adimplente (78,18%), `1` = Inadimplente (21,82%) | 0 |
| `loan_percent_income` | `float64` | Razão empréstimo/renda original (`0.00` a `0.83`) | 0 |
| `cb_person_default_on_file` | `object` | Histórico de inadimplência prévia (`Y` / `N`) | 0 |
| `cb_person_cred_hist_length` | `int64` | Tempo de histórico de crédito (anos) | 0 |

---

## 5. Especificação Funcional das Fases Técnicas

### Fase 1: Análise Exploratória de Dados (`src/fase1_eda.py`)
* **Entradas:** `data/bronze/credit_risk_dataset.csv`
* **Processamento & Saídas Obrigatórias:**
  1. **Metadados Estruturais:** Contagem de linhas/colunas, tipagem (`dtypes`), contagem e percentual de nulos por coluna e contagem de duplicatas exatas (`165` linhas).
  2. **Sumário Estatístico:** Tabela `df.describe()` (contagem, média, desvio padrão, mínimo, Q1, mediana, Q3, máximo) e assimetria (*skewness*).
  3. **Visualizações Obrigatórias (mínimo 4 gráficos):**
     * **Gráfico 1:** Histograma + Boxplot de `person_income` e `person_age` (evidenciando assimetria à direita e idades anômalas $> 80$).
     * **Gráfico 2:** Gráfico de barras da variável alvo `loan_status` com percentuais (`78,18%` vs `21,82%`), comprovando o desbalanceamento.
     * **Gráfico 3:** Mapa de calor (Heatmap) de correlação de Pearson entre variáveis numéricas (destacando `loan_percent_income` com $r \approx 0,38$ e `loan_int_rate` com $r \approx 0,34$).
     * **Gráfico 4:** Taxa de inadimplência por `loan_grade` e `person_home_ownership`.
  4. **Síntese Analítica:** Texto estruturado justificando como os achados orientam a limpeza, imputação pela mediana, controle de outliers, balanceamento e escalonamento.

---

### Fase 2: Tratamento e Limpeza — Bronze $\rightarrow$ Silver (`src/fase2_preprocessing.py`)

> **Ajuste Crítico Anti-Data Leakage:** Regras **determinísticas** (remoção de duplicatas e regras de domínio) são executadas na camada Silver. Estatísticas **aprendidas dos dados** (medianas de imputação e limites IQR de *clipping*) são ajustadas **exclusivamente no conjunto de treino** na Fase 4, ou registradas com rastreabilidade clara caso o desafio exija a tabela limpa antes do *split* — **esta especificação implementa o ajuste estatístico estritamente no Treino (`fit` no treino, `transform` no teste) dentro de um `Pipeline` do `scikit-learn`**, mantendo na camada Silver apenas a limpeza determinística e sanitização de inconsistências físicas.

* **Regras Determinísticas (Bronze $\rightarrow$ Silver):**
  1. **Duplicatas:** `df.drop_duplicates()` — remove os `165` registros idênticos.
  2. **Inconsistência de Idade:** Remover registros com `person_age > 80` (erros cadastrais como 84, 94, 123 e 144 anos; apenas 7 registros).
  3. **Inconsistência de Tempo de Emprego:** Converter para `np.nan` registros onde `person_emp_length > (person_age - 14)` (ex.: 123 anos de emprego para jovem de 21 anos), permitindo imputação segura posterior.
* **Regras Estatísticas (Ajustadas no Treino na Fase 4 com Justificativa):**
  1. **Imputação de Nulos (`person_emp_length` e `loan_int_rate`):** Utilizar **Mediana** (`SimpleImputer(strategy="median")`). *Justificativa:* Ambas as variáveis possuem distribuição assimétrica e presença de valores extremos, tornando a mediana estatisticamente mais robusta que a média.
  2. **Outliers em Renda (`person_income`):** Aplicar **IQR Clipping / Winsorization** ($[Q_1 - 1.5 \cdot IQR,\; Q_3 + 1.5 \cdot IQR]$) no pipeline do **KNN**. *Justificativa:* Preserva clientes legítimos de alta renda sem distorcer as distâncias euclidianas do KNN. Para a **Árvore de Decisão**, os dados são mantidos sem necessidade de clipping agressivo, demonstrando sua robustez a outliers por cortes monotônicos.
* **Saída Persistida:** `data/silver/silver_credit_risk.parquet` + relatório comparativo *Antes vs. Depois*.

---

### Fase 3: Feature Engineering — Silver $\rightarrow$ Gold (`src/fase3_feature_engineering.py`)
* **Pré-condição:** Garantir que as colunas base não contenham `NaN` ou zeros no denominador antes das operações matemáticas.
* **Variáveis Criadas:**
  1. **`comprometimento_renda` (Obrigatória):**
     $$\text{comprometimento\_renda} = \left(\frac{\text{loan\_amnt}}{\text{person\_income}}\right) \times 100$$
     *Nota técnica:* Expressa o comprometimento em escala percentual (`0` a `100`). Durante a Fase 4, analisar a colinearidade com `loan_percent_income` (mesma razão em escala `0` a `1`) e garantir que `comprometimento_renda` passe pelo `StandardScaler` no fluxo do KNN.
  2. **`valor_por_ano_credito` (Complementar):**
     $$\text{valor\_por\_ano\_credito} = \frac{\text{loan\_amnt}}{\text{cb\_person\_cred\_hist\_length} + 1}$$
     *Justificativa:* Mede a agressividade do crédito solicitado em relação à maturidade do histórico de crédito (o termo $+1$ evita divisão por zero).
  3. **`score_estabilidade` (Complementar — calculado após imputação segura):**
     $$\text{score\_estabilidade} = \text{person\_emp\_length} \times \text{cb\_person\_cred\_hist\_length}$$
     *Justificativa:* Captura a interação entre estabilidade profissional e maturidade creditícia.
* **Saída Persistida:** `data/gold/gold_credit_risk.parquet`

---

### Fase 4: Separação, Encoding, Balanceamento e Escalonamento Seguro (`src/fase4_data_split.py`)

O fluxo segue estritamente a ordem abaixo para eliminar qualquer vazamento de dados (*data leakage*):

```text
Dataset Gold
   │
   ▼
1. Separação X e y (alvo = loan_status)
   │
   ▼
2. Train / Test Split (test_size=0.20, stratify=y, random_state=42)
   │
   ├─────────────────────────────────────────────┐
   ▼ (Treino - 80%)                              ▼ (Teste - 20% Intocado)
3. Imputação (Mediana) + IQR Clipping         3'. Transform usando parâmetros do Treino
   │                                             │
   ▼                                             ▼
4. Encoding Categórico                        4'. Transform usando categorias do Treino
   - OneHotEncoder(drop='first', handle_unknown='ignore')
     para: person_home_ownership, loan_intent, cb_person_default_on_file
   - OrdinalEncoder (A=0 .. G=6) para: loan_grade (preserva ordem natural de risco)
   │                                             │
   ▼                                             │
5. Balanceamento (SMOTE, random_state=42)        │
   - Apenas no Treino (50% Classe 0 / 50% Classe 1)
   │                                             │
   ├──────────────────────┐                      ├──────────────────────┐
   ▼ (Fluxo KNN)          ▼ (Fluxo Árvore)       ▼ (Teste KNN)          ▼ (Teste Árvore)
6a. StandardScaler     6b. Sem Escalonamento  6a'. scaler.transform   6b'. Dados brutos
    (fit_transform)        (Dados originais)       (variáveis contínuas)  (codificados)
```

* **Justificativas Obrigatórias no Código e na UI:**
  * **Split Estratificado:** Mantém exatamente a proporção de `~21,82%` de inadimplentes em Treino e Teste.
  * **SMOTE Exclusivo no Treino:** Evita que amostras sintéticas contaminem o conjunto de teste, garantindo avaliação realista.
  * **StandardScaler Exclusivo para KNN:** O KNN calcula distâncias euclidianas ($d(p,q) = \sqrt{\sum (p_i - q_i)^2}$); sem escalonamento, variáveis de grande magnitude como `person_income` (dezenas de milhares) dominariam completamente variáveis pequenas como `loan_int_rate` ou `cb_person_cred_hist_length`. Já a Árvore de Decisão realiza partições ortogonais baseadas em limiares ($x_j \le t$), sendo invariante a transformações monotônicas de escala.

---

### Fase 5: Modelagem e Diagnóstico de Overfitting (`src/fase5_modelagem.py`)

#### 5.1 Grade de Experimentos
1. **K-Nearest Neighbors (`KNeighborsClassifier`):**
   * Hiperparâmetros testados: `n_neighbors` $\in \{3, 5, 7, 9, 15\}$ (mínimo exigido: 4; incluímos 5 valores para curva mais suave).
   * Dados de entrada: Treino balanceado e escalonado (`X_train_knn`) $\rightarrow$ Avaliação em `X_train_knn` e `X_test_knn`.
2. **Árvore de Decisão (`DecisionTreeClassifier(random_state=42)`):**
   * Hiperparâmetros testados: `max_depth` $\in \{3, 5, 7, 10, \text{None}\}$ (mínimo exigido: 4).
   * Dados de entrada: Treino balanceado **sem escalonamento** (`X_train_tree`) $\rightarrow$ Avaliação em `X_train_tree` e `X_test_tree`.

#### 5.2 Métricas Monitoradas e Critério de Overfitting
Para cada configuração, registrar em tabela e gráfico:
* `Accuracy (Treino)` e `Accuracy (Teste)`
* `F1-Score Classe 1 (Treino)` e `F1-Score Classe 1 (Teste)`
* `Recall Classe 1 (Teste)` e `Precision Classe 1 (Teste)`
* **Gap de Overfitting:** $\Delta_{\text{acc}} = \text{Acc}_{\text{treino}} - \text{Acc}_{\text{teste}}$ e $\Delta_{F1} = F1_{\text{treino}} - F1_{\text{teste}}$.
* **Regra de Diagnóstico:** Modelos com $\text{Acc}_{\text{treino}} > 95\%$ e $\Delta_{\text{acc}} > 10\%$ (ex.: `DecisionTree(max_depth=None)` ou `KNN(K=3)`) são diagnosticados com **Overfitting severo** (memorização de ruído e amostras sintéticas do SMOTE). O modelo ótimo maximiza `F1-Score (Classe 1)` no **Teste** mantendo $\Delta_{\text{acc}} < 5\%$.

---

### Fase 6: Avaliação Final e Veredito Executivo (`src/fase6_avaliacao.py`)

* **Comparação Direta (Melhor KNN vs. Melhor Árvore):**
  1. Exibição completa do `classification_report` (Precision, Recall, F1-Score, Support) na base de Teste.
  2. Plotagem lado a lado das **Matrizes de Confusão** (valores absolutos e percentuais normalizados por linha para leitura imediata da taxa de detecção de inadimplentes).
  3. **Simulação Financeira de Impacto (Diferencial Executivo):** Parâmetro interativo no Streamlit permitindo simular o ticket médio do empréstimo (ex.: `US$ 9.500`) e o custo médio de perda por default (LGD — *Loss Given Default*, ex.: `70%` do principal) versus margem líquida de juros perdida em um Falso Positivo (ex.: `11%` ao ano), traduzindo FP e FN diretamente em impacto monetário (US$).
  4. **Veredito para Produção:** Justificativa técnica e de negócios comparando latência de inferência ($O(1)$ na Árvore vs. $O(N \cdot d)$ no KNN sobre a base expandida pelo SMOTE), explicabilidade regulatória (regras auditáveis exigidas pelo setor bancário/BACEN) e robustez a *outliers*.

---

## 6. Especificação da Interface Streamlit (`src/app.py`)

### 6.1 Layout e Navegação
* **Configuração de Página:** `st.set_page_config(page_title="Credit Risk ML Pipeline", layout="wide")`.
* **Sidebar:**
  * Seletor de etapas (`st.radio` / `st.navigation`):
    1. `Fase 1 — EDA & Diagnóstico`
    2. `Fase 2 — Data Prep (Bronze → Silver)`
    3. `Fase 3 — Feature Engineering (Silver → Gold)`
    4. `Fase 4 — Split, SMOTE & Escalonamento`
    5. `Fase 5 — Modelagem & Overfitting`
    6. `Fase 6 — Avaliação & Veredito Executivo`
  * Card de status mostrando contagem de registros em cada camada Medallion (`Bronze` / `Silver` / `Gold`).

### 6.2 Componentes por Tela
* **Fase 1:** 4 abas (`Dimensões & Tipos`, `Estatística Descritiva`, `Gráficos Exploratórios`, `Síntese & Decisões`).
* **Fase 2:** 3 abas (`Duplicados & Inconsistências`, `Estratégia de Nulos & Outliers`, `Preview Camada Silver`) + KPIs comparativos (*Antes vs. Depois*).
* **Fase 3:** Fórmulas renderizadas em LaTeX, histograma de `comprometimento_renda` por `loan_status`, matriz de correlação atualizada e preview da camada Gold.
* **Fase 4:** Diagrama do fluxo anti-leakage, gráfico de barras comparando distribuição das classes *Antes do SMOTE (78% / 22%)* vs. *Depois do SMOTE (50% / 50%)* e tabela demonstrando `mean ≈ 0, std ≈ 1` apenas no dataset do KNN.
* **Fase 5:** Tabelas interativas para KNN e Árvore + **Gráfico de Curva de Complexidade (Treino vs. Teste)** destacando visualmente a zona de *overfitting* e o ponto ótimo de generalização.
* **Fase 6:** Classification Reports lado a lado, Matrizes de Confusão interativas, Simulador Financeiro de FP vs. FN e Card Executivo de Recomendação do Modelo Campeão.

---

## 7. Critérios de Aceite (Definition of Done)

- [ ] Projeto gerenciado integralmente com `uv` (`pyproject.toml` e `uv.lock` funcionais).
- [ ] Camadas `Bronze`, `Silver` e `Gold` geradas automaticamente e persistidas em Parquet.
- [ ] Zero *data leakage*: `train_test_split` executado antes de `SimpleImputer`, `IQR Clipping`, `OneHotEncoder`, `SMOTE` e `StandardScaler`.
- [ ] `StandardScaler` aplicado **exclusivamente** ao pipeline do KNN; Árvore treinada com variáveis não escalonadas, com justificativa explícita na interface e no código.
- [ ] Mínimo de 4 hiperparâmetros testados para KNN e 4 para Árvore de Decisão com comparação explícita Treino vs. Teste.
- [ ] Todas as decisões técnicas documentadas em células/blocos explicativos dentro da aplicação Streamlit e nos docstrings dos módulos `src/`.
```

Com essa especificação consolidada, você já pode iniciar o projeto com `uv init` ou pedir que eu gere os arquivos de código (`src/config.py`, módulos das fases 1 a 6 e `src/app.py`) seguindo exatamente este contrato. Qual etapa deseja implementar primeiro?