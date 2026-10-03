# Complete File & Folder Catalog
**Comprehensive Architectural Directory & File Purpose Reference**  
**Project**: Customer Churn Early-Warning Dashboard  
**Author**: Om Joshi  

---

## 1. Directory Tree Overview

```text
TelcoCustomerChurnDashboard/
│
├── .gitignore                                   <- Excludes cache, venvs, and secrets from Git
├── README.md                                    <- Primary repository landing page with badges & benchmarks
├── requirements.txt                             <- Python project dependencies with pinned versions
├── run_pipeline.py                              <- Headless CLI runner for end-to-end execution
├── app.py                                       <- Interactive 3-page Streamlit production web dashboard
│
├── data/                                        <- Data cache & serialized models
│   ├── telco_churn.csv                          <- Cleaned & cached raw customer dataset (7,043 rows)
│   └── models/                                  <- Serialized binaries
│       ├── trained_artifacts.pkl                <- Preprocessing pipeline, XGBoost & Logistic models
│       └── shap_cache.pkl                       <- Precomputed TreeSHAP matrix (1,409 x 30)
│
├── src/                                         <- Core Python modules & business logic
│   ├── __init__.py                              <- Python package identifier
│   ├── data_loader.py                           <- Automated ingestion & TotalCharges bug repair
│   ├── eda.py                                   <- Cohort aggregation & Plotly visualizers
│   ├── model_trainer.py                         <- Baseline LR, Tuned XGBoost & 5-fold cross-validation
│   ├── threshold_optimizer.py                   <- Cost curve & asymmetric threshold optimization
│   ├── explainability.py                        <- TreeSHAP calculations & customer-level waterfall charts
│   └── llm_advisor.py                           <- Grounded Gemini 2.5 Flash & offline domain rule engine
│
├── step_by_step_guide/                          <- Comprehensive step-by-step documentation folder
│   ├── 01_PREREQUISITES_AND_ENVIRONMENT_SETUP.md<- Environment creation, dependencies & verification
│   ├── 02_DATA_PIPELINE_AND_CLEANING.md         <- Source datasets, TotalCharges fix, schema validation
│   ├── 03_EXPLORATORY_DATA_ANALYSIS_GUIDE.md    <- Churn rates, contract dominance, tenure cliff
│   ├── 04_BASELINE_AND_XGBOOST_MODELING.md      <- Preprocessing, scale_pos_weight, CV metrics
│   ├── 05_BUSINESS_COST_THRESHOLD_OPTIMIZATION.md<- Asymmetric cost formulation ($20 vs $500), savings
│   ├── 06_SHAP_EXPLAINABILITY_DEEP_DIVE.md      <- Game-theoretic axioms, TreeSHAP, local drivers
│   ├── 07_LLM_RETENTION_ADVISOR_CONFIGURATION.md<- Gemini prompting, strict grounding, offline engine
│   ├── 08_STREAMLIT_DASHBOARD_WALKTHROUGH.md    <- 3-page UI architecture, filters, CSV exports
│   └── 09_STREAMLIT_COMMUNITY_CLOUD_DEPLOYMENT.md<- 1-click cloud deployment & secrets configuration
│
├── docs/                                        <- Technical documentation & reports
│   ├── PROJECT_REPORT.md                        <- 360-degree executive and technical project report
│   ├── IN_N_OUT_OF_PROJECT.md                   <- Architecture, data flow, math, edge cases & defenses
│   ├── PROJECT_QNA.md                           <- Comprehensive Q&A across business, data, model, SHAP
│   ├── INTERVIEW_QUESTIONS_AND_ANSWERS.md       <- 13+ rigorous technical ML & data science interview Q&A
│   └── FILE_AND_FOLDER_CATALOG.md               <- This line-by-line file & folder manifest
│
└── presentation/                                <- Executive presentation deck
    ├── SLIDES_OUTLINE.md                        <- 12-slide comprehensive presentation script
    ├── generate_deck.py                         <- python-pptx generator script
    └── Telco_Churn_AI_Dashboard.pptx            <- Generated widescreen PowerPoint presentation
```

---

## 2. Line-by-Line File Manifest & Purpose

### Root Files

#### `app.py`
- **Purpose**: The main interactive web application powered by Streamlit.
- **Key Responsibilities**:
  - Implements the 3 main navigation pages: 📋 Risk-Ranked Customer List, 🔍 Customer Deep-Dive & SHAP, and 📊 Segment Overview & Cost Matrix.
  - Exposes interactive sidebar widgets for dynamically altering intervention offer cost ($20 default), customer churn loss ($500 default), decision threshold, and optional Gemini API keys.
  - Leverages `@st.cache_resource` and `@st.cache_data` for lightning-fast sub-50ms page renders.
- **Dependencies**: `streamlit`, `plotly`, `pandas`, `numpy`, `src.*`.

#### `run_pipeline.py`
- **Purpose**: A standalone command-line pipeline runner enabling 1-click batch execution and validation.
- **Key Responsibilities**:
  - Ingests raw data and validates TotalCharges repair.
  - Trains Baseline Logistic Regression and Tuned XGBoost with 5-fold cross-validation.
  - Computes the business cost curve and prints financial comparison metrics.
  - Calculates TreeSHAP feature attributions.
  - Tests the LLM advice synthesizer on the highest-risk customer in the test set.
  - Reconfigures stdout to UTF-8 to prevent Windows console Unicode crashes.
- **Runtime**: ~5.3 seconds end-to-end on local machine.

#### `requirements.txt`
- **Purpose**: Pinned dependency list for virtual environment initialization and Streamlit Community Cloud deployments.
- **Packages Included**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `streamlit`, `plotly`, `google-genai`, `requests`, `matplotlib`, `python-pptx`.

#### `.gitignore`
- **Purpose**: Prevents binary caches, virtual environments (`.venv`), Python bytecode (`__pycache__`), IDE settings (`.vscode`), and secrets from leaking into Git version control.

#### `README.md`
- **Purpose**: The primary public-facing project landing page on GitHub.
- **Sections**: Badges, Executive Overview, Architecture Diagram, Benchmark Results Table, Cost Optimization ROI, Screenshot Gallery Guide, Installation, and Cloud Deployment instructions.

---

### `src/` Directory (Core Analytical & ML Engine)

#### `src/__init__.py`
- **Purpose**: Defines `src` as a formal Python package, enabling clean absolute and relative module imports across scripts and tests.

#### `src/data_loader.py`
- **Purpose**: Data ingestion, schema repair, and train/test feature isolation.
- **Key Functions**:
  - `load_raw_data()`: Implements a 3-tier fallback hierarchy (local disk $\rightarrow$ raw GitHub mirror $\rightarrow$ OpenML dataset 42178).
  - `clean_data(df)`: Strips whitespace, coerces `TotalCharges` text to float, imputes 0-tenure accounts to `$0.00`, and encodes `Churn` to binary 1/0.
  - `get_feature_splits(df)`: Drops identifiers and partitions features into numerical and categorical candidate lists.

#### `src/eda.py`
- **Purpose**: Statistical aggregations and interactive visual analytics.
- **Key Functions**:
  - `calculate_churn_metrics(df)`: Computes macro summary metrics (total accounts, churn percentage, average bill, median tenure).
  - `plot_churn_by_contract(df)`: Generates grouped Plotly bar chart highlighting the 42.7% Month-to-Month churn spike.
  - `plot_churn_by_tenure(df)`: Binned cohort analysis illustrating the "first-year cliff".
  - `plot_churn_by_monthly_charges(df)`: Dual-color overlay histogram with marginal box plots.
  - `plot_churn_by_internet_service(df)`: Comparative bar chart for Fiber Optic vs. DSL vs. Phone-only.

#### `src/model_trainer.py`
- **Purpose**: Supervised machine learning training, cross-validation, and metric benchmarking.
- **Key Functions**:
  - `build_preprocessor(num_cols, cat_cols)`: Builds a Scikit-Learn `ColumnTransformer` with `StandardScaler` and `OneHotEncoder(drop='first', handle_unknown='ignore')`.
  - `train_models()`: Fits Baseline Logistic Regression and Tuned XGBoost (`scale_pos_weight = 2.77`, `max_depth = 4`, `learning_rate = 0.04`); performs 5-fold stratified cross-validation; saves all models and test matrices into `data/models/trained_artifacts.pkl`.
  - `load_artifacts()`: Reads serialized models from disk; automatically triggers training if files are missing.

#### `src/threshold_optimizer.py`
- **Purpose**: Translates probabilistic machine learning into direct commercial dollars.
- **Key Functions**:
  - `compute_cost_curve(y_true, y_probs, cost_offer, cost_loss, retention_success_rate)`: Sweeps candidate thresholds $t \in [0.01, 0.99]$, computing $TP, FP, FN, TN$ and calculating expected business loss.
  - `find_optimal_threshold(...)`: Identifies the exact cost-minimizing threshold and computes net dollar savings compared to the default 0.50 threshold.
  - `plot_cost_vs_threshold(...)`: Renders the U-shaped financial loss curve in Plotly, highlighting the optimal intervention cutoff.

#### `src/explainability.py`
- **Purpose**: Game-theoretic feature attribution using TreeSHAP.
- **Key Functions**:
  - `compute_or_load_shap(xgb_model, X_test_df)`: Evaluates `shap.TreeExplainer` on the test cohort and caches matrices into `data/models/shap_cache.pkl`.
  - `get_global_importance(shap_data, top_n)`: Calculates mean absolute SHAP values across all test rows.
  - `plot_global_importance(df_importance)`: Renders horizontal Plotly bar chart of top enterprise drivers.
  - `get_customer_top_drivers(sample_idx, ...)`: Extracts the top 3 positive contributors pushing an individual customer toward churn.
  - `plot_customer_shap_waterfall(sample_idx, ...)`: Generates custom customer-level horizontal waterfall bar chart contrasting churn forces (red) against retention anchors (blue).

#### `src/llm_advisor.py`
- **Purpose**: Generative AI retention strategist combining LLM reasoning with strict feature grounding.
- **Key Functions**:
  - `generate_llm_advice(customer_id, churn_prob, threshold, top_drivers, ...)`: Injects customer SHAP drivers into a grounded prompt for the official Google Gemini 2.5 Flash SDK (`google-genai`).
  - `generate_grounded_fallback_advice(...)`: Deterministic domain rule engine that matches multi-driver combinations to proven retention offers, guaranteeing zero crashes offline.

---

### `step_by_step_guide/` Directory (Tutorial & How-To Guides)

- **`01_PREREQUISITES_AND_ENVIRONMENT_SETUP.md`**: Guide for virtual environment setup, pip dependencies, and API keys.
- **`02_DATA_PIPELINE_AND_CLEANING.md`**: Ingestion architecture, data dictionaries, and the `TotalCharges` 0-tenure repair.
- **`03_EXPLORATORY_DATA_ANALYSIS_GUIDE.md`**: Statistical breakdown of contract types, tenure cohorts, and billing distributions.
- **`04_BASELINE_AND_XGBOOST_MODELING.md`**: Preprocessing pipelines, `scale_pos_weight` math, 5-fold CV, and benchmark evaluation.
- **`05_BUSINESS_COST_THRESHOLD_OPTIMIZATION.md`**: Asymmetric cost matrices, mathematical loss functions, and financial ROI simulations.
- **`06_SHAP_EXPLAINABILITY_DEEP_DIVE.md`**: Cooperative game theory, Shapley axioms, TreeSHAP polynomial-time algorithm, and local waterfall plots.
- **`07_LLM_RETENTION_ADVISOR_CONFIGURATION.md`**: Gemini 2.5 Flash integration, strict grounding prompt engineering, and offline domain engines.
- **`08_STREAMLIT_DASHBOARD_WALKTHROUGH.md`**: Complete user manual for navigating and utilizing the 3-page Streamlit web application.
- **`09_STREAMLIT_COMMUNITY_CLOUD_DEPLOYMENT.md`**: 1-click cloud deployment guide for hosting the application publicly via GitHub.

---

### `docs/` Directory (Technical & Enterprise Documentation)

- **`PROJECT_REPORT.md`**: Complete 360-degree executive and technical report covering problem statement, methodology, benchmarks, and ROI.
- **`IN_N_OUT_OF_PROJECT.md`**: Exhaustive architectural blueprint detailing data flows, mathematical formulations, edge cases, and failure defenses.
- **`PROJECT_QNA.md`**: 16+ deep questions and answers covering business value, data engineering, modeling, TreeSHAP, and deployment.
- **`INTERVIEW_QUESTIONS_AND_ANSWERS.md`**: 13+ rigorous interview questions and comprehensive model answers tailored for Data Science, ML Engineering, and Leadership interviews.
- **`FILE_AND_FOLDER_CATALOG.md`**: This comprehensive manifest and architectural reference.

---

### `presentation/` Directory (Slide Deck Assets)

- **`SLIDES_OUTLINE.md`**: Full 12-slide textual presentation script with executive talking points.
- **`generate_deck.py`**: Python script leveraging `python-pptx` to build the PowerPoint presentation automatically.
- **`Telco_Churn_AI_Dashboard.pptx`**: The generated 16:9 widescreen PowerPoint presentation with custom styled cards, tables, and color themes.
