# In 'n' Out of the Project: Technical & Architectural Deep Dive
**Project**: Customer Churn Early-Warning Dashboard  
**Architecture Specification & Engineering Handbook**  
**Author**: Om Joshi  

---

## 1. System Overview & Macro Architecture

The **Customer Churn Early-Warning Dashboard** is designed around a decoupled, 5-tier architecture connecting raw enterprise data streams to frontline business action:

```
[ Tier 1: Ingestion & Quality Control ]
       │  • Multi-source retrieval (Local -> Raw CDN -> OpenML)
       │  • 'TotalCharges' whitespace fix & 0-tenure $0.00 imputation
       ▼
[ Tier 2: Feature Engineering & Preprocessing ]
       │  • Strict 80/20 Stratified Partitioning (No Data Leakage)
       │  • StandardScaler (Numeric) + OneHotEncoder(drop='first')
       ▼
[ Tier 3: Ensemble Modeling & Optimization ]
       │  • Baseline: Logistic Regression (class_weight='balanced')
       │  • Champion: Tuned XGBoost with scale_pos_weight = 2.77
       │  • 5-Fold Stratified Cross-Validation
       │  • Asymmetric Financial Cost-Benefit Threshold Optimization
       ▼
[ Tier 4: Explainability & Generative AI ]
       │  • TreeSHAP Polynomial-Time Decomposition
       │  • Macro Global Feature Importance + Micro Top 3 Customer Drivers
       │  • Google Gemini 2.5 Flash API + Grounded Domain Rule Engine
       ▼
[ Tier 5: Presentation & Frontline Delivery ]
       │  • Interactive 3-Page Streamlit Web Dashboard
       │  • Dynamic Plotly Visualizations & Live Cost Simulator
       │  • Automated PowerPoint (.pptx) Deck Generation
       │  • 1-Click CRM/Dialer CSV Export
```

---

## 2. In & Out Data Flow Specifications

### 2.1 Inbound Data Flow (Ingestion)
- **Source Input**: Tabular customer records (`data/telco_churn.csv`).
- **Raw Dimensions**: 7,043 rows $\times$ 21 columns.
- **Attributes**:
  - `customerID` (Identifier string)
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account Attributes: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
  - Service Subscriptions: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - Financial Metrics: `MonthlyCharges`, `TotalCharges`
  - Target: `Churn` ("Yes", "No")

### 2.2 Data Cleaning & Transformation Layer
| Raw Field | Input State | Transformation Applied | Output State |
| :--- | :--- | :--- | :--- |
| `TotalCharges` | Object string with `" "` for 11 rows | Coerce to numeric, impute 0-tenure rows with `0.00` | Clean `float64`, 0 nulls |
| `SeniorCitizen` | Integer (0/1) | Preserved as binary integer feature | Integer `int64` |
| `Churn` | Object ("Yes" / "No") | Lowercase string comparison to binary indicator | `Churn_Num` (`1` / `0`) |
| Column Headers | Strings with potential whitespace | `.str.strip()` applied to all headers and strings | Sanitized strings |

### 2.3 Feature Matrix Generation (`X_train`, `X_test`)
- **Training Set Dimensions**: 5,634 rows $\times$ 30 transformed features.
- **Test Set Dimensions**: 1,409 rows $\times$ 30 transformed features.
- **Transformed Features**:
  - Continuous (3): `tenure`, `MonthlyCharges`, `TotalCharges`
  - Categorical (27 one-hot binary columns): `Contract_One year`, `Contract_Two year`, `InternetService_Fiber optic`, `TechSupport_Yes`, etc.

### 2.4 Outbound Artifacts & Cache
All trained assets are serialized into disk for zero-latency runtime loading:
- `data/models/trained_artifacts.pkl`: Dictionary containing fitted preprocessing pipeline, trained models, test matrices, and evaluation metrics.
- `data/models/shap_cache.pkl`: Precomputed TreeSHAP matrix ($1,409 \times 30$), base value, and feature list.

---

## 3. Mathematical Foundations

### 3.1 Class Imbalance Handling (`scale_pos_weight`)
In standard cross-entropy loss, each sample contributes equally to the gradient:
$$\mathcal{L} = -\sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

In our dataset, only **26.54%** of customers churn. To prevent the majority negative class from dominating split selection, we compute:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{4,139}{1,495} \approx 2.77$$

In XGBoost, the objective function weights the positive gradient by $w$:
$$\mathcal{L}_{\text{XGB}} = -\sum_{i=1}^{N} \left[ w \cdot y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] + \Omega(\mathcal{T})$$
Where $\Omega(\mathcal{T}) = \gamma T + \frac{1}{2}\lambda \sum w_j^2 + \alpha \sum |w_j|$ provides tree complexity regularization.

### 3.2 TreeSHAP Game-Theoretic Decomposition
Shapley values compute feature attribution by evaluating the difference between predictions with and without feature $j$, averaged across all possible feature subsets $S \subseteq F \setminus \{j\}$:
$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left( f(S \cup \{j\}) - f(S) \right)$$

Traditional KernelSHAP requires exponential sampling $\mathcal{O}(2^{|F|})$. `shap.TreeExplainer` evaluates this in polynomial time $\mathcal{O}(T \cdot L \cdot D^2)$ (where $T = \text{trees}$, $L = \text{leaves}$, $D = \text{depth}$) by recursively keeping track of the proportion of training paths down each branch.

### 3.3 Asymmetric Business Cost Curve
Let:
- $C_{\text{loss}} = \$500$ (Customer lifetime value loss on defection)
- $C_{\text{offer}} = \$20$ (Cost of retention outreach incentive)
- $r = 0.50$ (Empirical retention intervention success rate)
- $t \in [0, 1]$ (Classification probability cutoff)

For any threshold $t$, the expected business cost is:
$$\text{Cost}(t) = FN(t) \cdot C_{\text{loss}} + FP(t) \cdot C_{\text{offer}} + TP(t) \cdot \left[ C_{\text{offer}} + (1 - r) \cdot C_{\text{loss}} \right]$$

The optimal threshold $t^*$ satisfies:
$$t^* = \arg\min_{t \in [0.01, 0.99]} \text{Cost}(t)$$

Because $C_{\text{loss}} \gg C_{\text{offer}}$, the cost curve is asymmetric, shifting the optimal cutoff downward into the **0.11 – 0.31** range, capturing significantly more churners before they defect.

---

## 4. Module-by-Module Code Catalog

### 4.1 Ingestion: `src/data_loader.py`
- **Functions**:
  - `load_raw_data()`: Checks local disk cache, tries raw GitHub mirrors, falls back to OpenML.
  - `clean_data(df)`: Strips whitespace, coerces `TotalCharges`, imputes 0-tenure rows with `0.00`, creates binary target.
  - `get_feature_splits(df)`: Drops ID/target columns, separates continuous and categorical feature names.

### 4.2 Exploratory Analytics: `src/eda.py`
- **Functions**:
  - `calculate_churn_metrics(df)`: Returns customer counts, churn rate, average bill, median tenure.
  - `plot_churn_by_contract(df)`: Interactive bar chart highlighting Month-to-Month attrition.
  - `plot_churn_by_tenure(df)`: Binned cohort analysis (0-12m, 13-24m, 25-48m, 49-72m).
  - `plot_churn_by_monthly_charges(df)`: Dual-color overlay histogram with marginal box plots.
  - `plot_churn_by_internet_service(df)`: Comparative bar chart for Fiber Optic vs. DSL vs. None.

### 4.3 Modeling Engine: `src/model_trainer.py`
- **Functions**:
  - `build_preprocessor(num_cols, cat_cols)`: Assembles `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
  - `train_models()`: Fits Baseline Logistic Regression and Tuned XGBoost; runs 5-fold cross-validation; saves `trained_artifacts.pkl`.
  - `load_artifacts()`: Loads serialized artifacts from disk; if absent, automatically triggers training.

### 4.4 Financial Threshold Optimizer: `src/threshold_optimizer.py`
- **Functions**:
  - `compute_cost_curve(y_true, y_probs, cost_offer, cost_loss, retention_success_rate)`: Computes $TP, FP, FN, TN$ across 100 thresholds.
  - `find_optimal_threshold(df_cost, y_true, y_probs, cost_offer, cost_loss)`: Returns cost-minimizing cutoff and dollar savings over 0.50.
  - `plot_cost_vs_threshold(df_cost, optimal_info)`: Plots the U-shaped financial loss curve with optimal cutoff markers.

### 4.5 Explainability Engine: `src/explainability.py`
- **Functions**:
  - `compute_or_load_shap(xgb_model, X_test_df)`: Evaluates TreeSHAP on test set and caches to disk.
  - `get_global_importance(shap_data, top_n)`: Computes mean absolute SHAP across all test rows.
  - `plot_global_importance(df_importance)`: Horizontal Plotly bar chart of top macroeconomic drivers.
  - `get_customer_top_drivers(sample_idx, shap_data, ...)`: Extracts top 3 positive contributors for an individual customer.
  - `plot_customer_shap_waterfall(sample_idx, shap_data)`: Local horizontal bar chart contrasting risk vs. retention forces.

### 4.6 Generative AI Advisor: `src/llm_advisor.py`
- **Functions**:
  - `generate_llm_advice(customer_id, churn_prob, threshold, top_drivers, ...)`: Injects customer SHAP drivers into a grounded prompt for Gemini 2.5 Flash; falls back to domain engine if offline.
  - `generate_grounded_fallback_advice(...)`: Deterministic domain synthesizer matching driver combinations to tailored retention offers.

### 4.7 Frontend Dashboard: `app.py`
- Multi-page Streamlit web app with sidebar controls, KPI metric cards, Plotly charts, data filters, and CSV export.

### 4.8 Headless Automation: `run_pipeline.py`
- Standalone CLI runner executing data loading, model training, evaluation, threshold optimization, SHAP calculation, and LLM advice verification in ~5 seconds.

### 4.9 Presentation Generator: `presentation/generate_deck.py`
- Automated Python script leveraging `python-pptx` to build the 12-slide widescreen PowerPoint presentation deck (`Telco_Churn_AI_Dashboard.pptx`).

---

## 5. Failure Modes, Edge Cases & Defenses

| Potential Failure Mode | Root Cause | Engineering Defense Implemented |
| :--- | :--- | :--- |
| **Missing Internet Access on First Run** | OpenML or GitHub raw mirror unreachable | 3-tier fallback hierarchy + local disk caching in `data/telco_churn.csv` |
| **`TotalCharges` Parsing Crash** | Blank string values (`" "`) | Explicit `pd.to_numeric(..., errors='coerce')` and `fillna(0.00)` |
| **Data Leakage in Preprocessing** | Fitting scalers/encoders on full dataset | Strict featurization ordering: split train/test first; fit solely on `X_train` |
| **Gemini API Outage / Missing Key** | Invalid API key, rate limit, or offline demo | Intelligent Grounded Domain Engine automatically catches exceptions and synthesizes structured retention advice |
| **Windows Console Unicode Errors** | Printing emojis (`\U0001f4e1`) to default cp1252 stdout | Explicit `sys.stdout.reconfigure(encoding='utf-8')` in runner scripts |
| **Streamlit Page Refresh Latency** | Re-running heavy TreeSHAP or XGBoost on every slider change | `@st.cache_resource` and `@st.cache_data` decorators cache models and SHAP matrices in RAM |
| **Unseen Categorical Levels at Inference** | Real-world customer with novel category | `OneHotEncoder(handle_unknown='ignore')` maps unseen categories to zeros without throwing exceptions |
