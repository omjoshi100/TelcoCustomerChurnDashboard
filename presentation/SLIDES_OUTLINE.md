# Presentation Slide Deck Outline
## Title: Customer Churn Early-Warning System: XGBoost, SHAP & LLM Retention Advisor

**Presenter**: Om Joshi  
**Project**: End-to-End Enterprise Customer Churn Dashboard  
**Stack**: Python | Scikit-Learn | XGBoost | SHAP | Streamlit | Plotly | Google Gemini  

---

### Slide 1: Title & Executive Summary
- **Title**: Customer Churn Early-Warning Dashboard
- **Subtitle**: Turning Churn Risk into Bottom-Line Retention with XGBoost, SHAP & Generative AI
- **Presenter**: Om Joshi
- **Key Takeaway**: A full-stack data science system that predicts churners, pinpoints root-cause drivers with game theory, optimizes decisions financially, and generates grounded retention offers via LLMs.

---

### Slide 2: The Enterprise Business Problem
- **The Telecommunications Reality**:
  - Customer acquisition is 5x to 7x more expensive than retention.
  - Telco industry churn averages 20% to 30% annually.
- **The Core Blindspot**:
  - Most companies react *after* a cancellation call is received.
  - Standard ML models predict *probabilities* without actionable *context*.
- **The Solution**: An early-warning dashboard identifying churn risks 30–60 days before contract expiration with personalized intervention strategies.

---

### Slide 3: Dataset Ingestion & The "TotalCharges" Data Trap
- **Benchmark Dataset**: 7,043 customer accounts with 21 demographic & service attributes.
- **The Hidden Data Trap**:
  - `TotalCharges` is loaded as an `object` string containing spaces (`" "`).
  - Standard models fail or drop these rows blindly.
- **Root-Cause Analysis**:
  - 11 customers had `tenure == 0` (brand-new accounts with 0 bills generated).
- **Engineering Fix**:
  - Explicit string strip, numeric coercion, and imputation to `0.00` to preserve data integrity and prevent leakage.

---

### Slide 4: Exploratory Data Analysis & Segment Insights
- **Overall Churn Rate**: 26.54% (Imbalanced ~3:1 ratio).
- **Contract Type Dominance**:
  - Month-to-Month contracts have a **42.7% churn rate** (represents 88% of all churners!).
  - 1-Year contracts drop to 11.3%; 2-Year contracts drop to 2.8%.
- **Tenure Vulnerability**:
  - Customers in their first 12 months exhibit a 47% churn rate.
- **Price Sensitivity**:
  - High monthly fee customers ($70 - $100) are vulnerable when lacking bundled tech support.

---

### Slide 5: Machine Learning Architecture & Imbalance Handling
- **Leakage-Free Preprocessing**:
  - 80/20 Stratified Train/Test Split.
  - Continuous variables: `StandardScaler`.
  - Categoricals: `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Baseline**: Logistic Regression with `class_weight='balanced'`.
- **Champion**: Tuned XGBoost with `scale_pos_weight = 2.77`.
  - Exactly balances positive/negative class loss gradients.
  - Regularized with `max_depth=4`, `colsample_bytree=0.8`, `gamma=0.2`.
- **Validation**: 5-Fold Stratified Cross-Validation (0.8477 ROC-AUC ± 0.011).

---

### Slide 6: Model Evaluation Benchmark
- **Head-to-Head Comparison on Test Set ($n=1,409$)**:
  - **ROC-AUC**: XGBoost (0.8460) vs Baseline (0.8417).
  - **PR-AUC**: XGBoost (**0.6650**) vs Baseline (0.6331) $\rightarrow$ +3.2 points improvement in minority class precision-recall!
  - **Recall**: 79.68% (captures 8 out of 10 leaving customers).
  - **F1-Score**: 0.6267 vs 0.6136.
- **Why Accuracy is Deceptive**:
  - Predicting "Nobody Churns" yields 73.5% accuracy but zero business value.

---

### Slide 7: Cost-Benefit Threshold Optimization
- **The Flaw of Standard 0.50 Threshold**:
  - Treats a $20 promotional offer the same as a $500 lost customer.
- **Asymmetric Cost Matrix**:
  - False Negative (Missed Churner) = **-$500**.
  - False Positive (Unnecessary Offer) = **-$20**.
  - True Positive (Intervened Churner) = **-$20 offer + 50% retention success**.
- **Financial Sweet Spot**:
  - Cost at 0.50 threshold: **$124,460**.
  - Cost at optimal threshold: **$115,880**.
  - **Net Direct Savings: $8,580 on the test cohort alone** (~$600k+ enterprise annual scale).

---

### Slide 8: SHAP Explainability (Macro & Micro)
- **Game-Theoretic Feature Attribution**:
  - Exact TreeSHAP polynomial-time decomposition.
- **Global Macro Drivers**:
  1. Month-to-Month contracts (+0.55 mean SHAP).
  2. Tenure length (strong retention anchor).
  3. Fiber optic internet with no tech support.
  4. Elevated monthly charges.
- **Local Micro Accountability**:
  - Waterfall chart per customer showing exact attributes pushing toward Churn vs. Retention.

---

### Slide 9: Generative AI Retention Advisor
- **Bridging the Gap from Prediction to Action**:
  - Customer reps cannot analyze log-odds during live calls.
- **Prompt Architecture**:
  - Input: Customer ID, Churn %, and Top 3 SHAP Drivers.
  - Strict grounding constraint: 2-sentence summary + 1 concrete retention action.
- **Dual-Engine Architecture**:
  - Primary: Google Gemini 2.5 Flash API (sub-second generation).
  - Resilient Fallback: Grounded Domain Rule Engine (zero downtime offline).

---

### Slide 10: 3-Page Streamlit Production Dashboard
- **Page 1: Risk-Ranked Priority List**:
  - Real-time customer filtering, risk tier badges, 1-click CSV dialer export.
- **Page 2: Customer Deep-Dive & SHAP**:
  - Interactive customer lookup, live SHAP waterfall, and AI Retention Advisor.
- **Page 3: Segment Overview & Cost Matrix**:
  - Interactive threshold tuning simulator, cohort distributions, model benchmark tables.

---

### Slide 11: Deployment & MLOps Architecture
- **Packaging**: Containerized Streamlit deployment on Streamlit Community Cloud.
- **Security**: Zero hardcoded credentials; environment variables for API keys.
- **Latency**: Caching via `@st.cache_data` and `@st.cache_resource` for sub-second UI interactions.
- **GitHub Repository**: Complete with automated setup scripts and comprehensive documentation.

---

### Slide 12: Business Impact & Strategic Roadmap
- **Immediate Value Delivered**:
  - 80% Churn Detection Recall.
  - $8,500+ immediate savings per 1,400 customers via threshold optimization.
  - Personalized frontline retention messaging.
- **Future Enhancements**:
  - Real-time event streaming ingestion via Kafka.
  - Multi-armed bandit testing for automated retention offer selection.
