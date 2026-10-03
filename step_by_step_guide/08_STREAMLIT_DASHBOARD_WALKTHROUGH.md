# Step 8: Streamlit Dashboard Walkthrough

This guide walks through the user experience, controls, and technical implementation of the 3-page interactive Streamlit dashboard (`app.py`).

---

## 1. How to Launch the Dashboard Locally

Activate your virtual environment and launch Streamlit:

### Windows (PowerShell)
```powershell
cd C:\Users\OM\.gemini\antigravity\scratch\TelcoCustomerChurnDashboard
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

### macOS / Linux
```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit will launch locally at `http://localhost:8501`.

---

## 2. Dashboard Layout and Architecture

The application is structured into three dedicated functional views accessible via the sidebar navigation:

```text
Streamlit Dashboard
├── Sidebar (Cost Parameters, Threshold Slider, Gemini API Key)
├── Page 1: 📋 Risk-Ranked Customer List
├── Page 2: 🔍 Customer Deep-Dive & SHAP
└── Page 3: 📊 Segment Overview & Cost Matrix
```

---

## 3. Page 1: Risk-Ranked Customer Priority List

### Purpose:
Provides retention campaign managers with a real-time, sortable target list of accounts at imminent risk of churning before the next monthly billing cycle.

### Key Features:
- **Executive Metric Cards**:
  - Total test cohort size ($n=1,409$).
  - Number of flagged customers based on the active threshold.
  - Estimated campaign outreach cost ($N \times \$20$).
  - Churn revenue exposure ($N_{\text{churners}} \times \$500$).
- **Multi-Select Filters**:
  - Filter by Contract (Month-to-month, One year, Two year).
  - Filter by Internet Service (Fiber optic, DSL, None).
  - Filter by Risk Tier (High Risk: >60%, Medium Risk: 30-60%, Low Risk: <30%).
- **Interactive Data Table**:
  - Displays customer ID, risk tier badge, predicted probability, tenure, charges, and recommendations.
- **1-Click CSV Export**:
  - Downloads the filtered target list directly to CSV for upload into CRM or dialer systems (Salesforce, HubSpot, etc.).

---

## 4. Page 2: Customer Deep-Dive, SHAP & AI Retention Advisor

### Purpose:
Gives call center agents and account managers an instant, 360-degree diagnostic view of any individual customer.

### Key Features:
- **Customer Selector**: Dropdown menu searching all customer IDs in the test set.
- **Account Snapshot Cards**:
  - Churn Probability (color-coded red/green).
  - Contract & Tenure duration.
  - Monthly bill amount.
  - Subscribed services.
- **Local SHAP Waterfall Bar Chart**:
  - Visualizes exact feature contributions pushing toward Churn (red) vs. Anchoring Retention (blue).
- **Top 3 Drivers Breakdown**:
  - Lists the top 3 algorithmic reasons for churn with human-readable labels and exact values.
- **🤖 AI Retention Strategist**:
  - Displays the 2-sentence executive summary and concrete retention offer.
  - Expandable box showing the raw prompt sent to the LLM.

---

## 5. Page 3: Segment Overview & Business Cost Matrix

### Purpose:
Provides leadership and data science executives with validation metrics, cohort distributions, and financial ROI simulations.

### Key Features:
1. **Model Evaluation Matrix**:
   - Head-to-head comparison of Baseline Logistic Regression vs. Tuned XGBoost (`ROC-AUC`, `PR-AUC`, `Recall`, `Precision`, `F1`).
2. **Business Cost Optimization Curve**:
   - Plotly chart illustrating the asymmetric cost curve.
   - Highlights the cost-minimizing threshold and displays total dollars saved versus standard 0.50 threshold.
3. **Cohort Distribution Visualizations**:
   - Churn by Contract Type (Month-to-month risk spike).
   - Churn by Customer Tenure (new user attrition).
   - Monthly Charges KDE/box distribution.
   - Internet Service attrition rates.
4. **Global TreeSHAP Importance**:
   - Ranked horizontal bar chart of the most influential macroeconomic drivers across the entire business.
