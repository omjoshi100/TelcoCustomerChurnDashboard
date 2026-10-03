# Step 3: Exploratory Data Analysis (EDA) Guide

This guide details the statistical distributions, key cohort behaviors, and visual findings in the Telco dataset.

---

## 1. High-Level Churn Metrics

Across the 7,043 customer accounts:
- **Overall Churn Rate**: **26.54%** (~1,869 churners vs 5,174 retained)
- **Average Monthly Bill**: **$64.76**
- **Median Customer Tenure**: **29 months**

This baseline indicates a noticeable class imbalance of roughly **3:1** (retained vs churned).

---

## 2. Key Exploratory Findings

### A. Contract Type Dominance
```python
contract_stats = df.groupby("Contract")["Churn_Num"].agg(["count", "mean"])
```
- **Month-to-month Contracts**: Churn rate is **42.7%**! 
  - Over 55% of all customers in the dataset are on month-to-month contracts.
  - They represent over **88% of all churn events**.
- **One-Year Contracts**: Churn rate drops dramatically to **11.3%**.
- **Two-Year Contracts**: Churn rate drops to **2.8%**.

**Strategic Insight**: Customer commitment duration is the single strongest buffer against defection. Transitioning month-to-month users into 1-year agreements is the highest-ROI retention initiative.

---

### B. Customer Tenure & New User Vulnerability
- Customers with **tenure < 12 months** experience a churn rate exceeding **47%**.
- Customers surviving past **24 months** have a churn rate under **15%**.
- Customers with **tenure > 48 months** have an ultra-low churn rate of **~7%**.

**Strategic Insight**: Churn is heavily front-loaded in the customer lifecycle. The first 90 to 180 days are critical; proactive onboarding and tech support calls can stabilize new accounts before habits solidify.

---

### C. Monthly Charges & Price Sensitivity
- Churned customers exhibit a high concentration in the **$70 – $100 / month** bracket (median: $79.65).
- Retained customers are concentrated in the lower price tier (**$20 – $40 / month**, median: $64.40).

**Strategic Insight**: Customers paying premium fees expect seamless reliability. When combined with fiber optic service issues or lack of tech support, high monthly bills trigger rapid attrition.

---

### D. Internet Service Type
- **Fiber Optic**: Churn rate is **41.9%**.
- **DSL**: Churn rate is **19.0%**.
- **No Internet (Phone only)**: Churn rate is **7.4%**.

**Why is Fiber Optic Churn So High?**
Further slice analysis reveals that Fiber Optic customers have high monthly fees ($70+) but frequently lack bundled **TechSupport** or **OnlineSecurity**. When technical issues arise, unassisted fiber customers defect quickly.

---

## 3. EDA Code and Visualizations

All Plotly chart generators are organized in `src/eda.py`:
- `plot_churn_by_contract()`: Grouped comparison of contract lengths.
- `plot_churn_by_tenure()`: Cohort breakdown by 1-year tenure buckets.
- `plot_churn_by_monthly_charges()`: Histogram with overlay box plots.
- `plot_churn_by_internet_service()`: Bar chart comparing Fiber, DSL, and Phone-only.
