# Step 6: SHAP Explainability Deep Dive

This guide explains how **SHAP (SHapley Additive exPlanations)** is applied to our XGBoost model to provide both global macroeconomic insights and individual customer-level accountability.

---

## 1. Why SHAP Over Traditional Feature Importance?

Traditional tree feature importance (such as Gini impurity decrease or weight/frequency) suffers from two fundamental limitations:
1. **Inconsistency**: Modifying a model to give more weight to a feature can paradoxically lower its Gini importance.
2. **Lack of Directionality**: Traditional importance tells you a feature is "important", but cannot indicate whether a high value *increases* or *decreases* churn risk.

### The Game-Theoretic Solution:
SHAP computes Shapley values from cooperative game theory. Each feature is treated as a "player" cooperating in a coalition to produce the model's prediction. The Shapley value is the average marginal contribution of a feature across all possible subsets of features.

Properties guaranteed by Shapley values:
- **Efficiency**: The sum of SHAP values plus base value equals the model output:
  $$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$
- **Symmetry**: Identical contributors receive identical attribution.
- **Additivity**: Attributions sum linearly across ensemble trees.

---

## 2. TreeSHAP Implementation

We utilize `shap.TreeExplainer`, which computes exact Shapley values in polynomial time $\mathcal{O}(TLD^2)$ rather than exponential time $\mathcal{O}(2^M)$:

```python
import shap

# Initialize TreeExplainer with trained XGBoost model
explainer = shap.TreeExplainer(artifacts["xgb_model"])
shap_values = explainer.shap_values(X_test_df)
base_value = explainer.expected_value
```

The computed values are cached locally in `data/models/shap_cache.pkl` to enable instantaneous UI responsiveness inside Streamlit.

---

## 3. Global Feature Importance

Global importance is quantified as the **Mean Absolute SHAP Value** across all test instances:
$$I_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_j^{(i)}|$$

### Top Global Churn Drivers Discovered:
1. **Contract: Month-to-Month** ($\text{SHAP} \approx +0.72$): By far the strongest driver pushing customers toward churn.
2. **Tenure** ($\text{SHAP} \approx 0.58$): Low tenure strongly pushes toward churn, while high tenure provides substantial negative SHAP (retention anchor).
3. **Internet Service: Fiber Optic** ($\text{SHAP} \approx +0.41$): Pushes toward churn when unbundled with security/support.
4. **TotalCharges & MonthlyCharges** ($\text{SHAP} \approx 0.35$): Price sensitivity indicators.
5. **Tech Support: No** ($\text{SHAP} \approx +0.28$): Customers without technical support are much more vulnerable to churn when service interruptions occur.

---

## 4. Local Explainability: Top 3 Drivers Per Customer

To empower retention agents on the phone, the system extracts the **top 3 risk drivers** for any selected customer:

```python
def get_customer_top_drivers(sample_idx, shap_data, X_test_df, X_test_raw, top_n=3):
    shap_values = shap_data["shap_values"][sample_idx]
    feature_names = shap_data["feature_names"]
    
    # Filter positive SHAP values (factors increasing churn probability)
    positive_drivers = sorted(
        [(feat, val) for feat, val in zip(feature_names, shap_values) if val > 0],
        key=lambda x: x[1],
        reverse=True
    )
    return positive_drivers[:top_n]
```

### Example Customer Attribution:
- **Customer ID**: `7590-VHVEG`
  - **Driver 1**: `Contract: Month-to-month` (SHAP: `+0.642` $\rightarrow$ Increases Risk)
  - **Driver 2**: `tenure: 1 month` (SHAP: `+0.418` $\rightarrow$ Increases Risk)
  - **Driver 3**: `TechSupport: No` (SHAP: `+0.211` $\rightarrow$ Increases Risk)

This explanation is rendered as an interactive horizontal waterfall bar chart in Streamlit, clearly contrasting risk-increasing factors in red and retention-anchoring factors in blue.
