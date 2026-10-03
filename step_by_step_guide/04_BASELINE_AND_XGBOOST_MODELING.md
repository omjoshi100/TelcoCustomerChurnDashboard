# Step 4: Baseline and XGBoost Modeling

This guide covers model training, preprocessing pipeline construction, handling class imbalance, cross-validation, and performance evaluation.

---

## 1. Strict Preprocessing Pipeline Architecture

In accordance with strict ML best practices, feature transformations are fitted **only** on the training set and applied to the test set to avoid data leakage:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["tenure", "MonthlyCharges", "TotalCharges"]),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)
```

- **Stratified Split**: 80% Train (`n=5,634`), 20% Test (`n=1,409`), stratified on `Churn_Num` to maintain identical class proportions.
- **Categoricals**: `drop='first'` prevents collinearity in the linear baseline; `handle_unknown='ignore'` provides inference robustness.
- **Continuous Features**: Z-score standardized to mean 0, variance 1.

---

## 2. Baseline Model: Logistic Regression

We first establish a simple, interpretable linear baseline using `scikit-learn`'s `LogisticRegression`.

```python
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr_model.fit(X_train_df, y_train)
```

- Setting `class_weight='balanced'` automatically weights classes inversely proportional to class frequencies.
- Evaluated on test set:
  - **ROC-AUC**: ~0.846
  - **PR-AUC**: ~0.648
  - **Recall**: ~79.1% (catches churners well, but precision is moderate at ~50.8%)

---

## 3. Champion Model: Tuned XGBoost with `scale_pos_weight`

XGBoost is a state-of-the-art gradient-boosted decision tree algorithm. In imbalanced problems, it natively supports `scale_pos_weight`:

### Computing `scale_pos_weight`:
$$\text{scale\_pos\_weight} = \frac{\text{Negative Samples (Retained)}}{\text{Positive Samples (Churned)}} = \frac{4,139}{1,495} \approx 2.77$$

This parameter scales the gradient for positive instances, forcing trees to prioritize the minority churn class without artificially oversampling rows.

### Tuned Hyperparameters:
```python
xgb_model = xgb.XGBClassifier(
    n_estimators=180,
    max_depth=4,           # Shallow depth prevents overfitting on noisy customer segments
    learning_rate=0.04,     # Conservative shrinkage
    subsample=0.85,         # Row bagging
    colsample_bytree=0.80,  # Feature bagging
    scale_pos_weight=2.77,  # Exact negative/positive ratio
    gamma=0.2,             # Minimum loss reduction for partition
    reg_alpha=0.1,         # L1 Lasso regularization on leaf weights
    reg_lambda=1.0,        # L2 Ridge regularization on leaf weights
    eval_metric="logloss",
    random_state=42,
)
```

---

## 4. 5-Fold Stratified Cross-Validation

Before final testing, the XGBoost model undergoes 5-Fold Stratified Cross-Validation on the training partition:
- **CV ROC-AUC**: `0.847 ± 0.012`
- **CV PR-AUC**: `0.654 ± 0.018`

This confirms consistent, stable performance across different demographic slices.

---

## 5. Model Evaluation Comparison Table

| Metric | Baseline (Logistic Regression) | Tuned XGBoost (`scale_pos_weight`) | Business Interpretation |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | 0.8462 | **0.8502** | Superior discriminatory power across all decision thresholds |
| **PR-AUC** | 0.6485 | **0.6587** | Higher area under Precision-Recall curve in minority class |
| **Recall (@0.5)** | 79.14% | **81.02%** | Catches 8 out of 10 leaving customers |
| **Precision (@0.5)** | 50.84% | **52.41%** | Fewer wasted retention calls |
| **F1-Score** | 0.6190 | **0.6364** | Harmonic mean balance favored by XGBoost |
| **Accuracy** | 74.31% | **76.08%** | Baseline accuracy |

*Note: In imbalanced churn prediction, Accuracy is a deceptive metric. A naive model predicting "Nobody Churns" would achieve 73.5% accuracy but 0% recall. We focus on PR-AUC, ROC-AUC, and financial cost-benefit.*
