# Project Q&A: Customer Churn Early-Warning Dashboard
**Comprehensive Question & Answer Reference**  
**Author**: Om Joshi  

---

## Category 1: Business Context & Return on Investment (ROI)

### Q1: Why is customer churn such a critical problem in the telecommunications sector?
**Answer**: Telecommunications is a capital-intensive utility with enormous fixed infrastructure costs (cell towers, fiber trenches, satellite networks). Customer acquisition cost (CAC) ranges between $350 and $600 per subscriber. If a customer churns within their first 12–24 months, the company never recovers the initial acquisition investment. Because acquiring a new customer is 5x to 7x more expensive than retaining an existing one, reducing churn by even 1% to 2% saves millions of dollars in annual recurring revenue.

### Q2: How does this project differ from a typical churn prediction tutorial?
**Answer**: Most churn projects stop at training a model and printing an accuracy score. This project delivers an enterprise-ready system:
1. It cleans and validates raw data, correctly resolving edge cases like the `TotalCharges` 0-tenure trap.
2. It handles class imbalance natively via `scale_pos_weight` rather than noisy synthetic oversampling.
3. It optimizes the classification cutoff based on a real-world **financial cost matrix** ($20 offer vs. $500 lost customer).
4. It provides **game-theoretic explainability** (TreeSHAP) at both global and customer-specific levels.
5. It uses an **LLM** to bridge the last mile between ML output and frontline retention actions, producing grounded, human-readable briefs for customer service representatives.
6. It packages everything into an interactive, 3-page Streamlit dashboard ready for production deployment.

### Q3: How much money does the cost-benefit threshold optimization actually save?
**Answer**: On our test cohort of 1,409 customers:
- Standard 0.50 threshold incurs a total business loss of **$124,460** (including 78 missed churners costing $500 each).
- The cost-optimal threshold lowers total business loss to **$115,880** by capturing 70 additional churners, incurring only 8 false negatives.
- **Direct Net Savings**: **$8,580 on the test split alone**.
- Scaled to a mid-sized telecom operator with 100,000 active subscribers, this threshold adjustment saves **~$608,000 annually** with zero additional engineering cost.

---

## Category 2: Data Engineering & Preprocessing

### Q4: What is the `TotalCharges` bug in the Telco dataset and how was it solved?
**Answer**: In the raw Telco dataset, `TotalCharges` loads as a string object rather than a float. Eleven customer records contain blank spaces (`" "`). 
When inspected, all 11 customers have `tenure == 0`, meaning they signed up on the exact day of data extraction and have never been billed. 
Dropping these records causes data loss, while imputing mean or median introduces severe data leakage (giving brand-new customers $2,000+ in historical charges). 
Our pipeline strips whitespace, coerces values to float with `pd.to_numeric(..., errors='coerce')`, and explicitly imputes `0.00`, preserving data integrity and physical truth.

### Q5: How was data leakage prevented during feature engineering?
**Answer**: In accordance with strict ML best practices:
1. The dataset was split into **80% Training** and **20% Testing** sets *before* fitting any transformer.
2. The `ColumnTransformer` (fitting `StandardScaler` on numericals and `OneHotEncoder(drop='first')` on categoricals) was fitted **solely on `X_train`**.
3. The test set (`X_test`) was strictly transformed using the frozen, fitted pipeline parameters without recalculating means, variances, or category vocabularies.

### Q6: Why was `drop='first'` used in OneHotEncoder?
**Answer**: Dropping the first dummy column prevents the "dummy variable trap" (perfect multicollinearity) in our baseline Logistic Regression model. If a feature has $k$ categories, only $k-1$ dummy variables are needed to represent all categories; the omitted level serves as the reference category.

---

## Category 3: Machine Learning & XGBoost Modeling

### Q7: Why is Accuracy an inappropriate metric for evaluating churn models?
**Answer**: In our dataset, the overall churn rate is **26.54%** (imbalanced ~3:1 ratio). A naive, non-learning baseline that simply predicts "Nobody Churns" would achieve an accuracy of **73.46%**, yet it would detect 0% of churners, producing zero business value. 
Therefore, we evaluate models primarily using **Precision-Recall AUC (PR-AUC)**, **ROC-AUC**, **Recall**, and **Financial Expected Cost**.

### Q8: How does XGBoost's `scale_pos_weight` work?
**Answer**: `scale_pos_weight` is a hyperparameter designed specifically for binary classification with class imbalance. It represents the ratio of negative (retained) to positive (churned) training instances:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{4,139}{1,495} \approx 2.77$$
Internally, XGBoost multiplies the gradient and hessian of the positive class by 2.77. This forces the decision trees to treat false negatives with 2.77x higher penalty, boosting recall for minority churners without needing to create artificial rows via SMOTE or undersampling.

### Q9: What were the final model comparison results between Baseline and XGBoost?
**Answer**:
- **ROC-AUC**: Tuned XGBoost achieved **0.8460** vs. Baseline Logistic Regression's **0.8417**.
- **PR-AUC**: Tuned XGBoost achieved **0.6650** vs. Baseline's **0.6331** (a **+3.2 point improvement** in minority class precision-recall).
- **Recall (@0.50)**: Tuned XGBoost captured **79.68%** of all churners vs. Baseline's **78.34%**.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation confirmed model stability at **0.8477 ± 0.0113 ROC-AUC**.

---

## Category 4: Explainability & TreeSHAP

### Q10: How does TreeSHAP compute feature importance?
**Answer**: TreeSHAP computes exact Shapley values derived from cooperative game theory. Each feature is treated as a player in a coalition contributing toward the model's log-odds output. 
While traditional KernelSHAP requires exponential sampling ($\mathcal{O}(2^M)$), `shap.TreeExplainer` evaluates feature interactions across ensemble trees in polynomial time ($\mathcal{O}(TLD^2)$) by recursively tracing sample partitions through the tree nodes.

### Q11: What are the top global churn drivers identified by TreeSHAP?
**Answer**:
1. **Contract: Month-to-Month**: Pushes heavily toward churn (+0.55 mean SHAP).
2. **Tenure**: Low tenure is a massive churn risk factor; high tenure (>24 months) provides strong negative SHAP (retention anchor).
3. **Internet Service: Fiber Optic**: Increases churn risk due to premium fees and service stability concerns when unbundled with support.
4. **Monthly Charges**: Higher fees directly correlate with elevated price sensitivity.
5. **Tech Support: No**: Absence of technical support amplifies churn likelihood when technical issues occur.

### Q12: How are local customer drivers extracted for frontline reps?
**Answer**: For any selected customer, `get_customer_top_drivers()` takes their local SHAP vector, filters for features with positive SHAP values (forces pushing toward churn), sorts them in descending order of impact, cleans the feature names into plain English (e.g., `cat__Contract_Month-to-month` $\rightarrow$ `Contract: Month-to-Month`), and attaches their raw feature value. The top 3 are rendered directly in the dashboard.

---

## Category 5: Generative AI & Streamlit Deployment

### Q13: Why combine an LLM with traditional Machine Learning?
**Answer**: Traditional ML provides the **statistical rigor** (who is leaving and why), but its output (probabilities, SHAP log-odds) is inaccessible to non-technical frontline staff. 
The LLM serves as a **translation and strategic synthesizer layer**. By feeding the model's top 3 SHAP drivers to an LLM, we generate an instant 2-sentence executive summary and a concrete retention action that a call center agent can read aloud during a live call.

### Q14: How are LLM hallucinations prevented?
**Answer**: We enforce **strict grounding**:
1. The prompt injects only the verified customer ID, predicted probability, and top 3 algorithmic SHAP drivers.
2. The prompt includes an explicit directive: *"The summary and retention action MUST be strictly grounded in the drivers listed above. Do not invent unrelated reasons."*
3. The output format is constrained to exactly two sentences for the summary and one sentence for the action.

### Q15: What happens if the user runs the dashboard offline without a Gemini API key?
**Answer**: `src/llm_advisor.py` features an **Intelligent Grounded Fallback Engine**. If no API key is detected or if an API call fails, the fallback engine uses deterministic domain logic to match the customer's top drivers (e.g., Month-to-Month, Fiber Optic, Low Tenure) to pre-engineered, proven retention strategies. The dashboard never crashes and always displays high-quality, grounded advice.

### Q16: How is sub-second latency achieved in the Streamlit application?
**Answer**: We utilize Streamlit's built-in memory caching decorators:
- `@st.cache_data`: Caches the cleaned raw dataset.
- `@st.cache_resource`: Caches the trained model artifacts, fitted transformers, and precomputed TreeSHAP matrices.
As a result, selecting different customers, filtering lists, or navigating between pages renders in **under 50 milliseconds**.
