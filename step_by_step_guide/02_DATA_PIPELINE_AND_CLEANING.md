# Step 2: Data Pipeline and Data Cleaning

This guide explains the data ingestion pipeline, the source datasets, and the critical data cleaning steps performed on the Telco Customer Churn dataset.

---

## 1. Dataset Origin & Sources

The project uses the standard benchmark **Telco Customer Churn** dataset originally published by IBM Cognos Analytics and hosted on Kaggle (`blastchar/telco-customer-churn`) and OpenML (`data_id=42178`).

To ensure 100% automated reliability and zero manual download requirements, `src/data_loader.py` implements a 3-tier fallback strategy:
1. **Local Disk Cache**: Checks `data/telco_churn.csv`.
2. **Primary CDN / GitHub Raw Mirrors**: Downloads directly from verified raw mirrors.
3. **OpenML API Fallback**: Queries `sklearn.datasets.fetch_openml(data_id=42178)`.

```python
from src.data_loader import load_raw_data, clean_data

# Ingest raw data
raw_df = load_raw_data()
print(f"Shape: {raw_df.shape}")  # (7043, 21)
```

---

## 2. Key Attributes & Schema

The dataset comprises **7,043 customers** with **21 attributes**:
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Information**: `tenure` (months with company), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`
- **Charges**: `MonthlyCharges` (float), `TotalCharges` (originally string with blanks!)
- **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target Variable**: `Churn` ("Yes" or "No")

---

## 3. The `TotalCharges` Data Quality Bug

### Problem Statement:
In the raw Telco dataset, the `TotalCharges` column loads as an **object (text string)** rather than a numeric float. 
When examining the data, 11 customer records contain whitespace strings `" "` or empty values.

Why does this happen?
These 11 customers have `tenure == 0`. They signed up on the very day of data extraction and have had **zero billing cycles generated**. Standard conversion tools throw parsing errors or convert them to `NaN`.

### Production Solution in `src/data_loader.py`:
```python
# Convert text to numeric, coercing spaces/blanks to NaN
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

# For customers with tenure == 0, impute TotalCharges to 0.0
df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)
```

**Why Impute 0.0 instead of Median or Mean?**
Imputing mean or median would introduce significant bias: a new customer who has been with the provider for 0 days has mathematically accumulated $0.00 in total charges. Imputing $2,000+ would mislead tree-based models and linear models alike.

---

## 4. Target Variable Encoding

The target column `Churn` contains `"Yes"` and `"No"`. We convert this into a binary numeric indicator:
- `1` = Churned (positive class, minority)
- `0` = Retained (negative class, majority)

```python
df_clean["Churn_Num"] = (df_clean["Churn"].str.lower() == "yes").astype(int)
```

---

## 5. Verification Command

You can run data validation directly using python:

```powershell
python -c "from src.data_loader import load_raw_data, clean_data; df = clean_data(load_raw_data()); print('Null TotalCharges:', df['TotalCharges'].isnull().sum()); print('Churn distribution:\n', df['Churn_Num'].value_counts(normalize=True))"
```
Output:
- `Null TotalCharges: 0`
- `0 (Retained): ~73.46%`
- `1 (Churned): ~26.54%` (matching the expected ~27% churn rate)
