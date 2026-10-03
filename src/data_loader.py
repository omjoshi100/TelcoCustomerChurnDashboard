"""
Data Loader and Preprocessing module for Telco Customer Churn.
Fixes TotalCharges, handles blank strings, and prepares features/targets.
"""

import os
import io
import requests
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

DATA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")
RAW_DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def load_raw_data() -> pd.DataFrame:
    """
    Loads raw Telco Customer Churn data from:
    1. Local cache if present
    2. Reliable raw GitHub mirrors
    3. OpenML dataset ID 42178
    """
    os.makedirs(os.path.dirname(DATA_CACHE_PATH), exist_ok=True)
    
    # 1. Check local cache
    if os.path.exists(DATA_CACHE_PATH):
        try:
            df = pd.read_csv(DATA_CACHE_PATH)
            if len(df) > 0 and "Churn" in df.columns:
                return df
        except Exception as e:
            print(f"Warning: could not read cached file: {e}")

    # 2. Try raw mirrors
    for url in RAW_DATA_URLS:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                df.to_csv(DATA_CACHE_PATH, index=False)
                return df
        except Exception:
            continue

    # 3. Fallback to OpenML
    try:
        data = fetch_openml(data_id=42178, as_frame=True, parser="auto")
        df = data.frame
        df.to_csv(DATA_CACHE_PATH, index=False)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Telco Churn dataset: {e}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Telco dataframe:
    - Strips whitespace from string columns
    - Coerces 'TotalCharges' from text to float
    - Imputes missing TotalCharges for 0-tenure accounts to 0.0
    - Maps 'Churn' to binary 1/0
    """
    df_clean = df.copy()
    
    # Strip whitespace from column names and string values
    df_clean.columns = [c.strip() for c in df_clean.columns]
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Fix TotalCharges: blank strings ' ' or '' become NaN, then fill with 0.0
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    null_count = df_clean["TotalCharges"].isnull().sum()
    if null_count > 0:
        # These 11 customers have tenure == 0 (brand new signups, 0 bills generated)
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Encode SeniorCitizen to standard string category if integer
    df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(int)

    # Target column mapping
    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == object or df_clean["Churn"].dtype == "string":
            df_clean["Churn_Num"] = (df_clean["Churn"].str.lower() == "yes").astype(int)
        else:
            df_clean["Churn_Num"] = df_clean["Churn"].astype(int)

    return df_clean


def get_feature_splits(df_clean: pd.DataFrame):
    """
    Returns X (features), y (target), customer_ids, and feature column types.
    """
    customer_ids = df_clean["customerID"] if "customerID" in df_clean.columns else df_clean.index.astype(str)
    
    drop_cols = ["customerID", "Churn", "Churn_Num"]
    drop_cols = [c for c in drop_cols if c in df_clean.columns]
    
    X = df_clean.drop(columns=drop_cols)
    y = df_clean["Churn_Num"]
    
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    return X, y, customer_ids, num_cols, cat_cols


if __name__ == "__main__":
    raw = load_raw_data()
    print(f"Loaded raw data: {raw.shape}")
    cleaned = clean_data(raw)
    print(f"Cleaned data: {cleaned.shape}")
    print(f"Churn distribution:\n{cleaned['Churn_Num'].value_counts(normalize=True)}")
    print(f"TotalCharges nulls after fix: {cleaned['TotalCharges'].isnull().sum()}")
