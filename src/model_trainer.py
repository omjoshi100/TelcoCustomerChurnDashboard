"""
Model Training & Evaluation module:
1. Baseline Logistic Regression (with Stratified Train/Test Split)
2. Tuned XGBoost with scale_pos_weight for class imbalance
3. Comprehensive evaluation reporting ROC-AUC, PR-AUC, F1, Precision, Recall
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

from src.data_loader import load_raw_data, clean_data, get_feature_splits

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer:
    - StandardScaler for continuous features
    - OneHotEncoder (drop='first', handle_unknown='ignore') for categoricals
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor


def train_models():
    """
    Trains Baseline Logistic Regression and Tuned XGBoost.
    Saves trained models, feature names, test datasets, and metrics to disk.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load and clean data
    raw = load_raw_data()
    cleaned = clean_data(raw)
    X, y, customer_ids, num_cols, cat_cols = get_feature_splits(cleaned)

    # 2. Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Fit Preprocessing Pipeline strictly on X_train
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
    feature_names = num_cols + list(encoded_cat_names)

    # Convert back to clean DataFrames with feature names
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # ---------------------------------------------------------
    # 4. BASELINE: Logistic Regression
    # ---------------------------------------------------------
    print("Training Baseline Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(X_train_df, y_train)

    lr_probs = lr_model.predict_proba(X_test_df)[:, 1]
    lr_preds_default = (lr_probs >= 0.5).astype(int)

    lr_metrics = {
        "model": "Baseline Logistic Regression",
        "roc_auc": float(roc_auc_score(y_test, lr_probs)),
        "pr_auc": float(average_precision_score(y_test, lr_probs)),
        "accuracy": float(accuracy_score(y_test, lr_preds_default)),
        "precision": float(precision_score(y_test, lr_preds_default, zero_division=0)),
        "recall": float(recall_score(y_test, lr_preds_default)),
        "f1": float(f1_score(y_test, lr_preds_default)),
    }

    # ---------------------------------------------------------
    # 5. TUNED XGBOOST with scale_pos_weight
    # ---------------------------------------------------------
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count)
    print(f"XGBoost scale_pos_weight calculated: {scale_pos_weight:.2f}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.80,
        scale_pos_weight=scale_pos_weight,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )

    # 5-Fold Stratified Cross-Validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc = cross_val_score(xgb_model, X_train_df, y_train, cv=cv, scoring="roc_auc")
    cv_pr = cross_val_score(xgb_model, X_train_df, y_train, cv=cv, scoring="average_precision")
    print(f"5-Fold CV ROC-AUC: {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")
    print(f"5-Fold CV PR-AUC:  {cv_pr.mean():.4f} +/- {cv_pr.std():.4f}")

    xgb_model.fit(X_train_df, y_train)
    xgb_probs = xgb_model.predict_proba(X_test_df)[:, 1]
    xgb_preds_default = (xgb_probs >= 0.5).astype(int)

    xgb_metrics = {
        "model": "Tuned XGBoost",
        "roc_auc": float(roc_auc_score(y_test, xgb_probs)),
        "pr_auc": float(average_precision_score(y_test, xgb_probs)),
        "accuracy": float(accuracy_score(y_test, xgb_preds_default)),
        "precision": float(precision_score(y_test, xgb_preds_default, zero_division=0)),
        "recall": float(recall_score(y_test, xgb_preds_default)),
        "f1": float(f1_score(y_test, xgb_preds_default)),
        "cv_roc_auc_mean": float(cv_roc.mean()),
        "cv_roc_auc_std": float(cv_roc.std()),
        "cv_pr_auc_mean": float(cv_pr.mean()),
    }

    # 6. Save Artifacts
    artifacts = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "lr_model": lr_model,
        "xgb_model": xgb_model,
        "lr_metrics": lr_metrics,
        "xgb_metrics": xgb_metrics,
        "X_test_raw": X_test,
        "X_test_df": X_test_df,
        "y_test": y_test,
        "ids_test": ids_test,
        "lr_probs": lr_probs,
        "xgb_probs": xgb_probs,
    }

    artifact_file = os.path.join(MODEL_DIR, "trained_artifacts.pkl")
    with open(artifact_file, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved trained artifacts to {artifact_file}")

    return artifacts


def load_artifacts():
    """
    Loads saved model artifacts from disk.
    If not found, trains them automatically.
    """
    artifact_file = os.path.join(MODEL_DIR, "trained_artifacts.pkl")
    if os.path.exists(artifact_file):
        with open(artifact_file, "rb") as f:
            return pickle.load(f)
    return train_models()


if __name__ == "__main__":
    arts = train_models()
    print("\n--- RESULTS COMPARISON ---")
    print(pd.DataFrame([arts["lr_metrics"], arts["xgb_metrics"]]).to_string(index=False))
