"""
Telco Churn Early-Warning Dashboard - End-to-End Execution Pipeline
Runs data acquisition, cleaning, baseline training, XGBoost tuning,
cost-benefit threshold analysis, SHAP computation, and LLM advice generation.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.data_loader import load_raw_data, clean_data, get_feature_splits
from src.eda import calculate_churn_metrics
from src.model_trainer import train_models
from src.threshold_optimizer import compute_cost_curve, find_optimal_threshold
from src.explainability import compute_or_load_shap, get_global_importance, get_customer_top_drivers
from src.llm_advisor import generate_llm_advice


def run_full_pipeline():
    print("=" * 75)
    print("📡 TELCO CUSTOMER CHURN EARLY-WARNING DASHBOARD - PIPELINE EXECUTION")
    print("=" * 75)
    start_time = time.time()

    # -------------------------------------------------------------
    # Step 1: Load and Clean Data
    # -------------------------------------------------------------
    print("\n[STEP 1/7] Loading and cleaning Telco Customer Churn dataset...")
    raw_df = load_raw_data()
    print(f"  ✓ Raw dataset loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    
    clean_df = clean_data(raw_df)
    print(f"  ✓ Cleaned dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
    print(f"  ✓ 'TotalCharges' blank strings converted & filled; nulls remaining: {clean_df['TotalCharges'].isnull().sum()}")

    # -------------------------------------------------------------
    # Step 2: Exploratory Data Metrics
    # -------------------------------------------------------------
    print("\n[STEP 2/7] Computing Churn & Cohort Statistics...")
    metrics = calculate_churn_metrics(clean_df)
    print(f"  ✓ Total Customers: {metrics['total_customers']:,}")
    print(f"  ✓ Churned Count:   {metrics['churn_count']:,} ({metrics['churn_rate']:.2%})")
    print(f"  ✓ Retained Count:  {metrics['non_churn_count']:,} ({1 - metrics['churn_rate']:.2%})")
    print(f"  ✓ Avg Monthly Bill: ${metrics['avg_monthly_charges']:.2f}")
    print(f"  ✓ Median Tenure:    {metrics['median_tenure']} months")

    # -------------------------------------------------------------
    # Step 3: Train Baseline and Tuned XGBoost Models
    # -------------------------------------------------------------
    print("\n[STEP 3/7] Training Baseline (Logistic Regression) & Tuned XGBoost...")
    artifacts = train_models()
    lr_m = artifacts["lr_metrics"]
    xgb_m = artifacts["xgb_metrics"]

    print("\n" + "-" * 75)
    print("🏆 MODEL EVALUATION COMPARISON MATRIX (Test Set n = 1,409)")
    print("-" * 75)
    comp_df = pd.DataFrame([
        {
            "Model": "Baseline Logistic Regression",
            "ROC-AUC": f"{lr_m['roc_auc']:.4f}",
            "PR-AUC": f"{lr_m['pr_auc']:.4f}",
            "Accuracy": f"{lr_m['accuracy']:.2%}",
            "Precision": f"{lr_m['precision']:.2%}",
            "Recall": f"{lr_m['recall']:.2%}",
            "F1-Score": f"{lr_m['f1']:.4f}",
        },
        {
            "Model": "Tuned XGBoost (scale_pos_weight)",
            "ROC-AUC": f"{xgb_m['roc_auc']:.4f}",
            "PR-AUC": f"{xgb_m['pr_auc']:.4f}",
            "Accuracy": f"{xgb_m['accuracy']:.2%}",
            "Precision": f"{xgb_m['precision']:.2%}",
            "Recall": f"{xgb_m['recall']:.2%}",
            "F1-Score": f"{xgb_m['f1']:.4f}",
        }
    ])
    print(comp_df.to_string(index=False))
    print(f"\n  ✓ 5-Fold Cross-Validation XGBoost ROC-AUC: {xgb_m['cv_roc_auc_mean']:.4f} ± {xgb_m['cv_roc_auc_std']:.4f}")

    # -------------------------------------------------------------
    # Step 4: Cost-Benefit Threshold Optimization
    # -------------------------------------------------------------
    print("\n[STEP 4/7] Optimizing Classification Decision Threshold...")
    cost_offer = 20.0  # Retention offer intervention cost
    cost_loss = 500.0  # Customer lifetime value lost if churns
    
    y_test = artifacts["y_test"].values
    xgb_probs = artifacts["xgb_probs"]
    
    df_cost = compute_cost_curve(y_test, xgb_probs, cost_offer=cost_offer, cost_loss=cost_loss)
    opt = find_optimal_threshold(df_cost, y_test, xgb_probs, cost_offer=cost_offer, cost_loss=cost_loss)

    print(f"  ✓ Standard 0.50 Threshold Cost:  ${opt['default_cost']:,.0f} (TP={opt['tp_default']}, FP={opt['fp_default']}, FN={opt['fn_default']})")
    print(f"  ✓ Cost-Optimal Threshold ({opt['optimal_threshold']:.2f}) Cost: ${opt['min_cost']:,.0f} (TP={opt['tp_optimal']}, FP={opt['fp_optimal']}, FN={opt['fn_optimal']})")
    print(f"  ✓ Direct Business Savings:       ${opt['savings_vs_default']:,.0f} saved by switching to optimal threshold!")

    # -------------------------------------------------------------
    # Step 5: TreeSHAP Feature Explainability
    # -------------------------------------------------------------
    print("\n[STEP 5/7] Computing TreeSHAP Global & Local Feature Attribution...")
    shap_data = compute_or_load_shap(artifacts["xgb_model"], artifacts["X_test_df"])
    global_imp = get_global_importance(shap_data, top_n=5)
    print("  ✓ Top 5 Global Churn Drivers (Mean |SHAP|):")
    for idx, row in global_imp.iterrows():
        print(f"     {idx+1}. {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")

    # -------------------------------------------------------------
    # Step 6: Sample Customer Deep-Dive & Top 3 Drivers
    # -------------------------------------------------------------
    print("\n[STEP 6/7] Extracting Local Customer Drivers for Sample High-Risk Account...")
    # Find highest churn risk customer in test set
    highest_idx = int(np.argmax(xgb_probs))
    sample_id = artifacts["ids_test"].iloc[highest_idx]
    sample_prob = xgb_probs[highest_idx]

    top_drivers = get_customer_top_drivers(
        highest_idx,
        shap_data,
        artifacts["X_test_df"],
        artifacts["X_test_raw"],
        top_n=3,
    )
    print(f"  ✓ Selected Customer ID: {sample_id} (Predicted Churn Probability: {sample_prob:.1%})")
    for i, d in enumerate(top_drivers, 1):
        print(f"     Driver {i}: {d['readable_label']}{d['raw_value_str']} -> {d['direction']} (SHAP: {d['shap_value']:+.3f})")

    # -------------------------------------------------------------
    # Step 7: LLM Grounded Retention Advice Generation
    # -------------------------------------------------------------
    print("\n[STEP 7/7] Synthesizing Grounded LLM Retention Advice...")
    sample_raw_row = artifacts["X_test_raw"].iloc[highest_idx].to_dict()
    advice = generate_llm_advice(
        customer_id=sample_id,
        churn_probability=sample_prob,
        threshold=opt["optimal_threshold"],
        top_drivers=top_drivers,
        customer_attributes=sample_raw_row,
    )
    print(f"  ✓ Advisor Engine: {advice['source']}")
    print("  ✓ Generated Advice Content:")
    print("-" * 50)
    print(advice["content"])
    print("-" * 50)

    elapsed = time.time() - start_time
    print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(f"   Launch the interactive Streamlit dashboard with: streamlit run app.py")
    print("=" * 75)


if __name__ == "__main__":
    run_full_pipeline()
