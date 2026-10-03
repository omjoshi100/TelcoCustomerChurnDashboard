"""
Customer Churn Early-Warning Dashboard
Streamlit Application with 3 Interactive Pages:
1. Risk-Ranked Customer List
2. Customer Detail & SHAP Explainability + LLM Retention Advice
3. Segment Overview & Business Cost Optimization
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path so src imports work smoothly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_raw_data, clean_data
from src.eda import (
    calculate_churn_metrics,
    plot_churn_by_contract,
    plot_churn_by_tenure,
    plot_churn_by_monthly_charges,
    plot_churn_by_internet_service,
)
from src.model_trainer import load_artifacts
from src.threshold_optimizer import (
    compute_cost_curve,
    find_optimal_threshold,
    plot_cost_vs_threshold,
)
from src.explainability import (
    compute_or_load_shap,
    get_global_importance,
    plot_global_importance,
    get_customer_top_drivers,
    plot_customer_shap_waterfall,
)
from src.llm_advisor import generate_llm_advice

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Telco Churn Early-Warning Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN CSS STYLING ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 14px;
        color: #555555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #1976D2;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #777777;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #212121;
    }
    .risk-high {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .risk-medium {
        background-color: #FFF3E0;
        color: #EF6C00;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .risk-low {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .driver-tag {
        display: inline-block;
        background-color: #F1F8FF;
        border: 1px solid #C8E1FF;
        color: #0366D6;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_data():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    return raw, cleaned


@st.cache_resource
def get_cached_artifacts():
    return load_artifacts()


# Load data and ML model artifacts
raw_df, clean_df = get_cached_data()
artifacts = get_cached_artifacts()
shap_data = compute_or_load_shap(artifacts["xgb_model"], artifacts["X_test_df"])

# Precompute cost curve and optimal threshold
y_test = artifacts["y_test"].values
xgb_probs = artifacts["xgb_probs"]
df_cost = compute_cost_curve(y_test, xgb_probs, cost_offer=20.0, cost_loss=500.0)
optimal_info = find_optimal_threshold(df_cost, y_test, xgb_probs, cost_offer=20.0, cost_loss=500.0)

# Build unified test dataset view with customer info & probabilities
test_indices = artifacts["X_test_raw"].index
test_customers = clean_df.loc[test_indices].copy()
test_customers["Churn_Prob"] = xgb_probs
test_customers["Churn_Prob_Pct"] = (test_customers["Churn_Prob"] * 100).round(1)

# --- SIDEBAR NAVIGATION & SETTINGS ---
st.sidebar.image(
    "https://img.icons8.com/fluency/96/artificial-intelligence.png",
    width=64,
)
st.sidebar.markdown("### **Telco Churn AI Studio**")
st.sidebar.markdown("*XGBoost + TreeSHAP + LLM Advisor*")

page = st.sidebar.radio(
    "Navigate",
    [
        "📋 1. Risk-Ranked Customer List",
        "🔍 2. Customer Deep-Dive & SHAP",
        "📊 3. Segment Overview & Cost Matrix",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### **Cost-Benefit Parameters**")
retention_cost = st.sidebar.number_input("Retention Offer Cost ($)", min_value=5.0, max_value=100.0, value=20.0, step=5.0)
churn_loss = st.sidebar.number_input("Customer Churn Loss ($)", min_value=100.0, max_value=2000.0, value=500.0, step=50.0)

# Decision threshold slider with quick-set buttons
optimal_t = optimal_info["optimal_threshold"]
use_optimal = st.sidebar.checkbox("Use Cost-Optimal Cutoff", value=True)
if use_optimal:
    active_threshold = optimal_t
    st.sidebar.caption(f"Cost-Minimizing Threshold: **{active_threshold:.2f}**")
else:
    active_threshold = st.sidebar.slider("Manual Decision Cutoff", 0.05, 0.95, 0.50, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("#### **LLM Advisor Settings**")
gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", help="Leave blank to use grounded expert domain engine")


# ==============================================================================
# PAGE 1: RISK-RANKED CUSTOMER LIST
# ==============================================================================
if page == "📋 1. Risk-Ranked Customer List":
    st.markdown('<div class="main-header">📋 Risk-Ranked Customer Priority List</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Active Intervention Cutoff: <b>{active_threshold:.1%}</b> | '
        f'Prioritizing high-churn accounts before the next billing cycle.</div>',
        unsafe_allow_html=True,
    )

    # Classify accounts
    test_customers["Action_Required"] = np.where(test_customers["Churn_Prob"] >= active_threshold, "Flagged for Offer", "Low Risk")
    test_customers["Risk_Tier"] = pd.cut(
        test_customers["Churn_Prob"],
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    flagged_df = test_customers[test_customers["Action_Required"] == "Flagged for Offer"].sort_values(by="Churn_Prob", ascending=False)
    
    # Summary Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Test Population</div>'
            f'<div class="metric-value">{len(test_customers):,}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Flagged for Outreach</div>'
            f'<div class="metric-value" style="color:#C62828;">{len(flagged_df):,} ({len(flagged_df)/len(test_customers):.1%})</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        campaign_cost = len(flagged_df) * retention_cost
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Campaign Outreach Cost</div>'
            f'<div class="metric-value">${campaign_cost:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        potential_loss = int(flagged_df["Churn_Num"].sum()) * churn_loss
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Identified Churn Exposure</div>'
            f'<div class="metric-value" style="color:#1565C0;">${potential_loss:,.0f}</div></div>',
            unsafe_allow_html=True,
        )

    # Filter Controls
    st.markdown("#### **Filters**")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        contract_filter = st.multiselect("Contract Type", options=clean_df["Contract"].unique(), default=clean_df["Contract"].unique())
    with fcol2:
        internet_filter = st.multiselect("Internet Service", options=clean_df["InternetService"].unique(), default=clean_df["InternetService"].unique())
    with fcol3:
        risk_filter = st.multiselect("Risk Tier", options=["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk"])

    filtered_view = test_customers[
        test_customers["Contract"].isin(contract_filter)
        & test_customers["InternetService"].isin(internet_filter)
        & test_customers["Risk_Tier"].isin(risk_filter)
    ].sort_values(by="Churn_Prob", ascending=False)

    display_cols = [
        "customerID",
        "Risk_Tier",
        "Churn_Prob_Pct",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "InternetService",
        "PaymentMethod",
        "Action_Required",
    ]

    st.dataframe(
        filtered_view[display_cols].rename(
            columns={
                "customerID": "Customer ID",
                "Risk_Tier": "Risk Tier",
                "Churn_Prob_Pct": "Churn Prob (%)",
                "tenure": "Tenure (Mo)",
                "MonthlyCharges": "Monthly ($)",
                "TotalCharges": "Total ($)",
                "InternetService": "Internet",
                "PaymentMethod": "Payment Method",
                "Action_Required": "Recommendation",
            }
        ),
        use_container_width=True,
        height=450,
    )

    # Download action list
    csv_data = filtered_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Retention Target List (CSV)",
        data=csv_data,
        file_name="telco_retention_priority_list.csv",
        mime="text/csv",
    )


# ==============================================================================
# PAGE 2: CUSTOMER DETAIL & SHAP EXPLAINABILITY + LLM ADVICE
# ==============================================================================
elif page == "🔍 2. Customer Deep-Dive & SHAP":
    st.markdown('<div class="main-header">🔍 Customer Deep-Dive, SHAP & AI Retention Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Select an individual customer to inspect their feature attribution and generate tailored retention actions.</div>',
        unsafe_allow_html=True,
    )

    # Customer Selection
    test_id_list = list(test_customers["customerID"])
    # Default to a high-risk customer for immediate impact
    high_risk_defaults = test_customers[test_customers["Churn_Prob"] > 0.70]
    default_id = high_risk_defaults.iloc[0]["customerID"] if len(high_risk_defaults) > 0 else test_id_list[0]
    default_idx = test_id_list.index(default_id) if default_id in test_id_list else 0

    selected_customer_id = st.selectbox(
        "Select Customer ID for Deep Analysis:",
        options=test_id_list,
        index=default_idx,
    )

    # Extract customer row
    cust_row = test_customers[test_customers["customerID"] == selected_customer_id].iloc[0]
    cust_idx_in_test = test_id_list.index(selected_customer_id)
    churn_p = float(cust_row["Churn_Prob"])

    # Customer Profile Cards
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Churn Probability</div>'
            f'<div class="metric-value" style="color:{"#C62828" if churn_p >= active_threshold else "#2E7D32"}">{churn_p:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with pcol2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Contract & Tenure</div>'
            f'<div class="metric-value">{cust_row["Contract"]} ({cust_row["tenure"]} mo)</div></div>',
            unsafe_allow_html=True,
        )
    with pcol3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Monthly Charges</div>'
            f'<div class="metric-value">${cust_row["MonthlyCharges"]:.2f} / mo</div></div>',
            unsafe_allow_html=True,
        )
    with pcol4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Internet & Support</div>'
            f'<div class="metric-value">{cust_row["InternetService"]} | {cust_row["TechSupport"]}</div></div>',
            unsafe_allow_html=True,
        )

    # Extract Top 3 Drivers
    top_3_drivers = get_customer_top_drivers(
        cust_idx_in_test,
        shap_data,
        artifacts["X_test_df"],
        artifacts["X_test_raw"],
        top_n=3,
    )

    st.markdown("---")
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown("#### **TreeSHAP Feature Attribution**")
        fig_waterfall = plot_customer_shap_waterfall(cust_idx_in_test, shap_data, top_k=7)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.markdown("#### **Top 3 Algorithmic Churn Drivers:**")
        for i, d in enumerate(top_3_drivers, 1):
            direction_color = "#C62828" if d["shap_value"] > 0 else "#2E7D32"
            st.markdown(
                f"- **Driver {i}:** <span style='color:{direction_color};font-weight:bold;'>{d['readable_label']}</span> "
                f"`{d['raw_value_str']}` — *SHAP: {d['shap_value']:+.3f} ({d['direction']})*",
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("#### **🤖 AI Retention Strategist**")
        st.caption("Prompted with exact customer SHAP drivers; strictly grounded output.")

        with st.spinner("Generating grounded retention advice..."):
            advice_result = generate_llm_advice(
                customer_id=selected_customer_id,
                churn_probability=churn_p,
                threshold=active_threshold,
                top_drivers=top_3_drivers,
                customer_attributes=cust_row.to_dict(),
                api_key=gemini_api_key,
            )

        st.info(f"**Advisor Engine:** {advice_result['source']}")
        st.markdown(advice_result["content"])

        with st.expander("View Model Prompt Sent to LLM"):
            st.code(advice_result.get("prompt_used", "Prompt not logged"), language="text")


# ==============================================================================
# PAGE 3: SEGMENT OVERVIEW & BUSINESS COST MATRIX
# ==============================================================================
elif page == "📊 3. Segment Overview & Cost Matrix":
    st.markdown('<div class="main-header">📊 Segment Analysis & Business Cost Optimization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Review historical segment distributions and the cost-benefit trade-offs of the classification threshold.</div>',
        unsafe_allow_html=True,
    )

    # 1. Model Performance Benchmark Table
    st.markdown("#### **1. Model Evaluation Matrix: Baseline vs Tuned XGBoost**")
    lr_m = artifacts["lr_metrics"]
    xgb_m = artifacts["xgb_metrics"]

    comparison_df = pd.DataFrame([
        {
            "Model": "Baseline Logistic Regression",
            "ROC-AUC": f"{lr_m['roc_auc']:.4f}",
            "PR-AUC": f"{lr_m['pr_auc']:.4f}",
            "Accuracy": f"{lr_m['accuracy']:.1%}",
            "Precision": f"{lr_m['precision']:.1%}",
            "Recall": f"{lr_m['recall']:.1%}",
            "F1-Score": f"{lr_m['f1']:.4f}",
        },
        {
            "Model": "Tuned XGBoost (with scale_pos_weight)",
            "ROC-AUC": f"{xgb_m['roc_auc']:.4f}",
            "PR-AUC": f"{xgb_m['pr_auc']:.4f}",
            "Accuracy": f"{xgb_m['accuracy']:.1%}",
            "Precision": f"{xgb_m['precision']:.1%}",
            "Recall": f"{xgb_m['recall']:.1%}",
            "F1-Score": f"{xgb_m['f1']:.4f}",
        },
    ])
    st.table(comparison_df)

    st.markdown("---")
    # 2. Cost-Benefit Optimization Curve
    st.markdown("#### **2. Business Cost Curve (Threshold Tuning)**")
    st.markdown(
        f"By balancing the **asymmetric costs** between sending a retention offer (**${retention_cost:,.0f}**) "
        f"and losing a customer (**${churn_loss:,.0f}**), we find that the standard `0.50` ML threshold is sub-optimal. "
        f"Shifting to the cost-optimal cutoff (**{optimal_t:.2f}**) prevents expensive customer walk-aways."
    )

    c_col1, c_col2 = st.columns([1.4, 0.6])
    with c_col1:
        fig_cost = plot_cost_vs_threshold(df_cost, optimal_info, retention_cost, churn_loss)
        st.plotly_chart(fig_cost, use_container_width=True)
    with c_col2:
        st.markdown("##### **Threshold Savings Breakdown**")
        st.metric(
            label="Optimal Threshold",
            value=f"{optimal_info['optimal_threshold']:.2f}",
            delta=f"{optimal_info['optimal_threshold'] - 0.50:+.2f} vs default",
        )
        st.metric(
            label="Cost at Optimal Threshold",
            value=f"${optimal_info['min_cost']:,.0f}",
        )
        st.metric(
            label="Cost at Standard 0.50 Threshold",
            value=f"${optimal_info['default_cost']:,.0f}",
        )
        st.metric(
            label="Net Financial Savings from Tuning",
            value=f"${optimal_info['savings_vs_default']:,.0f}",
            delta="Bottom-Line Dollar Savings",
        )

    st.markdown("---")
    # 3. Exploratory Data Visualizations
    st.markdown("#### **3. Churn Distribution by Segments**")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.plotly_chart(plot_churn_by_contract(clean_df), use_container_width=True)
        st.plotly_chart(plot_churn_by_monthly_charges(clean_df), use_container_width=True)
    with vcol2:
        st.plotly_chart(plot_churn_by_tenure(clean_df), use_container_width=True)
        st.plotly_chart(plot_churn_by_internet_service(clean_df), use_container_width=True)

    st.markdown("---")
    # 4. Global SHAP Feature Importance
    st.markdown("#### **4. Global Feature Importance (TreeSHAP)**")
    df_global = get_global_importance(shap_data, top_n=12)
    st.plotly_chart(plot_global_importance(df_global), use_container_width=True)
