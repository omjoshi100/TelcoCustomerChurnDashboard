"""
Exploratory Data Analysis (EDA) module for Telco Customer Churn.
Calculates churn rate and generates Plotly charts for contract type, tenure, and charges.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def calculate_churn_metrics(df: pd.DataFrame) -> dict:
    """
    Computes overall summary statistics for churn.
    """
    total_customers = len(df)
    churn_count = int(df["Churn_Num"].sum())
    non_churn_count = total_customers - churn_count
    churn_rate = churn_count / total_customers
    
    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "non_churn_count": non_churn_count,
        "churn_rate": churn_rate,
        "avg_monthly_charges": df["MonthlyCharges"].mean(),
        "median_tenure": df["tenure"].median(),
    }


def plot_churn_by_contract(df: pd.DataFrame) -> go.Figure:
    """
    Generates a grouped bar chart of Churn Rate by Contract Type.
    Highlights how Month-to-month contracts dominate churn.
    """
    contract_stats = (
        df.groupby("Contract")["Churn_Num"]
        .agg(Total="count", Churned="sum", ChurnRate="mean")
        .reset_index()
    )
    contract_stats["ChurnRatePct"] = contract_stats["ChurnRate"] * 100

    fig = px.bar(
        contract_stats,
        x="Contract",
        y="ChurnRatePct",
        color="Contract",
        text=contract_stats["ChurnRatePct"].apply(lambda v: f"{v:.1f}%"),
        title="<b>Churn Rate by Contract Type</b> (Month-to-Month High Risk)",
        labels={"ChurnRatePct": "Churn Rate (%)", "Contract": "Contract Type"},
        color_discrete_map={
            "Month-to-month": "#EF5350",
            "One year": "#FFA726",
            "Two year": "#66BB6A",
        },
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(ticksuffix="%", range=[0, 60]),
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif"),
    )
    return fig


def plot_churn_by_tenure(df: pd.DataFrame) -> go.Figure:
    """
    Plots tenure distribution for churned vs retained customers.
    Shows the dramatic drop in churn risk as customer tenure increases.
    """
    df_copy = df.copy()
    df_copy["Status"] = df_copy["Churn_Num"].map({1: "Churned", 0: "Retained"})
    
    # Create tenure buckets
    bins = [0, 12, 24, 48, 72]
    labels = ["0 - 12 Months", "13 - 24 Months", "25 - 48 Months", "49 - 72 Months"]
    df_copy["Tenure_Group"] = pd.cut(df_copy["tenure"], bins=bins, labels=labels, include_lowest=True)
    
    tenure_stats = (
        df_copy.groupby("Tenure_Group", observed=False)["Churn_Num"]
        .agg(Total="count", Churned="sum", ChurnRate="mean")
        .reset_index()
    )
    tenure_stats["ChurnRatePct"] = tenure_stats["ChurnRate"] * 100

    fig = px.bar(
        tenure_stats,
        x="Tenure_Group",
        y="ChurnRatePct",
        text=tenure_stats["ChurnRatePct"].apply(lambda v: f"{v:.1f}%"),
        title="<b>Churn Rate by Customer Tenure Cohort</b>",
        labels={"Tenure_Group": "Tenure Cohort", "ChurnRatePct": "Churn Rate (%)"},
        color="ChurnRatePct",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        yaxis=dict(ticksuffix="%", range=[0, 60]),
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif"),
    )
    return fig


def plot_churn_by_monthly_charges(df: pd.DataFrame) -> go.Figure:
    """
    Generates a box / KDE distribution plot of Monthly Charges by Churn status.
    Reveals that churned customers cluster at the high monthly charge segment ($70 - $100).
    """
    df_copy = df.copy()
    df_copy["Status"] = df_copy["Churn_Num"].map({1: "Churned", 0: "Retained"})

    fig = px.histogram(
        df_copy,
        x="MonthlyCharges",
        color="Status",
        barmode="overlay",
        marginal="box",
        title="<b>Monthly Charges Distribution: Retained vs Churned</b>",
        labels={"MonthlyCharges": "Monthly Charges ($)", "Status": "Customer Outcome"},
        color_discrete_map={"Retained": "#42A5F5", "Churned": "#EF5350"},
        opacity=0.65,
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif"),
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Customer Count",
    )
    return fig


def plot_churn_by_internet_service(df: pd.DataFrame) -> go.Figure:
    """
    Plots churn rate by Internet Service provider type (Fiber optic vs DSL vs No).
    """
    service_stats = (
        df.groupby("InternetService")["Churn_Num"]
        .agg(Total="count", Churned="sum", ChurnRate="mean")
        .reset_index()
    )
    service_stats["ChurnRatePct"] = service_stats["ChurnRate"] * 100

    fig = px.bar(
        service_stats,
        x="InternetService",
        y="ChurnRatePct",
        color="InternetService",
        text=service_stats["ChurnRatePct"].apply(lambda v: f"{v:.1f}%"),
        title="<b>Churn Rate by Internet Service Type</b>",
        labels={"InternetService": "Internet Service", "ChurnRatePct": "Churn Rate (%)"},
        color_discrete_map={
            "Fiber optic": "#EF5350",
            "DSL": "#FFA726",
            "No": "#66BB6A",
        },
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(ticksuffix="%", range=[0, 60]),
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif"),
    )
    return fig
