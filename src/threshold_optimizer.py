"""
Cost-Benefit Threshold Optimization module.
Calculates the optimal classification threshold based on asymmetric business costs:
- Retention Offer Cost: $20
- Customer Churn Loss: $500
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compute_cost_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    cost_offer: float = 20.0,
    cost_loss: float = 500.0,
    retention_success_rate: float = 0.50,
) -> pd.DataFrame:
    """
    Evaluates business cost across classification thresholds from 0.01 to 0.99.
    
    Business Logic:
    - False Negative (Missed Churner): Lost customer = $500
    - False Positive (Wasted Outreach): Retention offer spent on customer who would have stayed = $20
    - True Positive (Intervened Churner): $20 offer + (1 - success_rate) * $500 residual loss
    - True Negative (Correctly Left Alone): $0
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    records = []
    
    total_churners = int(np.sum(y_true))
    cost_no_action = total_churners * cost_loss

    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        
        cost_fn = fn * cost_loss
        cost_fp = fp * cost_offer
        cost_tp = tp * (cost_offer + (1.0 - retention_success_rate) * cost_loss)
        
        total_cost = cost_fn + cost_fp + cost_tp
        savings = cost_no_action - total_cost
        
        records.append({
            "threshold": thresh,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "cost_tp": cost_tp,
            "total_cost": total_cost,
            "savings": savings,
        })
        
    df_cost = pd.DataFrame(records)
    return df_cost


def find_optimal_threshold(
    df_cost: pd.DataFrame,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    cost_offer: float = 20.0,
    cost_loss: float = 500.0,
) -> dict:
    """
    Extracts the cost-minimizing threshold and compares it against default 0.50.
    """
    min_idx = df_cost["total_cost"].idxmin()
    best_row = df_cost.loc[min_idx]
    
    # Evaluate at default threshold 0.50
    default_row = df_cost.iloc[(df_cost["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

    cost_saved_over_default = float(default_row["total_cost"] - best_row["total_cost"])
    
    return {
        "optimal_threshold": float(best_row["threshold"]),
        "min_cost": float(best_row["total_cost"]),
        "default_cost": float(default_row["total_cost"]),
        "savings_vs_default": cost_saved_over_default,
        "default_threshold": 0.50,
        "tp_optimal": int(best_row["tp"]),
        "fp_optimal": int(best_row["fp"]),
        "fn_optimal": int(best_row["fn"]),
        "tp_default": int(default_row["tp"]),
        "fp_default": int(default_row["fp"]),
        "fn_default": int(default_row["fn"]),
    }


def plot_cost_vs_threshold(
    df_cost: pd.DataFrame,
    optimal_info: dict,
    cost_offer: float = 20.0,
    cost_loss: float = 500.0,
) -> go.Figure:
    """
    Plots the U-shaped business cost curve highlighting the optimal threshold sweet spot.
    """
    fig = go.Figure()

    # Total Cost line
    fig.add_trace(
        go.Scatter(
            x=df_cost["threshold"],
            y=df_cost["total_cost"],
            mode="lines",
            name="Total Expected Cost ($)",
            line=dict(color="#1976D2", width=3),
        )
    )

    # Cost of Missed Churners (FN)
    fig.add_trace(
        go.Scatter(
            x=df_cost["threshold"],
            y=df_cost["cost_fn"],
            mode="lines",
            name="Cost of Missed Churners ($500/FN)",
            line=dict(color="#D32F2F", width=2, dash="dash"),
        )
    )

    # Cost of Outreach (FP + TP offers)
    outreach_cost = (df_cost["tp"] + df_cost["fp"]) * cost_offer
    fig.add_trace(
        go.Scatter(
            x=df_cost["threshold"],
            y=outreach_cost,
            mode="lines",
            name="Retention Campaign Cost ($20/outreach)",
            line=dict(color="#388E3C", width=2, dash="dot"),
        )
    )

    # Vertical line for optimal threshold
    opt_t = optimal_info["optimal_threshold"]
    min_c = optimal_info["min_cost"]
    fig.add_vline(
        x=opt_t,
        line_width=2.5,
        line_dash="dash",
        line_color="#7B1FA2",
        annotation_text=f"<b>Optimal Cutoff ({opt_t:.2f})</b><br>${min_c:,.0f} Min Cost",
        annotation_position="top left",
    )

    # Vertical line for standard 0.50 threshold
    def_c = optimal_info["default_cost"]
    fig.add_vline(
        x=0.50,
        line_width=2,
        line_dash="dot",
        line_color="#757575",
        annotation_text=f"Default (0.50): ${def_c:,.0f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title="<b>Business Cost Optimization Curve (Retention $20 vs Lost Customer $500)</b>",
        xaxis_title="Classification Probability Threshold",
        yaxis_title="Total Financial Loss ($)",
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif"),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.35),
    )
    return fig
