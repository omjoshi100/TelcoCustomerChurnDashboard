"""
SHAP Explainability module:
- Computes TreeSHAP values for XGBoost
- Calculates Global Feature Importance
- Extracts Top 3 Churn Drivers per individual customer
- Generates interactive Plotly waterfall and summary charts
"""

import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap

SHAP_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "models", "shap_cache.pkl")


def compute_or_load_shap(xgb_model, X_test_df: pd.DataFrame):
    """
    Computes SHAP values using TreeExplainer, or loads cached values from disk.
    """
    if os.path.exists(SHAP_CACHE_FILE):
        try:
            with open(SHAP_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    print("Computing TreeSHAP values on test set...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_df)
    expected_value = explainer.expected_value

    shap_data = {
        "explainer": explainer,
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_names": list(X_test_df.columns),
    }

    os.makedirs(os.path.dirname(SHAP_CACHE_FILE), exist_ok=True)
    with open(SHAP_CACHE_FILE, "wb") as f:
        pickle.dump(shap_data, f)
    print(f"Cached SHAP values to {SHAP_CACHE_FILE}")

    return shap_data


def get_global_importance(shap_data: dict, top_n: int = 15) -> pd.DataFrame:
    """
    Calculates mean absolute SHAP value for each feature.
    """
    shap_values = shap_data["shap_values"]
    feature_names = shap_data["feature_names"]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    return df_importance.head(top_n)


def plot_global_importance(df_importance: pd.DataFrame) -> go.Figure:
    """
    Plots horizontal bar chart of top global SHAP features.
    """
    df_sorted = df_importance.sort_values(by="Mean_Abs_SHAP", ascending=True)

    fig = px.bar(
        df_sorted,
        x="Mean_Abs_SHAP",
        y="Feature",
        orientation="h",
        title="<b>Global Feature Importance (TreeSHAP)</b>",
        labels={"Mean_Abs_SHAP": "Mean |SHAP Value| (Impact on Model Output)", "Feature": ""},
        color="Mean_Abs_SHAP",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif"),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def clean_feature_label(raw_feature: str, feature_value=None) -> str:
    """
    Transforms encoded column names into clean, readable labels.
    Example: 'cat__Contract_One year' -> 'Contract: One Year'
    """
    cleaned = raw_feature.replace("cat__", "").replace("num__", "")
    parts = cleaned.split("_", 1)
    if len(parts) == 2:
        category, val = parts[0], parts[1]
        return f"{category}: {val}"
    return cleaned


def get_customer_top_drivers(
    sample_idx: int,
    shap_data: dict,
    X_test_df: pd.DataFrame,
    X_test_raw: pd.DataFrame = None,
    top_n: int = 3,
) -> list:
    """
    Extracts the top N positive contributors (risk drivers) and negative contributors (retention anchors)
    for a specific customer.
    """
    shap_values = shap_data["shap_values"][sample_idx]
    feature_names = shap_data["feature_names"]

    # Combine into DataFrame
    driver_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values,
        "Abs_SHAP": np.abs(shap_values),
    })

    # Sort strictly by positive SHAP (drivers increasing churn probability)
    positive_drivers = driver_df[driver_df["SHAP_Value"] > 0].sort_values(by="SHAP_Value", ascending=False)
    
    # If not enough positive drivers, take highest absolute
    if len(positive_drivers) < top_n:
        top_records = driver_df.sort_values(by="Abs_SHAP", ascending=False).head(top_n)
    else:
        top_records = positive_drivers.head(top_n)

    results = []
    for _, row in top_records.iterrows():
        feat = row["Feature"]
        shap_val = row["SHAP_Value"]
        readable_label = clean_feature_label(feat)
        
        # Look up raw value if provided
        raw_val_str = ""
        if X_test_raw is not None and sample_idx < len(X_test_raw):
            raw_row = X_test_raw.iloc[sample_idx]
            base_col = feat.replace("cat__", "").replace("num__", "").split("_")[0]
            if base_col in raw_row:
                raw_val_str = f" (Value: {raw_row[base_col]})"

        direction = "Increases Risk" if shap_val > 0 else "Lowers Risk"
        results.append({
            "feature": feat,
            "readable_label": readable_label,
            "raw_value_str": raw_val_str,
            "shap_value": float(shap_val),
            "direction": direction,
            "description": f"{readable_label}{raw_val_str} [{direction}, SHAP: {shap_val:+.3f}]",
        })

    return results


def plot_customer_shap_waterfall(
    sample_idx: int,
    shap_data: dict,
    top_k: int = 8,
) -> go.Figure:
    """
    Plots a customer-level SHAP impact bar chart showing features pushing toward Churn (+) or Retention (-).
    """
    shap_values = shap_data["shap_values"][sample_idx]
    feature_names = shap_data["feature_names"]

    df_local = pd.DataFrame({
        "Feature": [clean_feature_label(f) for f in feature_names],
        "SHAP_Value": shap_values,
        "Abs_SHAP": np.abs(shap_values),
    }).sort_values(by="Abs_SHAP", ascending=False).head(top_k)

    df_local = df_local.sort_values(by="SHAP_Value", ascending=True)
    df_local["Color"] = np.where(df_local["SHAP_Value"] > 0, "#EF5350", "#42A5F5")

    fig = go.Figure(
        go.Bar(
            x=df_local["SHAP_Value"],
            y=df_local["Feature"],
            orientation="h",
            marker=dict(color=df_local["Color"]),
            text=df_local["SHAP_Value"].apply(lambda v: f"{v:+.3f}"),
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Individual Customer Churn Risk Drivers (Local SHAP Breakdown)</b><br>"
              "<span style='font-size:11px;color:gray'>Red = Increases Churn Probability | Blue = Anchors Retention</span>",
        xaxis_title="SHAP Value (Contribution to Log-Odds)",
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif"),
        height=380,
    )
    return fig
