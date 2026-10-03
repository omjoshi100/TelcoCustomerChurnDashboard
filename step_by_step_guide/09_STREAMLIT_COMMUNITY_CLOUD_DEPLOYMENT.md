# Step 9: Streamlit Community Cloud Deployment Guide

This guide provides step-by-step instructions to deploy the **Telco Customer Churn Early-Warning Dashboard** live on the web using **Streamlit Community Cloud** (free hosting directly connected to GitHub).

---

## 1. Prerequisites for Deployment

1. A free [GitHub](https://github.com/) account (e.g., `omjoshi100`).
2. A free [Streamlit Community Cloud](https://share.streamlit.io/) account (sign in with your GitHub credentials).
3. The project code committed to your GitHub repository:
   `https://github.com/omjoshi100/TelcoCustomerChurnDashboard`

---

## 2. Repository Pre-Deployment Checklist

Before deploying, ensure the root directory contains:
- `app.py`: The main Streamlit entry point.
- `requirements.txt`: The list of all dependencies.
- `src/`: The directory containing analytical and model modules.
- `data/`: Contains `telco_churn.csv` and cached models (`data/models/trained_artifacts.pkl`).

*Note: If `trained_artifacts.pkl` is not committed or is generated at runtime, `app.py` automatically detects its absence and trains the model in ~4 seconds on startup!*

---

## 3. Step-by-Step Streamlit Cloud Deployment

### Step A: Sign In to Streamlit Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"Continue with GitHub"** and authorize Streamlit.

### Step B: Create a New App
1. Click the blue **"Create app"** or **"New app"** button in your dashboard.
2. Select **"I already have an app"** / **"Deploy a public app from GitHub"**.

### Step C: Enter Repository Details
Fill in the deployment configuration:
- **Repository**: `omjoshi100/TelcoCustomerChurnDashboard`
- **Branch**: `main` (or `master`)
- **Main file path**: `app.py`
- **App URL**: Choose a custom subdomain (e.g., `telco-churn-early-warning.streamlit.app`)

### Step D: Configure Secrets (Optional)
If you want the dashboard to use your Google Gemini API key automatically without requiring users to type it into the sidebar:
1. In the app settings on Streamlit Cloud, click **"Advanced settings..."** $\rightarrow$ **"Secrets"**.
2. Add your key in TOML format:
   ```toml
   GEMINI_API_KEY = "your-actual-gemini-api-key-here"
   ```
3. Save changes.

### Step E: Click "Deploy!"
1. Click the **"Deploy!"** button.
2. Streamlit Cloud will spin up a container, run `pip install -r requirements.txt`, and launch `app.py`.
3. In ~2 minutes, your dashboard will be live with a public HTTPS URL accessible to reviewers and interviewers worldwide!

---

## 4. Troubleshooting Common Deployment Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | Python path doesn't include app root | `app.py` includes `sys.path.append(os.path.abspath(os.path.dirname(__file__)))` which resolves this automatically |
| Build Timeout | Heavy packages taking too long | Streamlit caches packages after first build |
| Memory Limit Exceeded | TreeSHAP computing over too large a sample | We cache precomputed SHAP values in `data/models/shap_cache.pkl` |
