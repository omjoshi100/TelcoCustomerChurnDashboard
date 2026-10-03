# Step 1: Prerequisites and Environment Setup

This guide walks you through setting up the local runtime environment for the **Telco Customer Churn Early-Warning Dashboard**.

---

## 1. System Requirements

- **Operating System**: Windows 10/11, macOS (Apple Silicon / Intel), or Ubuntu Linux 20.04+
- **Python Version**: Python 3.10 or 3.11 (tested on Python 3.11.9)
- **RAM**: Minimum 4 GB (8 GB recommended for TreeSHAP calculations)
- **Disk Space**: ~1.5 GB for virtual environment, models, and dependencies

---

## 2. Clone or Navigate to the Repository

Open your terminal or PowerShell and navigate to the project directory:

```powershell
cd C:\Users\OM\.gemini\antigravity\scratch\TelcoCustomerChurnDashboard
```

---

## 3. Initialize Python Virtual Environment

Isolate the project packages to avoid system-level conflicts:

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### macOS / Linux (Bash / Zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4. Install Dependencies

Install all core analytical, ML, and dashboard dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Core Libraries Breakdown:
| Library | Purpose |
| :--- | :--- |
| `pandas` | High-performance tabular data manipulation |
| `numpy` | Vectorized numerical operations and arrays |
| `scikit-learn` | Preprocessing pipelines, Logistic Regression baseline, evaluation metrics |
| `xgboost` | Gradient-boosted decision trees with `scale_pos_weight` imbalance handling |
| `shap` | TreeSHAP calculation for local and global model interpretability |
| `streamlit` | Modern, responsive interactive web UI |
| `plotly` | Interactive SVG/WebGL data visualizations |
| `google-genai` | Google Gemini API SDK for generative retention advice |
| `python-pptx` | Automated PowerPoint slide deck generator |

---

## 5. Verify the Environment

Run a quick test to ensure all critical modules are recognized:

```powershell
python -c "import pandas, sklearn, xgboost, shap, streamlit, plotly; print('All ML modules loaded successfully!')"
```

---

## 6. Optional: Configure Google Gemini API Key

To enable real-time Gemini 2.5 Flash retention advice generation:

1. Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Set the environment variable in your terminal session:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key-here"

# Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key-here"
```

*Note: If no API key is set, the dashboard automatically utilizes the built-in Grounded Domain Expert Engine, ensuring zero crashes and full functionality offline.*
