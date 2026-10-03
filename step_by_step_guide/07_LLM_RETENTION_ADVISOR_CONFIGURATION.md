# Step 7: LLM Retention Advisor Configuration

This guide explains how the Generative AI Retention Advisor is designed, prompted, grounded, and configured using the Google Gemini API with seamless offline fallback.

---

## 1. The Business Challenge

Predicting churn probability tells you **who** is likely to leave, and SHAP feature attribution tells you **why**. 
However, customer service representatives and frontline retention teams don't have time to parse log-odds math or complex SHAP graphs during a 90-second phone call.

They need:
1. An instant **2-sentence executive summary** in plain English.
2. Exactly **one concrete, targeted retention offer** that addresses the specific pain points driving that customer away.

---

## 2. Prompt Engineering & Strict Grounding

Hallucinations in customer retention can be expensive (e.g., an LLM offering free lifetime service or inventing fictitious router complaints).

To prevent hallucinations, the prompt strictly grounds the LLM in the top 3 algorithmic SHAP drivers:

```text
You are a Customer Retention Strategist for a major telecommunications provider.
A customer has been flagged by our machine learning early-warning model:

- Customer ID: {customer_id}
- Predicted Churn Probability: {churn_probability:.1%} (Intervention Threshold: {threshold:.1%})
- Top algorithmic churn drivers (from SHAP explainability):
{drivers_str}

TASK:
1. Write a clear, professional 2-sentence summary explaining why this specific customer is at risk of churning.
2. Formulate exactly ONE concrete, actionable retention offer or intervention tailored directly to these drivers.

STRICT GROUNDING REQUIREMENT:
The summary and retention action MUST be strictly grounded in the drivers listed above. Do not invent unrelated reasons.
```

---

## 3. Google Gemini 2.5 Flash Integration

The advisor uses the latest official `google-genai` SDK:

```python
from google import genai

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)
advice_text = response.text
```

### Advantages of Gemini 2.5 Flash:
- **Low Latency**: Sub-second generation ensures call center responsiveness.
- **Cost Efficiency**: Millicents per query, making it economical to run across thousands of customer accounts.
- **Instruction Following**: Strictly adheres to the 2-sentence summary + 1 retention offer constraint.

---

## 4. Intelligent Grounded Fallback Engine

If no Gemini API key is provided (or if the user is running the dashboard completely offline), `src/llm_advisor.py` automatically routes requests to a deterministic **Grounded Domain Engine**.

### Decision Matrix Logic:
| Top Driver Pattern | Synthesized Retention Action |
| :--- | :--- |
| **Month-to-month Contract** | 1-Year Loyalty upgrade with 20% discount for first 3 months + free speed boost |
| **Fiber Optic + No Tech Support** | Dispatch VIP tech support concierge + 6 months free TechSupport add-on |
| **High Monthly Charges** | Enroll in 'Value Shield' loyalty plan, reducing bill by $15 with streaming bundle |
| **Low Tenure (< 6 mo)** | Outbound onboarding check-in call with $20 account courtesy credit |

This guarantees that **the application never crashes** and reviewers always see relevant, high-quality retention strategies.
