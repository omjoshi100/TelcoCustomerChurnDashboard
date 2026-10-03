"""
LLM Retention Advisor module:
Passes top 3 customer churn drivers to Gemini API (or grounded domain engine)
to generate a 2-sentence summary and 1 targeted retention action.
"""

import os
import re
import json


def generate_llm_advice(
    customer_id: str,
    churn_probability: float,
    threshold: float,
    top_drivers: list,
    customer_attributes: dict = None,
    api_key: str = None,
) -> dict:
    """
    Generates grounded retention advice based on customer's top 3 SHAP drivers.
    Tries Google Gemini API first; falls back to grounded domain rules if API key is not set.
    """
    # Format drivers list into readable text
    driver_texts = []
    for i, d in enumerate(top_drivers, 1):
        desc = d.get("readable_label", d.get("feature", "Unknown"))
        val = d.get("raw_value_str", "")
        driver_texts.append(f"{i}. {desc}{val} (Direction: {d.get('direction', 'Risk Factor')})")
    drivers_str = "\n".join(driver_texts)

    prompt = f"""You are a Customer Retention Strategist for a major telecommunications provider.
A customer has been flagged by our machine learning early-warning model:

- Customer ID: {customer_id}
- Predicted Churn Probability: {churn_probability:.1%} (Intervention Threshold: {threshold:.1%})
- Top algorithmic churn drivers (from SHAP explainability):
{drivers_str}

TASK:
1. Write a clear, professional 2-sentence summary explaining why this specific customer is at risk of churning.
2. Formulate exactly ONE concrete, actionable retention offer or intervention tailored directly to these drivers.

STRICT GROUNDING REQUIREMENT:
The summary and retention action MUST be strictly grounded in the drivers listed above. Do not invent unrelated reasons."""

    # 1. Check for Gemini API Key
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if effective_api_key:
        try:
            # Try official google.genai SDK
            from google import genai
            client = genai.Client(api_key=effective_api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw_text = response.text
            return {
                "source": "Gemini 2.5 Flash",
                "content": raw_text.strip(),
                "prompt_used": prompt,
            }
        except Exception as e:
            # If genai SDK fails or has quota issue, fall through to fallback
            print(f"Gemini API call notice: {e}. Using grounded domain fallback.")

    # 2. Intelligent Grounded Fallback Engine
    return generate_grounded_fallback_advice(customer_id, churn_probability, top_drivers, customer_attributes, prompt)


def generate_grounded_fallback_advice(
    customer_id: str,
    churn_probability: float,
    top_drivers: list,
    customer_attributes: dict = None,
    prompt_used: str = "",
) -> dict:
    """
    Rule-based grounded retention advice synthesizer.
    Ensures that even without an external API key, reviewers get high-quality,
    grounded 2-sentence summaries and actionable retention recommendations.
    """
    driver_labels = [d.get("readable_label", "").lower() for d in top_drivers]
    all_drivers_combined = " ".join(driver_labels)

    # 1. Synthesize 2-sentence summary
    if "month-to-month" in all_drivers_combined or "contract" in all_drivers_combined:
        sentence1 = f"Customer {customer_id} exhibits a high {churn_probability:.1%} churn risk primarily driven by being on a flexible month-to-month contract with zero long-term commitment."
    elif "tenure" in all_drivers_combined:
        sentence1 = f"Customer {customer_id} is at significant risk ({churn_probability:.1%} probability) due to low tenure, indicating an unsettled onboarding phase with our service."
    elif "fiber" in all_drivers_combined or "monthlycharges" in all_drivers_combined:
        sentence1 = f"Customer {customer_id} faces a {churn_probability:.1%} churn likelihood largely fueled by elevated monthly charges relative to their perceived service utility."
    else:
        first_driver = top_drivers[0]["readable_label"] if top_drivers else "usage patterns"
        sentence1 = f"Customer {customer_id} has been flagged with an elevated churn risk of {churn_probability:.1%} driven primarily by {first_driver}."

    # Second sentence linking other drivers
    other_drivers = [d["readable_label"] for d in top_drivers[1:]] if len(top_drivers) > 1 else []
    if other_drivers:
        sentence2 = f"This vulnerability is compounded by {', '.join(other_drivers)}, signaling frustration with price-to-service alignment."
    else:
        sentence2 = "Without a proactive retention touchpoint, the customer is likely to switch to an alternative provider in the next billing cycle."

    summary = f"{sentence1} {sentence2}"

    # 2. Synthesize exactly ONE actionable retention offer
    if "month-to-month" in all_drivers_combined or "contract" in all_drivers_combined:
        action = "Retention Action: Offer a 1-year loyalty contract upgrade featuring a 20% bill discount for the first 3 months plus complimentary speed tier boost."
    elif "fiber" in all_drivers_combined and ("techsupport" in all_drivers_combined or "support" in all_drivers_combined):
        action = "Retention Action: Dispatch a VIP technical support concierge to resolve fiber stability concerns and provide a free 6-month TechSupport add-on."
    elif "monthlycharges" in all_drivers_combined or "charges" in all_drivers_combined:
        action = "Retention Action: Enroll the customer in our 'Value Shield' loyalty plan, reducing their monthly bill by $15 while bundling streaming protection."
    elif "tenure" in all_drivers_combined:
        action = "Retention Action: Trigger an outbound onboarding specialist check-in call with a $20 account courtesy credit to cement service adoption."
    else:
        action = "Retention Action: Provide an immediate $20 promotional service credit paired with a personalized contract renewal incentive."

    output_text = f"**Executive Summary:**\n{summary}\n\n**Recommended Intervention:**\n{action}"

    return {
        "source": "Grounded Expert Engine",
        "content": output_text,
        "prompt_used": prompt_used,
    }
