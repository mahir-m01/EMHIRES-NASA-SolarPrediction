import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from state import GridAdvisorState

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
    "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
    "UK": "United Kingdom",
}

_SYSTEM_PROMPT = """You are a solar grid integration advisor.
Respond ONLY with valid JSON - no markdown, no code blocks, no extra text.

Output must match this exact structure:
{
  "forecast_summary": "2-3 sentence summary of the generation outlook for the day",
  "risk_periods": [
    {"period": "string", "risk": "string", "severity": "high|medium|low"}
  ],
  "strategies": [
    {"title": "string", "description": "string", "source": "retrieved|general"}
  ],
  "responsible_ai_note": "string"
}

IMPORTANT rules for risk_periods:
- Risk periods are PROBLEMS to manage, not opportunities.
- Real risks: morning ramp-up, evening ramp-down, overnight gap, high variability windows.
- Peak generation hours are an OPPORTUNITY — mention only in forecast_summary and strategies.
- severity "high" = grid stability threat, "medium" = requires active management, "low" = monitor only.

Base strategies on the provided grid management guidelines.
Mark source as "retrieved" if drawn from the guidelines, "general" if from general knowledge.
Acknowledge uncertainty where relevant."""


def recommendation_node(state: GridAdvisorState) -> dict:
    chunks = state["retrieved_chunks"]
    docs_block = "\n".join(f"[DOC {i+1}]: {c}" for i, c in enumerate(chunks))

    hourly_profile = state["hourly_profile"]
    peak_cf = max(hourly_profile) if hourly_profile else state["cf_value"]
    peak_hour = hourly_profile.index(peak_cf) if hourly_profile else 0

    country_name = _COUNTRY_NAMES.get(state["country"], state["country"])

    user_message = (
        f"Country: {country_name}\n"
        f"Forecast Date: {state['forecast_date']}\n"
        f"Peak Capacity Factor: {state['cf_value']:.3f}\n"
        f"Peak Generation Hour: {peak_hour:02d}:00\n"
        f"Installed Capacity: {state['capacity_kw']} kW\n"
        f"Variability Score: {state['risk_summary'].get('variability_score', 'N/A')}\n\n"
        f"Risk Flags:\n" + "\n".join(f"- {f}" for f in state["risk_flags"]) + "\n\n"
        f"Grid Management Guidelines (retrieved):\n{docs_block}"
    )

    try:
        llm = ChatOpenAI(
            model="openrouter/free",
            temperature=0.2,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_message)]
        response = llm.invoke(messages)
        recommendations = json.loads(response.content)
    except Exception as e:
        recommendations = {
            "forecast_summary": "LLM call failed - risk analysis above is still valid.",
            "risk_periods": [],
            "strategies": [],
            "responsible_ai_note": f"Recommendation generation failed: {str(e)}",
        }

    return {"final_recommendations": recommendations}
