from __future__ import annotations
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from .gemini import gemini_model_name, invoke_gemini


def _extract_generated_text(content: object) -> str:
    """Return only generated text, excluding provider metadata."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(text_parts)
    return ""


def generate_response(state: dict) -> str:
    sop = state.get("selected_sop")
    weather = state.get("weather", {})
    if not sop:
        return "There is no applicable policy for this activity and these live conditions."
    facts = f"Recommendation: {sop['advice']}\nCurrent conditions: temperature {weather['temperature_c']} C, wind {weather['wind_speed_kmh']} km/h, precipitation {weather['precipitation_mm']} mm, precipitation probability {weather['precipitation_probability_pct']}%, UV {weather['uv_index']}, condition {weather['weather_condition']}\nPolicy applied: {sop['id']} — {sop['name']}\nSeverity: {sop['severity']}"
    if not os.getenv("GEMINI_API_KEY"):
        return facts
    model = ChatGoogleGenerativeAI(model=gemini_model_name(), temperature=0, thinking_level="low")
    result = invoke_gemini(lambda: model.invoke([SystemMessage(content="Rewrite the supplied facts clearly. Preserve every number, SOP ID, severity, and recommendation exactly. Do not add advice."), HumanMessage(content=facts)]))
    return _extract_generated_text(result.content)
