from __future__ import annotations
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from .gemini import gemini_model_name, invoke_gemini


def _extract_generated_text(content: object) -> str:
    """Return only generated text, excluding provider metadata."""
    if isinstance(content, str):
        return _clean_text(content)
    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return _clean_text("".join(text_parts))
    return ""


def _clean_text(value: str) -> str:
    """Keep provider output as readable plain text for the chat response."""
    value = re.sub(r"^#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", value)
    value = value.replace("`", "")
    return value.strip()


def generate_response(state: dict) -> str:
    sop = state.get("selected_sop")
    weather = state.get("weather", {})
    if not sop:
        return "I don't have a specific weather-safety policy that covers this activity under the current conditions."
    facts = f"Recommendation: {sop['advice']}\nWeather facts: temperature {weather['temperature_c']} C, wind {weather['wind_speed_kmh']} km/h, precipitation {weather['precipitation_mm']} mm, precipitation probability {weather['precipitation_probability_pct']}%, UV {weather['uv_index']}, condition {weather['weather_condition']}"
    if not os.getenv("GEMINI_API_KEY"):
        return sop["advice"]
    model = ChatGoogleGenerativeAI(model=gemini_model_name(), temperature=0, thinking_level="low")
    result = invoke_gemini(lambda: model.invoke([SystemMessage(content="Write one or two plain-text sentences explaining only the supplied recommendation. Do not include headings, policy IDs, severity labels, weather tables, markdown, or new advice."), HumanMessage(content=facts)]))
    return _extract_generated_text(result.content)
