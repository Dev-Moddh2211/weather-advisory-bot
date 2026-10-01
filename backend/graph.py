from __future__ import annotations
import os
from typing import Any
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from .geocoding import GeocodingError, geocode_location
from .gemini import GeminiConfigurationError, GeminiUnavailableError, gemini_model_name, invoke_gemini
from .response_generator import generate_response
from .sop_loader import load_sops
from .sop_matcher import match_sops, select_sop
from .state import AdvisoryState
from .weather import WeatherError, fetch_weather

load_dotenv()


class Intent(BaseModel):
    activity: str | None = None
    intent: str | None = None
    user_group: str | None = None
    location: str | None = None
    requested_time: str = "now"
    is_follow_up: bool = False


def analyze_question(state: AdvisoryState) -> dict[str, Any]:
    message = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1]["content"]
    try:
        model_name = gemini_model_name()
    except GeminiConfigurationError as exc:
        return {"error_type": "configuration", "error": str(exc)}
    if not os.getenv("GEMINI_API_KEY"):
        return {"error_type": "configuration", "error": "GEMINI_API_KEY is not configured."}
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0, thinking_level="low").with_structured_output(Intent)
    previous = state.get("messages", [])[:-1]
    prompt = f"Extract context only, never safety advice. Previous turns: {previous}\nLatest message: {message}"
    try:
        intent = invoke_gemini(lambda: model.invoke(prompt))
    except GeminiUnavailableError as exc:
        return {"error_type": "gemini", "error": str(exc)}
    values = intent.model_dump(exclude_none=True)
    if not values.get("location") and state.get("location"): values["location"] = state["location"]
    if not values.get("activity") and state.get("activity"): values["activity"] = state["activity"]
    return values


async def resolve_location(state: AdvisoryState) -> dict[str, Any]:
    try:
        result = await geocode_location(state.get("location", ""))
        return {"location": result.name, "latitude": result.latitude, "longitude": result.longitude}
    except GeocodingError as exc:
        return {"error_type": "location", "error": str(exc)}


async def fetch_weather_node(state: AdvisoryState) -> dict[str, Any]:
    try:
        weather = await fetch_weather(state["latitude"], state["longitude"], state.get("requested_time", "now"))
        return {"weather": weather.model_dump()}
    except WeatherError as exc:
        return {"error_type": "weather", "error": str(exc)}


def route_error(state: AdvisoryState) -> str: return "error" if state.get("error") else "ok"
def match_node(state: AdvisoryState) -> dict[str, Any]:
    config = load_sops()
    matches = match_sops(state, state["weather"], config)
    return {"matched_sops": matches, "selected_sop": select_sop(matches, config["severity_order"])}
def route_sop(state: AdvisoryState) -> str: return "found" if state.get("selected_sop") else "none"
def response_node(state: AdvisoryState) -> dict[str, str]:
    try:
        return {"answer": generate_response(state)}
    except GeminiConfigurationError as exc:
        return {"error_type": "configuration", "error": str(exc), "answer": str(exc)}
    except GeminiUnavailableError as exc:
        return {"error_type": "gemini", "error": str(exc), "answer": str(exc)}
def error_node(state: AdvisoryState) -> dict[str, str]: return {"answer": state.get("error", "The request could not be completed.")}
def no_guidance_node(state: AdvisoryState) -> dict[str, str]: return {"answer": generate_response(state)}


def build_graph():
    graph = StateGraph(AdvisoryState)
    for name, fn in (("analyze_question", analyze_question), ("resolve_location", resolve_location), ("fetch_weather", fetch_weather_node), ("error", error_node), ("match_sops", match_node), ("no_guidance", no_guidance_node), ("generate_response", response_node)):
        graph.add_node(name, fn)
    graph.add_edge(START, "analyze_question")
    graph.add_conditional_edges("analyze_question", route_error, {"error": "error", "ok": "resolve_location"})
    graph.add_conditional_edges("resolve_location", route_error, {"error": "error", "ok": "fetch_weather"})
    graph.add_conditional_edges("fetch_weather", route_error, {"error": "error", "ok": "match_sops"})
    graph.add_conditional_edges("match_sops", route_sop, {"found": "generate_response", "none": "no_guidance"})
    graph.add_edge("no_guidance", END)
    graph.add_edge("generate_response", END)
    graph.add_edge("error", END)
    return graph.compile(checkpointer=InMemorySaver())
