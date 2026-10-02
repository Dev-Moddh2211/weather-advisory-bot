from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .geocoding import GeocodingError, geocode_location
from .gemini import (
    GeminiConfigurationError,
    GeminiUnavailableError,
    gemini_model_name,
    invoke_gemini,
)
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
    activity_confidence: float | None = Field(default=None, ge=0, le=1)
    activity_is_ambiguous: bool = False


MISSING_LOCATION_ERROR = (
    "I need a location to check the weather for this request."
)


def analyze_question(state: AdvisoryState) -> dict[str, Any]:
    latest_message = state["messages"][-1]
    message = (
        latest_message.content
        if hasattr(latest_message, "content")
        else latest_message["content"]
    )
    try:
        model_name = gemini_model_name()
    except GeminiConfigurationError as exc:
        return {"error_type": "configuration", "error": str(exc)}
    if not os.getenv("GEMINI_API_KEY"):
        return {"error_type": "configuration", "error": "GEMINI_API_KEY is not configured."}
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        thinking_level="low",
    ).with_structured_output(Intent)
    previous = state.get("messages", [])[:-1]
    supported_activities = sorted(
        {
            activity
            for sop in load_sops()["sops"]
            for activity in sop.get("conditions", {}).get("activity_any", [])
        }
    )
    prompt = f"""
Extract request context only; never provide safety advice and never evaluate a policy.

    Return activity as one canonical activity from this repository-supported vocabulary when
    the user's meaning clearly matches one of them: {supported_activities}. If the request
    names an unsupported activity such as fishing, preserve that distinction by returning
    null rather than mapping it to a supported activity.
Normalize paraphrases to the underlying activity rather than copying surface wording.
For example, an outdoor meal or spending time outside with food and friends can mean
the canonical activity 'picnic' when that is clearly the request. Do not classify every
outdoor activity as a picnic: cycling, running, walking, fishing, and an unspecified
request to go outside remain distinct or null. If the activity is genuinely ambiguous,
return null. Do not invent activities, SOP IDs, rules, weather values, or recommendations.

    Include activity_confidence from 0 to 1 and set activity_is_ambiguous=true when the
    activity is unclear. Do not claim high confidence for a vague request such as merely
    spending time outside.

    Previous turns: {previous}
Latest message: {message}
"""
    try:
        intent = invoke_gemini(lambda: model.invoke(prompt))
    except GeminiUnavailableError as exc:
        return {"error_type": "gemini", "error": str(exc)}
    values = intent.model_dump(exclude_none=True)
    supported = set(supported_activities)
    activity = values.get("activity")
    if values.get("activity_is_ambiguous") or (activity and activity not in supported):
        values["activity"] = None
    if (
        values.get("activity_confidence") is not None
        and values["activity_confidence"] < 0.6
    ):
        values["activity"] = None
    if not values.get("location") and state.get("location"):
        values["location"] = state["location"]
    if not values.get("activity") and state.get("activity"):
        values["activity"] = state["activity"]
    return values


async def resolve_location(state: AdvisoryState) -> dict[str, Any]:
    location = (state.get("location") or "").strip()
    if not location:
        return {"error_type": "location", "error": MISSING_LOCATION_ERROR}
    try:
        result = await geocode_location(location)
        return {
            "location": result.name,
            "latitude": result.latitude,
            "longitude": result.longitude,
        }
    except GeocodingError as exc:
        return {"error_type": "location", "error": str(exc)}


async def fetch_weather_node(state: AdvisoryState) -> dict[str, Any]:
    try:
        weather = await fetch_weather(
            state["latitude"],
            state["longitude"],
            state.get("requested_time", "now"),
        )
        return {
            "weather": weather.model_dump(exclude={"raw_payload"}),
            "raw_weather": weather.raw_payload,
        }
    except WeatherError as exc:
        return {"error_type": "weather", "error": str(exc)}


def route_error(state: AdvisoryState) -> str:
    return "error" if state.get("error") else "ok"


def match_node(state: AdvisoryState) -> dict[str, Any]:
    config = load_sops()
    matches = match_sops(state, state["weather"], config)
    situational_matches = [
        sop
        for sop in matches
        if sop.get("category", "").lower() == "situational"
        or "outdoor_activity"
        in sop.get("conditions", {}).get("activity_any", [])
    ]
    selected = select_sop(situational_matches or matches, config["severity_order"])
    return {
        "matched_sops": matches,
        "selected_sop": selected,
        "situational": bool(situational_matches),
        "situational_sop_id": (
            selected["id"] if situational_matches and selected else None
        ),
    }


def route_match(state: AdvisoryState) -> str:
    if state.get("error"):
        return "error"
    if state.get("situational"):
        return "override"
    return "found" if state.get("selected_sop") else "none"


def situational_override_node(state: AdvisoryState) -> dict[str, Any]:
    # Situational policies take precedence over activity-specific policies.
    return {"override": True}


def response_node(state: AdvisoryState) -> dict[str, str]:
    try:
        return {"answer": generate_response(state)}
    except GeminiConfigurationError as exc:
        return {
            "error_type": "configuration",
            "error": str(exc),
            "answer": str(exc),
        }
    except GeminiUnavailableError as exc:
        return {
            "error_type": "gemini",
            "error": str(exc),
            "answer": str(exc),
        }
def error_node(state: AdvisoryState) -> dict[str, str]:
    return {"answer": state.get("error", "The request could not be completed.")}


def no_guidance_node(state: AdvisoryState) -> dict[str, str]:
    return {"answer": generate_response(state)}


def build_graph():
    graph = StateGraph(AdvisoryState)
    nodes = (
        ("analyze_question", analyze_question),
        ("resolve_location", resolve_location),
        ("fetch_weather", fetch_weather_node),
        ("error", error_node),
        ("match_sops", match_node),
        ("override", situational_override_node),
        ("no_guidance", no_guidance_node),
        ("generate_response", response_node),
    )
    for name, fn in nodes:
        graph.add_node(name, fn)
    graph.add_edge(START, "analyze_question")
    graph.add_conditional_edges("analyze_question", route_error, {"error": "error", "ok": "resolve_location"})
    graph.add_conditional_edges("resolve_location", route_error, {"error": "error", "ok": "fetch_weather"})
    graph.add_conditional_edges("fetch_weather", route_error, {"error": "error", "ok": "match_sops"})
    graph.add_conditional_edges(
        "match_sops",
        route_match,
        {
            "error": "error",
            "override": "override",
            "found": "generate_response",
            "none": "no_guidance",
        },
    )
    graph.add_edge("no_guidance", END)
    graph.add_edge("generate_response", END)
    graph.add_edge("override", "generate_response")
    graph.add_edge("error", END)
    return graph.compile(checkpointer=InMemorySaver())
