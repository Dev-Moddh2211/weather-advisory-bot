from __future__ import annotations

from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages


class Weather(TypedDict, total=False):
    timestamp: str
    temperature_c: float
    wind_speed_kmh: float
    precipitation_mm: float
    precipitation_probability_pct: float
    uv_index: float
    weather_condition: str


class AdvisoryState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    location: str
    latitude: float
    longitude: float
    activity: str
    intent: str
    user_group: str
    requested_time: str
    is_follow_up: bool
    weather: Weather
    matched_sops: list[dict[str, Any]]
    selected_sop: dict[str, Any]
    severity: str
    answer: str
    error_type: str
    error: str
