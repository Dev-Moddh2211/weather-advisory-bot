from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, Field


class WeatherData(BaseModel):
    timestamp: str
    temperature_c: float
    wind_speed_kmh: float
    precipitation_mm: float
    precipitation_probability_pct: float
    uv_index: float
    weather_condition: str
    raw_payload: dict = Field(default_factory=dict, exclude=True)


class WeatherError(RuntimeError):
    pass


def _nearest_hourly_index(times: list[str], target: str) -> int:
    try:
        target_time = datetime.fromisoformat(target)
        return min(
            range(len(times)),
            key=lambda index: abs(datetime.fromisoformat(times[index]) - target_time),
        )
    except (ValueError, TypeError):
        return 0


def _target_hour(requested: str) -> int:
    if "morning" in requested:
        return 9
    if any(word in requested for word in ("evening", "tonight", "night", "later")):
        return 18
    return 13


def _location_now(timezone_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(timezone_name))
    except (KeyError, ValueError):
        return datetime.now(ZoneInfo("UTC"))


def weather_condition(code: int) -> str:
    if code == 0:
        return "clear"
    if code in (1, 2, 3):
        return "cloudy"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55, 56, 57):
        return "drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "thunderstorm"
    return "unknown"


async def fetch_weather(
    latitude: float,
    longitude: float,
    requested_time: str = "now",
    client: httpx.AsyncClient | None = None,
) -> WeatherData:
    requested = (requested_time or "now").strip().lower()
    current_request = requested in {"", "now", "right now", "current"}
    params = {"latitude": latitude, "longitude": longitude, "timezone": "auto", "forecast_days": 2}
    params["hourly"] = (
        "temperature_2m,wind_speed_10m,precipitation,rain,"
        "precipitation_probability,uv_index,weather_code"
    )
    if current_request:
        params["current"] = (
            "temperature_2m,wind_speed_10m,precipitation,weather_code"
        )
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=15)
    try:
        response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        payload = response.json()
        if not current_request:
            hourly = payload["hourly"]
            timezone_name = payload.get("timezone", "UTC")
            local_now = _location_now(timezone_name)
            target_date = (local_now + timedelta(days=1)).date() if "tomorrow" in requested else local_now.date()
            candidates = [i for i, value in enumerate(hourly["time"]) if value[:10] == target_date.isoformat()]
            if not candidates:
                raise WeatherError("The requested forecast time is unavailable.")
            target_hour = _target_hour(requested)
            index = min(candidates, key=lambda i: abs(int(hourly["time"][i][11:13]) - target_hour))
            values = {
                key: hourly[key][index]
                for key in (
                    "time",
                    "temperature_2m",
                    "wind_speed_10m",
                    "precipitation",
                    "precipitation_probability",
                    "uv_index",
                    "weather_code",
                )
            }
        else:
            current = payload["current"]
            hourly = payload.get("hourly", {})
            hourly_times = hourly.get("time", [])
            hourly_index = _nearest_hourly_index(hourly_times, current["time"]) if hourly_times else 0
            values = {
                "time": current["time"],
                "temperature_2m": current["temperature_2m"],
                "wind_speed_10m": current["wind_speed_10m"],
                "precipitation": current["precipitation"],
                "precipitation_probability": hourly.get(
                    "precipitation_probability", [0]
                )[hourly_index],
                "uv_index": hourly.get("uv_index", [0])[hourly_index],
                "weather_code": current["weather_code"],
            }
        return WeatherData(
            timestamp=values["time"],
            temperature_c=values["temperature_2m"],
            wind_speed_kmh=values["wind_speed_10m"],
            precipitation_mm=values["precipitation"],
            precipitation_probability_pct=values["precipitation_probability"],
            uv_index=values["uv_index"],
            weather_condition=weather_condition(values["weather_code"]),
            raw_payload=payload,
        )
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise WeatherError("Live weather could not be retrieved.") from exc
    finally:
        if owns_client:
            await client.aclose()
