from __future__ import annotations

from datetime import datetime, timedelta
import httpx
from pydantic import BaseModel


class WeatherData(BaseModel):
    timestamp: str
    temperature_c: float
    wind_speed_kmh: float
    precipitation_mm: float
    precipitation_probability_pct: float
    uv_index: float
    weather_condition: str


class WeatherError(RuntimeError):
    pass


def weather_condition(code: int) -> str:
    if code == 0: return "clear"
    if code in (1, 2, 3): return "cloudy"
    if code in (45, 48): return "fog"
    if code in (51, 53, 55, 56, 57): return "drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82): return "rain"
    if code in (71, 73, 75, 77, 85, 86): return "snow"
    if code in (95, 96, 99): return "thunderstorm"
    return "unknown"


async def fetch_weather(latitude: float, longitude: float, requested_time: str = "now", client: httpx.AsyncClient | None = None) -> WeatherData:
    future = requested_time not in {"", "now", "current", "today"}
    params = {"latitude": latitude, "longitude": longitude, "timezone": "auto", "forecast_days": 2}
    if future:
        params["hourly"] = "temperature_2m,wind_speed_10m,precipitation,rain,precipitation_probability,uv_index,weather_code"
    else:
        params["current"] = "temperature_2m,wind_speed_10m,precipitation,weather_code"
        params["hourly"] = "precipitation_probability,uv_index"
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=15)
    try:
        response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        payload = response.json()
        if future:
            hourly = payload["hourly"]
            target_hour = 18 if "evening" in requested_time.lower() or "night" in requested_time.lower() else 9 if "morning" in requested_time.lower() else 13 if "afternoon" in requested_time.lower() else None
            if "tomorrow" in requested_time.lower():
                target_date = (datetime.now() + timedelta(days=1)).date().isoformat()
                candidates = [i for i, value in enumerate(hourly["time"]) if value.startswith(target_date)]
            else:
                candidates = list(range(len(hourly["time"])))
            if target_hour is not None:
                index = min(candidates, key=lambda i: abs(int(hourly["time"][i][11:13]) - target_hour))
            else:
                index = candidates[0]
            values = {key: hourly[key][index] for key in ("time", "temperature_2m", "wind_speed_10m", "precipitation", "precipitation_probability", "uv_index", "weather_code")}
        else:
            current = payload["current"]
            hourly = payload.get("hourly", {})
            values = {"time": current["time"], "temperature_2m": current["temperature_2m"], "wind_speed_10m": current["wind_speed_10m"], "precipitation": current["precipitation"], "precipitation_probability": hourly.get("precipitation_probability", [0])[0], "uv_index": hourly.get("uv_index", [0])[0], "weather_code": current["weather_code"]}
        return WeatherData(timestamp=values["time"], temperature_c=values["temperature_2m"], wind_speed_kmh=values["wind_speed_10m"], precipitation_mm=values["precipitation"], precipitation_probability_pct=values["precipitation_probability"], uv_index=values["uv_index"], weather_condition=weather_condition(values["weather_code"]))
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise WeatherError("Live weather could not be retrieved.") from exc
    finally:
        if owns_client:
            await client.aclose()
