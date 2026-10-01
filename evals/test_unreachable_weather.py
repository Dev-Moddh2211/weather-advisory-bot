"""Mocked failure-path check; this never calls HTTP, Gemini, or /chat."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import graph
from backend.weather import WeatherError


async def test_unreachable_weather_is_error_not_no_match() -> None:
    async def unreachable(*args, **kwargs):
        raise WeatherError("Live weather could not be retrieved.")

    with patch("backend.graph.fetch_weather", unreachable):
        result = await graph.fetch_weather_node({"latitude": 28.61, "longitude": 77.21})

    assert result["error_type"] == "weather"
    assert "weather" in result["error"].lower()
    assert "matched_sops" not in result
    assert result.get("selected_sop") is None


if __name__ == "__main__":
    asyncio.run(test_unreachable_weather_is_error_not_no_match())
    print("PASS mocked unreachable-weather: weather error is not NO MATCHING POLICY")
