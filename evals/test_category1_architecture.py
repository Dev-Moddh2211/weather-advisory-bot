from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from backend import graph
from backend.sop_loader import load_sops


def test_invalid_sop_values_fail_loudly():
    config = load_sops()
    for mutate in (
        lambda data: data["sops"][0].update(severity="UNKNOWN"),
        lambda data: data["sops"][0].update(priority="high"),
        lambda data: data["sops"][0]["conditions"]["all"][0].update(operator="unknown"),
    ):
        data = yaml.safe_load(yaml.safe_dump(config))
        mutate(data)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sops.yaml"
            path.write_text(yaml.safe_dump(data))
            with pytest.raises(ValueError, match="Invalid SOP configuration"):
                load_sops(path)


def test_duplicate_sop_ids_fail():
    config = load_sops()
    data = yaml.safe_load(yaml.safe_dump(config))
    data["sops"].append(dict(data["sops"][0]))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sops.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match="duplicate SOP IDs"):
            load_sops(path)


def test_situational_match_selects_broad_policy_and_cites_it():
    config = load_sops()
    state = {"activity": "cycling"}
    weather = {
        "temperature_c": 20, "wind_speed_kmh": 10, "precipitation_mm": 4,
        "precipitation_probability_pct": 90, "uv_index": 2,
        "weather_condition": "thunderstorm", "timestamp": "fixture",
    }
    result = graph.match_node({**state, "weather": weather})
    assert result["situational"] is True
    assert result["situational_sop_id"] == "SOP-011"
    assert "SOP-011" in graph.generate_response({**result, "weather": weather})


def test_invalid_weather_values_are_rejected_before_matching():
    from pydantic import ValidationError
    from backend.weather import WeatherData

    with pytest.raises(ValidationError):
        WeatherData(
            timestamp="fixture",
            temperature_c=20,
            wind_speed_kmh=-1,
            precipitation_mm=0,
            precipitation_probability_pct=10,
            uv_index=3,
            weather_condition="clear",
        )

    with pytest.raises(ValidationError):
        WeatherData(
            timestamp="fixture",
            temperature_c=20,
            wind_speed_kmh=5,
            precipitation_mm=0,
            precipitation_probability_pct=101,
            uv_index=3,
            weather_condition="clear",
        )
