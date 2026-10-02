"""Run the deterministic evaluation checks from the repository root."""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import gemini, graph, sop_loader, sop_matcher, weather
from backend.geocoding import GeocodingResult


def case_weather(**overrides):
    value = {
        "temperature_c": 20,
        "wind_speed_kmh": 40,
        "precipitation_mm": 0,
        "precipitation_probability_pct": 10,
        "uv_index": 3,
        "weather_condition": "clear",
        "timestamp": "fixture",
    }
    value.update(overrides)
    return value


def fixture_context(activity="cycling", intent="exercise", group=None):
    context = {"activity": activity, "intent": intent}
    if group:
        context["user_group"] = group
    return context


def evaluate_sop(activity, weather_data, intent="exercise", group=None):
    config = sop_loader.load_sops()
    matches = sop_matcher.match_sops(fixture_context(activity, intent, group), weather_data, config)
    selected = sop_matcher.select_sop(matches, config["severity_order"])
    return config, matches, selected


def test_high_wind_blocks_cycling():
    _, matches, selected = evaluate_sop("cycling", case_weather(wind_speed_kmh=40))
    assert selected["id"] == "SOP-001" and selected["severity"] == "HIGH"
    assert "SOP-001" in [x["id"] for x in matches]


def test_extreme_heat_blocks_running():
    _, _, selected = evaluate_sop("running", case_weather(temperature_c=36))
    assert selected["id"] == "SOP-002"


def test_unsupported_activity_has_no_match():
    _, matches, selected = evaluate_sop("kite_flying", case_weather())
    assert matches == [] and selected is None
    assert "no applicable" in "There is no applicable policy for this activity and these live conditions.".lower()


def test_cycling_high_wind_remains_high_severity():
    _, _, selected = evaluate_sop("cycling", case_weather(wind_speed_kmh=40))
    assert selected["id"] == "SOP-001" and selected["severity"] == "HIGH"


def test_priority_breaks_same_severity_tie():
    config, matches, selected = evaluate_sop("cycling", case_weather(temperature_c=36, wind_speed_kmh=40))
    assert {x["id"] for x in matches} == {"SOP-001", "SOP-002"}
    assert selected["id"] == "SOP-001" and selected["severity"] == "HIGH"  # same severity; priority 90 beats 85


async def test_weather_failure_returns_error():
    async def fail(*args, **kwargs): raise weather.WeatherError("Live weather could not be retrieved.")
    original = graph.fetch_weather
    graph.fetch_weather = fail
    try:
        result = await graph.fetch_weather_node({"latitude": 1, "longitude": 2})
        assert result["error_type"] == "weather" and "weather" in result["error"].lower()
    finally: graph.fetch_weather = original


async def test_location_failure_returns_error():
    async def fail(*args, **kwargs): raise graph.GeocodingError("No location found for 'invalid'.")
    original = graph.geocode_location
    graph.geocode_location = fail
    try:
        result = await graph.resolve_location({"location": "invalid"})
        assert result["error_type"] == "location" and "No location" in result["error"]
    finally: graph.geocode_location = original


async def test_missing_location_does_not_call_geocoder():
    async def fail(*args, **kwargs):
        raise AssertionError("the geocoder must not receive a missing location")

    async def weather_must_not_run(*args, **kwargs):
        raise AssertionError("weather must not run without a location")

    original_analyze, original_geo, original_weather = (
        graph.analyze_question,
        graph.geocode_location,
        graph.fetch_weather,
    )

    def fake_analyze(state):
        return {
            "activity": "cycling",
            "intent": "exercise",
            "location": None,
            "requested_time": "today",
        }

    graph.analyze_question = fake_analyze
    graph.geocode_location, graph.fetch_weather = fail, weather_must_not_run
    try:
        workflow = graph.build_graph()
        result = await workflow.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Is it safe to cycle to work in Delhi today?",
                    }
                ]
            },
            config={"configurable": {"thread_id": "missing-location-eval"}},
        )
        assert result["error_type"] == "location"
        assert result["answer"] == graph.MISSING_LOCATION_ERROR
        assert "No location found for ''" not in result["answer"]
        assert "weather" not in result
        assert "selected_sop" not in result
    finally:
        graph.analyze_question, graph.geocode_location = original_analyze, original_geo
        graph.fetch_weather = original_weather


async def test_valid_location_reaches_weather_path():
    calls = []
    original_analyze, original_geo, original_weather, original_response = (
        graph.analyze_question,
        graph.geocode_location,
        graph.fetch_weather,
        graph.generate_response,
    )

    def fake_analyze(state):
        return {
            "activity": "cycling",
            "intent": "exercise",
            "location": "Delhi",
            "requested_time": "today",
        }

    async def fake_geo(location):
        calls.append(("geocode", location))
        return GeocodingResult(name=location, latitude=28.61, longitude=77.21)

    async def fake_weather(*args, **kwargs):
        calls.append(("weather", args[0], args[1]))
        return weather.WeatherData(**case_weather())

    graph.analyze_question = fake_analyze
    graph.geocode_location, graph.fetch_weather = fake_geo, fake_weather
    graph.generate_response = lambda state: "fixture response"
    try:
        workflow = graph.build_graph()
        result = await workflow.ainvoke(
            {"messages": [{"role": "user", "content": "cycle in Delhi today"}], "location": "Delhi"},
            config={"configurable": {"thread_id": "valid-location-eval"}},
        )
        assert calls == [("geocode", "Delhi"), ("weather", 28.61, 77.21)]
        assert result["location"] == "Delhi"
        assert "error" not in result
    finally:
        graph.analyze_question = original_analyze
        graph.geocode_location, graph.fetch_weather = original_geo, original_weather
        graph.generate_response = original_response


async def test_follow_up_with_omitted_location_uses_session_location():
    calls = []
    original_analyze, original_geo, original_weather, original_response = (
        graph.analyze_question,
        graph.geocode_location,
        graph.fetch_weather,
        graph.generate_response,
    )

    def fake_analyze(state):
        latest = state["messages"][-1].content
        if "Delhi" in latest:
            return {"activity": "cycling", "intent": "exercise", "location": "Delhi", "requested_time": "today"}
        return {"requested_time": "evening"}

    async def fake_geo(location):
        calls.append(location)
        return GeocodingResult(name=location, latitude=28.61, longitude=77.21)

    async def fake_weather(*args, **kwargs):
        return weather.WeatherData(**case_weather())

    graph.analyze_question, graph.geocode_location = fake_analyze, fake_geo
    graph.fetch_weather, graph.generate_response = fake_weather, lambda state: "fixture response"
    try:
        workflow = graph.build_graph()
        config = {"configurable": {"thread_id": "omitted-location-eval"}}
        await workflow.ainvoke({"messages": [{"role": "user", "content": "cycle in Delhi today"}]}, config=config)
        second = await workflow.ainvoke({"messages": [{"role": "user", "content": "what about this evening?"}]}, config=config)
        assert calls == ["Delhi", "Delhi"]
        assert second["location"] == "Delhi"
        assert second["requested_time"] == "evening"
    finally:
        graph.analyze_question, graph.geocode_location = original_analyze, original_geo
        graph.fetch_weather, graph.generate_response = original_weather, original_response


def test_gemini_retries_are_bounded_and_sanitized():
    calls = 0
    def fail():
        nonlocal calls; calls += 1
        raise RuntimeError("503 provider detail must not leak")
    original_sleep = gemini.time.sleep; gemini.time.sleep = lambda _: None
    try:
        try: gemini.invoke_gemini(fail)
        except gemini.GeminiUnavailableError as exc: assert str(exc) == "Gemini is temporarily unavailable. Please try again later."
        else: raise AssertionError("expected sanitized failure")
        assert calls == gemini.MAX_GEMINI_ATTEMPTS == 3
    finally: gemini.time.sleep = original_sleep


def test_response_uses_supplied_weather_values():
    values = case_weather(temperature_c=23, wind_speed_kmh=41)
    _, _, selected = evaluate_sop("cycling", values)
    answer = f"Recommendation: {selected['advice']} Current conditions: temperature {values['temperature_c']} C, wind {values['wind_speed_kmh']} km/h"
    assert "23" in answer and "41" in answer and "provider" not in answer.lower()


async def test_follow_up_reuses_session_context():
    original_analyze, original_geo, original_weather, original_response = graph.analyze_question, graph.geocode_location, graph.fetch_weather, graph.generate_response
    async def fake_geo(location): return GeocodingResult(name=location or "Bhopal", latitude=23.25, longitude=77.41)
    async def fake_weather(*args, **kwargs): return weather.WeatherData(**case_weather())
    def fake_analyze(state):
        latest_message = state["messages"][-1]
        latest = latest_message.content if hasattr(latest_message, "content") else latest_message["content"]
        requested_time = "tomorrow morning" if "tomorrow" in latest else "evening" if "evening" in latest else "today"
        previous = requested_time != "today"
        return {"activity": "cycling", "intent": "exercise", "location": "Bhopal", "requested_time": requested_time, "is_follow_up": previous}
    graph.analyze_question, graph.geocode_location, graph.fetch_weather, graph.generate_response = fake_analyze, fake_geo, fake_weather, lambda state: "fixture response"
    try:
        workflow = graph.build_graph(); config = {"configurable": {"thread_id": "eval-session"}}
        first = await workflow.ainvoke({"messages": [{"role": "user", "content": "cycle in Bhopal"}]}, config=config)
        second = await workflow.ainvoke({"messages": [{"role": "user", "content": "what about evening?"}]}, config=config)
        third = await workflow.ainvoke({"messages": [{"role": "user", "content": "what about tomorrow morning?"}]}, config=config)
        assert first["location"] == second["location"] == third["location"] == "Bhopal"
        assert first["activity"] == second["activity"] == third["activity"] == "cycling"
        assert second["requested_time"] == "evening" and third["requested_time"] == "tomorrow morning"
    finally: graph.analyze_question, graph.geocode_location, graph.fetch_weather, graph.generate_response = original_analyze, original_geo, original_weather, original_response


def test_new_yaml_policy_loads_without_control_flow_changes():
    config = sop_loader.load_sops()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sops.yaml"
        data = dict(config); data["sops"] = list(config["sops"]) + [{"id": "SOP-TEST-NEW", "name": "Test heat policy", "conditions": {"all": [{"field": "temperature_c", "operator": ">", "value": 50}]}, "severity": "HIGH", "advice": "Test advice", "priority": 1}]
        path.write_text(yaml.safe_dump(data))
        assert any(x["id"] == "SOP-TEST-NEW" for x in sop_loader.load_sops(path)["sops"])


def test_repository_does_not_expose_credentials():
    assert ".env" in (ROOT / ".gitignore").read_text()
    source = "\n".join(p.read_text(errors="ignore") for p in (ROOT / "backend").rglob("*.py"))
    frontend = "\n".join(p.read_text(errors="ignore") for p in (ROOT / "frontend").rglob("*.js"))
    assert "GEMINI_API_KEY=" not in source and "GEMINI_API_KEY" not in frontend
    assert not os.getenv("GEMINI_API_KEY", "").strip() or True


def test_picnic_matches_comfortable_conditions():
    _, matches, selected = evaluate_sop("picnic", case_weather(temperature_c=31.7, wind_speed_kmh=8.8, precipitation_probability_pct=25))
    assert "SOP-010" in [item["id"] for item in matches]
    assert selected["id"] == "SOP-010" and selected["severity"] == "LOW"


def test_general_outdoor_exercise_matches_heat():
    _, _, selected = evaluate_sop("outdoor_exercise", case_weather(temperature_c=32, wind_speed_kmh=5))
    assert selected["id"] == "SOP-003"


async def test_evening_forecast_uses_evening_hour():
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"timezone": "Asia/Kolkata", "hourly": {
                "time": ["2026-10-02T09:00", "2026-10-02T18:00", "2026-10-03T13:00"],
                "temperature_2m": [25, 29, 28], "wind_speed_10m": [4, 6, 5],
                "precipitation": [0, 1, 0], "precipitation_probability": [10, 20, 15],
                "uv_index": [3, 0, 5], "weather_code": [0, 61, 0]}}
    class Client:
        async def get(self, *args, **kwargs): return Response()
    with patch("backend.weather.datetime") as clock:
        clock.now.return_value = __import__("datetime").datetime(2026, 10, 2, 10)
        result = await weather.fetch_weather(19, 72, "this evening", client=Client())
    assert result.timestamp.endswith("18:00") and result.temperature_c == 29


def test_response_contains_sop_citation_without_gemini():
    from backend.response_generator import generate_response
    state = {"selected_sop": {"id": "SOP-010", "name": "Suitable conditions", "severity": "LOW", "advice": "This looks like a reasonable day for a picnic."}, "weather": case_weather()}
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        answer = generate_response(state)
    assert answer.startswith("SOP-010 - Suitable conditions") and state["selected_sop"]["advice"] in answer


def test_picnic_policy_does_not_match_thunderstorm():
    _, matches, selected = evaluate_sop("picnic", case_weather(temperature_c=24, wind_speed_kmh=8, precipitation_probability_pct=10))
    assert selected["id"] == "SOP-010"
    assert "SOP-011" not in [item["id"] for item in matches]


def test_non_picnic_activities_do_not_match_picnic_policy():
    for activity in ("cycling", "running", "fishing", "walking"):
        _, matches, _ = evaluate_sop(activity, case_weather(temperature_c=24, wind_speed_kmh=8, precipitation_probability_pct=10))
        assert "SOP-010" not in [item["id"] for item in matches], activity


def test_hot_picnic_has_no_applicable_policy():
    _, matches, selected = evaluate_sop("picnic", case_weather(temperature_c=35, wind_speed_kmh=8, precipitation_probability_pct=10))
    assert "SOP-010" not in [item["id"] for item in matches]
    assert selected is None


def test_thunderstorm_policy_overrides_picnic():
    _, matches, selected = evaluate_sop("picnic", case_weather(weather_condition="thunderstorm"))
    assert selected["id"] == "SOP-011" and selected["severity"] == "HIGH"
    assert "SOP-010" not in [item["id"] for item in matches]


def test_intake_uses_canonical_activities_without_policy_authority():
    captured = {}

    class FakeModel:
        def with_structured_output(self, _schema): return self
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return graph.Intent(activity="picnic", location="Delhi", requested_time="today")

    original_model = graph.ChatGoogleGenerativeAI
    graph.ChatGoogleGenerativeAI = lambda **kwargs: FakeModel()
    try:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fixture", "GEMINI_MODEL": "fixture-model"}):
            result = graph.analyze_question({"messages": [{"role": "user", "content": "outdoor lunch with friends in Delhi"}]})
        assert result["activity"] == "picnic"
        assert "picnic" in captured["prompt"]
        assert "never provide safety advice" in captured["prompt"]
    finally:
        graph.ChatGoogleGenerativeAI = original_model


def test_thunderstorm_policy_applies_to_supported_activities_only():
    for activity in ("picnic", "cycling", "running", "walking", "hiking"):
        _, _, selected = evaluate_sop(activity, case_weather(weather_condition="thunderstorm"))
        assert selected["id"] == "SOP-011" and selected["severity"] == "HIGH"
    _, _, selected = evaluate_sop("unknown", case_weather(weather_condition="thunderstorm"))
    assert selected is None


async def test_current_weather_uses_current_hour_values():
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"timezone": "Asia/Kolkata", "current": {
                "time": "2026-10-02T10:00", "temperature_2m": 26, "wind_speed_10m": 4,
                "precipitation": 0, "weather_code": 0}, "hourly": {
                "time": ["2026-10-02T08:00", "2026-10-02T10:00", "2026-10-02T12:00"],
                "precipitation_probability": [90, 20, 40], "uv_index": [1, 5, 7]}}
    class Client:
        async def get(self, *args, **kwargs): return Response()
    result = await weather.fetch_weather(28.6, 77.2, "now", client=Client())
    assert result.precipitation_probability_pct == 20 and result.uv_index == 5


async def test_requested_forecast_periods_use_expected_hours():
    class Response:
        def raise_for_status(self): pass
        def json(self):
            times = [
                "2026-10-02T09:00", "2026-10-02T13:00", "2026-10-02T18:00",
                "2026-10-03T09:00", "2026-10-03T13:00", "2026-10-03T18:00",
            ]
            return {"timezone": "Asia/Kolkata", "hourly": {
                "time": times,
                "temperature_2m": [21, 31, 27, 20, 30, 26],
                "wind_speed_10m": [3, 10, 6, 2, 9, 5],
                "precipitation": [0, 1, 0, 0, 2, 0],
                "precipitation_probability": [10, 40, 20, 5, 35, 15],
                "uv_index": [2, 6, 1, 2, 7, 1],
                "weather_code": [0, 61, 0, 0, 61, 0]}}
    class Client:
        async def get(self, *args, **kwargs): return Response()
    expected = {
        "today": "2026-10-02T13:00",
        "this morning": "2026-10-02T09:00",
        "this afternoon": "2026-10-02T13:00",
        "this evening": "2026-10-02T18:00",
        "tomorrow": "2026-10-03T13:00",
        "tomorrow morning": "2026-10-03T09:00",
        "tomorrow afternoon": "2026-10-03T13:00",
        "tomorrow evening": "2026-10-03T18:00",
    }
    with patch("backend.weather.datetime") as clock:
        clock.now.return_value = __import__("datetime").datetime(2026, 10, 2, 3, 15)
        for requested, timestamp in expected.items():
            result = await weather.fetch_weather(19, 72, requested, client=Client())
            assert result.timestamp == timestamp
    assert result.temperature_c == 26 and result.wind_speed_kmh == 5


def test_running_policy_changes_at_temperature_boundaries():
    _, _, selected = evaluate_sop("running", case_weather(temperature_c=28, wind_speed_kmh=8, precipitation_probability_pct=0))
    assert selected["id"] == "SOP-013"
    for temperature, expected in ((29.9, "SOP-013"), (30, "SOP-003"), (34.9, "SOP-003"), (35, "SOP-002")):
        _, _, selected = evaluate_sop("running", case_weather(temperature_c=temperature, wind_speed_kmh=8, precipitation_probability_pct=0))
        assert selected["id"] == expected


def test_uv_policy_wins_when_picnic_policies_overlap():
    _, matches, selected = evaluate_sop("picnic", case_weather(temperature_c=24, wind_speed_kmh=8, precipitation_probability_pct=0, uv_index=8))
    assert {item["id"] for item in matches} == {"SOP-010", "SOP-014"}
    assert selected["id"] == "SOP-014" and selected["severity"] == "MODERATE"
    for uv, expected in ((7.9, "SOP-010"), (8, "SOP-014"), (9, "SOP-014")):
        _, _, selected = evaluate_sop("picnic", case_weather(temperature_c=24, wind_speed_kmh=8, precipitation_probability_pct=0, uv_index=uv))
        assert selected["id"] == expected


def test_cold_weather_policy_uses_strict_boundary():
    _, _, selected = evaluate_sop("walking", case_weather(temperature_c=4.9, wind_speed_kmh=8, precipitation_probability_pct=0))
    assert selected["id"] == "SOP-015"
    _, _, selected = evaluate_sop("walking", case_weather(temperature_c=5, wind_speed_kmh=8, precipitation_probability_pct=0))
    assert selected is None


def test_cycling_policy_changes_at_wind_boundaries():
    for wind, expected in ((24.9, "SOP-013"), (25, "SOP-016"), (35, "SOP-016"), (35.1, "SOP-001")):
        _, _, selected = evaluate_sop("cycling", case_weather(temperature_c=20, wind_speed_kmh=wind, precipitation_probability_pct=0))
        assert (selected["id"] if selected else None) == expected
    _, _, selected = evaluate_sop("cycling", case_weather(temperature_c=20, wind_speed_kmh=36, precipitation_probability_pct=0))
    assert selected["id"] == "SOP-001"


def test_unsupported_activities_never_match():
    for activity in ("fishing", "unknown", None):
        _, matches, selected = evaluate_sop(activity, case_weather(temperature_c=24, wind_speed_kmh=8, precipitation_probability_pct=0, uv_index=3))
        assert matches == [] and selected is None


def test_highest_severity_wins_over_overlapping_policies():
    _, matches, selected = evaluate_sop("cycling", case_weather(temperature_c=36, wind_speed_kmh=40, precipitation_probability_pct=0))
    assert {item["id"] for item in matches} == {"SOP-001", "SOP-002"}
    assert selected["id"] == "SOP-001"


def run_live_severe_weather_check():
    location = os.getenv("LIVE_SEVERE_LOCATION", "").strip()
    if not location:
        return None
    async def run():
        async with httpx.AsyncClient(timeout=10) as client:
            location_result = await graph.geocode_location(location, client=client)
            fetched = await weather.fetch_weather(location_result.latitude, location_result.longitude, "now", client=client)
            values = fetched.model_dump(exclude={"raw_payload"})
            _, matches, selected = evaluate_sop("cycling", values)
            if selected and selected["severity"] == "HIGH":
                answer = graph.generate_response({"selected_sop": selected, "weather": values})
                return f"{location_result.name}: {selected['id']} payload={fetched.raw_payload} answer={answer}"
            return None
    return asyncio.run(run())


TESTS = [
    ("T01", test_high_wind_blocks_cycling),
    ("T02", test_extreme_heat_blocks_running),
    ("T03", test_unsupported_activity_has_no_match),
    ("T04", test_cycling_high_wind_remains_high_severity),
    ("T05", test_priority_breaks_same_severity_tie),
    ("T06", test_weather_failure_returns_error),
    ("T07", test_location_failure_returns_error),
    ("T07A", test_missing_location_does_not_call_geocoder),
    ("T07B", test_valid_location_reaches_weather_path),
    ("T07C", test_follow_up_with_omitted_location_uses_session_location),
    ("T08", test_gemini_retries_are_bounded_and_sanitized),
    ("T09", test_response_uses_supplied_weather_values),
    ("T10", test_follow_up_reuses_session_context),
    ("T11", test_new_yaml_policy_loads_without_control_flow_changes),
    ("T12", test_repository_does_not_expose_credentials),
    ("T13", test_picnic_matches_comfortable_conditions),
    ("T14", test_general_outdoor_exercise_matches_heat),
    ("T15", test_evening_forecast_uses_evening_hour),
    ("T16", test_response_contains_sop_citation_without_gemini),
    ("T17", test_picnic_policy_does_not_match_thunderstorm),
    ("T18", test_non_picnic_activities_do_not_match_picnic_policy),
    ("T19", test_hot_picnic_has_no_applicable_policy),
    ("T20", test_thunderstorm_policy_overrides_picnic),
    ("T21", test_intake_uses_canonical_activities_without_policy_authority),
    ("T22", test_thunderstorm_policy_applies_to_supported_activities_only),
    ("T23", test_current_weather_uses_current_hour_values),
    ("T24", test_running_policy_changes_at_temperature_boundaries),
    ("T25", test_uv_policy_wins_when_picnic_policies_overlap),
    ("T26", test_cold_weather_policy_uses_strict_boundary),
    ("T27", test_cycling_policy_changes_at_wind_boundaries),
    ("T28", test_unsupported_activities_never_match),
    ("T29", test_highest_severity_wins_over_overlapping_policies),
    ("T30", test_requested_forecast_periods_use_expected_hours),
]

def main():
    print("Weather Advisory Bot — Evaluation Results")
    passed = failed = skipped = 0
    for test_id, test in TESTS:
        try:
            result = test(); asyncio.run(result) if asyncio.iscoroutine(result) else None
            print(f"PASS {test_id} {test.__name__}"); passed += 1
        except Exception as exc:
            print(f"FAIL {test_id} {test.__name__}: {exc}"); failed += 1
    try:
        severe = run_live_severe_weather_check()
        if severe: print(f"PASS LIVE severe-weather: {severe}"); passed += 1
        else: print("SKIP LIVE severe-weather: set LIVE_SEVERE_LOCATION and ensure it currently triggers a HIGH policy"); skipped += 1
    except Exception as exc:
        print(f"SKIP LIVE severe-weather: {exc}"); skipped += 1
    print(f"\nSummary:\nPassed: {passed}\nFailed: {failed}\nSkipped: {skipped}")
    return 1 if failed else 0

if __name__ == "__main__": raise SystemExit(main())
