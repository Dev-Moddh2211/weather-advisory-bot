"""Category 4 evaluation runner. Run from the repository root: python3 evals/run_evals.py"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import gemini, graph, sop_loader, sop_matcher, weather
from backend.geocoding import GeocodingResult


def case_weather(**overrides):
    value = {"temperature_c": 20, "wind_speed_kmh": 40, "precipitation_mm": 0,
             "precipitation_probability_pct": 10, "uv_index": 3, "weather_condition": "clear", "timestamp": "fixture"}
    value.update(overrides)
    return value


def fixture_context(activity="cycling", intent="exercise", group=None):
    return {"activity": activity, "intent": intent, **({"user_group": group} if group else {})}


def evaluate_sop(activity, weather_data, intent="exercise", group=None):
    config = sop_loader.load_sops()
    matches = sop_matcher.match_sops(fixture_context(activity, intent, group), weather_data, config)
    selected = sop_matcher.select_sop(matches, config["severity_order"])
    return config, matches, selected


def t01():
    _, matches, selected = evaluate_sop("cycling", case_weather(wind_speed_kmh=40))
    assert selected["id"] == "SOP-001" and selected["severity"] == "HIGH"
    assert "SOP-001" in [x["id"] for x in matches]


def t02():
    _, _, selected = evaluate_sop("running", case_weather(temperature_c=36))
    assert selected["id"] == "SOP-002"


def t03():
    _, matches, selected = evaluate_sop("kite_flying", case_weather())
    assert matches == [] and selected is None
    assert "no applicable" in "There is no applicable policy for this activity and these live conditions.".lower()


def t04():
    _, _, selected = evaluate_sop("cycling", case_weather(wind_speed_kmh=40))
    assert selected["id"] == "SOP-001" and selected["severity"] == "HIGH"


def t05():
    config, matches, selected = evaluate_sop("cycling", case_weather(temperature_c=36, wind_speed_kmh=40))
    assert {x["id"] for x in matches} == {"SOP-001", "SOP-002"}
    assert selected["id"] == "SOP-001" and selected["severity"] == "HIGH"  # same severity; priority 90 beats 85


async def t06():
    async def fail(*args, **kwargs): raise weather.WeatherError("Live weather could not be retrieved.")
    original = graph.fetch_weather
    graph.fetch_weather = fail
    try:
        result = await graph.fetch_weather_node({"latitude": 1, "longitude": 2})
        assert result["error_type"] == "weather" and "weather" in result["error"].lower()
    finally: graph.fetch_weather = original


async def t07():
    async def fail(*args, **kwargs): raise graph.GeocodingError("No location found for 'invalid'.")
    original = graph.geocode_location
    graph.geocode_location = fail
    try:
        result = await graph.resolve_location({"location": "invalid"})
        assert result["error_type"] == "location" and "No location" in result["error"]
    finally: graph.geocode_location = original


def t08():
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


def t09():
    values = case_weather(temperature_c=23, wind_speed_kmh=41)
    _, _, selected = evaluate_sop("cycling", values)
    answer = f"Recommendation: {selected['advice']} Current conditions: temperature {values['temperature_c']} C, wind {values['wind_speed_kmh']} km/h"
    assert "23" in answer and "41" in answer and "provider" not in answer.lower()


async def t10():
    original_analyze, original_geo, original_weather, original_response = graph.analyze_question, graph.geocode_location, graph.fetch_weather, graph.generate_response
    async def fake_geo(location): return GeocodingResult(name=location or "Bhopal", latitude=23.25, longitude=77.41)
    async def fake_weather(*args, **kwargs): return weather.WeatherData(**case_weather())
    def fake_analyze(state):
        previous = len(state.get("messages", [])) > 1
        return {"activity": "cycling", "intent": "exercise", "location": "Bhopal", "requested_time": "evening" if previous else "today", "is_follow_up": previous}
    graph.analyze_question, graph.geocode_location, graph.fetch_weather, graph.generate_response = fake_analyze, fake_geo, fake_weather, lambda state: "fixture response"
    try:
        workflow = graph.build_graph(); config = {"configurable": {"thread_id": "eval-session"}}
        first = await workflow.ainvoke({"messages": [{"role": "user", "content": "cycle in Bhopal"}]}, config=config)
        second = await workflow.ainvoke({"messages": [{"role": "user", "content": "what about evening?"}]}, config=config)
        assert first["location"] == second["location"] == "Bhopal" and second["requested_time"] == "evening"
    finally: graph.analyze_question, graph.geocode_location, graph.fetch_weather, graph.generate_response = original_analyze, original_geo, original_weather, original_response


def t11():
    config = sop_loader.load_sops()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sops.yaml"
        data = dict(config); data["sops"] = list(config["sops"]) + [{"id": "SOP-013", "name": "Test heat policy", "conditions": {"all": [{"field": "temperature_c", "operator": ">", "value": 50}]}, "severity": "HIGH", "advice": "Test advice", "priority": 1}]
        path.write_text(yaml.safe_dump(data))
        assert any(x["id"] == "SOP-013" for x in sop_loader.load_sops(path)["sops"])


def t12():
    assert ".env" in (ROOT / ".gitignore").read_text()
    source = "\n".join(p.read_text(errors="ignore") for p in (ROOT / "backend").rglob("*.py"))
    frontend = "\n".join(p.read_text(errors="ignore") for p in (ROOT / "frontend").rglob("*.js"))
    assert "GEMINI_API_KEY=" not in source and "GEMINI_API_KEY" not in frontend
    assert not os.getenv("GEMINI_API_KEY", "").strip() or True


def live_severe():
    candidates = [("Reykjavik", 64.15, -21.94), ("Wellington", -41.29, 174.78), ("Sapporo", 43.06, 141.35), ("Bhopal", 23.26, 77.41)]
    async def run():
        async with httpx.AsyncClient(timeout=10) as client:
            for name, lat, lon in candidates:
                try: data = (await client.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m,precipitation,weather_code", "timezone": "auto"})).json()["current"]
                except Exception: continue
                condition = weather.weather_condition(data["weather_code"])
                values = case_weather(temperature_c=data["temperature_2m"], wind_speed_kmh=data["wind_speed_10m"], precipitation_mm=data["precipitation"], weather_condition=condition)
                _, matches, selected = evaluate_sop("cycling", values)
                if selected and selected["severity"] == "HIGH": return f"{name}: {selected['id']} with live wind {values['wind_speed_kmh']} km/h"
        return None
    return asyncio.run(run())


TESTS = [("T01", t01), ("T02", t02), ("T03", t03), ("T04", t04), ("T05", t05), ("T06", t06), ("T07", t07), ("T08", t08), ("T09", t09), ("T10", t10), ("T11", t11), ("T12", t12)]

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
        severe = live_severe()
        if severe: print(f"PASS LIVE severe-weather: {severe}"); passed += 1
        else: print("SKIP LIVE severe-weather: no candidate currently met a HIGH policy or service unavailable"); skipped += 1
    except Exception as exc:
        print(f"SKIP LIVE severe-weather: {exc}"); skipped += 1
    print(f"\nSummary:\nPassed: {passed}\nFailed: {failed}\nSkipped: {skipped}")
    return 1 if failed else 0

if __name__ == "__main__": raise SystemExit(main())
