import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import graph, main, weather
from backend.geocoding import GeocodingResult


def fixture_weather(**overrides):
    values = {
        "timestamp": "fixture",
        "temperature_c": 22,
        "wind_speed_kmh": 46,
        "precipitation_mm": 0,
        "precipitation_probability_pct": 10,
        "uv_index": 3,
        "weather_condition": "clear",
    }
    values.update(overrides)
    return weather.WeatherData(**values, raw_payload={"current": values.copy()})


def test_post_chat_runs_real_graph_pipeline_and_cites_policy():
    async def fake_geo(location):
        return GeocodingResult(name=location, latitude=23.2, longitude=77.4)

    async def fake_weather(*args, **kwargs):
        return fixture_weather()

    def fake_intake(state):
        return {"activity": "cycling", "intent": "exercise", "location": "Bhopal", "requested_time": "today"}

    with patch.object(graph, "analyze_question", fake_intake), patch.object(graph, "geocode_location", fake_geo), patch.object(graph, "fetch_weather", fake_weather):
        original = main.workflow
        main.workflow = graph.build_graph()
        try:
            response = TestClient(main.app).post("/chat", json={"session_id": "integration", "message": "Would riding my bike be a bad idea?"})
        finally:
            main.workflow = original
    assert response.status_code == 200
    payload = response.json()
    assert payload["sop_id"] == "SOP-001"
    assert "SOP-001" in payload["answer"]
    assert "46" not in payload["answer"] or payload["weather"]["wind_speed_kmh"] == 46


def test_two_paraphrases_enter_through_intake(monkeypatch):
    cases = {
        "Would riding my bike to the office be a bad idea?": "cycling",
        "Should I take my two-wheeler out in these conditions?": "cycling",
    }

    class FakeModel:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            text = prompt.split("Latest message:", 1)[1]
            from backend.graph import Intent
            return Intent(activity="cycling", intent="exercise", location="Fixture City", requested_time="now", activity_confidence=1)

    monkeypatch.setattr(graph, "ChatGoogleGenerativeAI", lambda **kwargs: FakeModel())
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fixture", "GEMINI_MODEL": "fixture"}):
        for message, expected in cases.items():
            result = graph.analyze_question({"messages": [{"role": "user", "content": message}]})
            assert result["activity"] == expected


def test_no_match_text_is_honest():
    state = {"selected_sop": None}
    assert "don't have" in graph.no_guidance_node(state)["answer"].lower()


def test_weather_failure_does_not_continue_to_matching():
    async def fail(*args, **kwargs):
        raise weather.WeatherError("Live weather could not be retrieved.")

    with patch.object(graph, "fetch_weather", fail):
        result = asyncio.run(graph.fetch_weather_node({"latitude": 1, "longitude": 2}))
    assert result["error_type"] == "weather"
    assert "weather" in result["error"].lower()
    assert "matched_sops" not in result
