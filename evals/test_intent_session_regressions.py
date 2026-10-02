import asyncio
from unittest.mock import patch

from backend import graph, weather
from backend.geocoding import GeocodingError, GeocodingResult


def test_request_boundary_clears_request_facts_but_preserves_conversation_memory():
    result = graph.reset_request_state(
        {
            "conversation_location": "Delhi",
            "conversation_activity": "cycling",
            "weather": {"temperature_c": 40},
            "selected_sop": {"id": "SOP-003"},
            "error": "old error",
        }
    )
    assert result["weather"] is None
    assert result["selected_sop"] is None
    assert result["error"] is None
    assert result["location"] is None
    # The partial update leaves these existing checkpoint keys untouched.
    assert "conversation_location" not in result


def test_missing_location_does_not_call_geocoder_or_continue():
    with patch.object(graph, "geocode_location") as geocode:
        result = asyncio.run(graph.resolve_location({"location": None}))
    geocode.assert_not_called()
    assert result["error_type"] == "location"
    assert result["weather"] is None
    assert result["selected_sop"] is None


def test_location_failure_clears_stale_weather_and_sop():
    async def fail(_location):
        raise GeocodingError("Location could not be resolved.")

    state = {
        "location": "near Delhi",
        "weather": {"temperature_c": 40},
        "raw_weather": {"old": True},
        "selected_sop": {"id": "SOP-003"},
        "situational": True,
    }
    with patch.object(graph, "geocode_location", fail):
        result = asyncio.run(graph.resolve_location(state))
    assert result["error_type"] == "location"
    assert result["weather"] is None
    assert result["raw_weather"] is None
    assert result["selected_sop"] is None
    assert result["situational"] is False


def test_weather_failure_clears_stale_weather_and_sop():
    async def fail(*_args, **_kwargs):
        raise weather.WeatherError("Live weather could not be retrieved.")

    state = {
        "latitude": 28.6,
        "longitude": 77.2,
        "weather": {"temperature_c": 40},
        "raw_weather": {"old": True},
        "selected_sop": {"id": "SOP-003"},
    }
    with patch.object(graph, "fetch_weather", fail):
        result = asyncio.run(graph.fetch_weather_node(state))
    assert result["error_type"] == "weather"
    assert result["weather"] is None
    assert result["raw_weather"] is None
    assert result["selected_sop"] is None


def test_successful_location_updates_conversation_memory():
    async def fake_geo(location):
        assert location == "Delhi"
        return GeocodingResult(name="Delhi", latitude=28.6, longitude=77.2)

    with patch.object(graph, "geocode_location", fake_geo):
        result = asyncio.run(
            graph.resolve_location({"location": "Delhi", "activity": "cycling"})
        )
    assert result["conversation_location"] == "Delhi"
    assert result["conversation_activity"] == "cycling"


def test_follow_up_uses_only_explicitly_marked_conversation_memory(monkeypatch):
    class FakeModel:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _prompt):
            return graph.Intent(is_follow_up=True, requested_time="evening")

    monkeypatch.setattr(graph, "ChatGoogleGenerativeAI", lambda **_kwargs: FakeModel())
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fixture", "GEMINI_MODEL": "fixture"}):
        result = graph.analyze_question(
            {
                "messages": [{"role": "user", "content": "What about this evening?"}],
                "conversation_location": "Delhi",
                "conversation_activity": "cycling",
            }
        )
    assert result["location"] == "Delhi"
    assert result["activity"] == "cycling"
    assert result["requested_time"] == "evening"


def test_intake_normalizes_relational_location_to_place_name(monkeypatch):
    class FakeModel:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _prompt):
            return graph.Intent(
                location="Delhi", activity="hiking", requested_time="afternoon"
            )

    monkeypatch.setattr(graph, "ChatGoogleGenerativeAI", lambda **_kwargs: FakeModel())
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fixture", "GEMINI_MODEL": "fixture"}):
        result = graph.analyze_question(
            {"messages": [{"role": "user", "content": "hiking near Delhi"}]}
        )
    assert result["location"] == "Delhi"
    assert result["location"] != "near Delhi"


def test_intake_keeps_travel_child_and_unsupported_activities_distinct(monkeypatch):
    cases = {
        "take my scooter to work": graph.Intent(
            activity="scooter_travel", intent="travel", location="Delhi"
        ),
        "take my child to the park": graph.Intent(
            activity="park_visit", intent="outdoor_activity", user_group="child", location="Delhi"
        ),
        "fly a kite": graph.Intent(
            activity="kite_flying", intent="outdoor_activity", location="Delhi"
        ),
    }

    class FakeModel:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            message = prompt.split("Latest message:", 1)[1].strip()
            return cases[next(text for text in cases if text in message)]

    monkeypatch.setattr(graph, "ChatGoogleGenerativeAI", lambda **_kwargs: FakeModel())
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fixture", "GEMINI_MODEL": "fixture"}):
        scooter = graph.analyze_question(
            {"messages": [{"role": "user", "content": "take my scooter to work"}]}
        )
        child = graph.analyze_question(
            {"messages": [{"role": "user", "content": "take my child to the park"}]}
        )
        kite = graph.analyze_question(
            {"messages": [{"role": "user", "content": "fly a kite"}]}
        )

    assert scooter["activity"] == "scooter_travel"
    assert scooter["intent"] == "travel"
    assert child["user_group"] == "child"
    assert child["activity"] == "park_visit"
    assert kite["activity"] == "kite_flying"
