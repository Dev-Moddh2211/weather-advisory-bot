from __future__ import annotations

from typing import Any


ALIASES = {
    "bicycling": "cycling",
    "bike_ride": "cycling",
    "biking": "cycling",
    "jogging": "running",
    "outdoor_exercise": "running",
    "exercise_outdoors": "running",
    "senior": "older_adult",
    "infant": "child",
}


def _value(context: dict[str, Any], weather: dict[str, Any], field: str) -> Any:
    return context.get(field, weather.get(field))


def _condition(condition: dict[str, Any], context: dict[str, Any], weather: dict[str, Any]) -> bool:
    actual = _value(context, weather, condition.get("field", ""))
    operator, expected = condition.get("operator"), condition.get("value")
    if actual is None:
        return False
    if operator in ("equals", "=="):
        return actual == expected
    if operator == ">":
        return actual > expected
    if operator == ">=":
        return actual >= expected
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    if operator in ("in", "membership"):
        return actual in expected
    if operator == "between":
        return condition["min"] <= actual <= condition["max"]
    return False


def _membership(value: float, definition: dict[str, Any]) -> float:
    kind = definition["kind"]
    points = definition["points"]
    if kind == "ascending":
        left, right = points
        return 0.0 if value <= left else 1.0 if value >= right else (value - left) / (right - left)
    if kind == "descending":
        left, right = points
        return 1.0 if value <= left else 0.0 if value >= right else (right - value) / (right - left)
    if kind == "triangle":
        left, peak, right = points
        if value <= left or value >= right:
            return 0.0
        return (value - left) / (peak - left) if value <= peak else (right - value) / (right - peak)
    left, rising_end, falling_start, right = points
    if value <= left or value >= right:
        return 0.0
    if value < rising_end:
        return (value - left) / (rising_end - left)
    if value <= falling_start:
        return 1.0
    return (right - value) / (right - falling_start)


def _fuzzy_condition(condition: dict[str, Any], context: dict[str, Any], weather: dict[str, Any]) -> bool:
    weighted_score = 0.0
    total_weight = 0.0
    for signal in condition.get("signals", []):
        actual = _value(context, weather, signal.get("field", ""))
        if actual is None:
            return False
        weight = signal["weight"]
        weighted_score += weight * _membership(float(actual), signal["membership"])
        total_weight += weight
    return total_weight > 0 and weighted_score / total_weight >= condition["minimum_score"]


def _matches(sop: dict[str, Any], context: dict[str, Any], weather: dict[str, Any], known_activities: set[str]) -> bool:
    conditions = sop.get("conditions", {})
    for key, values in (("activity_any", [context.get("activity")]), ("intent_any", [context.get("intent")]), ("group_any", [context.get("user_group")])):
        if key in conditions:
            allowed = [ALIASES.get(item, item) for item in conditions[key]]
            matches_outdoor_scope = key == "activity_any" and "outdoor_activity" in allowed and any(
                ALIASES.get(value, value) in known_activities for value in values if value
            )
            if not matches_outdoor_scope and not any(
                ALIASES.get(value, value) in allowed for value in values if value
            ):
                return False
    if "all" in conditions and not all(_condition(item, context, weather) for item in conditions["all"]): return False
    if "any" in conditions and not any(_condition(item, context, weather) for item in conditions["any"]): return False
    if "fuzzy" in conditions and not _fuzzy_condition(conditions["fuzzy"], context, weather): return False
    return True


def match_sops(context: dict[str, Any], weather: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    known_activities = {
        ALIASES.get(activity, activity)
        for sop in config["sops"]
        for activity in sop.get("conditions", {}).get("activity_any", [])
        if activity != "outdoor_activity"
    }
    return [
        sop
        for sop in config["sops"]
        if _matches(sop, context, weather, known_activities)
    ]


def select_sop(matches: list[dict[str, Any]], severity_order: dict[str, int]) -> dict[str, Any] | None:
    return max(matches, key=lambda sop: (severity_order.get(sop["severity"], -1), sop["priority"])) if matches else None
