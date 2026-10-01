from __future__ import annotations
from typing import Any


ALIASES = {
    "bicycling": "cycling", "bike_ride": "cycling", "biking": "cycling",
    "jogging": "running", "outdoor_exercise": "running", "exercise_outdoors": "running",
    "senior": "older_adult", "infant": "child",
}


def _value(context: dict[str, Any], weather: dict[str, Any], field: str) -> Any:
    return context.get(field, weather.get(field))


def _condition(condition: dict[str, Any], context: dict[str, Any], weather: dict[str, Any]) -> bool:
    actual = _value(context, weather, condition.get("field", ""))
    operator, expected = condition.get("operator"), condition.get("value")
    if actual is None: return False
    if operator in ("equals", "=="): return actual == expected
    if operator == ">": return actual > expected
    if operator == ">=": return actual >= expected
    if operator == "<": return actual < expected
    if operator == "<=": return actual <= expected
    if operator in ("in", "membership"): return actual in expected
    if operator == "between": return condition["min"] <= actual <= condition["max"]
    return False


def _matches(sop: dict[str, Any], context: dict[str, Any], weather: dict[str, Any]) -> bool:
    conditions = sop.get("conditions", {})
    for key, values in (("activity_any", [context.get("activity")]), ("intent_any", [context.get("intent")]), ("group_any", [context.get("user_group")])):
        if key in conditions and not any(ALIASES.get(value, value) in [ALIASES.get(item, item) for item in conditions[key]] for value in values if value): return False
    if "all" in conditions and not all(_condition(item, context, weather) for item in conditions["all"]): return False
    if "any" in conditions and not any(_condition(item, context, weather) for item in conditions["any"]): return False
    return True


def match_sops(context: dict[str, Any], weather: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [sop for sop in config["sops"] if _matches(sop, context, weather)]


def select_sop(matches: list[dict[str, Any]], severity_order: dict[str, int]) -> dict[str, Any] | None:
    return max(matches, key=lambda sop: (severity_order.get(sop["severity"], -1), sop["priority"])) if matches else None
