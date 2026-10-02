from __future__ import annotations

def generate_response(state: dict) -> str:
    """Return only policy-owned text.

    Gemini is intentionally not used here.  The intake model may interpret the
    question, but this boundary makes the loaded SOP the sole authority for the
    recommendation and makes SOP citation deterministic.
    """
    sop = state.get("selected_sop")
    if not sop:
        return "I don't have a specific weather-safety policy that covers this activity under the current conditions."
    citation = sop.get("cite_as") or f"{sop['id']} - {sop['name']}"
    return f"{citation}\n{sop['advice']}"
