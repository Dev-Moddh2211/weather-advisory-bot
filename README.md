# Weather Advisory

This project is a small weather-based advisory bot. A user asks whether an outdoor activity is suitable for current or requested conditions. The application gets weather from Open-Meteo, matches the request against project-defined SOPs, and returns a recommendation with the policy used.

## Problem

Outdoor questions depend on location, time, and changing weather. Gemini understands the conversational question, but it does not invent safety rules. Advice comes from written, reviewable policies and includes the selected SOP and severity when one applies.

## Architecture

```text
User → Intent / Context Extraction → Location Resolution → Live Weather
     → SOP Matching → SOP Selection → Response Generation → User
```

Gemini extracts activity, intent, group, location, and requested time, then turns approved results into natural language. Open-Meteo supplies geocoding and forecast data. The SOP layer decides whether a policy matches, its severity, and the prescribed advice. Gemini is not the safety authority; unsupported activities return no applicable guidance.

## LangGraph Flow

```text
START → Analyze Question → Resolve Location → Fetch Weather
                              ↓ failure       ↓ failure
                             Error            Error

Fetch Weather → Match SOPs → SOP found? ── no ──→ No Guidance → END
                                      └─ yes ─→ Generate Response → END
```

The failure branches cover unavailable location/weather services and allow unsupported activities to return an honest no-guidance response.

## SOPs

Policies live in [`sops/sops.yaml`](sops/sops.yaml). The current configuration has 12 policies across Outdoor Exercise, Travel, Vulnerable Groups, and General Outdoor Activities. Policies use stable IDs and LOW, MODERATE, or HIGH severity. Thresholds are project-defined prototype values, not official safety standards. The picnic policy combines temperature, rain probability, and wind.

All matching policies are retained. The highest severity wins; numeric priority breaks ties. Selection is deterministic and does not ask Gemini to choose a policy.

This is an explicit severity-first resolution strategy: matching policies are collected
before selection, then the greatest configured severity is selected and `priority` breaks
ties. A severe-weather policy can therefore override a lower-severity activity policy
when both match.

## Adding a Policy

Add another entry to `sops/sops.yaml` using the existing fields:

```yaml
- id: SOP-013
  name: Example policy
  conditions:
    activity_any: [walking]
    all:
      - field: wind_speed_kmh
        operator: ">"
        value: 30
  severity: MODERATE
  advice: Use a sheltered route and reconsider the activity.
  priority: 50
```

Adding a policy only requires another configuration entry; the graph control flow does not need to change.

## Weather Data

The application resolves a city through Open-Meteo geocoding, converts it to latitude/longitude, and requests a forecast. Weather is normalized into temperature, wind, precipitation, precipitation probability, UV, timestamp, and a readable condition. Current questions use current conditions plus hourly rain probability/UV; future requests select a relevant hourly forecast.

## Session Memory

LangGraph keeps conversation state by session ID. After `Is it safe to cycle in Bhopal today?`, a follow-up such as `What about this evening?` can reuse Bhopal and cycling while changing the requested time. This memory is session-scoped in process memory, not permanent user storage.

## Setup

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GEMINI_API_KEY in .env locally if using Gemini extraction/wording.
uvicorn backend.main:app --reload
```

The API is `POST http://127.0.0.1:8000/chat` with `{"session_id":"a-session-id","message":"Can I go running in Bhopal?"}`.

### Frontend

In a second terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

The frontend uses `VITE_API_BASE_URL` and keeps one UUID in `sessionStorage`. It does not receive server credentials.

## Evaluation

The evaluator covers deterministic SOP matching, canonical paraphrase normalization,
false-positive protection, adversarial requests, overlapping policies, a deterministic
HIGH-severity thunderstorm path, failure paths, bounded Gemini retries, response
grounding, three-turn session memory, temporary SOP extensibility, and repository
security checks.

```bash
.venv/bin/python evals/run_evals.py
```

It also probes candidate locations against live Open-Meteo data for a current HIGH-severity condition. That case is skipped when no candidate meets the policy or the service is unavailable; it does not fabricate weather.

The live severe-weather probe is informational: `PASS` means a real candidate currently
matched a HIGH policy, `SKIP` means no candidate matched or the service was unavailable,
and neither result is converted into the other. The deterministic mocked thunderstorm
case separately proves that the HIGH-severity branch works without depending on current
weather.

### Mocked versus live checks

`evals/run_evals.py` is the automated evaluator. Its normal cases use fixtures and monkeypatches; for example, the session-memory case replaces intent extraction, geocoding, weather, and response generation. It does not call Gemini or `POST /chat`. The separate `evals/test_unreachable_weather.py` check deliberately mocks an unreachable weather node and verifies that the graph returns a weather error before SOP matching, rather than treating the failure as no matching policy.

For local live verification, start the backend with the real `.env`, then run:

```bash
python3 scripts/live_integration.py
```

This script calls the running `POST /chat` endpoint for a Delhi picnic and a Bhopal cycling request. It does not mock Gemini, Open-Meteo, or the backend. It prints the configured model name only (never the API key), the HTTP status, live weather, selected SOP, and final advisory. The Bhopal case expects no applicable SOP under the live conditions; if current weather matches a cycling policy, the check fails rather than fabricating or overriding the result.

## Scope

There is no authentication, database, account system, or deployment configuration. The backend and frontend are intended for local running and review.
