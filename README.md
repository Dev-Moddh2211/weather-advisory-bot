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

Policies live in [`sops/sops.yaml`](sops/sops.yaml). The current configuration has 16 policies across Outdoor Exercise, Travel, Vulnerable Groups, and General Outdoor Activities. Policies are validated with Pydantic when loaded, use stable IDs and LOW, MODERATE, or HIGH severity, and keep thresholds in YAML. Thresholds are project-defined prototype values, not official safety standards. The picnic policy combines temperature, rain probability, and wind.

All matching policies are retained. The highest severity wins; numeric priority breaks ties. Selection is deterministic and does not ask Gemini to choose a policy.

This is an explicit severity-first resolution strategy: matching policies are collected
before selection, then the greatest configured severity is selected and `priority` breaks
ties. A severe-weather policy can therefore override a lower-severity activity policy
when both match.

The matcher is deterministic because safety decisions must be reproducible. Broad
`outdoor_activity` policies are marked as situational overrides and use a dedicated
graph branch. The raw Open-Meteo JSON is retained alongside typed weather values for
traceability; Gemini only composes language from the selected validated SOP and fetched
facts.

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

The application resolves a city through Open-Meteo geocoding, converts it to latitude/longitude, and requests a forecast. Weather is normalized into temperature, wind, precipitation, precipitation probability, UV, timestamp, and a readable condition. Explicit present-tense requests (`now`, `right now`, and `current`) use the current Open-Meteo snapshot, with rain probability and UV taken from the nearest current-hourly record. `today` uses a representative 13:00 local hourly forecast rather than treating the current instant as the whole day. `this morning`, `this afternoon`, and `this evening` use 09:00, 13:00, and 18:00 local time; tomorrow variants use the next local date at the same representative hour. If the exact hour is unavailable, the nearest available record on the requested local date is selected. All fields used for a forecast decision come from that same hourly record, and the returned Open-Meteo timestamp is preserved.

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

The browser renders only the response returned by the backend. It does not call
Open-Meteo, calculate weather values, select an SOP, or persist accounts. A
single conversation area keeps the current session's messages, and errors are
shown as ordinary assistant messages.

## API contract

`POST /chat` accepts:

```json
{"session_id":"a-session-id","message":"Can I go running in Bhopal?"}
```

The response includes `answer`, the resolved `location`, requested time,
weather facts when available, `sop_id`, `severity`, and error fields when a
location, weather, or provider step fails. `GET /health` returns `{"status":"ok"}`
for deployment health checks.

## Deployment

The included [`render.yaml`](render.yaml) describes a small Render web service
for the FastAPI backend. It installs `requirements.txt`, binds to Render's
`PORT`, and uses `/health` for health checks. Configure `GEMINI_API_KEY`,
`GEMINI_MODEL`, and `CORS_ORIGINS` as Render environment variables; secrets are
never sent to the browser.

The Vite frontend can be deployed as a static site on Vercel, Cloudflare Pages,
or another static host with the project root set to `frontend`, build command
`npm run build`, and output directory `dist`. Set `VITE_API_BASE_URL` to the
deployed backend URL in that host's build environment. Do not use the local
`.env.example` value in production.

Deployment and browser verification are environment-dependent. This checkout
does not claim a current public deployment URL until those services are
actually deployed and checked.

## Demo and submission links

- Repository: [weather-advisory-bot](https://github.com/Dev-Moddh2211/weather-advisory-bot)
- Deployed project: pending deployment verification
- Screen recording: pending recording

For the 5–10 minute recording, demonstrate a normal SOP-backed question, a
follow-up such as “What about this evening?” in the same chat, a no-SOP case,
an API/weather failure case, and a short walkthrough of the graph, YAML SOPs,
and session ID handling. Do not stage or claim a severe-weather result unless
the live conditions actually trigger one.

## Evaluation

The evaluator covers deterministic SOP matching, canonical paraphrase normalization,
false-positive protection, adversarial requests, overlapping policies, a deterministic
HIGH-severity thunderstorm path, failure paths, bounded Gemini retries, response
grounding, three-turn session memory, temporary SOP extensibility, and repository
security checks.

```bash
.venv/bin/python -m pytest
.venv/bin/python evals/run_evals.py
```

Install test dependencies with `pip install -r requirements.txt`. Pytest tests
cover SOP validation, duplicate IDs, situational routing, the `/chat` pipeline,
weather failure, no-match behavior, response citation, and deterministic conflict
selection. The custom runner additionally covers paraphrase/intake fixtures,
session memory, retry handling, and YAML-only SOP extensibility.

The live severe-weather check uses the normal geocoder and weather client when
`LIVE_SEVERE_LOCATION` is set, for example:

```bash
LIVE_SEVERE_LOCATION="a location to review" .venv/bin/python evals/run_evals.py
```

It records the actual Open-Meteo payload only when a current HIGH-severity policy
is genuinely triggered. Otherwise it reports `SKIP`; it never fabricates weather.

The live severe-weather probe is informational: `PASS` means a real candidate currently
matched a HIGH policy, `SKIP` means no candidate matched or the service was unavailable,
and neither result is converted into the other. The deterministic mocked thunderstorm
case separately proves that the HIGH-severity branch works without depending on current
weather.

### Mocked versus live checks

`evals/run_evals.py` is the deterministic evaluator. Its normal cases use fixtures
and monkeypatches and do not claim live Gemini success. Pytest includes a mocked
end-to-end `POST /chat` path through LangGraph, while `evals/test_unreachable_weather.py`
deliberately mocks an unreachable weather node and verifies that the graph returns a
weather error before SOP matching.

For local live verification, start the backend with the real `.env`, then run:

```bash
python3 scripts/live_integration.py
```

This script calls the running `POST /chat` endpoint for a Delhi picnic and a Bhopal cycling request. It does not mock Gemini, Open-Meteo, or the backend. It prints the configured model name only (never the API key), the HTTP status, live weather, selected SOP, and final advisory. The Bhopal case expects no applicable SOP under the live conditions; if current weather matches a cycling policy, the check fails rather than fabricating or overriding the result.

## Scope and limitations

There is no authentication, database, account system, analytics dashboard,
voice assistant, or permanent chat history. Session memory is process-local and
lasts only while the backend process retains the LangGraph checkpoint. Gemini
live success, current severe-weather matches, public deployment, and browser
behavior remain separate verification steps rather than assumptions from a
successful build or mocked test.
