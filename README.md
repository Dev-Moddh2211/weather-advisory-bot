# Weather Advisory Support Bot

This project answers questions about outdoor activities using live weather data
from Open-Meteo and a set of written safety policies. A request is parsed into
structured context, matched against the configured policies, and returned with
the policy used when one applies.

## Architecture

```text
User message
    ↓
Intent and context extraction
    ↓
Location resolution → Weather retrieval
    ↓                      ↓ failure
SOP matching          Error response
    ↓
Policy selection → No matching policy / Response generation
```

The backend is a LangGraph workflow with separate nodes for intake, location,
weather, matching, failure handling, and response generation. Session state is
checkpointed by `session_id` so follow-up questions can reuse earlier context.

## Design decisions

- Safety decisions are deterministic. SOPs are loaded from YAML, evaluated
  against the extracted intent and fetched weather, and ranked by severity and
  then priority.
- Gemini is used for intent extraction and wording only. It does not select
  thresholds, choose the winning policy, or invent weather values.
- Situational policies, such as a thunderstorm rule that applies broadly to
  outdoor activities, take precedence over activity-specific matches.
- When no policy applies, the backend returns a fixed no-guidance response
  instead of asking the model to improvise.
- Location and weather failures stop the workflow before policy matching. The
  response does not contain forecast values that were not retrieved.

## SOPs

Policies live in [`sops/sops.yaml`](sops/sops.yaml). The current configuration
contains 16 policies across Outdoor Exercise, Travel, Vulnerable Groups, and
General Outdoor Activities. Each policy contains its conditions, severity,
advice, and priority. Thresholds are project-defined prototype values, not
official weather-safety standards.

The loader validates the YAML with Pydantic. Adding a policy only requires
adding another entry to the YAML file; graph control flow and weather code do
not need to change.

Example:

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

## Weather data

The application resolves a city through Open-Meteo geocoding, then requests a
forecast with explicit current and hourly fields. Current requests use the
current snapshot. Requests for today, morning, afternoon, evening, or tomorrow
use the nearest matching local forecast hour. The selected timestamp and raw
Open-Meteo payload are retained for response and debugging context.

## Session memory

Conversation state is held in process memory through LangGraph's
`InMemorySaver`. For example, a follow-up such as “What about this evening?”
can reuse the location and activity from an earlier cycling question. State is
not persistent across backend restarts or users.

## Setup

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GEMINI_API_KEY locally when using Gemini extraction or wording.
uvicorn backend.main:app --reload
```

The API is `POST http://127.0.0.1:8000/chat`:

```json
{"session_id":"a-session-id","message":"Can I go running in Bhopal?"}
```

`GET /health` returns `{"status":"ok"}`.

### Frontend

In a second terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

The frontend uses `VITE_API_BASE_URL` and stores one session ID in
`sessionStorage`. It does not receive the Gemini key or call Open-Meteo
directly.

## Evaluation

Run the deterministic checks from the repository root:

```bash
.venv/bin/python -m pytest
.venv/bin/python evals/run_evals.py
```

The suite covers policy matching, paraphrased activity intent, overlapping
policies, situational overrides, session follow-ups, weather and location
failures, bounded Gemini retries, response citations, weather grounding, and
adding a temporary policy through YAML.

The optional live severe-weather probe uses the normal geocoder and weather
client:

```bash
LIVE_SEVERE_LOCATION="a location to review" .venv/bin/python evals/run_evals.py
```

It reports `PASS` only when the current API response genuinely triggers a HIGH
policy. Otherwise it reports `SKIP`; current weather conditions are not assumed
to remain stable.

For local HTTP verification, start the backend with the real `.env` and run:

```bash
python3 scripts/live_integration.py
```

This calls the running `/chat` endpoint without mocking Gemini, Open-Meteo, or
the backend. It prints the configured model name, response status, weather,
selected policy, and final advisory, but never the API key.

## Deployment

[`render.yaml`](render.yaml) describes a Render web service for the FastAPI
backend. Configure `GEMINI_API_KEY`, `GEMINI_MODEL`, and `CORS_ORIGINS` as
Render environment variables.

The Vite frontend can be deployed as a static site with project root
`frontend`, build command `npm run build`, and output directory `dist`. Set
`VITE_API_BASE_URL` to the deployed backend URL at build time.

This checkout does not claim a public deployment URL. Deployment, browser
behavior, live Gemini success, and current severe-weather matches require
separate environment verification.

## Scope

There is no authentication, database, account system, analytics dashboard,
voice assistant, or permanent chat history. The application is a focused
weather-advisory prototype.
