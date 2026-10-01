"""Local-only live integration checks for a running Weather Advisory backend.

Unlike ``evals/run_evals.py``, this script does not import or patch backend
nodes. It calls the running HTTP API, which in turn calls Open-Meteo and,
when GEMINI_API_KEY is configured, Gemini.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv


def _print_case(name: str, payload: dict[str, Any], expected_sop: str | None) -> None:
    selected = payload.get("selected_sop")
    selected_id = selected.get("id") if isinstance(selected, dict) else None
    print(f"\n{name}")
    print(f"HTTP status: {payload['_http_status']}")
    print(f"selected SOP: {selected_id or 'NONE'}")
    print(f"weather: {payload.get('weather')}")
    print(f"final advisory: {payload.get('answer', '<missing>')}")
    if expected_sop != selected_id:
        expected = expected_sop or "NONE"
        raise AssertionError(f"expected selected SOP {expected}, got {selected_id or 'NONE'}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Call a running backend with live Gemini/Open-Meteo data.")
    parser.add_argument("--base-url", default=os.getenv("LIVE_BACKEND_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()

    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model:
        print("FAIL: GEMINI_MODEL is not configured.", file=sys.stderr)
        return 2
    print(f"Gemini model: {model}")
    print("Mode: LIVE HTTP /chat; Gemini and Open-Meteo are not mocked.")

    cases = (
        ("A. Delhi picnic SOP-match", "live-delhi-picnic", "Can I have a picnic in Delhi today?", "SOP-010"),
        ("B. Bhopal cycling no-SOP", "live-bhopal-cycling", "Is cycling in Bhopal safe today?", None),
    )
    try:
        with httpx.Client(timeout=60) as client:
            for name, session_id, message, expected_sop in cases:
                response = client.post(f"{args.base_url.rstrip('/')}/chat", json={"session_id": session_id, "message": message})
                response.raise_for_status()
                payload = response.json()
                payload["_http_status"] = response.status_code
                _print_case(name, payload, expected_sop)
    except httpx.ConnectError as exc:
        print(f"FAIL: backend unavailable at {args.base_url}: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"FAIL: backend returned HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except (httpx.HTTPError, ValueError, AssertionError) as exc:
        print(f"FAIL: live integration check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
