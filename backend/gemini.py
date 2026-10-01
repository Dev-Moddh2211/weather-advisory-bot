from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")

MAX_GEMINI_ATTEMPTS = 3
GEMINI_RETRY_BASE_SECONDS = 1.0


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini remains unavailable after bounded retries."""


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource exhausted",
            "temporarily unavailable",
            "service unavailable",
            "high demand",
            "internal server error",
        )
    )


def invoke_gemini(invoke: Callable[[], T]) -> T:
    """Invoke Gemini with bounded retries for transient provider failures."""
    for attempt in range(MAX_GEMINI_ATTEMPTS):
        try:
            return invoke()
        except Exception as exc:
            if not _is_transient_error(exc) or attempt == MAX_GEMINI_ATTEMPTS - 1:
                raise GeminiUnavailableError(
                    "Gemini is temporarily unavailable. Please try again later."
                ) from None
            delay = GEMINI_RETRY_BASE_SECONDS * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
    raise AssertionError("unreachable")


def gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
