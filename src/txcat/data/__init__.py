"""Dataset loaders. Every loader emits the processed schema and drops PII at the boundary."""

from __future__ import annotations

import requests

RETRIES = 5


def is_fatal_http_error(exc: BaseException) -> bool:
    """True for client errors retrying cannot fix: any 4xx except 429 (rate limited)."""
    if not isinstance(exc, requests.HTTPError):
        return False
    response = getattr(exc, "response", None)
    if response is None:
        return False
    return 400 <= response.status_code < 500 and response.status_code != 429
