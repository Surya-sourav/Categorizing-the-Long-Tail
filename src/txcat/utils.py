"""Shared helpers: seeding, hashing, logging, slugs, timestamps."""

from __future__ import annotations

import atexit
import hashlib
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

_LOG_HANDLERS: dict[str, tuple[Path, int]] = {}
_ATEXIT_REGISTERED = False


def set_seed(seed: int) -> None:
    """Seed ``random``, the numpy global RNG and torch (cpu + mps, if available).

    This covers library internals that read global state. Downstream code must use an
    explicit ``np.random.default_rng(seed)`` for anything whose reproducibility matters.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def sha1_text(text: str) -> str:
    """Stable SHA-1 hex digest of a UTF-8 string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Stable SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for a model id (``org/name`` -> ``org__name``)."""
    return model_name.replace("/", "__").replace(":", "_")


def now_iso() -> str:
    """UTC timestamp in ISO-8601 with seconds precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(log_dir: str | Path, name: str) -> tuple[Path, int]:
    """Add a file sink under ``log_dir`` and return ``(path, handler_id)``.

    Idempotent per ``name``: a second call with the same name returns the existing
    sink instead of adding a duplicate (which would double every log line).
    """
    global _ATEXIT_REGISTERED
    if name in _LOG_HANDLERS:
        return _LOG_HANDLERS[name]

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
    handler_id = logger.add(path, level="INFO", enqueue=True)
    _LOG_HANDLERS[name] = (path, handler_id)

    if not _ATEXIT_REGISTERED:
        atexit.register(logger.complete)
        _ATEXIT_REGISTERED = True

    return path, handler_id
