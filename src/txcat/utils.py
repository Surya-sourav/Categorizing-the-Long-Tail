"""Shared helpers: seeding, hashing, logging, slugs, timestamps."""

from __future__ import annotations

import hashlib
import os
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (if present) and PYTHONHASHSEED for determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
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


def setup_logging(log_dir: str | Path, name: str) -> Path:
    """Add a file sink under ``log_dir`` and return the log path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(path, level="INFO", enqueue=True)
    return path
