"""Run manifest: everything a reader needs to know to trust a results directory."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from txcat.config import _ROOT, rel_to_root
from txcat.utils import now_iso, sha256_text

PACKAGES = [
    "numpy",
    "pandas",
    "hnswlib",
    "sentence-transformers",
    "torch",
    "scikit-learn",
    "openai",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_manifest(prompt_paths: list[str | Path], config: dict, model_versions: dict) -> dict:
    return {
        "timestamp": now_iso(),
        "git_commit": _git_commit(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "python": sys.version,
        "packages": {p: _ver(p) for p in PACKAGES},
        "prompt_hashes": {rel_to_root(p): sha256_text(Path(p).read_text()) for p in prompt_paths},
        "hnsw_params": config.get("index", {}),
        "model_versions": model_versions,
        "config": _relativize(config),
    }


def _relativize(obj):
    """Config paths are absolute in memory; the committed manifest must not leak a home dir."""
    if isinstance(obj, dict):
        return {k: _relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(str(_ROOT)):
        return rel_to_root(obj)
    return obj


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "missing"
