"""Pack per-call JSON cache files into one gzip JSONL per group, and read packed caches.

Live runs write one file per API call (crash-safe, resumable). At freeze time ``pack`` folds them
into ``<cache_dir>/packed/<group>.jsonl.gz`` (LLM: one per model x prompt file; search: one per
provider) so the committed cache is a handful of files instead of tens of thousands. Readers consult
the per-call file first, then the packed index; both layouts stay valid.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

from loguru import logger

from txcat.utils import model_slug


def _group_llm(rec: dict) -> str:
    return f"{model_slug(rec['model'])}__{Path(rec['prompt_file']).stem}"


def _group_search(rec: dict) -> str:
    return rec["provider"]


def pack(cache_dir: str | Path, kind: str, remove: bool = False) -> dict[str, int]:
    """Fold ``<cache_dir>/*.json`` into ``<cache_dir>/packed/<group>.jsonl.gz``.

    Each line is ``{"key": <file stem>, **record}``. Returns records packed per group. Existing
    packed records are kept and de-duplicated by key (per-call files win).
    """
    cache_dir = Path(cache_dir)
    packed_dir = cache_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)
    grouper = _group_llm if kind == "llm" else _group_search
    groups: dict[str, dict[str, dict]] = {}
    for gz in packed_dir.glob("*.jsonl.gz"):
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                groups.setdefault(gz.name[: -len(".jsonl.gz")], {})[rec["key"]] = rec
    files = list(cache_dir.glob("*.json"))
    for f in files:
        rec = json.loads(f.read_text())
        rec["key"] = f.stem
        groups.setdefault(grouper(rec), {})[f.stem] = rec
    counts = {}
    for g, recs in groups.items():
        final = packed_dir / f"{g}.jsonl.gz"
        tmp = packed_dir / f"{g}.jsonl.gz.tmp"
        # Whole file, then atomic rename. mtime=0 + sorted keys => byte-identical repacks, so a
        # reproduction run never dirties the committed cache.
        with (
            gzip.GzipFile(tmp, "wb", mtime=0) as raw,
            io.TextIOWrapper(raw, encoding="utf-8") as fh,
        ):
            for key in sorted(recs):
                fh.write(json.dumps(recs[key], ensure_ascii=False) + "\n")
        tmp.replace(final)
        counts[g] = len(recs)
    if remove:
        for f in files:
            f.unlink()
    logger.info(
        f"packed {sum(counts.values())} {kind} records into {len(counts)} files under {packed_dir}"
    )
    return counts


def load_packed(cache_dir: str | Path) -> dict[str, dict]:
    """Load every packed record under ``<cache_dir>/packed`` into a dict keyed by cache key."""
    out: dict[str, dict] = {}
    packed_dir = Path(cache_dir) / "packed"
    if not packed_dir.exists():
        return out
    for gz in sorted(packed_dir.glob("*.jsonl.gz")):
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out
