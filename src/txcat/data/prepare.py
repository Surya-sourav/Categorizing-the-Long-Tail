"""Processed parquet -> analysis frame: adds ``merchant`` (normalized) and label columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from txcat.data.taxonomy import attach_labels
from txcat.normalizer import normalize_merchant


def prepare(df: pd.DataFrame, taxonomy_dir: str | Path) -> pd.DataFrame:
    out = df.copy()
    uniq = out["raw_merchant"].drop_duplicates()
    norm = {r: normalize_merchant(r).text for r in uniq}
    out["merchant"] = out["raw_merchant"].map(norm)
    out = attach_labels(out, taxonomy_dir)
    return out


def load_prepared(processed_path: str | Path, taxonomy_dir: str | Path) -> pd.DataFrame:
    return prepare(pd.read_parquet(processed_path), taxonomy_dir)
