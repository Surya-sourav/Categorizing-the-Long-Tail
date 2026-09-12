"""The one processed transaction schema every loader must emit."""

from __future__ import annotations

import pandas as pd

PROCESSED_COLUMNS = ["txn_id", "date", "raw_merchant", "mcc_description", "source"]
SOURCES = {"dc", "ok"}


def validate_processed(df: pd.DataFrame) -> None:
    """Raise ValueError unless ``df`` has exactly the processed columns with sane dtypes."""
    if list(df.columns) != PROCESSED_COLUMNS:
        raise ValueError(
            f"processed columns must be exactly {PROCESSED_COLUMNS}, got {list(df.columns)}"
        )
    if len(df) == 0:
        raise ValueError("processed frame is empty")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("date must be datetime64")
    if df["raw_merchant"].isna().any() or df["date"].isna().any():
        raise ValueError("raw_merchant and date must be non-null")
    if (df["raw_merchant"].str.strip() == "").any():
        raise ValueError("raw_merchant must be non-empty after stripping")
    unknown = set(df["source"].unique()) - SOURCES
    if unknown:
        raise ValueError(f"source must be one of {sorted(SOURCES)}, got {sorted(unknown)}")
    if df["txn_id"].duplicated().any():
        raise ValueError("txn_id must be unique")
