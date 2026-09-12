"""Oklahoma State PCard loader (CKAN monthly CSVs).

Cardholder, amount, item and agency are dropped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import IO

import pandas as pd
import requests
from loguru import logger

from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

CKAN_SHOW = "https://data.ok.gov/api/3/action/package_show?id={pkg}"
KEEP = ["ROWID", "TRANSACTION_DATE", "MERCHANT", "MCC_DESCRIPTION"]


def csv_to_frame(src: str | Path | IO[str]) -> pd.DataFrame:
    """Read one monthly CSV, keep only the four needed columns, emit the processed schema."""
    raw = pd.read_csv(src, usecols=KEEP, dtype=str, keep_default_na=False)
    raw = raw[(raw["MERCHANT"].str.strip() != "") & (raw["TRANSACTION_DATE"].str.strip() != "")]
    df = pd.DataFrame(
        {
            "txn_id": "ok:" + raw["ROWID"].str.strip(),
            "date": pd.to_datetime(
                raw["TRANSACTION_DATE"].str.strip(), format="%d-%b-%y", errors="coerce"
            ),
            "raw_merchant": raw["MERCHANT"].str.strip(),
            "mcc_description": raw["MCC_DESCRIPTION"].str.strip(),
            "source": "ok",
        }
    )[PROCESSED_COLUMNS]
    df = df[df["date"].notna()].reset_index(drop=True)
    return df


def list_resource_urls(packages: list[str]) -> list[tuple[str, str]]:
    """Return (name, url) of every CSV resource across the CKAN packages."""
    out = []
    for pkg in packages:
        r = requests.get(CKAN_SHOW.format(pkg=pkg), timeout=60)
        r.raise_for_status()
        for res in r.json()["result"]["resources"]:
            if res.get("format", "").upper() == "CSV":
                out.append((res["name"], res["url"]))
    return out


def download_oklahoma(raw_dir: str | Path, packages: list[str]) -> Path:
    """Download every monthly CSV into ``raw_dir`` (skips files already present)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in list_resource_urls(packages):
        dest = raw_dir / (Path(name).stem + ".csv")
        if dest.exists():
            continue
        for attempt in range(5):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                dest.write_bytes(r.content)
                logger.info(f"OK downloaded {dest.name} ({len(r.content) / 1e6:.1f} MB)")
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{name} failed ({e}); retry")
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Oklahoma download failed: {name}")
    return raw_dir


def build_processed(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Concatenate all monthly CSVs to the processed parquet."""
    frames = [csv_to_frame(p) for p in sorted(Path(raw_dir).glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("txn_id").sort_values("date")
    df = df.reset_index(drop=True)
    validate_processed(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Oklahoma processed: {len(df)} rows -> {out_path}")
    return df
