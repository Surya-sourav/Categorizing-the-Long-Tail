"""Oklahoma State PCard loader (CKAN monthly CSVs).

Cardholder, amount, item and agency are dropped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import IO

import pandas as pd
import requests
from loguru import logger

from txcat.data import RETRIES, is_fatal_http_error
from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

CKAN_SHOW = "https://data.ok.gov/api/3/action/package_show?id={pkg}"
KEEP = ["ROWID", "TRANSACTION_DATE", "MERCHANT", "MCC_DESCRIPTION"]
REQUIRED = {"TRANSACTION_DATE", "MERCHANT", "MCC_DESCRIPTION"}


def csv_to_frame(src: str | Path | IO[str], file_id: str = "") -> pd.DataFrame:
    """Read one monthly CSV, keep only the four needed columns, emit the processed schema.

    ``ROWID`` is an Oracle rowid that the publisher reuses across monthly extracts, so it is unique
    only within one file; ``file_id`` (the file stem) namespaces it into a corpus-unique txn_id.
    Two published months (Aug/Sep 2023) ship without ``ROWID`` at all; for those the txn_id falls
    back to the source row position, which is stable because the downloaded files are immutable.
    """
    raw = pd.read_csv(src, usecols=lambda c: c in KEEP, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"{file_id or src}: CSV is missing required columns {sorted(missing)}")
    has_rowid = "ROWID" in raw.columns
    if not has_rowid:
        logger.warning(
            f"{file_id or src}: no ROWID column; falling back to source row positions for txn_id"
        )
    raw = raw[(raw["MERCHANT"].str.strip() != "") & (raw["TRANSACTION_DATE"].str.strip() != "")]
    prefix = f"ok:{file_id}:" if file_id else "ok:"
    if has_rowid:
        txn_id = prefix + raw["ROWID"].str.strip()
    else:
        txn_id = prefix + "row" + raw.index.astype(str)
    df = pd.DataFrame(
        {
            "txn_id": txn_id,
            "date": pd.to_datetime(
                raw["TRANSACTION_DATE"].str.strip(), format="%d-%b-%y", errors="coerce"
            ),
            "raw_merchant": raw["MERCHANT"].str.strip(),
            "mcc_description": raw["MCC_DESCRIPTION"].str.strip(),
            "source": "ok",
        }
    )[PROCESSED_COLUMNS]
    df = df[df["date"].notna()].reset_index(drop=True)
    # One unit across both loaders so a cross-dataset concat never has to reconcile resolutions.
    df["date"] = df["date"].astype("datetime64[us]")
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


def _write_bytes_atomic(dest: Path, content: bytes, expected_size: int | None) -> None:
    """Write via a .tmp sibling and os.replace; a short body is never promoted to the real name."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_bytes(content)
        size = tmp.stat().st_size
        if expected_size is not None and size != expected_size:
            raise RuntimeError(
                f"{dest.name}: wrote {size} bytes but Content-Length said {expected_size}"
            )
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _download_one(url: str, dest: Path, name: str) -> None:
    """Fetch one monthly CSV, retrying transient failures, and land it atomically."""
    last_exc: BaseException | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            # requests transparently decompresses, so a compressed Content-Length would not match
            compressed = bool(r.headers.get("Content-Encoding"))
            declared = None if compressed else r.headers.get("Content-Length")
            _write_bytes_atomic(dest, r.content, int(declared) if declared else None)
            logger.info(f"OK downloaded {dest.name} ({len(r.content) / 1e6:.1f} MB)")
            return
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if is_fatal_http_error(e):
                logger.error(f"{name} failed with a client error ({e}); not retrying")
                break
            if attempt < RETRIES - 1:
                wait = 2**attempt
                logger.warning(f"{name} failed ({e}); retry in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Oklahoma download failed: {name}") from last_exc


def download_oklahoma(raw_dir: str | Path, packages: list[str]) -> Path:
    """Download every monthly CSV into ``raw_dir`` (skips files already present)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in list_resource_urls(packages):
        dest = raw_dir / (Path(name).stem + ".csv")
        if dest.exists():
            continue
        _download_one(url, dest, name)
    return raw_dir


def build_processed(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Concatenate all monthly CSVs to the processed parquet."""
    frames = [csv_to_frame(p, file_id=p.stem) for p in sorted(Path(raw_dir).glob("*.csv"))]
    if not frames:
        raise ValueError(f"no Oklahoma CSVs found under {raw_dir}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("txn_id")
    df = df.sort_values(["date", "txn_id"], kind="stable").reset_index(drop=True)
    validate_processed(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Oklahoma processed: {len(df)} rows -> {out_path}")
    return df
