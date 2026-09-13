"""Washington DC Purchase Card Transactions loader (ArcGIS REST, paged, 2019+).

Only OBJECTID, TRANSACTION_DATE, VENDOR_NAME, MCC_DESCRIPTION are ever requested; agency, amount and
vendor state never leave the API response.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from txcat.data import RETRIES, is_fatal_http_error
from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

LAYER_URL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Service_WebMercator/MapServer/50/query"
)
OUT_FIELDS = "OBJECTID,TRANSACTION_DATE,VENDOR_NAME,MCC_DESCRIPTION"
PAGE_SIZE = 1000  # layer maxRecordCount
META_NAME = "_meta.json"


def _where(start_date: str) -> str:
    return f"TRANSACTION_DATE >= DATE '{start_date}'"


def build_query_params(start_date: str, offset: int, page_size: int = PAGE_SIZE) -> dict:
    """ArcGIS query params for one page of rows on/after ``start_date`` (YYYY-MM-DD)."""
    return {
        "where": _where(start_date),
        "outFields": OUT_FIELDS,
        "returnGeometry": "false",
        "orderByFields": "OBJECTID ASC",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "f": "json",
    }


def build_count_params(start_date: str) -> dict:
    """ArcGIS params for the row count of the same window, used to verify a finished download."""
    return {"where": _where(start_date), "returnCountOnly": "true", "f": "json"}


def features_to_frame(features: list[dict]) -> pd.DataFrame:
    """Convert ArcGIS features to the processed schema, dropping PII and null rows."""
    rows = []
    for f in features:
        a = f["attributes"]
        if a.get("TRANSACTION_DATE") is None or not a.get("VENDOR_NAME"):
            continue
        rows.append(
            {
                "txn_id": f"dc:{a['OBJECTID']}",
                "date": pd.to_datetime(a["TRANSACTION_DATE"], unit="ms").normalize(),
                "raw_merchant": str(a["VENDOR_NAME"]),
                "mcc_description": str(a.get("MCC_DESCRIPTION") or ""),
                "source": "dc",
            }
        )
    df = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
    # One unit across both loaders so a cross-dataset concat never has to reconcile resolutions.
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
    return df


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write via a .tmp sibling and os.replace so an interrupted write never looks complete."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _get_json(session: requests.Session, params: dict, what: str) -> dict:
    """GET one ArcGIS response, retrying transient failures; ArcGIS error payloads are failures."""
    last_exc: BaseException | None = None
    for attempt in range(RETRIES):
        try:
            r = session.get(LAYER_URL, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(f"ArcGIS error payload: {payload['error']}")
            return payload
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if is_fatal_http_error(e):
                logger.error(f"DC {what} failed with a client error ({e}); not retrying")
                break
            if attempt < RETRIES - 1:
                wait = 2**attempt
                logger.warning(f"DC {what} failed ({e}); retry in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"DC download failed at {what}") from last_exc


def _read_meta(raw_dir: Path) -> dict | None:
    meta_path = raw_dir / META_NAME
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"{meta_path} is corrupt; ignoring it")
        return None


def write_meta(raw_dir: str | Path, start_date: str, page_size: int, total_count: int) -> Path:
    """Record what window these raw pages came from, so a later resume cannot mix windows."""
    meta_path = Path(raw_dir) / META_NAME
    meta_path.write_text(
        json.dumps(
            {
                "layer_url": LAYER_URL,
                "where": _where(start_date),
                "start_date": start_date,
                "page_size": page_size,
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "total_count": total_count,
            },
            indent=2,
        )
    )
    return meta_path


def _check_meta(raw_dir: Path, start_date: str) -> None:
    meta = _read_meta(raw_dir)
    if meta is None:
        return
    if meta.get("where") != _where(start_date) or meta.get("start_date") != start_date:
        raise RuntimeError(
            f"{raw_dir} holds pages for start_date={meta.get('start_date')!r} "
            f"(where={meta.get('where')!r}) but this run asks for {start_date!r}; "
            f"clear {raw_dir} before downloading a different window"
        )


def _load_page(page_path: Path) -> dict | None:
    """Return a cached page, or None if it is missing or corrupt (then it is deleted)."""
    if not page_path.exists():
        return None
    try:
        return json.loads(page_path.read_text())
    except json.JSONDecodeError as e:
        logger.warning(f"{page_path.name} is corrupt ({e}); deleting and refetching")
        page_path.unlink()
        return None


def download_dc(
    raw_dir: str | Path,
    start_date: str = "2019-01-01",
    sleep_s: float = 0.2,
    page_size: int = PAGE_SIZE,
) -> Path:
    """Page through the layer and checkpoint each page to ``raw_dir`` as JSON. Resumable.

    Paging follows the layer's ``exceededTransferLimit`` flag rather than page fullness, and the
    finished download is checked against a ``returnCountOnly`` query (the layer is static per day,
    so the tolerance is zero).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _check_meta(raw_dir, start_date)
    session = requests.Session()
    offset = 0
    total = 0
    while True:
        page_path = raw_dir / f"page_{offset:07d}.json"
        payload = _load_page(page_path)
        if payload is None:
            payload = _get_json(session, build_query_params(start_date, offset, page_size),
                                f"offset {offset}")
            _write_json_atomic(page_path, payload)
            time.sleep(sleep_s)
        if "features" not in payload:
            raise RuntimeError(
                f"DC offset {offset}: response has no 'features' key (got {sorted(payload)})"
            )
        feats = payload["features"]
        total += len(feats)
        logger.info(f"DC offset {offset}: {len(feats)} features")
        if not payload.get("exceededTransferLimit"):
            break
        offset += page_size

    expected = _get_json(session, build_count_params(start_date), "count")["count"]
    if total != expected:
        raise RuntimeError(
            f"DC download incomplete: {total} rows across pages but the layer reports {expected}"
        )
    write_meta(raw_dir, start_date, page_size, expected)
    logger.info(f"DC download verified: {total} rows == layer count")
    return raw_dir


def _page_features(page_path: Path) -> list[dict]:
    """Features of a checkpointed page; a page without them is a bug, not an empty page."""
    payload = json.loads(page_path.read_text())
    if "features" not in payload:
        raise ValueError(f"{page_path} has no 'features' key (got {sorted(payload)})")
    return payload["features"]


def build_processed(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Concatenate checkpointed pages into the processed parquet."""
    frames = [features_to_frame(_page_features(p))
              for p in sorted(Path(raw_dir).glob("page_*.json"))]
    if not frames:
        raise ValueError(f"no DC pages found under {raw_dir}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("txn_id")
    df = df.sort_values(["date", "txn_id"], kind="stable").reset_index(drop=True)
    meta_path = Path(raw_dir) / "_meta.json"
    if meta_path.exists():
        expected = json.loads(meta_path.read_text()).get("total_count")
        if expected is not None and len(df) != expected:
            raise RuntimeError(
                f"DC processed has {len(df)} rows but _meta.json recorded {expected}; "
                "a page file is missing or truncated - re-run the download"
            )
    validate_processed(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"DC processed: {len(df)} rows -> {out_path}")
    return df
