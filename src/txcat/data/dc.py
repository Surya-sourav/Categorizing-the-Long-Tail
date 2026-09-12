"""Washington DC Purchase Card Transactions loader (ArcGIS REST, paged, 2019+).

Only OBJECTID, TRANSACTION_DATE, VENDOR_NAME, MCC_DESCRIPTION are ever requested; agency, amount and
vendor state never leave the API response.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

LAYER_URL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Service_WebMercator/MapServer/50/query"
)
OUT_FIELDS = "OBJECTID,TRANSACTION_DATE,VENDOR_NAME,MCC_DESCRIPTION"
PAGE_SIZE = 1000  # layer maxRecordCount


def build_query_params(start_date: str, offset: int, page_size: int = PAGE_SIZE) -> dict:
    """ArcGIS query params for one page of rows on/after ``start_date`` (YYYY-MM-DD)."""
    return {
        "where": f"TRANSACTION_DATE >= DATE '{start_date}'",
        "outFields": OUT_FIELDS,
        "returnGeometry": "false",
        "orderByFields": "OBJECTID ASC",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "f": "json",
    }


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
    df["date"] = pd.to_datetime(df["date"])
    return df


def download_dc(raw_dir: str | Path, start_date: str = "2019-01-01", sleep_s: float = 0.2) -> Path:
    """Page through the layer and checkpoint each page to ``raw_dir`` as JSON. Resumable."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    session = requests.Session()
    while True:
        page_path = raw_dir / f"page_{offset:07d}.json"
        if page_path.exists():
            feats = json.loads(page_path.read_text()).get("features", [])
        else:
            for attempt in range(5):
                try:
                    params = build_query_params(start_date, offset)
                    r = session.get(LAYER_URL, params=params, timeout=60)
                    r.raise_for_status()
                    payload = r.json()
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    break
                except Exception as e:  # noqa: BLE001
                    wait = 2**attempt
                    logger.warning(f"DC page {offset} failed ({e}); retry in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"DC download failed at offset {offset}")
            page_path.write_text(json.dumps(payload))
            feats = payload.get("features", [])
            time.sleep(sleep_s)
        logger.info(f"DC offset {offset}: {len(feats)} features")
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return raw_dir


def build_processed(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Concatenate checkpointed pages into the processed parquet."""
    frames = [features_to_frame(json.loads(p.read_text()).get("features", []))
              for p in sorted(Path(raw_dir).glob("page_*.json"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("txn_id").sort_values("date")
    df = df.reset_index(drop=True)
    validate_processed(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"DC processed: {len(df)} rows -> {out_path}")
    return df
