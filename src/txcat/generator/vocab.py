"""Real merchant vocabulary for the semi-synthetic benchmark, from the OSM name-suggestion-index."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

NSI_VERSION = "8.0.20260729"
NSI_URL = f"https://cdn.jsdelivr.net/npm/name-suggestion-index@{NSI_VERSION}/dist/json/nsi.min.json"
US_CODES = {"us", "001"}


def download_nsi(dest: str | Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(NSI_URL, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        logger.info(f"NSI {NSI_VERSION} -> {dest} ({len(r.content) / 1e6:.1f} MB)")
    return dest


def load_tag_map(path: str | Path) -> dict[tuple[str, str], str]:
    df = pd.read_csv(path)
    return {(r.osm_key, r.osm_value): r.category for r in df.itertuples()}


def items_to_vocab(nsi: dict, tag_map: dict[tuple[str, str], str]) -> pd.DataFrame:
    rows = []
    for path, block in nsi["nsi"].items():
        if not path.startswith("brands/"):
            continue
        _, key, value = path.split("/", 2)
        cat = tag_map.get((key, value))
        if cat is None:
            continue
        for it in block.get("items", []):
            # include entries: country codes, "001" (world), region codes, or [lon, lat, r] points
            inc = {x for x in it.get("locationSet", {}).get("include", []) if isinstance(x, str)}
            if not (inc & US_CODES):
                continue
            name = it["tags"].get("brand") or it["tags"].get("name") or it["displayName"]
            rows.append(
                {
                    "merchant_id": it["id"],
                    "name": name,
                    "osm_key": key,
                    "osm_value": value,
                    "category": cat,
                    "wikidata": it["tags"].get("brand:wikidata", ""),
                }
            )
    v = pd.DataFrame(
        rows, columns=["merchant_id", "name", "osm_key", "osm_value", "category", "wikidata"]
    )
    return v.drop_duplicates("name").sort_values("merchant_id").reset_index(drop=True)


def build_vocab(
    raw_dir: str | Path, tag_map_path: str | Path, out_path: str | Path
) -> pd.DataFrame:
    nsi = json.loads(download_nsi(Path(raw_dir) / f"nsi-{NSI_VERSION}.min.json").read_text())
    v = items_to_vocab(nsi, load_tag_map(tag_map_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    v.to_parquet(out_path, index=False)
    logger.info(f"vocab: {len(v)} US brands across {v['category'].nunique()} categories")
    return v
