"""Download raw datasets (never redistributed) and build processed parquets.

Usage: python data/download.py [dc] [oklahoma] [mcc_codes] [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from loguru import logger

from txcat.config import load_config
from txcat.data import dc, oklahoma

MCC_CODES_URL = "https://raw.githubusercontent.com/greggles/mcc-codes/main/mcc_codes.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", choices=["dc", "oklahoma", "mcc_codes"])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if "mcc_codes" in args.targets:
        dest = Path(cfg.taxonomy_dir) / "mcc_codes_source.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(requests.get(MCC_CODES_URL, timeout=60).content)
        logger.info(f"mcc codes -> {dest}")
    if "dc" in args.targets:
        d = cfg.datasets["dc"]
        dc.download_dc(d.raw_dir, start_date=d.window.train_start)
        dc.build_processed(d.raw_dir, d.processed_path)
    if "oklahoma" in args.targets:
        d = cfg.datasets["oklahoma"]
        oklahoma.download_oklahoma(d.raw_dir, d.ckan_packages)
        oklahoma.build_processed(d.raw_dir, d.processed_path)


if __name__ == "__main__":
    main()
