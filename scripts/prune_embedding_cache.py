"""Keep only embedding-cache vectors for real-data merchant strings (DC, Oklahoma, FES).

Synthetic generator strings are dropped: they regenerate locally in minutes and would bloat the
committed cache.

Usage: python scripts/prune_embedding_cache.py [--config configs/dc.yaml] [--dry-run]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from txcat.config import load_config
from txcat.data.prepare import load_prepared
from txcat.utils import sha1_text

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/dc.yaml")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)

keep_texts: set[str] = set()
for name in ("dc", "oklahoma"):
    keep_texts |= set(
        load_prepared(cfg.datasets[name].processed_path, cfg.taxonomy_dir)["merchant"]
    )
keep = {sha1_text(t) for t in keep_texts}
print(f"real-data merchant strings: {len(keep):,}")

for d in sorted(Path(cfg.embed.cache_dir).iterdir()):
    kp, vp = d / "keys.json", d / "vectors.npy"
    if not (kp.exists() and vp.exists()):
        continue
    keys = json.loads(kp.read_text())
    vecs = np.load(vp)
    mask = np.array([k in keep for k in keys])
    before, after = vecs.nbytes / 1e6, vecs[mask].nbytes / 1e6
    kept = int(mask.sum())
    print(f"{d.name}: {len(keys):,} -> {kept:,} vectors ({before:.0f} MB -> {after:.0f} MB)")
    if not args.dry_run and mask.sum() < len(keys):
        np.save(vp, vecs[mask])
        kp.write_text(json.dumps([k for k, m in zip(keys, mask, strict=True) if m]))
