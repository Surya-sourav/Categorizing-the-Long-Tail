"""Freeze the API caches: pack per-call JSON files into gzip JSONL per group and remove the per-call
files (they stay reproducible from the packed set). Run once before committing the cache.

Usage: python scripts/pack_cache.py [--keep-files]
"""

import argparse

from txcat.cachepack import pack
from txcat.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/dc.yaml")
ap.add_argument("--keep-files", action="store_true", help="leave the per-call JSON files in place")
args = ap.parse_args()
cfg = load_config(args.config)
for cache_dir, kind in ((cfg.llm.cache_dir, "llm"), (cfg.search.cache_dir, "search")):
    counts = pack(cache_dir, kind, remove=not args.keep_files)
    for g, n in sorted(counts.items()):
        print(f"{kind:6s} {g:60s} {n:6d}")
