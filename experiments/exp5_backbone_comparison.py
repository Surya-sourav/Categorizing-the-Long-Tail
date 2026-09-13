"""Exp 5: kNN head/tail/overall macro-F1 and embedding latency per backbone on the DC FES + full test set.

Local backbones run on CPU; the OpenAI backbone is skipped unless OPENAI_API_KEY is set (or in
reproduce mode, unless its cache is complete).

Usage: python -m experiments.exp5_backbone_comparison --config configs/dc.yaml --seeds 42 43 44 [--mode reproduce]
Outputs: results/tables/tab4_backbone_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import freq_of, merchant_frequencies, temporal_split
from txcat.embedder import CacheMissError, Embedder
from txcat.knn import build_merchant_index, predict_knn
from txcat.metrics import bootstrap_ci, macro_f1
from txcat.utils import set_seed, setup_logging

ALL_BACKBONES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "ProsusAI/finbert",
    "text-embedding-3-small",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--mode", choices=["live", "reproduce"], default="live")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    live = args.mode == "live"
    setup_logging(cfg.logs_dir, "exp5")
    tables = Path(cfg.results_dir, "tables")
    tables.mkdir(parents=True, exist_ok=True)

    d = cfg.datasets["dc"]
    decision = json.loads(Path(cfg.results_dir, "window_decision.json").read_text())
    win = WindowCfg(**decision["chosen_window"])
    df = load_prepared(d.processed_path, cfg.taxonomy_dir)
    df = df[df["category"].notna()]
    train, test = temporal_split(df, win)
    freqs = merchant_frequencies(train)
    test = test[test["ambiguous"] == 0].copy()
    test["is_tail"] = freq_of(test, freqs) <= cfg.tail_k

    rows = []
    for bb in args.backbones or ALL_BACKBONES:
        if bb.startswith("text-embedding") and live and not os.environ.get("OPENAI_API_KEY"):
            logger.warning(f"skipping {bb}: OPENAI_API_KEY not set")
            continue
        emb = Embedder(bb, cfg.embed.cache_dir, allow_live=live, batch_size=cfg.embed.batch_size)
        try:
            uniq = test["merchant"].drop_duplicates().tolist()
            # time the raw backend on a fresh batch so cache hits do not flatter the number
            if live:
                sample = [f"{m} TIMING-{i}" for i, m in enumerate(uniq[:200])]
                t0 = time.perf_counter()
                emb.backend.encode(sample)
                embed_ms_per_merchant = (time.perf_counter() - t0) * 1000 / len(sample)
            else:
                embed_ms_per_merchant = float("nan")  # not measurable without live encoding
            emb.embed(uniq)
            f1s = []
            for seed in seeds:
                set_seed(seed)
                idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
                p = predict_knn(test, idx, emb, cfg.index.k, cfg.index.ef_search)
                f1s.append(
                    {
                        "overall": macro_f1(test["category"], p["pred"]),
                        "head": macro_f1(test["category"], p["pred"], ~test["is_tail"]),
                        "tail": macro_f1(test["category"], p["pred"], test["is_tail"]),
                        "acc_tail": float((p["pred"] == test["category"])[test["is_tail"]].mean()),
                    }
                )
        except CacheMissError as e:
            logger.warning(f"skipping {bb} in reproduce mode: {e}")
            continue
        row = {
            "backbone": bb,
            "dim": int(emb.embed([uniq[0]]).shape[1]),
            "n_seeds": len(seeds),
            "embed_ms_per_merchant": round(embed_ms_per_merchant, 3),
        }
        for k in ("overall", "head", "tail", "acc_tail"):
            lo, mean, hi = bootstrap_ci([f[k] for f in f1s])
            row[f"f1_{k}" if k != "acc_tail" else k] = round(mean, 4)
            row[(f"f1_{k}" if k != "acc_tail" else k) + "_lo"] = round(lo, 4)
            row[(f"f1_{k}" if k != "acc_tail" else k) + "_hi"] = round(hi, 4)
        rows.append(row)
        logger.info(row)
    out = pd.DataFrame(rows)
    out.to_csv(tables / "tab4_backbone_comparison.csv", index=False)
    logger.info("\n" + out.to_string())


if __name__ == "__main__":
    main()
