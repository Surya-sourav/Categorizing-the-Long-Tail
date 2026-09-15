"""Exp 1: tail characterization on the DC primary window (kNN with the first configured backbone).

Usage: python -m experiments.exp1_tail_characterization --config configs/dc.yaml --seeds 42 43 44 45 46
Outputs: results/figures/fig1_zipf.{pdf,png}, fig2_acc_by_freq, fig3_acc_vs_sim,
         results/tables/tab1_tail_stats.csv, results/tables/tab1_acc_by_freq.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analysis.plots import fig_acc_by_freq, fig_acc_vs_sim, fig_zipf
from loguru import logger

from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import freq_of, merchant_frequencies, temporal_split
from txcat.embedder import Embedder
from txcat.knn import build_merchant_index, predict_knn
from txcat.utils import set_seed, setup_logging

BINS = [
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 20, "6-20"),
    (21, 100, "21-100"),
    (101, 10**9, "100+"),
]


def bin_label(f: int) -> str:
    for lo, hi, lab in BINS:
        if lo <= f <= hi:
            return lab
    return "100+"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--mode", choices=["live", "reproduce"], default="live")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    backbone = args.backbone or cfg.embed.backbones[0]
    setup_logging(cfg.logs_dir, "exp1")
    figs, tables = Path(cfg.results_dir, "figures"), Path(cfg.results_dir, "tables")
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    d = cfg.datasets["dc"]
    win = WindowCfg(
        **json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"]
    )
    df = load_prepared(d.processed_path, cfg.taxonomy_dir)
    df = df[df["category"].notna()]
    train, test = temporal_split(df, win)
    freqs = merchant_frequencies(train)
    test = test[test["ambiguous"] == 0].copy()  # headline slice
    test["train_freq"] = freq_of(test, freqs)
    test["is_tail"] = test["train_freq"] <= cfg.tail_k

    fig_zipf(freqs, figs / "fig1_zipf")

    emb = Embedder(
        backbone,
        cfg.embed.cache_dir,
        allow_live=(args.mode == "live"),
        batch_size=cfg.embed.batch_size,
    )
    per_seed = []
    for seed in seeds:
        set_seed(seed)
        idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
        pred = predict_knn(test, idx, emb, cfg.index.k, cfg.index.ef_search)
        t = test.assign(pred=pred["pred"].values, sim1=pred["sim1"].values, seed=seed)
        t["correct"] = t["pred"] == t["category"]
        per_seed.append(t)
        tail_acc = t.loc[t.is_tail, "correct"].mean()
        logger.info(f"seed {seed}: overall acc {t['correct'].mean():.4f}, tail acc {tail_acc:.4f}")
    allp = pd.concat(per_seed, ignore_index=True)

    # fig2: accuracy by frequency bin (pooled over seeds; CI = Wilson on pooled rows / n_seeds)
    allp["bin"] = allp["train_freq"].map(bin_label)
    rows = []
    for _, _, lab in BINS:
        s = allp[allp["bin"] == lab]
        n = len(s) // len(seeds)
        k = int(s["correct"].sum() / len(seeds))
        lo, hi = wilson(k, n)
        rows.append({"bin": lab, "accuracy": k / n if n else 0.0, "n": n, "lo": lo, "hi": hi})
    acc = pd.DataFrame(rows)
    acc.to_csv(tables / "tab1_acc_by_freq.csv", index=False)
    fig_acc_by_freq(acc, figs / "fig2_acc_by_freq")

    # fig3: accuracy vs similarity, head vs tail (first seed is representative; all seeds in CSV)
    fig_acc_vs_sim(per_seed[0][["sim1", "correct", "is_tail"]], figs / "fig3_acc_vs_sim")

    # tab1: tail stats at k in {2,3,5}, mean ± CI across seeds
    out = []
    for k in [cfg.tail_k, *cfg.tail_sensitivity]:
        m_all = allp["train_freq"] <= k
        accs = [g.loc[g["train_freq"] <= k, "correct"].mean() for g in per_seed]
        haccs = [g.loc[g["train_freq"] > k, "correct"].mean() for g in per_seed]
        out.append(
            {
                "tail_k": k,
                "backbone": backbone,
                "window": f"{win.train_start}..{win.test_start}+",
                "tail_merchants": int(allp.loc[m_all, "merchant"].nunique()),
                "tail_merchant_share": round(
                    allp.loc[m_all, "merchant"].nunique() / allp["merchant"].nunique(), 4
                ),
                "tail_txn_share": round(float(m_all.mean()), 4),
                "tail_acc_mean": round(float(np.mean(accs)), 4),
                "tail_acc_std": round(float(np.std(accs)), 4),
                "head_acc_mean": round(float(np.mean(haccs)), 4),
                "head_acc_std": round(float(np.std(haccs)), 4),
                "n_seeds": len(seeds),
            }
        )
    pd.DataFrame(out).to_csv(tables / "tab1_tail_stats.csv", index=False)
    allp[
        ["txn_id", "merchant", "category", "pred", "sim1", "train_freq", "is_tail", "seed"]
    ].to_parquet(Path(cfg.results_dir) / "exp1_predictions.parquet", index=False)
    logger.info("\n" + pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
