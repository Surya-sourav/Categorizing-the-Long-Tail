"""Exp 1 (generator panel): kNN tail/head accuracy as a function of the Zipf exponent alpha. $0.

Usage: python -m experiments.exp1_generator_sweep --config configs/dc.yaml --alphas 0.8 1.0 1.2 1.5 --seeds 42 43 44 45 46
Outputs: results/tables/tab1b_alpha_sweep.csv, results/figures/fig1b_alpha_sweep.{pdf,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from analysis import style
from loguru import logger

from txcat.config import WindowCfg, load_config
from txcat.data.splits import freq_of, merchant_frequencies, temporal_split
from txcat.embedder import Embedder
from txcat.generator.generate import generate
from txcat.knn import build_merchant_index, predict_knn
from txcat.normalizer import normalize_merchant
from txcat.utils import set_seed, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-txns", type=int, default=60000)
    ap.add_argument("--novel-shares", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--fixed-alpha", type=float, default=1.1)
    ap.add_argument("--mode", choices=["live", "reproduce"], default="live")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    setup_logging(cfg.logs_dir, "exp1_gen")
    vocab = pd.read_parquet("data/processed/gen_vocab.parquet")
    bb = cfg.embed.backbones[0]
    # local backbones may embed in any mode (CPU compute, not an API call)
    emb = Embedder(
        bb,
        cfg.embed.cache_dir,
        allow_live=(args.mode == "live") or not bb.startswith("text-embedding"),
    )
    win = WindowCfg(
        train_start="2019-01-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2025-12-31",
    )

    def run_one(alpha: float, novel_share: float, seed: int) -> dict:
        set_seed(seed)
        g = generate(
            vocab, args.n_txns, alpha, seed, novel_share=novel_share, novel_after=win.test_start
        )
        g["merchant"] = g["raw_merchant"].map(lambda r: normalize_merchant(r).text)
        g["ambiguous"] = 0
        train, test = temporal_split(g, win)
        freqs = merchant_frequencies(train)
        test = test.assign(train_freq=freq_of(test, freqs))
        idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
        p = predict_knn(test, idx, emb, cfg.index.k, cfg.index.ef_search)
        correct = p["pred"].values == test["category"].values
        tail = test["train_freq"].values <= cfg.tail_k
        unseen_entity = ~test["merchant_id"].isin(set(train["merchant_id"])).values
        return {
            "alpha": alpha,
            "novel_share": novel_share,
            "seed": seed,
            "tail_txn_share": tail.mean(),
            "acc_tail": correct[tail].mean(),
            "acc_head": correct[~tail].mean(),
            "acc_overall": correct.mean(),
            "unseen_entity_txn_share": unseen_entity.mean(),
            "acc_unseen_entity": correct[unseen_entity].mean()
            if unseen_entity.any()
            else float("nan"),
            "acc_seen_entity": correct[~unseen_entity].mean(),
            "n_train_merchants": len(idx),
        }

    # panel B: novel-brand share at fixed alpha (the tail-severity dial)
    nrows = [run_one(args.fixed_alpha, ns, seed) for ns in args.novel_shares for seed in seeds]
    ndf = pd.DataFrame(nrows)
    nagg = ndf.groupby("novel_share").agg(["mean", "std"]).reset_index()
    nagg.columns = ["_".join(c).strip("_") for c in nagg.columns]
    Path(cfg.results_dir, "tables").mkdir(parents=True, exist_ok=True)
    nagg.to_csv(Path(cfg.results_dir, "tables", "tab1c_novel_share_sweep.csv"), index=False)
    style.apply()
    fig, ax = plt.subplots()
    ax.errorbar(
        nagg["novel_share"],
        nagg["acc_overall_mean"],
        yerr=nagg["acc_overall_std"],
        marker="o",
        color=style.SERIES["head"],
        label="overall",
    )
    ax.errorbar(
        nagg["novel_share"],
        nagg["acc_tail_mean"],
        yerr=nagg["acc_tail_std"],
        marker="o",
        color=style.SERIES["tail"],
        label="tail (freq ≤ 3)",
    )
    ax.errorbar(
        nagg["novel_share"],
        nagg["acc_unseen_entity_mean"],
        yerr=nagg["acc_unseen_entity_std"],
        marker="s",
        color=style.SERIES["third"],
        label="brand never seen in training",
    )
    ax.set_xlabel("share of brands that first appear in the test period")
    ax.set_ylabel("kNN top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Tail severity sweep: novel brands (α = {args.fixed_alpha})", loc="left", color=style.INK
    )
    ax.legend(loc="lower left")
    style.save(fig, Path(cfg.results_dir, "figures", "fig1c_novel_share_sweep"))
    plt.close(fig)
    logger.info(
        "\n"
        + nagg[
            [
                "novel_share",
                "unseen_entity_txn_share_mean",
                "acc_unseen_entity_mean",
                "acc_tail_mean",
                "acc_overall_mean",
            ]
        ]
        .round(3)
        .to_string()
    )

    rows = []
    for alpha in args.alphas:
        for seed in seeds:
            set_seed(seed)
            g = generate(vocab, args.n_txns, alpha, seed)
            g["merchant"] = g["raw_merchant"].map(lambda r: normalize_merchant(r).text)
            g["ambiguous"] = 0
            train, test = temporal_split(g, win)
            freqs = merchant_frequencies(train)
            test = test.assign(train_freq=freq_of(test, freqs))
            idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
            p = predict_knn(test, idx, emb, cfg.index.k, cfg.index.ef_search)
            correct = p["pred"].values == test["category"].values
            tail = test["train_freq"].values <= cfg.tail_k
            unseen_entity = ~test["merchant_id"].isin(set(train["merchant_id"])).values
            rows.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "tail_txn_share": tail.mean(),
                    "acc_tail": correct[tail].mean(),
                    "acc_head": correct[~tail].mean(),
                    "acc_overall": correct.mean(),
                    "unseen_entity_txn_share": unseen_entity.mean(),
                    "acc_unseen_entity": correct[unseen_entity].mean()
                    if unseen_entity.any()
                    else float("nan"),
                    "acc_seen_entity": correct[~unseen_entity].mean(),
                    "n_train_merchants": len(idx),
                }
            )
            logger.info(rows[-1])
    df = pd.DataFrame(rows)
    agg = df.groupby("alpha").agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    Path(cfg.results_dir, "tables").mkdir(parents=True, exist_ok=True)
    agg.to_csv(Path(cfg.results_dir, "tables", "tab1b_alpha_sweep.csv"), index=False)
    style.apply()
    fig, ax = plt.subplots()
    ax.errorbar(
        agg["alpha"],
        agg["acc_head_mean"],
        yerr=agg["acc_head_std"],
        marker="o",
        color=style.SERIES["head"],
        label="head",
    )
    ax.errorbar(
        agg["alpha"],
        agg["acc_tail_mean"],
        yerr=agg["acc_tail_std"],
        marker="o",
        color=style.SERIES["tail"],
        label="tail (freq ≤ 3)",
    )
    ax.errorbar(
        agg["alpha"],
        agg["acc_unseen_entity_mean"],
        yerr=agg["acc_unseen_entity_std"],
        marker="s",
        color=style.SERIES["third"],
        label="brand never seen in training",
    )
    ax.set_xlabel("Zipf exponent α (larger = heavier head, thinner tail)")
    ax.set_ylabel("kNN top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Tail severity sweep on the semi-synthetic benchmark", loc="left", color=style.INK)
    ax.legend(loc="lower right")
    style.save(fig, Path(cfg.results_dir, "figures", "fig1b_alpha_sweep"))
    plt.close(fig)


if __name__ == "__main__":
    main()
