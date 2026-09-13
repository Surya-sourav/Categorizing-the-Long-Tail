"""Exp 3: replay the DC test window in temporal order under three write-back policies (cache replay, $0).
Merchants without a cached LLM result are counted as fallback-uncached and served by kNN.

Usage: python -m experiments.exp3_streaming_convergence --config configs/dc.yaml --seeds 42 43 44 [--max-txns 100000]
Outputs: fig5_fallback_decay, fig6_cumulative_cost, fig7_cumulative_f1 (+_tail), tab5_writeback_policies.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from analysis.plots import fig_stream_lines
from loguru import logger

from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import temporal_split
from txcat.embedder import Embedder
from txcat.gate import ConfidenceGate
from txcat.knn import build_merchant_index
from txcat.llm_fallback import CacheMissError, LLMFallback, LLMResult
from txcat.stream import run_stream
from txcat.utils import set_seed, setup_logging
from txcat.web_search import CacheMissError as SearchMiss
from txcat.web_search import WebSearchClient
from txcat.writeback import WriteBackPolicy

POLICIES = ["never", "always", "confidence_gated"]


class CachedOrUncached:
    """Wraps LLMFallback: on a cache miss return an invalid result (kNN keeps its label) and count it."""

    def __init__(self, fb: LLMFallback):
        self.fb, self.uncached = fb, 0

    def categorize(self, merchant, evidence, taxonomy):
        try:
            return self.fb.categorize(merchant, evidence, taxonomy)
        except CacheMissError:
            self.uncached += 1
            return LLMResult("INVALID", 0.0, "uncached", False, self.fb.m.name, 0, 0, 0.0, True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--max-txns", type=int, default=100000)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument(
        "--model", default=None, help="LLM model name for the fallback (default: first configured)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    setup_logging(cfg.logs_dir, "exp3")
    figs, tables = Path(cfg.results_dir, "figures"), Path(cfg.results_dir, "tables")
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    d = cfg.datasets["dc"]
    win = WindowCfg(
        **json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"]
    )
    dc = load_prepared(d.processed_path, cfg.taxonomy_dir)
    dc = dc[dc["category"].notna() & (dc["ambiguous"] == 0)]
    train, test = temporal_split(dc, win)
    test = test.sort_values("date", kind="stable").head(args.max_txns).reset_index(drop=True)
    mcfg = next(m for m in cfg.llm.models if args.model is None or m.name == args.model)
    ws = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, None, allow_live=False)

    def evidence(m):
        try:
            return ws.search(m, cfg.search.num_results)
        except SearchMiss:
            return None

    bb = cfg.embed.backbones[0]
    emb = Embedder(bb, cfg.embed.cache_dir, allow_live=False)
    windows_by_policy, summary_rows = {}, []
    for policy in POLICIES:
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
            fb = CachedOrUncached(
                LLMFallback(mcfg, cfg.llm.cache_dir, cfg.llm.prompt_with_web, allow_live=False)
            )
            log = run_stream(
                test,
                idx,
                emb,
                ConfidenceGate(cfg.gate.mode, cfg.gate.threshold),
                fb,
                WriteBackPolicy(policy, 0.8),
                evidence,
                window=args.window,
                k=cfg.index.k,
                ef_search=cfg.index.ef_search,
                price_in_per_1m=mcfg.price_in_per_1m,
                price_out_per_1m=mcfg.price_out_per_1m,
                search_price=cfg.search.price_per_1k_usd / 1000,
                tail_k=cfg.tail_k,
            )
            log.summary.update(
                {
                    "policy": policy,
                    "seed": seed,
                    "model": mcfg.name,
                    "fallback_uncached": fb.uncached,
                }
            )
            summary_rows.append(log.summary)
            per_seed.append(log.windows)
            logger.info(f"{policy} seed {seed}: {log.summary}")
        w = pd.concat(per_seed).groupby("window").mean(numeric_only=True).reset_index()
        w.attrs["window_size"] = args.window
        windows_by_policy[policy] = w
    fig_stream_lines(
        windows_by_policy,
        "fallback_rate",
        "fallback rate (per window)",
        figs / "fig5_fallback_decay",
        "Fallback rate over the stream",
    )
    fig_stream_lines(
        windows_by_policy,
        "cum_cost",
        "cumulative cost (USD)",
        figs / "fig6_cumulative_cost",
        "Cumulative fallback cost",
    )
    fig_stream_lines(
        windows_by_policy,
        "cum_macro_f1",
        "cumulative macro-F1",
        figs / "fig7_cumulative_f1",
        "Cumulative macro-F1 (overall)",
    )
    fig_stream_lines(
        windows_by_policy,
        "cum_macro_f1_tail",
        "cumulative macro-F1 (tail)",
        figs / "fig7_cumulative_f1_tail",
        "Cumulative macro-F1 (tail)",
    )
    s = pd.DataFrame(summary_rows)
    agg = (
        s.groupby("policy")
        .agg(
            fallback_rate=("fallback_rate", "mean"),
            macro_f1=("macro_f1", "mean"),
            total_cost=("total_cost", "mean"),
            n_writebacks=("n_writebacks", "mean"),
            writeback_error_rate=("writeback_error_rate", "mean"),
            final_index_size=("final_index_size", "mean"),
            fallback_uncached=("fallback_uncached", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    agg.to_csv(tables / "tab5_writeback_policies.csv", index=False)
    s.to_csv(Path(cfg.results_dir, "exp3_per_seed.csv"), index=False)
    logger.info("\n" + agg.to_string())


if __name__ == "__main__":
    main()
