"""Exp 2 cache pass (LIVE). Builds/loads the frozen FES lists, then for every FES merchant:
Brave search (cached), and each configured LLM in both conditions (cached). Safe to re-run; only misses
hit the network. Stops with BudgetExceeded before crossing the cap.

Usage: python -m experiments.exp2_cache_pass --config configs/dc.yaml [--stage fes|search|llm|all] [--models ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from txcat.budget import BudgetExceeded, SpendLedger
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import merchant_frequencies, temporal_split
from txcat.data.taxonomy import CATEGORIES
from txcat.fes import build_dc_fes, build_oklahoma_coldstart_fes, load_fes, save_fes
from txcat.llm_fallback import LLMFallback
from txcat.utils import RateLimiter, run_parallel, set_seed, setup_logging
from txcat.web_search import WebSearchClient


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--stage", choices=["fes", "search", "llm", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    setup_logging(cfg.logs_dir, "exp2_cache_pass")
    fes_dir = Path(cfg.results_dir, "fes")
    dc_path, ok_path = fes_dir / "dc_fes.parquet", fes_dir / "oklahoma_coldstart_fes.parquet"

    if args.stage in ("fes", "all"):
        d = cfg.datasets["dc"]
        win = WindowCfg(
            **json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"]
        )
        dc = load_prepared(d.processed_path, cfg.taxonomy_dir)
        dc = dc[dc["category"].notna()]
        train, test = temporal_split(dc, win)
        freqs = merchant_frequencies(train)
        if not dc_path.exists():
            save_fes(
                build_dc_fes(test, freqs, cfg.tail_k, cfg.fes.n_tail, cfg.fes.n_head, cfg.seed),
                dc_path,
            )
        if not ok_path.exists():
            ok = load_prepared(cfg.datasets["oklahoma"].processed_path, cfg.taxonomy_dir)
            ok = ok[ok["category"].notna()]
            save_fes(
                build_oklahoma_coldstart_fes(
                    ok, set(train["merchant"].unique()), cfg.fes.n_oklahoma, cfg.seed
                ),
                ok_path,
            )
        for p in (dc_path, ok_path):
            f = load_fes(p)
            logger.info(
                f"{p.name}: {f['merchant'].nunique()} merchants, {len(f)} rows, tail merchants {f.loc[f.is_tail, 'merchant'].nunique()}"
            )

    merchants = (
        pd.concat([load_fes(dc_path), load_fes(ok_path)])["merchant"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    ledger = SpendLedger(cfg.budget.ledger_path, cfg.budget.max_usd)
    logger.info(f"{len(merchants)} unique FES merchants; spend so far {ledger.total:.2f} USD")

    ws = WebSearchClient(
        cfg.search.provider,
        cfg.search.cache_dir,
        os.environ.get(cfg.search.api_key_env),
        ledger=ledger,
        min_interval_s=cfg.search.min_interval_s,
        price_per_1k=cfg.search.price_per_1k_usd,
    )
    if args.stage in ("search", "all"):
        try:
            search_limiter = RateLimiter(cfg.concurrency.search_rpm)

            def _search(m):
                search_limiter.acquire()
                return ws.search(m, cfg.search.num_results)

            run_parallel(_search, merchants, cfg.concurrency.search_workers, "search")
        except BudgetExceeded as e:
            logger.error(f"STOPPED: {e}")
            return
        logger.info(f"search stage complete; spend {ledger.total:.2f} USD")

    if args.stage in ("llm", "all"):
        ws_ro = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, None, allow_live=False)
        models = [m for m in cfg.llm.models if not args.models or m.name in args.models]
        limiters = {
            prov: RateLimiter(cfg.concurrency.llm_rpm if prov == "nvidia" else 400)
            for prov in {m.provider for m in models}
        }
        try:
            for mcfg in models:
                if not os.environ.get(mcfg.api_key_env):
                    logger.warning(f"skip {mcfg.name}: {mcfg.api_key_env} unset")
                    continue
                for cond, prompt in (
                    ("no_web", cfg.llm.prompt_no_web),
                    ("with_web", cfg.llm.prompt_with_web),
                ):
                    fb = LLMFallback(
                        mcfg,
                        cfg.llm.cache_dir,
                        prompt,
                        ledger=ledger,
                        max_completion_tokens=cfg.llm.max_completion_tokens,
                        limiter=limiters[mcfg.provider],
                    )

                    def _categorize(m, fb=fb, cond=cond):
                        ev = ws_ro.search(m, cfg.search.num_results) if cond == "with_web" else None
                        return fb.categorize(m, ev, CATEGORIES)

                    run_parallel(
                        _categorize, merchants, cfg.concurrency.llm_workers, f"{mcfg.name}/{cond}"
                    )
                    logger.info(f"{mcfg.name}/{cond} done; spend {ledger.total:.2f} USD")
        except BudgetExceeded as e:
            logger.error(f"STOPPED: {e}")
            return
    logger.info(
        f"cache pass complete. spend by kind: {ledger.by_kind()}; total {ledger.total:.2f} USD"
    )


if __name__ == "__main__":
    main()
