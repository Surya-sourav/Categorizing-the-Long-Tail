"""Pilot: 100 seeded tail merchants -> Brave search -> each LLM with and without web. Reports accuracy
per condition and spend, then freezes prompt hashes and pins the model snapshot ids.

Usage: python -m experiments.pilot --config configs/dc.yaml --n 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from txcat.budget import SpendLedger
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import merchant_frequencies, sample_fes, temporal_split
from txcat.data.taxonomy import CATEGORIES
from txcat.llm_fallback import LLMFallback
from txcat.utils import RateLimiter, run_parallel, set_seed, setup_logging, sha256_text
from txcat.web_search import WebSearchClient


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names to run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    setup_logging(cfg.logs_dir, "pilot")
    n = args.n or cfg.fes.pilot_n

    d = cfg.datasets["dc"]
    win = WindowCfg(
        **json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"]
    )
    df = load_prepared(d.processed_path, cfg.taxonomy_dir)
    df = df[df["category"].notna()]
    train, test = temporal_split(df, win)
    freqs = merchant_frequencies(train)
    fes = sample_fes(test, freqs, cfg.tail_k, n_tail=n, n_head=0, seed=cfg.seed)
    merchants = fes.drop_duplicates("merchant")[["merchant", "category"]].reset_index(drop=True)
    logger.info(f"pilot on {len(merchants)} tail merchants")

    ledger = SpendLedger(cfg.budget.ledger_path, cfg.budget.max_usd)
    ws = WebSearchClient(
        cfg.search.provider,
        cfg.search.cache_dir,
        os.environ.get(cfg.search.api_key_env),
        ledger=ledger,
        min_interval_s=cfg.search.min_interval_s,
        price_per_1k=cfg.search.price_per_1k_usd,
    )
    search_limiter = RateLimiter(cfg.concurrency.search_rpm)

    def _search(m):
        search_limiter.acquire()
        return m, ws.search(m, cfg.search.num_results)

    evidence = dict(
        run_parallel(_search, list(merchants["merchant"]), cfg.concurrency.search_workers, "search")
    )
    logger.info(
        f"search done; empty results for {sum(1 for v in evidence.values() if not v)} merchants; spend {ledger.total:.3f}"
    )

    rows, model_ids = [], {}
    # one requests-per-minute budget per provider: NVIDIA free tier is 40 RPM per key
    limiters = {
        prov: RateLimiter(cfg.concurrency.llm_rpm if prov == "nvidia" else 400)
        for prov in {m.provider for m in cfg.llm.models}
    }
    models = [m for m in cfg.llm.models if not args.models or m.name in args.models]
    for mcfg in models:
        if not os.environ.get(mcfg.api_key_env):
            logger.warning(f"skipping {mcfg.name}: {mcfg.api_key_env} not set")
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
                max_completion_tokens=mcfg.max_completion_tokens or cfg.llm.max_completion_tokens,
                limiter=limiters[mcfg.provider],
            )

            def _categorize(r, fb=fb, cond=cond, mcfg=mcfg):
                ev = evidence[r["merchant"]] if cond == "with_web" else None
                res = fb.categorize(r["merchant"], ev, CATEGORIES)
                return {
                    "model": mcfg.name,
                    "condition": cond,
                    "merchant": r["merchant"],
                    "gold": r["category"],
                    "pred": res.category,
                    "correct": res.category == r["category"],
                    "valid": res.valid,
                    "confidence": res.confidence,
                    "latency_ms": res.latency_ms,
                    "tokens_in": res.tokens_in,
                    "tokens_out": res.tokens_out,
                    "model_id": res.model_id,
                }

            out = run_parallel(
                _categorize,
                [r for _, r in merchants.iterrows()],
                cfg.concurrency.nvidia_workers
                if mcfg.provider == "nvidia"
                else cfg.concurrency.llm_workers,
                f"{mcfg.name}/{cond}",
                on_error="collect",
            )
            failed = [o for o in out if isinstance(o, Exception)]
            ok = [o for o in out if not isinstance(o, Exception)]
            if failed:
                logger.warning(
                    f"{mcfg.name}/{cond}: {len(failed)} merchants failed after retries "
                    f"(left uncached; re-run to retry). First: {failed[0]}"
                )
            rows.extend(ok)
            if ok:
                model_ids[mcfg.name] = ok[-1]["model_id"]
            logger.info(f"{mcfg.name} {cond} done; spend so far {ledger.total:.3f} USD")

    out = pd.DataFrame(rows)
    Path(cfg.results_dir, "pilot").mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(cfg.results_dir, "pilot", "pilot_results.csv"), index=False)
    summary = out.groupby(["model", "condition"]).agg(
        acc=("correct", "mean"),
        invalid=("valid", lambda s: 1 - s.mean()),
        p50_ms=("latency_ms", "median"),
        n=("correct", "size"),
    )
    logger.info("\n" + summary.to_string())
    logger.info(f"spend by kind: {ledger.by_kind()}  total {ledger.total:.3f} USD")

    Path("prompts/PROMPT_HASHES.json").write_text(
        json.dumps(
            {
                p: sha256_text(Path(p).read_text())
                for p in (cfg.llm.prompt_no_web, cfg.llm.prompt_with_web)
            },
            indent=2,
        )
    )
    versions = {
        "embedding_models": {b: b for b in cfg.embed.backbones}
        | {"openai": cfg.embed.openai_model},
        "llm_models": model_ids,
        "web_search": {
            "provider": cfg.search.provider,
            "pilot_date": pd.Timestamp.now("UTC").date().isoformat(),
        },
    }
    Path("prompts/model_versions.json").write_text(json.dumps(versions, indent=2))
    logger.info("prompts frozen (PROMPT_HASHES.json) and model ids pinned (model_versions.json)")


if __name__ == "__main__":
    main()
