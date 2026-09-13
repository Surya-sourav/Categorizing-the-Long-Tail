"""Exp 4: failure taxonomy. Samples ~200 tail merchants the routed system still gets wrong (from the
exp2 caches), writes a labelling template, and, once labelled, aggregates the taxonomy table.

Usage:
  python -m experiments.exp4_failure_taxonomy --config configs/dc.yaml --stage sample   # writes template
  python -m experiments.exp4_failure_taxonomy --config configs/dc.yaml --stage table    # after labelling
Outputs: results/tables/failure_sample_template.csv, results/tables/tab6_failure_taxonomy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from txcat.config import load_config
from txcat.data.taxonomy import CATEGORIES
from txcat.fes import load_fes
from txcat.llm_fallback import LLMFallback
from txcat.utils import set_seed, setup_logging
from txcat.web_search import WebSearchClient

FAILURE_CLASSES = [
    "ambiguous_merchant",
    "no_web_presence",
    "web_search_failure",
    "llm_reasoning_error",
    "taxonomy_mismatch",
    "normalization_failure",
    "label_noise",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--stage", choices=["sample", "table"], default="sample")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    setup_logging(cfg.logs_dir, "exp4")
    tables = Path(cfg.results_dir, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    template = tables / "failure_sample_template.csv"

    if args.stage == "sample":
        mcfg = next(m for m in cfg.llm.models if args.model is None or m.name == args.model)
        fes = load_fes(Path(cfg.results_dir, "fes", "dc_fes.parquet"))
        tail = fes[(fes["ambiguous"] == 0) & fes["is_tail"]].drop_duplicates("merchant")
        ws = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, None, allow_live=False)
        fb = LLMFallback(mcfg, cfg.llm.cache_dir, cfg.llm.prompt_with_web, allow_live=False)
        rows = []
        for r in tail.itertuples(index=False):
            ev = ws.search(r.merchant, cfg.search.num_results)
            res = fb.categorize(r.merchant, ev, CATEGORIES)
            if res.category != r.category:
                rows.append(
                    {
                        "merchant": r.merchant,
                        "raw_merchant": r.raw_merchant,
                        "gold": r.category,
                        "pred": res.category,
                        "confidence": res.confidence,
                        "llm_evidence": res.evidence,
                        "n_search_results": len(ev),
                        "top_result": ev[0].title if ev else "",
                        "top_url": ev[0].url if ev else "",
                        "failure_class": "",
                        "note": "",
                    }
                )
        errs = pd.DataFrame(rows)
        rng = np.random.default_rng(cfg.seed)
        pick = errs.iloc[np.sort(rng.choice(len(errs), size=min(args.n, len(errs)), replace=False))]
        pick.to_csv(template, index=False)
        logger.info(
            f"{len(errs)} tail errors for {mcfg.name}; template with {len(pick)} rows -> {template}. "
            f"Allowed failure_class values: {FAILURE_CLASSES}"
        )
    else:
        lab = pd.read_csv(template)
        lab = lab[lab["failure_class"].notna() & (lab["failure_class"] != "")]
        bad = set(lab["failure_class"]) - set(FAILURE_CLASSES)
        if bad:
            raise SystemExit(f"unknown failure_class values: {bad}")
        agg = (
            lab.groupby("failure_class")
            .agg(
                count=("merchant", "size"),
                example=("merchant", "first"),
                example_gold=("gold", "first"),
                example_pred=("pred", "first"),
            )
            .reindex(FAILURE_CLASSES)
            .fillna({"count": 0})
            .reset_index()
        )
        agg["share"] = (agg["count"] / max(1, agg["count"].sum())).round(4)
        agg.to_csv(tables / "tab6_failure_taxonomy.csv", index=False)
        logger.info("\n" + agg.to_string())


if __name__ == "__main__":
    main()
