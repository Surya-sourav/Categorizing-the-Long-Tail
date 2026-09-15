"""Exp 0: dataset stats, taxonomy coverage, volume-rule window decision, label-noise audit sample.

Usage: python -m experiments.exp0_data_audit --config configs/dc.yaml [--seed 42]
Outputs: results/tables/tab0_dataset_stats.csv, results/tables/tab0_taxonomy_coverage.csv,
         results/window_decision.json, data/taxonomy/label_noise_audit_sample.csv (300 rows)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import merchant_frequencies, tail_mask, temporal_split, volume_rule_ok
from txcat.utils import now_iso, set_seed, setup_logging


def window_stats(df: pd.DataFrame, w: WindowCfg, k: int, ks_sens: list[int]) -> dict:
    tr, te = temporal_split(df, w)
    f = merchant_frequencies(tr)
    row = {
        "train_start": w.train_start,
        "train_end": w.train_end,
        "test_start": w.test_start,
        "n_train_txns": len(tr),
        "n_train_merchants": int(tr["merchant"].nunique()),
        "n_test_txns": len(te),
        "n_test_merchants": int(te["merchant"].nunique()),
    }
    for kk in [k, *ks_sens]:
        m = tail_mask(te, f, kk)
        row[f"tail_le{kk}_txn_share"] = round(float(m.mean()), 4)
        row[f"tail_le{kk}_merchants"] = int(te.loc[m, "merchant"].nunique())
    row["unseen_txn_share"] = round(float((te["merchant"].map(f).fillna(0) == 0).mean()), 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg.seed
    set_seed(seed)
    setup_logging(cfg.logs_dir, "exp0")
    tables = Path(cfg.results_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    stats_rows, coverage_rows = [], []
    for name in ["dc", "oklahoma"]:
        d = cfg.datasets[name]
        if not Path(d.processed_path).exists():
            logger.warning(f"{name}: processed file missing, skipping")
            continue
        df = load_prepared(d.processed_path, cfg.taxonomy_dir)
        n = len(df)
        coverage_rows.append(
            {
                "dataset": name,
                "n_rows": n,
                "rows_with_mcc": int(df["mcc"].notna().sum()),
                "rows_ambiguous": int(df["ambiguous"].sum()),
                "share_unlabeled": round(float(df["mcc"].isna().mean()), 4),
                "share_ambiguous": round(float(df["ambiguous"].mean()), 4),
                "n_categories_present": int(df["category"].dropna().nunique()),
            }
        )
        if name == "dc":
            primary = window_stats(df, d.window, cfg.tail_k, cfg.tail_sensitivity)
            tr, te = temporal_split(df, d.window)
            ok, vs = volume_rule_ok(
                te,
                merchant_frequencies(tr),
                cfg.tail_k,
                d.volume_rule.min_test_txns,
                d.volume_rule.min_tail_merchants,
            )
            decision = {
                "dataset": "dc",
                "checked_at": now_iso(),
                "primary_window": d.window.model_dump(),
                "primary_stats": vs,
                "primary_ok": ok,
            }
            chosen = d.window
            if not ok:
                fb = d.volume_rule.fallback_window
                tr2, te2 = temporal_split(df, fb)
                ok2, vs2 = volume_rule_ok(
                    te2,
                    merchant_frequencies(tr2),
                    cfg.tail_k,
                    d.volume_rule.min_test_txns,
                    d.volume_rule.min_tail_merchants,
                )
                decision.update(
                    {"fallback_window": fb.model_dump(), "fallback_stats": vs2, "fallback_ok": ok2}
                )
                chosen = fb
                logger.warning(f"primary window failed volume rule {vs}; sliding to {fb}")
            decision["chosen_window"] = chosen.model_dump()
            Path(cfg.results_dir, "window_decision.json").write_text(json.dumps(decision, indent=2))
            stats_rows.append(
                {"dataset": "dc", **window_stats(df, chosen, cfg.tail_k, cfg.tail_sensitivity)}
            )
            if chosen.model_dump() != d.window.model_dump():
                stats_rows.append({"dataset": "dc_primary_window_rejected", **primary})

            # label-noise audit sample: 300 unique merchants, non-ambiguous, seeded
            pool = df[(df["ambiguous"] == 0)].drop_duplicates("merchant")
            rng = np.random.default_rng(seed)
            pick = pool.iloc[
                np.sort(rng.choice(len(pool), size=min(300, len(pool)), replace=False))
            ]
            audit = pick[["raw_merchant", "merchant", "mcc_description", "mcc", "category"]].copy()
            audit_path = Path(cfg.taxonomy_dir) / "label_noise_audit_sample.csv"
            if audit_path.exists():
                prior = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
                if "judgement" in prior and (prior["judgement"] != "").any():
                    logger.info("label-noise audit already judged; keeping the committed file")
                    audit = None
            if audit is not None:
                audit["judgement"] = ""  # correct | wrong | unclear
                audit["note"] = ""
                audit.to_csv(audit_path, index=False)
                logger.info(f"label-noise audit sample: {len(audit)} rows written")
        else:
            stats_rows.append(
                {
                    "dataset": name,
                    "n_train_txns": 0,
                    "n_test_txns": n,
                    "n_test_merchants": int(df["merchant"].nunique()),
                    "train_start": "",
                    "train_end": "",
                    "test_start": str(df["date"].min().date()),
                }
            )

    pd.DataFrame(stats_rows).to_csv(tables / "tab0_dataset_stats.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(tables / "tab0_taxonomy_coverage.csv", index=False)
    logger.info("\n" + pd.DataFrame(stats_rows).T.to_string())
    logger.info("\n" + pd.DataFrame(coverage_rows).to_string())


if __name__ == "__main__":
    main()
