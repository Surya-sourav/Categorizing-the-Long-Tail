"""Exp 2 (offline). Every LLM/search number comes from cache; kNN is recomputed per seed from the embedding cache.

Usage: python -m experiments.exp2_fallback_comparison --config configs/dc.yaml --seeds 42 43 44 [--mode reproduce]
Outputs: tab2_main_results.csv, tab2b_oklahoma_coldstart.csv, tab3_critical_ablation.csv, tab3b_prompt_sensitivity.csv,
         tab3c_agentic_search.csv, fig4_acc_cost_frontier.{pdf,png}, exp2_frontier.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from analysis.plots import fig_frontier
from dotenv import load_dotenv
from loguru import logger

from txcat.agentic_search import AgenticSearchCategorizer
from txcat.baselines import EmbeddingLRBaseline
from txcat.budget import SpendLedger
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import temporal_split
from txcat.data.taxonomy import CATEGORIES
from txcat.embedder import Embedder
from txcat.fes import load_fes
from txcat.knn import build_merchant_index, predict_knn
from txcat.llm_fallback import LLMFallback
from txcat.metrics import (
    bootstrap_ci,
    cost_per_1k,
    latency_p50_p95,
    macro_f1,
    paired_bootstrap_diff,
)
from txcat.utils import set_seed, setup_logging
from txcat.web_search import WebSearchClient

THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.951, 0.05)]


def f1_row(df: pd.DataFrame, pred_col: str) -> dict:
    return {
        "f1_overall": macro_f1(df["category"], df[pred_col]),
        "f1_head": macro_f1(df["category"], df[pred_col], ~df["is_tail"]),
        "f1_tail": macro_f1(df["category"], df[pred_col], df["is_tail"]),
    }


def llm_preds(fes_merchants: list[str], mcfg, prompt: str, cfg, ws_ro) -> pd.DataFrame:
    fb = LLMFallback(mcfg, cfg.llm.cache_dir, prompt, allow_live=False)
    rows = []
    for m in fes_merchants:
        ev = ws_ro.search(m, cfg.search.num_results) if "with_web" in Path(prompt).stem else None
        r = fb.categorize(m, ev, CATEGORIES)
        rows.append(
            {
                "merchant": m,
                "pred": r.category,
                "conf": r.confidence,
                "lat": r.latency_ms,
                "tin": r.tokens_in,
                "tout": r.tokens_out,
            }
        )
    return pd.DataFrame(rows).set_index("merchant")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--mode", choices=["live", "reproduce"], default="reproduce")
    ap.add_argument("--agentic-n", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    setup_logging(cfg.logs_dir, "exp2")
    figs, tables = Path(cfg.results_dir, "figures"), Path(cfg.results_dir, "tables")
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    live = args.mode == "live"

    d = cfg.datasets["dc"]
    win = WindowCfg(
        **json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"]
    )
    dc = load_prepared(d.processed_path, cfg.taxonomy_dir)
    dc = dc[dc["category"].notna()]
    train, _ = temporal_split(dc, win)
    fes = load_fes(Path(cfg.results_dir, "fes", "dc_fes.parquet"))
    fes = fes[fes["ambiguous"] == 0].reset_index(drop=True)
    ok_fes = load_fes(Path(cfg.results_dir, "fes", "oklahoma_coldstart_fes.parquet"))
    ok_fes = ok_fes[ok_fes["ambiguous"] == 0].reset_index(drop=True)
    merchants = fes["merchant"].drop_duplicates().tolist()
    ws_ro = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, None, allow_live=False)
    n_txn = len(fes)
    backbones = [*cfg.embed.backbones, cfg.embed.openai_model]

    # ---- LLM rows (cached) ----
    llm = {}  # (model, cond) -> DataFrame indexed by merchant
    for mcfg in cfg.llm.models:
        for cond, prompt in (
            ("no_web", cfg.llm.prompt_no_web),
            ("with_web", cfg.llm.prompt_with_web),
        ):
            try:
                llm[(mcfg.name, cond)] = llm_preds(merchants, mcfg, prompt, cfg, ws_ro)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"skipping {mcfg.name}/{cond}: {e}")

    rows, frontier, knn_cache = [], [], {}
    for bb in backbones:
        emb = Embedder(bb, cfg.embed.cache_dir, allow_live=live, batch_size=cfg.embed.batch_size)
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
            t0 = time.perf_counter()
            p = predict_knn(fes, idx, emb, cfg.index.k, cfg.index.ef_search)
            lat_ms = (time.perf_counter() - t0) * 1000 / max(1, len(merchants))
            per_seed.append(
                (seed, fes.assign(pred=p["pred"].values, sim1=p["sim1"].values), lat_ms)
            )
        knn_cache[bb] = per_seed
        f1s = [f1_row(t, "pred") for _, t, _ in per_seed]
        rows.append(
            {
                "method": f"knn/{bb}",
                **{k: bootstrap_ci([f[k] for f in f1s])[1] for k in f1s[0]},
                **{k + "_lo": bootstrap_ci([f[k] for f in f1s])[0] for k in f1s[0]},
                **{k + "_hi": bootstrap_ci([f[k] for f in f1s])[2] for k in f1s[0]},
                "p50_ms": np.mean([lat_ms for _, _, lat_ms in per_seed]),
                "p95_ms": np.mean([lat_ms for _, _, lat_ms in per_seed]),
                "cost_per_1k": 0.0,
                "fallback_rate": 0.0,
                "n_seeds": len(seeds),
            }
        )
        if bb == cfg.embed.backbones[0]:
            lr = EmbeddingLRBaseline(emb, seed=seeds[0]).fit(
                train["merchant"].tolist(), train["category"].tolist()
            )
            t = fes.assign(pred=lr.predict(fes["merchant"].tolist()))
            rows.append(
                {
                    "method": f"emb_lr/{bb}",
                    **f1_row(t, "pred"),
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "cost_per_1k": 0.0,
                    "fallback_rate": 0.0,
                    "n_seeds": 1,
                }
            )

    for (model, cond), lp in llm.items():
        mcfg = next(m for m in cfg.llm.models if m.name == model)
        t = fes.assign(pred=fes["merchant"].map(lp["pred"]).values)
        p50, p95 = latency_p50_p95(lp["lat"])
        cost = cost_per_1k(
            n_txn,
            int(lp["tin"].sum()),
            int(lp["tout"].sum()),
            len(lp) if cond == "with_web" else 0,
            mcfg.price_in_per_1m,
            mcfg.price_out_per_1m,
            cfg.search.price_per_1k_usd,
        )
        rows.append(
            {
                "method": f"llm_{cond}/{model}",
                **f1_row(t, "pred"),
                "p50_ms": p50,
                "p95_ms": p95,
                "cost_per_1k": cost,
                "fallback_rate": 1.0,
                "n_seeds": 1,
            }
        )

    # ---- routed system: kNN + gate + LLM-with-web; sweep thresholds ----
    bb0 = cfg.embed.backbones[0]
    for model in {m for m, c in llm if c == "with_web"}:
        lp = llm[(model, "with_web")]
        mcfg = next(m for m in cfg.llm.models if m.name == model)
        for t_thr in THRESHOLDS:
            f1s, costs, fr = [], [], []
            for _seed, t, _lat in knn_cache[bb0]:
                fb_mask = t["sim1"] < t_thr
                routed = t["pred"].where(~fb_mask, t["merchant"].map(lp["pred"]))
                tt = t.assign(routed=routed)
                f1s.append(f1_row(tt, "routed"))
                fb_m = tt.loc[fb_mask, "merchant"].drop_duplicates()
                sub = lp.loc[fb_m]
                costs.append(
                    cost_per_1k(
                        n_txn,
                        int(sub["tin"].sum()),
                        int(sub["tout"].sum()),
                        len(sub),
                        mcfg.price_in_per_1m,
                        mcfg.price_out_per_1m,
                        cfg.search.price_per_1k_usd,
                    )
                )
                fr.append(float(fb_mask.mean()))
            rec = {
                "model": model,
                "threshold": t_thr,
                "macro_f1": np.mean([f["f1_overall"] for f in f1s]),
                "f1_tail": np.mean([f["f1_tail"] for f in f1s]),
                "f1_head": np.mean([f["f1_head"] for f in f1s]),
                "cost_per_1k": np.mean(costs),
                "fallback_rate": np.mean(fr),
            }
            frontier.append(rec)
            if t_thr == round(cfg.gate.threshold, 2):
                rows.append(
                    {
                        "method": f"routed/{bb0}/{model}@{t_thr}",
                        "f1_overall": rec["macro_f1"],
                        "f1_head": rec["f1_head"],
                        "f1_tail": rec["f1_tail"],
                        "p50_ms": np.nan,
                        "p95_ms": np.nan,
                        "cost_per_1k": rec["cost_per_1k"],
                        "fallback_rate": rec["fallback_rate"],
                        "n_seeds": len(seeds),
                    }
                )
    fr_df = pd.DataFrame(frontier)
    fr_df.to_csv(Path(cfg.results_dir, "exp2_frontier.csv"), index=False)
    # knee: largest second difference of F1 w.r.t. cost, per first model
    knee = None
    if not fr_df.empty:
        g = fr_df[fr_df["model"] == fr_df["model"].iloc[0]].sort_values("cost_per_1k")
        if len(g) > 4:
            d1 = np.gradient(g["macro_f1"].values, g["cost_per_1k"].values + 1e-9)
            knee = float(g["threshold"].values[int(np.argmax(-np.gradient(d1)))])
        fig_frontier(fr_df, figs / "fig4_acc_cost_frontier", knee_threshold=knee)
    pd.DataFrame(rows).to_csv(tables / "tab2_main_results.csv", index=False)

    # ---- critical ablation: tail-only, with-web vs no-web, paired bootstrap over merchants ----
    tail_m = (
        fes.loc[fes["is_tail"]]
        .drop_duplicates("merchant")[["merchant", "category"]]
        .set_index("merchant")["category"]
    )
    abl = []
    for model in {m for m, _ in llm}:
        if (model, "with_web") not in llm or (model, "no_web") not in llm:
            continue
        a = (llm[(model, "with_web")].loc[tail_m.index, "pred"] == tail_m).values
        b = (llm[(model, "no_web")].loc[tail_m.index, "pred"] == tail_m).values
        diff, lo, hi = paired_bootstrap_diff(a, b, seed=seeds[0])
        abl.append(
            {
                "model": model,
                "n_tail_merchants": len(tail_m),
                "acc_with_web": a.mean(),
                "acc_no_web": b.mean(),
                "diff": diff,
                "ci_lo": lo,
                "ci_hi": hi,
                "significant_95": bool(lo > 0 or hi < 0),
            }
        )
    pd.DataFrame(abl).to_csv(tables / "tab3_critical_ablation.csv", index=False)

    # ---- Oklahoma cold-start rows: kNN (DC index) vs LLM with web ----
    okrows = []
    ok_m = ok_fes["merchant"].drop_duplicates().tolist()
    emb0 = Embedder(bb0, cfg.embed.cache_dir, allow_live=live)
    set_seed(seeds[0])
    idx0 = build_merchant_index(train, emb0, cfg.index.M, cfg.index.ef_construction, seeds[0])
    p = predict_knn(ok_fes, idx0, emb0, cfg.index.k, cfg.index.ef_search)
    t = ok_fes.assign(pred=p["pred"].values)
    okrows.append(
        {
            "method": f"knn/{bb0}",
            "f1_overall": macro_f1(t["category"], t["pred"]),
            "n_merchants": len(ok_m),
        }
    )
    for (model, cond), _ in llm.items():
        try:
            mcfg = next(m for m in cfg.llm.models if m.name == model)
            lp = llm_preds(
                ok_m,
                mcfg,
                cfg.llm.prompt_with_web if cond == "with_web" else cfg.llm.prompt_no_web,
                cfg,
                ws_ro,
            )
            tt = ok_fes.assign(pred=ok_fes["merchant"].map(lp["pred"]).values)
            okrows.append(
                {
                    "method": f"llm_{cond}/{model}",
                    "f1_overall": macro_f1(tt["category"], tt["pred"]),
                    "n_merchants": len(ok_m),
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"oklahoma {model}/{cond} skipped: {e}")
    pd.DataFrame(okrows).to_csv(tables / "tab2b_oklahoma_coldstart.csv", index=False)

    # ---- prompt sensitivity: 2 variants x 300 tail merchants x each model (cached in live mode on first run) ----
    ps_m = tail_m.index[:300].tolist()
    ledger = SpendLedger(cfg.budget.ledger_path, cfg.budget.max_usd)
    ps = []
    for mcfg in cfg.llm.models:
        if live and not os.environ.get(mcfg.api_key_env):
            continue
        for cond, v1, v2 in (
            ("no_web", cfg.llm.prompt_no_web, "prompts/fallback_no_web_v2.txt"),
            ("with_web", cfg.llm.prompt_with_web, "prompts/fallback_with_web_v2.txt"),
        ):
            try:
                accs = []
                for prompt in (v1, v2):
                    fb = LLMFallback(
                        mcfg,
                        cfg.llm.cache_dir,
                        prompt,
                        ledger=ledger if live else None,
                        allow_live=live,
                    )
                    ok_n = 0
                    for m in ps_m:
                        ev = ws_ro.search(m, cfg.search.num_results) if cond == "with_web" else None
                        ok_n += fb.categorize(m, ev, CATEGORIES).category == tail_m[m]
                    accs.append(ok_n / len(ps_m))
                ps.append(
                    {
                        "model": mcfg.name,
                        "condition": cond,
                        "acc_v1": accs[0],
                        "acc_v2": accs[1],
                        "abs_diff": abs(accs[0] - accs[1]),
                        "n": len(ps_m),
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"prompt sensitivity {mcfg.name}/{cond} skipped: {e}")
    pd.DataFrame(ps).to_csv(tables / "tab3b_prompt_sensitivity.csv", index=False)

    # ---- agentic search ablation: OpenAI built-in web_search on 300 tail merchants ----
    ag_model = next(
        (m for m in cfg.llm.models if m.provider == "openai" and "mini" in m.name), None
    )
    if ag_model is not None:
        ag = AgenticSearchCategorizer(
            ag_model.name,
            Path(cfg.llm.cache_dir) / "agentic",
            ledger=ledger if live else None,
            price_in_per_1m=ag_model.price_in_per_1m,
            price_out_per_1m=ag_model.price_out_per_1m,
            allow_live=live,
        )
        try:
            res = {m: ag.categorize(m, CATEGORIES) for m in ps_m[: args.agentic_n]}
            acc_ag = np.mean([res[m]["category"] == tail_m[m] for m in res])
            acc_fixed = np.mean(
                [llm[(ag_model.name, "with_web")].loc[m, "pred"] == tail_m[m] for m in res]
            )
            acc_none = np.mean(
                [llm[(ag_model.name, "no_web")].loc[m, "pred"] == tail_m[m] for m in res]
            )
            pd.DataFrame(
                [
                    {
                        "model": ag_model.name,
                        "n": len(res),
                        "acc_agentic_search": acc_ag,
                        "acc_fixed_snippets": acc_fixed,
                        "acc_no_web": acc_none,
                        "mean_searches_per_merchant": np.mean(
                            [r["n_searches"] for r in res.values()]
                        ),
                        "usd_per_1k_merchants": 1000 * np.mean([r["usd"] for r in res.values()]),
                    }
                ]
            ).to_csv(tables / "tab3c_agentic_search.csv", index=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"agentic ablation skipped: {e}")
    logger.info("\n" + pd.DataFrame(rows).to_string())
    logger.info("\n" + pd.DataFrame(abl).to_string())


if __name__ == "__main__":
    main()
