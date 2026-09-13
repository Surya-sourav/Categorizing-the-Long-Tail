"""Temporal stream: kNN -> gate -> (search + LLM) -> optional write-back, with windowed tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from txcat.data.taxonomy import CATEGORIES
from txcat.embedder import Embedder
from txcat.gate import ConfidenceGate
from txcat.index import HNSWIndex
from txcat.metrics import macro_f1
from txcat.writeback import WriteBackPolicy


@dataclass
class StreamLog:
    per_txn: pd.DataFrame
    windows: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_stream(
    transactions: pd.DataFrame,
    index: HNSWIndex,
    emb: Embedder,
    gate: ConfidenceGate,
    fallback,
    writeback: WriteBackPolicy,
    evidence_fn: Callable[[str], list | None],
    window: int = 500,
    k: int = 5,
    ef_search: int = 64,
    price_in_per_1m: float = 0.0,
    price_out_per_1m: float = 0.0,
    search_price: float = 0.0,
    tail_k: int = 3,
) -> StreamLog:
    """Process ``transactions`` (sorted by date) one at a time.

    ``fallback.categorize(merchant, evidence, taxonomy)`` must return an LLMResult-like object.
    ``evidence_fn(merchant)`` returns cached search results or None.
    """
    txns = transactions.sort_values("date", kind="stable").reset_index(drop=True)
    rows, cum_cost, n_wb, n_wb_wrong = [], 0.0, 0, 0
    written: dict[str, str] = {}
    for r in txns.itertuples(index=False):
        vec = emb.embed([r.merchant])[0]
        q = index.query(vec, k=k, ef_search=ef_search)
        fb = gate.should_fallback(q)
        pred, cost = q.labels[0], 0.0
        if fb:
            res = fallback.categorize(r.merchant, evidence_fn(r.merchant), CATEGORIES)
            pred = res.category if res.valid else q.labels[0]
            billed = (
                res.tokens_in + res.tokens_out > 0
            )  # an uncached miss (replay mode) costs nothing
            cost = (
                (
                    res.tokens_in / 1e6 * price_in_per_1m
                    + res.tokens_out / 1e6 * price_out_per_1m
                    + search_price
                )
                if billed
                else 0.0
            )
            if writeback.should_write(res) and r.merchant not in written:
                index.add(
                    vec.reshape(1, -1),
                    [res.category],
                    [r.merchant],
                    [1],
                    pd.to_datetime([r.date]),
                    source="writeback",
                )
                written[r.merchant] = res.category
                n_wb += 1
                n_wb_wrong += int(res.category != r.category)
        cum_cost += cost
        rows.append(
            {
                "txn_id": r.txn_id,
                "date": r.date,
                "merchant": r.merchant,
                "category": r.category,
                "pred": pred,
                "fallback": fb,
                "sim1": q.sims[0],
                "top1_freq": q.top1_freq,
                "cost": cost,
                "cum_cost": cum_cost,
                "index_size": len(index),
            }
        )
    per = pd.DataFrame(rows)
    per["correct"] = per["pred"] == per["category"]
    per["is_tail"] = (
        per["top1_freq"] <= tail_k
    )  # approximation for streaming: top-1 neighbour's frequency
    per["window"] = np.arange(len(per)) // window
    win = per.groupby("window").agg(
        start=("date", "min"),
        fallback_rate=("fallback", "mean"),
        acc=("correct", "mean"),
        cum_cost=("cum_cost", "last"),
        index_size=("index_size", "last"),
    )
    win["cum_macro_f1"] = [
        macro_f1(
            per.loc[: (i + 1) * window - 1, "category"], per.loc[: (i + 1) * window - 1, "pred"]
        )
        for i in win.index
    ]
    win["cum_macro_f1_tail"] = [
        macro_f1(
            per.loc[: (i + 1) * window - 1, "category"],
            per.loc[: (i + 1) * window - 1, "pred"],
            per.loc[: (i + 1) * window - 1, "is_tail"],
        )
        for i in win.index
    ]
    summary = {
        "n_txns": len(per),
        "fallback_rate": float(per["fallback"].mean()),
        "macro_f1": macro_f1(per["category"], per["pred"]),
        "total_cost": float(cum_cost),
        "n_writebacks": n_wb,
        "writeback_error_rate": (n_wb_wrong / n_wb) if n_wb else 0.0,
        "final_index_size": len(index),
    }
    return StreamLog(per_txn=per, windows=win.reset_index(), summary=summary)
