"""Build the merchant index from a training frame and produce kNN predictions for a test frame."""

from __future__ import annotations

import pandas as pd

from txcat.embedder import Embedder
from txcat.index import HNSWIndex


def merchant_table(train: pd.DataFrame) -> pd.DataFrame:
    """One row per unique merchant: majority category, frequency, first seen. Ambiguous rows are
    excluded from label voting but still counted in frequency."""
    g = train.groupby("merchant")
    freq = g.size().rename("freq")
    first_seen = g["date"].min().rename("first_seen")
    lab_src = train[train["ambiguous"] == 0] if "ambiguous" in train else train
    label = (
        lab_src.groupby("merchant")["category"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("label")
    )
    tbl = pd.concat([freq, first_seen, label], axis=1).reset_index()
    tbl = tbl[tbl["label"].notna()].reset_index(drop=True)
    return tbl


def build_merchant_index(
    train: pd.DataFrame, emb: Embedder, M: int, ef_construction: int, seed: int
) -> HNSWIndex:
    tbl = merchant_table(train)
    vecs = emb.embed(tbl["merchant"].tolist())
    idx = HNSWIndex(
        dim=vecs.shape[1],
        M=M,
        ef_construction=ef_construction,
        seed=seed,
        max_elements=max(len(tbl) * 2, 1000),
    )
    idx.add(
        vecs,
        tbl["label"].tolist(),
        tbl["merchant"].tolist(),
        tbl["freq"].astype(int).tolist(),
        tbl["first_seen"],
    )
    return idx


def predict_knn(
    test: pd.DataFrame, idx: HNSWIndex, emb: Embedder, k: int, ef_search: int
) -> pd.DataFrame:
    """Top-1 kNN prediction plus similarity features per test row.

    Unique merchants are embedded and queried once.
    """
    uniq = test["merchant"].drop_duplicates().tolist()
    vecs = emb.embed(uniq)
    res = idx.query_batch(vecs, k=k, ef_search=ef_search)
    per_m = {m: r for m, r in zip(uniq, res, strict=True)}
    rows = []
    for m in test["merchant"]:
        r = per_m[m]
        s2 = r.sims[1] if len(r.sims) > 1 else 0.0
        rows.append(
            {
                "pred": r.labels[0],
                "sim1": r.sims[0],
                "sim2": s2,
                "margin": r.sims[0] - s2,
                "top1_freq": r.freqs[0],
                "top1_merchant": r.merchant_ids[0],
            }
        )
    return pd.DataFrame(rows, index=test.index)
