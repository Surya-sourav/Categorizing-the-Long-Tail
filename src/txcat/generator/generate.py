"""Zipf-distributed synthetic transactions over the NSI vocabulary, in the processed schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from txcat.data.schema import PROCESSED_COLUMNS
from txcat.generator.noise import noisy_descriptor

CITIES = [
    ("Washington", "DC"),
    ("Oklahoma City", "OK"),
    ("Tulsa", "OK"),
    ("Seattle", "WA"),
    ("Austin", "TX"),
    ("Denver", "CO"),
    ("Chicago", "IL"),
    ("Atlanta", "GA"),
    ("Phoenix", "AZ"),
    ("Boston", "MA"),
]


def generate(
    vocab: pd.DataFrame,
    n_transactions: int,
    alpha: float,
    seed: int,
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    novel_share: float = 0.0,
    novel_after: str = "2024-01-01",
) -> pd.DataFrame:
    """Zipf-distributed transactions over ``vocab``.

    ``novel_share`` is the fraction of brands that only start transacting on/after ``novel_after``
    (the tail-severity dial: these brands are unseen by any index trained before that date).
    """
    rng = np.random.default_rng(seed)
    v = vocab.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # random rank assignment
    ranks = np.arange(1, len(v) + 1)
    p = ranks ** (-alpha)
    p /= p.sum()
    idx = rng.choice(len(v), size=n_transactions, p=p)
    novel = np.zeros(len(v), dtype=bool)
    if novel_share > 0:
        novel[rng.choice(len(v), size=int(round(novel_share * len(v))), replace=False)] = True
    t0, t1, tn = pd.to_datetime(start), pd.to_datetime(end), pd.to_datetime(novel_after)
    span_all = (t1 - t0).days + 1
    span_novel = (t1 - tn).days + 1
    offsets = rng.integers(0, span_all, size=n_transactions)
    is_novel_txn = novel[idx]
    offsets[is_novel_txn] = (tn - t0).days + rng.integers(
        0, span_novel, size=int(is_novel_txn.sum())
    )
    order = np.argsort(offsets, kind="stable")
    idx, offsets = idx[order], offsets[order]
    dates = t0 + pd.to_timedelta(offsets, unit="D")
    rows = []
    for i, k in enumerate(idx):
        city, state = CITIES[int(rng.integers(0, len(CITIES)))]
        rows.append(
            {
                "txn_id": f"gen:{seed}:{i}",
                "date": dates[i],
                "raw_merchant": noisy_descriptor(v.loc[k, "name"], city, state, rng),
                "mcc_description": "",
                "source": "gen",
                "category": v.loc[k, "category"],
                "merchant_id": v.loc[k, "merchant_id"],
                "alpha": alpha,
                "novel_brand": bool(novel[k]),
            }
        )
    return pd.DataFrame(
        rows, columns=PROCESSED_COLUMNS + ["category", "merchant_id", "alpha", "novel_brand"]
    )


def generate_to_parquet(
    vocab_path: str | Path, out_path: str | Path, n_transactions: int, alpha: float, seed: int
) -> pd.DataFrame:
    df = generate(pd.read_parquet(vocab_path), n_transactions, alpha, seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
