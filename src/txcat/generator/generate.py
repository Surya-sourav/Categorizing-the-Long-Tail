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
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = vocab.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # random rank assignment
    ranks = np.arange(1, len(v) + 1)
    p = ranks ** (-alpha)
    p /= p.sum()
    idx = rng.choice(len(v), size=n_transactions, p=p)
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.sort(
            rng.integers(
                0, (pd.to_datetime(end) - pd.to_datetime(start)).days + 1, size=n_transactions
            )
        ),
        unit="D",
    )
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
            }
        )
    return pd.DataFrame(rows, columns=PROCESSED_COLUMNS + ["category", "merchant_id", "alpha"])


def generate_to_parquet(
    vocab_path: str | Path, out_path: str | Path, n_transactions: int, alpha: float, seed: int
) -> pd.DataFrame:
    df = generate(pd.read_parquet(vocab_path), n_transactions, alpha, seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
