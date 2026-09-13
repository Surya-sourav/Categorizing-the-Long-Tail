"""Temporal splits and merchant-level sampling. No row-level randomization anywhere."""

from __future__ import annotations

import numpy as np
import pandas as pd

from txcat.config import WindowCfg


def temporal_split(df: pd.DataFrame, w: WindowCfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on ``date`` only.

    train: [train_start, train_end]; test: [test_start, test_end or max date].
    """
    d = df["date"]
    train = df[(d >= pd.Timestamp(w.train_start)) & (d <= pd.Timestamp(w.train_end))]
    hi = pd.Timestamp(w.test_end) if w.test_end else d.max()
    test = df[(d >= pd.Timestamp(w.test_start)) & (d <= hi)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def merchant_frequencies(train: pd.DataFrame) -> pd.Series:
    """Transactions per normalized merchant in the training window."""
    return train.groupby("merchant").size()


def freq_of(test: pd.DataFrame, freqs: pd.Series) -> pd.Series:
    """Training frequency of each test row's merchant (0 if unseen)."""
    return test["merchant"].map(freqs).fillna(0).astype(int)


def tail_mask(test: pd.DataFrame, freqs: pd.Series, k: int) -> pd.Series:
    """True where the row's merchant has training frequency <= k (unseen merchants are tail)."""
    return freq_of(test, freqs) <= k


def volume_rule_ok(
    test: pd.DataFrame, freqs: pd.Series, k: int, min_test_txns: int, min_tail_merchants: int
) -> tuple[bool, dict]:
    """Spec section 3 conditional volume rule."""
    is_tail = tail_mask(test, freqs, k)
    stats = {
        "n_test_txns": int(len(test)),
        "n_test_merchants": int(test["merchant"].nunique()),
        "n_tail_merchants": int(test.loc[is_tail, "merchant"].nunique()),
        "tail_txn_share": float(is_tail.mean()) if len(test) else 0.0,
    }
    ok = stats["n_test_txns"] >= min_test_txns and stats["n_tail_merchants"] >= min_tail_merchants
    return ok, stats


def sample_fes(
    test: pd.DataFrame,
    freqs: pd.Series,
    k: int,
    n_tail: int,
    n_head: int,
    seed: int,
    exclude_ambiguous: bool = True,
) -> pd.DataFrame:
    """Fallback evaluation set: sample merchants (not rows), keep all their test rows.

    Tail merchants (freq <= k) and head merchants sampled separately with a seeded RNG; if fewer
    exist than requested, all are taken. Adds ``is_tail`` and ``train_freq`` columns.
    """
    t = test.copy()
    t["train_freq"] = freq_of(t, freqs)
    t["is_tail"] = t["train_freq"] <= k
    pool = t[t["ambiguous"] == 0] if exclude_ambiguous else t
    rng = np.random.default_rng(seed)
    tail_m = np.sort(pool.loc[pool["is_tail"], "merchant"].unique())
    head_m = np.sort(pool.loc[~pool["is_tail"], "merchant"].unique())
    pick_t = rng.choice(tail_m, size=min(n_tail, len(tail_m)), replace=False) if len(tail_m) else []
    pick_h = rng.choice(head_m, size=min(n_head, len(head_m)), replace=False) if len(head_m) else []
    chosen = set(pick_t) | set(pick_h)
    return t[t["merchant"].isin(chosen)].reset_index(drop=True)
