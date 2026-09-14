"""Fallback evaluation sets: frozen merchant lists every API-backed method is evaluated on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from txcat.data.splits import freq_of, sample_fes


def build_dc_fes(
    test: pd.DataFrame, freqs: pd.Series, k: int, n_tail: int, n_head: int, seed: int
) -> pd.DataFrame:
    t = test[test["ambiguous"] == 0]
    n_tail_avail = int(t.loc[freq_of(t, freqs) <= k, "merchant"].nunique())
    if n_tail_avail <= n_tail:
        return sample_fes(test, freqs, k, n_tail=n_tail_avail, n_head=n_head, seed=seed)
    return sample_fes(test, freqs, k, n_tail=n_tail, n_head=n_head, seed=seed)


def build_oklahoma_coldstart_fes(
    ok: pd.DataFrame, dc_train_merchants: set[str], n: int, seed: int
) -> pd.DataFrame:
    t = ok[(ok["ambiguous"] == 0) & (~ok["merchant"].isin(dc_train_merchants))].copy()
    rng = np.random.default_rng(seed)
    uniq = np.sort(t["merchant"].unique())
    chosen = set(rng.choice(uniq, size=min(n, len(uniq)), replace=False))
    out = t[t["merchant"].isin(chosen)].reset_index(drop=True)
    out["train_freq"] = 0
    out["is_tail"] = True
    return out


def save_fes(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_fes(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def tail_subset(fes: pd.DataFrame, n: int) -> list[str]:
    """The first ``n`` unique non-ambiguous tail merchants in FES order.

    Shared by the prompt-sensitivity check, the agentic-search ablation and the free-endpoint
    open-weight rows, so every subsample is the same 300 merchants.
    """
    t = fes[(fes["ambiguous"] == 0) & fes["is_tail"]].drop_duplicates("merchant")
    return t["merchant"].head(n).tolist()
