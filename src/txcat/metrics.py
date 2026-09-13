"""Evaluation metrics: masked macro-F1, latency percentiles, cost per 1k, bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred, mask=None) -> float:
    """Macro-F1 over the union of labels in (masked) truth and predictions.

    Returns NaN if the mask selects nothing.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if mask is not None:
        y_true, y_pred = y_true[np.asarray(mask, bool)], y_pred[np.asarray(mask, bool)]
    if len(y_true) == 0:
        return float("nan")
    # labels = union of truth and prediction (sklearn default); predicted-but-absent scores 0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def latency_p50_p95(timings_ms) -> tuple[float, float]:
    t = np.asarray(timings_ms, dtype=float)
    return float(np.percentile(t, 50)), float(np.percentile(t, 95))


def cost_per_1k(
    n_txns: int,
    tokens_in: int,
    tokens_out: int,
    search_calls: int,
    price_in_per_1m: float,
    price_out_per_1m: float,
    search_price_per_1k: float,
) -> float:
    """USD per 1,000 transactions given total token and search usage over ``n_txns`` rows."""
    usd = (
        tokens_in / 1e6 * price_in_per_1m
        + tokens_out / 1e6 * price_out_per_1m
        + search_calls / 1000 * search_price_per_1k
    )
    return float(usd / n_txns * 1000)


def bootstrap_ci(
    values, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """(lo, mean, hi) percentile bootstrap over a small vector (e.g. per-seed metrics)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(v.mean()), float(np.quantile(means, 1 - a))


def paired_bootstrap_diff(
    correct_a, correct_b, n_boot: int = 5000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Paired bootstrap over units (merchants).

    Resamples indices and computes mean(a) - mean(b). Returns (diff, lo, hi).
    """
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    al = (1 - level) / 2
    return (
        float(a.mean() - b.mean()),
        float(np.quantile(diffs, al)),
        float(np.quantile(diffs, 1 - al)),
    )
