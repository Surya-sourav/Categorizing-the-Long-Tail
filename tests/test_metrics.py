import numpy as np
import pytest

from txcat.metrics import (
    bootstrap_ci,
    cost_per_1k,
    latency_p50_p95,
    macro_f1,
    paired_bootstrap_diff,
)


def test_macro_f1_with_mask():
    y = np.array(["a", "a", "b", "b", "c"])
    p = np.array(["a", "b", "b", "b", "c"])
    assert macro_f1(y, p) == pytest.approx((2 / 3 + 0.8 + 1.0) / 3, abs=1e-6)
    assert macro_f1(y, p, mask=np.array([True, True, False, False, False])) == pytest.approx(
        0.5 * (2 / 3 + 0.0), abs=1e-6
    )
    assert macro_f1(y, p, mask=np.zeros(5, bool)) != macro_f1(y, p)  # empty mask returns nan
    assert np.isnan(macro_f1(y, p, mask=np.zeros(5, bool)))


def test_latency_and_cost():
    p50, p95 = latency_p50_p95([10, 20, 30, 40, 1000])
    assert p50 == 30 and p95 > 500
    usd = cost_per_1k(
        n_txns=2000,
        tokens_in=1_000_000,
        tokens_out=100_000,
        search_calls=500,
        price_in_per_1m=0.05,
        price_out_per_1m=0.40,
        search_price_per_1k=5.0,
    )
    assert usd == pytest.approx((0.05 + 0.04 + 2.5) / 2, abs=1e-6)


def test_bootstrap_ci_and_paired_diff():
    lo, mean, hi = bootstrap_ci([0.5, 0.52, 0.48], n_boot=500, seed=0)
    assert lo <= mean <= hi and abs(mean - 0.5) < 0.02
    a = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], bool)
    b = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1], bool)
    d, lo, hi = paired_bootstrap_diff(a, b, n_boot=1000, seed=0)
    assert d == pytest.approx(0.4) and lo > 0.0  # a beats b, CI excludes zero
