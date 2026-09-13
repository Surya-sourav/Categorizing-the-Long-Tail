import numpy as np

from txcat.gate import ConfidenceGate
from txcat.index import QueryResult


def qr(s1, s2, f1=5):
    return QueryResult(labels=["a", "b"], sims=[s1, s2], merchant_ids=["x", "y"], freqs=[f1, 1])


def test_threshold_mode():
    g = ConfidenceGate("threshold", 0.65)
    assert g.should_fallback(qr(0.60, 0.5)) is True
    assert g.should_fallback(qr(0.70, 0.5)) is False


def test_margin_mode():
    g = ConfidenceGate("margin", 0.10)
    assert g.should_fallback(qr(0.80, 0.75)) is True
    assert g.should_fallback(qr(0.80, 0.60)) is False


def test_calibrated_lr_learns_that_low_sim_means_wrong():
    rng = np.random.default_rng(0)
    results, correct = [], []
    for _ in range(400):
        s1 = rng.uniform(0.3, 1.0)
        results.append(qr(s1, s1 - rng.uniform(0, 0.2), f1=int(rng.integers(1, 50))))
        correct.append(bool(rng.random() < s1))  # higher sim -> more likely correct
    g = ConfidenceGate("calibrated_lr", 0.5).fit(results, correct)
    assert g.should_fallback(qr(0.35, 0.30)) is True
    assert g.should_fallback(qr(0.98, 0.60)) is False
    p = g.p_wrong(qr(0.5, 0.4))
    assert 0.0 <= p <= 1.0
