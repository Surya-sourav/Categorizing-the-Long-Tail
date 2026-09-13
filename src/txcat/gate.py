"""Decide whether a kNN result is trusted or routed to the LLM fallback."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from txcat.index import QueryResult

MODES = ("threshold", "margin", "calibrated_lr")


def _features(r: QueryResult) -> list[float]:
    s2 = r.sims[1] if len(r.sims) > 1 else 0.0
    return [r.sims[0], s2, r.sims[0] - s2, float(np.log1p(r.top1_freq))]


class ConfidenceGate:
    """``threshold``: fallback if sim1 < t. ``margin``: fallback if sim1 - sim2 < t.
    ``calibrated_lr``: fallback if P(top-1 wrong) > t, LR on [sim1, sim2, margin, log1p(freq)]."""

    def __init__(self, mode: str = "threshold", threshold: float = 0.65):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.threshold = mode, threshold
        self._lr: LogisticRegression | None = None

    def fit(self, results: list[QueryResult], correct: list[bool]) -> ConfidenceGate:
        """Fit the calibrated LR on a calibration slice (label 1 = top-1 wrong)."""
        X = np.asarray([_features(r) for r in results])
        y = 1 - np.asarray(correct, dtype=int)
        self._lr = LogisticRegression(max_iter=1000).fit(X, y)
        return self

    def p_wrong(self, r: QueryResult) -> float:
        if self._lr is None:
            raise RuntimeError("calibrated_lr gate must be fit() first")
        return float(self._lr.predict_proba(np.asarray([_features(r)]))[0, 1])

    def should_fallback(self, r: QueryResult) -> bool:
        if self.mode == "threshold":
            return r.sims[0] < self.threshold
        if self.mode == "margin":
            return r.margin < self.threshold
        return self.p_wrong(r) > self.threshold
