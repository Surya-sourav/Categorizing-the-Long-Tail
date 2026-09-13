"""Embeddings + logistic regression baseline (row 2 of the main table)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from txcat.embedder import Embedder


class EmbeddingLRBaseline:
    def __init__(self, emb: Embedder, C: float = 4.0, seed: int = 42):
        self.emb = emb
        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, merchants: list[str], labels: list[str]) -> EmbeddingLRBaseline:
        self.clf.fit(self.emb.embed(merchants), np.asarray(labels))
        return self

    def predict(self, merchants: list[str]) -> np.ndarray:
        return self.clf.predict(self.emb.embed(merchants))

    def predict_proba(self, merchants: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.emb.embed(merchants))
