"""hnswlib cosine index with per-entry metadata (label, merchant id, frequency, first seen,
source)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd

META_COLS = ["merchant_id", "label", "freq", "first_seen", "source"]


@dataclass
class QueryResult:
    labels: list[str]
    sims: list[float]
    merchant_ids: list[str]
    freqs: list[int]

    @property
    def margin(self) -> float:
        return self.sims[0] - (self.sims[1] if len(self.sims) > 1 else 0.0)

    @property
    def top1_freq(self) -> int:
        return self.freqs[0]


class HNSWIndex:
    """One entry per unique training merchant. Grows on write-back."""

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200, seed: int = 42,
                 max_elements: int = 1000):
        self.dim, self.M, self.ef_construction, self.seed = dim, M, ef_construction, seed
        self._ix = hnswlib.Index(space="cosine", dim=dim)
        self._ix.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M,
                            random_seed=seed)
        # single-threaded inserts keep graph construction seed-deterministic
        self._ix.set_num_threads(1)
        self.meta = pd.DataFrame(columns=META_COLS)

    def __len__(self) -> int:
        return int(self._ix.get_current_count())

    def add(self, vectors: np.ndarray, labels: list[str], merchant_ids: list[str],
            freqs: list[int], first_seen: pd.Series | pd.DatetimeIndex,
            source: str = "train") -> None:
        n = len(labels)
        if len(self) + n > self._ix.get_max_elements():
            self._ix.resize_index(max(len(self) + n, 2 * self._ix.get_max_elements()))
        ids = np.arange(len(self), len(self) + n)
        self._ix.add_items(np.asarray(vectors, dtype=np.float32), ids, num_threads=1)
        new = pd.DataFrame({"merchant_id": list(merchant_ids), "label": list(labels),
                            "freq": list(freqs),
                            "first_seen": pd.to_datetime(list(first_seen)), "source": source},
                           index=ids)
        self.meta = new if self.meta.empty else pd.concat([self.meta, new])

    def query(self, vector: np.ndarray, k: int = 5, ef_search: int = 64) -> QueryResult:
        k = min(k, len(self))
        self._ix.set_ef(max(ef_search, k))
        ids, dists = self._ix.knn_query(np.asarray(vector, dtype=np.float32).reshape(1, -1), k=k)
        ids, sims = ids[0].tolist(), (1.0 - dists[0]).tolist()
        rows = self.meta.loc[ids]
        return QueryResult(labels=rows["label"].tolist(), sims=sims,
                           merchant_ids=rows["merchant_id"].tolist(),
                           freqs=rows["freq"].astype(int).tolist())

    def query_batch(
        self, vectors: np.ndarray, k: int = 5, ef_search: int = 64
    ) -> list[QueryResult]:
        k = min(k, len(self))
        self._ix.set_ef(max(ef_search, k))
        ids, dists = self._ix.knn_query(np.asarray(vectors, dtype=np.float32), k=k)
        out = []
        for row_ids, row_d in zip(ids, dists, strict=True):
            rows = self.meta.loc[row_ids.tolist()]
            out.append(QueryResult(rows["label"].tolist(), (1.0 - row_d).tolist(),
                                   rows["merchant_id"].tolist(),
                                   rows["freq"].astype(int).tolist()))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._ix.save_index(str(path / "index.bin"))
        self.meta.to_parquet(path / "meta.parquet")
        (path / "params.txt").write_text(f"{self.dim},{self.M},{self.ef_construction},{self.seed}")

    @classmethod
    def load(cls, path: str | Path) -> HNSWIndex:
        path = Path(path)
        dim, M, efc, seed = (int(x) for x in (path / "params.txt").read_text().split(","))
        obj = cls(dim, M, efc, seed)
        obj.meta = pd.read_parquet(path / "meta.parquet")
        obj._ix.load_index(str(path / "index.bin"), max_elements=max(len(obj.meta) * 2, 1000))
        return obj
