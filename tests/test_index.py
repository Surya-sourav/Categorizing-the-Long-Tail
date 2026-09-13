import numpy as np
import pandas as pd
import pytest

from txcat.index import HNSWIndex


def _unit(rows):
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_add_query_margin_and_metadata():
    idx = HNSWIndex(dim=3, M=8, ef_construction=50, seed=1)
    vecs = _unit([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    idx.add(vecs, labels=["a", "b", "a"], merchant_ids=["m1", "m2", "m3"], freqs=[10, 1, 3],
            first_seen=pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"]))
    r = idx.query(_unit([[1, 0, 0]])[0], k=3, ef_search=50)
    assert r.labels[0] == "a" and r.merchant_ids[0] == "m1"
    assert r.sims[0] == pytest.approx(1.0, abs=1e-4)
    assert r.margin == pytest.approx(r.sims[0] - r.sims[1], abs=1e-6)
    assert r.freqs[0] == 10
    assert len(idx) == 3


def test_add_grows_and_writeback_tagging():
    idx = HNSWIndex(dim=2, M=8, ef_construction=50, seed=1, max_elements=2)
    idx.add(_unit([[1, 0], [0, 1]]), ["x", "y"], ["m1", "m2"], [1, 1],
            pd.to_datetime(["2019-01-01"] * 2))
    idx.add(_unit([[1, 1]]), ["z"], ["m3"], [1], pd.to_datetime(["2024-01-01"]), source="writeback")
    assert len(idx) == 3
    assert idx.meta.loc[2, "source"] == "writeback"


def test_save_load_roundtrip(tmp_path):
    idx = HNSWIndex(dim=2, M=8, ef_construction=50, seed=3)
    idx.add(_unit([[1, 0], [0, 1]]), ["x", "y"], ["m1", "m2"], [4, 2],
            pd.to_datetime(["2019-01-01"] * 2))
    idx.save(tmp_path / "ix")
    idx2 = HNSWIndex.load(tmp_path / "ix")
    r = idx2.query(_unit([[0, 1]])[0], k=1)
    assert r.labels[0] == "y" and idx2.meta.loc[1, "freq"] == 2


def test_same_seed_same_neighbors():
    rng = np.random.default_rng(0)
    vecs = _unit(rng.standard_normal((500, 16)))
    q = _unit(rng.standard_normal((1, 16)))[0]
    outs = []
    for _ in range(2):
        idx = HNSWIndex(dim=16, M=8, ef_construction=50, seed=11)
        idx.add(vecs, [str(i) for i in range(500)], [str(i) for i in range(500)], [1] * 500,
                pd.to_datetime(["2019-01-01"] * 500))
        outs.append(idx.query(q, k=5, ef_search=50).merchant_ids)
    assert outs[0] == outs[1]
