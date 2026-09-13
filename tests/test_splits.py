import pandas as pd

from txcat.config import WindowCfg
from txcat.data.splits import (
    merchant_frequencies,
    sample_fes,
    tail_mask,
    temporal_split,
    volume_rule_ok,
)


def _df():
    dates = [
        "2019-05-01",
        "2020-05-01",
        "2021-05-01",
        "2023-12-31",
        "2024-01-01",
        "2024-06-01",
        "2025-01-01",
    ]
    merchants = ["A", "A", "B", "C", "A", "B", "D"]
    return pd.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(7)],
            "date": pd.to_datetime(dates),
            "merchant": merchants,
            "category": ["x"] * 7,
            "ambiguous": [0] * 7,
        }
    )


def test_temporal_split_is_by_date_only():
    w = WindowCfg(train_start="2019-01-01", train_end="2023-12-31", test_start="2024-01-01")
    tr, te = temporal_split(_df(), w)
    assert list(tr["txn_id"]) == ["t0", "t1", "t2", "t3"]
    assert list(te["txn_id"]) == ["t4", "t5", "t6"]


def test_frequencies_and_tail_mask():
    w = WindowCfg(train_start="2019-01-01", train_end="2023-12-31", test_start="2024-01-01")
    tr, te = temporal_split(_df(), w)
    f = merchant_frequencies(tr)
    assert f["A"] == 2 and f["B"] == 1 and f["C"] == 1 and "D" not in f
    m = tail_mask(te, f, k=1)
    assert list(m) == [False, True, True]  # A freq 2 > 1; B freq 1; D freq 0 (unseen) is tail


def test_volume_rule():
    te = pd.DataFrame({"merchant": ["m1", "m2", "m3"], "date": pd.to_datetime(["2024-01-01"] * 3)})
    f = pd.Series({"m1": 1})
    ok, stats = volume_rule_ok(te, f, k=3, min_test_txns=3, min_tail_merchants=3)
    assert ok is True and stats["n_test_txns"] == 3 and stats["n_tail_merchants"] == 3
    ok, _ = volume_rule_ok(te, f, k=3, min_test_txns=4, min_tail_merchants=3)
    assert ok is False


def test_sample_fes_is_by_merchant_and_seeded():
    te = pd.DataFrame(
        {
            "merchant": [f"m{i // 3}" for i in range(60)],
            "ambiguous": [0] * 60,
            "category": ["c"] * 60,
        }
    )
    f = pd.Series({f"m{i}": (10 if i < 5 else 1) for i in range(20)})  # 5 head, 15 tail merchants
    a = sample_fes(te, f, k=3, n_tail=4, n_head=2, seed=1)
    b = sample_fes(te, f, k=3, n_tail=4, n_head=2, seed=1)
    assert a["merchant"].nunique() == 6 and set(a["merchant"]) == set(b["merchant"])
    assert (a.groupby("merchant").size() == 3).all()  # all of a merchant's rows come along
    assert a["is_tail"].sum() == 4 * 3
    c = sample_fes(
        te, f, k=3, n_tail=100, n_head=100, seed=1
    )  # asks for more than exist -> takes all
    assert c["merchant"].nunique() == 20
