import pandas as pd

from txcat.fes import build_dc_fes, build_oklahoma_coldstart_fes


def test_dc_fes_takes_all_tail_when_small():
    test = pd.DataFrame(
        {"merchant": ["a", "a", "b", "c", "d"], "ambiguous": [0] * 5, "category": ["x"] * 5}
    )
    freqs = pd.Series({"a": 10, "b": 1})
    f = build_dc_fes(test, freqs, k=3, n_tail=2500, n_head=500, seed=1)
    assert set(f["merchant"]) == {"a", "b", "c", "d"} and f["is_tail"].sum() == 3


def test_oklahoma_coldstart_excludes_dc_merchants():
    ok = pd.DataFrame(
        {
            "merchant": ["STAPLES", "LOCAL FARM CO", "TULSA WELDING", "STAPLES"],
            "ambiguous": [0] * 4,
            "category": ["office_supplies", "groceries", "industrial_hardware", "office_supplies"],
        }
    )
    dc_train_merchants = {"STAPLES"}
    f = build_oklahoma_coldstart_fes(ok, dc_train_merchants, n=800, seed=1)
    assert set(f["merchant"]) == {"LOCAL FARM CO", "TULSA WELDING"}
    assert (f["train_freq"] == 0).all() and f["is_tail"].all()
