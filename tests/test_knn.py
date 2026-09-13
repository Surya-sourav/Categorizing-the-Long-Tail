import pandas as pd

from txcat.embedder import Embedder, FakeHashBackend
from txcat.knn import build_merchant_index, predict_knn


def test_build_and_predict(tmp_path):
    train = pd.DataFrame(
        {
            "merchant": ["STAPLES", "STAPLES", "STAPLES", "USPS", "GRAINGER"],
            "category": [
                "office_supplies",
                "office_supplies",
                "retail",
                "financial_postal_shipping",
                "industrial_hardware",
            ],
            "date": pd.to_datetime(
                ["2019-01-01", "2019-02-01", "2019-03-01", "2020-01-01", "2021-01-01"]
            ),
            "ambiguous": [0] * 5,
        }
    )
    emb = Embedder("fake/m", tmp_path, backend=FakeHashBackend(16))
    idx = build_merchant_index(train, emb, M=8, ef_construction=50, seed=1)
    assert len(idx) == 3
    row = idx.meta[idx.meta["merchant_id"] == "STAPLES"].iloc[0]
    assert row["label"] == "office_supplies" and row["freq"] == 3  # majority label, frequency
    assert str(row["first_seen"].date()) == "2019-01-01"

    test = pd.DataFrame({"merchant": ["STAPLES", "NEVER SEEN"]})
    pred = predict_knn(test, idx, emb, k=2, ef_search=50)
    assert list(pred.columns) == ["pred", "sim1", "sim2", "margin", "top1_freq", "top1_merchant"]
    assert pred.loc[0, "pred"] == "office_supplies" and pred.loc[0, "sim1"] > 0.999
