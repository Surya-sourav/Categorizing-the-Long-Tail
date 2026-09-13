import numpy as np
import pandas as pd

from txcat.baselines import EmbeddingLRBaseline
from txcat.embedder import Embedder, FakeHashBackend


def test_lr_baseline_fits_and_predicts(tmp_path):
    emb = Embedder("fake/m", tmp_path, backend=FakeHashBackend(32))
    train = pd.DataFrame(
        {"merchant": [f"M{i}" for i in range(60)], "category": ["a", "b", "c"] * 20}
    )
    clf = EmbeddingLRBaseline(emb).fit(train["merchant"].tolist(), train["category"].tolist())
    pred = clf.predict(["M0", "M1", "ZZZ"])
    assert len(pred) == 3 and pred[0] == "a" and pred[1] == "b"
    proba = clf.predict_proba(["M0"])
    assert proba.shape == (1, 3) and np.isclose(proba.sum(), 1.0)
