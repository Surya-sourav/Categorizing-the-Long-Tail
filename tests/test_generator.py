import numpy as np
import pandas as pd

from txcat.data.schema import PROCESSED_COLUMNS
from txcat.generator.generate import generate
from txcat.generator.noise import noisy_descriptor
from txcat.normalizer import normalize_merchant


def test_noise_is_seeded_and_varied():
    rng = np.random.default_rng(0)
    outs = {noisy_descriptor("Blue Bottle Coffee", "San Francisco", "CA", rng) for _ in range(30)}
    assert len(outs) > 5
    rng2 = np.random.default_rng(0)
    assert noisy_descriptor("Blue Bottle Coffee", "San Francisco", "CA", rng2) in outs
    for o in outs:
        assert o == o.upper() and len(o) <= 40


def test_noise_survives_normalizer_most_of_the_time():
    rng = np.random.default_rng(1)
    hits = sum(
        "BLUE BOTTLE"
        in normalize_merchant(noisy_descriptor("Blue Bottle Coffee", "Oakland", "CA", rng)).text
        for _ in range(200)
    )
    assert hits >= 150  # truncation/abbreviation may drop it sometimes; most survive


def test_generate_schema_and_zipf():
    vocab = pd.DataFrame(
        {
            "merchant_id": [f"m{i}" for i in range(300)],
            "name": [f"Brand {i}" for i in range(300)],
            "osm_key": "shop",
            "osm_value": "x",
            "category": ["groceries", "retail", "health"] * 100,
            "wikidata": "",
        }
    )
    df = generate(
        vocab, n_transactions=5000, alpha=1.1, seed=3, start="2019-01-01", end="2020-12-31"
    )
    assert list(df.columns) == PROCESSED_COLUMNS + [
        "category",
        "merchant_id",
        "alpha",
        "novel_brand",
    ]
    assert len(df) == 5000 and df["source"].iloc[0] == "gen"
    counts = df["merchant_id"].value_counts()
    assert counts.iloc[0] > 20 * counts.median()  # heavy head
    assert (counts <= 3).sum() > 50  # real tail exists
    assert df["date"].is_monotonic_increasing
    df2 = generate(
        vocab, n_transactions=5000, alpha=1.1, seed=3, start="2019-01-01", end="2020-12-31"
    )
    assert df2["raw_merchant"].equals(df["raw_merchant"])


def test_novel_brands_only_appear_after_the_cutoff():
    vocab = pd.DataFrame(
        {
            "merchant_id": [f"m{i}" for i in range(300)],
            "name": [f"Brand {i}" for i in range(300)],
            "osm_key": "shop",
            "osm_value": "x",
            "category": ["groceries", "retail", "health"] * 100,
            "wikidata": "",
        }
    )
    df = generate(
        vocab,
        6000,
        1.1,
        seed=5,
        start="2019-01-01",
        end="2025-12-31",
        novel_share=0.2,
        novel_after="2024-01-01",
    )
    novel = df[df["novel_brand"]]
    assert 50 <= novel["merchant_id"].nunique() <= 60  # rare Zipf brands may never be drawn
    assert (novel["date"] >= pd.Timestamp("2024-01-01")).all()
    assert df["date"].is_monotonic_increasing
    early = df[df["date"] < pd.Timestamp("2024-01-01")]
    assert not early["novel_brand"].any()
