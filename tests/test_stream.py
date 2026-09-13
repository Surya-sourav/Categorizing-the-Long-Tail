import pandas as pd

from txcat.embedder import Embedder, FakeHashBackend
from txcat.gate import ConfidenceGate
from txcat.index import HNSWIndex
from txcat.llm_fallback import LLMResult
from txcat.stream import run_stream
from txcat.writeback import WriteBackPolicy


class OracleFallback:
    """Stands in for LLMFallback: gold label at confidence 0.9, or a wrong label for `wrong`."""

    def __init__(self, gold, wrong=()):
        self.gold, self.wrong, self.calls = gold, set(wrong), 0

    def categorize(self, merchant, evidence, taxonomy):
        self.calls += 1
        cat = "WRONGCAT" if merchant in self.wrong else self.gold[merchant]
        return LLMResult(cat, 0.9, "", True, "m", 100, 20, 50.0, False)


def _setup(tmp_path):
    emb = Embedder("fake/m", tmp_path, backend=FakeHashBackend(16))
    idx = HNSWIndex(dim=16, M=8, ef_construction=50, seed=1)
    v = emb.embed(["STAPLES"])
    idx.add(v, ["office_supplies"], ["STAPLES"], [50], pd.to_datetime(["2019-01-01"]))
    txns = pd.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(6)],
            "date": pd.to_datetime([f"2024-01-0{i + 1}" for i in range(6)]),
            "merchant": ["STAPLES", "NEWCO", "NEWCO", "NEWCO", "OTHERCO", "STAPLES"],
            "category": [
                "office_supplies",
                "retail",
                "retail",
                "retail",
                "health",
                "office_supplies",
            ],
            "ambiguous": [0] * 6,
        }
    )
    gold = dict(zip(txns["merchant"], txns["category"], strict=False))
    return emb, idx, txns, gold


def test_always_writeback_reduces_fallbacks(tmp_path):
    emb, idx, txns, gold = _setup(tmp_path)
    fb = OracleFallback(gold)
    log = run_stream(
        txns,
        idx,
        emb,
        ConfidenceGate("threshold", 0.99),
        fb,
        WriteBackPolicy("always"),
        lambda m: None,
        window=2,
        price_in_per_1m=1.0,
        price_out_per_1m=1.0,
        search_price=0.005,
    )
    # NEWCO falls back once, is written back, then is served by kNN twice; OTHERCO falls back once
    assert fb.calls == 2
    assert len(idx) == 3 and (idx.meta["source"] == "writeback").sum() == 2
    assert log.per_txn["fallback"].tolist() == [False, True, False, False, True, False]
    assert log.per_txn["pred"].tolist() == txns["category"].tolist()
    assert log.windows["fallback_rate"].tolist() == [0.5, 0.0, 0.5]
    assert log.summary["writeback_error_rate"] == 0.0 and log.summary["n_writebacks"] == 2


def test_never_writeback_keeps_falling_back_and_error_rate_counts_bad_writes(tmp_path):
    emb, idx, txns, gold = _setup(tmp_path)
    fb = OracleFallback(gold)
    log = run_stream(
        txns,
        idx,
        emb,
        ConfidenceGate("threshold", 0.99),
        fb,
        WriteBackPolicy("never"),
        lambda m: None,
        window=2,
    )
    assert fb.calls == 4 and len(idx) == 1
    emb, idx, txns, gold = _setup(tmp_path / "b")
    fb = OracleFallback(gold, wrong={"NEWCO"})
    log = run_stream(
        txns,
        idx,
        emb,
        ConfidenceGate("threshold", 0.99),
        fb,
        WriteBackPolicy("always"),
        lambda m: None,
        window=2,
    )
    assert log.summary["writeback_error_rate"] == 0.5  # NEWCO written wrong, OTHERCO written right
    assert log.per_txn["pred"].tolist()[2] == "WRONGCAT"  # pollution propagates via kNN
