from txcat.llm_fallback import LLMResult
from txcat.writeback import WriteBackPolicy


def res(conf, valid=True):
    return LLMResult(
        category="retail" if valid else "INVALID",
        confidence=conf,
        evidence="",
        valid=valid,
        model_id="m",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1.0,
        cached=True,
    )


def test_modes():
    assert WriteBackPolicy("never").should_write(res(0.99)) is False
    assert WriteBackPolicy("always").should_write(res(0.1)) is True
    assert (
        WriteBackPolicy("always").should_write(res(0.9, valid=False)) is False
    )  # never write INVALID
    g = WriteBackPolicy("confidence_gated", 0.8)
    assert g.should_write(res(0.85)) is True and g.should_write(res(0.79)) is False
