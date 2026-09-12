import pytest

from txcat.budget import BudgetExceeded, SpendLedger


def test_ledger_accumulates_persists_and_caps(tmp_path):
    p = tmp_path / "ledger.json"
    led = SpendLedger(p, max_usd=1.0)
    led.add("search", 0.4, "brave")
    led.add("llm", 0.5, "gpt-5-nano")
    assert led.total == pytest.approx(0.9)
    led2 = SpendLedger(p, max_usd=1.0)
    assert led2.total == pytest.approx(0.9)
    assert led2.by_kind()["search"] == pytest.approx(0.4)
    with pytest.raises(BudgetExceeded):
        led2.add("llm", 0.2, "gpt-5-nano")
    assert led2.total == pytest.approx(0.9)  # rejected call not recorded
