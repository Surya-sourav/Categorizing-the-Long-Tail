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


def test_uncapped_kind_is_recorded_but_not_capped(tmp_path):
    led = SpendLedger(tmp_path / "l.json", max_usd=1.0, uncapped_kinds=("search",))
    led.add("search", 5.0, "firecrawl")  # prepaid credits: reference USD only
    led.add("llm", 0.9, "gpt")
    assert led.total == pytest.approx(5.9) and led.capped_total == pytest.approx(0.9)
    with pytest.raises(BudgetExceeded):
        led.add("llm", 0.2, "gpt")


def test_from_config_exempts_prepaid_search(tmp_path):
    from types import SimpleNamespace

    from txcat.budget import SpendLedger

    cfg = SimpleNamespace(
        budget=SimpleNamespace(ledger_path=tmp_path / "l.json", max_usd=1.0),
        search=SimpleNamespace(prepaid=True),
    )
    led = SpendLedger.from_config(cfg)
    led.add("search", 50.0)  # prepaid credits: tracked, never capped
    led.add("llm", 0.5)
    assert led.capped_total == 0.5
    cfg.search.prepaid = False
    assert SpendLedger.from_config(cfg).uncapped_kinds == set()
