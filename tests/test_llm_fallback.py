import json

import pytest

from txcat.budget import SpendLedger
from txcat.config import LLMModelCfg
from txcat.llm_fallback import CacheMissError, LLMFallback, render_prompt
from txcat.web_search import SearchResult

TAX = ["groceries", "restaurants", "industrial_hardware"]
MODEL = LLMModelCfg(name="gpt-5-nano", provider="openai", api_key_env="OPENAI_API_KEY",
                    price_in_per_1m=0.05, price_out_per_1m=0.40, reasoning_effort="minimal",
                    json_mode="json_schema")


class _Obj:
    """Attribute bag standing in for SDK response objects."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeCompletions:
    def __init__(self, content):
        self.content, self.calls, self.last_kwargs = content, 0, None

    def create(self, **kw):
        self.calls += 1
        self.last_kwargs = kw
        usage = _Obj(prompt_tokens=120, completion_tokens=30)
        choice = _Obj(message=_Obj(content=self.content))
        return _Obj(choices=[choice], usage=usage, model="gpt-5-nano-2025-08-07")


class FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {})()
        self.chat.completions = FakeCompletions(content)


def test_render_prompt_with_and_without_evidence(tmp_path):
    ev = [SearchResult("Grainger", "MRO supplies distributor", "https://grainger.com")]
    with_web = render_prompt("prompts/fallback_with_web.txt", "WW GRAINGER", ev, TAX)
    no_web = render_prompt("prompts/fallback_no_web.txt", "WW GRAINGER", None, TAX)
    assert "1. Grainger — MRO supplies distributor (https://grainger.com)" in with_web
    assert "Web search results" not in no_web and "WW GRAINGER" in no_web
    assert "- groceries\n- restaurants\n- industrial_hardware" in with_web


def test_categorize_parses_caches_and_bills(tmp_path):
    client = FakeClient(json.dumps({"category": "industrial_hardware", "confidence": 0.9,
                                    "evidence": "Grainger sells MRO."}))
    led = SpendLedger(tmp_path / "l.json", 5)
    fb = LLMFallback(MODEL, tmp_path / "llm", "prompts/fallback_no_web.txt", ledger=led,
                     client=client)
    r = fb.categorize("WW GRAINGER", None, TAX)
    assert r.category == "industrial_hardware" and r.confidence == 0.9 and r.valid
    assert r.tokens_in == 120 and r.tokens_out == 30 and r.model_id == "gpt-5-nano-2025-08-07"
    assert client.chat.completions.last_kwargs["model"] == "gpt-5-nano"
    assert client.chat.completions.last_kwargs["reasoning_effort"] == "minimal"
    assert led.total == pytest.approx(120 / 1e6 * 0.05 + 30 / 1e6 * 0.40)
    r2 = fb.categorize("WW GRAINGER", None, TAX)
    assert client.chat.completions.calls == 1 and r2.category == r.category
    assert len(list((tmp_path / "llm").glob("*.json"))) == 1


def test_invalid_category_is_flagged_not_crashed(tmp_path):
    client = FakeClient('{"category": "Hardware Store", "confidence": 0.5, "evidence": "x"}')
    fb = LLMFallback(MODEL, tmp_path / "llm", "prompts/fallback_no_web.txt", ledger=None,
                     client=client)
    r = fb.categorize("X", None, TAX)
    assert r.valid is False and r.category == "INVALID"


def test_reproduce_mode(tmp_path):
    fb = LLMFallback(MODEL, tmp_path / "llm", "prompts/fallback_no_web.txt", ledger=None,
                     client=FakeClient("{}"), allow_live=False)
    with pytest.raises(CacheMissError):
        fb.categorize("X", None, TAX)
