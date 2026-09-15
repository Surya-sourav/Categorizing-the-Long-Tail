import json

from txcat.agentic_search import AgenticSearchCategorizer

TAX = ["groceries", "restaurants"]


class FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1

        class Out:
            pass

        r = Out()
        r.output_text = json.dumps(
            {"category": "groceries", "confidence": 0.8, "evidence": "Safeway is a supermarket [1]"}
        )
        r.model = "gpt-5-mini-2025-08-07"

        class U:
            input_tokens = 9000
            output_tokens = 40

        r.usage = U()
        r.output = [
            type("W", (), {"type": "web_search_call", "action": {"query": "SAFEWAY store"}})(),
            type("M", (), {"type": "message"})(),
        ]
        return r


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_agentic_categorize_caches_queries_and_result(tmp_path):
    c = AgenticSearchCategorizer(
        "gpt-5-mini",
        tmp_path,
        client=FakeClient(),
        price_in_per_1m=0.25,
        price_out_per_1m=2.0,
        search_price_per_call=0.01,
    )
    r = c.categorize("SAFEWAY", TAX)
    assert (
        r["category"] == "groceries" and r["queries"] == ["SAFEWAY store"] and r["n_searches"] == 1
    )
    r2 = c.categorize("SAFEWAY", TAX)
    assert c.client.responses.calls == 1 and r2["category"] == "groceries"
    assert r["usd"] > 0.01


def test_unparseable_answer_becomes_invalid_not_exception(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from txcat.agentic_search import AgenticSearchCategorizer

    resp = SimpleNamespace(
        output=[],
        output_text='{"category": "RETAIL", "evidence": "unterminated',
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model="m",
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: resp))
    ag = AgenticSearchCategorizer("m", tmp_path, client=client)
    out = ag.categorize("acme", ["RETAIL", "OTHER"])
    assert out["category"] == "INVALID" and out["valid"] is False
