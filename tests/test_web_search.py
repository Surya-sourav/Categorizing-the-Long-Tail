import json

import pytest

from txcat.budget import SpendLedger
from txcat.web_search import CacheMissError, SearchResult, WebSearchClient

BRAVE_PAYLOAD = {"web": {"results": [
    {"title": "WW Grainger - Industrial Supply",
     "description": "Grainger is a distributor of MRO supplies.",
     "url": "https://www.grainger.com/"},
    {"title": "Grainger wiki",
     "description": "W. W. Grainger, Inc. is an American industrial supply company.",
     "url": "https://en.wikipedia.org/wiki/W._W._Grainger"},
]}}


class FakeHTTP:
    def __init__(self):
        self.calls = 0

    def get(self, url, params, headers, timeout):
        self.calls += 1
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return BRAVE_PAYLOAD
        return R()


def test_search_caches_and_bills_once(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 10)
    http = FakeHTTP()
    c = WebSearchClient("brave", tmp_path / "ws", api_key="k", ledger=led, http=http,
                        min_interval_s=0)
    r1 = c.search("WW GRAINGER", num_results=5)
    assert [x.url for x in r1][:1] == ["https://www.grainger.com/"]
    assert isinstance(r1[0], SearchResult) and r1[0].snippet.startswith("Grainger")
    r2 = c.search("WW GRAINGER", num_results=5)
    assert http.calls == 1 and r2 == r1
    assert led.total == pytest.approx(0.005)
    files = list((tmp_path / "ws").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["query"] == "WW GRAINGER" and payload["provider"] == "brave"
    assert "timestamp" in payload
    assert set(payload["results"][0]) == {"title", "snippet", "url"}


def test_reproduce_mode_never_calls_network(tmp_path):
    c = WebSearchClient("brave", tmp_path / "ws", api_key=None, ledger=None, http=FakeHTTP(),
                        allow_live=False)
    with pytest.raises(CacheMissError):
        c.search("UNSEEN")


def test_query_is_only_the_merchant_string(tmp_path):
    c = WebSearchClient("brave", tmp_path / "ws", api_key="k",
                        ledger=SpendLedger(tmp_path / "l", 1),
                        http=FakeHTTP(), min_interval_s=0)
    with pytest.raises(ValueError):
        c.search("STAPLES $45.00 2024-01-01")  # digits that look like an amount/date are rejected
