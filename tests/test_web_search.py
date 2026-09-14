import json

import pytest

from txcat.budget import SpendLedger
from txcat.web_search import CacheMissError, SearchResult, WebSearchClient

BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "WW Grainger - Industrial Supply",
                "description": "Grainger is a distributor of MRO supplies.",
                "url": "https://www.grainger.com/",
            },
            {
                "title": "Grainger wiki",
                "description": "W. W. Grainger, Inc. is an American industrial supply company.",
                "url": "https://en.wikipedia.org/wiki/W._W._Grainger",
            },
        ]
    }
}


class FakeHTTP:
    def __init__(self):
        self.calls = 0

    def get(self, url, params, headers, timeout):
        self.calls += 1

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return BRAVE_PAYLOAD

        return R()


def test_search_caches_and_bills_once(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 10)
    http = FakeHTTP()
    c = WebSearchClient(
        "brave", tmp_path / "ws", api_key="k", ledger=led, http=http, min_interval_s=0
    )
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
    c = WebSearchClient(
        "brave", tmp_path / "ws", api_key=None, ledger=None, http=FakeHTTP(), allow_live=False
    )
    with pytest.raises(CacheMissError):
        c.search("UNSEEN")


def test_query_is_only_the_merchant_string(tmp_path):
    c = WebSearchClient(
        "brave",
        tmp_path / "ws",
        api_key="k",
        ledger=SpendLedger(tmp_path / "l", 1),
        http=FakeHTTP(),
        min_interval_s=0,
    )
    with pytest.raises(ValueError):
        c.search("STAPLES $45.00 2024-01-01")  # digits that look like an amount/date are rejected


TAVILY_PAYLOAD = {
    "query": "WW GRAINGER",
    "results": [
        {
            "title": "Grainger Industrial Supply",
            "url": "https://www.grainger.com/",
            "content": "Grainger is your premier industrial supplies provider.",
            "score": 0.93,
        },
        {
            "title": "W. W. Grainger - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/W._W._Grainger",
            "content": "American industrial supply company.",
            "score": 0.81,
        },
    ],
}


class FakeTavilyHTTP:
    def __init__(self):
        self.calls, self.last = 0, None

    def post(self, url, json, headers, timeout):
        self.calls += 1
        self.last = (url, json, headers)

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return TAVILY_PAYLOAD

        return R()


def test_tavily_provider_posts_and_maps_content_to_snippet(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 10)
    http = FakeTavilyHTTP()
    c = WebSearchClient(
        "tavily",
        tmp_path / "ws",
        api_key="tk",
        ledger=led,
        http=http,
        min_interval_s=0,
        price_per_1k=8.0,
    )
    r = c.search("WW GRAINGER", num_results=5)
    assert http.last[0].endswith("/search") and http.last[1]["query"] == "WW GRAINGER"
    assert http.last[1]["search_depth"] == "basic" and http.last[2]["Authorization"] == "Bearer tk"
    assert r[0].snippet.startswith("Grainger is") and r[0].url == "https://www.grainger.com/"
    assert led.total == pytest.approx(0.008)
    payload = json.loads(next((tmp_path / "ws").glob("*.json")).read_text())
    assert payload["provider"] == "tavily" and set(payload["results"][0]) == {
        "title",
        "snippet",
        "url",
    }
    assert c.search("WW GRAINGER").__len__() == 2 and http.calls == 1  # cached


def test_unknown_provider_rejected(tmp_path):
    with pytest.raises(ValueError):
        WebSearchClient("google", tmp_path, api_key="k")


FIRECRAWL_PAYLOAD = {
    "success": True,
    "creditsUsed": 2,
    "data": {
        "web": [
            {
                "url": "https://www.grainger.com/",
                "title": "Grainger Industrial Supply",
                "description": "Grainger is your premier industrial supplies provider.",
                "category": "",
            },
            {
                "url": "https://en.wikipedia.org/wiki/W._W._Grainger",
                "title": "W. W. Grainger",
                "description": "American industrial supply company.",
                "category": "",
            },
        ]
    },
}


class FakeFirecrawlHTTP:
    def __init__(self):
        self.calls, self.last = 0, None

    def post(self, url, json, headers, timeout):
        self.calls += 1
        self.last = (url, json, headers)

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return FIRECRAWL_PAYLOAD

        return R()


def test_firecrawl_provider_uses_v2_search_and_records_credits(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 10)
    http = FakeFirecrawlHTTP()
    c = WebSearchClient(
        "firecrawl",
        tmp_path / "ws",
        api_key="fc",
        ledger=led,
        http=http,
        min_interval_s=0,
        price_per_1k=10.7,
    )
    r = c.search("WW GRAINGER", num_results=5)
    assert http.last[0].endswith("/v2/search")
    assert http.last[1] == {"query": "WW GRAINGER", "limit": 5, "sources": ["web"], "country": "US"}
    assert http.last[2]["Authorization"] == "Bearer fc"
    assert r[0].title == "Grainger Industrial Supply" and r[0].snippet.startswith("Grainger is")
    payload = json.loads(next((tmp_path / "ws").glob("*.json")).read_text())
    assert payload["provider"] == "firecrawl" and payload["credits_used"] == 2
    assert led.total == pytest.approx(0.0107)
    assert len(c.search("WW GRAINGER")) == 2 and http.calls == 1


class FlakyHTTP:
    """First N responses are 502, then a normal Firecrawl payload."""

    def __init__(self, fail_first=1, status=502):
        self.calls, self.fail_first, self.status = 0, fail_first, status

    def post(self, url, json, headers, timeout):
        self.calls += 1
        status = self.status if self.calls <= self.fail_first else 200

        class R:
            status_code = status

            def raise_for_status(self):
                if status >= 400:
                    import requests as _rq

                    raise _rq.HTTPError(f"{status}", response=self)

            def json(self):
                return FIRECRAWL_PAYLOAD

        return R()


def test_search_retries_transient_5xx_then_succeeds(tmp_path, monkeypatch):
    import txcat.web_search as ws_mod

    monkeypatch.setattr(ws_mod.time, "sleep", lambda s: None)
    http = FlakyHTTP(fail_first=2)
    c = WebSearchClient(
        "firecrawl", tmp_path / "ws", api_key="k", ledger=None, http=http, min_interval_s=0
    )
    assert len(c.search("WW GRAINGER")) == 2 and http.calls == 3


def test_search_does_not_retry_4xx_and_refunds_ledger(tmp_path, monkeypatch):
    import txcat.web_search as ws_mod

    monkeypatch.setattr(ws_mod.time, "sleep", lambda s: None)
    led = SpendLedger(tmp_path / "l.json", 10)
    http = FlakyHTTP(fail_first=99, status=404)
    c = WebSearchClient(
        "firecrawl", tmp_path / "ws", api_key="k", ledger=led, http=http, min_interval_s=0
    )
    with pytest.raises(RuntimeError):
        c.search("WW GRAINGER")
    assert http.calls == 1 and led.total == 0.0
