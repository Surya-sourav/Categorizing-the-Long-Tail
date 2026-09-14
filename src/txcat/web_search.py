"""Web search evidence for a merchant name. Only the normalized merchant string is ever sent.
Every response is cached as JSON: {query, provider, timestamp, results:[{title,snippet,url}]}."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from loguru import logger

from txcat.cachepack import load_packed
from txcat.utils import now_iso, sha1_text

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_URL = "https://api.tavily.com/search"
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/search"
PROVIDERS = ("brave", "tavily", "firecrawl")
_AMOUNT_OR_DATE = re.compile(r"\$\s?\d|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")


class CacheMissError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str


class WebSearchClient:
    def __init__(
        self,
        provider: str,
        cache_dir: str | Path,
        api_key: str | None,
        ledger=None,
        http=None,
        allow_live: bool = True,
        min_interval_s: float = 1.1,
        price_per_1k: float = 5.0,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        self.provider, self.dir = provider, Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.api_key, self.ledger, self.http = api_key, ledger, http or requests
        self.allow_live, self.min_interval_s = allow_live, min_interval_s
        self.price = price_per_1k / 1000
        self._last = 0.0
        self._packed = load_packed(
            self.dir
        )  # frozen cache (packed JSONL), consulted after per-call files

    def _request(self, query: str, num_results: int):
        if self.provider == "brave":
            return self.http.get(
                BRAVE_URL,
                params={"q": query, "count": num_results},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                timeout=30,
            )
        if self.provider == "firecrawl":
            # Firecrawl v2 search: 2 credits per request (<=10 results); web source, no scraping
            return self.http.post(
                FIRECRAWL_URL,
                json={"query": query, "limit": num_results, "sources": ["web"], "country": "US"},
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
        # Tavily: JSON body, bearer auth; basic depth = 1 credit; evidence only
        return self.http.post(
            TAVILY_URL,
            json={
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=30,
        )

    def _parse(self, payload: dict, num_results: int) -> list[SearchResult]:
        if self.provider == "brave":
            raw = payload.get("web", {}).get("results", [])
            return [
                SearchResult(
                    title=x.get("title", ""), snippet=x.get("description", ""), url=x.get("url", "")
                )
                for x in raw[:num_results]
            ]
        if self.provider == "firecrawl":
            raw = (payload.get("data") or {}).get("web", []) or []
            return [
                SearchResult(
                    title=x.get("title", ""), snippet=x.get("description", ""), url=x.get("url", "")
                )
                for x in raw[:num_results]
            ]
        raw = payload.get("results", [])
        return [
            SearchResult(
                title=x.get("title", ""), snippet=x.get("content", ""), url=x.get("url", "")
            )
            for x in raw[:num_results]
        ]

    def _path(self, query: str) -> Path:
        return self.dir / f"{sha1_text(f'{self.provider}|{query}')}.json"

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        if _AMOUNT_OR_DATE.search(query):
            raise ValueError(
                "query looks like it contains an amount or date; only merchant names are allowed"
            )
        p = self._path(query)
        if p.exists() or p.stem in self._packed:
            data = json.loads(p.read_text()) if p.exists() else self._packed[p.stem]
            return [SearchResult(**r) for r in data["results"][:num_results]]
        if not self.allow_live:
            raise CacheMissError(f"no cached search for {query!r}")
        if not self.api_key:
            raise RuntimeError(f"{self.provider.upper()}_API_KEY missing")
        if self.ledger is not None:
            self.ledger.add("search", self.price, self.provider)
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r, last_exc = None, None
        for attempt in range(5):
            try:
                r = self._request(query, num_results)
                status = getattr(r, "status_code", 200)
                if status == 429 or status >= 500:  # rate limited or transient upstream failure
                    raise requests.HTTPError(f"{status} from {self.provider}", response=r)
                r.raise_for_status()
                break
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                last_exc, r = e, None
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    break  # permanent client error: do not retry
                logger.warning(f"search {query!r} attempt {attempt}: {e}")
                if attempt < 4:
                    time.sleep(3 * (attempt + 1))
        if r is None:
            if self.ledger is not None:
                self.ledger.adjust_last("search", self.provider, 0.0)  # nothing was served
            raise RuntimeError(f"search failed for {query!r}") from last_exc
        self._last = time.time()
        payload = r.json()
        results = self._parse(payload, num_results)
        credits_used = payload.get("creditsUsed") if isinstance(payload, dict) else None
        p.write_text(
            json.dumps(
                {
                    "query": query,
                    "provider": self.provider,
                    "timestamp": now_iso(),
                    "results": [asdict(x) for x in results],
                    "credits_used": credits_used,
                },
                ensure_ascii=False,
            )
        )
        logger.debug(f"search {query!r}: {len(results)} results")
        return results
