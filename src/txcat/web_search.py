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

from txcat.utils import now_iso, sha1_text

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_AMOUNT_OR_DATE = re.compile(r"\$\s?\d|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")


class CacheMissError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str


class WebSearchClient:
    def __init__(self, provider: str, cache_dir: str | Path, api_key: str | None, ledger=None,
                 http=None, allow_live: bool = True, min_interval_s: float = 1.1,
                 price_per_1k: float = 5.0):
        if provider != "brave":
            raise ValueError("only provider='brave' is implemented for evidence retrieval")
        self.provider, self.dir = provider, Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.api_key, self.ledger, self.http = api_key, ledger, http or requests
        self.allow_live, self.min_interval_s = allow_live, min_interval_s
        self.price = price_per_1k / 1000
        self._last = 0.0

    def _path(self, query: str) -> Path:
        return self.dir / f"{sha1_text(f'{self.provider}|{query}')}.json"

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        if _AMOUNT_OR_DATE.search(query):
            raise ValueError(
                "query looks like it contains an amount or date; only merchant names are allowed"
            )
        p = self._path(query)
        if p.exists():
            data = json.loads(p.read_text())
            return [SearchResult(**r) for r in data["results"][:num_results]]
        if not self.allow_live:
            raise CacheMissError(f"no cached search for {query!r}")
        if not self.api_key:
            raise RuntimeError("BRAVE_API_KEY missing")
        if self.ledger is not None:
            self.ledger.add("search", self.price, self.provider)
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            r = self.http.get(
                BRAVE_URL,
                params={"q": query, "count": num_results},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                timeout=30,
            )
            if getattr(r, "status_code", 200) == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        self._last = time.time()
        raw = r.json().get("web", {}).get("results", [])
        results = [
            SearchResult(title=x.get("title", ""), snippet=x.get("description", ""),
                         url=x.get("url", ""))
            for x in raw[:num_results]
        ]
        p.write_text(json.dumps({"query": query, "provider": self.provider, "timestamp": now_iso(),
                                 "results": [asdict(x) for x in results]}, ensure_ascii=False))
        logger.debug(f"search {query!r}: {len(results)} results")
        return results
