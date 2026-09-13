"""Ablation only: let the model run its own web searches (OpenAI Responses API `web_search` tool).
Cached like everything else; only the normalized merchant string enters the prompt."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from txcat.utils import now_iso, sha256_text

PROMPT = """You categorize the merchant behind a card transaction descriptor. Use web search to
find out what the merchant is, then answer.

Merchant descriptor (normalized): {merchant}

Choose exactly one category from this list:
{taxonomy}

Return JSON only, with keys "category" (copied exactly from the list), "confidence" (0-1), and
"evidence" (one sentence)."""


class AgenticSearchCategorizer:
    def __init__(
        self,
        model: str,
        cache_dir: str | Path,
        client=None,
        price_in_per_1m: float = 0.25,
        price_out_per_1m: float = 2.0,
        search_price_per_call: float = 0.01,
        ledger=None,
        allow_live: bool = True,
    ):
        self.model, self.dir = model, Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._client, self.ledger, self.allow_live = client, ledger, allow_live
        self.p_in, self.p_out, self.p_search = (
            price_in_per_1m,
            price_out_per_1m,
            search_price_per_call,
        )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def categorize(self, merchant: str, taxonomy: list[str]) -> dict:
        prompt = PROMPT.format(merchant=merchant, taxonomy="\n".join(f"- {t}" for t in taxonomy))
        p = self.dir / f"{sha256_text(self.model + '|agentic|' + prompt)}.json"
        if p.exists():
            return json.loads(p.read_text())["result"]
        if not self.allow_live:
            raise RuntimeError(f"no cached agentic result for {merchant!r}")
        if self.ledger is not None:
            self.ledger.add("agentic_search", 0.03, self.model)
        t0 = time.perf_counter()
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            tools=[{"type": "web_search", "search_context_size": "low"}],
        )
        latency = (time.perf_counter() - t0) * 1000
        queries = [
            getattr(o, "action", {}).get("query", "")
            if isinstance(getattr(o, "action", None), dict)
            else getattr(getattr(o, "action", None), "query", "")
            for o in resp.output
            if getattr(o, "type", "") == "web_search_call"
        ]
        text = resp.output_text or ""
        m = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(m.group(0)) if m else {}
        cat = str(parsed.get("category", "")).strip()
        usd = (
            resp.usage.input_tokens / 1e6 * self.p_in
            + resp.usage.output_tokens / 1e6 * self.p_out
            + len(queries) * self.p_search
        )
        if self.ledger is not None:
            self.ledger.entries[-1]["usd"] = usd
            self.ledger.path.write_text(json.dumps(self.ledger.entries))
        result = {
            "category": cat if cat in taxonomy else "INVALID",
            "valid": cat in taxonomy,
            "confidence": float(parsed.get("confidence", 0.0)),
            "evidence": str(parsed.get("evidence", "")),
            "queries": queries,
            "n_searches": len(queries),
            "model_id": getattr(resp, "model", self.model),
            "tokens_in": resp.usage.input_tokens,
            "tokens_out": resp.usage.output_tokens,
            "latency_ms": latency,
            "usd": usd,
        }
        p.write_text(
            json.dumps(
                {
                    "merchant": merchant,
                    "model": self.model,
                    "timestamp": now_iso(),
                    "raw": text,
                    "result": result,
                }
            )
        )
        return result
