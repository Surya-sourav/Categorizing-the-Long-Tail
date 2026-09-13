"""LLM fallback categorizer over an OpenAI-compatible chat API (OpenAI, Together via base_url).

Cache key = sha256(model | prompt_file_hash | rendered prompt). Reproduce mode never calls the API.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from txcat.config import LLMModelCfg, rel_to_root
from txcat.utils import now_iso, sha256_text
from txcat.web_search import SearchResult

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["category", "confidence", "evidence"],
    "additionalProperties": False,
}


class CacheMissError(RuntimeError):
    pass


@dataclass
class LLMResult:
    category: str
    confidence: float
    evidence: str
    valid: bool
    model_id: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool


def render_prompt(
    template_path: str | Path,
    merchant: str,
    evidence: list[SearchResult] | None,
    taxonomy: list[str],
) -> str:
    tpl = Path(template_path).read_text()
    block = (
        "\n".join(
            f"{i}. {r.title} — {r.snippet} ({r.url})" for i, r in enumerate(evidence or [], 1)
        )
        or "(no results)"
    )
    return tpl.format(
        merchant=merchant, taxonomy="\n".join(f"- {t}" for t in taxonomy), evidence_block=block
    )


def _extract_json(text: str) -> dict:
    """Parse the model's answer. Tolerates ``` fences and reasoning preambles: the LAST parseable
    flat JSON object in the text wins (reasoning models may echo earlier braces)."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    candidates = re.findall(r"\{[^{}]*\}", text, re.S)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no JSON object found", text, 0)


class LLMFallback:
    def __init__(
        self,
        model: LLMModelCfg,
        cache_dir: str | Path,
        prompt_path: str | Path,
        ledger=None,
        client=None,
        allow_live: bool = True,
        max_completion_tokens: int = 300,
        limiter=None,
    ):
        self.m, self.dir, self.prompt_path = model, Path(cache_dir), Path(prompt_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prompt_hash = sha256_text(self.prompt_path.read_text())
        self.ledger, self._client, self.allow_live = ledger, client, allow_live
        self.max_completion_tokens = max_completion_tokens
        self.limiter = limiter  # optional shared RateLimiter (requests per minute across threads)
        self._last_call = 0.0

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get(self.m.api_key_env)
            if not key:
                raise RuntimeError(f"{self.m.api_key_env} missing")
            self._client = OpenAI(
                api_key=key, base_url=self.m.base_url, timeout=self.m.timeout_s, max_retries=0
            )
        return self._client

    def _request_kwargs(self, prompt: str) -> dict:
        kw: dict = {
            "model": self.m.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.m.reasoning_effort:
            kw["reasoning_effort"] = self.m.reasoning_effort
        if self.m.json_mode == "json_schema":
            js = {"name": "categorization", "schema": SCHEMA}
            if self.m.provider == "openai":
                js["strict"] = True  # Together rejects the strict key
            kw["response_format"] = {"type": "json_schema", "json_schema": js}
        elif self.m.json_mode == "json_object":
            kw["response_format"] = {"type": "json_object"}
        if self.m.extra_body:
            kw["extra_body"] = self.m.extra_body
        return kw

    def categorize(
        self, merchant: str, evidence: list[SearchResult] | None, taxonomy: list[str]
    ) -> LLMResult:
        prompt = render_prompt(self.prompt_path, merchant, evidence, taxonomy)
        key = sha256_text(f"{self.m.name}|{self.prompt_hash}|{prompt}")
        p = self.dir / f"{key}.json"
        if p.exists():
            d = json.loads(p.read_text())["result"]
            d["cached"] = True
            return LLMResult(**d)
        if not self.allow_live:
            raise CacheMissError(f"no cached LLM result for {self.m.name} / {merchant!r}")
        kw = self._request_kwargs(prompt)
        est_usd = (
            len(prompt) / 4
        ) / 1e6 * self.m.price_in_per_1m + 150 / 1e6 * self.m.price_out_per_1m
        if self.ledger is not None:
            self.ledger.add("llm", est_usd, self.m.name)
        wait = self.m.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        t0 = time.perf_counter()
        resp, last_exc = None, None
        for attempt in range(4):
            if self.limiter is not None:
                self.limiter.acquire()
            try:
                resp = self.client.chat.completions.create(**kw)
                break
            except Exception as e:  # noqa: BLE001
                last_exc = e
                logger.warning(
                    f"{self.m.name} attempt {attempt}: {type(e).__name__}: {str(e)[:160]}"
                )
                if attempt < 3:
                    time.sleep(2**attempt)
        if resp is None:
            raise RuntimeError(f"LLM call failed for {merchant!r}") from last_exc
        latency = (time.perf_counter() - t0) * 1000
        self._last_call = time.time()
        text = resp.choices[0].message.content or ""
        try:
            parsed = _extract_json(text)
            cat = str(parsed.get("category", "")).strip()
            valid = cat in taxonomy
            res = LLMResult(
                category=cat if valid else "INVALID",
                confidence=float(parsed.get("confidence", 0.0)),
                evidence=str(parsed.get("evidence", "")),
                valid=valid,
                model_id=getattr(resp, "model", self.m.name),
                tokens_in=int(resp.usage.prompt_tokens),
                tokens_out=int(resp.usage.completion_tokens),
                latency_ms=latency,
                cached=False,
            )
        except Exception:  # noqa: BLE001
            res = LLMResult(
                "INVALID",
                0.0,
                text[:200],
                False,
                getattr(resp, "model", self.m.name),
                int(resp.usage.prompt_tokens),
                int(resp.usage.completion_tokens),
                latency,
                False,
            )
        if self.ledger is not None:  # true-up: replace estimate with actual usage
            actual = (
                res.tokens_in / 1e6 * self.m.price_in_per_1m
                + res.tokens_out / 1e6 * self.m.price_out_per_1m
            )
            self.ledger.adjust_last("llm", self.m.name, actual)
        p.write_text(
            json.dumps(
                {
                    "model": self.m.name,
                    "prompt_hash": self.prompt_hash,
                    "prompt_file": rel_to_root(self.prompt_path),
                    "merchant": merchant,
                    "timestamp": now_iso(),
                    "request": kw,
                    "raw_response": text,
                    "result": asdict(res),
                },
                ensure_ascii=False,
            )
        )
        return res
