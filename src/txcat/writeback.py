"""Decide whether an LLM-resolved merchant is inserted into the index."""

from __future__ import annotations

from txcat.llm_fallback import LLMResult

MODES = ("never", "always", "confidence_gated")


class WriteBackPolicy:
    def __init__(self, mode: str = "confidence_gated", confidence_threshold: float = 0.8):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.threshold = mode, confidence_threshold

    def should_write(self, r: LLMResult) -> bool:
        if not r.valid or self.mode == "never":
            return False
        if self.mode == "always":
            return True
        return r.confidence >= self.threshold
