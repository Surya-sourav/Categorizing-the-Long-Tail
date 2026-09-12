"""Estimated API spend ledger. Every live client calls ``add`` BEFORE the request; crossing the cap
raises ``BudgetExceeded`` so the request never happens."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from txcat.utils import now_iso


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    def __init__(self, path: str | Path, max_usd: float):
        self.path, self.max_usd = Path(path), max_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    @property
    def total(self) -> float:
        return float(sum(e["usd"] for e in self.entries))

    def by_kind(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in self.entries:
            out[e["kind"]] += e["usd"]
        return dict(out)

    def add(self, kind: str, usd: float, detail: str = "") -> None:
        if self.total + usd > self.max_usd:
            raise BudgetExceeded(
                f"spend {self.total:.2f} + {usd:.4f} would exceed cap {self.max_usd:.2f} USD"
            )
        self.entries.append({"ts": now_iso(), "kind": kind, "usd": float(usd), "detail": detail})
        self.path.write_text(json.dumps(self.entries))
