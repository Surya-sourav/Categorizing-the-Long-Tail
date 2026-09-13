"""Estimated API spend ledger. Every live client calls ``add`` BEFORE the request; crossing the cap
raises ``BudgetExceeded`` so the request never happens."""

from __future__ import annotations

import json
import threading
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
        self._lock = threading.RLock()

    @property
    def total(self) -> float:
        return float(sum(e["usd"] for e in self.entries))

    def by_kind(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in self.entries:
            out[e["kind"]] += e["usd"]
        return dict(out)

    def add(self, kind: str, usd: float, detail: str = "") -> None:
        with self._lock:
            if self.total + usd > self.max_usd:
                raise BudgetExceeded(
                    f"spend {self.total:.2f} + {usd:.4f} would exceed cap {self.max_usd:.2f} USD"
                )
            self.entries.append(
                {"ts": now_iso(), "kind": kind, "usd": float(usd), "detail": detail}
            )
            self._flush()

    def adjust_last(self, kind: str, detail: str, usd: float) -> None:
        """Replace the estimated cost of the most recent matching entry with the actual cost."""
        with self._lock:
            for e in reversed(self.entries):
                if e["kind"] == kind and e["detail"] == detail:
                    e["usd"] = float(usd)
                    break
            self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.entries))
        tmp.replace(self.path)
