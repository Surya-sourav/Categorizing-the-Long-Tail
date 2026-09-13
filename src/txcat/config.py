"""Typed configuration loaded from YAML (defaults deep-merged with an override file)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _ROOT / "configs/default.yaml"

# (section path, keys) for every path-valued field; resolved against _ROOT when relative.
_PATH_FIELDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    ((), ("results_dir", "logs_dir", "taxonomy_dir")),
    (("budget",), ("ledger_path",)),
    (("embed",), ("cache_dir",)),
    (("search",), ("cache_dir",)),
    (("llm",), ("cache_dir", "prompt_with_web", "prompt_no_web")),
)
_DATASET_PATH_KEYS = ("raw_dir", "processed_path")


class _Base(BaseModel):
    """Strict base: unknown keys are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowCfg(_Base):
    train_start: str
    train_end: str
    test_start: str
    test_end: str | None = None

    @field_validator("train_start", "train_end", "test_start", "test_end", mode="before")
    @classmethod
    def _iso_date(cls, v: Any) -> Any:
        """Accept unquoted YAML dates and validate the ISO-8601 format."""
        if v is None:
            return v
        if isinstance(v, date):
            return v.isoformat()
        date.fromisoformat(str(v))  # raises ValueError on a malformed date
        return str(v)


class VolumeRuleCfg(_Base):
    min_test_txns: int = 20000
    min_tail_merchants: int = 2500
    fallback_window: WindowCfg


class DatasetCfg(_Base):
    name: str
    raw_dir: str
    processed_path: str
    window: WindowCfg | None = None
    volume_rule: VolumeRuleCfg | None = None
    ckan_packages: list[str] = Field(default_factory=list)


class FESCfg(_Base):
    n_tail: int = 2500
    n_head: int = 500
    n_oklahoma: int = 800
    pilot_n: int = 100


class EmbedCfg(_Base):
    cache_dir: str
    backbones: list[str]
    openai_model: str = "text-embedding-3-small"
    openai_price_per_1m: float = 0.02
    batch_size: int = 256


class IndexCfg(_Base):
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    k: int = 5


class GateCfg(_Base):
    mode: Literal["threshold", "margin", "calibrated_lr"] = "threshold"
    threshold: float = 0.65


class BudgetCfg(_Base):
    max_usd: float = 40.0
    ledger_path: str = "cache/spend_ledger.json"


class SearchCfg(_Base):
    provider: Literal["brave", "tavily"] = "brave"
    cache_dir: str = "cache/web_search"
    num_results: int = 5
    price_per_1k_usd: float = 5.0
    min_interval_s: float = 1.1
    api_key_env: str = "BRAVE_API_KEY"


class LLMModelCfg(_Base):
    name: str
    provider: Literal["openai", "together", "nvidia"]
    api_key_env: str
    base_url: str | None = None
    price_in_per_1m: float
    price_out_per_1m: float
    reasoning_effort: str | None = None
    min_interval_s: float = 0.0  # client-side throttle (NVIDIA free endpoints allow ~40 RPM)
    note: str | None = None  # e.g. "free developer endpoint; prices are reference rates"
    json_mode: Literal["json_schema", "json_object", "none"] = "json_schema"


class LLMCfg(_Base):
    cache_dir: str
    prompt_with_web: str
    prompt_no_web: str
    max_completion_tokens: int = 300
    models: list[LLMModelCfg]


class Config(_Base):
    seed: int = 42
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    tail_k: int = 3
    tail_sensitivity: list[int] = Field(default_factory=lambda: [2, 5])
    results_dir: str = "results"
    logs_dir: str = "results/logs"
    taxonomy_dir: str = "data/taxonomy"
    active_dataset: str = "dc"
    datasets: dict[str, DatasetCfg]
    fes: FESCfg = FESCfg()
    embed: EmbedCfg
    index: IndexCfg = IndexCfg()
    gate: GateCfg = GateCfg()
    budget: BudgetCfg = BudgetCfg()
    search: SearchCfg = SearchCfg()
    llm: LLMCfg

    @model_validator(mode="after")
    def _active_dataset_exists(self) -> Config:
        if self.active_dataset not in self.datasets:
            raise ValueError(
                f"active_dataset {self.active_dataset!r} is not in datasets "
                f"{sorted(self.datasets)}"
            )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _anchor(value: Any) -> Any:
    """Resolve a relative path string against the repo root; leave anything else alone."""
    if not isinstance(value, str):
        return value
    p = Path(value)
    return str(p if p.is_absolute() else (_ROOT / p))


def _resolve_paths(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite every path-valued key in the merged mapping to a root-anchored absolute path."""
    out = dict(data)
    for section, keys in _PATH_FIELDS:
        target = out
        for part in section:
            child = target.get(part)
            if not isinstance(child, dict):
                target = {}
                break
            target[part] = dict(child)
            target = target[part]
        for key in keys:
            if key in target:
                target[key] = _anchor(target[key])

    datasets = out.get("datasets")
    if isinstance(datasets, dict):
        resolved: dict[str, Any] = {}
        for name, ds in datasets.items():
            if isinstance(ds, dict):
                ds = dict(ds)
                for key in _DATASET_PATH_KEYS:
                    if key in ds:
                        ds[key] = _anchor(ds[key])
            resolved[name] = ds
        out["datasets"] = resolved
    return out


def rel_to_root(p: str | Path) -> str:
    """Repo-relative POSIX string for paths written into committed artifacts (caches, manifests)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _resolve_config_path(path: str | Path) -> Path:
    """Take ``path`` as given; fall back to ``_ROOT / path`` for a relative path from elsewhere."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        rooted = _ROOT / p
        if rooted.exists():
            return rooted
    return p


def load_config(path: str | Path | None = None, default_path: Path = DEFAULT_PATH) -> Config:
    """Load defaults, deep-merge ``path`` on top (if given and different), validate.

    Mappings are merged key by key; list-valued keys (``seeds``, ``embed.backbones``,
    ``llm.models``, ``tail_sensitivity``, ``datasets.*.ckan_packages``) are **replaced
    wholesale** by the override, never element-merged or appended. Path-valued fields
    come back absolute, anchored at the repo root, so any CWD works. ``.env`` is loaded
    as a side effect so every entry point picks up API keys.
    """
    load_dotenv()
    base = yaml.safe_load(default_path.read_text()) or {}
    if path is not None:
        resolved = _resolve_config_path(path)
        if resolved.resolve() != default_path.resolve():
            override = yaml.safe_load(resolved.read_text()) or {}
            base = _deep_merge(base, override)
    return Config.model_validate(_resolve_paths(base))
