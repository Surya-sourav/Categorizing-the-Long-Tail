"""Typed configuration loaded from YAML (defaults deep-merged with an override file)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_PATH = Path("configs/default.yaml")


class WindowCfg(BaseModel):
    train_start: str
    train_end: str
    test_start: str
    test_end: str | None = None


class VolumeRuleCfg(BaseModel):
    min_test_txns: int = 20000
    min_tail_merchants: int = 2500
    fallback_window: WindowCfg


class DatasetCfg(BaseModel):
    name: str
    raw_dir: str
    processed_path: str
    window: WindowCfg | None = None
    volume_rule: VolumeRuleCfg | None = None
    ckan_packages: list[str] = Field(default_factory=list)


class FESCfg(BaseModel):
    n_tail: int = 2500
    n_head: int = 500
    n_oklahoma: int = 800
    pilot_n: int = 100


class EmbedCfg(BaseModel):
    cache_dir: str
    backbones: list[str]
    openai_model: str = "text-embedding-3-small"
    openai_price_per_1m: float = 0.02
    batch_size: int = 256


class IndexCfg(BaseModel):
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    k: int = 5


class GateCfg(BaseModel):
    mode: str = "threshold"
    threshold: float = 0.65


class BudgetCfg(BaseModel):
    max_usd: float = 40.0
    ledger_path: str = "cache/spend_ledger.json"


class SearchCfg(BaseModel):
    provider: str = "brave"
    cache_dir: str = "cache/web_search"
    num_results: int = 5
    price_per_1k_usd: float = 5.0
    min_interval_s: float = 1.1
    api_key_env: str = "BRAVE_API_KEY"


class LLMModelCfg(BaseModel):
    name: str
    provider: str
    api_key_env: str
    base_url: str | None = None
    price_in_per_1m: float
    price_out_per_1m: float
    reasoning_effort: str | None = None
    json_mode: str = "json_schema"  # json_schema | json_object | none


class LLMCfg(BaseModel):
    cache_dir: str
    prompt_with_web: str
    prompt_no_web: str
    max_completion_tokens: int = 300
    models: list[LLMModelCfg]


class Config(BaseModel):
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, default_path: Path = DEFAULT_PATH) -> Config:
    """Load defaults, deep-merge ``path`` on top (if given and different), validate."""
    base = yaml.safe_load(default_path.read_text()) or {}
    if path is not None and Path(path).resolve() != default_path.resolve():
        override = yaml.safe_load(Path(path).read_text()) or {}
        base = _deep_merge(base, override)
    return Config.model_validate(base)
