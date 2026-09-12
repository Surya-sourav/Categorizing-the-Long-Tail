# longtail-txcat Day 1: Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the reproducible data layer, taxonomy, normalizer, cached embedder, HNSW index, gate, Exp 0 and Exp 1 on the DC purchase-card data, and a 100-merchant search+LLM pilot that freezes the prompts.

**Architecture:** An installable package `txcat` under `src/` with one module per responsibility (loaders, taxonomy, splits, normalizer, embedder, index, gate, web_search, llm_fallback), a pydantic config loaded from YAML, disk caches keyed by content hash for every embedding / search / LLM call, and thin experiment scripts under `experiments/` that only orchestrate. Exp 2/3/reproduce/generator are Plan B (Day 2); Exp 4/5/Docker are Plan C.

**Tech Stack:** Python 3.12 via `uv`; pandas, pyarrow, numpy, hnswlib, sentence-transformers, scikit-learn, rapidfuzz, requests, openai (also used for Together via `base_url`), pydantic, PyYAML, loguru, matplotlib, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-12-longtail-txcat-design.md` (FINAL). Read sections 2, 3, 4 before starting.

---

## File structure (what each file owns)

```
pyproject.toml                      deps, ruff, pytest config
configs/default.yaml                every knob with defaults; configs/dc.yaml overrides window etc.
prompts/fallback_with_web.txt       LLM prompt WITH evidence block
prompts/fallback_no_web.txt         identical prompt WITHOUT evidence block
prompts/model_versions.json         pinned model ids (written by pilot)
prompts/PROMPT_HASHES.json          sha256 of each frozen prompt (written by pilot)
data/download.py                    CLI: download dc | oklahoma | mcc_codes
data/taxonomy/mcc_to_category.csv   committed MCC -> category mapping (generated once, then hand-edited)
data/taxonomy/mcc_description_aliases.csv  observed description -> mcc
data/taxonomy/unmatched_descriptions.csv   leftovers for manual review
scripts/build_taxonomy.py           generates mcc_to_category.csv from rules
scripts/build_aliases.py            generates alias file from processed data
src/txcat/__init__.py
src/txcat/utils.py                  seeds, hashing, logger, slugs, time
src/txcat/config.py                 pydantic Config + load_config (deep-merge yaml)
src/txcat/data/dc.py                ArcGIS paged download -> processed parquet (4 columns)
src/txcat/data/oklahoma.py          CKAN monthly CSVs -> processed parquet (4 columns)
src/txcat/data/schema.py            PROCESSED_COLUMNS, validate_processed()
src/txcat/data/taxonomy.py          load mapping + aliases, attach_labels(df)
src/txcat/data/splits.py            temporal split, frequencies, tail mask, volume rule, FES sampling
src/txcat/normalizer.py             normalize_merchant
src/txcat/embedder.py               Embedder + backends + disk cache
src/txcat/index.py                  HNSWIndex + QueryResult
src/txcat/gate.py                   ConfidenceGate
src/txcat/budget.py                 SpendLedger, BudgetExceeded
src/txcat/web_search.py             WebSearchClient (brave) + cache
src/txcat/llm_fallback.py           LLMFallback + prompt rendering + cache
src/txcat/knn.py                    build_merchant_index(train_df, embedder, cfg, seed), predict(test)
analysis/style.py                   matplotlib palette/style (dataviz-validated)
analysis/plots.py                   fig1..fig3 functions
experiments/exp0_data_audit.py
experiments/exp1_tail_characterization.py
experiments/pilot.py
tests/                              one test file per module
```

Processed transaction schema (every loader must emit exactly this):

| column | dtype | meaning |
|---|---|---|
| `txn_id` | str | source row id, prefixed with source (`dc:123`, `ok:AAAJ...`) |
| `date` | datetime64 (pandas 3 emits `[us]`; check with `is_datetime64_any_dtype`, never a literal `[ns]`) | transaction date |
| `raw_merchant` | str | descriptor exactly as posted |
| `mcc_description` | str | label text exactly as posted |
| `source` | str | `dc` or `ok` |

---

### Task 0: Environment and package scaffold

**Files:**
- Create: `pyproject.toml`, `src/txcat/__init__.py`, `src/txcat/utils.py`, `tests/test_utils.py`, `.env.example`, `configs/default.yaml`, `configs/dc.yaml`, `README.md` (stub)

- [ ] **Step 1: Install uv (user-space) and pin Python 3.12**

Run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.12
```
Expected: `uv 0.x.y` printed; Python 3.12 installed.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "txcat"
version = "0.1.0"
description = "Categorizing the Long Tail: web-search-augmented fallback for embedding-based transaction classification"
authors = [{ name = "Surya Parida" }]
requires-python = ">=3.12,<3.13"
dependencies = [
  "pandas>=2.2",
  "pyarrow>=17",
  "numpy>=1.26,<3",
  "hnswlib>=0.8.0",
  "sentence-transformers>=3.0",
  "torch>=2.2",
  "scikit-learn>=1.5",
  "rapidfuzz>=3.9",
  "requests>=2.32",
  "openai>=1.50",
  "pydantic>=2.8",
  "pyyaml>=6.0",
  "loguru>=0.7",
  "python-dotenv>=1.0",
  "matplotlib>=3.9",
  "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/txcat"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 3: Create venv, install, export pinned requirements**

Run:
```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
uv pip freeze > requirements.txt
.venv/bin/python -c "import hnswlib, sentence_transformers, torch; print('ok', torch.__version__)"
```
Expected: `ok 2.x.y`. If `hnswlib` fails to build, run `uv pip install hnswlib --no-binary hnswlib` and re-check (clang is present at `/Library/Developer/CommandLineTools`).

- [ ] **Step 4: Write the failing test for utils**

`tests/test_utils.py`:
```python
import random

import numpy as np

from txcat.utils import model_slug, set_seed, sha1_text, sha256_text


def test_set_seed_is_reproducible():
    set_seed(7)
    a = (random.random(), np.random.rand())
    set_seed(7)
    b = (random.random(), np.random.rand())
    assert a == b


def test_hashes_are_stable():
    assert sha1_text("x") == "11f6ad8ec52a2984abaafd7c3b516503785c2072"
    assert len(sha1_text("x")) == 40
    assert len(sha256_text("x")) == 64
    assert sha1_text("a") != sha1_text("b")


def test_model_slug():
    slug = model_slug("sentence-transformers/all-MiniLM-L6-v2")
    assert slug == "sentence-transformers__all-MiniLM-L6-v2"
    assert model_slug("text-embedding-3-small") == "text-embedding-3-small"
```

- [ ] **Step 5: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'txcat.utils'`

- [ ] **Step 6: Implement `src/txcat/__init__.py` and `src/txcat/utils.py`**

`src/txcat/__init__.py`:
```python
"""txcat: long-tail transaction categorization research pipeline."""

__version__ = "0.1.0"
```

`src/txcat/utils.py`:
```python
"""Shared helpers: seeding, hashing, logging, slugs, timestamps."""

from __future__ import annotations

import hashlib
import os
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (if present) and PYTHONHASHSEED for determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def sha1_text(text: str) -> str:
    """Stable SHA-1 hex digest of a UTF-8 string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Stable SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for a model id (``org/name`` -> ``org__name``)."""
    return model_name.replace("/", "__").replace(":", "_")


def now_iso() -> str:
    """UTC timestamp in ISO-8601 with seconds precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(log_dir: str | Path, name: str) -> Path:
    """Add a file sink under ``log_dir`` and return the log path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(path, level="INFO", enqueue=True)
    return path
```

- [ ] **Step 7: Run tests, verify pass**

Run: `.venv/bin/pytest tests/test_utils.py -v`
Expected: 3 passed.

- [ ] **Step 8: Write configs, env example, README stub**

`configs/default.yaml`:
```yaml
seed: 42
seeds: [42, 43, 44]
tail_k: 3
tail_sensitivity: [2, 5]
results_dir: results
logs_dir: results/logs
taxonomy_dir: data/taxonomy

datasets:
  dc:
    name: dc
    raw_dir: data/raw/dc
    processed_path: data/processed/dc.parquet
    window:
      train_start: "2019-01-01"
      train_end: "2023-12-31"
      test_start: "2024-01-01"
      test_end: null
    volume_rule:
      min_test_txns: 20000
      min_tail_merchants: 2500
      fallback_window:
        train_start: "2019-01-01"
        train_end: "2022-12-31"
        test_start: "2023-01-01"
        test_end: null
  oklahoma:
    name: oklahoma
    raw_dir: data/raw/oklahoma
    processed_path: data/processed/oklahoma.parquet
    ckan_packages: [purchase-card-pcard-fiscal-year-2023, purchase-card-pcard-fiscal-year-2024, purchase-card-pcard-fiscal-year-2025]

fes:
  n_tail: 2500
  n_head: 500
  n_oklahoma: 800
  pilot_n: 100

embed:
  cache_dir: cache/embeddings
  backbones:
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-small-en-v1.5
  openai_model: text-embedding-3-small
  openai_price_per_1m: 0.02
  batch_size: 256

index:
  M: 32
  ef_construction: 200
  ef_search: 64
  k: 5

gate:
  mode: threshold
  threshold: 0.65

budget:
  max_usd: 40.0
  ledger_path: cache/spend_ledger.json

search:
  provider: brave
  cache_dir: cache/web_search
  num_results: 5
  price_per_1k_usd: 5.0
  min_interval_s: 1.1
  api_key_env: BRAVE_API_KEY

llm:
  cache_dir: cache/llm
  prompt_with_web: prompts/fallback_with_web.txt
  prompt_no_web: prompts/fallback_no_web.txt
  max_completion_tokens: 300
  models:
    - name: gpt-5-nano
      provider: openai
      api_key_env: OPENAI_API_KEY
      price_in_per_1m: 0.05
      price_out_per_1m: 0.40
      reasoning_effort: minimal
      json_mode: json_schema
    - name: gpt-5-mini
      provider: openai
      api_key_env: OPENAI_API_KEY
      price_in_per_1m: 0.25
      price_out_per_1m: 2.00
      reasoning_effort: minimal
      json_mode: json_schema
    - name: Qwen/Qwen3.5-9B
      provider: together
      base_url: https://api.together.xyz/v1
      api_key_env: TOGETHER_API_KEY
      price_in_per_1m: 0.17
      price_out_per_1m: 0.25
      reasoning_effort: null
      json_mode: json_schema
```

`configs/dc.yaml`:
```yaml
# Primary dataset run. Window is verified/possibly slid by exp0 (see results/window_decision.json).
active_dataset: dc
```

`.env.example`:
```
OPENAI_API_KEY=
BRAVE_API_KEY=
TOGETHER_API_KEY=
```

`README.md` (stub, expanded in Plan B):
```markdown
# longtail-txcat

Code and frozen caches for *Categorizing the Long Tail: An Empirical Study of Web-Search-Augmented
Fallback for Embedding-Based Transaction Classification* (Surya Parida, Independent Researcher).

Setup: `curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv --python 3.12 .venv && uv pip install -e ".[dev]"`
Data: `.venv/bin/python data/download.py dc oklahoma mcc_codes` (datasets are not redistributed).
Tests: `.venv/bin/pytest`.
```

- [ ] **Step 9: Extend `.gitignore`** (large or unfrozen artifacts; frozen caches are committed in Plan B)

Append to `.gitignore`:
```
cache/embeddings/*/vectors.npy
results/*.parquet
results/index/
```

- [ ] **Step 10: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt src/txcat tests/test_utils.py configs .env.example README.md
git commit -m "chore: scaffold txcat package, configs, utils"
```

---

### Task 1: Config loading

**Files:**
- Create: `src/txcat/config.py`, `tests/test_config.py`

- [ ] **Step 1: Failing test**

`tests/test_config.py`:
```python
from txcat.config import load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.tail_k == 3
    assert cfg.datasets["dc"].window.train_start == "2019-01-01"
    assert cfg.llm.models[0].name == "gpt-5-nano"
    assert cfg.budget.max_usd == 40.0


def test_override_merges(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text("tail_k: 5\nindex:\n  M: 16\n")
    cfg = load_config(str(p))
    assert cfg.tail_k == 5
    assert cfg.index.M == 16
    assert cfg.index.ef_construction == 200  # untouched default survives deep merge
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/bin/pytest tests/test_config.py -v` -> FAIL `No module named 'txcat.config'`

- [ ] **Step 3: Implement `src/txcat/config.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass** — `.venv/bin/pytest tests/test_config.py -v` -> 2 passed
- [ ] **Step 5: Commit** — `git add src/txcat/config.py tests/test_config.py && git commit -m "feat: typed yaml config with deep merge"`

---

### Task 2: Processed schema + DC loader (ArcGIS paged download, PII dropped at the boundary)

**Files:**
- Create: `src/txcat/data/__init__.py`, `src/txcat/data/schema.py`, `src/txcat/data/dc.py`, `tests/test_data_dc.py`

- [ ] **Step 1: Failing tests**

`tests/test_data_dc.py`:
```python
import pandas as pd
import pytest

from txcat.data.dc import features_to_frame, build_query_params
from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

FAKE_FEATURES = [
    {"attributes": {"OBJECTID": 1, "AGENCY": "Office of Latino Affairs", "TRANSACTION_DATE": 1704067200000,
                    "TRANSACTION_AMOUNT": 16.8, "VENDOR_NAME": "USPS 1050050275    QQQ",
                    "VENDOR_STATE_PROVINCE": "DC", "MCC_DESCRIPTION": "Postage Services-Government Only"}},
    {"attributes": {"OBJECTID": 2, "AGENCY": "DDOT", "TRANSACTION_DATE": 1704153600000,
                    "TRANSACTION_AMOUNT": 229.5, "VENDOR_NAME": "WW GRAINGER 912",
                    "VENDOR_STATE_PROVINCE": "DC", "MCC_DESCRIPTION": "Industrial Supplies, Not Elsewhere Classified"}},
    {"attributes": {"OBJECTID": 3, "AGENCY": "DDOT", "TRANSACTION_DATE": None,
                    "TRANSACTION_AMOUNT": 1.0, "VENDOR_NAME": None, "VENDOR_STATE_PROVINCE": "DC",
                    "MCC_DESCRIPTION": "X"}},
]


def test_features_to_frame_drops_pii_and_bad_rows():
    df = features_to_frame(FAKE_FEATURES)
    assert list(df.columns) == PROCESSED_COLUMNS
    assert len(df) == 2  # row with null date / vendor dropped
    assert df.loc[0, "txn_id"] == "dc:1"
    assert df.loc[0, "raw_merchant"] == "USPS 1050050275    QQQ"
    assert df.loc[0, "source"] == "dc"
    assert str(df.loc[0, "date"].date()) == "2024-01-01"
    for forbidden in ("AGENCY", "TRANSACTION_AMOUNT", "VENDOR_STATE_PROVINCE"):
        assert forbidden not in df.columns
    validate_processed(df)  # does not raise


def test_validate_processed_rejects_extra_columns():
    df = features_to_frame(FAKE_FEATURES)
    df["amount"] = 1.0
    with pytest.raises(ValueError):
        validate_processed(df)


def test_build_query_params_pages_and_filters():
    p = build_query_params(start_date="2019-01-01", offset=3000, page_size=1000)
    assert p["where"] == "TRANSACTION_DATE >= DATE '2019-01-01'"
    assert p["resultOffset"] == 3000 and p["resultRecordCount"] == 1000
    assert p["outFields"] == "OBJECTID,TRANSACTION_DATE,VENDOR_NAME,MCC_DESCRIPTION"
    assert p["orderByFields"] == "OBJECTID ASC"
```

- [ ] **Step 2: Run, verify fail** — `.venv/bin/pytest tests/test_data_dc.py -v` -> FAIL (module missing)

- [ ] **Step 3: Implement schema and loader**

`src/txcat/data/__init__.py`: empty docstring file `"""Dataset loaders. Every loader emits the processed schema and drops PII at the boundary."""`

`src/txcat/data/schema.py`:
```python
"""The one processed transaction schema every loader must emit."""

from __future__ import annotations

import pandas as pd

PROCESSED_COLUMNS = ["txn_id", "date", "raw_merchant", "mcc_description", "source"]


def validate_processed(df: pd.DataFrame) -> None:
    """Raise ValueError unless ``df`` has exactly the processed columns with sane dtypes."""
    if list(df.columns) != PROCESSED_COLUMNS:
        raise ValueError(f"processed columns must be exactly {PROCESSED_COLUMNS}, got {list(df.columns)}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("date must be datetime64")
    if df["raw_merchant"].isna().any() or df["date"].isna().any():
        raise ValueError("raw_merchant and date must be non-null")
    if df["txn_id"].duplicated().any():
        raise ValueError("txn_id must be unique")
```

`src/txcat/data/dc.py`:
```python
"""Washington DC Purchase Card Transactions loader (ArcGIS REST, paged, 2019+).

Only OBJECTID, TRANSACTION_DATE, VENDOR_NAME, MCC_DESCRIPTION are ever requested; agency, amount and
vendor state never leave the API response.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

LAYER_URL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
    "Public_Service_WebMercator/MapServer/50/query"
)
OUT_FIELDS = "OBJECTID,TRANSACTION_DATE,VENDOR_NAME,MCC_DESCRIPTION"
PAGE_SIZE = 1000  # layer maxRecordCount


def build_query_params(start_date: str, offset: int, page_size: int = PAGE_SIZE) -> dict:
    """ArcGIS query params for one page of rows on/after ``start_date`` (YYYY-MM-DD)."""
    return {
        "where": f"TRANSACTION_DATE >= DATE '{start_date}'",
        "outFields": OUT_FIELDS,
        "returnGeometry": "false",
        "orderByFields": "OBJECTID ASC",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "f": "json",
    }


def features_to_frame(features: list[dict]) -> pd.DataFrame:
    """Convert ArcGIS features to the processed schema, dropping PII and null rows."""
    rows = []
    for f in features:
        a = f["attributes"]
        if a.get("TRANSACTION_DATE") is None or not a.get("VENDOR_NAME"):
            continue
        rows.append(
            {
                "txn_id": f"dc:{a['OBJECTID']}",
                "date": pd.to_datetime(a["TRANSACTION_DATE"], unit="ms").normalize(),
                "raw_merchant": str(a["VENDOR_NAME"]),
                "mcc_description": str(a.get("MCC_DESCRIPTION") or ""),
                "source": "dc",
            }
        )
    df = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def download_dc(raw_dir: str | Path, start_date: str = "2019-01-01", sleep_s: float = 0.2) -> Path:
    """Page through the layer and checkpoint each page to ``raw_dir`` as JSON. Resumable."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    session = requests.Session()
    while True:
        page_path = raw_dir / f"page_{offset:07d}.json"
        if page_path.exists():
            feats = json.loads(page_path.read_text()).get("features", [])
        else:
            for attempt in range(5):
                try:
                    r = session.get(LAYER_URL, params=build_query_params(start_date, offset), timeout=60)
                    r.raise_for_status()
                    payload = r.json()
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    break
                except Exception as e:  # noqa: BLE001
                    wait = 2**attempt
                    logger.warning(f"DC page {offset} failed ({e}); retry in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"DC download failed at offset {offset}")
            page_path.write_text(json.dumps(payload))
            feats = payload.get("features", [])
            time.sleep(sleep_s)
        logger.info(f"DC offset {offset}: {len(feats)} features")
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return raw_dir


def build_processed(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Concatenate checkpointed pages into the processed parquet."""
    frames = [features_to_frame(json.loads(p.read_text()).get("features", []))
              for p in sorted(Path(raw_dir).glob("page_*.json"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("txn_id").sort_values("date")
    df = df.reset_index(drop=True)
    validate_processed(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"DC processed: {len(df)} rows -> {out_path}")
    return df
```

- [ ] **Step 4: Run tests, verify pass** — 3 passed
- [ ] **Step 5: Commit** — `git add src/txcat/data tests/test_data_dc.py && git commit -m "feat: DC ArcGIS loader with PII-dropping processed schema"`

---

### Task 3: Oklahoma loader (CKAN monthly CSVs)

> **Implemented with one approved deviation (2026-09-13):** Oklahoma `ROWID` is an Oracle rowid the publisher reuses across monthly extracts for different transactions (28k collisions between two adjacent months), and `PCard_Public_202308/202309.csv` have no `ROWID` column. The shipped loader therefore emits `txn_id = ok:<file stem>:<ROWID>`, falling back to `ok:<file stem>:row<pos>` when the column is absent, and is BOM-tolerant. The bare-`ROWID` behaviour below is kept when no `file_id` is passed, so the test in Step 1 is unchanged. Real result: 1,256,589 rows (the plan's dedupe-on-bare-ROWID would have kept only 358,170).

**Files:**
- Create: `src/txcat/data/oklahoma.py`, `tests/test_data_oklahoma.py`

- [ ] **Step 1: Failing test**

`tests/test_data_oklahoma.py`:
```python
import io

import pandas as pd

from txcat.data.oklahoma import csv_to_frame
from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

CSV = """CALENDAR_YEAR,CALENDAR_MONTH,AGENCYNBR,AGENCYNAME,LAST_NAME,FIRST_INITIAL,ITEM_DESCR,AMOUNT,MERCHANT,TRANSACTION_DATE,POST_DATE,MCC_DESCRIPTION,ROWID
2024,07, 01000,OKLAHOMA STATE UNIVERSITY,Tivis,J,Bar Audio Wired In-Ear Hea PCE,24.99,AMZN Mktp US RC9J295A2,29-Jun-24,01-Jul-24,BOOK STORES,AAAJGhAANAANWuXAAO
2024,07, 01000,OKLAHOMA STATE UNIVERSITY,Wadley,M,GENERAL PURCHASE,49.98,WAL-MART #4241,28-Jun-24,01-Jul-24,GROCERY STORES  SUPERMARKETS,AAAJGhAANAANWuXAAU
2024,07, 01000,OKLAHOMA STATE UNIVERSITY,Nobody,X,,1.00,,28-Jun-24,01-Jul-24,X,AAAJGhAANAANWuXAAZ
"""


def test_csv_to_frame_keeps_only_processed_columns():
    df = csv_to_frame(io.StringIO(CSV))
    assert list(df.columns) == PROCESSED_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "txn_id"] == "ok:AAAJGhAANAANWuXAAO"
    assert df.loc[0, "raw_merchant"] == "AMZN Mktp US RC9J295A2"
    assert str(df.loc[0, "date"].date()) == "2024-06-29"
    assert df.loc[1, "mcc_description"] == "GROCERY STORES  SUPERMARKETS"
    validate_processed(df)
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/data/oklahoma.py`**

```python
"""Oklahoma State PCard loader (CKAN monthly CSVs). Cardholder, amount, item and agency are dropped."""

from __future__ import annotations

import time
from pathlib import Path
from typing import IO

import pandas as pd
import requests
from loguru import logger

from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

CKAN_SHOW = "https://data.ok.gov/api/3/action/package_show?id={pkg}"
KEEP = ["ROWID", "TRANSACTION_DATE", "MERCHANT", "MCC_DESCRIPTION"]


def csv_to_frame(src: str | Path | IO[str]) -> pd.DataFrame:
    """Read one monthly CSV, keep only the four needed columns, emit the processed schema."""
    raw = pd.read_csv(src, usecols=KEEP, dtype=str, keep_default_na=False)
    raw = raw[(raw["MERCHANT"].str.strip() != "") & (raw["TRANSACTION_DATE"].str.strip() != "")]
    df = pd.DataFrame(
        {
            "txn_id": "ok:" + raw["ROWID"].str.strip(),
            "date": pd.to_datetime(raw["TRANSACTION_DATE"].str.strip(), format="%d-%b-%y", errors="coerce"),
            "raw_merchant": raw["MERCHANT"].str.strip(),
            "mcc_description": raw["MCC_DESCRIPTION"].str.strip(),
            "source": "ok",
        }
    )[PROCESSED_COLUMNS]
    df = df[df["date"].notna()].reset_index(drop=True)
    return df


def list_resource_urls(packages: list[str]) -> list[tuple[str, str]]:
    """Return (name, url) of every CSV resource across the CKAN packages."""
    out = []
    for pkg in packages:
        r = requests.get(CKAN_SHOW.format(pkg=pkg), timeout=60)
        r.raise_for_status()
        for res in r.json()["result"]["resources"]:
            if res.get("format", "").upper() == "CSV":
                out.append((res["name"], res["url"]))
    return out


def download_oklahoma(raw_dir: str | Path, packages: list[str]) -> Path:
    """Download every monthly CSV into ``raw_dir`` (skips files already present)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in list_resource_urls(packages):
        dest = raw_dir / (Path(name).stem + ".csv")
        if dest.exists():
            continue
        for attempt in range(5):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                dest.write_bytes(r.content)
                logger.info(f"OK downloaded {dest.name} ({len(r.content)/1e6:.1f} MB)")
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{name} failed ({e}); retry")
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Oklahoma download failed: {name}")
    return raw_dir


def build_processed(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Concatenate all monthly CSVs to the processed parquet."""
    frames = [csv_to_frame(p) for p in sorted(Path(raw_dir).glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("txn_id").sort_values("date")
    df = df.reset_index(drop=True)
    validate_processed(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Oklahoma processed: {len(df)} rows -> {out_path}")
    return df
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git commit -am "feat: Oklahoma CKAN loader"` (after `git add`)

---

### Task 4: `data/download.py` CLI and first real download

**Files:**
- Create: `data/download.py`, `data/raw/.gitkeep`, `data/processed/.gitkeep`

- [ ] **Step 1: Write the CLI**

`data/download.py`:
```python
"""Download raw datasets (never redistributed) and build processed parquets.

Usage: python data/download.py [dc] [oklahoma] [mcc_codes] [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from loguru import logger

from txcat.config import load_config
from txcat.data import dc, oklahoma

MCC_CODES_URL = "https://raw.githubusercontent.com/greggles/mcc-codes/main/mcc_codes.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", choices=["dc", "oklahoma", "mcc_codes"])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if "mcc_codes" in args.targets:
        dest = Path(cfg.taxonomy_dir) / "mcc_codes_source.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(requests.get(MCC_CODES_URL, timeout=60).content)
        logger.info(f"mcc codes -> {dest}")
    if "dc" in args.targets:
        d = cfg.datasets["dc"]
        dc.download_dc(d.raw_dir, start_date=d.window.train_start)
        dc.build_processed(d.raw_dir, d.processed_path)
    if "oklahoma" in args.targets:
        d = cfg.datasets["oklahoma"]
        oklahoma.download_oklahoma(d.raw_dir, d.ckan_packages)
        oklahoma.build_processed(d.raw_dir, d.processed_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the downloads** (DC is ~280 pages at 1000 rows; ~3–5 min. Oklahoma 36 files ~280 MB.)

Run: `.venv/bin/python data/download.py mcc_codes dc oklahoma`
Expected: logs ending with `DC processed: ~280000 rows` and `Oklahoma processed: ~1.2M rows`; files `data/processed/dc.parquet`, `data/processed/oklahoma.parquet`, `data/taxonomy/mcc_codes_source.csv`.

- [ ] **Step 3: Sanity check**

Run:
```bash
.venv/bin/python -c "
import pandas as pd
for n in ['dc','oklahoma']:
    df=pd.read_parquet(f'data/processed/{n}.parquet'); print(n, len(df), df.date.min().date(), df.date.max().date(), df.raw_merchant.nunique(), 'unique raw merchants')"
```
Expected: dc ~280k rows, 2019-01-01..2026-07-31; oklahoma ~1.2M rows.

- [ ] **Step 4: Commit** — `git add data/download.py data/raw/.gitkeep data/processed/.gitkeep data/taxonomy/mcc_codes_source.csv && git commit -m "feat: dataset download CLI"`

---

### Task 5: MCC -> 15-category taxonomy (committed, rule-generated, inspectable)

**Files:**
- Create: `scripts/build_taxonomy.py`, `src/txcat/data/taxonomy.py` (constants only for now), `tests/test_taxonomy.py`, output `data/taxonomy/mcc_to_category.csv`

Category slugs (15) + `AMBIGUOUS`:
`groceries`, `restaurants`, `retail`, `office_supplies`, `software_electronics`, `telecom_utilities`, `airlines_travel`, `lodging`, `transport_auto_fuel`, `industrial_hardware`, `professional_services`, `health`, `education_gov_membership`, `entertainment_media`, `financial_postal_shipping`.

- [ ] **Step 1: Failing test**

`tests/test_taxonomy.py`:
```python
import pandas as pd

from txcat.data.taxonomy import CATEGORIES, AMBIGUOUS_MCCS, categorize_mcc


def test_fifteen_categories():
    assert len(CATEGORIES) == 15
    assert "AMBIGUOUS" not in CATEGORIES


def test_spot_checks():
    assert categorize_mcc(5411) == ("groceries", 0)
    assert categorize_mcc(5812) == ("restaurants", 0)
    assert categorize_mcc(7011) == ("lodging", 0)
    assert categorize_mcc(3501) == ("lodging", 0)        # Holiday Inn brand MCC
    assert categorize_mcc(3000) == ("airlines_travel", 0)  # United brand MCC
    assert categorize_mcc(3351) == ("transport_auto_fuel", 0)  # car rental brand MCC
    assert categorize_mcc(5734) == ("software_electronics", 0)
    assert categorize_mcc(4814) == ("telecom_utilities", 0)
    assert categorize_mcc(5943) == ("office_supplies", 0)
    assert categorize_mcc(5085) == ("industrial_hardware", 0)
    assert categorize_mcc(8062) == ("health", 0)
    assert categorize_mcc(8220) == ("education_gov_membership", 0)
    assert categorize_mcc(9402) == ("financial_postal_shipping", 0)
    assert categorize_mcc(7829) == ("entertainment_media", 0)
    for m in (5999, 5399, 7399, 8999):
        cat, amb = categorize_mcc(m)
        assert amb == 1 and m in AMBIGUOUS_MCCS


def test_every_source_mcc_is_mapped():
    src = pd.read_csv("data/taxonomy/mcc_codes_source.csv", dtype={"mcc": str})
    for m in src["mcc"]:
        cat, amb = categorize_mcc(int(m))
        assert cat in CATEGORIES or (cat == "AMBIGUOUS" and amb == 1)
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/data/taxonomy.py` (rules) and the build script**

`src/txcat/data/taxonomy.py`:
```python
"""MCC -> macro category rules and label attachment.

The committed artifact is data/taxonomy/mcc_to_category.csv (generated by scripts/build_taxonomy.py
from the rules below, then hand-reviewed). Ranges are the default; OVERRIDES win.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CATEGORIES = [
    "groceries", "restaurants", "retail", "office_supplies", "software_electronics",
    "telecom_utilities", "airlines_travel", "lodging", "transport_auto_fuel", "industrial_hardware",
    "professional_services", "health", "education_gov_membership", "entertainment_media",
    "financial_postal_shipping",
]

# "we don't know" codes: excluded from headline tables, reported as their own slice
AMBIGUOUS_MCCS = {5999, 5399, 7399, 8999}

# (lo, hi, category) inclusive; first match wins after OVERRIDES
RANGES = [
    (1, 1499, "industrial_hardware"),          # agricultural services, contractors
    (1500, 2999, "industrial_hardware"),       # contractors, misc manufacturing
    (3000, 3299, "airlines_travel"),           # airline brand codes
    (3300, 3499, "transport_auto_fuel"),       # car rental brand codes
    (3500, 3999, "lodging"),                   # hotel brand codes
    (4000, 4799, "transport_auto_fuel"),       # transportation
    (4800, 4999, "telecom_utilities"),
    (5000, 5199, "industrial_hardware"),       # wholesale durable/nondurable
    (5200, 5299, "industrial_hardware"),       # home supply, lumber, hardware
    (5300, 5399, "retail"),
    (5400, 5499, "groceries"),
    (5500, 5599, "transport_auto_fuel"),       # auto dealers, service stations
    (5600, 5699, "retail"),                    # apparel
    (5700, 5799, "retail"),                    # home furnishings
    (5800, 5899, "restaurants"),
    (5900, 5999, "retail"),                    # misc retail
    (6000, 6999, "financial_postal_shipping"),
    (7000, 7099, "lodging"),
    (7200, 7299, "professional_services"),     # personal services
    (7300, 7499, "professional_services"),     # business services
    (7500, 7599, "transport_auto_fuel"),       # auto rental/repair
    (7600, 7699, "industrial_hardware"),       # repair services
    (7800, 7999, "entertainment_media"),
    (8000, 8099, "health"),
    (8100, 8199, "professional_services"),     # legal
    (8200, 8399, "education_gov_membership"),
    (8600, 8699, "education_gov_membership"),  # membership orgs
    (8700, 8999, "professional_services"),     # engineering, accounting
    (9000, 9999, "education_gov_membership"),  # government
]

OVERRIDES = {
    # office / printing
    2741: "office_supplies", 2791: "office_supplies", 5021: "office_supplies", 5044: "office_supplies",
    5111: "office_supplies", 5192: "office_supplies", 5943: "office_supplies", 5978: "office_supplies",
    7338: "office_supplies",
    # software / computers / electronics
    4816: "software_electronics", 5045: "software_electronics", 5065: "software_electronics",
    5732: "software_electronics", 5734: "software_electronics", 5815: "software_electronics",
    5816: "software_electronics", 5817: "software_electronics", 5818: "software_electronics",
    7371: "software_electronics", 7372: "software_electronics", 7375: "software_electronics",
    7379: "software_electronics",
    # telecom already 4800-4999; money transfer inside that range -> financial
    4829: "financial_postal_shipping",
    # shipping / postal / courier
    4214: "financial_postal_shipping", 4215: "financial_postal_shipping", 4225: "financial_postal_shipping",
    9402: "financial_postal_shipping",
    # travel agencies / cruise -> airlines_travel
    4411: "airlines_travel", 4511: "airlines_travel", 4582: "airlines_travel", 4722: "airlines_travel",
    # boats / marinas -> entertainment
    4457: "entertainment_media", 4468: "entertainment_media",
    # ambulance -> health
    4119: "health",
    # fuel dealers, petroleum -> transport_auto_fuel
    5172: "transport_auto_fuel", 5983: "transport_auto_fuel", 5013: "transport_auto_fuel",
    # retail overrides inside wholesale/industrial ranges
    5094: "retail", 5137: "retail", 5139: "retail", 5300: "retail", 5309: "retail",
    5931: "retail", 5932: "retail", 5937: "retail",
    5935: "industrial_hardware", 5933: "financial_postal_shipping",
    # health inside retail range
    5047: "health", 5122: "health", 5912: "health", 5975: "health", 5976: "health",
    # entertainment / media inside retail range
    5733: "entertainment_media", 5735: "entertainment_media", 5940: "entertainment_media",
    5941: "entertainment_media", 5945: "entertainment_media", 5946: "entertainment_media",
    5970: "entertainment_media", 5971: "entertainment_media", 5994: "entertainment_media",
    # personal services that are health/entertainment
    7221: "entertainment_media", 7298: "health", 7297: "health",
    # veterinary -> professional (animal health is not human health)
    742: "professional_services",
    # lodging-range personal services -> lodging stays for 7011/7012/7032/7033 only
    # child care, charities
    8351: "education_gov_membership", 8398: "education_gov_membership",
    # education inside 8200 already; testing labs
    8734: "professional_services",
}


def categorize_mcc(mcc: int) -> tuple[str, int]:
    """Return (category, ambiguous_flag) for an MCC using OVERRIDES then RANGES."""
    if mcc in AMBIGUOUS_MCCS:
        return "AMBIGUOUS", 1
    if mcc in OVERRIDES:
        return OVERRIDES[mcc], 0
    for lo, hi, cat in RANGES:
        if lo <= mcc <= hi:
            return cat, 0
    return "AMBIGUOUS", 1


def load_mapping(taxonomy_dir: str | Path) -> pd.DataFrame:
    """Load the committed mcc_to_category.csv."""
    return pd.read_csv(Path(taxonomy_dir) / "mcc_to_category.csv", dtype={"mcc": int})
```

`scripts/build_taxonomy.py`:
```python
"""Generate data/taxonomy/mcc_to_category.csv from the rules in txcat.data.taxonomy."""

from pathlib import Path

import pandas as pd

from txcat.data.taxonomy import categorize_mcc

src = pd.read_csv("data/taxonomy/mcc_codes_source.csv", dtype={"mcc": str})
rows = []
for _, r in src.iterrows():
    mcc = int(r["mcc"])
    cat, amb = categorize_mcc(mcc)
    note = "NEC / we-don't-know code" if amb else ""
    rows.append({"mcc": mcc, "mcc_description": r["edited_description"], "category": cat,
                 "ambiguous": amb, "notes": note})
out = pd.DataFrame(rows).sort_values("mcc")
Path("data/taxonomy").mkdir(parents=True, exist_ok=True)
out.to_csv("data/taxonomy/mcc_to_category.csv", index=False)
print(out["category"].value_counts())
```

- [ ] **Step 4: Generate the CSV and run tests**

Run: `.venv/bin/python scripts/build_taxonomy.py && .venv/bin/pytest tests/test_taxonomy.py -v`
Expected: value counts printed, 3 passed.

- [ ] **Step 5: Eyeball the mapping** — `column -s, -t < data/taxonomy/mcc_to_category.csv | less` and fix any obviously wrong OVERRIDES; re-run step 4.

- [ ] **Step 6: Commit** — `git add scripts/build_taxonomy.py src/txcat/data/taxonomy.py tests/test_taxonomy.py data/taxonomy/mcc_to_category.csv && git commit -m "feat: committed MCC->15-category taxonomy with ambiguous flags"`

---

### Task 6: Description aliases and `attach_labels`

**Files:**
- Modify: `src/txcat/data/taxonomy.py` (append), Create: `scripts/build_aliases.py`, `tests/test_aliases.py`; input (already committed): `data/taxonomy/manual_aliases.csv`; outputs `data/taxonomy/mcc_description_aliases.csv`, `data/taxonomy/unmatched_descriptions.csv`

- [ ] **Step 1: Failing test**

`tests/test_aliases.py`:
```python
import pandas as pd

from txcat.data.taxonomy import match_description, norm_desc, attach_labels

SOURCE = pd.DataFrame({
    "mcc": [5411, 5943, 3503, 5999],
    "edited_description": ["Grocery Stores, Supermarkets", "Stationery Stores, Office and School Supply Stores",
                           "Sheraton Hotels", "Miscellaneous and Specialty Retail Stores"],
    "combined_description": ["Grocery Stores, Supermarkets", "Stationery, Office & School Supply Stores",
                             "Sheraton", "Miscellaneous Retail"],
})


def test_norm_desc():
    assert norm_desc("GROCERY STORES  SUPERMARKETS") == "GROCERY STORES SUPERMARKETS"
    assert norm_desc("Stationery, Office & School Supply Stores") == "STATIONERY OFFICE SCHOOL SUPPLY STORES"


def test_match_exact_then_fuzzy():
    assert match_description("GROCERY STORES  SUPERMARKETS", SOURCE) == (5411, "exact")
    assert match_description("Stationery, Office & School Supply Stores", SOURCE) == (5943, "exact")
    assert match_description("SHERATON", SOURCE) == (3503, "exact")
    mcc, how = match_description("Stationery Office School Supply Store", SOURCE)
    assert mcc == 5943 and how == "fuzzy"
    assert match_description("TOTALLY UNKNOWN THING", SOURCE) == (None, "none")


def test_attach_labels(tmp_path):
    (tmp_path / "mcc_to_category.csv").write_text(
        "mcc,mcc_description,category,ambiguous,notes\n5411,Grocery,groceries,0,\n5999,Misc,AMBIGUOUS,1,x\n")
    (tmp_path / "mcc_description_aliases.csv").write_text(
        "source,observed_description,mcc,match\ndc,GROCERY STORES,5411,exact\ndc,MISC RETAIL,5999,exact\n")
    df = pd.DataFrame({"mcc_description": ["GROCERY STORES", "MISC RETAIL", "???"], "source": ["dc"] * 3})
    out = attach_labels(df, tmp_path)
    assert list(out["category"][:2]) == ["groceries", "AMBIGUOUS"] and pd.isna(out["category"].iloc[2])
    assert list(out["ambiguous"]) == [0, 1, 1]
    assert list(out["mcc"].fillna(-1).astype(int)) == [5411, 5999, -1]
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Append to `src/txcat/data/taxonomy.py`** (move the `import re` / `rapidfuzz` lines up to the module's import block so ruff E402 stays clean)

```python
import re

from rapidfuzz import fuzz, process

_DESC_COLS = ["edited_description", "combined_description", "usda_description", "irs_description"]


def norm_desc(s: str) -> str:
    """Uppercase, strip punctuation, collapse whitespace."""
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", str(s)).upper()
    return re.sub(r"\s+", " ", s).strip()


def match_description(observed: str, source: pd.DataFrame, fuzzy_cutoff: int = 90) -> tuple[int | None, str]:
    """Map an observed MCC description to an MCC via exact normalized match, then fuzzy (token_set)."""
    target = norm_desc(observed)
    lookup: dict[str, int] = {}
    for col in _DESC_COLS:
        if col in source.columns:
            for mcc, d in zip(source["mcc"], source[col], strict=False):
                if isinstance(d, str) and d:
                    lookup.setdefault(norm_desc(d), int(mcc))
    if target in lookup:
        return lookup[target], "exact"
    best = process.extractOne(target, list(lookup.keys()), scorer=fuzz.token_set_ratio)
    if best and best[1] >= fuzzy_cutoff:
        return lookup[best[0]], "fuzzy"
    return None, "none"


def attach_labels(df: pd.DataFrame, taxonomy_dir: str | Path) -> pd.DataFrame:
    """Add ``mcc``, ``category``, ``ambiguous`` columns using the committed alias + mapping files.

    Rows whose description has no alias get category None and ambiguous=1 (they are excluded from
    headline tables like other ambiguous rows, and counted in the coverage table).
    """
    taxonomy_dir = Path(taxonomy_dir)
    aliases = pd.read_csv(taxonomy_dir / "mcc_description_aliases.csv", dtype={"mcc": "Int64"})
    mapping = pd.read_csv(taxonomy_dir / "mcc_to_category.csv", dtype={"mcc": int})
    alias_key = aliases.set_index(["source", "observed_description"])["mcc"]
    keys = list(zip(df["source"], df["mcc_description"], strict=False))
    mcc = pd.Series([alias_key.get(k, pd.NA) for k in keys], index=df.index, dtype="Int64")
    cat_map = mapping.set_index("mcc")["category"]
    amb_map = mapping.set_index("mcc")["ambiguous"]
    out = df.copy()
    out["mcc"] = mcc
    out["category"] = [cat_map.get(int(m)) if pd.notna(m) else None for m in mcc]
    out["ambiguous"] = [int(amb_map.get(int(m), 1)) if pd.notna(m) else 1 for m in mcc]
    return out
```

`scripts/build_aliases.py`:
```python
"""Build data/taxonomy/mcc_description_aliases.csv from observed descriptions in processed data."""

import pandas as pd

from txcat.data.taxonomy import match_description, norm_desc

src = pd.read_csv("data/taxonomy/mcc_codes_source.csv", dtype={"mcc": int})
# Hand-written aliases for descriptions the automatic matcher cannot resolve (abbreviated MCC text such as
# "TRANSPRTN-SUBRBN + LOCAL COMTR PSNGR", and hotel/airline brand names mapped to the generic MCC of the
# same category). Committed and reviewed; keyed on norm_desc so spelling variants across sources match.
manual = pd.read_csv("data/taxonomy/manual_aliases.csv")
manual_map = {norm_desc(d): int(m) for d, m in zip(manual["observed_description"], manual["mcc"], strict=True)}
rows, unmatched = [], []
for name in ["dc", "oklahoma"]:
    df = pd.read_parquet(f"data/processed/{name}.parquet")
    counts = df.groupby("mcc_description").size().sort_values(ascending=False)
    for desc, n in counts.items():
        mcc, how = match_description(desc, src)
        if mcc is None and norm_desc(desc) in manual_map:
            mcc, how = manual_map[norm_desc(desc)], "manual"
        if mcc is None:
            unmatched.append({"source": df["source"].iloc[0], "observed_description": desc, "n_rows": n, "mcc": ""})
        else:
            rows.append({"source": df["source"].iloc[0], "observed_description": desc, "mcc": mcc, "match": how, "n_rows": n})
pd.DataFrame(rows).to_csv("data/taxonomy/mcc_description_aliases.csv", index=False)
pd.DataFrame(unmatched).to_csv("data/taxonomy/unmatched_descriptions.csv", index=False)
matched_rows = sum(r["n_rows"] for r in rows); un_rows = sum(u["n_rows"] for u in unmatched)
print(f"aliases: {len(rows)} matched descriptions covering {matched_rows} rows; "
      f"{len(unmatched)} unmatched covering {un_rows} rows ({100*un_rows/(matched_rows+un_rows):.2f}%)")
```

- [ ] **Step 4: Run tests, then build aliases**

Run: `.venv/bin/pytest tests/test_aliases.py -v && .venv/bin/python scripts/build_aliases.py`
Expected: 3 passed; coverage line printed. With `data/taxonomy/manual_aliases.csv` (158 hand-mapped descriptions, already committed by the controller after inspecting the real unmatched list) the unmatched row share should be about 0.1% (only `Unknown`, `OTHER FEES`, `POI FUNDING TRANSACTIONS...`, which are deliberately left unmapped). If it is higher, list the new entries in `data/taxonomy/unmatched_descriptions.csv` in your report rather than guessing at MCCs.

- [ ] **Step 5: Commit** — `git add scripts/build_aliases.py src/txcat/data/taxonomy.py tests/test_aliases.py data/taxonomy/*.csv && git commit -m "feat: description->MCC aliases and attach_labels"`

---

### Task 7: Normalizer (conservative, every rule logged)

**Files:**
- Create: `src/txcat/normalizer.py`, `tests/test_normalizer.py`

- [ ] **Step 1: Failing tests (25 real-style examples)**

`tests/test_normalizer.py`:
```python
import pytest

from txcat.normalizer import normalize_merchant

CASES = [
    # (raw, expected)
    ("STAPLES       00102186", "STAPLES"),
    ("WW GRAINGER 912", "WW GRAINGER"),
    ("USPS 1050050275    QQQ", "USPS"),
    ("AMZN Mktp US RC9J295A2", "AMZN MKTP US"),
    ("AMAZON MKTPL RC6DU71U2", "AMAZON MKTPL"),
    ("WAL-MART #4241", "WAL-MART"),
    ("WESTERN OKLAHOMA DAIRY SU", "WESTERN OKLAHOMA DAIRY SU"),
    ("SQ *BLUE BOTTLE COFFEE", "BLUE BOTTLE COFFEE"),
    ("TST* GOLDEN DRAGON - OAKLAND", "GOLDEN DRAGON - OAKLAND"),
    ("PAYPAL *EBAY", "EBAY"),
    ("PP*SAFEWAY", "SAFEWAY"),
    ("PYPL*ETSY", "ETSY"),
    ("DD *DOORDASH CHIPOTLE", "DOORDASH CHIPOTLE"),
    ("UBER *TRIP HELP.UBER.COM", "UBER TRIP"),
    ("AMAZON.COM*2K4R7 SEATTLE", "AMAZON SEATTLE"),
    ("IN *ACME WIDGETS LLC", "ACME WIDGETS LLC"),
    ("POS PURCHASE STARBUCKS 00123", "STARBUCKS"),
    ("HOME DEPOT #1234 WASHINGTON DC", "HOME DEPOT WASHINGTON DC"),
    ("SHELL OIL 57544099307", "SHELL OIL"),
    ("MARRIOTT 337W2 WASHINGTON", "MARRIOTT WASHINGTON"),
    ("7-ELEVEN 34567", "7-ELEVEN"),
    ("3M COMPANY", "3M COMPANY"),
    ("JOTFORM INC", "JOTFORM INC"),
    ("DELTA AIR 0062341234567", "DELTA AIR"),
    ("  lowe's  #02214  ", "LOWE'S"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize_cases(raw, expected):
    assert normalize_merchant(raw).text == expected


def test_steps_are_logged():
    r = normalize_merchant("SQ *BLUE BOTTLE COFFEE #12345")
    assert "strip_prefix:SQ *" in r.steps
    assert any(s.startswith("drop_store_number") for s in r.steps)


def test_empty_after_stripping_falls_back_to_uppercase_raw():
    r = normalize_merchant("PAYPAL *")
    assert r.text == "PAYPAL"


def test_never_returns_empty():
    assert normalize_merchant("   ").text == ""
    assert normalize_merchant("12345").text == "12345"
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/normalizer.py`**

```python
"""Rule-based merchant descriptor normalization. Conservative: strips processor noise and
reference codes, never maps brands. Every applied rule is recorded in ``steps`` for audit."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Processor / channel prefixes. Matched at string start, case-insensitive, optional space/asterisk.
PREFIXES = [
    "SQ *", "SQ*", "TST* ", "TST*", "PAYPAL *", "PAYPAL*", "PYPL*", "PP*", "DD *", "DD*",
    "IN *", "IN*", "CLV*", "IC*", "POS PURCHASE ", "POS ", "DBT PURCHASE ",
    "CKCD ", "PURCHASE ", "DEBIT CARD PURCHASE ",
]
_PREFIX_RE = re.compile(r"^(?:" + "|".join(re.escape(p) for p in PREFIXES) + r")", re.IGNORECASE)
_STORE_NUM_RE = re.compile(r"#\s*\d+")
_TLD_RE = re.compile(r"\.(?:COM|NET|ORG|CO|US|IO)$", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
_PAD_RE = re.compile(r"\bQQQ\b")
_WS_RE = re.compile(r"\s+")


@dataclass
class NormalizeResult:
    text: str
    steps: list[str] = field(default_factory=list)


def _is_ref_code(tok: str) -> bool:
    """Digit-heavy tokens are store/reference codes, not merchant words."""
    digits = sum(c.isdigit() for c in tok)
    letters = sum(c.isalpha() for c in tok)
    if tok.isdigit() and len(tok) >= 3:
        return True
    return digits >= 3 and letters >= 1 and "-" not in tok and "'" not in tok


def normalize_merchant(raw: str) -> NormalizeResult:
    """Normalize a raw card descriptor. Returns cleaned uppercase text plus the rules applied."""
    steps: list[str] = []
    s = raw.strip()
    if not s:
        return NormalizeResult("", steps)
    s = s.upper()
    steps.append("upper")

    m = _PREFIX_RE.match(s)
    if m:
        s = s[m.end():]
        steps.append(f"strip_prefix:{m.group(0).strip().upper()}")

    if _STORE_NUM_RE.search(s):
        s = _STORE_NUM_RE.sub(" ", s)
        steps.append("drop_store_number:#")
    if "*" in s:
        s = s.replace("*", " ")
        steps.append("replace_asterisk")
    toks_url = []
    for tok in s.split():
        if _TLD_RE.search(tok):
            if tok.count(".") >= 2 or tok.upper().startswith("WWW."):
                steps.append(f"drop_url:{tok}")
                continue  # subdomain-ish token (HELP.UBER.COM) carries no merchant identity
            tok = _TLD_RE.sub("", tok)  # AMAZON.COM -> AMAZON
            steps.append("strip_tld")
        toks_url.append(tok)
    s = " ".join(toks_url)
    if _DATE_RE.search(s):
        s = _DATE_RE.sub(" ", s)
        steps.append("drop_date")
    if _PAD_RE.search(s):
        s = _PAD_RE.sub(" ", s)
        steps.append("drop_padding:QQQ")

    toks = s.split()
    kept = [t for t in toks if not _is_ref_code(t)]
    if len(kept) != len(toks):
        steps.append(f"drop_ref_codes:{len(toks) - len(kept)}")
    s = " ".join(kept)

    s = s.strip(" *-/|,.")
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        # everything was noise; fall back to the uppercased raw minus punctuation
        s = _WS_RE.sub(" ", re.sub(r"[*#]", " ", raw.upper())).strip()
        steps.append("fallback_raw")
    return NormalizeResult(s, steps)
```

- [ ] **Step 4: Run tests, iterate until all pass** — `.venv/bin/pytest tests/test_normalizer.py -v`. If a case fails, adjust the regex for that case only; do not add brand-specific rules.
- [ ] **Step 5: Commit** — `git add src/txcat/normalizer.py tests/test_normalizer.py && git commit -m "feat: rule-based merchant normalizer with audit steps"`

---

### Task 8: Embedder with content-hash disk cache

**Files:**
- Create: `src/txcat/embedder.py`, `tests/test_embedder.py`

- [ ] **Step 1: Failing tests**

`tests/test_embedder.py`:
```python
import numpy as np
import pytest

from txcat.embedder import Embedder, FakeHashBackend, CacheMissError


def test_embed_is_normalized_and_cached(tmp_path):
    be = FakeHashBackend(dim=8)
    e = Embedder("fake/model", tmp_path, backend=be)
    v = e.embed(["STAPLES", "USPS", "STAPLES"])
    assert v.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v[0], v[2])
    assert be.calls == 1 and be.n_texts == 2  # deduped before hitting the backend

    e2 = Embedder("fake/model", tmp_path, backend=be)  # fresh instance reloads cache
    v2 = e2.embed(["USPS", "STAPLES"])
    assert be.calls == 1  # no new backend calls
    np.testing.assert_allclose(v2[1], v[0])


def test_reproduce_mode_raises_on_miss(tmp_path):
    e = Embedder("fake/model", tmp_path, backend=FakeHashBackend(8), allow_live=False)
    with pytest.raises(CacheMissError):
        e.embed(["NEVER SEEN"])


def test_cache_dir_is_per_model(tmp_path):
    Embedder("a/x", tmp_path, backend=FakeHashBackend(4)).embed(["Q"])
    Embedder("b/y", tmp_path, backend=FakeHashBackend(4)).embed(["Q"])
    assert (tmp_path / "a__x" / "vectors.npy").exists()
    assert (tmp_path / "b__y" / "vectors.npy").exists()
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/embedder.py`**

```python
"""Embedding backbones behind one interface, with an append-only float16 disk cache keyed by
sha1(text). Reproduction mode (allow_live=False) never computes a new vector."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from txcat.utils import model_slug, sha1_text


class CacheMissError(RuntimeError):
    """Raised in reproduce mode when a text is not in the embedding cache."""


class Backend(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class FakeHashBackend:
    """Deterministic pseudo-embeddings for tests."""

    def __init__(self, dim: int = 8):
        self.dim, self.calls, self.n_texts = dim, 0, 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.n_texts += len(texts)
        rows = []
        for t in texts:
            rng = np.random.default_rng(int(sha1_text(t)[:8], 16))
            rows.append(rng.standard_normal(self.dim))
        return np.asarray(rows, dtype=np.float32)


class SentenceTransformerBackend:
    """Local CPU sentence-transformers model."""

    def __init__(self, model_name: str, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 5000,
                              convert_to_numpy=True, normalize_embeddings=False),
            dtype=np.float32,
        )


class FinBERTMeanPoolBackend:
    """ProsusAI/finbert used as an encoder: mean-pool last hidden state (documented choice)."""

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.torch = torch
        self.batch_size = batch_size
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                b = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True, max_length=32,
                             return_tensors="pt")
                h = self.model(**b).last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class OpenAIEmbeddingBackend:
    """OpenAI embeddings API; spend is recorded on the ledger if one is given."""

    def __init__(self, model: str, dim: int = 1536, price_per_1m: float = 0.02, ledger=None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model, self.dim, self.price, self.ledger = model, dim, price_per_1m, ledger

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for i in range(0, len(texts), 512):
            chunk = texts[i:i + 512]
            r = self.client.embeddings.create(model=self.model, input=chunk)
            if self.ledger is not None:
                self.ledger.add("embedding", r.usage.total_tokens / 1e6 * self.price, self.model)
            rows.extend(d.embedding for d in r.data)
        return np.asarray(rows, dtype=np.float32)


def make_backend(model_name: str, batch_size: int = 256, ledger=None) -> Backend:
    """Pick a backend from the model id."""
    if model_name.startswith("text-embedding-3"):
        return OpenAIEmbeddingBackend(model_name, ledger=ledger)
    if "finbert" in model_name.lower():
        return FinBERTMeanPoolBackend(model_name)
    return SentenceTransformerBackend(model_name, batch_size=batch_size)


class Embedder:
    """Cached, L2-normalized embeddings for one model."""

    def __init__(self, model_name: str, cache_dir: str | Path, backend: Backend | None = None,
                 allow_live: bool = True, batch_size: int = 256, ledger=None):
        self.model_name = model_name
        self.dir = Path(cache_dir) / model_slug(model_name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self._batch_size, self._ledger = batch_size, ledger
        self.allow_live = allow_live
        self._keys: list[str] = []
        self._pos: dict[str, int] = {}
        self._vecs = np.zeros((0, 0), dtype=np.float16)
        self._load()

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = make_backend(self.model_name, self._batch_size, self._ledger)
        return self._backend

    def _load(self) -> None:
        kp, vp = self.dir / "keys.json", self.dir / "vectors.npy"
        if kp.exists() and vp.exists():
            self._keys = json.loads(kp.read_text())
            self._vecs = np.load(vp)
            self._pos = {k: i for i, k in enumerate(self._keys)}
            logger.info(f"embedding cache {self.model_name}: {len(self._keys)} vectors")

    def _save(self) -> None:
        tmp_v, tmp_k = self.dir / "vectors.npy.tmp", self.dir / "keys.json.tmp"
        with open(tmp_v, "wb") as fh:  # file handle: np.save must not append ".npy"
            np.save(fh, self._vecs)
        tmp_k.write_text(json.dumps(self._keys))
        os.replace(tmp_v, self.dir / "vectors.npy")
        os.replace(tmp_k, self.dir / "keys.json")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 L2-normalized vectors, computing and caching any misses."""
        hashes = [sha1_text(t) for t in texts]
        missing = sorted({h for h in hashes if h not in self._pos})
        if missing:
            if not self.allow_live:
                raise CacheMissError(f"{len(missing)} texts missing from cache for {self.model_name}")
            h2t = {sha1_text(t): t for t in texts}
            new_texts = [h2t[h] for h in missing]
            new = self.backend.encode(new_texts).astype(np.float32)
            new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
            new16 = new.astype(np.float16)
            self._vecs = new16 if self._vecs.size == 0 else np.vstack([self._vecs, new16])
            for h in missing:
                self._pos[h] = len(self._keys)
                self._keys.append(h)
            self._save()
        out = self._vecs[[self._pos[h] for h in hashes]].astype(np.float32)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Smoke test a real backbone** — `.venv/bin/python -c "from txcat.embedder import Embedder; e=Embedder('sentence-transformers/all-MiniLM-L6-v2','cache/embeddings'); print(e.embed(['STAPLES','WW GRAINGER']).shape)"` -> `(2, 384)` (first run downloads ~90 MB).
- [ ] **Step 6: Commit** — `git add src/txcat/embedder.py tests/test_embedder.py && git commit -m "feat: cached multi-backbone embedder"` (do NOT commit `cache/embeddings` yet; caches are committed once frozen in Plan B).

---

### Task 9: HNSW index over unique merchants

**Files:**
- Create: `src/txcat/index.py`, `tests/test_index.py`

- [ ] **Step 1: Failing tests**

`tests/test_index.py`:
```python
import numpy as np
import pandas as pd
import pytest

from txcat.index import HNSWIndex


def _unit(rows):
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_add_query_margin_and_metadata():
    idx = HNSWIndex(dim=3, M=8, ef_construction=50, seed=1)
    vecs = _unit([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    idx.add(vecs, labels=["a", "b", "a"], merchant_ids=["m1", "m2", "m3"], freqs=[10, 1, 3],
            first_seen=pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"]))
    r = idx.query(_unit([[1, 0, 0]])[0], k=3, ef_search=50)
    assert r.labels[0] == "a" and r.merchant_ids[0] == "m1"
    assert r.sims[0] == pytest.approx(1.0, abs=1e-4)
    assert r.margin == pytest.approx(r.sims[0] - r.sims[1], abs=1e-6)
    assert r.freqs[0] == 10
    assert len(idx) == 3


def test_add_grows_and_writeback_tagging():
    idx = HNSWIndex(dim=2, M=8, ef_construction=50, seed=1, max_elements=2)
    idx.add(_unit([[1, 0], [0, 1]]), ["x", "y"], ["m1", "m2"], [1, 1], pd.to_datetime(["2019-01-01"] * 2))
    idx.add(_unit([[1, 1]]), ["z"], ["m3"], [1], pd.to_datetime(["2024-01-01"]), source="writeback")
    assert len(idx) == 3
    assert idx.meta.loc[2, "source"] == "writeback"


def test_save_load_roundtrip(tmp_path):
    idx = HNSWIndex(dim=2, M=8, ef_construction=50, seed=3)
    idx.add(_unit([[1, 0], [0, 1]]), ["x", "y"], ["m1", "m2"], [4, 2], pd.to_datetime(["2019-01-01"] * 2))
    idx.save(tmp_path / "ix")
    idx2 = HNSWIndex.load(tmp_path / "ix")
    r = idx2.query(_unit([[0, 1]])[0], k=1)
    assert r.labels[0] == "y" and idx2.meta.loc[1, "freq"] == 2


def test_same_seed_same_neighbors():
    rng = np.random.default_rng(0)
    vecs = _unit(rng.standard_normal((500, 16)))
    q = _unit(rng.standard_normal((1, 16)))[0]
    outs = []
    for _ in range(2):
        idx = HNSWIndex(dim=16, M=8, ef_construction=50, seed=11)
        idx.add(vecs, [str(i) for i in range(500)], [str(i) for i in range(500)], [1] * 500,
                pd.to_datetime(["2019-01-01"] * 500))
        outs.append(idx.query(q, k=5, ef_search=50).merchant_ids)
    assert outs[0] == outs[1]
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/index.py`**

```python
"""hnswlib cosine index with per-entry metadata (label, merchant id, frequency, first seen, source)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd

META_COLS = ["merchant_id", "label", "freq", "first_seen", "source"]


@dataclass
class QueryResult:
    labels: list[str]
    sims: list[float]
    merchant_ids: list[str]
    freqs: list[int]

    @property
    def margin(self) -> float:
        return self.sims[0] - (self.sims[1] if len(self.sims) > 1 else 0.0)

    @property
    def top1_freq(self) -> int:
        return self.freqs[0]


class HNSWIndex:
    """One entry per unique training merchant. Grows on write-back."""

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200, seed: int = 42,
                 max_elements: int = 1000):
        self.dim, self.M, self.ef_construction, self.seed = dim, M, ef_construction, seed
        self._ix = hnswlib.Index(space="cosine", dim=dim)
        self._ix.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M, random_seed=seed)
        self._ix.set_num_threads(1)  # single-threaded inserts keep graph construction seed-deterministic
        self.meta = pd.DataFrame(columns=META_COLS)

    def __len__(self) -> int:
        return int(self._ix.get_current_count())

    def add(self, vectors: np.ndarray, labels: list[str], merchant_ids: list[str], freqs: list[int],
            first_seen: pd.Series | pd.DatetimeIndex, source: str = "train") -> None:
        n = len(labels)
        if len(self) + n > self._ix.get_max_elements():
            self._ix.resize_index(max(len(self) + n, 2 * self._ix.get_max_elements()))
        ids = np.arange(len(self), len(self) + n)
        self._ix.add_items(np.asarray(vectors, dtype=np.float32), ids, num_threads=1)
        new = pd.DataFrame({"merchant_id": list(merchant_ids), "label": list(labels), "freq": list(freqs),
                            "first_seen": pd.to_datetime(list(first_seen)), "source": source}, index=ids)
        self.meta = new if self.meta.empty else pd.concat([self.meta, new])

    def query(self, vector: np.ndarray, k: int = 5, ef_search: int = 64) -> QueryResult:
        k = min(k, len(self))
        self._ix.set_ef(max(ef_search, k))
        ids, dists = self._ix.knn_query(np.asarray(vector, dtype=np.float32).reshape(1, -1), k=k)
        ids, sims = ids[0].tolist(), (1.0 - dists[0]).tolist()
        rows = self.meta.loc[ids]
        return QueryResult(labels=rows["label"].tolist(), sims=sims,
                           merchant_ids=rows["merchant_id"].tolist(), freqs=rows["freq"].astype(int).tolist())

    def query_batch(self, vectors: np.ndarray, k: int = 5, ef_search: int = 64) -> list[QueryResult]:
        k = min(k, len(self))
        self._ix.set_ef(max(ef_search, k))
        ids, dists = self._ix.knn_query(np.asarray(vectors, dtype=np.float32), k=k)
        out = []
        for row_ids, row_d in zip(ids, dists, strict=True):
            rows = self.meta.loc[row_ids.tolist()]
            out.append(QueryResult(rows["label"].tolist(), (1.0 - row_d).tolist(),
                                   rows["merchant_id"].tolist(), rows["freq"].astype(int).tolist()))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._ix.save_index(str(path / "index.bin"))
        self.meta.to_parquet(path / "meta.parquet")
        (path / "params.txt").write_text(f"{self.dim},{self.M},{self.ef_construction},{self.seed}")

    @classmethod
    def load(cls, path: str | Path) -> HNSWIndex:
        path = Path(path)
        dim, M, efc, seed = (int(x) for x in (path / "params.txt").read_text().split(","))
        obj = cls(dim, M, efc, seed)
        obj.meta = pd.read_parquet(path / "meta.parquet")
        obj._ix.load_index(str(path / "index.bin"), max_elements=max(len(obj.meta) * 2, 1000))
        return obj
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git add src/txcat/index.py tests/test_index.py && git commit -m "feat: hnswlib merchant index with metadata"`

---

### Task 10: Confidence gate (threshold / margin / calibrated LR)

**Files:**
- Create: `src/txcat/gate.py`, `tests/test_gate.py`

- [ ] **Step 1: Failing tests**

`tests/test_gate.py`:
```python
import numpy as np

from txcat.gate import ConfidenceGate
from txcat.index import QueryResult


def qr(s1, s2, f1=5):
    return QueryResult(labels=["a", "b"], sims=[s1, s2], merchant_ids=["x", "y"], freqs=[f1, 1])


def test_threshold_mode():
    g = ConfidenceGate("threshold", 0.65)
    assert g.should_fallback(qr(0.60, 0.5)) is True
    assert g.should_fallback(qr(0.70, 0.5)) is False


def test_margin_mode():
    g = ConfidenceGate("margin", 0.10)
    assert g.should_fallback(qr(0.80, 0.75)) is True
    assert g.should_fallback(qr(0.80, 0.60)) is False


def test_calibrated_lr_learns_that_low_sim_means_wrong():
    rng = np.random.default_rng(0)
    results, correct = [], []
    for _ in range(400):
        s1 = rng.uniform(0.3, 1.0)
        results.append(qr(s1, s1 - rng.uniform(0, 0.2), f1=int(rng.integers(1, 50))))
        correct.append(bool(rng.random() < s1))  # higher sim -> more likely correct
    g = ConfidenceGate("calibrated_lr", 0.5).fit(results, correct)
    assert g.should_fallback(qr(0.35, 0.30)) is True
    assert g.should_fallback(qr(0.98, 0.60)) is False
    p = g.p_wrong(qr(0.5, 0.4))
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/gate.py`**

```python
"""Decide whether a kNN result is trusted or routed to the LLM fallback."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from txcat.index import QueryResult

MODES = ("threshold", "margin", "calibrated_lr")


def _features(r: QueryResult) -> list[float]:
    s2 = r.sims[1] if len(r.sims) > 1 else 0.0
    return [r.sims[0], s2, r.sims[0] - s2, float(np.log1p(r.top1_freq))]


class ConfidenceGate:
    """``threshold``: fallback if sim1 < t. ``margin``: fallback if sim1 - sim2 < t.
    ``calibrated_lr``: fallback if P(top-1 wrong) > t, LR on [sim1, sim2, margin, log1p(freq)]."""

    def __init__(self, mode: str = "threshold", threshold: float = 0.65):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.threshold = mode, threshold
        self._lr: LogisticRegression | None = None

    def fit(self, results: list[QueryResult], correct: list[bool]) -> ConfidenceGate:
        """Fit the calibrated LR on a calibration slice (label 1 = top-1 wrong)."""
        X = np.asarray([_features(r) for r in results])
        y = 1 - np.asarray(correct, dtype=int)
        self._lr = LogisticRegression(max_iter=1000).fit(X, y)
        return self

    def p_wrong(self, r: QueryResult) -> float:
        if self._lr is None:
            raise RuntimeError("calibrated_lr gate must be fit() first")
        return float(self._lr.predict_proba(np.asarray([_features(r)]))[0, 1])

    def should_fallback(self, r: QueryResult) -> bool:
        if self.mode == "threshold":
            return r.sims[0] < self.threshold
        if self.mode == "margin":
            return r.margin < self.threshold
        return self.p_wrong(r) > self.threshold
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git add src/txcat/gate.py tests/test_gate.py && git commit -m "feat: confidence gate with three modes"`

---

### Task 11: Temporal splits, frequencies, tail mask, volume rule, FES sampling

**Files:**
- Create: `src/txcat/data/splits.py`, `tests/test_splits.py`

- [ ] **Step 1: Failing tests**

`tests/test_splits.py`:
```python
import pandas as pd
import pytest

from txcat.config import WindowCfg
from txcat.data.splits import (merchant_frequencies, sample_fes, tail_mask, temporal_split,
                               volume_rule_ok)


def _df():
    dates = ["2019-05-01", "2020-05-01", "2021-05-01", "2023-12-31", "2024-01-01", "2024-06-01", "2025-01-01"]
    merchants = ["A", "A", "B", "C", "A", "B", "D"]
    return pd.DataFrame({"txn_id": [f"t{i}" for i in range(7)], "date": pd.to_datetime(dates),
                         "merchant": merchants, "category": ["x"] * 7, "ambiguous": [0] * 7})


def test_temporal_split_is_by_date_only():
    w = WindowCfg(train_start="2019-01-01", train_end="2023-12-31", test_start="2024-01-01")
    tr, te = temporal_split(_df(), w)
    assert list(tr["txn_id"]) == ["t0", "t1", "t2", "t3"]
    assert list(te["txn_id"]) == ["t4", "t5", "t6"]


def test_frequencies_and_tail_mask():
    w = WindowCfg(train_start="2019-01-01", train_end="2023-12-31", test_start="2024-01-01")
    tr, te = temporal_split(_df(), w)
    f = merchant_frequencies(tr)
    assert f["A"] == 2 and f["B"] == 1 and f["C"] == 1 and "D" not in f
    m = tail_mask(te, f, k=1)
    assert list(m) == [False, True, True]  # A freq 2 > 1; B freq 1; D freq 0 (unseen) is tail


def test_volume_rule():
    te = pd.DataFrame({"merchant": ["m1", "m2", "m3"], "date": pd.to_datetime(["2024-01-01"] * 3)})
    f = pd.Series({"m1": 1})
    ok, stats = volume_rule_ok(te, f, k=3, min_test_txns=3, min_tail_merchants=3)
    assert ok is True and stats["n_test_txns"] == 3 and stats["n_tail_merchants"] == 3
    ok, _ = volume_rule_ok(te, f, k=3, min_test_txns=4, min_tail_merchants=3)
    assert ok is False


def test_sample_fes_is_by_merchant_and_seeded():
    te = pd.DataFrame({"merchant": [f"m{i // 3}" for i in range(60)], "ambiguous": [0] * 60,
                       "category": ["c"] * 60})
    f = pd.Series({f"m{i}": (10 if i < 5 else 1) for i in range(20)})  # 5 head, 15 tail merchants
    a = sample_fes(te, f, k=3, n_tail=4, n_head=2, seed=1)
    b = sample_fes(te, f, k=3, n_tail=4, n_head=2, seed=1)
    assert a["merchant"].nunique() == 6 and set(a["merchant"]) == set(b["merchant"])
    assert (a.groupby("merchant").size() == 3).all()  # all of a merchant's rows come along
    assert a["is_tail"].sum() == 4 * 3
    c = sample_fes(te, f, k=3, n_tail=100, n_head=100, seed=1)  # asks for more than exist -> takes all
    assert c["merchant"].nunique() == 20
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/data/splits.py`**

```python
"""Temporal splits and merchant-level sampling. No row-level randomization anywhere."""

from __future__ import annotations

import numpy as np
import pandas as pd

from txcat.config import WindowCfg


def temporal_split(df: pd.DataFrame, w: WindowCfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on ``date`` only. train: [train_start, train_end]; test: [test_start, test_end or max]."""
    d = df["date"]
    train = df[(d >= pd.Timestamp(w.train_start)) & (d <= pd.Timestamp(w.train_end))]
    hi = pd.Timestamp(w.test_end) if w.test_end else d.max()
    test = df[(d >= pd.Timestamp(w.test_start)) & (d <= hi)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def merchant_frequencies(train: pd.DataFrame) -> pd.Series:
    """Transactions per normalized merchant in the training window."""
    return train.groupby("merchant").size()


def freq_of(test: pd.DataFrame, freqs: pd.Series) -> pd.Series:
    """Training frequency of each test row's merchant (0 if unseen)."""
    return test["merchant"].map(freqs).fillna(0).astype(int)


def tail_mask(test: pd.DataFrame, freqs: pd.Series, k: int) -> pd.Series:
    """True where the row's merchant has training frequency <= k (unseen merchants count as tail)."""
    return freq_of(test, freqs) <= k


def volume_rule_ok(test: pd.DataFrame, freqs: pd.Series, k: int, min_test_txns: int,
                   min_tail_merchants: int) -> tuple[bool, dict]:
    """Spec section 3 conditional volume rule."""
    is_tail = tail_mask(test, freqs, k)
    stats = {"n_test_txns": int(len(test)), "n_test_merchants": int(test["merchant"].nunique()),
             "n_tail_merchants": int(test.loc[is_tail, "merchant"].nunique()),
             "tail_txn_share": float(is_tail.mean()) if len(test) else 0.0}
    ok = stats["n_test_txns"] >= min_test_txns and stats["n_tail_merchants"] >= min_tail_merchants
    return ok, stats


def sample_fes(test: pd.DataFrame, freqs: pd.Series, k: int, n_tail: int, n_head: int,
               seed: int, exclude_ambiguous: bool = True) -> pd.DataFrame:
    """Fallback evaluation set: sample merchants (not rows), keep all their test rows.

    Tail merchants (freq <= k) and head merchants sampled separately with a seeded RNG; if fewer
    exist than requested, all are taken. Adds ``is_tail`` and ``train_freq`` columns.
    """
    t = test.copy()
    t["train_freq"] = freq_of(t, freqs)
    t["is_tail"] = t["train_freq"] <= k
    pool = t[t["ambiguous"] == 0] if exclude_ambiguous else t
    rng = np.random.default_rng(seed)
    tail_m = np.sort(pool.loc[pool["is_tail"], "merchant"].unique())
    head_m = np.sort(pool.loc[~pool["is_tail"], "merchant"].unique())
    pick_t = rng.choice(tail_m, size=min(n_tail, len(tail_m)), replace=False) if len(tail_m) else []
    pick_h = rng.choice(head_m, size=min(n_head, len(head_m)), replace=False) if len(head_m) else []
    chosen = set(pick_t) | set(pick_h)
    return t[t["merchant"].isin(chosen)].reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git add src/txcat/data/splits.py tests/test_splits.py && git commit -m "feat: temporal splits, tail mask, volume rule, FES sampling"`

---

### Task 12: kNN driver (build merchant index from train, predict test)

**Files:**
- Create: `src/txcat/knn.py`, `tests/test_knn.py`

- [ ] **Step 1: Failing test**

`tests/test_knn.py`:
```python
import pandas as pd

from txcat.embedder import Embedder, FakeHashBackend
from txcat.knn import build_merchant_index, predict_knn


def test_build_and_predict(tmp_path):
    train = pd.DataFrame({
        "merchant": ["STAPLES", "STAPLES", "STAPLES", "USPS", "GRAINGER"],
        "category": ["office_supplies", "office_supplies", "retail", "financial_postal_shipping", "industrial_hardware"],
        "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01", "2020-01-01", "2021-01-01"]),
        "ambiguous": [0] * 5,
    })
    emb = Embedder("fake/m", tmp_path, backend=FakeHashBackend(16))
    idx = build_merchant_index(train, emb, M=8, ef_construction=50, seed=1)
    assert len(idx) == 3
    row = idx.meta[idx.meta["merchant_id"] == "STAPLES"].iloc[0]
    assert row["label"] == "office_supplies" and row["freq"] == 3  # majority label, frequency
    assert str(row["first_seen"].date()) == "2019-01-01"

    test = pd.DataFrame({"merchant": ["STAPLES", "NEVER SEEN"]})
    pred = predict_knn(test, idx, emb, k=2, ef_search=50)
    assert list(pred.columns) == ["pred", "sim1", "sim2", "margin", "top1_freq", "top1_merchant"]
    assert pred.loc[0, "pred"] == "office_supplies" and pred.loc[0, "sim1"] > 0.999
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/knn.py`**

```python
"""Build the merchant index from a training frame and produce kNN predictions for a test frame."""

from __future__ import annotations

import pandas as pd

from txcat.embedder import Embedder
from txcat.index import HNSWIndex


def merchant_table(train: pd.DataFrame) -> pd.DataFrame:
    """One row per unique merchant: majority category, frequency, first seen. Ambiguous rows are
    excluded from label voting but still counted in frequency."""
    g = train.groupby("merchant")
    freq = g.size().rename("freq")
    first_seen = g["date"].min().rename("first_seen")
    lab_src = train[train["ambiguous"] == 0] if "ambiguous" in train else train
    label = lab_src.groupby("merchant")["category"].agg(lambda s: s.value_counts().idxmax()).rename("label")
    tbl = pd.concat([freq, first_seen, label], axis=1).reset_index()
    tbl = tbl[tbl["label"].notna()].reset_index(drop=True)
    return tbl


def build_merchant_index(train: pd.DataFrame, emb: Embedder, M: int, ef_construction: int,
                         seed: int) -> HNSWIndex:
    tbl = merchant_table(train)
    vecs = emb.embed(tbl["merchant"].tolist())
    idx = HNSWIndex(dim=vecs.shape[1], M=M, ef_construction=ef_construction, seed=seed,
                    max_elements=max(len(tbl) * 2, 1000))
    idx.add(vecs, tbl["label"].tolist(), tbl["merchant"].tolist(), tbl["freq"].astype(int).tolist(),
            tbl["first_seen"])
    return idx


def predict_knn(test: pd.DataFrame, idx: HNSWIndex, emb: Embedder, k: int, ef_search: int) -> pd.DataFrame:
    """Top-1 kNN prediction plus similarity features for every test row (unique merchants embedded once)."""
    uniq = test["merchant"].drop_duplicates().tolist()
    vecs = emb.embed(uniq)
    res = idx.query_batch(vecs, k=k, ef_search=ef_search)
    per_m = {m: r for m, r in zip(uniq, res, strict=True)}
    rows = []
    for m in test["merchant"]:
        r = per_m[m]
        s2 = r.sims[1] if len(r.sims) > 1 else 0.0
        rows.append({"pred": r.labels[0], "sim1": r.sims[0], "sim2": s2, "margin": r.sims[0] - s2,
                     "top1_freq": r.freqs[0], "top1_merchant": r.merchant_ids[0]})
    return pd.DataFrame(rows, index=test.index)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git add src/txcat/knn.py tests/test_knn.py && git commit -m "feat: kNN merchant index driver"`

---

### Task 13: Exp 0 — data audit, window decision, label-noise sample

**Files:**
- Create: `src/txcat/data/prepare.py`, `experiments/__init__.py` (empty), `experiments/exp0_data_audit.py`, `tests/test_prepare.py`

- [ ] **Step 1: Failing test for the prepare step (normalize + labels in one call)**

`tests/test_prepare.py`:
```python
import pandas as pd

from txcat.data.prepare import prepare


def test_prepare_adds_merchant_and_labels(tmp_path):
    (tmp_path / "mcc_to_category.csv").write_text(
        "mcc,mcc_description,category,ambiguous,notes\n5943,Stationery,office_supplies,0,\n")
    (tmp_path / "mcc_description_aliases.csv").write_text(
        "source,observed_description,mcc,match\ndc,Stationery Stores,5943,exact\n")
    df = pd.DataFrame({"txn_id": ["dc:1"], "date": pd.to_datetime(["2024-01-01"]),
                       "raw_merchant": ["STAPLES       00102186"], "mcc_description": ["Stationery Stores"],
                       "source": ["dc"]})
    out = prepare(df, tmp_path)
    assert out.loc[0, "merchant"] == "STAPLES"
    assert out.loc[0, "category"] == "office_supplies" and out.loc[0, "ambiguous"] == 0
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/txcat/data/prepare.py`**

```python
"""Processed parquet -> analysis frame: adds ``merchant`` (normalized) and label columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from txcat.data.taxonomy import attach_labels
from txcat.normalizer import normalize_merchant


def prepare(df: pd.DataFrame, taxonomy_dir: str | Path) -> pd.DataFrame:
    out = df.copy()
    uniq = out["raw_merchant"].drop_duplicates()
    norm = {r: normalize_merchant(r).text for r in uniq}
    out["merchant"] = out["raw_merchant"].map(norm)
    out = attach_labels(out, taxonomy_dir)
    return out


def load_prepared(processed_path: str | Path, taxonomy_dir: str | Path) -> pd.DataFrame:
    return prepare(pd.read_parquet(processed_path), taxonomy_dir)
```

- [ ] **Step 4: Run test, verify pass**

- [ ] **Step 5: Write `experiments/exp0_data_audit.py`**

```python
"""Exp 0: dataset stats, taxonomy coverage, volume-rule window decision, label-noise audit sample.

Usage: python -m experiments.exp0_data_audit --config configs/dc.yaml [--seed 42]
Outputs: results/tables/tab0_dataset_stats.csv, results/tables/tab0_taxonomy_coverage.csv,
         results/window_decision.json, data/taxonomy/label_noise_audit_sample.csv (300 rows)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import merchant_frequencies, tail_mask, temporal_split, volume_rule_ok
from txcat.utils import now_iso, set_seed, setup_logging


def window_stats(df: pd.DataFrame, w: WindowCfg, k: int, ks_sens: list[int]) -> dict:
    tr, te = temporal_split(df, w)
    f = merchant_frequencies(tr)
    row = {"train_start": w.train_start, "train_end": w.train_end, "test_start": w.test_start,
           "n_train_txns": len(tr), "n_train_merchants": int(tr["merchant"].nunique()),
           "n_test_txns": len(te), "n_test_merchants": int(te["merchant"].nunique())}
    for kk in [k, *ks_sens]:
        m = tail_mask(te, f, kk)
        row[f"tail_le{kk}_txn_share"] = round(float(m.mean()), 4)
        row[f"tail_le{kk}_merchants"] = int(te.loc[m, "merchant"].nunique())
    row["unseen_txn_share"] = round(float((te["merchant"].map(f).fillna(0) == 0).mean()), 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg.seed
    set_seed(seed)
    setup_logging(cfg.logs_dir, "exp0")
    tables = Path(cfg.results_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    stats_rows, coverage_rows = [], []
    for name in ["dc", "oklahoma"]:
        d = cfg.datasets[name]
        if not Path(d.processed_path).exists():
            logger.warning(f"{name}: processed file missing, skipping")
            continue
        df = load_prepared(d.processed_path, cfg.taxonomy_dir)
        n = len(df)
        coverage_rows.append({"dataset": name, "n_rows": n,
                              "rows_with_mcc": int(df["mcc"].notna().sum()),
                              "rows_ambiguous": int(df["ambiguous"].sum()),
                              "share_unlabeled": round(float(df["mcc"].isna().mean()), 4),
                              "share_ambiguous": round(float(df["ambiguous"].mean()), 4),
                              "n_categories_present": int(df["category"].dropna().nunique())})
        if name == "dc":
            primary = window_stats(df, d.window, cfg.tail_k, cfg.tail_sensitivity)
            tr, te = temporal_split(df, d.window)
            ok, vs = volume_rule_ok(te, merchant_frequencies(tr), cfg.tail_k,
                                    d.volume_rule.min_test_txns, d.volume_rule.min_tail_merchants)
            decision = {"dataset": "dc", "checked_at": now_iso(), "primary_window": d.window.model_dump(),
                        "primary_stats": vs, "primary_ok": ok}
            chosen = d.window
            if not ok:
                fb = d.volume_rule.fallback_window
                tr2, te2 = temporal_split(df, fb)
                ok2, vs2 = volume_rule_ok(te2, merchant_frequencies(tr2), cfg.tail_k,
                                          d.volume_rule.min_test_txns, d.volume_rule.min_tail_merchants)
                decision.update({"fallback_window": fb.model_dump(), "fallback_stats": vs2, "fallback_ok": ok2})
                chosen = fb
                logger.warning(f"primary window failed volume rule {vs}; sliding to {fb}")
            decision["chosen_window"] = chosen.model_dump()
            Path(cfg.results_dir, "window_decision.json").write_text(json.dumps(decision, indent=2))
            stats_rows.append({"dataset": "dc", **window_stats(df, chosen, cfg.tail_k, cfg.tail_sensitivity)})
            if chosen.model_dump() != d.window.model_dump():
                stats_rows.append({"dataset": "dc_primary_window_rejected", **primary})

            # label-noise audit sample: 300 unique merchants, non-ambiguous, seeded
            pool = df[(df["ambiguous"] == 0)].drop_duplicates("merchant")
            rng = np.random.default_rng(seed)
            pick = pool.iloc[np.sort(rng.choice(len(pool), size=min(300, len(pool)), replace=False))]
            audit = pick[["raw_merchant", "merchant", "mcc_description", "mcc", "category"]].copy()
            audit["judgement"] = ""  # correct | wrong | unclear
            audit["note"] = ""
            audit.to_csv(Path(cfg.taxonomy_dir) / "label_noise_audit_sample.csv", index=False)
            logger.info(f"label-noise audit sample: {len(audit)} rows written")
        else:
            stats_rows.append({"dataset": name, "n_train_txns": 0, "n_test_txns": n,
                               "n_test_merchants": int(df["merchant"].nunique()),
                               "train_start": "", "train_end": "", "test_start": str(df["date"].min().date())})

    pd.DataFrame(stats_rows).to_csv(tables / "tab0_dataset_stats.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(tables / "tab0_taxonomy_coverage.csv", index=False)
    logger.info("\n" + pd.DataFrame(stats_rows).T.to_string())
    logger.info("\n" + pd.DataFrame(coverage_rows).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run Exp 0**

Run: `.venv/bin/python -m experiments.exp0_data_audit --config configs/dc.yaml`
Expected: `results/window_decision.json` with `"primary_ok": true` (DC has 90k test rows and ~14.5k raw test vendors, so ≥2,500 tail merchants is expected); `tab0_dataset_stats.csv`, `tab0_taxonomy_coverage.csv`; `data/taxonomy/label_noise_audit_sample.csv` with 300 rows. If `share_unlabeled` > 0.02 for either dataset, go back to Task 6 step 4 and resolve unmatched descriptions.

- [ ] **Step 7: Pre-fill the audit judgements** — open `data/taxonomy/label_noise_audit_sample.csv`, fill `judgement` for all 300 rows (`correct` if the mapped category is what the merchant obviously is, `wrong` if not, `unclear` if you cannot tell from the name). Author verifies. Then:
```bash
.venv/bin/python -c "
import pandas as pd; a=pd.read_csv('data/taxonomy/label_noise_audit_sample.csv'); vc=a.judgement.value_counts(); print(vc); 
pd.DataFrame([{'n':len(a),'correct':vc.get('correct',0),'wrong':vc.get('wrong',0),'unclear':vc.get('unclear',0),'noise_rate_excl_unclear': round(vc.get('wrong',0)/max(1,vc.get('correct',0)+vc.get('wrong',0)),4)}]).to_csv('results/tables/tab0_label_noise.csv',index=False)"
```

- [ ] **Step 8: Commit** — `git add src/txcat/data/prepare.py tests/test_prepare.py experiments/__init__.py experiments/exp0_data_audit.py results/window_decision.json results/tables/tab0_*.csv data/taxonomy/label_noise_audit_sample.csv && git commit -m "feat: exp0 data audit, window decision, label-noise sample"`

---

### Task 14: Plot style + Exp 1 tail characterization

**Files:**
- Create: `analysis/__init__.py`, `analysis/style.py`, `analysis/plots.py`, `experiments/exp1_tail_characterization.py`, `tests/test_plots.py`

Design rules applied (dataviz skill): one y-axis per chart; fixed categorical order head=blue `#2a78d6`, tail=orange `#eb6834`, third series aqua `#1baf7a`; thin 2px lines, ≥6pt markers; recessive grey grid; legend present when ≥2 series; text in ink colors never series colors; no rainbow; sequential = single blue ramp.

- [ ] **Step 1: Failing test (plots render to files without error)**

`tests/test_plots.py`:
```python
import numpy as np
import pandas as pd

from analysis.plots import fig_acc_by_freq, fig_acc_vs_sim, fig_zipf


def test_plots_write_pdf_and_png(tmp_path):
    freqs = pd.Series(np.random.default_rng(0).zipf(1.2, 2000))
    fig_zipf(freqs, tmp_path / "fig1_zipf")
    acc = pd.DataFrame({"bin": ["0", "1", "2-5", "6-20", "21-100", "100+"],
                        "accuracy": [0.3, 0.5, 0.6, 0.8, 0.9, 0.95], "n": [100, 50, 80, 60, 40, 20],
                        "lo": [0.25, 0.4, 0.55, 0.75, 0.85, 0.9], "hi": [0.35, 0.6, 0.65, 0.85, 0.95, 0.99]})
    fig_acc_by_freq(acc, tmp_path / "fig2")
    df = pd.DataFrame({"sim1": np.random.default_rng(1).uniform(0.2, 1, 500),
                       "correct": np.random.default_rng(2).random(500) > 0.4,
                       "is_tail": np.random.default_rng(3).random(500) > 0.5})
    fig_acc_vs_sim(df, tmp_path / "fig3")
    for stem in ("fig1_zipf", "fig2", "fig3"):
        assert (tmp_path / f"{stem}.pdf").exists() and (tmp_path / f"{stem}.png").exists()
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement style and plots**

`analysis/__init__.py`: empty.

`analysis/style.py`:
```python
"""Paper figure style. Palette validated with the dataviz skill's validator (adjacent CVD ΔE ≥ 8)."""

import matplotlib as mpl

SERIES = {"head": "#2a78d6", "tail": "#eb6834", "third": "#1baf7a", "fourth": "#eda100", "fifth": "#e87ba4"}
BACKBONE_ORDER = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5",
                  "sentence-transformers/all-mpnet-base-v2", "ProsusAI/finbert", "text-embedding-3-small"]
BACKBONE_COLORS = dict(zip(BACKBONE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], strict=True))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def apply() -> None:
    mpl.rcParams.update({
        "figure.figsize": (5.2, 3.4), "figure.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def save(fig, stem) -> None:
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
```

`analysis/plots.py`:
```python
"""Figure functions for Exp 1 (fig1–fig3). Each takes tidy data and a path stem (no extension)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style

style.apply()


def fig_zipf(freqs: pd.Series, stem: str | Path, title: str = "Merchant frequency (training window)") -> None:
    """Rank–frequency plot on log–log axes with a fitted slope in the legend."""
    f = np.sort(np.asarray(freqs, dtype=float))[::-1]
    rank = np.arange(1, len(f) + 1)
    slope, intercept = np.polyfit(np.log(rank), np.log(f), 1)
    fig, ax = plt.subplots()
    ax.loglog(rank, f, color=style.SERIES["head"], lw=1.6, label="merchants")
    ax.loglog(rank, np.exp(intercept) * rank ** slope, color=style.INK2, lw=1, ls="--",
              label=f"power-law fit, slope {slope:.2f}")
    ax.set_xlabel("merchant rank")
    ax.set_ylabel("transactions")
    ax.set_title(title, loc="left", color=style.INK)
    ax.legend(loc="upper right")
    style.save(fig, stem)
    plt.close(fig)


def fig_acc_by_freq(acc: pd.DataFrame, stem: str | Path, title: str = "kNN top-1 accuracy by training frequency") -> None:
    """Bars with 95% CI whiskers; ``acc`` has columns bin, accuracy, n, lo, hi (bins in display order)."""
    fig, ax = plt.subplots()
    x = np.arange(len(acc))
    colors = [style.SERIES["tail"] if b in ("0", "1") else style.SERIES["head"] for b in acc["bin"]]
    ax.bar(x, acc["accuracy"], width=0.62, color=colors, edgecolor="white", linewidth=1)
    ax.errorbar(x, acc["accuracy"], yerr=[acc["accuracy"] - acc["lo"], acc["hi"] - acc["accuracy"]],
                fmt="none", ecolor=style.INK2, elinewidth=1, capsize=2)
    for xi, (a, n) in enumerate(zip(acc["accuracy"], acc["n"], strict=True)):
        ax.text(xi, min(a + 0.04, 0.97), f"n={n:,}", ha="center", va="bottom", fontsize=7, color=style.INK2)
    ax.set_xticks(x, acc["bin"])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("merchant frequency in training index")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(title, loc="left", color=style.INK)
    ax.plot([], [], color=style.SERIES["tail"], lw=6, label="unseen or singleton (freq ≤ 1)")
    ax.plot([], [], color=style.SERIES["head"], lw=6, label="freq ≥ 2")
    ax.legend(loc="lower right")
    style.save(fig, stem)
    plt.close(fig)


def fig_acc_vs_sim(df: pd.DataFrame, stem: str | Path, nbins: int = 12,
                   title: str = "Accuracy vs top-1 similarity") -> None:
    """Binned accuracy against sim1, head and tail as two lines. ``df`` has sim1, correct, is_tail."""
    edges = np.linspace(df["sim1"].min(), 1.0, nbins + 1)
    fig, ax = plt.subplots()
    for name, mask, color in (("head", ~df["is_tail"], style.SERIES["head"]),
                              ("tail (freq ≤ 3)", df["is_tail"], style.SERIES["tail"])):
        sub = df[mask]
        if sub.empty:
            continue
        b = np.clip(np.digitize(sub["sim1"], edges) - 1, 0, nbins - 1)
        g = sub.groupby(b)["correct"].agg(["mean", "size"])
        g = g[g["size"] >= 10]
        centers = (edges[g.index] + edges[g.index + 1]) / 2
        ax.plot(centers, g["mean"], marker="o", color=color, label=name)
    ax.set_xlabel("top-1 cosine similarity")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left", color=style.INK)
    ax.legend(loc="lower right")
    style.save(fig, stem)
    plt.close(fig)
```

- [ ] **Step 4: Run test, verify pass** — `.venv/bin/pytest tests/test_plots.py -v`

- [ ] **Step 5: Write `experiments/exp1_tail_characterization.py`**

```python
"""Exp 1: tail characterization on the DC primary window (kNN with the first configured backbone).

Usage: python -m experiments.exp1_tail_characterization --config configs/dc.yaml --seeds 42 43 44
Outputs: results/figures/fig1_zipf.{pdf,png}, fig2_acc_by_freq, fig3_acc_vs_sim,
         results/tables/tab1_tail_stats.csv, results/tables/tab1_acc_by_freq.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from analysis.plots import fig_acc_by_freq, fig_acc_vs_sim, fig_zipf
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import freq_of, merchant_frequencies, temporal_split
from txcat.embedder import Embedder
from txcat.knn import build_merchant_index, predict_knn
from txcat.utils import set_seed, setup_logging

BINS = [(0, 0, "0"), (1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 100, "21-100"), (101, 10**9, "100+")]


def bin_label(f: int) -> str:
    for lo, hi, lab in BINS:
        if lo <= f <= hi:
            return lab
    return "100+"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--mode", choices=["live", "reproduce"], default="live")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    backbone = args.backbone or cfg.embed.backbones[0]
    setup_logging(cfg.logs_dir, "exp1")
    figs, tables = Path(cfg.results_dir, "figures"), Path(cfg.results_dir, "tables")
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    d = cfg.datasets["dc"]
    win = WindowCfg(**json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"])
    df = load_prepared(d.processed_path, cfg.taxonomy_dir)
    df = df[df["category"].notna()]
    train, test = temporal_split(df, win)
    freqs = merchant_frequencies(train)
    test = test[test["ambiguous"] == 0].copy()  # headline slice
    test["train_freq"] = freq_of(test, freqs)
    test["is_tail"] = test["train_freq"] <= cfg.tail_k

    fig_zipf(freqs, figs / "fig1_zipf")

    emb = Embedder(backbone, cfg.embed.cache_dir, allow_live=(args.mode == "live"),
                   batch_size=cfg.embed.batch_size)
    per_seed = []
    for seed in seeds:
        set_seed(seed)
        idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
        pred = predict_knn(test, idx, emb, cfg.index.k, cfg.index.ef_search)
        t = test.assign(pred=pred["pred"].values, sim1=pred["sim1"].values, seed=seed)
        t["correct"] = t["pred"] == t["category"]
        per_seed.append(t)
        logger.info(f"seed {seed}: overall acc {t['correct'].mean():.4f}, tail acc {t.loc[t.is_tail,'correct'].mean():.4f}")
    allp = pd.concat(per_seed, ignore_index=True)

    # fig2: accuracy by frequency bin (pooled over seeds; CI = Wilson on pooled rows / n_seeds)
    allp["bin"] = allp["train_freq"].map(bin_label)
    rows = []
    for _, _, lab in BINS:
        s = allp[allp["bin"] == lab]
        n = len(s) // len(seeds)
        k = int(s["correct"].sum() / len(seeds))
        lo, hi = wilson(k, n)
        rows.append({"bin": lab, "accuracy": k / n if n else 0.0, "n": n, "lo": lo, "hi": hi})
    acc = pd.DataFrame(rows)
    acc.to_csv(tables / "tab1_acc_by_freq.csv", index=False)
    fig_acc_by_freq(acc, figs / "fig2_acc_by_freq")

    # fig3: accuracy vs similarity, head vs tail (first seed is representative; all seeds in CSV)
    fig_acc_vs_sim(per_seed[0][["sim1", "correct", "is_tail"]], figs / "fig3_acc_vs_sim")

    # tab1: tail stats at k in {2,3,5}, mean ± CI across seeds
    out = []
    for k in [cfg.tail_k, *cfg.tail_sensitivity]:
        m_all = allp["train_freq"] <= k
        accs = [g.loc[g["train_freq"] <= k, "correct"].mean() for g in per_seed]
        haccs = [g.loc[g["train_freq"] > k, "correct"].mean() for g in per_seed]
        out.append({"tail_k": k, "backbone": backbone, "window": f"{win.train_start}..{win.test_start}+",
                    "tail_merchants": int(allp.loc[m_all, "merchant"].nunique()),
                    "tail_merchant_share": round(allp.loc[m_all, "merchant"].nunique() / allp["merchant"].nunique(), 4),
                    "tail_txn_share": round(float(m_all.mean()), 4),
                    "tail_acc_mean": round(float(np.mean(accs)), 4), "tail_acc_std": round(float(np.std(accs)), 4),
                    "head_acc_mean": round(float(np.mean(haccs)), 4), "head_acc_std": round(float(np.std(haccs)), 4),
                    "n_seeds": len(seeds)})
    pd.DataFrame(out).to_csv(tables / "tab1_tail_stats.csv", index=False)
    allp[["txn_id", "merchant", "category", "pred", "sim1", "train_freq", "is_tail", "seed"]].to_parquet(
        Path(cfg.results_dir) / "exp1_predictions.parquet", index=False)
    logger.info("\n" + pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run Exp 1** — `.venv/bin/python -m experiments.exp1_tail_characterization --config configs/dc.yaml --seeds 42 43 44` (run from repo root so `analysis` is importable)
Expected: three figures in `results/figures/`, `tab1_tail_stats.csv` showing tail accuracy well below head accuracy. Open the PNGs and check for label collisions and axis sanity (dataviz step 7).

- [ ] **Step 7: Commit** — `git add analysis experiments/exp1_tail_characterization.py tests/test_plots.py results/figures/fig1_zipf.* results/figures/fig2_acc_by_freq.* results/figures/fig3_acc_vs_sim.* results/tables/tab1_*.csv && git commit -m "feat: exp1 tail characterization figures and tables"`

---

### Task 15: Spend ledger with hard cap

**Files:**
- Create: `src/txcat/budget.py`, `tests/test_budget.py`

- [ ] **Step 1: Failing test**

`tests/test_budget.py`:
```python
import pytest

from txcat.budget import BudgetExceeded, SpendLedger


def test_ledger_accumulates_persists_and_caps(tmp_path):
    p = tmp_path / "ledger.json"
    led = SpendLedger(p, max_usd=1.0)
    led.add("search", 0.4, "brave")
    led.add("llm", 0.5, "gpt-5-nano")
    assert led.total == pytest.approx(0.9)
    led2 = SpendLedger(p, max_usd=1.0)
    assert led2.total == pytest.approx(0.9)
    assert led2.by_kind()["search"] == pytest.approx(0.4)
    with pytest.raises(BudgetExceeded):
        led2.add("llm", 0.2, "gpt-5-nano")
    assert led2.total == pytest.approx(0.9)  # rejected call not recorded
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement `src/txcat/budget.py`**

```python
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
            raise BudgetExceeded(f"spend {self.total:.2f} + {usd:.4f} would exceed cap {self.max_usd:.2f} USD")
        self.entries.append({"ts": now_iso(), "kind": kind, "usd": float(usd), "detail": detail})
        self.path.write_text(json.dumps(self.entries))
```

- [ ] **Step 4: Run test, verify pass**
- [ ] **Step 5: Commit** — `git add src/txcat/budget.py tests/test_budget.py && git commit -m "feat: spend ledger with hard cap"`

---

### Task 16: Web search client (Brave) with JSON cache

**Files:**
- Create: `src/txcat/web_search.py`, `tests/test_web_search.py`, `cache/web_search/.gitkeep`

- [ ] **Step 1: Failing tests (HTTP mocked)**

`tests/test_web_search.py`:
```python
import json

import pytest

from txcat.budget import SpendLedger
from txcat.web_search import CacheMissError, SearchResult, WebSearchClient

BRAVE_PAYLOAD = {"web": {"results": [
    {"title": "WW Grainger - Industrial Supply", "description": "Grainger is a distributor of MRO supplies.", "url": "https://www.grainger.com/"},
    {"title": "Grainger wiki", "description": "W. W. Grainger, Inc. is an American industrial supply company.", "url": "https://en.wikipedia.org/wiki/W._W._Grainger"},
]}}


class FakeHTTP:
    def __init__(self):
        self.calls = 0

    def get(self, url, params, headers, timeout):
        self.calls += 1
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return BRAVE_PAYLOAD
        return R()


def test_search_caches_and_bills_once(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 10)
    http = FakeHTTP()
    c = WebSearchClient("brave", tmp_path / "ws", api_key="k", ledger=led, http=http, min_interval_s=0)
    r1 = c.search("WW GRAINGER", num_results=5)
    assert [x.url for x in r1][:1] == ["https://www.grainger.com/"]
    assert isinstance(r1[0], SearchResult) and r1[0].snippet.startswith("Grainger")
    r2 = c.search("WW GRAINGER", num_results=5)
    assert http.calls == 1 and r2 == r1
    assert led.total == pytest.approx(0.005)
    files = list((tmp_path / "ws").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["query"] == "WW GRAINGER" and payload["provider"] == "brave" and "timestamp" in payload
    assert set(payload["results"][0]) == {"title", "snippet", "url"}


def test_reproduce_mode_never_calls_network(tmp_path):
    c = WebSearchClient("brave", tmp_path / "ws", api_key=None, ledger=None, http=FakeHTTP(), allow_live=False)
    with pytest.raises(CacheMissError):
        c.search("UNSEEN")


def test_query_is_only_the_merchant_string(tmp_path):
    c = WebSearchClient("brave", tmp_path / "ws", api_key="k", ledger=SpendLedger(tmp_path / "l", 1),
                        http=FakeHTTP(), min_interval_s=0)
    with pytest.raises(ValueError):
        c.search("STAPLES $45.00 2024-01-01")  # digits that look like an amount/date are rejected
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement `src/txcat/web_search.py`**

```python
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
                 http=None, allow_live: bool = True, min_interval_s: float = 1.1, price_per_1k: float = 5.0):
        if provider != "brave":
            raise ValueError("only provider='brave' is implemented for evidence retrieval")
        self.provider, self.dir = provider, Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.api_key, self.ledger, self.http = api_key, ledger, http or requests
        self.allow_live, self.min_interval_s, self.price = allow_live, min_interval_s, price_per_1k / 1000
        self._last = 0.0

    def _path(self, query: str) -> Path:
        return self.dir / f"{sha1_text(f'{self.provider}|{query}')}.json"

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        if _AMOUNT_OR_DATE.search(query):
            raise ValueError("query looks like it contains an amount or date; only merchant names are allowed")
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
            r = self.http.get(BRAVE_URL, params={"q": query, "count": num_results},
                              headers={"Accept": "application/json", "X-Subscription-Token": self.api_key}, timeout=30)
            if getattr(r, "status_code", 200) == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        self._last = time.time()
        raw = r.json().get("web", {}).get("results", [])
        results = [SearchResult(title=x.get("title", ""), snippet=x.get("description", ""), url=x.get("url", ""))
                   for x in raw[:num_results]]
        p.write_text(json.dumps({"query": query, "provider": self.provider, "timestamp": now_iso(),
                                 "results": [asdict(x) for x in results]}, ensure_ascii=False))
        logger.debug(f"search {query!r}: {len(results)} results")
        return results
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git add src/txcat/web_search.py tests/test_web_search.py cache/web_search/.gitkeep && git commit -m "feat: Brave web search client with committed JSON cache"`

---

### Task 17: LLM fallback (OpenAI-compatible), prompts, JSON cache

Path hygiene: config paths are absolute in memory (root-anchored). Anything written into a COMMITTED artifact must be repo-relative. Add this helper to `src/txcat/config.py` as the first step of this task (with a one-line test in tests/test_config.py):

```python
def rel_to_root(p: str | Path) -> str:
    """Repo-relative POSIX string for paths written into committed artifacts (caches, manifests)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return p.as_posix()
```

Provider notes (verified 2026-09-12): Together's structured output uses `response_format={"type":"json_schema","json_schema":{"name":..., "schema":...}}` with NO `strict` key; OpenAI accepts the same plus `"strict": true`. The open-weight row is `Qwen/Qwen3.5-9B` on Together ($0.17/$0.25 per 1M, structured outputs supported); Llama 3.1 8B is no longer served there. gpt-5-nano's current snapshot is `gpt-5-nano-2025-08-07`.

**Files:**
- Create: `prompts/fallback_with_web.txt`, `prompts/fallback_no_web.txt`, `src/txcat/llm_fallback.py`, `tests/test_llm_fallback.py`, `cache/llm/.gitkeep`

- [ ] **Step 1: Write the two prompts (identical except the evidence block)**

`prompts/fallback_no_web.txt`:
```
You categorize the merchant behind a card transaction descriptor.

Merchant descriptor (normalized): {merchant}

Choose exactly one category from this list:
{taxonomy}

Return JSON only, with keys:
- "category": one of the category names above, copied exactly
- "confidence": a number from 0 to 1 for how sure you are
- "evidence": one sentence naming what you believe the merchant is and why
```

`prompts/fallback_with_web.txt`:
```
You categorize the merchant behind a card transaction descriptor.

Merchant descriptor (normalized): {merchant}

Web search results for this descriptor:
{evidence_block}

Choose exactly one category from this list:
{taxonomy}

Return JSON only, with keys:
- "category": one of the category names above, copied exactly
- "confidence": a number from 0 to 1 for how sure you are
- "evidence": one sentence naming what you believe the merchant is and why, citing a result number if used
```

- [ ] **Step 2: Failing tests (client mocked)**

`tests/test_llm_fallback.py`:
```python
import json

import pytest

from txcat.budget import SpendLedger
from txcat.config import LLMModelCfg
from txcat.llm_fallback import LLMFallback, render_prompt, CacheMissError
from txcat.web_search import SearchResult

TAX = ["groceries", "restaurants", "industrial_hardware"]
MODEL = LLMModelCfg(name="gpt-5-nano", provider="openai", api_key_env="OPENAI_API_KEY",
                    price_in_per_1m=0.05, price_out_per_1m=0.40, reasoning_effort="minimal", json_mode="json_schema")


class FakeCompletions:
    def __init__(self, content):
        self.content, self.calls, self.last_kwargs = content, 0, None

    def create(self, **kw):
        self.calls += 1
        self.last_kwargs = kw
        class Msg: pass
        class Choice: pass
        class Usage: prompt_tokens = 120; completion_tokens = 30
        class Resp: pass
        m = Msg(); m.content = self.content
        c = Choice(); c.message = m
        r = Resp(); r.choices = [c]; r.usage = Usage(); r.model = "gpt-5-nano-2025-08-07"
        return r


class FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {})()
        self.chat.completions = FakeCompletions(content)


def test_render_prompt_with_and_without_evidence(tmp_path):
    ev = [SearchResult("Grainger", "MRO supplies distributor", "https://grainger.com")]
    with_web = render_prompt("prompts/fallback_with_web.txt", "WW GRAINGER", ev, TAX)
    no_web = render_prompt("prompts/fallback_no_web.txt", "WW GRAINGER", None, TAX)
    assert "1. Grainger — MRO supplies distributor (https://grainger.com)" in with_web
    assert "Web search results" not in no_web and "WW GRAINGER" in no_web
    assert "- groceries\n- restaurants\n- industrial_hardware" in with_web


def test_categorize_parses_caches_and_bills(tmp_path):
    client = FakeClient(json.dumps({"category": "industrial_hardware", "confidence": 0.9, "evidence": "Grainger sells MRO."}))
    led = SpendLedger(tmp_path / "l.json", 5)
    fb = LLMFallback(MODEL, tmp_path / "llm", "prompts/fallback_no_web.txt", ledger=led, client=client)
    r = fb.categorize("WW GRAINGER", None, TAX)
    assert r.category == "industrial_hardware" and r.confidence == 0.9 and r.valid
    assert r.tokens_in == 120 and r.tokens_out == 30 and r.model_id == "gpt-5-nano-2025-08-07"
    assert client.chat.completions.last_kwargs["model"] == "gpt-5-nano"
    assert client.chat.completions.last_kwargs["reasoning_effort"] == "minimal"
    assert led.total == pytest.approx(120 / 1e6 * 0.05 + 30 / 1e6 * 0.40)
    r2 = fb.categorize("WW GRAINGER", None, TAX)
    assert client.chat.completions.calls == 1 and r2.category == r.category
    assert len(list((tmp_path / "llm").glob("*.json"))) == 1


def test_invalid_category_is_flagged_not_crashed(tmp_path):
    client = FakeClient('{"category": "Hardware Store", "confidence": 0.5, "evidence": "x"}')
    fb = LLMFallback(MODEL, tmp_path / "llm", "prompts/fallback_no_web.txt", ledger=None, client=client)
    r = fb.categorize("X", None, TAX)
    assert r.valid is False and r.category == "INVALID"


def test_reproduce_mode(tmp_path):
    fb = LLMFallback(MODEL, tmp_path / "llm", "prompts/fallback_no_web.txt", ledger=None,
                     client=FakeClient("{}"), allow_live=False)
    with pytest.raises(CacheMissError):
        fb.categorize("X", None, TAX)
```

- [ ] **Step 3: Run, verify fail**
- [ ] **Step 4: Implement `src/txcat/llm_fallback.py`**

```python
"""LLM fallback categorizer over an OpenAI-compatible chat API (OpenAI, Together via base_url).
Cache key = sha256(model | prompt_file_hash | rendered prompt). Reproduce mode never calls the API."""

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
    "properties": {"category": {"type": "string"}, "confidence": {"type": "number"}, "evidence": {"type": "string"}},
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


def render_prompt(template_path: str | Path, merchant: str, evidence: list[SearchResult] | None,
                  taxonomy: list[str]) -> str:
    tpl = Path(template_path).read_text()
    block = "\n".join(f"{i}. {r.title} — {r.snippet} ({r.url})" for i, r in enumerate(evidence or [], 1)) or "(no results)"
    return tpl.format(merchant=merchant, taxonomy="\n".join(f"- {t}" for t in taxonomy), evidence_block=block)


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class LLMFallback:
    def __init__(self, model: LLMModelCfg, cache_dir: str | Path, prompt_path: str | Path, ledger=None,
                 client=None, allow_live: bool = True, max_completion_tokens: int = 300):
        self.m, self.dir, self.prompt_path = model, Path(cache_dir), Path(prompt_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prompt_hash = sha256_text(self.prompt_path.read_text())
        self.ledger, self._client, self.allow_live = ledger, client, allow_live
        self.max_completion_tokens = max_completion_tokens

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get(self.m.api_key_env)
            if not key:
                raise RuntimeError(f"{self.m.api_key_env} missing")
            self._client = OpenAI(api_key=key, base_url=self.m.base_url)
        return self._client

    def _request_kwargs(self, prompt: str) -> dict:
        kw: dict = {"model": self.m.name, "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": self.max_completion_tokens}
        if self.m.reasoning_effort:
            kw["reasoning_effort"] = self.m.reasoning_effort
        if self.m.json_mode == "json_schema":
            js = {"name": "categorization", "schema": SCHEMA}
            if self.m.provider == "openai":
                js["strict"] = True  # Together rejects the strict key
            kw["response_format"] = {"type": "json_schema", "json_schema": js}
        elif self.m.json_mode == "json_object":
            kw["response_format"] = {"type": "json_object"}
        return kw

    def categorize(self, merchant: str, evidence: list[SearchResult] | None, taxonomy: list[str]) -> LLMResult:
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
        est_usd = (len(prompt) / 4) / 1e6 * self.m.price_in_per_1m + 150 / 1e6 * self.m.price_out_per_1m
        if self.ledger is not None:
            self.ledger.add("llm", est_usd, self.m.name)
        t0 = time.perf_counter()
        resp = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**kw)
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{self.m.name} attempt {attempt}: {e}")
                time.sleep(2 ** attempt)
        if resp is None:
            raise RuntimeError(f"LLM call failed for {merchant!r}")
        latency = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        try:
            parsed = _extract_json(text)
            cat = str(parsed.get("category", "")).strip()
            valid = cat in taxonomy
            res = LLMResult(category=cat if valid else "INVALID", confidence=float(parsed.get("confidence", 0.0)),
                            evidence=str(parsed.get("evidence", "")), valid=valid, model_id=getattr(resp, "model", self.m.name),
                            tokens_in=int(resp.usage.prompt_tokens), tokens_out=int(resp.usage.completion_tokens),
                            latency_ms=latency, cached=False)
        except Exception:  # noqa: BLE001
            res = LLMResult("INVALID", 0.0, text[:200], False, getattr(resp, "model", self.m.name),
                            int(resp.usage.prompt_tokens), int(resp.usage.completion_tokens), latency, False)
        if self.ledger is not None:  # true-up: replace estimate with actual usage
            actual = res.tokens_in / 1e6 * self.m.price_in_per_1m + res.tokens_out / 1e6 * self.m.price_out_per_1m
            self.ledger.entries[-1]["usd"] = actual
            self.ledger.path.write_text(json.dumps(self.ledger.entries))
        p.write_text(json.dumps({"model": self.m.name, "prompt_hash": self.prompt_hash, "prompt_file": rel_to_root(self.prompt_path),
                                 "merchant": merchant, "timestamp": now_iso(), "request": kw, "raw_response": text,
                                 "result": asdict(res)}, ensure_ascii=False))
        return res
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit** — `git add prompts/fallback_*.txt src/txcat/llm_fallback.py tests/test_llm_fallback.py cache/llm/.gitkeep && git commit -m "feat: LLM fallback client with prompts and JSON cache"`

---

### Task 18: 100-merchant pilot, freeze prompts, pin model ids

**Files:**
- Create: `experiments/pilot.py`, outputs `results/pilot/pilot_results.csv`, `prompts/PROMPT_HASHES.json`, `prompts/model_versions.json`

Prerequisite: `.env` contains `OPENAI_API_KEY`, `BRAVE_API_KEY`, (optionally) `TOGETHER_API_KEY`. Spend for this task is about $1.

- [ ] **Step 1: Write `experiments/pilot.py`**

```python
"""Pilot: 100 seeded tail merchants -> Brave search -> each LLM with and without web. Reports accuracy
per condition and spend, then freezes prompt hashes and pins the model snapshot ids.

Usage: python -m experiments.pilot --config configs/dc.yaml --n 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from txcat.budget import SpendLedger
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import merchant_frequencies, sample_fes, temporal_split
from txcat.data.taxonomy import CATEGORIES
from txcat.llm_fallback import LLMFallback
from txcat.utils import set_seed, setup_logging, sha256_text
from txcat.web_search import WebSearchClient


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names to run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    setup_logging(cfg.logs_dir, "pilot")
    n = args.n or cfg.fes.pilot_n

    d = cfg.datasets["dc"]
    win = WindowCfg(**json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"])
    df = load_prepared(d.processed_path, cfg.taxonomy_dir)
    df = df[df["category"].notna()]
    train, test = temporal_split(df, win)
    freqs = merchant_frequencies(train)
    fes = sample_fes(test, freqs, cfg.tail_k, n_tail=n, n_head=0, seed=cfg.seed)
    merchants = fes.drop_duplicates("merchant")[["merchant", "category"]].reset_index(drop=True)
    logger.info(f"pilot on {len(merchants)} tail merchants")

    ledger = SpendLedger(cfg.budget.ledger_path, cfg.budget.max_usd)
    ws = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, os.environ.get(cfg.search.api_key_env),
                         ledger=ledger, min_interval_s=cfg.search.min_interval_s, price_per_1k=cfg.search.price_per_1k_usd)
    evidence = {m: ws.search(m, cfg.search.num_results) for m in merchants["merchant"]}
    logger.info(f"search done; empty results for {sum(1 for v in evidence.values() if not v)} merchants; spend {ledger.total:.3f}")

    rows, model_ids = [], {}
    models = [m for m in cfg.llm.models if not args.models or m.name in args.models]
    for mcfg in models:
        if not os.environ.get(mcfg.api_key_env):
            logger.warning(f"skipping {mcfg.name}: {mcfg.api_key_env} not set")
            continue
        for cond, prompt in (("no_web", cfg.llm.prompt_no_web), ("with_web", cfg.llm.prompt_with_web)):
            fb = LLMFallback(mcfg, cfg.llm.cache_dir, prompt, ledger=ledger, max_completion_tokens=cfg.llm.max_completion_tokens)
            for _, r in merchants.iterrows():
                ev = evidence[r["merchant"]] if cond == "with_web" else None
                res = fb.categorize(r["merchant"], ev, CATEGORIES)
                model_ids[mcfg.name] = res.model_id
                rows.append({"model": mcfg.name, "condition": cond, "merchant": r["merchant"], "gold": r["category"],
                             "pred": res.category, "correct": res.category == r["category"], "valid": res.valid,
                             "confidence": res.confidence, "latency_ms": res.latency_ms,
                             "tokens_in": res.tokens_in, "tokens_out": res.tokens_out})
            logger.info(f"{mcfg.name} {cond} done; spend so far {ledger.total:.3f} USD")

    out = pd.DataFrame(rows)
    Path(cfg.results_dir, "pilot").mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(cfg.results_dir, "pilot", "pilot_results.csv"), index=False)
    summary = out.groupby(["model", "condition"]).agg(acc=("correct", "mean"), invalid=("valid", lambda s: 1 - s.mean()),
                                                     p50_ms=("latency_ms", "median"), n=("correct", "size"))
    logger.info("\n" + summary.to_string())
    logger.info(f"spend by kind: {ledger.by_kind()}  total {ledger.total:.3f} USD")

    Path("prompts/PROMPT_HASHES.json").write_text(json.dumps(
        {p: sha256_text(Path(p).read_text()) for p in (cfg.llm.prompt_no_web, cfg.llm.prompt_with_web)}, indent=2))
    versions = {"embedding_models": {b: b for b in cfg.embed.backbones} | {"openai": cfg.embed.openai_model},
                "llm_models": model_ids, "web_search": {"provider": cfg.search.provider, "pilot_date": pd.Timestamp.now("UTC").date().isoformat()}}
    Path("prompts/model_versions.json").write_text(json.dumps(versions, indent=2))
    logger.info("prompts frozen (PROMPT_HASHES.json) and model ids pinned (model_versions.json)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the pilot** — `.venv/bin/python -m experiments.pilot --config configs/dc.yaml --n 100`
Expected: summary table with `with_web` accuracy above `no_web` for each model on these tail merchants, invalid rate < 5%, total spend ≈ $0.6–1.0. If invalid rate is high for the Together model, switch its `json_mode` to `none` in `configs/default.yaml` and rerun (cache keys include the prompt, not the response_format, so results are reused).

- [ ] **Step 3: Freeze decision** — read 10 wrong `with_web` rows in `results/pilot/pilot_results.csv`. If the prompt wording itself is at fault (not evidence quality), edit the prompt ONCE, delete `cache/llm/*.json`, rerun step 2. After this point the prompts are frozen: any later edit must be a new file (used only by the prompt-sensitivity ablation).

- [ ] **Step 4: Commit** — `git add experiments/pilot.py results/pilot/pilot_results.csv prompts/PROMPT_HASHES.json prompts/model_versions.json cache/web_search cache/llm && git commit -m "feat: 100-merchant pilot; freeze prompts and pin model ids"`

---

## Day 1 exit criteria

- `.venv/bin/pytest` green (≥ 14 test files).
- `results/window_decision.json` says `primary_ok: true` (or documents the slide).
- `results/tables/tab0_*.csv`, `tab1_*.csv`; `results/figures/fig1–fig3` PDF+PNG.
- `prompts/PROMPT_HASHES.json` and `prompts/model_versions.json` exist; `cache/spend_ledger.json` total < $2.
- Plan B (Day 2: generator, FES full cache pass, Exp 2, write-back, stream, Exp 3, reproduce.py, README) is written next, informed by the pilot numbers.
