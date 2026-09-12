# longtail-txcat Day 2: Cache Pass, Exp 2, Streaming, Reproduce Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the one expensive cache pass on the fallback evaluation set, then produce every Exp 2 and Exp 3 table and figure from cache, add the generator for the tail-severity sweep, and make `reproduce.py` regenerate everything with zero live calls.

**Architecture:** Reuses Day 1 modules unchanged. Adds `generator/` (NSI vocab + noise + Zipf), `metrics.py`, `baselines.py`, `writeback.py`, `stream.py`, a resumable `exp2_cache_pass` that is the only script allowed to make live calls, and offline `exp2_fallback_comparison` / `exp3_streaming_convergence` that re-mix cached results. `reproduce.py` runs Exp 0–3 in `--mode reproduce`.

**Tech Stack:** as Day 1, plus `scipy` (bootstrap), `openai` Responses API for the 300-merchant agentic-search ablation.

**Spec:** `docs/superpowers/specs/2026-09-12-longtail-txcat-design.md` sections 3, 4.7–4.9, 5, 6, 7.

---

## File structure

```
src/txcat/generator/__init__.py
src/txcat/generator/vocab.py          download pinned NSI, filter US brands, map OSM tags -> 15 categories
src/txcat/generator/noise.py          descriptor noise templates (prefix, ref code, store #, city/state, truncate, pad, abbreviate)
src/txcat/generator/generate.py       Zipf sampling -> processed-schema parquet (+ category, merchant_id, alpha)
data/taxonomy/osm_tag_to_category.csv committed OSM tag -> category map
src/txcat/metrics.py                  macro_f1 (masked), latency_p50_p95, cost_per_1k, bootstrap_ci, paired_bootstrap_diff
src/txcat/baselines.py                embeddings + logistic regression
src/txcat/writeback.py                WriteBackPolicy
src/txcat/stream.py                   run_stream -> StreamLog
src/txcat/agentic_search.py           OpenAI Responses web_search ablation client (300 merchants, cached)
src/txcat/fes.py                      build/load the frozen FES merchant lists (DC + Oklahoma cold-start)
experiments/exp2_cache_pass.py        LIVE: search + LLM on every FES merchant, resumable, spend-capped
experiments/exp2_fallback_comparison.py  OFFLINE: tab2, tab2b, tab3, tab3b, tab3c, fig4
experiments/exp3_streaming_convergence.py OFFLINE: fig5-7, tab5
experiments/exp1_generator_sweep.py   OFFLINE ($0): kNN tail accuracy vs alpha -> fig1b/tab1b
reproduce.py                          runs exp0, exp1, exp1_generator_sweep, exp2_fallback_comparison, exp3 in reproduce mode; logs versions
analysis/plots.py                     + fig_frontier, fig_stream_*
prompts/fallback_with_web_v2.txt, fallback_no_web_v2.txt   prompt-sensitivity variants (wording only)
```

### Task B1: Generator vocabulary from NSI (pinned) + OSM tag map

**Files:**
- Create: `src/txcat/generator/__init__.py`, `src/txcat/generator/vocab.py`, `data/taxonomy/osm_tag_to_category.csv`, `tests/test_generator_vocab.py`

NSI bulk file (pinned): `https://cdn.jsdelivr.net/npm/name-suggestion-index@8.0.20260729/dist/json/nsi.min.json` (12 MB).
Structure: `{"nsi": {"<tree>/<key>/<value>": {"items": [{"displayName", "id", "locationSet": {"include": [...]}, "tags": {...}}]}}}`.
Keep items whose `locationSet.include` contains `"us"` or `"001"` (worldwide) and whose path starts with `brands/`.

- [ ] **Step 1: Write the tag map** `data/taxonomy/osm_tag_to_category.csv` (columns `osm_key,osm_value,category`). Rows:

```
osm_key,osm_value,category
shop,supermarket,groceries
shop,convenience,groceries
shop,greengrocer,groceries
shop,bakery,groceries
shop,butcher,groceries
shop,deli,groceries
shop,frozen_food,groceries
shop,health_food,groceries
shop,alcohol,groceries
shop,beverages,groceries
shop,wholesale,retail
amenity,restaurant,restaurants
amenity,fast_food,restaurants
amenity,cafe,restaurants
amenity,bar,restaurants
amenity,pub,restaurants
amenity,ice_cream,restaurants
amenity,food_court,restaurants
shop,coffee,restaurants
shop,department_store,retail
shop,variety_store,retail
shop,clothes,retail
shop,shoes,retail
shop,jewelry,retail
shop,furniture,retail
shop,houseware,retail
shop,gift,retail
shop,toys,entertainment_media
shop,sports,entertainment_media
shop,books,retail
shop,music,entertainment_media
shop,video_games,entertainment_media
shop,pet,retail
shop,florist,retail
shop,cosmetics,retail
shop,beauty,professional_services
shop,hairdresser,professional_services
shop,massage,health
shop,stationery,office_supplies
shop,copyshop,office_supplies
shop,printing,office_supplies
shop,computer,software_electronics
shop,electronics,software_electronics
shop,mobile_phone,telecom_utilities
shop,telecommunication,telecom_utilities
shop,hardware,industrial_hardware
shop,doityourself,industrial_hardware
shop,paint,industrial_hardware
shop,trade,industrial_hardware
shop,garden_centre,industrial_hardware
shop,car,transport_auto_fuel
shop,car_parts,transport_auto_fuel
shop,car_repair,transport_auto_fuel
shop,tyres,transport_auto_fuel
shop,fuel,transport_auto_fuel
amenity,fuel,transport_auto_fuel
amenity,car_rental,transport_auto_fuel
amenity,car_wash,transport_auto_fuel
amenity,parking,transport_auto_fuel
amenity,charging_station,transport_auto_fuel
tourism,hotel,lodging
tourism,motel,lodging
tourism,hostel,lodging
tourism,guest_house,lodging
amenity,pharmacy,health
shop,chemist,health
shop,optician,health
shop,medical_supply,health
shop,hearing_aids,health
amenity,clinic,health
amenity,dentist,health
amenity,doctors,health
amenity,hospital,health
amenity,veterinary,professional_services
healthcare,laboratory,health
amenity,bank,financial_postal_shipping
amenity,atm,financial_postal_shipping
amenity,money_transfer,financial_postal_shipping
office,insurance,financial_postal_shipping
office,financial,financial_postal_shipping
amenity,post_office,financial_postal_shipping
amenity,cinema,entertainment_media
amenity,theatre,entertainment_media
leisure,fitness_centre,entertainment_media
leisure,bowling_alley,entertainment_media
leisure,amusement_arcade,entertainment_media
amenity,gym,entertainment_media
shop,dry_cleaning,professional_services
shop,laundry,professional_services
shop,tailor,professional_services
shop,funeral_directors,professional_services
office,accountant,professional_services
office,lawyer,professional_services
office,estate_agent,financial_postal_shipping
office,employment_agency,professional_services
office,telecommunication,telecom_utilities
office,energy_supplier,telecom_utilities
amenity,school,education_gov_membership
amenity,college,education_gov_membership
amenity,university,education_gov_membership
amenity,childcare,education_gov_membership
amenity,kindergarten,education_gov_membership
amenity,language_school,education_gov_membership
amenity,driving_school,education_gov_membership
amenity,library,education_gov_membership
```

- [ ] **Step 2: Failing test**

`tests/test_generator_vocab.py`:
```python
import pandas as pd

from txcat.generator.vocab import items_to_vocab, load_tag_map

FAKE_NSI = {"nsi": {
    "brands/shop/supermarket": {"items": [
        {"displayName": "Safeway", "id": "safeway-1", "locationSet": {"include": ["us"]},
         "tags": {"brand": "Safeway", "name": "Safeway", "shop": "supermarket", "brand:wikidata": "Q1"}},
        {"displayName": "Tesco", "id": "tesco-1", "locationSet": {"include": ["gb"]},
         "tags": {"brand": "Tesco", "name": "Tesco", "shop": "supermarket"}},
    ]},
    "brands/amenity/cafe": {"items": [
        {"displayName": "Starbucks", "id": "sb-1", "locationSet": {"include": ["001"]},
         "tags": {"brand": "Starbucks", "name": "Starbucks", "amenity": "cafe"}}]},
    "brands/shop/unknown_thing": {"items": [
        {"displayName": "Mystery", "id": "m-1", "locationSet": {"include": ["us"]},
         "tags": {"brand": "Mystery", "name": "Mystery", "shop": "unknown_thing"}}]},
    "operators/amenity/post_office": {"items": [
        {"displayName": "USPS", "id": "usps-1", "locationSet": {"include": ["us"]},
         "tags": {"operator": "USPS", "name": "USPS", "amenity": "post_office"}}]},
}}


def test_items_to_vocab_filters_us_and_maps_categories(tmp_path):
    tm = tmp_path / "osm_tag_to_category.csv"
    tm.write_text("osm_key,osm_value,category\nshop,supermarket,groceries\namenity,cafe,restaurants\n")
    v = items_to_vocab(FAKE_NSI, load_tag_map(tm))
    assert list(v.columns) == ["merchant_id", "name", "osm_key", "osm_value", "category", "wikidata"]
    assert set(v["name"]) == {"Safeway", "Starbucks"}  # Tesco not US, Mystery unmapped, USPS not brands/
    assert v.set_index("name").loc["Safeway", "category"] == "groceries"
    assert v.set_index("name").loc["Safeway", "wikidata"] == "Q1"
```

- [ ] **Step 3: Run, verify fail**
- [ ] **Step 4: Implement `src/txcat/generator/vocab.py`** (`__init__.py` empty)

```python
"""Real merchant vocabulary for the semi-synthetic benchmark, from the OSM name-suggestion-index."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

NSI_VERSION = "8.0.20260729"
NSI_URL = f"https://cdn.jsdelivr.net/npm/name-suggestion-index@{NSI_VERSION}/dist/json/nsi.min.json"
US_CODES = {"us", "001"}


def download_nsi(dest: str | Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(NSI_URL, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        logger.info(f"NSI {NSI_VERSION} -> {dest} ({len(r.content)/1e6:.1f} MB)")
    return dest


def load_tag_map(path: str | Path) -> dict[tuple[str, str], str]:
    df = pd.read_csv(path)
    return {(r.osm_key, r.osm_value): r.category for r in df.itertuples()}


def items_to_vocab(nsi: dict, tag_map: dict[tuple[str, str], str]) -> pd.DataFrame:
    rows = []
    for path, block in nsi["nsi"].items():
        if not path.startswith("brands/"):
            continue
        _, key, value = path.split("/", 2)
        cat = tag_map.get((key, value))
        if cat is None:
            continue
        for it in block.get("items", []):
            inc = set(it.get("locationSet", {}).get("include", []))
            if not (inc & US_CODES):
                continue
            name = it["tags"].get("brand") or it["tags"].get("name") or it["displayName"]
            rows.append({"merchant_id": it["id"], "name": name, "osm_key": key, "osm_value": value,
                         "category": cat, "wikidata": it["tags"].get("brand:wikidata", "")})
    v = pd.DataFrame(rows, columns=["merchant_id", "name", "osm_key", "osm_value", "category", "wikidata"])
    return v.drop_duplicates("name").sort_values("merchant_id").reset_index(drop=True)


def build_vocab(raw_dir: str | Path, tag_map_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    nsi = json.loads(download_nsi(Path(raw_dir) / f"nsi-{NSI_VERSION}.min.json").read_text())
    v = items_to_vocab(nsi, load_tag_map(tag_map_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    v.to_parquet(out_path, index=False)
    logger.info(f"vocab: {len(v)} US brands across {v['category'].nunique()} categories")
    return v
```

- [ ] **Step 5: Run tests, then build the real vocab** — `.venv/bin/python -c "from txcat.generator.vocab import build_vocab; v=build_vocab('data/raw/nsi','data/taxonomy/osm_tag_to_category.csv','data/processed/gen_vocab.parquet'); print(v.category.value_counts())"`. Expected several thousand brands; every one of the 15 categories present with ≥ 30 brands. If a category is thin, add more OSM tag rows for it to the tag map and rebuild.
- [ ] **Step 6: Commit** — `git add src/txcat/generator data/taxonomy/osm_tag_to_category.csv tests/test_generator_vocab.py && git commit -m "feat: generator vocabulary from pinned NSI"`

---

### Task B2: Descriptor noise + Zipf generator

**Files:**
- Create: `src/txcat/generator/noise.py`, `src/txcat/generator/generate.py`, `tests/test_generator.py`

- [ ] **Step 1: Failing tests**

`tests/test_generator.py`:
```python
import numpy as np
import pandas as pd

from txcat.data.schema import PROCESSED_COLUMNS
from txcat.generator.generate import generate
from txcat.generator.noise import noisy_descriptor
from txcat.normalizer import normalize_merchant


def test_noise_is_seeded_and_varied():
    rng = np.random.default_rng(0)
    outs = {noisy_descriptor("Blue Bottle Coffee", "San Francisco", "CA", rng) for _ in range(30)}
    assert len(outs) > 5
    rng2 = np.random.default_rng(0)
    assert noisy_descriptor("Blue Bottle Coffee", "San Francisco", "CA", rng2) in outs
    for o in outs:
        assert o == o.upper() and len(o) <= 40


def test_noise_survives_normalizer_most_of_the_time():
    rng = np.random.default_rng(1)
    hits = sum("BLUE BOTTLE" in normalize_merchant(noisy_descriptor("Blue Bottle Coffee", "Oakland", "CA", rng)).text
               for _ in range(200))
    assert hits >= 150  # truncation/abbreviation may drop it sometimes; most survive


def test_generate_schema_and_zipf():
    vocab = pd.DataFrame({"merchant_id": [f"m{i}" for i in range(300)], "name": [f"Brand {i}" for i in range(300)],
                          "osm_key": "shop", "osm_value": "x", "category": ["groceries", "retail", "health"] * 100,
                          "wikidata": ""})
    df = generate(vocab, n_transactions=5000, alpha=1.1, seed=3, start="2019-01-01", end="2020-12-31")
    assert list(df.columns) == PROCESSED_COLUMNS + ["category", "merchant_id", "alpha"]
    assert len(df) == 5000 and df["source"].iloc[0] == "gen"
    counts = df["merchant_id"].value_counts()
    assert counts.iloc[0] > 20 * counts.median()  # heavy head
    assert (counts <= 3).sum() > 50  # real tail exists
    assert df["date"].is_monotonic_increasing
    df2 = generate(vocab, n_transactions=5000, alpha=1.1, seed=3, start="2019-01-01", end="2020-12-31")
    assert df2["raw_merchant"].equals(df["raw_merchant"])
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement `src/txcat/generator/noise.py`**

```python
"""Templated bank-statement noise applied to a clean brand name. Patterns mirror those observed in
DC/Oklahoma descriptors: processor prefixes, alphanumeric reference codes, store numbers, city/state
suffixes, hard truncation at 22/25 chars, whitespace padding, vowel-dropped abbreviation."""

from __future__ import annotations

import re

import numpy as np

PREFIXES = ["SQ *", "TST* ", "PP*", "PAYPAL *", "PYPL*", "IN *", "DD *", "CLV*", "POS PURCHASE ", ""]
PREFIX_P = [0.08, 0.06, 0.04, 0.04, 0.02, 0.03, 0.03, 0.02, 0.03, 0.65]
ALNUM = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"))


def _ref_code(rng: np.random.Generator) -> str:
    n = int(rng.integers(5, 10))
    return "".join(rng.choice(ALNUM, size=n))


def _abbrev(name: str, rng: np.random.Generator) -> str:
    words = name.split()
    out = []
    for w in words:
        if len(w) > 5 and rng.random() < 0.5:
            w = w[0] + re.sub(r"[AEIOU]", "", w[1:].upper())
        out.append(w.upper())
    return " ".join(out)


def noisy_descriptor(name: str, city: str, state: str, rng: np.random.Generator) -> str:
    s = name.upper()
    if rng.random() < 0.15:
        s = _abbrev(name, rng)
    s = np.random.default_rng(int(rng.integers(0, 2**31))).choice(PREFIXES, p=PREFIX_P) + s
    r = rng.random()
    if r < 0.30:
        s += f" #{int(rng.integers(1, 9999))}"
    elif r < 0.55:
        s += f" {_ref_code(rng)}"
    elif r < 0.70:
        s += f" {int(rng.integers(100, 99999)):05d}"
    r = rng.random()
    if r < 0.35:
        s += f" {city.upper()} {state}"
    elif r < 0.50:
        s += f" {state}"
    if rng.random() < 0.20:
        s = s[: int(rng.choice([22, 25]))]
    if rng.random() < 0.10:
        parts = s.split(" ", 1)
        s = parts[0] + " " * int(rng.integers(3, 9)) + (parts[1] if len(parts) > 1 else "")
    return s.strip()[:40]
```

- [ ] **Step 4: Implement `src/txcat/generator/generate.py`**

```python
"""Zipf-distributed synthetic transactions over the NSI vocabulary, in the processed schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from txcat.data.schema import PROCESSED_COLUMNS
from txcat.generator.noise import noisy_descriptor

CITIES = [("Washington", "DC"), ("Oklahoma City", "OK"), ("Tulsa", "OK"), ("Seattle", "WA"), ("Austin", "TX"),
          ("Denver", "CO"), ("Chicago", "IL"), ("Atlanta", "GA"), ("Phoenix", "AZ"), ("Boston", "MA")]


def generate(vocab: pd.DataFrame, n_transactions: int, alpha: float, seed: int,
             start: str = "2019-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = vocab.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # random rank assignment
    ranks = np.arange(1, len(v) + 1)
    p = ranks ** (-alpha)
    p /= p.sum()
    idx = rng.choice(len(v), size=n_transactions, p=p)
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.sort(rng.integers(0, (pd.to_datetime(end) - pd.to_datetime(start)).days + 1, size=n_transactions)), unit="D")
    rows = []
    for i, k in enumerate(idx):
        city, state = CITIES[int(rng.integers(0, len(CITIES)))]
        rows.append({"txn_id": f"gen:{seed}:{i}", "date": dates[i], "raw_merchant": noisy_descriptor(v.loc[k, "name"], city, state, rng),
                     "mcc_description": "", "source": "gen", "category": v.loc[k, "category"],
                     "merchant_id": v.loc[k, "merchant_id"], "alpha": alpha})
    return pd.DataFrame(rows, columns=PROCESSED_COLUMNS + ["category", "merchant_id", "alpha"])


def generate_to_parquet(vocab_path: str | Path, out_path: str | Path, n_transactions: int, alpha: float, seed: int) -> pd.DataFrame:
    df = generate(pd.read_parquet(vocab_path), n_transactions, alpha, seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit** — `git add src/txcat/generator tests/test_generator.py && git commit -m "feat: descriptor noise and Zipf transaction generator"`

---

### Task B3: Metrics and the embeddings + logistic-regression baseline

**Files:**
- Create: `src/txcat/metrics.py`, `src/txcat/baselines.py`, `tests/test_metrics.py`, `tests/test_baselines.py`

- [ ] **Step 1: Failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pytest

from txcat.metrics import bootstrap_ci, cost_per_1k, latency_p50_p95, macro_f1, paired_bootstrap_diff


def test_macro_f1_with_mask():
    y = np.array(["a", "a", "b", "b", "c"])
    p = np.array(["a", "b", "b", "b", "c"])
    assert macro_f1(y, p) == pytest.approx((2/3 + 0.8 + 1.0) / 3, abs=1e-6)
    assert macro_f1(y, p, mask=np.array([True, True, False, False, False])) == pytest.approx(0.5 * (2/3 + 0.0), abs=1e-6)
    assert macro_f1(y, p, mask=np.zeros(5, bool)) != macro_f1(y, p)  # empty mask returns nan
    assert np.isnan(macro_f1(y, p, mask=np.zeros(5, bool)))


def test_latency_and_cost():
    p50, p95 = latency_p50_p95([10, 20, 30, 40, 1000])
    assert p50 == 30 and p95 > 500
    usd = cost_per_1k(n_txns=2000, tokens_in=1_000_000, tokens_out=100_000, search_calls=500,
                      price_in_per_1m=0.05, price_out_per_1m=0.40, search_price_per_1k=5.0)
    assert usd == pytest.approx((0.05 + 0.04 + 2.5) / 2, abs=1e-6)


def test_bootstrap_ci_and_paired_diff():
    lo, mean, hi = bootstrap_ci([0.5, 0.52, 0.48], n_boot=500, seed=0)
    assert lo <= mean <= hi and abs(mean - 0.5) < 0.02
    a = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], bool)
    b = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1], bool)
    d, lo, hi = paired_bootstrap_diff(a, b, n_boot=1000, seed=0)
    assert d == pytest.approx(0.4) and lo > 0.0  # a beats b, CI excludes zero
```

`tests/test_baselines.py`:
```python
import numpy as np
import pandas as pd

from txcat.baselines import EmbeddingLRBaseline
from txcat.embedder import Embedder, FakeHashBackend


def test_lr_baseline_fits_and_predicts(tmp_path):
    emb = Embedder("fake/m", tmp_path, backend=FakeHashBackend(32))
    train = pd.DataFrame({"merchant": [f"M{i}" for i in range(60)], "category": ["a", "b", "c"] * 20})
    clf = EmbeddingLRBaseline(emb).fit(train["merchant"].tolist(), train["category"].tolist())
    pred = clf.predict(["M0", "M1", "ZZZ"])
    assert len(pred) == 3 and pred[0] == "a" and pred[1] == "b"
    proba = clf.predict_proba(["M0"])
    assert proba.shape == (1, 3) and np.isclose(proba.sum(), 1.0)
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement `src/txcat/metrics.py`**

```python
"""Evaluation metrics: masked macro-F1, latency percentiles, cost per 1k transactions, bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred, mask=None) -> float:
    """Macro-F1 over the union of labels in the (masked) truth and predictions. NaN if mask selects nothing."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if mask is not None:
        y_true, y_pred = y_true[np.asarray(mask, bool)], y_pred[np.asarray(mask, bool)]
    if len(y_true) == 0:
        return float("nan")
    # labels = union of truth and prediction (sklearn default): a predicted-but-absent class counts as F1 = 0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def latency_p50_p95(timings_ms) -> tuple[float, float]:
    t = np.asarray(timings_ms, dtype=float)
    return float(np.percentile(t, 50)), float(np.percentile(t, 95))


def cost_per_1k(n_txns: int, tokens_in: int, tokens_out: int, search_calls: int,
                price_in_per_1m: float, price_out_per_1m: float, search_price_per_1k: float) -> float:
    """USD per 1,000 transactions given total token and search usage over ``n_txns`` transactions."""
    usd = tokens_in / 1e6 * price_in_per_1m + tokens_out / 1e6 * price_out_per_1m + search_calls / 1000 * search_price_per_1k
    return float(usd / n_txns * 1000)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(lo, mean, hi) percentile bootstrap over a small vector (e.g. per-seed metrics)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(v.mean()), float(np.quantile(means, 1 - a))


def paired_bootstrap_diff(correct_a, correct_b, n_boot: int = 5000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Paired bootstrap over units (merchants): resample indices, compute mean(a) - mean(b). Returns (diff, lo, hi)."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    al = (1 - level) / 2
    return float(a.mean() - b.mean()), float(np.quantile(diffs, al)), float(np.quantile(diffs, 1 - al))
```

- [ ] **Step 4: Implement `src/txcat/baselines.py`**

```python
"""Embeddings + logistic regression baseline (row 2 of the main table)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from txcat.embedder import Embedder


class EmbeddingLRBaseline:
    def __init__(self, emb: Embedder, C: float = 4.0, seed: int = 42):
        self.emb = emb
        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, merchants: list[str], labels: list[str]) -> EmbeddingLRBaseline:
        self.clf.fit(self.emb.embed(merchants), np.asarray(labels))
        return self

    def predict(self, merchants: list[str]) -> np.ndarray:
        return self.clf.predict(self.emb.embed(merchants))

    def predict_proba(self, merchants: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.emb.embed(merchants))
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit** — `git add src/txcat/metrics.py src/txcat/baselines.py tests/test_metrics.py tests/test_baselines.py && git commit -m "feat: metrics and embedding+LR baseline"`

---

### Task B4: Frozen FES lists and the resumable live cache pass (the one expensive step)

**Files:**
- Create: `src/txcat/fes.py`, `experiments/exp2_cache_pass.py`, `tests/test_fes.py`, outputs `results/fes/dc_fes.parquet`, `results/fes/oklahoma_coldstart_fes.parquet`

FES rules (spec §3): DC = all tail merchants in the test window if ≤ 2,500 else seeded sample of 2,500 tail + 500 head merchants, with all their test rows. Oklahoma cold-start = 800 seeded Oklahoma merchants that never appear in DC train (all frequency 0 in the DC index). Both frozen to parquet and committed so every method sees the same merchants.

- [ ] **Step 1: Failing test**

`tests/test_fes.py`:
```python
import pandas as pd

from txcat.fes import build_dc_fes, build_oklahoma_coldstart_fes


def test_dc_fes_takes_all_tail_when_small():
    test = pd.DataFrame({"merchant": ["a", "a", "b", "c", "d"], "ambiguous": [0] * 5, "category": ["x"] * 5})
    freqs = pd.Series({"a": 10, "b": 1})
    f = build_dc_fes(test, freqs, k=3, n_tail=2500, n_head=500, seed=1)
    assert set(f["merchant"]) == {"a", "b", "c", "d"} and f["is_tail"].sum() == 3


def test_oklahoma_coldstart_excludes_dc_merchants():
    ok = pd.DataFrame({"merchant": ["STAPLES", "LOCAL FARM CO", "TULSA WELDING", "STAPLES"], "ambiguous": [0] * 4,
                       "category": ["office_supplies", "groceries", "industrial_hardware", "office_supplies"]})
    dc_train_merchants = {"STAPLES"}
    f = build_oklahoma_coldstart_fes(ok, dc_train_merchants, n=800, seed=1)
    assert set(f["merchant"]) == {"LOCAL FARM CO", "TULSA WELDING"}
    assert (f["train_freq"] == 0).all() and f["is_tail"].all()
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement `src/txcat/fes.py`**

```python
"""Fallback evaluation sets: frozen merchant lists every API-backed method is evaluated on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from txcat.data.splits import freq_of, sample_fes


def build_dc_fes(test: pd.DataFrame, freqs: pd.Series, k: int, n_tail: int, n_head: int, seed: int) -> pd.DataFrame:
    t = test[test["ambiguous"] == 0]
    n_tail_avail = int(t.loc[freq_of(t, freqs) <= k, "merchant"].nunique())
    if n_tail_avail <= n_tail:
        return sample_fes(test, freqs, k, n_tail=n_tail_avail, n_head=n_head, seed=seed)
    return sample_fes(test, freqs, k, n_tail=n_tail, n_head=n_head, seed=seed)


def build_oklahoma_coldstart_fes(ok: pd.DataFrame, dc_train_merchants: set[str], n: int, seed: int) -> pd.DataFrame:
    t = ok[(ok["ambiguous"] == 0) & (~ok["merchant"].isin(dc_train_merchants))].copy()
    rng = np.random.default_rng(seed)
    uniq = np.sort(t["merchant"].unique())
    chosen = set(rng.choice(uniq, size=min(n, len(uniq)), replace=False))
    out = t[t["merchant"].isin(chosen)].reset_index(drop=True)
    out["train_freq"] = 0
    out["is_tail"] = True
    return out


def save_fes(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_fes(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Write `experiments/exp2_cache_pass.py`** (LIVE; resumable; spend-capped; the only script that may call APIs besides the pilot)

```python
"""Exp 2 cache pass (LIVE). Builds/loads the frozen FES lists, then for every FES merchant:
Brave search (cached), and each configured LLM in both conditions (cached). Safe to re-run; only misses
hit the network. Stops with BudgetExceeded before crossing the cap.

Usage: python -m experiments.exp2_cache_pass --config configs/dc.yaml [--stage fes|search|llm|all] [--models ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from txcat.budget import BudgetExceeded, SpendLedger
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import merchant_frequencies, temporal_split
from txcat.data.taxonomy import CATEGORIES
from txcat.fes import build_dc_fes, build_oklahoma_coldstart_fes, load_fes, save_fes
from txcat.llm_fallback import LLMFallback
from txcat.utils import set_seed, setup_logging
from txcat.web_search import WebSearchClient


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--stage", choices=["fes", "search", "llm", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    setup_logging(cfg.logs_dir, "exp2_cache_pass")
    fes_dir = Path(cfg.results_dir, "fes")
    dc_path, ok_path = fes_dir / "dc_fes.parquet", fes_dir / "oklahoma_coldstart_fes.parquet"

    if args.stage in ("fes", "all"):
        d = cfg.datasets["dc"]
        win = WindowCfg(**json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"])
        dc = load_prepared(d.processed_path, cfg.taxonomy_dir)
        dc = dc[dc["category"].notna()]
        train, test = temporal_split(dc, win)
        freqs = merchant_frequencies(train)
        if not dc_path.exists():
            save_fes(build_dc_fes(test, freqs, cfg.tail_k, cfg.fes.n_tail, cfg.fes.n_head, cfg.seed), dc_path)
        if not ok_path.exists():
            ok = load_prepared(cfg.datasets["oklahoma"].processed_path, cfg.taxonomy_dir)
            ok = ok[ok["category"].notna()]
            save_fes(build_oklahoma_coldstart_fes(ok, set(train["merchant"].unique()), cfg.fes.n_oklahoma, cfg.seed), ok_path)
        for p in (dc_path, ok_path):
            f = load_fes(p)
            logger.info(f"{p.name}: {f['merchant'].nunique()} merchants, {len(f)} rows, tail merchants {f.loc[f.is_tail,'merchant'].nunique()}")

    merchants = pd.concat([load_fes(dc_path), load_fes(ok_path)])["merchant"].drop_duplicates().sort_values().tolist()
    ledger = SpendLedger(cfg.budget.ledger_path, cfg.budget.max_usd)
    logger.info(f"{len(merchants)} unique FES merchants; spend so far {ledger.total:.2f} USD")

    ws = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, os.environ.get(cfg.search.api_key_env), ledger=ledger,
                         min_interval_s=cfg.search.min_interval_s, price_per_1k=cfg.search.price_per_1k_usd)
    if args.stage in ("search", "all"):
        try:
            for m in tqdm(merchants, desc="search"):
                ws.search(m, cfg.search.num_results)
        except BudgetExceeded as e:
            logger.error(f"STOPPED: {e}"); return
        logger.info(f"search stage complete; spend {ledger.total:.2f} USD")

    if args.stage in ("llm", "all"):
        ws_ro = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, None, allow_live=False)
        models = [m for m in cfg.llm.models if not args.models or m.name in args.models]
        try:
            for mcfg in models:
                if not os.environ.get(mcfg.api_key_env):
                    logger.warning(f"skip {mcfg.name}: {mcfg.api_key_env} unset"); continue
                for cond, prompt in (("no_web", cfg.llm.prompt_no_web), ("with_web", cfg.llm.prompt_with_web)):
                    fb = LLMFallback(mcfg, cfg.llm.cache_dir, prompt, ledger=ledger, max_completion_tokens=cfg.llm.max_completion_tokens)
                    for m in tqdm(merchants, desc=f"{mcfg.name}/{cond}"):
                        ev = ws_ro.search(m, cfg.search.num_results) if cond == "with_web" else None
                        fb.categorize(m, ev, CATEGORIES)
                    logger.info(f"{mcfg.name}/{cond} done; spend {ledger.total:.2f} USD")
        except BudgetExceeded as e:
            logger.error(f"STOPPED: {e}"); return
    logger.info(f"cache pass complete. spend by kind: {ledger.by_kind()}; total {ledger.total:.2f} USD")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Build FES lists only and inspect** — `.venv/bin/python -m experiments.exp2_cache_pass --stage fes`. Expected: DC FES ≈ 2,500 tail + 500 head merchants; Oklahoma 800 merchants. Commit the two parquet files.
- [ ] **Step 7 (author runs): the live pass** — `.venv/bin/python -m experiments.exp2_cache_pass --stage all`. ~1 hour for search at 1 rps, then LLM stages. Re-run the same command to resume after any interruption. Expected final spend ≈ $20.
- [ ] **Step 8: Commit** — `git add src/txcat/fes.py tests/test_fes.py experiments/exp2_cache_pass.py results/fes/*.parquet cache/web_search cache/llm cache/spend_ledger.json && git commit -m "feat: frozen FES lists and completed exp2 cache pass"`

---

### Task B5: Exp 2 offline analysis — main table, critical ablation, frontier, prompt sensitivity, agentic search

**Files:**
- Create: `src/txcat/agentic_search.py`, `prompts/fallback_with_web_v2.txt`, `prompts/fallback_no_web_v2.txt`, `experiments/exp2_fallback_comparison.py`, `tests/test_agentic_search.py`
- Modify: `analysis/plots.py` (add `fig_frontier`)

Method rows (all evaluated on the DC FES rows, headline slice = `ambiguous == 0`):
1. `knn/<backbone>` for each backbone in `cfg.embed.backbones` + `text-embedding-3-small`
2. `emb_lr/<first backbone>`
3. `llm_no_web/<model>` for each model
4. `llm_with_web/<model>` for each model
5. `routed/<backbone>/<model>@t` — kNN unless gate (threshold t) says fallback, then LLM-with-web. Reported at t = cfg.gate.threshold in tab2; swept 0.30–0.95 for fig4.

Latency: kNN latency measured live in this script (embed + query, per merchant, ms). LLM/search latency = `latency_ms` stored in the caches (live measurements from the cache pass). Cost: tokens from LLM cache + one search per fallback merchant, priced from config.

- [ ] **Step 1: Prompt-sensitivity variants** — write `prompts/fallback_no_web_v2.txt` and `prompts/fallback_with_web_v2.txt` with the SAME structure but reworded instructions:

`prompts/fallback_no_web_v2.txt`:
```
Task: assign a spending category to a merchant from a bank card statement.

Statement text for the merchant (already cleaned): {merchant}

Allowed categories (pick exactly one, copy the name verbatim):
{taxonomy}

Answer with a single JSON object with fields "category", "confidence" (0-1), and "evidence" (one sentence saying what the merchant is).
```

`prompts/fallback_with_web_v2.txt`:
```
Task: assign a spending category to a merchant from a bank card statement.

Statement text for the merchant (already cleaned): {merchant}

Search engine results for that text:
{evidence_block}

Allowed categories (pick exactly one, copy the name verbatim):
{taxonomy}

Answer with a single JSON object with fields "category", "confidence" (0-1), and "evidence" (one sentence saying what the merchant is; mention the result number you relied on, if any).
```

- [ ] **Step 2: Failing test for the agentic-search client**

`tests/test_agentic_search.py`:
```python
import json

from txcat.agentic_search import AgenticSearchCategorizer

TAX = ["groceries", "restaurants"]


class FakeResponses:
    def __init__(self):
        self.calls = 0
    def create(self, **kw):
        self.calls += 1
        class Out: pass
        r = Out(); r.output_text = json.dumps({"category": "groceries", "confidence": 0.8, "evidence": "Safeway is a supermarket [1]"})
        r.model = "gpt-5-mini-2025-08-07"
        class U: input_tokens = 9000; output_tokens = 40
        r.usage = U()
        r.output = [type("W", (), {"type": "web_search_call", "action": {"query": "SAFEWAY store"}})(),
                    type("M", (), {"type": "message"})()]
        return r


class FakeClient:
    def __init__(self): self.responses = FakeResponses()


def test_agentic_categorize_caches_queries_and_result(tmp_path):
    c = AgenticSearchCategorizer("gpt-5-mini", tmp_path, client=FakeClient(), price_in_per_1m=0.25, price_out_per_1m=2.0,
                                 search_price_per_call=0.01)
    r = c.categorize("SAFEWAY", TAX)
    assert r["category"] == "groceries" and r["queries"] == ["SAFEWAY store"] and r["n_searches"] == 1
    r2 = c.categorize("SAFEWAY", TAX)
    assert c.client.responses.calls == 1 and r2["category"] == "groceries"
    assert r["usd"] > 0.01
```

- [ ] **Step 3: Implement `src/txcat/agentic_search.py`**

```python
"""Ablation only: let the model run its own web searches (OpenAI Responses API `web_search` tool).
Cached like everything else; only the normalized merchant string enters the prompt."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from txcat.utils import now_iso, sha256_text

PROMPT = """You categorize the merchant behind a card transaction descriptor. Use web search to find out what
the merchant is, then answer.

Merchant descriptor (normalized): {merchant}

Choose exactly one category from this list:
{taxonomy}

Return JSON only, with keys "category" (copied exactly from the list), "confidence" (0-1), and "evidence" (one sentence)."""


class AgenticSearchCategorizer:
    def __init__(self, model: str, cache_dir: str | Path, client=None, price_in_per_1m: float = 0.25,
                 price_out_per_1m: float = 2.0, search_price_per_call: float = 0.01, ledger=None, allow_live: bool = True):
        self.model, self.dir = model, Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._client, self.ledger, self.allow_live = client, ledger, allow_live
        self.p_in, self.p_out, self.p_search = price_in_per_1m, price_out_per_1m, search_price_per_call

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
        resp = self.client.responses.create(model=self.model, input=prompt,
                                            tools=[{"type": "web_search", "search_context_size": "low"}])
        latency = (time.perf_counter() - t0) * 1000
        queries = [getattr(o, "action", {}).get("query", "") if isinstance(getattr(o, "action", None), dict)
                   else getattr(getattr(o, "action", None), "query", "")
                   for o in resp.output if getattr(o, "type", "") == "web_search_call"]
        text = resp.output_text or ""
        m = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(m.group(0)) if m else {}
        cat = str(parsed.get("category", "")).strip()
        usd = resp.usage.input_tokens / 1e6 * self.p_in + resp.usage.output_tokens / 1e6 * self.p_out + len(queries) * self.p_search
        if self.ledger is not None:
            self.ledger.entries[-1]["usd"] = usd
            self.ledger.path.write_text(json.dumps(self.ledger.entries))
        result = {"category": cat if cat in taxonomy else "INVALID", "valid": cat in taxonomy,
                  "confidence": float(parsed.get("confidence", 0.0)), "evidence": str(parsed.get("evidence", "")),
                  "queries": queries, "n_searches": len(queries), "model_id": getattr(resp, "model", self.model),
                  "tokens_in": resp.usage.input_tokens, "tokens_out": resp.usage.output_tokens, "latency_ms": latency, "usd": usd}
        p.write_text(json.dumps({"merchant": merchant, "model": self.model, "timestamp": now_iso(), "raw": text, "result": result}))
        return result
```

- [ ] **Step 4: Run test, verify pass**

- [ ] **Step 5: Add `fig_frontier` to `analysis/plots.py`**

```python
def fig_frontier(frontier: pd.DataFrame, stem: str | Path, knee_threshold: float | None = None,
                 title: str = "Accuracy–cost frontier over the gate threshold") -> None:
    """``frontier`` columns: threshold, macro_f1, cost_per_1k, model. One line per model, points labeled by threshold."""
    fig, ax = plt.subplots()
    colors = [style.SERIES["head"], style.SERIES["tail"], style.SERIES["third"], style.SERIES["fourth"]]
    for (model, g), c in zip(frontier.groupby("model", sort=False), colors, strict=False):
        g = g.sort_values("cost_per_1k")
        ax.plot(g["cost_per_1k"], g["macro_f1"], marker="o", color=c, label=model)
        for _, r in g.iloc[::3].iterrows():
            ax.annotate(f"{r['threshold']:.2f}", (r["cost_per_1k"], r["macro_f1"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color=style.INK2)
    if knee_threshold is not None:
        k = frontier[frontier["threshold"].round(2) == round(knee_threshold, 2)]
        ax.scatter(k["cost_per_1k"], k["macro_f1"], s=90, facecolors="none", edgecolors=style.INK, linewidths=1.2, zorder=5,
                   label=f"knee t={knee_threshold:.2f}")
    ax.set_xlabel("cost per 1k transactions (USD)")
    ax.set_ylabel("overall macro-F1")
    ax.set_title(title, loc="left", color=style.INK)
    ax.legend(loc="lower right")
    style.save(fig, stem)
    plt.close(fig)
```

- [ ] **Step 6: Write `experiments/exp2_fallback_comparison.py`** (OFFLINE: reads caches only)

```python
"""Exp 2 (offline). Every LLM/search number comes from cache; kNN is recomputed per seed from the embedding cache.

Usage: python -m experiments.exp2_fallback_comparison --config configs/dc.yaml --seeds 42 43 44 [--mode reproduce]
Outputs: tab2_main_results.csv, tab2b_oklahoma_coldstart.csv, tab3_critical_ablation.csv, tab3b_prompt_sensitivity.csv,
         tab3c_agentic_search.csv, fig4_acc_cost_frontier.{pdf,png}, exp2_frontier.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from analysis.plots import fig_frontier
from txcat.agentic_search import AgenticSearchCategorizer
from txcat.baselines import EmbeddingLRBaseline
from txcat.budget import SpendLedger
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import merchant_frequencies, temporal_split
from txcat.data.taxonomy import CATEGORIES
from txcat.embedder import Embedder
from txcat.fes import load_fes
from txcat.knn import build_merchant_index, predict_knn
from txcat.llm_fallback import LLMFallback
from txcat.metrics import bootstrap_ci, cost_per_1k, latency_p50_p95, macro_f1, paired_bootstrap_diff
from txcat.utils import set_seed, setup_logging
from txcat.web_search import WebSearchClient

THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.951, 0.05)]


def f1_row(df: pd.DataFrame, pred_col: str) -> dict:
    return {"f1_overall": macro_f1(df["category"], df[pred_col]),
            "f1_head": macro_f1(df["category"], df[pred_col], ~df["is_tail"]),
            "f1_tail": macro_f1(df["category"], df[pred_col], df["is_tail"])}


def llm_preds(fes_merchants: list[str], mcfg, prompt: str, cfg, ws_ro) -> pd.DataFrame:
    fb = LLMFallback(mcfg, cfg.llm.cache_dir, prompt, allow_live=False)
    rows = []
    for m in fes_merchants:
        ev = ws_ro.search(m, cfg.search.num_results) if "with_web" in Path(prompt).stem else None
        r = fb.categorize(m, ev, CATEGORIES)
        rows.append({"merchant": m, "pred": r.category, "conf": r.confidence, "lat": r.latency_ms, "tin": r.tokens_in, "tout": r.tokens_out})
    return pd.DataFrame(rows).set_index("merchant")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--mode", choices=["live", "reproduce"], default="reproduce")
    ap.add_argument("--agentic-n", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    setup_logging(cfg.logs_dir, "exp2")
    figs, tables = Path(cfg.results_dir, "figures"), Path(cfg.results_dir, "tables")
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    live = args.mode == "live"

    d = cfg.datasets["dc"]
    win = WindowCfg(**json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"])
    dc = load_prepared(d.processed_path, cfg.taxonomy_dir)
    dc = dc[dc["category"].notna()]
    train, _ = temporal_split(dc, win)
    fes = load_fes(Path(cfg.results_dir, "fes", "dc_fes.parquet"))
    fes = fes[fes["ambiguous"] == 0].reset_index(drop=True)
    ok_fes = load_fes(Path(cfg.results_dir, "fes", "oklahoma_coldstart_fes.parquet"))
    ok_fes = ok_fes[ok_fes["ambiguous"] == 0].reset_index(drop=True)
    merchants = fes["merchant"].drop_duplicates().tolist()
    ws_ro = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, None, allow_live=False)
    n_txn = len(fes)
    backbones = [*cfg.embed.backbones, cfg.embed.openai_model]

    # ---- LLM rows (cached) ----
    llm = {}  # (model, cond) -> DataFrame indexed by merchant
    for mcfg in cfg.llm.models:
        for cond, prompt in (("no_web", cfg.llm.prompt_no_web), ("with_web", cfg.llm.prompt_with_web)):
            try:
                llm[(mcfg.name, cond)] = llm_preds(merchants, mcfg, prompt, cfg, ws_ro)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"skipping {mcfg.name}/{cond}: {e}")

    rows, frontier, knn_cache = [], [], {}
    for bb in backbones:
        emb = Embedder(bb, cfg.embed.cache_dir, allow_live=live, batch_size=cfg.embed.batch_size)
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
            t0 = time.perf_counter()
            p = predict_knn(fes, idx, emb, cfg.index.k, cfg.index.ef_search)
            lat_ms = (time.perf_counter() - t0) * 1000 / max(1, len(merchants))
            per_seed.append((seed, fes.assign(pred=p["pred"].values, sim1=p["sim1"].values), lat_ms))
        knn_cache[bb] = per_seed
        f1s = [f1_row(t, "pred") for _, t, _ in per_seed]
        rows.append({"method": f"knn/{bb}", **{k: bootstrap_ci([f[k] for f in f1s])[1] for k in f1s[0]},
                     **{k + "_lo": bootstrap_ci([f[k] for f in f1s])[0] for k in f1s[0]},
                     **{k + "_hi": bootstrap_ci([f[k] for f in f1s])[2] for k in f1s[0]},
                     "p50_ms": np.mean([l for _, _, l in per_seed]), "p95_ms": np.mean([l for _, _, l in per_seed]),
                     "cost_per_1k": 0.0, "fallback_rate": 0.0, "n_seeds": len(seeds)})
        if bb == cfg.embed.backbones[0]:
            lr = EmbeddingLRBaseline(emb, seed=seeds[0]).fit(train["merchant"].tolist(), train["category"].tolist())
            t = fes.assign(pred=lr.predict(fes["merchant"].tolist()))
            rows.append({"method": f"emb_lr/{bb}", **f1_row(t, "pred"), "p50_ms": 0.0, "p95_ms": 0.0, "cost_per_1k": 0.0,
                         "fallback_rate": 0.0, "n_seeds": 1})

    for (model, cond), lp in llm.items():
        mcfg = next(m for m in cfg.llm.models if m.name == model)
        t = fes.assign(pred=fes["merchant"].map(lp["pred"]).values)
        p50, p95 = latency_p50_p95(lp["lat"])
        cost = cost_per_1k(n_txn, int(lp["tin"].sum()), int(lp["tout"].sum()), len(lp) if cond == "with_web" else 0,
                           mcfg.price_in_per_1m, mcfg.price_out_per_1m, cfg.search.price_per_1k_usd)
        rows.append({"method": f"llm_{cond}/{model}", **f1_row(t, "pred"), "p50_ms": p50, "p95_ms": p95,
                     "cost_per_1k": cost, "fallback_rate": 1.0, "n_seeds": 1})

    # ---- routed system: kNN + gate + LLM-with-web; sweep thresholds ----
    bb0 = cfg.embed.backbones[0]
    for model in {m for m, c in llm if c == "with_web"}:
        lp = llm[(model, "with_web")]
        mcfg = next(m for m in cfg.llm.models if m.name == model)
        for t_thr in THRESHOLDS:
            f1s, costs, fr = [], [], []
            for seed, t, lat in knn_cache[bb0]:
                fb_mask = t["sim1"] < t_thr
                routed = t["pred"].where(~fb_mask, t["merchant"].map(lp["pred"]))
                tt = t.assign(routed=routed)
                f1s.append(f1_row(tt, "routed"))
                fb_m = tt.loc[fb_mask, "merchant"].drop_duplicates()
                sub = lp.loc[fb_m]
                costs.append(cost_per_1k(n_txn, int(sub["tin"].sum()), int(sub["tout"].sum()), len(sub),
                                         mcfg.price_in_per_1m, mcfg.price_out_per_1m, cfg.search.price_per_1k_usd))
                fr.append(float(fb_mask.mean()))
            rec = {"model": model, "threshold": t_thr, "macro_f1": np.mean([f["f1_overall"] for f in f1s]),
                   "f1_tail": np.mean([f["f1_tail"] for f in f1s]), "f1_head": np.mean([f["f1_head"] for f in f1s]),
                   "cost_per_1k": np.mean(costs), "fallback_rate": np.mean(fr)}
            frontier.append(rec)
            if t_thr == round(cfg.gate.threshold, 2):
                rows.append({"method": f"routed/{bb0}/{model}@{t_thr}", "f1_overall": rec["macro_f1"], "f1_head": rec["f1_head"],
                             "f1_tail": rec["f1_tail"], "p50_ms": np.nan, "p95_ms": np.nan, "cost_per_1k": rec["cost_per_1k"],
                             "fallback_rate": rec["fallback_rate"], "n_seeds": len(seeds)})
    fr_df = pd.DataFrame(frontier)
    fr_df.to_csv(Path(cfg.results_dir, "exp2_frontier.csv"), index=False)
    # knee: largest second difference of F1 w.r.t. cost, per first model
    knee = None
    if not fr_df.empty:
        g = fr_df[fr_df["model"] == fr_df["model"].iloc[0]].sort_values("cost_per_1k")
        if len(g) > 4:
            d1 = np.gradient(g["macro_f1"].values, g["cost_per_1k"].values + 1e-9)
            knee = float(g["threshold"].values[int(np.argmax(-np.gradient(d1)))])
        fig_frontier(fr_df, figs / "fig4_acc_cost_frontier", knee_threshold=knee)
    pd.DataFrame(rows).to_csv(tables / "tab2_main_results.csv", index=False)

    # ---- critical ablation: tail-only, with-web vs no-web, paired bootstrap over merchants ----
    tail_m = fes.loc[fes["is_tail"]].drop_duplicates("merchant")[["merchant", "category"]].set_index("merchant")["category"]
    abl = []
    for model in {m for m, _ in llm}:
        if (model, "with_web") not in llm or (model, "no_web") not in llm:
            continue
        a = (llm[(model, "with_web")].loc[tail_m.index, "pred"] == tail_m).values
        b = (llm[(model, "no_web")].loc[tail_m.index, "pred"] == tail_m).values
        diff, lo, hi = paired_bootstrap_diff(a, b, seed=seeds[0])
        abl.append({"model": model, "n_tail_merchants": len(tail_m), "acc_with_web": a.mean(), "acc_no_web": b.mean(),
                    "diff": diff, "ci_lo": lo, "ci_hi": hi, "significant_95": bool(lo > 0 or hi < 0)})
    pd.DataFrame(abl).to_csv(tables / "tab3_critical_ablation.csv", index=False)

    # ---- Oklahoma cold-start rows: kNN (DC index) vs LLM with web ----
    okrows = []
    ok_m = ok_fes["merchant"].drop_duplicates().tolist()
    emb0 = Embedder(bb0, cfg.embed.cache_dir, allow_live=live)
    set_seed(seeds[0])
    idx0 = build_merchant_index(train, emb0, cfg.index.M, cfg.index.ef_construction, seeds[0])
    p = predict_knn(ok_fes, idx0, emb0, cfg.index.k, cfg.index.ef_search)
    t = ok_fes.assign(pred=p["pred"].values)
    okrows.append({"method": f"knn/{bb0}", "f1_overall": macro_f1(t["category"], t["pred"]), "n_merchants": len(ok_m)})
    for (model, cond), _ in llm.items():
        try:
            mcfg = next(m for m in cfg.llm.models if m.name == model)
            lp = llm_preds(ok_m, mcfg, cfg.llm.prompt_with_web if cond == "with_web" else cfg.llm.prompt_no_web, cfg, ws_ro)
            tt = ok_fes.assign(pred=ok_fes["merchant"].map(lp["pred"]).values)
            okrows.append({"method": f"llm_{cond}/{model}", "f1_overall": macro_f1(tt["category"], tt["pred"]), "n_merchants": len(ok_m)})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"oklahoma {model}/{cond} skipped: {e}")
    pd.DataFrame(okrows).to_csv(tables / "tab2b_oklahoma_coldstart.csv", index=False)

    # ---- prompt sensitivity: 2 variants x 300 tail merchants x each model (cached in live mode on first run) ----
    ps_m = tail_m.index[:300].tolist()
    ledger = SpendLedger(cfg.budget.ledger_path, cfg.budget.max_usd)
    ps = []
    for mcfg in cfg.llm.models:
        if live and not os.environ.get(mcfg.api_key_env):
            continue
        for cond, v1, v2 in (("no_web", cfg.llm.prompt_no_web, "prompts/fallback_no_web_v2.txt"),
                             ("with_web", cfg.llm.prompt_with_web, "prompts/fallback_with_web_v2.txt")):
            try:
                accs = []
                for prompt in (v1, v2):
                    fb = LLMFallback(mcfg, cfg.llm.cache_dir, prompt, ledger=ledger if live else None, allow_live=live)
                    ok_n = 0
                    for m in ps_m:
                        ev = ws_ro.search(m, cfg.search.num_results) if cond == "with_web" else None
                        ok_n += fb.categorize(m, ev, CATEGORIES).category == tail_m[m]
                    accs.append(ok_n / len(ps_m))
                ps.append({"model": mcfg.name, "condition": cond, "acc_v1": accs[0], "acc_v2": accs[1], "abs_diff": abs(accs[0] - accs[1]), "n": len(ps_m)})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"prompt sensitivity {mcfg.name}/{cond} skipped: {e}")
    pd.DataFrame(ps).to_csv(tables / "tab3b_prompt_sensitivity.csv", index=False)

    # ---- agentic search ablation: OpenAI built-in web_search on 300 tail merchants ----
    ag_model = next((m for m in cfg.llm.models if m.provider == "openai" and "mini" in m.name), None)
    if ag_model is not None:
        ag = AgenticSearchCategorizer(ag_model.name, Path(cfg.llm.cache_dir) / "agentic", ledger=ledger if live else None,
                                      price_in_per_1m=ag_model.price_in_per_1m, price_out_per_1m=ag_model.price_out_per_1m,
                                      allow_live=live)
        try:
            res = {m: ag.categorize(m, CATEGORIES) for m in ps_m[: args.agentic_n]}
            acc_ag = np.mean([res[m]["category"] == tail_m[m] for m in res])
            acc_fixed = np.mean([llm[(ag_model.name, "with_web")].loc[m, "pred"] == tail_m[m] for m in res])
            acc_none = np.mean([llm[(ag_model.name, "no_web")].loc[m, "pred"] == tail_m[m] for m in res])
            pd.DataFrame([{"model": ag_model.name, "n": len(res), "acc_agentic_search": acc_ag, "acc_fixed_snippets": acc_fixed,
                           "acc_no_web": acc_none, "mean_searches_per_merchant": np.mean([r["n_searches"] for r in res.values()]),
                           "usd_per_1k_merchants": 1000 * np.mean([r["usd"] for r in res.values()])}]
                         ).to_csv(tables / "tab3c_agentic_search.csv", index=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"agentic ablation skipped: {e}")
    logger.info("\n" + pd.DataFrame(rows).to_string())
    logger.info("\n" + pd.DataFrame(abl).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: First run in live mode** (author; fills the prompt-sensitivity and agentic caches, ~$8) — `.venv/bin/python -m experiments.exp2_fallback_comparison --config configs/dc.yaml --seeds 42 43 44 --mode live`
- [ ] **Step 8: Re-run in reproduce mode** and confirm identical tables — `--mode reproduce`, then `git diff --stat results/tables` shows no change.
- [ ] **Step 9: Commit** — `git add src/txcat/agentic_search.py prompts/*_v2.txt experiments/exp2_fallback_comparison.py tests/test_agentic_search.py analysis/plots.py results/tables/tab2*.csv results/tables/tab3*.csv results/figures/fig4_* results/exp2_frontier.csv cache/llm && git commit -m "feat: exp2 fallback comparison, ablations, frontier"`

---

### Task B6: Write-back policy, stream runner, Exp 3 streaming convergence

**Files:**
- Create: `src/txcat/writeback.py`, `src/txcat/stream.py`, `tests/test_writeback.py`, `tests/test_stream.py`, `experiments/exp3_streaming_convergence.py`
- Modify: `analysis/plots.py` (add `fig_stream_lines`)

- [ ] **Step 1: Failing tests**

`tests/test_writeback.py`:
```python
from txcat.llm_fallback import LLMResult
from txcat.writeback import WriteBackPolicy


def res(conf, valid=True):
    return LLMResult(category="retail" if valid else "INVALID", confidence=conf, evidence="", valid=valid,
                     model_id="m", tokens_in=1, tokens_out=1, latency_ms=1.0, cached=True)


def test_modes():
    assert WriteBackPolicy("never").should_write(res(0.99)) is False
    assert WriteBackPolicy("always").should_write(res(0.1)) is True
    assert WriteBackPolicy("always").should_write(res(0.9, valid=False)) is False  # never write INVALID
    g = WriteBackPolicy("confidence_gated", 0.8)
    assert g.should_write(res(0.85)) is True and g.should_write(res(0.79)) is False
```

`tests/test_stream.py`:
```python
import numpy as np
import pandas as pd

from txcat.embedder import Embedder, FakeHashBackend
from txcat.gate import ConfidenceGate
from txcat.index import HNSWIndex
from txcat.llm_fallback import LLMResult
from txcat.stream import run_stream
from txcat.writeback import WriteBackPolicy


class OracleFallback:
    """Stands in for LLMFallback: returns the gold label with confidence 0.9, or a wrong label for merchants in `wrong`."""
    def __init__(self, gold, wrong=()):
        self.gold, self.wrong, self.calls = gold, set(wrong), 0
    def categorize(self, merchant, evidence, taxonomy):
        self.calls += 1
        cat = "WRONGCAT" if merchant in self.wrong else self.gold[merchant]
        return LLMResult(cat, 0.9, "", True, "m", 100, 20, 50.0, False)


def _setup(tmp_path):
    emb = Embedder("fake/m", tmp_path, backend=FakeHashBackend(16))
    idx = HNSWIndex(dim=16, M=8, ef_construction=50, seed=1)
    v = emb.embed(["STAPLES"])
    idx.add(v, ["office_supplies"], ["STAPLES"], [50], pd.to_datetime(["2019-01-01"]))
    txns = pd.DataFrame({"txn_id": [f"t{i}" for i in range(6)],
                         "date": pd.to_datetime(["2024-01-0%d" % (i + 1) for i in range(6)]),
                         "merchant": ["STAPLES", "NEWCO", "NEWCO", "NEWCO", "OTHERCO", "STAPLES"],
                         "category": ["office_supplies", "retail", "retail", "retail", "health", "office_supplies"],
                         "ambiguous": [0] * 6})
    gold = dict(zip(txns["merchant"], txns["category"], strict=False))
    return emb, idx, txns, gold


def test_always_writeback_reduces_fallbacks(tmp_path):
    emb, idx, txns, gold = _setup(tmp_path)
    fb = OracleFallback(gold)
    log = run_stream(txns, idx, emb, ConfidenceGate("threshold", 0.99), fb, WriteBackPolicy("always"),
                     lambda m: None, window=2, price_in_per_1m=1.0, price_out_per_1m=1.0, search_price=0.005)
    # NEWCO falls back once, is written back, then is served by kNN twice; OTHERCO falls back once
    assert fb.calls == 2
    assert len(idx) == 3 and (idx.meta["source"] == "writeback").sum() == 2
    assert log.per_txn["fallback"].tolist() == [False, True, False, False, True, False]
    assert log.per_txn["pred"].tolist() == txns["category"].tolist()
    assert log.windows["fallback_rate"].tolist() == [0.5, 0.0, 0.5]
    assert log.summary["writeback_error_rate"] == 0.0 and log.summary["n_writebacks"] == 2


def test_never_writeback_keeps_falling_back_and_error_rate_counts_bad_writes(tmp_path):
    emb, idx, txns, gold = _setup(tmp_path)
    fb = OracleFallback(gold)
    log = run_stream(txns, idx, emb, ConfidenceGate("threshold", 0.99), fb, WriteBackPolicy("never"),
                     lambda m: None, window=2)
    assert fb.calls == 4 and len(idx) == 1
    emb, idx, txns, gold = _setup(tmp_path / "b")
    fb = OracleFallback(gold, wrong={"NEWCO"})
    log = run_stream(txns, idx, emb, ConfidenceGate("threshold", 0.99), fb, WriteBackPolicy("always"), lambda m: None, window=2)
    assert log.summary["writeback_error_rate"] == 0.5  # NEWCO written wrong, OTHERCO written right
    assert log.per_txn["pred"].tolist()[2] == "WRONGCAT"  # pollution propagates via kNN
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement `src/txcat/writeback.py`**

```python
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
```

- [ ] **Step 4: Implement `src/txcat/stream.py`**

```python
"""Temporal stream: kNN -> gate -> (search + LLM) -> optional write-back, with windowed tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from txcat.data.taxonomy import CATEGORIES
from txcat.embedder import Embedder
from txcat.gate import ConfidenceGate
from txcat.index import HNSWIndex
from txcat.metrics import macro_f1
from txcat.writeback import WriteBackPolicy


@dataclass
class StreamLog:
    per_txn: pd.DataFrame
    windows: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_stream(transactions: pd.DataFrame, index: HNSWIndex, emb: Embedder, gate: ConfidenceGate, fallback,
               writeback: WriteBackPolicy, evidence_fn: Callable[[str], list | None], window: int = 500,
               k: int = 5, ef_search: int = 64, price_in_per_1m: float = 0.0, price_out_per_1m: float = 0.0,
               search_price: float = 0.0, tail_k: int = 3) -> StreamLog:
    """Process ``transactions`` (sorted by date) one at a time. ``fallback.categorize(merchant, evidence, taxonomy)``
    must return an LLMResult-like object. ``evidence_fn(merchant)`` returns cached search results or None."""
    txns = transactions.sort_values("date", kind="stable").reset_index(drop=True)
    rows, cum_cost, n_wb, n_wb_wrong = [], 0.0, 0, 0
    written: dict[str, str] = {}
    for r in txns.itertuples(index=False):
        vec = emb.embed([r.merchant])[0]
        q = index.query(vec, k=k, ef_search=ef_search)
        fb = gate.should_fallback(q)
        pred, cost = q.labels[0], 0.0
        if fb:
            res = fallback.categorize(r.merchant, evidence_fn(r.merchant), CATEGORIES)
            pred = res.category if res.valid else q.labels[0]
            billed = res.tokens_in + res.tokens_out > 0  # an uncached miss (replay mode) costs nothing
            cost = (res.tokens_in / 1e6 * price_in_per_1m + res.tokens_out / 1e6 * price_out_per_1m + search_price) if billed else 0.0
            if writeback.should_write(res) and r.merchant not in written:
                index.add(vec.reshape(1, -1), [res.category], [r.merchant], [1], pd.to_datetime([r.date]), source="writeback")
                written[r.merchant] = res.category
                n_wb += 1
                n_wb_wrong += int(res.category != r.category)
        cum_cost += cost
        rows.append({"txn_id": r.txn_id, "date": r.date, "merchant": r.merchant, "category": r.category, "pred": pred,
                     "fallback": fb, "sim1": q.sims[0], "top1_freq": q.top1_freq, "cost": cost, "cum_cost": cum_cost,
                     "index_size": len(index)})
    per = pd.DataFrame(rows)
    per["correct"] = per["pred"] == per["category"]
    per["is_tail"] = per["top1_freq"] <= tail_k  # approximation for streaming: top-1 neighbour's frequency
    per["window"] = np.arange(len(per)) // window
    win = per.groupby("window").agg(start=("date", "min"), fallback_rate=("fallback", "mean"), acc=("correct", "mean"),
                                    cum_cost=("cum_cost", "last"), index_size=("index_size", "last"))
    win["cum_macro_f1"] = [macro_f1(per.loc[: (i + 1) * window - 1, "category"], per.loc[: (i + 1) * window - 1, "pred"])
                           for i in win.index]
    win["cum_macro_f1_tail"] = [macro_f1(per.loc[: (i + 1) * window - 1, "category"], per.loc[: (i + 1) * window - 1, "pred"],
                                         per.loc[: (i + 1) * window - 1, "is_tail"]) for i in win.index]
    summary = {"n_txns": len(per), "fallback_rate": float(per["fallback"].mean()), "macro_f1": macro_f1(per["category"], per["pred"]),
               "total_cost": float(cum_cost), "n_writebacks": n_wb,
               "writeback_error_rate": (n_wb_wrong / n_wb) if n_wb else 0.0, "final_index_size": len(index)}
    return StreamLog(per_txn=per, windows=win.reset_index(), summary=summary)
```

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Add `fig_stream_lines` to `analysis/plots.py`**

```python
POLICY_COLORS = {"never": style.SERIES["head"], "always": style.SERIES["tail"], "confidence_gated": style.SERIES["third"]}


def fig_stream_lines(windows_by_policy: dict[str, pd.DataFrame], ycol: str, ylabel: str, stem: str | Path, title: str) -> None:
    """One line per write-back policy over stream windows (x = transactions processed)."""
    fig, ax = plt.subplots()
    for policy, w in windows_by_policy.items():
        x = (w["window"] + 1) * (w.attrs.get("window_size", 500))
        ax.plot(x, w[ycol], color=POLICY_COLORS.get(policy, style.INK2), label=policy.replace("_", " "))
    ax.set_xlabel("transactions processed")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=style.INK)
    ax.legend(loc="best")
    style.save(fig, stem)
    plt.close(fig)
```

- [ ] **Step 7: Write `experiments/exp3_streaming_convergence.py`** (OFFLINE cache replay)

```python
"""Exp 3: replay the DC test window in temporal order under three write-back policies (cache replay, $0).
Merchants without a cached LLM result are counted as fallback-uncached and served by kNN.

Usage: python -m experiments.exp3_streaming_convergence --config configs/dc.yaml --seeds 42 43 44 [--max-txns 100000]
Outputs: fig5_fallback_decay, fig6_cumulative_cost, fig7_cumulative_f1 (+_tail), tab5_writeback_policies.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from analysis.plots import fig_stream_lines
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import load_prepared
from txcat.data.splits import temporal_split
from txcat.embedder import Embedder
from txcat.gate import ConfidenceGate
from txcat.knn import build_merchant_index
from txcat.llm_fallback import CacheMissError, LLMFallback, LLMResult
from txcat.stream import run_stream
from txcat.utils import set_seed, setup_logging
from txcat.web_search import WebSearchClient
from txcat.web_search import CacheMissError as SearchMiss
from txcat.writeback import WriteBackPolicy

POLICIES = ["never", "always", "confidence_gated"]


class CachedOrUncached:
    """Wraps LLMFallback: on a cache miss return an invalid result (kNN keeps its label) and count it."""
    def __init__(self, fb: LLMFallback):
        self.fb, self.uncached = fb, 0
    def categorize(self, merchant, evidence, taxonomy):
        try:
            return self.fb.categorize(merchant, evidence, taxonomy)
        except CacheMissError:
            self.uncached += 1
            return LLMResult("INVALID", 0.0, "uncached", False, self.fb.m.name, 0, 0, 0.0, True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--max-txns", type=int, default=100000)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--model", default=None, help="LLM model name for the fallback (default: first configured)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    setup_logging(cfg.logs_dir, "exp3")
    figs, tables = Path(cfg.results_dir, "figures"), Path(cfg.results_dir, "tables")
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    d = cfg.datasets["dc"]
    win = WindowCfg(**json.loads(Path(cfg.results_dir, "window_decision.json").read_text())["chosen_window"])
    dc = load_prepared(d.processed_path, cfg.taxonomy_dir)
    dc = dc[dc["category"].notna() & (dc["ambiguous"] == 0)]
    train, test = temporal_split(dc, win)
    test = test.sort_values("date", kind="stable").head(args.max_txns).reset_index(drop=True)
    mcfg = next(m for m in cfg.llm.models if args.model is None or m.name == args.model)
    ws = WebSearchClient(cfg.search.provider, cfg.search.cache_dir, None, allow_live=False)

    def evidence(m):
        try:
            return ws.search(m, cfg.search.num_results)
        except SearchMiss:
            return None

    bb = cfg.embed.backbones[0]
    emb = Embedder(bb, cfg.embed.cache_dir, allow_live=False)
    windows_by_policy, summary_rows = {}, []
    for policy in POLICIES:
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
            fb = CachedOrUncached(LLMFallback(mcfg, cfg.llm.cache_dir, cfg.llm.prompt_with_web, allow_live=False))
            log = run_stream(test, idx, emb, ConfidenceGate(cfg.gate.mode, cfg.gate.threshold), fb,
                             WriteBackPolicy(policy, 0.8), evidence, window=args.window, k=cfg.index.k,
                             ef_search=cfg.index.ef_search, price_in_per_1m=mcfg.price_in_per_1m,
                             price_out_per_1m=mcfg.price_out_per_1m, search_price=cfg.search.price_per_1k_usd / 1000,
                             tail_k=cfg.tail_k)
            log.summary.update({"policy": policy, "seed": seed, "model": mcfg.name, "fallback_uncached": fb.uncached})
            summary_rows.append(log.summary)
            per_seed.append(log.windows)
            logger.info(f"{policy} seed {seed}: {log.summary}")
        w = pd.concat(per_seed).groupby("window").mean(numeric_only=True).reset_index()
        w.attrs["window_size"] = args.window
        windows_by_policy[policy] = w
    fig_stream_lines(windows_by_policy, "fallback_rate", "fallback rate (per window)", figs / "fig5_fallback_decay",
                     "Fallback rate over the stream")
    fig_stream_lines(windows_by_policy, "cum_cost", "cumulative cost (USD)", figs / "fig6_cumulative_cost", "Cumulative fallback cost")
    fig_stream_lines(windows_by_policy, "cum_macro_f1", "cumulative macro-F1", figs / "fig7_cumulative_f1", "Cumulative macro-F1 (overall)")
    fig_stream_lines(windows_by_policy, "cum_macro_f1_tail", "cumulative macro-F1 (tail)", figs / "fig7_cumulative_f1_tail",
                     "Cumulative macro-F1 (tail)")
    s = pd.DataFrame(summary_rows)
    agg = s.groupby("policy").agg(fallback_rate=("fallback_rate", "mean"), macro_f1=("macro_f1", "mean"),
                                  total_cost=("total_cost", "mean"), n_writebacks=("n_writebacks", "mean"),
                                  writeback_error_rate=("writeback_error_rate", "mean"), final_index_size=("final_index_size", "mean"),
                                  fallback_uncached=("fallback_uncached", "mean"), n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(tables / "tab5_writeback_policies.csv", index=False)
    s.to_csv(Path(cfg.results_dir, "exp3_per_seed.csv"), index=False)
    logger.info("\n" + agg.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run** — `.venv/bin/python -m experiments.exp3_streaming_convergence --config configs/dc.yaml --seeds 42 43 44`. Expected: `always` and `confidence_gated` show fallback rate decaying across windows and a lower total cost than `never`; `always` has a higher write-back error rate than `confidence_gated`.
- [ ] **Step 9: Commit** — `git add src/txcat/writeback.py src/txcat/stream.py tests/test_writeback.py tests/test_stream.py experiments/exp3_streaming_convergence.py analysis/plots.py results/figures/fig5_* results/figures/fig6_* results/figures/fig7_* results/tables/tab5_writeback_policies.csv results/exp3_per_seed.csv && git commit -m "feat: write-back policies, stream runner, exp3 streaming convergence"`

---

### Task B7: Generator tail-severity sweep (Exp 1 panel)

**Files:**
- Create: `experiments/exp1_generator_sweep.py`

- [ ] **Step 1: Write the script**

```python
"""Exp 1 (generator panel): kNN tail/head accuracy as a function of the Zipf exponent alpha. $0.

Usage: python -m experiments.exp1_generator_sweep --config configs/dc.yaml --alphas 0.8 1.0 1.2 1.5 --seeds 42 43 44
Outputs: results/tables/tab1b_alpha_sweep.csv, results/figures/fig1b_alpha_sweep.{pdf,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from analysis import style
from txcat.config import WindowCfg, load_config
from txcat.data.prepare import prepare
from txcat.data.splits import freq_of, merchant_frequencies, temporal_split
from txcat.embedder import Embedder
from txcat.generator.generate import generate
from txcat.knn import build_merchant_index, predict_knn
from txcat.normalizer import normalize_merchant
from txcat.utils import set_seed, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-txns", type=int, default=60000)
    ap.add_argument("--mode", choices=["live", "reproduce"], default="live")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    setup_logging(cfg.logs_dir, "exp1_gen")
    vocab = pd.read_parquet("data/processed/gen_vocab.parquet")
    emb = Embedder(cfg.embed.backbones[0], cfg.embed.cache_dir, allow_live=(args.mode == "live"))
    win = WindowCfg(train_start="2019-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2025-12-31")
    rows = []
    for alpha in args.alphas:
        for seed in seeds:
            set_seed(seed)
            g = generate(vocab, args.n_txns, alpha, seed)
            g["merchant"] = g["raw_merchant"].map(lambda r: normalize_merchant(r).text)
            g["ambiguous"] = 0
            train, test = temporal_split(g, win)
            freqs = merchant_frequencies(train)
            test = test.assign(train_freq=freq_of(test, freqs))
            idx = build_merchant_index(train, emb, cfg.index.M, cfg.index.ef_construction, seed)
            p = predict_knn(test, idx, emb, cfg.index.k, cfg.index.ef_search)
            correct = p["pred"].values == test["category"].values
            tail = test["train_freq"].values <= cfg.tail_k
            rows.append({"alpha": alpha, "seed": seed, "tail_txn_share": tail.mean(), "acc_tail": correct[tail].mean(),
                         "acc_head": correct[~tail].mean(), "acc_overall": correct.mean(), "n_train_merchants": len(idx)})
            logger.info(rows[-1])
    df = pd.DataFrame(rows)
    agg = df.groupby("alpha").agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    Path(cfg.results_dir, "tables").mkdir(parents=True, exist_ok=True)
    agg.to_csv(Path(cfg.results_dir, "tables", "tab1b_alpha_sweep.csv"), index=False)
    style.apply()
    fig, ax = plt.subplots()
    ax.errorbar(agg["alpha"], agg["acc_head_mean"], yerr=agg["acc_head_std"], marker="o", color=style.SERIES["head"], label="head")
    ax.errorbar(agg["alpha"], agg["acc_tail_mean"], yerr=agg["acc_tail_std"], marker="o", color=style.SERIES["tail"], label="tail (freq ≤ 3)")
    ax.set_xlabel("Zipf exponent α (larger = heavier head, thinner tail)")
    ax.set_ylabel("kNN top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Tail severity sweep on the semi-synthetic benchmark", loc="left", color=style.INK)
    ax.legend(loc="lower right")
    style.save(fig, Path(cfg.results_dir, "figures", "fig1b_alpha_sweep"))
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** — `.venv/bin/python -m experiments.exp1_generator_sweep --config configs/dc.yaml`. Expected: tail accuracy well below head at every alpha; tail transaction share falls as alpha rises.
- [ ] **Step 3: Commit** — `git add experiments/exp1_generator_sweep.py results/tables/tab1b_alpha_sweep.csv results/figures/fig1b_* && git commit -m "feat: generator tail-severity sweep"`

---

### Task B8: `reproduce.py`, run manifest, README, cache commit

**Files:**
- Create: `reproduce.py`, `src/txcat/manifest.py`, `tests/test_manifest.py`; Modify: `README.md`

- [ ] **Step 1: Failing test**

`tests/test_manifest.py`:
```python
import json

from txcat.manifest import build_manifest


def test_manifest_has_required_keys(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    m = build_manifest(prompt_paths=[tmp_path / "a.txt"], config={"index": {"M": 32}}, model_versions={"x": "y"})
    for k in ("timestamp", "git_commit", "host", "python", "packages", "prompt_hashes", "hnsw_params", "model_versions"):
        assert k in m
    assert m["hnsw_params"] == {"M": 32}
    assert "/Users/" not in json.dumps(m)  # no absolute paths in the committed manifest
    assert len(next(iter(m["prompt_hashes"].values()))) == 64
    json.dumps(m)  # serializable
```

- [ ] **Step 2: Implement `src/txcat/manifest.py`**

```python
"""Run manifest: everything a reader needs to know to trust a results directory."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from txcat.config import _ROOT, rel_to_root
from txcat.utils import now_iso, sha256_text

PACKAGES = ["numpy", "pandas", "hnswlib", "sentence-transformers", "torch", "scikit-learn", "openai"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_manifest(prompt_paths: list[str | Path], config: dict, model_versions: dict) -> dict:
    return {"timestamp": now_iso(), "git_commit": _git_commit(),
            "host": {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node()},
            "python": sys.version,
            "packages": {p: _ver(p) for p in PACKAGES},
            "prompt_hashes": {rel_to_root(p): sha256_text(Path(p).read_text()) for p in prompt_paths},
            "hnsw_params": config.get("index", {}), "model_versions": model_versions,
            "config": _relativize(config)}


def _relativize(obj):
    """Config paths are absolute in memory; the committed manifest must not leak the author's home dir."""
    if isinstance(obj, dict):
        return {k: _relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(str(_ROOT)):
        return rel_to_root(obj)
    return obj


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "missing"
```

- [ ] **Step 3: Write `reproduce.py`**

```python
"""Regenerate every table and figure from committed caches. Zero live API calls. Fails loudly on cache misses.

python reproduce.py --config configs/dc.yaml --seeds 42 43 44 --output results/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from txcat.config import load_config
from txcat.manifest import build_manifest

STEPS = [
    ["experiments.exp0_data_audit"],
    ["experiments.exp1_tail_characterization", "--mode", "reproduce"],
    ["experiments.exp1_generator_sweep", "--mode", "reproduce"],
    ["experiments.exp2_fallback_comparison", "--mode", "reproduce"],
    ["experiments.exp3_streaming_convergence"],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = [str(s) for s in (args.seeds or cfg.seeds)]
    for req in ("prompts/PROMPT_HASHES.json", "prompts/model_versions.json", "cache/web_search", "cache/llm", "cache/embeddings"):
        if not Path(req).exists():
            sys.exit(f"missing required artifact: {req} (caches must be present; no live calls are made)")
    frozen = json.loads(Path("prompts/PROMPT_HASHES.json").read_text())
    for p, h in frozen.items():
        from txcat.utils import sha256_text
        if sha256_text(Path(p).read_text()) != h:
            sys.exit(f"prompt {p} differs from frozen hash; refusing to reproduce with an edited prompt")
    manifest = build_manifest(list(frozen), cfg.model_dump(), json.loads(Path("prompts/model_versions.json").read_text()))
    Path(args.output, "logs").mkdir(parents=True, exist_ok=True)
    Path(args.output, "logs", f"manifest_{manifest['timestamp'].replace(':', '')}.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"git {manifest['git_commit'][:10]} | {manifest['host']['platform']} | prompts {list(frozen)}")
    for step in STEPS:
        cmd = [sys.executable, "-m", *step, "--config", args.config]
        if step[0] != "experiments.exp0_data_audit":
            cmd += ["--seeds", *seeds]
        logger.info("RUN " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    logger.info("reproduction complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then run `reproduce.py` end-to-end** — `.venv/bin/python reproduce.py --config configs/dc.yaml --seeds 42 43 44`. Expected: completes with no network; tables/figures identical to the committed ones (`git status` clean except the new manifest).

- [ ] **Step 5: README** — replace the stub with sections: Overview (paper title, author), Evidence stack (DC / Oklahoma / generator / DoDataThings table), Setup (uv), Data download (never redistributed), Reproduce from cache (`python reproduce.py ...`), Re-running the live cache pass (keys, spend cap, resumability), Repository layout, Privacy statement (only normalized merchant strings leave the machine), Taxonomy (mcc_to_category.csv, ambiguous codes, label-noise audit), Citation placeholder. No employer mentioned anywhere.

- [ ] **Step 6: Cache size check and commit.** `.gitignore` currently ignores `cache/embeddings/*/vectors.npy` (added in Day 1 Task 0 while caches were unfrozen). A fresh clone would then have `keys.json` pointing at vectors that do not exist and `reproduce.py` would raise on every text. Resolve it here, one way or the other:
```bash
du -sh cache/web_search cache/llm cache/embeddings
```
If `cache/embeddings` > 100 MB total, keep `cache/embeddings/*/vectors.npy` ignored and add a `make release-assets` note in README (upload the npy files as a GitHub release asset; `reproduce.py` downloads them if missing). Otherwise remove that ignore line and commit the embeddings too.
```bash
git add reproduce.py src/txcat/manifest.py tests/test_manifest.py README.md results/logs/manifest_*.json cache
git commit -m "feat: reproduce.py with run manifest; README; frozen caches"
```

## Day 2 exit criteria
- `python reproduce.py` regenerates Exp 0–3 outputs from cache with zero network calls.
- `results/tables/tab2_main_results.csv`, `tab3_critical_ablation.csv`, `tab5_writeback_policies.csv`, `fig4`–`fig7` committed.
- `cache/spend_ledger.json` total ≤ $40.
