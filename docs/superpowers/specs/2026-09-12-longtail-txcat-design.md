# longtail-txcat — Design Spec

**Paper:** "Categorizing the Long Tail: An Empirical Study of Web-Search-Augmented Fallback for Embedding-Based Transaction Classification"
**Author:** Surya Parida, Independent Researcher
**Date:** 2026-09-12
**Status:** FINAL — decisions locked by author 2026-09-12. Build against this.

## 1. Research question

Embedding + kNN classifiers for card transactions work on merchants seen before and fail on the
long tail (merchants with frequency <= 3 in the index). The tail is small in unique-merchant terms
but large in transaction volume, and it is where top-1 similarity is low. We ask:

1. **Tail characterization.** How does kNN accuracy fall with merchant frequency and with top-1
   similarity, on real card-network descriptors?
2. **Web evidence.** When a low-confidence retrieval is routed to an LLM, does web-search evidence
   about the merchant improve categorization versus the same LLM with no evidence? Measured on tail
   merchants only, paired bootstrap CI on the difference.
3. **Self-expansion.** If LLM-resolved merchants are written back into the index, does the fallback
   rate decay over a temporally ordered stream, and what does write-back cost in index pollution
   under never / always / confidence-gated policies?

Contribution is empirical: a routed architecture, an accuracy-cost frontier over the gate threshold,
a failure taxonomy, and a reproducible public benchmark with frozen caches. CPU + API only.

## 2. Datasets (evidence stack)

| Role | Dataset | Source | Fields kept | Notes |
|---|---|---|---|---|
| Primary (real) | Washington DC Purchase Card Transactions, **2019-01-01 → present** | ArcGIS REST, `maps2.dcgis.dc.gov/.../Public_Service_WebMercator/MapServer/50` (639k records total, 2009–present; we use 2019+) | `OBJECTID`, `TRANSACTION_DATE`, `VENDOR_NAME`, `MCC_DESCRIPTION` | All headline numbers. Real acquirer descriptors (`STAPLES       00102186`, `WW GRAINGER 912`), real MCC labels, real dates |
| Second real | Oklahoma State PCard FY2023–2025 | CKAN `data.ok.gov`, 36 monthly CSVs, ~8 MB each | `ROWID`, `TRANSACTION_DATE`, `MERCHANT`, `MCC_DESCRIPTION` | **Cross-dataset cold-start only:** index built from DC train, tested on Oklahoma merchants that never appear in DC. No Oklahoma-trained index. |
| Instrument | Own generator | OSM name-suggestion-index (BSD-3) vocabulary + templated statement noise + Zipf(alpha) frequencies + synthetic dates | `txn_id`, `date`, `raw`, `category`, `merchant_id`, `alpha` | **Tail-severity sweep only.** No headline numbers. |
| Sanity | DoDataThings/us-bank-transaction-categories-v2 | HuggingFace, MIT, 68k rows | `description`, `category` | Internal check only. **No paper numbers.** |

**Privacy rule.** Loaders drop cardholder names, amounts, item descriptions, agency, and any field
not listed above before anything reaches the pipeline. Only the *normalized merchant string* is ever
sent to a web search or LLM API. Enforced by a unit test on the loader output schema and by the
`WebSearchClient` / `LLMFallback` accepting a `merchant: str` argument only.

**Redistribution.** Raw data is not committed. `data/download.py` fetches it; `data/processed/` is
gitignored. Caches (which contain normalized merchant strings only) are committed.

**Label construction.** Both real datasets label with MCC *descriptions* (different spellings per
source). Two committed files:

- `data/taxonomy/mcc_to_category.csv` — columns `mcc, mcc_description, category, ambiguous, notes`.
  Maps the ~980 MCCs (from the public greggles/mcc-codes list) to 15 macro categories + an
  `AMBIGUOUS` bucket. Brand-specific hotel/airline/car-rental MCCs (3000–3999) collapse into
  Lodging / Airlines / Ground Transport.
- `data/taxonomy/mcc_description_aliases.csv` — columns `source, observed_description, mcc`. Maps each
  dataset's observed description string to an MCC. Built by normalized exact match, then fuzzy
  match, then manual review of leftovers; the review is committed.

Macro categories (15): Groceries & Food Stores; Restaurants & Dining; Retail & General Merchandise;
Office, Stationery & Printing; Software, Computers & Electronics; Telecom & Utilities; Airlines;
Lodging; Ground Transport, Auto & Fuel; Industrial, Construction & Hardware; Professional & Business
Services; Health & Medical; Education, Membership & Government; Entertainment, Recreation & Media;
Financial, Insurance, Postal & Shipping.

**Ambiguous MCCs** (5999 Misc Retail, 5399 Misc General Merchandise, 7399 Business Services NEC,
8999 Professional Services NEC, and any "NOT ELSEWHERE CLASSIFIED" description that does not fit a
category) are flagged `ambiguous=1`. They are **excluded from headline tables** and **reported as
their own slice** in every main table.

**Label-noise audit.** `data/taxonomy/label_noise_audit_sample.csv`: 300 random DC
(merchant, MCC description, mapped category) rows with columns `judgement`
(`correct` / `wrong` / `unclear`) and `note`. Pre-filled by the implementer, verified by the author,
reported as one sentence + `results/tables/tab0_label_noise.csv`.

## 3. Splits and evaluation sets

**Temporal split, never random. No row-level randomization anywhere.**

- **DC train / index:** 2019-01-01 → 2023-12-31. **DC test:** 2024-01-01 → present.
- **Conditional volume rule (verified by Exp 0):** the test window must contain >= 20,000
  transactions and >= 2,500 unique tail merchants. If short, slide to train 2019–2022 / test 2023+.
  The chosen window is written to `configs/dc.yaml` and reported in the paper.
- **Tail definition:** normalized merchant frequency in the *training index* <= 3. Sensitivity at
  <= 2 and <= 5. Head = frequency > 3.
- **Oklahoma cold-start:** index = DC train. Test = Oklahoma transactions whose normalized merchant
  never appears in DC train (every such merchant is, by construction, frequency 0 in the index).
- **Full test set** — used for kNN-only rows and Exp 1 (no API cost).
- **Fallback evaluation set (FES)** — budget-bounded, drawn *by merchant*: all tail merchants in the
  DC test window if <= 2,500, else a seeded sample of 2,500 tail merchants plus 500 head merchants,
  with all their test transactions. Oklahoma cold-start FES: 800 merchants. Generator FES: 800
  merchants at each alpha in the sweep only for the kNN rows (no API calls; generator is $0).
  Every API-backed method (search, each LLM, with/without web) runs on **every** FES merchant once
  and is cached. The gate-threshold frontier is a pure offline re-mix of cached results.
- **Streaming set** — the DC test window in real temporal order; merchants outside the FES reuse
  cached results where present and are counted as "fallback (uncached)" otherwise; count reported.

## 4. System architecture

```
raw row ──► loader (drop PII) ──► normalize_merchant ──► Embedder (cache) ──► HNSWIndex.query
                                                                                   │
                                                                     ConfidenceGate.should_fallback?
                                                                        │ no                │ yes
                                                                        ▼                   ▼
                                                                  kNN top-1 label   WebSearchClient (cache)
                                                                                            │
                                                                                     LLMFallback (cache)
                                                                                            │
                                                                                 WriteBackPolicy.should_write?
                                                                                            │ yes
                                                                                   HNSWIndex.add(vector,label)
```

Every module has one job, a typed interface, and is testable alone.

### 4.1 `src/normalizer.py`
`normalize_merchant(raw: str) -> NormalizeResult(text: str, steps: list[str])`. Rule pipeline:
uppercase; strip processor prefixes (`SQ *`, `TST*`, `PP*`, `PAYPAL *`, `PYPL*`, `AMZN Mktp US`,
`AMAZON MKTPL`, `CLV*`, `IC*`, `DD *`, `UBER *`, `POS `, `DBT PURCHASE`); strip trailing
alphanumeric reference codes (`RC9J295A2`, `K8R2M5VN7`), store numbers (`#4241`, `00102186`,
`912`), trailing US state codes and city tokens when preceded by a known merchant token; collapse
whitespace and `QQQ`-style padding artifacts; drop dates/times. Never touches amounts (they never
enter). Every applied rule is appended to `steps` for auditability. Tests: >= 20 real examples
drawn from DC/Oklahoma descriptors.

### 4.2 `src/embedder.py`
`Embedder(model_name, cache_dir).embed(texts) -> np.ndarray` (L2-normalized, float32 in memory,
float16 on disk). Cache per model: `cache/embeddings/<model_slug>/vectors.npy` +
`manifest.json` (sha1(text) -> row). Local backbones via sentence-transformers:
`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `all-mpnet-base-v2`, `ProsusAI/finbert` (mean
pooling, documented). API: OpenAI `text-embedding-3-small` (pinned in `prompts/model_versions.json`).
Reproduction mode raises if a text is missing from cache.

### 4.3 `src/index.py`
`HNSWIndex(dim, M=32, ef_construction=200, seed)` over hnswlib cosine space.
`add(vectors, labels, merchant_ids, first_seen)` maintains per-entry `label`, `merchant_id`,
`freq`, `first_seen` in a sidecar `pandas.DataFrame`. `query(vector, k=5, ef_search=50) ->
QueryResult(labels, sims, merchant_ids, freqs, margin=sims[0]-sims[1])`. Persist/load `.bin` +
`.parquet`. `ef_search` sweep {16, 32, 64, 128} logged once per dataset.

### 4.4 `src/gate.py`
`ConfidenceGate(mode, threshold)`; modes `threshold` (sim1 < t), `margin` (sim1 - sim2 < t),
`calibrated_lr` (sklearn LogisticRegression on `[sim1, sim2, margin, log1p(rank1_freq)]`, trained
on a 10% calibration slice of train, predicts P(top-1 wrong); fallback if P > t).

### 4.5 `src/web_search.py`
`WebSearchClient(provider, cache_dir).search(query, num_results=5) -> list[SearchResult]`.
Provider `firecrawl` (POST `api.firecrawl.dev/v2/search`, bearer auth, 2 prepaid credits per query, results
under `data.web[]`; `tavily` and `brave` also implemented). The author holds 10k credits, enough for the
~3,900 pilot + FES queries with ~2k spare for Exp 4 page scrapes. Cost column uses Firecrawl's list price
($0.0053/credit) as a reference. Resumable, sequential (2-concurrency cap). Cache file `cache/web_search/<sha1(provider|query)>.json` with schema
`{"query", "provider", "timestamp", "results": [{"title","snippet","url"}]}`. Optional provider
`openai_builtin` (Responses API `web_search` tool, model-controlled query, ablation only) stores
the model's issued queries and `url_citation`s in the same schema. Reproduction mode never calls
the network.

### 4.6 `src/llm_fallback.py`
`LLMFallback(model, cache_dir, prompt_template).categorize(merchant, evidence | None, taxonomy)
-> LLMResult(category, confidence, evidence_used, tokens_in, tokens_out, latency_ms)`. One
OpenAI-compatible client with `base_url` switch: OpenAI (`gpt-5-nano`, `gpt-5-mini`, pinned
snapshot ids, `reasoning.effort="minimal"`), NVIDIA NIM (`google/gemma-4-31b-it` and `nvidia/nemotron-3.5-lightning-30b-a3b`, the open-weight rows;
free developer endpoint at `integrate.api.nvidia.com/v1`, ~40 requests/minute, so the client throttles to
1.6 s between calls; cost column uses a labelled reference rate because the endpoint itself is free; latency on
a shared free endpoint is reported but not treated as a deployment number).
JSON-schema structured output `{category: enum(taxonomy), confidence: number, evidence: string}`.
Prompts `prompts/fallback_with_web.txt` and `prompts/fallback_no_web.txt` differ only by the
evidence block. Cache key `sha256(model|prompt_hash|rendered_prompt)`; file stores full request
and response. Prompts are frozen after a 100-merchant pilot; `prompts/PROMPT_HASHES.json` pins them.

### 4.7 `src/writeback.py` and `src/stream.py`
`WriteBackPolicy(mode, confidence_threshold=0.8).should_write(llm_result)`; modes `never`,
`always`, `confidence_gated`. `run_stream(transactions, index, gate, fallback, writeback,
window=500) -> StreamLog` processes in temporal order, tracks windowed fallback rate, cumulative
cost, cumulative macro-F1 (overall/tail), index size, write-back error rate (written label !=
gold). Written entries are tagged `source="writeback"` so pollution can be measured.

### 4.8 `src/baselines.py`, `src/metrics.py`, `src/costs.py`
Embeddings + logistic regression baseline. Metrics: `macro_f1(y_true, y_pred, mask)`,
`latency_p50_p95`, `cost_per_1k`, `bootstrap_ci` across seeds, `paired_bootstrap_diff` on
per-merchant correctness for the critical ablation. Prices in `configs/model_prices.yaml`.

### 4.9 Generator `src/generator/`
`build_vocab.py`: download NSI, keep brands with `locationSet` including `us` or `001`, map
`shop=*`/`amenity=*`/`office=*`/`healthcare=*`/`leisure=*` tags to the 15 categories via
`data/taxonomy/osm_tag_to_category.csv`. `generate.py --alpha --n-merchants --n-transactions
--seed`: Zipf merchant frequencies, noise templates (prefixes, ref codes, store numbers, city/state,
truncation at 22/25 chars, whitespace padding, vowel-dropped abbreviations), synthetic dates over a
2-year window. Output parquet with same schema as real loaders.

## 5. Experiments — exactly six, run in order 0 → 1 → 2 → 3 → 4 → 5

| # | Script | Cost | Paper output |
|---|---|---|---|
| 0 | `exp0_data_audit.py` | $0 | `tab0_dataset_stats.csv` (rows, unique merchants, tail share, per dataset/window), `tab0_label_noise.csv` (300-row audit), taxonomy coverage, **final window decision** logged to config |
| 1 | `exp1_tail_characterization.py` | $0 | `fig1_zipf`, `fig2_acc_by_freq` (bins 1, 2–5, 6–20, 21–100, 100+), `fig3_acc_vs_sim` head/tail overlay, `tab1_tail_stats.csv` (tail at <=2, <=3, <=5). Also the generator alpha-sweep panel. |
| 2 | `exp2_fallback_comparison.py` | ~$20 (the one expensive cache pass) | `tab2_main_results.csv`: rows kNN per backbone, emb+LR, LLM no-web, LLM with-web, routed system; cols macro-F1 head/tail/overall/ambiguous-slice, p50/p95 latency, cost/1k, fallback rate. LLM rows are reported on head too so "kNN beats LLM on head" is visible. `tab3_critical_ablation.csv`: tail-only with-web vs no-web, paired bootstrap CI, per model. `fig4_acc_cost_frontier` (threshold 0.30–0.95 step 0.05, knee marked). **Folded in:** `tab3b_prompt_sensitivity.csv` (2 prompt variants x 300 tail merchants) and `tab3c_agentic_search.csv` (OpenAI built-in web search, 300 merchants, ~$7). Oklahoma cold-start rows reported in `tab2b_oklahoma_coldstart.csv`. |
| 3 | `exp3_streaming_convergence.py` | ~$0 (cache replay) | `fig5_fallback_decay`, `fig6_cumulative_cost`, `fig7_cumulative_f1`, `tab5_writeback_policies.csv` (never / always / confidence-gated) |
| 4 | `exp4_failure_taxonomy.py` | $0 (author labels ~200 tail errors) | `failure_sample_template.csv` with categories: ambiguous merchant / no web presence / web search failure / LLM reasoning error / taxonomy mismatch / normalization failure / label noise. After labelling -> `tab6_failure_taxonomy.csv` + qualitative section examples |
| 5 | `exp5_backbone_comparison.py` | < $1 | `tab4_backbone_comparison.csv`: MiniLM-L6-v2, bge-small-en-v1.5, mpnet-base-v2, FinBERT-as-encoder, text-embedding-3-small x head/tail/overall macro-F1 + embed latency |

Figures saved as PDF + PNG. All tables CSV. Every experiment takes `--config`, `--seeds`,
`--dataset`, `--mode {live,reproduce}`.

## 6. Reproducibility contract

- `reproduce.py --config configs/dc.yaml --seeds 42 43 44 --output results/` regenerates every
  table and figure from caches; raises if any cache entry is missing; logs embedding + LLM model
  ids, prompt hashes, HNSW params, git commit, timestamp, host, Python/lib versions to
  `results/logs/run_<ts>.json`.
- `prompts/model_versions.json` pins every model id and the Brave snapshot date.
- Seeds: `random`, `numpy`, `torch`, hnswlib `random_seed` (single-threaded inserts). Anything that
  samples uses an explicit `np.random.default_rng(seed)`; `set_seed` is a best-effort global helper.
  `PYTHONHASHSEED` is set in the Dockerfile/`reproduce.py` environment, not from inside Python.
  **3 seeds for development, 5 for the final paper run.**
- **Hard API spend cap** in config (`budget.max_usd`, default 40). Every live client accumulates
  estimated spend in `cache/spend_ledger.json` and raises `BudgetExceeded` before the call that
  would cross the cap.
- Exact LLM snapshot ids are resolved from the provider's models endpoint at the pilot run and
  written to `prompts/model_versions.json`; all later calls use the pinned id.
- Tooling: `uv` with Python 3.12 (hnswlib/torch wheels), `pyproject.toml` + pinned
  `requirements.txt`, `loguru`, `pydantic-settings` for YAML configs, `pytest`, `ruff`. Secrets via
  `.env` (gitignored). Dockerfile after the core lands.

## 7. Budget (target ~$30–40, hard cap $40)

| Item | Estimate |
|---|---|
| Firecrawl: ~3,900 queries x 2 credits from the author's prepaid 10k | $0 incremental (~$41 list-price reference) |
| Two small OpenAI models: ~3,300 merchants x 2 conditions x 2 models | ~$3–5 |
| Two open-weight models via NVIDIA NIM free endpoint: same calls | $0 (~3 h each at 40 RPM) |
| text-embedding-3-small: unique strings across DC + Oklahoma + generator | < $1 |
| Prompt-sensitivity: 2 variants x 300 tail merchants x 3 models | ~$1 |
| OpenAI built-in web search ablation, 300 merchants | ~$7 |
| **Total** | **~$26–33** |

## 8. Two-day core scope and cut lines

**Day 1:** repo scaffold, loaders + download scripts (DC 2019+, Oklahoma, NSI, DoDataThings),
taxonomy files, normalizer + tests, embedder + cache, index + tests, gate + tests, Exp 0 (window
decision, audit sample) + Exp 1 on DC, 100-merchant pilot of search + LLM, prompts frozen, spend cap.
**Day 2 (author runs the cache pass):** full FES cache run (Exp 2 live mode), Exp 2 tables and
frontier, write-back + stream + tests, Exp 3 replay, `reproduce.py`, README.
**Days 3–5:** Exp 5 with mpnet + FinBERT at 5 seeds, Exp 4 labelling (author), final 5-seed run,
Dockerfile, coverage > 80%.

## 9. Risks

- Brave rate limit (1 rps) makes the search pass ~1 hour; the pass is resumable and idempotent.
- Any prompt edit after the LLM cache is frozen invalidates it — hence the pilot-then-freeze rule.
- Institutional spend skews the category mix; stated as a limitation.
- Oklahoma cold-start merchants are frequency 0 in the index by construction, so kNN there measures
  pure generalization; stated explicitly so it is not read as a tail-<=3 result.
- MCC label noise bounds achievable accuracy; measured and reported (Exp 0).
- DC ArcGIS `maxRecordCount` paging (~1–2k/request) means ~400–640 requests; download script retries
  and checkpoints.
