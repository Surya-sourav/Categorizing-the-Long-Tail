# longtail-txcat — Design Spec

**Paper:** "Categorizing the Long Tail: An Empirical Study of Web-Search-Augmented Fallback for Embedding-Based Transaction Classification"
**Author:** Surya Parida, Independent Researcher
**Date:** 2026-09-12
**Status:** approved design, pre-implementation

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
| Primary (real) | Washington DC Purchase Card Transactions | ArcGIS REST, `maps2.dcgis.dc.gov/.../Public_Service_WebMercator/MapServer/50`, 639k records, 2009–present | `OBJECTID`, `TRANSACTION_DATE`, `VENDOR_NAME`, `MCC_DESCRIPTION` | Real acquirer descriptors (`STAPLES       00102186`, `WW GRAINGER 912`), real MCC labels, real dates |
| Second real | Oklahoma State PCard FY2023–2025 | CKAN `data.ok.gov`, 36 monthly CSVs, ~8 MB each | `ROWID`, `TRANSACTION_DATE`, `MERCHANT`, `MCC_DESCRIPTION` | Descriptors like `AMZN Mktp US RC9J295A2`, `WAL-MART #4241` |
| Controlled instrument | Own generator | OSM name-suggestion-index (BSD-3) vocabulary + templated statement noise + Zipf(alpha) frequencies + synthetic dates | `txn_id`, `date`, `raw`, `category`, `merchant_id`, `alpha` | Only dataset where tail severity can be dialed |
| Sanity only | DoDataThings/us-bank-transaction-categories-v2 | HuggingFace, MIT, 68k rows | `description`, `category` | Template-generated; quick check, not evidence |

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
8999 Professional Services NEC, and any description containing "NOT ELSEWHERE CLASSIFIED" that
does not fit a category) are flagged `ambiguous=1`. All headline tables are reported with these rows
excluded; an appendix table reports them included.

**Label-noise audit.** `data/taxonomy/label_noise_audit_sample.csv`: 100 random DC
(merchant, MCC description, mapped category) rows with columns `judgement`
(`correct` / `wrong` / `unclear`) and `note`. Pre-filled by the implementer, verified by the author,
reported as one sentence + `results/tables/tab0_label_noise.csv`.

## 3. Splits and evaluation sets

- **Temporal split** on real data: train = transactions before cutoff date T, test = after. DC:
  T = 2020-01-01 (train ~2009–2019, test 2020–present). Oklahoma: train = FY2023–FY2024, test = FY2025.
- **Tail definition:** normalized merchant frequency in the *training index* <= 3. Sensitivity at
  <= 2 and <= 5. Head = frequency > 3.
- **Full test set** — used for kNN-only rows and Exp 1 (no API cost).
- **Fallback evaluation set (FES)** — budget-bounded: a stratified random sample of unique
  normalized test merchants, 2,000 tail + 500 head for DC, 800 + 200 for Oklahoma, 800 + 200 for
  the generator, with all their test transactions. Every API-backed method (search, LLM with/without
  web, all models) is run on **every** FES merchant once and cached. The gate-threshold frontier is
  then a pure offline re-mix of cached results, so the sweep costs nothing.
- **Streaming set** — the DC test period in real temporal order, capped at 100k transactions, with
  merchants outside the FES falling back to cached results where present and counted as
  "fallback (uncached)" otherwise; the uncached count is reported.

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
Provider `brave` (GET `api.search.brave.com/res/v1/web/search`, header `X-Subscription-Token`,
1 req/s throttle, resumable). Cache file `cache/web_search/<sha1(provider|query)>.json` with schema
`{"query", "provider", "timestamp", "results": [{"title","snippet","url"}]}`. Optional provider
`openai_builtin` (Responses API `web_search` tool, model-controlled query, ablation only) stores
the model's issued queries and `url_citation`s in the same schema. Reproduction mode never calls
the network.

### 4.6 `src/llm_fallback.py`
`LLMFallback(model, cache_dir, prompt_template).categorize(merchant, evidence | None, taxonomy)
-> LLMResult(category, confidence, evidence_used, tokens_in, tokens_out, latency_ms)`. One
OpenAI-compatible client with `base_url` switch: OpenAI (`gpt-5-nano`, `gpt-5-mini`, pinned
snapshot ids, `reasoning.effort="minimal"`), Together (`meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`).
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

## 5. Experiments and outputs

| Exp | Script | Outputs |
|---|---|---|
| 0 | `experiments/exp0_data_audit.py` | `tab0_dataset_stats.csv`, `tab0_label_noise.csv`, taxonomy coverage |
| 1 | `exp1_tail_characterization.py` | `fig1_zipf`, `fig2_acc_by_freq` (bins 1, 2–5, 6–20, 21–100, 100+), `fig3_acc_vs_sim` head/tail overlay, `tab1_tail_stats.csv` (tail at <=2, <=3, <=5) |
| 2 | `exp2_fallback_comparison.py` | `tab2_main_results.csv` (rows: kNN per backbone, emb+LR, LLM no-web, LLM with-web, routed system; cols: macro-F1 head/tail/overall, p50/p95 latency, cost/1k, fallback rate), `tab3_critical_ablation.csv` (tail-only with-web vs no-web, paired bootstrap CI, per model), `fig4_acc_cost_frontier` (threshold 0.30–0.95 step 0.05, knee marked) |
| 3 | `exp3_streaming_convergence.py` | `fig5_fallback_decay`, `fig6_cumulative_cost`, `fig7_cumulative_f1`, `tab5_writeback_policies.csv` |
| 4 | `exp4_failure_taxonomy.py` | `failure_sample_template.csv` (~200 tail errors, categories: ambiguous merchant / no web presence / web search failure / LLM reasoning error / taxonomy mismatch / normalization failure / label noise); after labelling -> `tab6_failure_taxonomy.csv` |
| 5 | `exp5_backbone_comparison.py` | `tab4_backbone_comparison.csv` (5 backbones x head/tail/overall F1, embed latency) |

Figures saved as PDF + PNG. All tables CSV. Every experiment takes `--config`, `--seed(s)`,
`--dataset`, `--mode {live,reproduce}`.

## 6. Reproducibility contract

- `reproduce.py --config configs/dc.yaml --seeds 42 43 44 --output results/` regenerates every
  table and figure from caches; raises if any cache entry is missing; logs embedding + LLM model
  ids, prompt hashes, HNSW params, git commit, timestamp, host, Python/lib versions to
  `results/logs/run_<ts>.json`.
- `prompts/model_versions.json` pins every model id and the Brave snapshot date.
- Seeds: `PYTHONHASHSEED`, `random`, `numpy`, `torch`, hnswlib `random_seed`. 3 seeds in the 2-day
  core, 5 for the final paper run.
- Tooling: `uv` with Python 3.12 (hnswlib/torch wheels), `pyproject.toml` + pinned
  `requirements.txt`, `loguru`, `pydantic-settings` for YAML configs, `pytest`, `ruff`. Secrets via
  `.env` (gitignored). Dockerfile after the core lands.

## 7. Budget (target < $50)

| Item | Estimate |
|---|---|
| Brave: ~3,600 FES merchants (DC 2,500 + OK 1,000 + gen 1,000, minus overlap) | ~$13–18 after $5 credit |
| gpt-5-nano + gpt-5-mini: ~3,600 merchants x 2 conditions x 2 models | ~$3–5 |
| Llama 3.1 8B via Together: same calls | ~$2 |
| text-embedding-3-small: ~150k unique strings | < $1 |
| Optional: OpenAI built-in web search ablation, 300 merchants | ~$7 |
| **Total** | **~$25–33** |

## 8. Two-day core scope and cut lines

**Day 1:** repo scaffold, loaders + download scripts (DC, OK, NSI, DoDataThings), taxonomy files,
normalizer + tests, embedder + cache, index + tests, gate + tests, Exp 0 + Exp 1 on DC, 100-merchant
pilot of search + LLM, prompts frozen.
**Day 2 (author runs the cache passes):** full FES cache run, Exp 2 (main table, critical ablation,
frontier), write-back + stream + tests, Exp 3 on DC stream, `reproduce.py`, README.
**Days 3–5:** Oklahoma as second real set through Exp 1–3, Exp 5 with mpnet + FinBERT at 5 seeds,
Exp 4 labelling (author), Dockerfile, coverage > 80%, OpenAI agentic-search ablation.

## 9. Risks

- Brave rate limit (1 rps) makes the search pass ~1 hour; the pass is resumable and idempotent.
- Any prompt edit after the LLM cache is frozen invalidates it — hence the pilot-then-freeze rule.
- Institutional spend skews the category mix; stated as a limitation, generator covers consumer mix.
- MCC label noise bounds achievable accuracy; measured and reported (Exp 0).
- DC ArcGIS `maxRecordCount` paging (~1–2k/request) means ~400–640 requests; download script retries
  and checkpoints.
