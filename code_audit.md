# Code audit against the paper's claims

**Commit audited:** `70c054ed733477c933a04ee57ca9f17dbce36e76`
**Working tree:** clean at the start of the audit (only untracked `table1_values.json` from a prior
task). This audit added `code_audit.md` and `audit_output/`; it modified no source file, table,
figure, prompt or cache. The embedding cache was hashed before and after the streaming replay in
section E and was **unchanged**; that replay used a scratch copy of the cache.

**Table numbering** follows the compiled `paper/main.pdf`: T1 datasets, T2 accuracy by frequency,
T3 novel-share sweep, T4 main comparison, T5 critical ablation, T6 Oklahoma cold start, T7 agentic
search, T8 write-back policies, T9 backbones, T10 failure taxonomy, T11 spend.

---

## A. Scoring

### A1. Every macro-F1 call site — **CONTRADICTED**

`macro_f1` is defined at `src/txcat/metrics.py:9-20` and calls
`sklearn.metrics.f1_score(..., average="macro", zero_division=0)` with **no `labels` argument**
(`src/txcat/metrics.py:20`). scikit-learn therefore uses the union of labels present in truth and
prediction for that one comparison. No call site overrides this.

| # | file:line | `labels` passed | effective class set |
|---|---|---|---|
| 1 | `src/txcat/metrics.py:20` | no | union of y_true ∪ y_pred |
| 2 | `src/txcat/stream.py:112` | no | union |
| 3 | `src/txcat/stream.py:118` | no | union |
| 4 | `src/txcat/stream.py:128` | no | union |
| 5 | `experiments/exp2_fallback_comparison.py:50` | no | union |
| 6 | `experiments/exp2_fallback_comparison.py:51` | no | union |
| 7 | `experiments/exp2_fallback_comparison.py:52` | no | union |
| 8 | `experiments/exp2_fallback_comparison.py:335` | no | union |
| 9 | `experiments/exp2_fallback_comparison.py:356` | no | union |
| 10 | `experiments/exp5_backbone_comparison.py:87` | no | union |
| 11 | `experiments/exp5_backbone_comparison.py:88` | no | union |
| 12 | `experiments/exp5_backbone_comparison.py:89` | no | union |

The paper's Section 3.4 states "the class set is the fixed 15-category taxonomy" and that "a
category with zero support and zero predictions contributes zero". Neither holds: absent categories
are **excluded from the average**, not scored zero, and the denominator varies per row.

### A2. Malformed-response sentinel — **CONFIRMED that it exists; CONTRADICTED that it is mapped into the taxonomy**

A sentinel string `"INVALID"` is produced at four places:

| Condition | file:line |
|---|---|
| JSON parsed but `category` not in the taxonomy | `src/txcat/llm_fallback.py:193` |
| Response unparseable (exception path) | `src/txcat/llm_fallback.py:205` |
| Agentic-search answer off-taxonomy or unparseable | `src/txcat/agentic_search.py:82,92` |
| Cache miss during streaming replay | `experiments/exp3_streaming_convergence.py:43` |

`"INVALID"` is **never mapped into the 15 macro categories**. It is written into the prediction
column and reaches `macro_f1` as an ordinary label, so it becomes its own 16th class that scores
F1 = 0 and enlarges the denominator. The one exception is the streaming path, which substitutes the
kNN label when `res.valid` is false (`src/txcat/stream.py:57`), so T8 is unaffected.

### A3. Class-set size per T4 row — **CONTRADICTED (rows are not comparable)**

Sizes differ by row. Full table: `audit_output/A3_class_set_sizes.csv`.

| T4 row | class-set size | extra class |
|---|---|---|
| kNN, MiniLM-L6-v2 | 15 | — |
| llm_no_web / gpt-5-nano | 16 | INVALID |
| llm_with_web / gpt-5-nano | 16 | INVALID |
| llm_no_web / gpt-5-mini | 16 | INVALID |
| llm_with_web / gpt-5-mini | 16 | INVALID |
| llm_no_web / DiffusionGemma 26B | 15 | — |
| llm_with_web / DiffusionGemma 26B | 15 | — |
| llm_no_web / Nemotron 3 Super | 16 | INVALID |
| llm_with_web / Nemotron 3 Super | 16 | INVALID |
| llm_no_web / Muse Glimmer 30B | 15 | — |
| llm_with_web / Muse Glimmer 30B | 15 | — |
| routed / gpt-5-mini @0.65 | 15 | — |
| **routed / gpt-5-nano @0.65** | **16** | **INVALID** |

All 15 taxonomy categories appear in every row, so the only variation is the extra INVALID class.
Rows scored over 16 classes are divided by a larger denominator than rows scored over 15, which
makes the T4 column not strictly comparable across rows.

### A4. Malformed responses per model — **CONFIRMED (counted)**

Full table: `audit_output/A4_malformed_per_model.csv`. Counted over every record in
`cache/llm/packed/*.jsonl.gz`.

| model | prompt | calls | invalid | % | unparseable | parsed but off-taxonomy |
|---|---|---|---|---|---|---|
| DiffusionGemma 26B | no_web | 300 | 0 | 0.000 | 0 | 0 |
| DiffusionGemma 26B | with_web | 300 | 0 | 0.000 | 0 | 0 |
| gpt-5-mini | no_web | 3,869 | 28 | 0.724 | 0 | 28 |
| gpt-5-mini | no_web_v2 | 300 | 1 | 0.333 | 0 | 1 |
| gpt-5-mini | with_web | 3,869 | 6 | 0.155 | 0 | 6 |
| gpt-5-mini | with_web_v2 | 300 | 0 | 0.000 | 0 | 0 |
| gpt-5-nano | no_web | 3,869 | 73 | 1.887 | 0 | 73 |
| gpt-5-nano | no_web_v2 | 300 | 7 | 2.333 | 0 | 7 |
| gpt-5-nano | with_web | 3,869 | 27 | 0.698 | 1 | 26 |
| gpt-5-nano | with_web_v2 | 300 | 2 | 0.667 | 0 | 2 |
| Muse Glimmer 30B | no_web | 300 | 0 | 0.000 | 0 | 0 |
| Muse Glimmer 30B | with_web | 298 | 0 | 0.000 | 0 | 0 |
| Nemotron 3 Super | no_web | 378 | 1 | 0.265 | 1 | 0 |
| Nemotron 3 Super | with_web | 376 | 4 | 1.064 | 4 | 0 |

Almost all invalid answers are well-formed JSON carrying a category string outside the taxonomy, not
parse failures. Only 6 responses in 18,628 were unparseable.

### A5. T4 rescored with a fixed 15-class set — **materially changes four rows**

Rescoring rule: `labels` fixed to the 15 sorted macro categories, and any INVALID prediction remapped
to a deterministic **wrong-but-valid** class (the first category in sorted order that differs from
that row's gold label). Committed tables were not touched; results are in
`audit_output/A5_rescored_table4.csv`.

| T4 row | slice | committed (union) | rescored (fixed 15) | Δ |
|---|---|---|---|---|
| kNN, MiniLM | overall | 0.8066 | 0.8066 | 0.0000 |
| kNN, MiniLM | tail | 0.6072 | 0.6072 | 0.0000 |
| llm_no_web / gpt-5-nano | overall | 0.4436 | 0.4709 | **+0.0273** |
| llm_with_web / gpt-5-nano | overall | 0.5419 | 0.5769 | **+0.0349** |
| llm_no_web / gpt-5-mini | overall | 0.5068 | 0.5399 | **+0.0331** |
| llm_with_web / gpt-5-mini | overall | 0.5883 | 0.6229 | **+0.0346** |
| llm_no_web / Nemotron | overall | 0.4208 | 0.4467 | **+0.0260** |
| llm_with_web / Nemotron | overall | 0.4314 | 0.4579 | **+0.0265** |
| llm_* / DiffusionGemma, Muse Glimmer | all | unchanged | unchanged | 0.0000 |
| routed / gpt-5-mini @0.65 | overall | 0.8611 | 0.8611 | 0.0000 |
| routed / gpt-5-mini @0.65 | tail | 0.7333 | 0.7333 | 0.0000 |
| **routed / gpt-5-nano @0.65** | **overall** | **0.7994** | **0.8524** | **+0.0530** |
| **routed / gpt-5-nano @0.65** | **tail** | **0.6704** | **0.7095** | **+0.0392** |

The headline routed/gpt-5-mini row is unaffected. The routed/gpt-5-nano row moves by +0.053 overall,
which changes the paper's nano-versus-mini comparison: the gap narrows from 0.062 to 0.009 overall
macro-F1. Every LLM-only row that emits INVALID is understated by 0.026 to 0.035.

---

## B. Confidence intervals and tests

### B1. Resampling unit per table

| Table | intervals present | unit | n_resamples | file:line |
|---|---|---|---|---|
| T2 accuracy by frequency | yes | **binomial closed form** (Wilson, z = 1.96) over transactions | n/a | `experiments/exp1_tail_characterization.py:43,107` |
| T4 main comparison | **kNN rows only** | **five seed values** | 2,000 | `experiments/exp2_fallback_comparison.py:163-165` |
| T5 critical ablation | yes | **merchant identity** (paired) | 5,000 | `src/txcat/metrics.py:57-75`, called at `experiments/exp2_fallback_comparison.py:306` |
| T8 write-back policies | **none** — **ABSENT** | n/a | n/a | columns of `results/tables/tab5_writeback_policies.csv` carry no lo/hi |
| T9 backbones | yes | **five seed values** | 2,000 | `experiments/exp5_backbone_comparison.py:103` |

Two notes on T2 and T4. T2's `n` is the pooled row count divided by the number of seeds
(`exp1_tail_characterization.py:105-106`), so the Wilson interval is computed on a single seed's
worth of transactions. In T4 only the three kNN rows carry intervals; the logistic-regression row,
all ten LLM-only rows and **both routed rows have no interval at all**, so the paper's headline
routed numbers are point estimates.

### B2. Tables that bootstrap the five seed values — **T4 and T9**, n_resamples = 2,000

Per-seed values are in `audit_output/B2_per_seed_values.csv`; the intervals they generate are in
`audit_output/B2_B3_interval_provenance.csv`. Tail macro-F1 example:

| table | backbone | five per-seed values | spread (pp) | bootstrap interval | width (pp) |
|---|---|---|---|---|---|
| T9 | MiniLM-L6-v2 | 0.632832, 0.632780, 0.633236, 0.633216, 0.632993 | 0.046 | [0.632844, 0.633179] | 0.034 |
| T9 | bge-small-en-v1.5 | 0.634038, 0.633549, 0.633935, 0.633906, 0.633904 | 0.049 | [0.633698, 0.633985] | 0.029 |
| T9 | mpnet-base-v2 | 0.639605, 0.639183, 0.638557, 0.639521, 0.639262 | 0.105 | [0.638891, 0.639504] | 0.061 |
| T9 | FinBERT | 0.537018, 0.537018, 0.537018, 0.537025, 0.537006 | 0.002 | [0.537011, 0.537022] | 0.001 |
| T9 | text-embedding-3-small | 0.643596, 0.643602, 0.643709, 0.643622, 0.643467 | 0.024 | [0.643524, 0.643665] | 0.014 |
| T4 | MiniLM-L6-v2 | 0.606986, 0.607146, 0.607176, 0.607188, 0.607438 | 0.045 | [0.607067, 0.607330] | 0.026 |
| T4 | bge-small-en-v1.5 | 0.637795, 0.636525, 0.636369, 0.637228, 0.637332 | 0.143 | [0.636603, 0.637541] | 0.094 |
| T4 | text-embedding-3-small | 0.643468, 0.643704, 0.643468, 0.643368, 0.643468 | 0.034 | [0.643408, 0.643609] | 0.020 |

These intervals describe the dispersion of five nearly identical numbers. They are not sampling
intervals over transactions or merchants and cannot be read as such.

### B3. T9 interval widths and the FinBERT case — **partly CONFIRMED**

| backbone | f1_overall | f1_head | f1_tail | acc_tail |
|---|---|---|---|---|
| MiniLM-L6-v2 | 0.0001 | 0.0001 | 0.0004 | 0.0003 |
| bge-small-en-v1.5 | 0.0001 | 0.0001 | 0.0003 | 0.0002 |
| mpnet-base-v2 | 0.0005 | 0.0006 | 0.0006 | 0.0004 |
| **FinBERT** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| text-embedding-3-small | 0.0001 | 0.0000 | 0.0002 | 0.0002 |

Widths are as printed in the committed table (4 decimal places). At full precision FinBERT's
`f1_head` and `acc_tail` widths are **exactly zero**, because all five seeds produce bit-identical
values (0.932517 and 0.593122 respectively): a bootstrap over a constant vector returns that constant
at every quantile. FinBERT's `f1_tail` and `f1_overall` are not exactly zero (0.0011 pp and 0.0004 pp)
but round to 0.0000 in the table. The cause is that FinBERT's mean-pooled embeddings place tail
merchants far enough from any neighbour that HNSW seed variation changes almost no top-1 result.

### B4. McNemar's test — **ABSENT**

Searched case-insensitively for `mcnemar` across the whole repository excluding `.venv`
(`src/`, `experiments/`, `scripts/`, `tests/`, `reproduce.py`, `analysis/`, `paper/`). No match. The
paper's Section 3.4 claims "binary paired accuracy comparisons additionally report McNemar's test".

### B5. Holm–Bonferroni or any multiple-comparison correction — **ABSENT**

Searched for `holm`, `bonferroni`, `multipletests`, `fdr_`, `p_adjust` across the same scope. No
match. T5 reports raw 95% paired-bootstrap intervals for five models with no correction. The paper's
Section 3.4 claims the correction "is applied across the five evidence-toggle model comparisons".

### B6. T5 paired bootstrap — **CONFIRMED**

`src/txcat/metrics.py:57-75`: resamples row indices with replacement and returns
`mean(a) − mean(b)`; `n_boot` defaults to 5,000 and is not overridden at the call site
(`experiments/exp2_fallback_comparison.py:306`, `seed=seeds[0]` = 42). The unit is merchant
identity: `common` is the intersection of tail merchants with both a with-web and a no-web cached
prediction (`exp2_fallback_comparison.py:297-299`), and `a` and `b` are correctness vectors indexed
by that same merchant list (`:304-305`). The pairing is therefore on the same merchant across the two
conditions, and one bootstrap index vector is applied to both arms.

---

## C. The gate

Population: the analysis slice, 83,115 transactions and 9,174 merchants, MiniLM backbone,
t = 0.65. Counts are for seed 42; the five-seed spread is under 0.1 pp on every quantity.

### C1. Gate decision × merchant status

| | unseen, f = 0 | seen, f ≥ 1 | total |
|---|---|---|---|
| route | 6,401 | 153 | 6,554 |
| no route | 8,582 | 67,979 | 76,561 |

As a share of all test transactions: routed-and-unseen 7.70%, routed-and-seen 0.18%,
not-routed-and-unseen **10.33%**, not-routed-and-seen 81.79%.

### C2. Gate decision × retrieval correctness

| | retrieval wrong | retrieval correct | total |
|---|---|---|---|
| route | 4,423 | 2,131 | 6,554 |
| no route | 5,068 | 71,493 | 76,561 |

Shares: 5.32%, 2.56%, 6.10%, 86.02%.

### C3. Gate as a detector of "retrieval is wrong"

| seed | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 42 | 4,423 | 2,131 | 5,068 | 0.6749 | 0.4660 | 0.5513 |
| 43 | 4,425 | 2,112 | 5,068 | 0.6769 | 0.4661 | 0.5521 |
| 44 | 4,416 | 2,134 | 5,068 | 0.6742 | 0.4656 | 0.5508 |
| 45 | 4,415 | 2,136 | 5,068 | 0.6739 | 0.4656 | 0.5507 |
| 46 | 4,421 | 2,129 | 5,068 | 0.6750 | 0.4659 | 0.5513 |

Mean precision 0.675, recall 0.466, F1 0.551.

### C4. Gate recall on unseen volume

**42.7%** (mean over seeds; 0.4261 to 0.4272). The gate leaves **57.3% of unseen transactions
unrouted**, because an unseen merchant string can still sit within 0.65 cosine of some indexed
merchant.

### C5–C7. Counterfactual gates

Cache coverage limits this to merchants with a cached gpt-5-mini with-web answer: **3,079 of 9,174
merchants (33.6%) and 20,493 of 83,115 transactions (24.7%)**. All rows below are computed on that
covered subset only, mean over five seeds
(`audit_output/C5_C7_gate_variants.csv`, per-seed in `..._per_seed.csv`).

| gate | routed | overall F1 | unseen F1 | observed-tail F1 |
|---|---|---|---|---|
| none (kNN only) | 0.0% | 0.8047 | 0.4808 | 0.9700 |
| threshold t = 0.65 | 10.5% | 0.8600 | 0.6482 | 0.9724 |
| **C5 oracle** (route exactly the wrong) | 15.6% | **0.8854** | **0.7178** | 0.9831 |
| **C6 random** @ matched coverage | 10.5% | **0.7377** | 0.4711 | 0.9380 |
| **C7 membership** (not in index) | 27.4% | 0.7870 | 0.5246 | 0.9698 |
| **C7 hybrid** (not in index OR cos < 0.65) | 27.4% | 0.7870 | 0.5246 | 0.9698 |

Three observations. The random gate at matched coverage scores **below doing nothing** (0.738 vs
0.805), so the similarity gate is doing real work rather than the fallback being uniformly helpful.
The oracle gate is 2.5 points of overall macro-F1 above the threshold gate, which bounds what a
better gate could buy. The hybrid is identical to the membership gate because every routed-by-cosine
row is already a not-in-index row: an indexed merchant matches its own string at similarity 1.0.

One caveat on C7. The membership gate is **not** identical to f = 0
(`audit_output/C_meta.json`: `membership_gate_equals_f0: false`). The index holds 14,878 merchants
while the unfiltered training window has 17,166, because `merchant_table` drops merchants with no
label-bearing training row (`src/txcat/knn.py:12-25`). Membership therefore routes 27.4% against the
18.0% unseen share.

---

## D. Re-stratification (f = 0 vs 1 ≤ f ≤ 3)

Computed from the same cached predictions, no new API calls. Full table:
`audit_output/D_restratified.csv`.

### D1/D3. T4 rows by stratum

| method | unseen f = 0 | observed tail 1 ≤ f ≤ 3 |
|---|---|---|
| kNN, MiniLM | **0.4795** | **0.9703** |
| llm_no_web / gpt-5-nano | 0.3459 | 0.2958 |
| llm_with_web / gpt-5-nano | 0.4879 | 0.4335 |
| llm_no_web / gpt-5-mini | 0.4093 | 0.4394 |
| llm_with_web / gpt-5-mini | 0.5288 | 0.5920 |
| llm_no_web / DiffusionGemma | 0.3342 | 0.3817 |
| llm_with_web / DiffusionGemma | 0.4294 | 0.5490 |
| llm_no_web / Nemotron | 0.4549 | 0.3721 |
| llm_with_web / Nemotron | 0.3704 | 0.4783 |
| llm_no_web / Muse Glimmer | 0.3962 | 0.4334 |
| llm_with_web / Muse Glimmer | 0.4366 | 0.5908 |
| routed / gpt-5-nano @0.65 | 0.5877 | 0.9728 |
| routed / gpt-5-mini @0.65 | **0.6492** | **0.9728** |

Stratum sizes: unseen 5,434 rows over 1,941 merchants; observed tail 1,865 rows over 559 merchants.
Open-weight rows cover only 180 to 201 merchants in the unseen stratum and 118 to 124 in the observed
tail.

This is the sharpest result in the audit. Retrieval already scores **0.970** on the observed tail and
**0.480** on unseen merchants, and routing moves the observed tail by **+0.0025** while moving unseen
by **+0.170**. The pooled f ≤ 3 "tail" number in the paper averages a solved stratum with an unsolved
one, and every point of the routed gain comes from the unseen stratum.

### D2. T5 evidence toggle by stratum (merchant-level accuracy)

| model | stratum | with web | no web | diff | merchants |
|---|---|---|---|---|---|
| gpt-5-nano | unseen | 0.5312 | 0.3601 | +0.1710 | 1,941 |
| gpt-5-nano | observed tail | 0.5581 | 0.3685 | +0.1896 | 559 |
| gpt-5-mini | unseen | 0.6203 | 0.4977 | +0.1226 | 1,941 |
| gpt-5-mini | observed tail | 0.6386 | 0.5116 | +0.1270 | 559 |
| DiffusionGemma 26B | unseen | 0.6243 | 0.5193 | +0.1050 | 181 |
| DiffusionGemma 26B | observed tail | 0.6387 | 0.4454 | +0.1933 | 119 |
| Muse Glimmer 30B | unseen | 0.5944 | 0.5056 | +0.0889 | 180 |
| Muse Glimmer 30B | observed tail | 0.6271 | 0.5000 | +0.1271 | 118 |
| Nemotron 3 Super | unseen | 0.5888 | 0.5279 | +0.0609 | 197 |
| Nemotron 3 Super | observed tail | 0.5574 | 0.4836 | +0.0738 | 122 |

The evidence effect survives in both strata for all five models and is, if anything, slightly larger
in the observed tail.

**T6 (Oklahoma) cannot be stratified — not determinable.** Every Oklahoma cold-start merchant is by
construction absent from the DC index, so f = 0 for all 800 and the observed-tail stratum is empty
(`src/txcat/fes.py:24-33` sets `train_freq = 0` for the whole set).

---

## E. Write-back

### E1. Definition of a wrong write — **CONFIRMED, with a caveat**

`src/txcat/stream.py:81`: `n_wb_wrong += int(res.category != r.category)`. It is an exact string
comparison between the model's category and the gold category of **the single transaction that
triggered the write**, not a merchant-level majority vote and not a comparison against the merchant's
modal label. Writes happen once per merchant, on first fallback
(`src/txcat/stream.py:70`), and only when `res.valid` is true (`src/txcat/writeback.py:17`).

### E2. The 948 writes — **CONFIRMED and reproduced exactly**

Replaying the stream at seed 42 with policy `always` reproduces the committed summary exactly:
948 writes, error rate 46.84%, macro-F1 0.9055, total cost $10.38, final index 15,826. **444 of the
948 written labels are wrong.** Top confusion pairs
(`audit_output/E2_writeback_confusion_pairs.csv`):

| gold | predicted | count |
|---|---|---|
| education_gov_membership | professional_services | 33 |
| industrial_hardware | professional_services | 28 |
| software_electronics | professional_services | 22 |
| software_electronics | education_gov_membership | 18 |
| professional_services | entertainment_media | 17 |
| software_electronics | entertainment_media | 14 |
| retail | professional_services | 13 |
| professional_services | health | 13 |
| professional_services | retail | 13 |
| professional_services | education_gov_membership | 13 |

`professional_services` is the predicted label in 5 of the top 10 pairs, absorbing 89 wrong writes;
it behaves as the model's default guess. This overlaps heavily with the taxonomy-mismatch class that
dominates T10.

### E3. Export — **done**

`audit_output/E3_wrong_writes.csv` (444 rows: merchant, predicted category, gold category, model
confidence, top-1 similarity, date, txn_id) and `audit_output/E3_all_writes.csv` (all 948 with a
`wrong` flag), ready to hand-audit against the T10 taxonomy.

Confidence separates the two groups only weakly
(`audit_output/E3_confidence_by_correctness.csv`): correct writes mean 0.605, median 0.62; wrong
writes mean 0.528, median 0.58. The distributions overlap across the whole range, which is the
mechanism behind the paper's finding that confidence gating on the write path removes the savings
along with the errors.

### E4. The 70,333-transaction stream — **CONFIRMED**

Filter chain, from `experiments/exp3_streaming_convergence.py:83-95`:

| stage | rows |
|---|---|
| full DC test window | 90,466 |
| after `category.notna() & ambiguous == 0` (`exp3:83`) | 83,115 |
| after scope `fes_tail`: keep rows where train_freq > 3 **or** merchant ∈ frozen 3,000-merchant set (`exp3:85-94`) | **70,333** |

12,782 rows are dropped by the scope filter: tail merchants outside the evaluation set, excluded so
that every fallback has a cached answer. Note the frequency used for `train_freq > 3` here is
computed on the **filtered** training frame (`exp3:90`), unlike T1 and T2, which use unfiltered
training counts.

---

## F. Cost accounting

### F1. T11 recomputed from cached usage records — **CONTRADICTED for the OpenAI line**

Recomputed by summing `result.tokens_in` and `result.tokens_out` over every record in
`cache/llm/packed/` and applying the configured list prices
(`audit_output/F1_usage_from_cache.csv`):

| model | calls | tokens in | tokens out | USD at list price | billable |
|---|---|---|---|---|---|
| gpt-5-mini | 8,338 | 3,699,743 | 526,105 | 1.9771 | yes |
| gpt-5-nano | 8,338 | 3,699,743 | 491,485 | 0.3816 | yes |
| DiffusionGemma 26B | 600 | 269,167 | 35,300 | 0.0410 | no (free endpoint) |
| Muse Glimmer 30B | 598 | 269,599 | 94,414 | 0.1106 | no (free endpoint) |
| Nemotron 3 Super | 754 | 341,751 | 163,358 | 0.2495 | no (free endpoint) |

| line | T11 / ledger | recomputed from cache | gap |
|---|---|---|---|
| OpenAI chat completions | $3.0673 | **$2.3587** | **−$0.7086** |
| OpenAI built-in web search | $6.3783 | $6.3783 (300 records) | 0.00 |
| Firecrawl search | $41.4411 | $41.3983 (3,869 records × $0.0107) | −$0.043 |

The OpenAI chat line cannot be reconstructed from the cache. The ledger holds **18,964** `llm`
entries against **18,628** cached records, a surplus of 336. `SpendLedger.add` is called before each
request (`src/txcat/llm_fallback.py:159`) and `adjust_last` replaces the estimate only on success
(`:220`), so calls that ultimately failed leave their pre-flight estimate in the ledger with no cache
record. T11's $3.07 is an estimate-inclusive figure; $2.36 is the cost of the work that actually
landed in the frozen cache.

### F2. The $41.44 Firecrawl figure — **credits consumed, valued at a reference price; NOT a prepaid balance**

The ledger holds 4,388 `search` entries, of which **3,873 are non-zero** (the other 515 were refunded
to zero on failure, `src/txcat/web_search.py:158`). 3,873 × $0.0107 = $41.44, and $0.0107 is the
configured reference rate (`configs/default.yaml:51`, `price_per_1k_usd: 10.7`), flagged
`prepaid: true` on line 54. So the figure is the number of search calls these runs actually issued,
multiplied by a list price that was never paid: the credits were bought in advance. It is neither a
balance nor cash spent. For scale, the stated 10,000-credit prepaid balance at 2 credits per search
is 5,000 searches, so these runs consumed roughly 77% of it.

### F3. The $0.54 per-1k figure — **CONFIRMED, and it is 97% search**

Recomputed per seed (`audit_output/F3_cost_per_1k_split.csv`), routed gpt-5-mini at t = 0.65 over
20,304 evaluation rows:

| component | USD per 1k | share |
|---|---|---|
| model tokens | 0.0143 | 2.6% |
| web search | 0.5275 | 97.3% |
| **total** | **0.5418** | 100% |

Matches the committed 0.5417. The paper's claim that cost "is dominated by the search call, not by
tokens" is confirmed and can now be quantified: search is 97% of the routed cost, which is why
gpt-5-nano and gpt-5-mini cost nearly the same to route.

### F4. The abstract's "$9.45 total" — **CONFIRMED that it excludes Firecrawl**

$9.45 = $3.0673 (OpenAI chat) + $6.3783 (OpenAI built-in search), rounded. It excludes the $41.44
Firecrawl line that appears in T11 of the same paper. Including it, the reference-priced total is
**$50.89**. Cash actually paid to OpenAI is $9.45; Firecrawl credits were prepaid separately, and the
open-weight models were free.

---

## G. Loose ends

### G1. Appendix A with a 16-class sensitivity comparison — **ABSENT**

Searched `paper/main.tex`, `paper/README.md`, `paper/tables/`, `docs/` and `README.md` for
`appendix`, `16-class`, `16 class`, `sixteen`. No match anywhere, and `paper/main.tex` contains no
`\appendix` command. Section 3.2's forward reference has no target. (Separately, note that the
*de facto* 16th class already exists in the scoring as INVALID — see A3 — which is not what the
promised sensitivity analysis describes.)

### G2. Normalization rule set — **defined in code; no labelled paper section**

Defined at `src/txcat/normalizer.py:55` (`normalize_merchant`), with the prefix list at
`src/txcat/normalizer.py:9-30` and the reference-code heuristic at `:44-52`. The paper covers it in
an **unnumbered** `\paragraph{Normalization.}` inside Section 4 "System" (`paper/main.tex:188`).
There is no numbered section or subsection for it, so Section 3.3's "normalized merchant m" has no
labelled cross-reference target.

### G3. 10.4% versus 7.9% fallback rate — **both correct, different populations**

| figure | population | rows | rate |
|---|---|---|---|
| T4 fallback rate | frozen evaluation set (3,000 merchants, tail-weighted) | 20,304 | **0.1036** |
| Section 6.1 | full analysis slice | 83,115 | **0.0788** |

The evaluation set deliberately over-samples tail merchants (2,500 tail to 500 head), so a larger
share of its rows fall below the threshold. Neither number is wrong; they are not interchangeable and
the paper should label each with its population.

### G4. MiniLM 0.607 versus 0.633 — **both current; explained**

Both numbers reproduce exactly from the current caches
(`audit_output/G4_minilm_discrepancy.csv`), so neither is stale:

| backbone | T4 population (evaluation set) | T9 population (full slice) | difference |
|---|---|---|---|
| MiniLM-L6-v2 | 0.6072 | 0.6330 | **+0.0258** |
| bge-small-en-v1.5 | 0.6370 | 0.6339 | −0.0032 |
| text-embedding-3-small | 0.6435 | 0.6436 | +0.0001 |

The cause is thin-class variance amplified by unweighted macro averaging
(`audit_output/G4_per_class_decomposition.csv`). The evaluation-set tail has very small support in
several categories — groceries 16 rows, financial_postal_shipping 59, airlines_travel 69 — yet each
still carries 1/15 of the macro average. Per-class deltas between the two populations:

| category | evaluation-set support | MiniLM Δ | bge Δ |
|---|---|---|---|
| groceries | 16 | +0.2678 | +0.1138 |
| financial_postal_shipping | 59 | +0.1092 | +0.0716 |
| airlines_travel | 69 | **+0.0871** | **−0.1742** |
| retail | 380 | +0.2454 | +0.2503 |

For MiniLM the three thinnest classes all push the same way, contributing +0.031 of the +0.026 net
move. For bge they cancel, contributing +0.001, because bge loses 0.17 on airlines_travel exactly
where MiniLM gains 0.09. This is sampling noise in classes with fewer than 100 rows, not a backbone
property, and it is a direct consequence of macro-F1 weighting a 16-row class as heavily as a
4,060-row one.

### G5. T1's 8.9% and 0.17% — **base is all processed DC rows, and 8.9% double-counts**

Source: `results/tables/tab0_taxonomy_coverage.csv`, written at
`experiments/exp0_data_audit.py:69-71`. The base is **280,047 rows — every processed DC row, train
plus test plus rows outside the chosen window** — not the test set and not the analysis slice.

| quantity | reported | base |
|---|---|---|
| ambiguous | 8.9% | `df["ambiguous"].mean()` over 280,047 rows |
| unlabeled | 0.17% | `df["mcc"].isna().mean()` over 280,047 rows |

Two problems. The `ambiguous` flag is 1 for **both** the four NEC codes **and** rows with no usable
MCC (`src/txcat/data/taxonomy.py:224`), so the 8.9% figure includes the 0.17% unlabeled rows and the
two lines cannot be added. And on test rows alone the four NEC codes are 7.97% and unlabeled 0.16%,
so T1's percentages describe a different population from every other table in the paper.

### G6. Was t = 0.65 selected on the evaluation set? — **CONTRADICTED (no held-out selection)**

The sweep runs over `THRESHOLDS` from 0.30 to 0.95 (`experiments/exp2_fallback_comparison.py:45`)
using `knn_cache[bb0]`, which is the kNN prediction on **the same frozen 3,000-merchant evaluation
set that T4 reports** (`:232-248`). The reported operating point is the configuration constant
`gate.threshold: 0.65` (`configs/default.yaml:58`), and Figure 2's marker is the **argmax of overall
macro-F1 on that same set** (`:283`, `g.loc[g["macro_f1"].idxmax(), "threshold"]`).

For gpt-5-mini those coincide exactly: the argmax is 0.65 at 0.8611. For gpt-5-nano the argmax is
0.40 at 0.8069, while the paper reports 0.65 at 0.7994. So the headline routed/gpt-5-mini number is
reported at the threshold that maximizes performance on the very set it is evaluated on. There is no
validation split anywhere in the threshold selection path.

---

## PAPER CLAIMS NOT SUPPORTED BY CODE

Every CONTRADICTED and ABSENT finding, in one place.

| # | Claim in the paper | Verdict | Evidence |
|---|---|---|---|
| 1 | "The class set is the fixed 15-category taxonomy" (§3.4) | **CONTRADICTED** | `src/txcat/metrics.py:20` passes no `labels`; class set is the per-comparison union |
| 2 | "A category with zero support and zero predictions contributes zero" (§3.4) | **CONTRADICTED** | absent categories are excluded from the average, not scored zero |
| 3 | Implicit: T4 rows are comparable | **CONTRADICTED** | 6 of 13 rows score over 16 classes, 7 over 15 (A3); rescoring moves routed/gpt-5-nano by +0.053 (A5) |
| 4 | "Binary paired accuracy comparisons additionally report McNemar's test" (§3.4) | **ABSENT** | no `mcnemar` anywhere in the repository |
| 5 | "Holm–Bonferroni correction is applied across the five evidence-toggle model comparisons" (§3.4) | **ABSENT** | no correction implemented; T5 reports raw intervals |
| 6 | Seeds are "summarized by mean and standard deviation or min–max, **not bootstrapped as five independent observations**" (§3.4) | **CONTRADICTED** | `exp2:163-165` and `exp5:103` bootstrap the five seed values with `n_boot = 2000` |
| 7 | Implicit: T4's headline routed rows carry intervals | **ABSENT** | only the three kNN rows have lo/hi; both routed rows and all LLM rows are point estimates |
| 8 | Implicit: T8 write-back numbers carry uncertainty | **ABSENT** | `tab5_writeback_policies.csv` has no interval columns |
| 9 | "Appendix A specifies the corresponding 16-class sensitivity comparison" (§3.2) | **ABSENT** | no appendix exists in `paper/main.tex` |
| 10 | T11's OpenAI chat line ($3.07) | **CONTRADICTED** | only $2.36 is reconstructible from cached usage; 336 ledger entries have no cache record |
| 11 | Abstract's "$9.45 total" presented as the study's total cost | **CONTRADICTED** | excludes the $41.44 Firecrawl line printed in T11; reference-priced total is $50.89 |
| 12 | Implicit: t = 0.65 is an independent operating point | **CONTRADICTED** | the threshold sweep, the argmax marker and the reported result all use the same 3,000-merchant set |
| 13 | T1's exclusion percentages describe the analysis population | **CONTRADICTED** | they use all 280,047 processed rows, and the 8.9% figure double-counts the 0.17% unlabeled rows |
| 14 | Implicit: the pooled f ≤ 3 "tail" is one population | **CONTRADICTED** | retrieval scores 0.970 on observed tail and 0.480 on unseen; routing moves the former by +0.003 and the latter by +0.170 (D1) |

Items 1 to 3 and 6 to 8 are scoring and statistics claims that can be fixed either by changing the
code and regenerating, or by rewriting Section 3.4 to describe what the code does. Items 4, 5 and 9
are claims of work that does not exist and should be removed or implemented. Item 12 needs either a
validation split or an explicit statement that the threshold is selected in-sample. Item 14 is not an
error so much as the strongest result the current tables obscure.

---

## Files written by this audit

All under `audit_output/`. Nothing else in the repository was created or modified.

| file | contents |
|---|---|
| `A3_class_set_sizes.csv` | class-set size and extra classes per T4 row |
| `A4_malformed_per_model.csv` | invalid / unparseable / off-taxonomy counts per model and prompt |
| `A5_rescored_table4.csv` | T4 before and after fixed-15 rescoring, per slice |
| `B2_per_seed_values.csv` | per-seed metrics for T4 and T9, all backbones |
| `B2_B3_interval_provenance.csv` | per-seed values, spread and the bootstrap interval each produces |
| `C1_gate_vs_merchant_status.csv` | 2×2 gate × unseen |
| `C2_gate_vs_correctness.csv` | 2×2 gate × retrieval correctness |
| `C3_C4_gate_detection.csv` | precision, recall, F1, unseen recall per seed |
| `C5_C7_gate_variants.csv`, `C5_C7_gate_variants_per_seed.csv` | oracle, random, membership and hybrid gates |
| `C_meta.json` | cache coverage and the membership-gate caveat |
| `D_restratified.csv` | T4 and T5 split into unseen and observed-tail strata |
| `E1_E2_summary.json` | reproduced stream summary and the definition of a wrong write |
| `E2_writeback_confusion_pairs.csv` | gold × predicted pairs for the 444 wrong writes |
| `E3_wrong_writes.csv`, `E3_all_writes.csv` | per-write export for hand-auditing |
| `E3_confidence_by_correctness.csv` | confidence distribution of right versus wrong writes |
| `E4_stream_scope.json` | the 90,466 → 83,115 → 70,333 filter chain |
| `F1_usage_from_cache.csv` | token usage and list-price cost per model |
| `F3_cost_per_1k_split.csv` | routed cost per 1k split into tokens and search |
| `G4_minilm_discrepancy.csv` | tail F1 on both populations for three backbones |
| `G4_per_class_decomposition.csv` | per-class tail F1 across populations, MiniLM and bge |
| `G4_tail_class_support.csv` | class support in the evaluation-set tail versus the full tail |
| `G_loose_ends.json` | G3 and G5 supporting numbers |

---

# Addendum: three unverifiable items plus the Figure 2 replot

Same commit `70c054ed73`. No tracked file was modified; new outputs are listed at the end.
Table numbering in this addendum follows the revised manuscript the request cites (T6 main
comparison, T7 critical ablation, T8 Oklahoma, T12 spend).

## 1. Cost reconciliation — the $0.71 gap has two independent causes

**Neither cause is a missing experiment.** Everything the request asked about is already inside the
$2.3587 figure, except the streaming experiment, which issues no live calls at all: `exp3` wraps the
fallback in `CachedOrUncached` with `allow_live=False`
(`experiments/exp3_streaming_convergence.py:32-44`), so T10's cost column is replay accounting, not
spend.

Full itemisation of cached OpenAI chat cost (`audit_output/V1_openai_cost_itemised.csv`):

| component | merchants | gpt-5-mini | gpt-5-nano | subtotal |
|---|---|---|---|---|
| DC evaluation set, both main prompts | 3,000 | $1.4366 | $0.2769 | **$1.7135** |
| Oklahoma cold-start set, both main prompts | 798 | $0.3712 | $0.0722 | **$0.4434** |
| Pilot remainder (merchants in neither frozen set) | 71 | $0.0327 | $0.0063 | **$0.0390** |
| Prompt-sensitivity `_v2` passes, both conditions | 300 | $0.1367 | $0.0264 | **$0.1631** |
| Streaming experiment | — | $0 | $0 | **$0.0000** |
| **total** | | **$1.9772** | **$0.3818** | **$2.3590** |

(The $2.3590 versus $2.3587 difference is rounding of the per-line figures.)

**Cause A — the line is not OpenAI-only ($0.4573).** T12's "OpenAI chat completions" figure is the
ledger's entire `llm` kind, which also carries every NVIDIA NIM call priced at reference rates. Those
calls were free.

| vendor | model | entries | ledger USD |
|---|---|---|---|
| OpenAI (paid) | gpt-5-mini | 8,338 | 2.1915 |
| OpenAI (paid) | gpt-5-nano | 8,338 | 0.4185 |
| NIM (free) | nemotron-3-super-120b-a12b | 793 | 0.2637 |
| NIM (free) | muse-glimmer-30b | 600 | 0.1110 |
| NIM (free) | diffusiongemma-26b-a4b-it | 600 | 0.0410 |
| NIM (free) | deepseek-v4-flash-0731 | 275 | 0.0384 |
| NIM (free) | gemma-4-31b-it | 10 | 0.0016 |
| NIM (free) | nemotron-3.5-lightning-30b-a3b | 10 | 0.0016 |
| | **OpenAI subtotal** | 16,676 | **2.6100** |
| | **NIM subtotal** | 2,288 | **0.4573** |
| | **T12 line as printed** | 18,964 | **3.0673** |

Three of those NIM models (deepseek-v4-flash, gemma-4-31b, nemotron-3.5-lightning) were abandoned
during setup and appear in no table, yet their reference cost is inside the published total.

**Cause B — the OpenAI portion itself over-counts by $0.2513.** The ledger's OpenAI subtotal is
$2.6100 against $2.3587 recomputable from recorded token usage, with **entry counts matching exactly**
(16,676 each, zero surplus). The cause is a race in the true-up:
`SpendLedger.add` writes a pre-flight estimate (`src/txcat/llm_fallback.py:155-159`) and
`adjust_last` replaces "the most recent matching entry" (`src/txcat/budget.py:63-70`). With 16
concurrent workers several calls update the same latest entry, so some estimates are never replaced.
The per-day evidence confirms it (`audit_output/V1_ledger_vs_cache_by_day.csv`):

| date | run | gap (ledger − cache) |
|---|---|---|
| 2026-09-13 | pilot, concurrent | +$0.0119 |
| 2026-09-14 | main cache pass, 16 workers | **+$0.2394** |
| 2026-09-15 | extras, sequential | **+$0.0000** |

The sequential run has a gap of exactly zero. The concurrent run carries the whole discrepancy.

**Which figure belongs in T12: $2.36.** It is reconstructible from the committed cache, contains only
paid calls, and is the amount attributable to work that survives in the artifact. The free NIM models
belong on their own line at $0.00, with reference rates in a footnote if wanted. If the intent is to
report cash actually charged by OpenAI, that is $2.61 and it should be labelled as an
estimate-inclusive ledger total that the cache cannot reproduce.

## 2. Nemotron merchant count — **CONFIRMED, 319 is the paired intersection**

`audit_output/V2_ablation_merchant_counts.csv`:

| model | with-web arm | no-web arm | paired intersection | union |
|---|---|---|---|---|
| gpt-5-nano | 2,500 | 2,500 | 2,500 | 2,500 |
| gpt-5-mini | 2,500 | 2,500 | 2,500 | 2,500 |
| DiffusionGemma 26B | 300 | 300 | 300 | 300 |
| Muse Glimmer 30B | 298 | 300 | 298 | 300 |
| **Nemotron 3 Super** | **322** | **324** | **319** | **327** |

319 is the count of **tail merchants holding a cached answer in both arms**, built at
`experiments/exp2_fallback_comparison.py:297-299` as
`tail_m ∩ with_web.index ∩ no_web.index`. That is the correct denominator for a paired test, since
the bootstrap resamples merchant identities present in both conditions.

The audit's earlier D-table figures (324 and 322) are per-arm coverage counts, which is why they
disagree: 5 merchants have a no-web answer but no with-web answer, and 3 the reverse. **No error in
the paper.** The definition is worth one clause in the caption, because a reader cannot otherwise
tell why Nemotron's n is neither 322 nor 324.

## 3. Previously unaudited results

### 3a. Oklahoma cold start (T8) — **all five rows reproduce exactly**

| method | published | recomputed | class-set size |
|---|---|---|---|
| kNN, MiniLM on the DC index | 0.392 | **0.3920** | 15 |
| llm_no_web / gpt-5-nano | 0.321 | **0.3212** | 16 (INVALID) |
| llm_with_web / gpt-5-nano | 0.459 | **0.4587** | 16 (INVALID) |
| llm_no_web / gpt-5-mini | 0.503 | **0.5028** | 16 (INVALID) |
| llm_with_web / gpt-5-mini | 0.576 | **0.5757** | 16 (INVALID) |

Population: **800 merchants, 6,318 transaction rows**, all with f = 0 by construction. Two caveats
for the caption. The metric is computed over the 6,318 rows while the column is headed
"merchants", so it is transaction-weighted despite the label. And the four LLM rows are again scored
over 16 classes against the kNN row's 15, the A3 problem repeating here: the gap between retrieval
and the fallback is understated by the same mechanism as in T6.

### 3b. Agentic search (§6.4) — **accuracy CONFIRMED, cost comparison CONTRADICTED**

| quantity | published | recomputed | verdict |
|---|---|---|---|
| accuracy, fixed snippets | 0.613 | **0.6133** | confirmed |
| accuracy, agent-issued | 0.620 | **0.6200** | confirmed |
| accuracy, no web (context) | 0.513 | **0.5133** | confirmed |
| searches per merchant, fixed | 1.00 | **1.0000** | confirmed |
| searches per merchant, agentic | 1.64 | **1.6367** | confirmed |
| USD per 1k, agentic | 21.16 | **21.161** | confirmed |
| **USD per 1k, fixed snippets** | **1.63** | **11.00** | **contradicted** |

The two cost figures use different denominators. The agentic figure is dollars per thousand
**merchants**, computed at `experiments/exp2_fallback_comparison.py:447-449` as
`1000 × mean(usd per merchant)`. The $1.63 is lifted from T6's `llm_with_web/gpt-5-mini` row
(`scripts/make_paper_tables.py`, `extras()`), which is dollars per thousand **transactions** over the
20,304-row evaluation set, where 3,000 searches are amortised across many repeat transactions per
merchant.

On a common per-thousand-merchant basis the fixed-snippet strategy costs **$11.00** (one search at
$0.0107 plus tokens). The correct statement is that agent-issued search costs **1.9× more**, not 13×.
The qualitative conclusion survives, but the headline multiple does not.

### 3c. Logistic-regression baseline (T6) — **reproduces**

| slice | published | recomputed | Δ |
|---|---|---|---|
| overall | 0.619 | 0.6191 | +0.0001 |
| head (f > 3) | 0.749 | 0.7494 | +0.0004 |
| old tail (f ≤ 3) | 0.399 | 0.3991 | +0.0001 |

Single fit, seed 42, `C = 4.0`, `max_iter = 2000` (`src/txcat/baselines.py:12-18`), trained on all
training merchants and evaluated on the frozen evaluation set. Note it is a one-seed row sitting in a
five-seed table.

### 3d. Prompt sensitivity (§6.3) — **reproduces; range is 1.3 to 2.7 points**

| model | condition | prompt v1 | prompt v2 | absolute difference |
|---|---|---|---|---|
| gpt-5-nano | no web | 0.3533 | 0.3700 | **1.67 pp** |
| gpt-5-nano | with web | 0.5233 | 0.5500 | **2.67 pp** |
| gpt-5-mini | no web | 0.5133 | 0.5367 | **2.33 pp** |
| gpt-5-mini | with web | 0.6133 | 0.6267 | **1.33 pp** |

All four on the same 300-merchant tail subset. The published range of 1.3 to 2.7 points is correct.
In every pair the rewritten prompt scores **higher**, which is worth a clause: the check bounds
prompt sensitivity but does not establish that the frozen prompt is the better of the two.

## 4. Figure 2 replot — written to new files, committed figure untouched

*The request's final sentence was truncated mid-word ("so the caption no"). I regenerated the left
panel from data and report its statistics so any caption claim can be checked; say if you meant
something else.*

**Right panel, three series, same axes and same binning** (12 fixed-width bins from the minimum
similarity to 1.0, bins with fewer than 50 rows dropped, exactly as
`analysis/plots.py:83-110`): `audit_output/figures/fig2_right_three_series.{pdf,png}`, data in
`audit_output/V4_fig2_right_data.csv`.

Under that binning the grouping degenerates, and this exposes a defect in the committed figure:

| series | bins that clear the 50-row filter | rows |
|---|---|---|
| unseen (f = 0) | 12 | 14,983 |
| observed tail (1 ≤ f ≤ 3) | **1** | 4,937 |
| head (f > 3) | **2** | 63,195 |

The cause is that indexed merchants match their own string at similarity 1.0
(`audit_output/V4_similarity_concentration.csv`):

| series | rows | ≥ 0.99 | ≥ 0.95 | < 0.65 | median |
|---|---|---|---|---|---|
| unseen | 14,983 | 0.81% | 5.17% | 42.72% | 0.684 |
| observed tail | 4,937 | **98.52%** | 98.52% | 0.97% | 1.000 |
| head | 63,195 | **99.79%** | 99.79% | 0.17% | 1.000 |

**The committed figure's head line is two points joined by a straight segment**, one at similarity
0.485 carrying 62 rows at accuracy 0.000 and one at 0.973 carrying 63,064 rows at accuracy 0.949. The
apparent smooth relationship between similarity and accuracy for head merchants is interpolation
across a region that contains 62 of 63,195 head transactions. The same applies to the committed
two-series version, where that line is labelled "head". This should not be presented as a trend.

**Supplementary variant**, per-series quantile bins (10 per series, same 50-row filter):
`audit_output/figures/fig2_right_three_series_quantile.{pdf,png}`, data in
`audit_output/V4_fig2_right_quantile_data.csv`. It gives unseen 10 bins and the two seen series 5
each, and shows the real structure: unseen accuracy climbs from 0.23 to about 0.9 across the
similarity range and crosses 0.5 almost exactly at the gate threshold, while both seen series sit
flat above 0.93. If the panel is meant to justify the gate, this variant carries the argument and the
fixed-width one does not.

**Left panel regenerated from data**: `audit_output/figures/fig2_left_zipf.{pdf,png}`, statistics in
`audit_output/V4_fig2_left_stats.json`.

| quantity | value |
|---|---|
| fitted power-law slope | **−1.1029** |
| training merchants | 17,166 |
| merchants seen exactly once | 8,036 (**46.81%**) |
| largest merchant frequency | 24,781 |
| frequency basis | unfiltered training window, normalized merchants |

The paper's −1.10 and 47% both hold.

## Addendum outputs

| file | contents |
|---|---|
| `V1_openai_cost_itemised.csv` | cached OpenAI cost by model, prompt and merchant set |
| `V1_ledger_vs_cache_by_day.csv` | ledger versus cache per day, showing the concurrency gap |
| `V2_ablation_merchant_counts.csv` | per-arm, intersection and union merchant counts per model |
| `V3a_oklahoma_llm_rows.csv`, `V3a_oklahoma_knn.csv` | T8 recomputed with class-set sizes |
| `V3b_agentic.csv` | agentic versus fixed-snippet accuracy, searches and cost |
| `V3c_lr_baseline.csv` | logistic-regression row recomputed |
| `V3d_prompt_sensitivity.csv` | all four model/condition pairs |
| `V4_fig2_right_data.csv`, `V4_fig2_right_quantile_data.csv` | plotted points for both variants |
| `V4_similarity_concentration.csv` | similarity distribution per series |
| `V4_fig2_left_stats.json` | regenerated left-panel statistics |
| `figures/fig2_right_three_series.{pdf,png}` | three-series replot, original binning |
| `figures/fig2_right_three_series_quantile.{pdf,png}` | three-series replot, quantile binning |
| `figures/fig2_left_zipf.{pdf,png}` | left panel regenerated |
