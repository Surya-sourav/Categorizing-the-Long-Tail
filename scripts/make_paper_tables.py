"""Generate the paper's LaTeX tables from the committed result CSVs.

Every number in paper/main.tex that is not prose comes from here, so the paper cannot drift from
results/. Run after reproduce.py:

    .venv/bin/python scripts/make_paper_tables.py

Writes paper/tables/*.tex (booktabs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "tables"

BACKBONE_SHORT = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5": "bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2": "mpnet-base-v2",
    "ProsusAI/finbert": "FinBERT",
    "text-embedding-3-small": "text-embedding-3-small",
}
MODEL_SHORT = {
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5-mini": "gpt-5-mini",
    "google/diffusiongemma-26b-a4b-it": "DiffusionGemma 26B-A4B",
    "nvidia/nemotron-3-super-120b-a12b": "Nemotron 3 Super 120B-A12B",
    "meta/muse-glimmer-30b": "Muse Glimmer 30B",
}
CLASS_LABEL = {
    "taxonomy_mismatch": "taxonomy mismatch",
    "label_noise": "label noise",
    "llm_reasoning_error": "LLM reasoning error",
    "no_web_presence": "no web presence",
    "web_search_failure": "web search failure",
    "ambiguous_merchant": "ambiguous merchant",
    "normalization_failure": "normalization failure",
}


def esc(s: str) -> str:
    """Escape the LaTeX specials that occur in merchant strings and model ids."""
    for a, b in (("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def f3(x: float | None) -> str:
    return "--" if x is None or pd.isna(x) else f"{float(x):.3f}"


def pct(x: float, nd: int = 1) -> str:
    return "--" if pd.isna(x) else f"{100 * float(x):.{nd}f}\\%"


def table(body: str, spec: str, header: str, caption: str, label: str, note: str = "") -> str:
    notes = f"\n\\vspace{{2pt}}\n\\footnotesize {note}\n" if note else "\n"
    return (
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header}\\\\\n\\midrule\n{body}\n"
        f"\\bottomrule\n\\end{{tabular}}{notes}\\end{{table}}\n"
    )


def datasets() -> str:
    d = pd.read_csv(TAB / "tab0_dataset_stats.csv").set_index("dataset").loc["dc"]
    cov = pd.read_csv(TAB / "tab0_taxonomy_coverage.csv").set_index("dataset")
    noise = pd.read_csv(TAB / "tab0_label_noise.csv").iloc[0]
    rows = [
        ("Training window", f"{d.train_start} to {d.train_end}"),
        ("Test window", f"{d.test_start} onward"),
        (
            "Training transactions / merchants",
            f"{int(d.n_train_txns):,} / {int(d.n_train_merchants):,}",
        ),
        ("Test transactions / merchants", f"{int(d.n_test_txns):,} / {int(d.n_test_merchants):,}"),
        ("Tail merchants (train freq $\\leq 3$)", f"{int(d.tail_le3_merchants):,}"),
        ("Tail share of test transactions", pct(d.tail_le3_txn_share)),
        (
            "\\quad sensitivity: freq $\\leq 2$ / $\\leq 5$",
            f"{pct(d.tail_le2_txn_share)} / {pct(d.tail_le5_txn_share)}",
        ),
        ("Test transactions from never-seen merchants", pct(d.unseen_txn_share)),
        ("Rows without a usable MCC", pct(cov.loc["dc", "share_unlabeled"], 2)),
        ("Rows with an ambiguous MCC (excluded)", pct(cov.loc["dc", "share_ambiguous"])),
        (
            "Label noise (300-row audit)",
            f"{pct(noise.noise_rate_excl_unclear)} wrong, "
            f"{pct(noise.noise_rate_upper_bound)} upper bound",
        ),
    ]
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows)
    return table(
        body,
        "lr",
        "\\textbf{Washington DC Purchase Card} & \\textbf{Value}",
        "Primary dataset and temporal split. The tail is a quarter of test volume, and one test "
        "transaction in five comes from a merchant the index has never seen.",
        "datasets",
    )


def acc_by_freq() -> str:
    a = pd.read_csv(TAB / "tab1_acc_by_freq.csv")
    label = {"0": "0 (unseen)"}
    body = "\n".join(
        f"{label.get(str(r.bin), str(r.bin))} & {int(r.n):,} & {f3(r.accuracy)} & "
        f"[{f3(r.lo)}, {f3(r.hi)}] \\\\"
        for r in a.itertuples()
    )
    return table(
        body,
        "lrrc",
        "Training frequency & Test txns & Top-1 accuracy & 95\\% CI",
        "kNN accuracy by training frequency (MiniLM-L6-v2, full DC test set). The cliff is at "
        "zero: "
        "a merchant seen once is classified as well as a merchant seen a thousand times.",
        "accfreq",
    )


def main_results() -> str:
    m = pd.read_csv(TAB / "tab2_main_results.csv").set_index("method")

    def usd(x: float) -> str:
        if x <= 0:
            return "--"  # local CPU only
        return f"{x:.2f}" if x >= 0.01 else "$<$0.01"

    def row(key: str, name: str) -> str:
        r = m.loc[key]
        fb = "--" if r.fallback_rate == 0 else pct(r.fallback_rate, 1)
        return (
            f"{name} & {f3(r.f1_overall)} & {f3(r.f1_head)} & {f3(r.f1_tail)} & "
            f"{usd(r.cost_per_1k)} & {fb} \\\\"
        )

    blocks = [
        ("\\textit{Retrieval only}", []),
        (
            None,
            [
                row("knn/sentence-transformers/all-MiniLM-L6-v2", "kNN, MiniLM-L6-v2"),
                row("knn/BAAI/bge-small-en-v1.5", "kNN, bge-small-en-v1.5"),
                row("knn/text-embedding-3-small", "kNN, text-embedding-3-small"),
                row(
                    "emb_lr/sentence-transformers/all-MiniLM-L6-v2",
                    "Logistic regression on MiniLM embeddings",
                ),
            ],
        ),
        ("\\textit{LLM only, every merchant}", []),
        (
            None,
            [
                row("llm_no_web/gpt-5-nano", "gpt-5-nano, no web"),
                row("llm_with_web/gpt-5-nano", "gpt-5-nano, web evidence"),
                row("llm_no_web/gpt-5-mini", "gpt-5-mini, no web"),
                row("llm_with_web/gpt-5-mini", "gpt-5-mini, web evidence"),
            ],
        ),
        ("\\textit{Routed: kNN, similarity gate at 0.65, LLM with web evidence}", []),
        (
            None,
            [
                row(
                    "routed/sentence-transformers/all-MiniLM-L6-v2/gpt-5-nano@0.65",
                    "Routed, gpt-5-nano",
                ),
                row(
                    "routed/sentence-transformers/all-MiniLM-L6-v2/gpt-5-mini@0.65",
                    "\\textbf{Routed, gpt-5-mini}",
                ),
            ],
        ),
    ]
    parts = []
    for headline, rows in blocks:
        if headline:
            parts.append(f"\\multicolumn{{6}}{{l}}{{{headline}}} \\\\")
        parts.extend(rows)
    body = "\n".join(parts)
    return table(
        body,
        "lrrrrr",
        "Method & Overall & Head & Tail & USD/1k & Fallback",
        "Main comparison on the 3,000-merchant fallback evaluation set (2,500 tail, 500 head), "
        "macro-F1, five seeds. Cost is US dollars per 1,000 transactions at list prices; fallback "
        "is the share of transactions sent to the LLM.",
        "main",
        "Open-weight models are evaluated on a 300-merchant tail subset and appear in "
        "Table~\\ref{tab:ablation}. The routed system is the only configuration that keeps head "
        "accuracy while moving the tail.",
    )


def ablation() -> str:
    a = pd.read_csv(TAB / "tab3_critical_ablation.csv").sort_values("diff", ascending=False)
    body = "\n".join(
        f"{esc(MODEL_SHORT.get(r.model, r.model))} & {int(r.n_tail_merchants):,} & "
        f"{f3(r.acc_no_web)} & {f3(r.acc_with_web)} & \\textbf{{{r.diff:+.3f}}} & "
        f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] \\\\"
        for r in a.itertuples()
    )
    return table(
        body,
        "lrrrrc",
        "Fallback model & Merchants & No web & With web & Difference & 95\\% CI",
        "The critical ablation: identical prompts, merchants and taxonomy, with and without web "
        "search evidence. Paired bootstrap over merchants; every interval excludes zero.",
        "ablation",
        "The three open-weight models run on NVIDIA's free NIM endpoint and are evaluated on the "
        "shared 300-merchant tail subset, which is why their intervals are wider.",
    )


def coldstart() -> str:
    c = pd.read_csv(TAB / "tab2b_oklahoma_coldstart.csv")
    name = {
        "knn/sentence-transformers/all-MiniLM-L6-v2": "kNN on the DC index",
        "llm_no_web/gpt-5-nano": "gpt-5-nano, no web",
        "llm_with_web/gpt-5-nano": "gpt-5-nano, web evidence",
        "llm_no_web/gpt-5-mini": "gpt-5-mini, no web",
        "llm_with_web/gpt-5-mini": "gpt-5-mini, web evidence",
    }
    body = "\n".join(
        f"{name.get(r.method, r.method)} & {f3(r.f1_overall)} \\\\" for r in c.itertuples()
    )
    return table(
        body,
        "lr",
        "Method & Macro-F1",
        "Cross-dataset cold start: 800 Oklahoma merchants that never appear in the DC training "
        "index, classified with the DC index and taxonomy.",
        "coldstart",
    )


def writeback() -> str:
    w = pd.read_csv(TAB / "tab5_writeback_policies.csv")
    order = ["never", "gated@0.8", "gated@0.7", "gated@0.6", "gated@0.5", "always"]
    w = w.set_index("policy").loc[[p for p in order if p in set(w.policy)]].reset_index()
    body = "\n".join(
        f"{esc(r.policy)} & {pct(r.fallback_rate, 2)} & {f3(r.macro_f1)} & {r.total_cost:.2f} & "
        f"{int(round(r.n_writebacks)):,} & {pct(r.writeback_error_rate)} & "
        f"{int(round(r.final_index_size)):,} \\\\"
        for r in w.itertuples()
    )
    return table(
        body,
        "lrrrrrr",
        "Write-back policy & Fallback & Macro-F1 & Total USD & Writes & Wrong writes & Index size",
        "Streaming write-back over 70,333 test transactions in arrival order (141 windows, five "
        "seeds). Writing every resolved merchant back into the index halves the fallback bill and "
        "costs 0.3 points of macro-F1, but nearly half of the written labels are wrong.",
        "writeback",
    )


def backbones() -> str:
    b = pd.read_csv(TAB / "tab4_backbone_comparison.csv")
    body = "\n".join(
        f"{esc(BACKBONE_SHORT.get(r.backbone, r.backbone))} & {int(r.dim)} & "
        f"{r.embed_ms_per_merchant:.2f} & {f3(r.f1_overall)} & {f3(r.f1_head)} & {f3(r.f1_tail)} & "
        f"{f3(r.acc_tail)} \\\\"
        for r in b.itertuples()
    )
    return table(
        body,
        "lrrrrrr",
        "Backbone & Dim & ms/merchant & Overall & Head & Tail & Tail acc.",
        "Embedding backbones under the same index, gate and taxonomy, full DC test set, five "
        "seeds. Twenty-three times the embedding latency buys one point of tail macro-F1.",
        "backbones",
    )


def failures() -> str:
    f = pd.read_csv(TAB / "tab6_failure_taxonomy.csv").sort_values("share", ascending=False)
    body = "\n".join(
        f"{CLASS_LABEL.get(r.failure_class, r.failure_class)} & {int(r.count)} & {pct(r.share)} & "
        f"\\texttt{{{esc(str(r.example))}}} \\\\"
        for r in f.itertuples()
    )
    return table(
        body,
        "lrrl",
        "Failure class & Count & Share & Example merchant",
        "Manual taxonomy of 200 residual tail errors made by the routed system with web evidence. "
        "Two thirds are annotation or taxonomy artifacts rather than model failures.",
        "failures",
    )


def novel_share() -> str:
    n = pd.read_csv(TAB / "tab1c_novel_share_sweep.csv")
    body = "\n".join(
        f"{r.novel_share:.2f} & {pct(r.unseen_entity_txn_share_mean)} & {f3(r.acc_overall_mean)} & "
        f"{f3(r.acc_tail_mean)} & {f3(r.acc_unseen_entity_mean)} & "
        f"{f3(r.acc_seen_entity_mean)} \\\\"
        for r in n.itertuples()
    )
    return table(
        body,
        "rrrrrr",
        "Novel brand share & Unseen txns & Overall & Tail & Unseen entities & Seen entities",
        "Generator sweep over the share of brands that appear only after the split date, with the "
        "Zipf exponent held at 1.1 (five seeds). Accuracy on genuinely unseen entities is pinned "
        "near 0.5 regardless of the dial; only their share of traffic changes.",
        "novelshare",
    )


def extras() -> str:
    a = pd.read_csv(TAB / "tab3c_agentic_search.csv").iloc[0]
    main_t = pd.read_csv(TAB / "tab2_main_results.csv").set_index("method")
    fixed_cost = float(main_t.loc["llm_with_web/gpt-5-mini", "cost_per_1k"])
    p = pd.read_csv(TAB / "tab3b_prompt_sensitivity.csv")
    rows = [
        ("No web evidence", f3(a.acc_no_web), "--", "--"),
        (
            "Fixed web snippets (this paper)",
            f3(a.acc_fixed_snippets),
            "1.00",
            f"{fixed_cost:.2f}",
        ),
        (
            "Agent-issued searches",
            f3(a.acc_agentic_search),
            f"{a.mean_searches_per_merchant:.2f}",
            f"{a.usd_per_1k_merchants:.2f}",
        ),
    ]
    body = "\n".join(f"{n} & {acc} & {s} & {c} \\\\" for n, acc, s, c in rows)
    spread = ", ".join(
        f"{esc(r.model)} {r.condition.replace('_', ' ')} {r.abs_diff * 100:.1f} pts"
        for r in p.itertuples()
    )
    return table(
        body,
        "lrrr",
        "Evidence strategy & Accuracy & Searches/merchant & USD/1k",
        "Letting the model run its own searches instead of reading five fixed snippets "
        "(gpt-5-mini, the shared 300-merchant tail subset).",
        "agentic",
        f"Prompt sensitivity on the same subset, two independently written prompt templates: "
        f"{spread}. Prompt wording moves accuracy by a few points; web evidence moves it by ten to "
        f"seventeen.",
    )


def spend() -> str:
    led = json.loads((ROOT / "cache" / "spend_ledger.json").read_text())
    by: dict[str, float] = {}
    for e in led:
        by[e["kind"]] = by.get(e["kind"], 0.0) + e["usd"]
    n_search = sum(1 for e in led if e["kind"] == "search")
    n_llm = sum(1 for e in led if e["kind"] in ("llm", "agentic_search"))
    rows = [
        ("OpenAI chat completions (cache pass, all experiments)", f"{by.get('llm', 0):.2f}"),
        (
            "OpenAI built-in web search ablation (300 merchants)",
            f"{by.get('agentic_search', 0):.2f}",
        ),
        ("Firecrawl search (prepaid credits, reference price)", f"{by.get('search', 0):.2f}"),
        ("NVIDIA NIM open-weight models (free endpoint)", "0.00"),
        (
            "\\textbf{Total cash spent (OpenAI)}",
            f"\\textbf{{{by.get('llm', 0) + by.get('agentic_search', 0):.2f}}}",
        ),
    ]
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows)
    return table(
        body,
        "lr",
        "Item & USD",
        "Total API spend for every experiment in this paper. "
        f"The frozen caches hold {n_llm:,} LLM calls and {n_search:,} search calls; "
        "reproduce.py replays them and issues no requests.",
        "spend",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("datasets", datasets),
        ("accfreq", acc_by_freq),
        ("main", main_results),
        ("ablation", ablation),
        ("coldstart", coldstart),
        ("writeback", writeback),
        ("backbones", backbones),
        ("failures", failures),
        ("novelshare", novel_share),
        ("agentic", extras),
        ("spend", spend),
    ):
        (OUT / f"{name}.tex").write_text(fn())
        print(f"wrote paper/tables/{name}.tex")


if __name__ == "__main__":
    main()
