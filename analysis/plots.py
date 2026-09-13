"""Figure functions for Exp 1 (fig1–fig3). Each takes tidy data and a path stem (no extension)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style

style.apply()


def fig_zipf(
    freqs: pd.Series, stem: str | Path, title: str = "Merchant frequency (training window)"
) -> None:
    """Rank–frequency plot on log–log axes with a fitted slope in the legend."""
    f = np.sort(np.asarray(freqs, dtype=float))[::-1]
    rank = np.arange(1, len(f) + 1)
    slope, intercept = np.polyfit(np.log(rank), np.log(f), 1)
    fig, ax = plt.subplots()
    ax.loglog(rank, f, color=style.SERIES["head"], lw=1.6, label="merchants")
    ax.loglog(
        rank,
        np.exp(intercept) * rank**slope,
        color=style.INK2,
        lw=1,
        ls="--",
        label=f"power-law fit, slope {slope:.2f}",
    )
    ax.set_xlabel("merchant rank")
    ax.set_ylabel("transactions")
    ax.set_title(title, loc="left", color=style.INK)
    ax.legend(loc="upper right")
    style.save(fig, stem)
    plt.close(fig)


def fig_acc_by_freq(
    acc: pd.DataFrame, stem: str | Path, title: str = "kNN top-1 accuracy by training frequency"
) -> None:
    """Bars with 95% CI whiskers.

    ``acc`` has columns bin, accuracy, n, lo, hi, in display order.
    """
    fig, ax = plt.subplots()
    x = np.arange(len(acc))
    colors = [style.SERIES["tail"] if b in ("0", "1") else style.SERIES["head"] for b in acc["bin"]]
    ax.bar(x, acc["accuracy"], width=0.62, color=colors, edgecolor="white", linewidth=1)
    ax.errorbar(
        x,
        acc["accuracy"],
        yerr=[acc["accuracy"] - acc["lo"], acc["hi"] - acc["accuracy"]],
        fmt="none",
        ecolor=style.INK2,
        elinewidth=1,
        capsize=2,
    )
    for xi, (a, n) in enumerate(zip(acc["accuracy"], acc["n"], strict=True)):
        ax.text(
            xi,
            min(a + 0.04, 0.97),
            f"n={n:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=style.INK2,
        )
    ax.set_xticks(x, acc["bin"])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("merchant frequency in training index")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(title, loc="left", color=style.INK)
    ax.plot([], [], color=style.SERIES["tail"], lw=6, label="unseen or singleton (freq ≤ 1)")
    ax.plot([], [], color=style.SERIES["head"], lw=6, label="freq ≥ 2")
    ax.legend(loc="lower right")
    style.save(fig, stem)
    plt.close(fig)


def fig_acc_vs_sim(
    df: pd.DataFrame, stem: str | Path, nbins: int = 12, title: str = "Accuracy vs top-1 similarity"
) -> None:
    """Binned accuracy against sim1, head and tail as two lines.

    ``df`` has columns sim1, correct, is_tail.
    """
    edges = np.linspace(df["sim1"].min(), 1.0, nbins + 1)
    fig, ax = plt.subplots()
    for name, mask, color in (
        ("head", ~df["is_tail"], style.SERIES["head"]),
        ("tail (freq ≤ 3)", df["is_tail"], style.SERIES["tail"]),
    ):
        sub = df[mask]
        if sub.empty:
            continue
        b = np.clip(np.digitize(sub["sim1"], edges) - 1, 0, nbins - 1)
        g = sub.groupby(b)["correct"].agg(["mean", "size"])
        g = g[g["size"] >= 50]  # thin bins (rare low-similarity head merchants) add only noise
        centers = (edges[g.index] + edges[g.index + 1]) / 2
        ax.plot(centers, g["mean"], marker="o", color=color, label=name)
    ax.set_xlabel("top-1 cosine similarity")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left", color=style.INK)
    ax.legend(loc="lower right")
    style.save(fig, stem)
    plt.close(fig)
