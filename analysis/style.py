"""Paper figure style. Palette validated for adjacent-pair CVD separation (ΔE ≥ 8)."""

import matplotlib as mpl

SERIES = {
    "head": "#2a78d6",
    "tail": "#eb6834",
    "third": "#1baf7a",
    "fourth": "#eda100",
    "fifth": "#e87ba4",
}
BACKBONE_ORDER = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "ProsusAI/finbert",
    "text-embedding-3-small",
]
BACKBONE_COLORS = dict(
    zip(BACKBONE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], strict=True)
)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (5.2, 3.4),
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def save(fig, stem) -> None:
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
