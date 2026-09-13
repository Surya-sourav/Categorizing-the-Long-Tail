import numpy as np
import pandas as pd
from analysis.plots import fig_acc_by_freq, fig_acc_vs_sim, fig_zipf


def test_plots_write_pdf_and_png(tmp_path):
    freqs = pd.Series(np.random.default_rng(0).zipf(1.2, 2000))
    fig_zipf(freqs, tmp_path / "fig1_zipf")
    acc = pd.DataFrame(
        {
            "bin": ["0", "1", "2-5", "6-20", "21-100", "100+"],
            "accuracy": [0.3, 0.5, 0.6, 0.8, 0.9, 0.95],
            "n": [100, 50, 80, 60, 40, 20],
            "lo": [0.25, 0.4, 0.55, 0.75, 0.85, 0.9],
            "hi": [0.35, 0.6, 0.65, 0.85, 0.95, 0.99],
        }
    )
    fig_acc_by_freq(acc, tmp_path / "fig2")
    df = pd.DataFrame(
        {
            "sim1": np.random.default_rng(1).uniform(0.2, 1, 500),
            "correct": np.random.default_rng(2).random(500) > 0.4,
            "is_tail": np.random.default_rng(3).random(500) > 0.5,
        }
    )
    fig_acc_vs_sim(df, tmp_path / "fig3")
    for stem in ("fig1_zipf", "fig2", "fig3"):
        assert (tmp_path / f"{stem}.pdf").exists() and (tmp_path / f"{stem}.png").exists()
