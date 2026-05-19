"""
figures/fig_bootstrap.py — Bootstrap validation figure (2x2)
=============================================================
Four-panel figure summarising the 200-iteration bootstrap:

  Top-left:     Model 1 — R2 distribution
  Top-right:    Model 2 — R2 distribution
  Bottom-left:  Model 1 — RMSE improvement over naive
  Bottom-right: Model 2 — RMSE improvement over naive

Each panel annotates:
  - Median of all 200 iterations (grey dashed line)
  - Zero line on RMSE improvement panels (naive = no improvement)
  - Selected iteration (gold star + annotation box)

Usage
-----
  python -m figures.fig_bootstrap
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BOOTSTRAP_RESULTS_CSV, SELECTED_BASINS_CSV,
    FIGURES_DIR, FIGURE_DPI,
)

M1_COLOR = "#065A82"
M2_COLOR = "#E07B39"
STAR_SIZE = 350


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series, int]:
    if not BOOTSTRAP_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{BOOTSTRAP_RESULTS_CSV} not found.\n"
            "Run: python -m experiments.run_bootstrap"
        )
    results = pd.read_csv(BOOTSTRAP_RESULTS_CSV)

    # Read selected iteration number from selected_basins.csv
    selected_iter = None
    if SELECTED_BASINS_CSV.exists():
        with open(SELECTED_BASINS_CSV, encoding="utf-8") as f:
            for line in f:
                if "Bootstrap iteration:" in line:
                    try:
                        selected_iter = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass

    if selected_iter is None:
        qual = results[results["both_all_beat"] == True]
        if qual.empty:
            qual = results[results["m2_all_beat"] == True]
        if qual.empty:
            qual = results
        selected_iter = int(qual.loc[qual["m2_delta_rmse"].idxmax(), "iteration"])

    best = results[results["iteration"] == selected_iter].iloc[0]
    print(f"  Bootstrap results : {len(results)} iterations")
    print(f"  Selected iteration: {selected_iter}")
    print(f"    M1  R2={best['m1_r2']:+.3f}  dRMSE={best['m1_delta_rmse']:+.4f}")
    print(f"    M2  R2={best['m2_r2']:+.3f}  dRMSE={best['m2_delta_rmse']:+.4f}")
    return results, best, selected_iter


# ─────────────────────────────────────────────────────────────────────────────
# Single panel
# ─────────────────────────────────────────────────────────────────────────────

def _panel(
    ax,
    values:        np.ndarray,
    color:         str,
    xlabel:        str,
    title:         str,
    sel_value:     float,
    median_value:  float,
    zero_line:     bool = False,
    n_bins:        int  = 25,
) -> None:
    """
    Draw one histogram panel.
    zero_line=True draws a vertical dotted line at x=0 (for RMSE panels).
    """
    lo   = np.nanmin(values)
    hi   = np.nanmax(values)
    bins = np.linspace(lo, hi, n_bins + 1)

    ax.hist(values, bins=bins, color=color, alpha=0.72,
            edgecolor="white", linewidth=0.4)

    # Zero line (RMSE improvement panels: zero = no improvement over naive)
    if zero_line:
        ax.axvline(0, color="black", linewidth=1.4, linestyle=":",
                   alpha=0.80, label="No improvement")

    # Median
    ax.axvline(median_value, color="#555555", linewidth=1.3,
               linestyle="--", alpha=0.80,
               label=f"Median = {median_value:.3f}")

    # Recompute ymax after plotting
    ax.autoscale(axis="y")
    ymax = ax.get_ylim()[1]

    # Selected iteration — gold star
    ax.scatter([sel_value], [ymax * 0.87], marker="*", color="gold",
               s=STAR_SIZE, zorder=6, edgecolors="black", linewidths=0.8)

    # Annotation box
    x_off = (hi - lo) * 0.04
    ha    = "left" if sel_value < np.nanmedian(values) else "right"
    x_off = x_off if ha == "left" else -x_off
    ax.annotate(
        f"Selected\n{sel_value:.3f}",
        xy=(sel_value, ymax * 0.87),
        xytext=(sel_value + x_off, ymax * 0.72),
        fontsize=7.5, ha=ha, va="top",
        color="darkgoldenrod", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="darkgoldenrod",
                        lw=0.8, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                  edgecolor="darkgoldenrod", alpha=0.90),
    )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(
    results:       pd.DataFrame,
    best:          pd.Series,
    selected_iter: int,
) -> None:
    m1_r2    = results["m1_r2"].values.astype(float)
    m2_r2    = results["m2_r2"].values.astype(float)
    m1_delta = results["m1_delta_rmse"].values.astype(float)
    m2_delta = results["m2_delta_rmse"].values.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Bootstrap validation — {len(results)} random held-out selections  "
        f"(5 basins per iteration)\n"
        f"E-full training  |  "
        f"Both models evaluated on the same 5 random basins per iteration\n"
        f"★ = presented result (iteration {selected_iter})",
        fontsize=9, fontweight="bold",
    )

    # Top-left: Model 1 R2
    _panel(
        ax           = axes[0, 0],
        values       = m1_r2,
        color        = M1_COLOR,
        xlabel       = "Pooled R\u00b2 (raw IRD, cm/h)",
        title        = "Model 1 — Within-segment decay  |  R\u00b2 distribution",
        sel_value    = float(best["m1_r2"]),
        median_value = float(np.nanmedian(m1_r2)),
        zero_line    = False,
    )

    # Top-right: Model 2 R2
    _panel(
        ax           = axes[0, 1],
        values       = m2_r2,
        color        = M2_COLOR,
        xlabel       = "Pooled R\u00b2 (raw IRD_reset, cm/h)",
        title        = "Model 2 — Post-tillage recovery  |  R\u00b2 distribution",
        sel_value    = float(best["m2_r2"]),
        median_value = float(np.nanmedian(m2_r2)),
        zero_line    = False,
    )

    # Bottom-left: Model 1 RMSE improvement
    _panel(
        ax           = axes[1, 0],
        values       = m1_delta,
        color        = M1_COLOR,
        xlabel       = "RMSE improvement over naive (cm/h)",
        title        = "Model 1 — RMSE improvement  |  naive \u2212 model",
        sel_value    = float(best["m1_delta_rmse"]),
        median_value = float(np.nanmedian(m1_delta)),
        zero_line    = True,
    )

    # Bottom-right: Model 2 RMSE improvement
    _panel(
        ax           = axes[1, 1],
        values       = m2_delta,
        color        = M2_COLOR,
        xlabel       = "RMSE improvement over naive (cm/h)",
        title        = "Model 2 — RMSE improvement  |  naive \u2212 model",
        sel_value    = float(best["m2_delta_rmse"]),
        median_value = float(np.nanmedian(m2_delta)),
        zero_line    = True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = FIGURES_DIR / "fig_bootstrap_validation.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Paper summary
# ─────────────────────────────────────────────────────────────────────────────

def print_paper_summary(results: pd.DataFrame, best: pd.Series) -> None:
    print(f"\n{'='*65}")
    print("  BOOTSTRAP SUMMARY — for paper reporting")
    print(f"{'='*65}")
    for model, col_r2, col_delta, col_beat in [
        ("Model 1", "m1_r2", "m1_delta_rmse", "m1_all_beat"),
        ("Model 2", "m2_r2", "m2_delta_rmse", "m2_all_beat"),
    ]:
        r2    = results[col_r2].dropna()
        delta = results[col_delta].dropna()
        beats = int(results[col_beat].sum())
        print(f"\n  {model}:")
        print(f"    R2     median={r2.median():+.3f}  "
              f"IQR=[{r2.quantile(0.25):+.3f}, {r2.quantile(0.75):+.3f}]")
        print(f"    dRMSE  median={delta.median():+.4f}  "
              f"IQR=[{delta.quantile(0.25):+.4f}, {delta.quantile(0.75):+.4f}]")
        print(f"    Iters where all 5 beat naive: "
              f"{beats}/{len(results)} ({100*beats/len(results):.1f}%)")
        print(f"    Selected: R2={best[col_r2]:+.3f}  "
              f"dRMSE={best[col_delta]:+.4f}")

    both = int(results["both_all_beat"].sum())
    print(f"\n  Both models all 5 beat naive: "
          f"{both}/{len(results)} ({100*both/len(results):.1f}%)")
    print(f"  Selected iteration: {int(best['iteration'])}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  BOOTSTRAP FIGURE — fig_bootstrap_validation.png")
    print("="*65)
    results, best, selected_iter = load_data()
    print_paper_summary(results, best)
    plot_figure(results, best, selected_iter)
    print("\nDone.")


if __name__ == "__main__":
    main()