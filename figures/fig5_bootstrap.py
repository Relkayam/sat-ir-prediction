"""
figures/fig5_bootstrap.py — Figure 5: bootstrap validation (2×2)
=================================================================
Four-panel summary of the v2 bootstrap (run_bootstrap_v2.py):

  Top-left:     Model 1 — R² distribution
  Top-right:    Model 2 — R² distribution
  Bottom-left:  Model 1 — RMSE improvement over naive (naive − model)
  Bottom-right: Model 2 — RMSE improvement over naive

Reads the V2 outputs:
  data/bootstrap_results_v2.csv   (all iterations)
  data/selected_basins_v2.csv     (selected iteration, parsed from header)

Caption + text numbers are generated from the data — nothing hardcoded.

Usage
-----
  python -m figures.fig5_bootstrap
"""
from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FIGURES_DIR, FIGURE_DPI

# ── V2 bootstrap outputs (match run_bootstrap_v2.py) ──────────────────────────
DATA_DIR             = Path(__file__).parent.parent / "data"
BOOTSTRAP_RESULTS_V2 = DATA_DIR / "bootstrap_results_v2.csv"
SELECTED_BASINS_V2   = DATA_DIR / "selected_basins_v2.csv"

M1_COLOR  = "#065A82"
M2_COLOR  = "#E07B39"
STAR_SIZE = 350

# ── Fonts — aligned to plot_style.FONT (was hardcoded 8–9; now journal-scale) ──
FONTS = dict(
    axis_label = 12,
    title      = 12,
    tick       = 11,
    legend     = 10,
    annotation = 9,
)

REQUIRED_FIELDS = {3, 4, 5, 6, 7}


def _covers_all_fields(held_out_str: str) -> bool:
    basin_nums = [int(x) for x in re.findall(r'\b\d{4}\b', str(held_out_str))]
    return {int(str(bn)[0]) for bn in basin_nums} == REQUIRED_FIELDS


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series, int]:
    # if not BOOTSTRAP_RESULTS_V2.exists():
    #     raise FileNotFoundError(
    #         f"{BOOTSTRAP_RESULTS_V2} not found.\n"
    #         "Run: python -m experiments.run_bootstrap_v2"
    #     )
    results = pd.read_csv(BOOTSTRAP_RESULTS_V2)

    selected_iter = None
    if SELECTED_BASINS_V2.exists():
        with open(SELECTED_BASINS_V2, encoding="utf-8") as f:
            for line in f:
                if "Bootstrap draw:" in line or "Bootstrap iteration:" in line:
                    m = re.search(r"(?:draw|iteration):\s*(\d+)", line)
                    if m:
                        selected_iter = int(m.group(1))

    if selected_iter is None or selected_iter not in set(results["iteration"]):
        print("  (selected iteration not parsed from header — using "
              "median-closest fallback)")
        total_median = float(results["m2_delta_rmse"].median())
        qual = results[results["both_all_beat"] == True].copy()
        if qual.empty:
            qual = results.copy()
        fmask = qual["held_out"].apply(_covers_all_fields)
        if fmask.any():
            qual = qual[fmask]
        idx = (qual["m2_delta_rmse"] - total_median).abs().idxmin()
        selected_iter = int(qual.loc[idx, "iteration"])

    best = results[results["iteration"] == selected_iter].iloc[0]
    print(f"  Bootstrap results : {len(results)} random draws")
    print(f"  Selected iteration: {selected_iter}")
    print(f"    M1  R²={best['m1_r2']:+.3f}  ΔRMSE={best['m1_delta_rmse']:+.4f}")
    print(f"    M2  R²={best['m2_r2']:+.3f}  ΔRMSE={best['m2_delta_rmse']:+.4f}")
    return results, best, selected_iter


# ─────────────────────────────────────────────────────────────────────────────
# Single panel
# ─────────────────────────────────────────────────────────────────────────────

def _panel(ax, values, color, xlabel, title, sel_value, median_value,
           zero_line=False, n_bins=25) -> None:
    values = values[np.isfinite(values)]
    lo, hi = np.nanmin(values), np.nanmax(values)
    bins   = np.linspace(lo, hi, n_bins + 1)

    ax.hist(values, bins=bins, color=color, alpha=0.72,
            edgecolor="white", linewidth=0.4)

    if zero_line:
        ax.axvline(0, color="black", linewidth=1.4, linestyle=":",
                   alpha=0.80, label="No improvement")
    ax.axvline(median_value, color="#555555", linewidth=1.3,
               linestyle="--", alpha=0.80,
               label=f"Median = {median_value:.3f}")

    ax.autoscale(axis="y")
    ymax = ax.get_ylim()[1]
    ax.scatter([sel_value], [ymax * 0.87], marker="*", color="gold",
               s=STAR_SIZE, zorder=6, edgecolors="black", linewidths=0.8)

    x_off = (hi - lo) * 0.04
    ha    = "left" if sel_value < np.nanmedian(values) else "right"
    x_off = x_off if ha == "left" else -x_off
    ax.annotate(
        f"Selected\n{sel_value:.3f}",
        xy=(sel_value, ymax * 0.87),
        xytext=(sel_value + x_off, ymax * 0.72),
        fontsize=FONTS["annotation"], ha=ha, va="top",
        color="darkgoldenrod", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="darkgoldenrod", lw=0.8, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                  edgecolor="darkgoldenrod", alpha=0.90),
    )

    ax.set_xlabel(xlabel, fontsize=FONTS["axis_label"])
    ax.set_ylabel("Count", fontsize=FONTS["axis_label"])
    ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
    ax.legend(fontsize=FONTS["legend"], loc="upper left")
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=FONTS["tick"])


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(results: pd.DataFrame, best: pd.Series, selected_iter: int) -> None:
    m1_r2    = results["m1_r2"].values.astype(float)
    m2_r2    = results["m2_r2"].values.astype(float)
    m1_delta = results["m1_delta_rmse"].values.astype(float)
    m2_delta = results["m2_delta_rmse"].values.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    _panel(axes[0, 0], m1_r2, M1_COLOR,
           "Pooled R\u00b2 (raw IRD, cm/h)",
           "Model 1 — Within-segment decay  |  R\u00b2 distribution",
           float(best["m1_r2"]), float(np.nanmedian(m1_r2)), zero_line=False)
    _panel(axes[0, 1], m2_r2, M2_COLOR,
           "Pooled R\u00b2 (raw IRD_reset, cm/h)",
           "Model 2 — Post-tillage recovery  |  R\u00b2 distribution",
           float(best["m2_r2"]), float(np.nanmedian(m2_r2)), zero_line=False)
    _panel(axes[1, 0], m1_delta, M1_COLOR,
           "RMSE improvement over naive (cm/h)",
           "Model 1 — RMSE improvement  |  naive \u2212 model",
           float(best["m1_delta_rmse"]), float(np.nanmedian(m1_delta)), zero_line=True)
    _panel(axes[1, 1], m2_delta, M2_COLOR,
           "RMSE improvement over naive (cm/h)",
           "Model 2 — RMSE improvement  |  naive \u2212 model",
           float(best["m2_delta_rmse"]), float(np.nanmedian(m2_delta)), zero_line=True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = FIGURES_DIR / "fig5_bootstrap_validation.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Paper summary (text-routed numbers)
# ─────────────────────────────────────────────────────────────────────────────

def print_paper_summary(results: pd.DataFrame, best: pd.Series) -> None:
    print(f"\n{'='*70}")
    print("  TEXT NUMBERS — bootstrap summary (route to §2.5 / §3.2 / §3.3):")
    print(f"{'='*70}")
    for model, col_r2, col_delta, col_beat in [
        ("Model 1", "m1_r2", "m1_delta_rmse", "m1_all_beat"),
        ("Model 2", "m2_r2", "m2_delta_rmse", "m2_all_beat"),
    ]:
        r2    = results[col_r2].dropna()
        delta = results[col_delta].dropna()
        beats = int(results[col_beat].sum())
        print(f"\n  {model}:")
        print(f"    R²     median={r2.median():+.3f}  "
              f"IQR=[{r2.quantile(0.25):+.3f}, {r2.quantile(0.75):+.3f}]")
        print(f"    ΔRMSE  median={delta.median():+.4f}  "
              f"IQR=[{delta.quantile(0.25):+.4f}, {delta.quantile(0.75):+.4f}]")
        print(f"    Iters where all 5 beat naive: "
              f"{beats}/{len(results)} ({100*beats/len(results):.1f}%)")
        print(f"    Selected: R²={best[col_r2]:+.3f}  ΔRMSE={best[col_delta]:+.4f}")

    both = int(results["both_all_beat"].sum())
    print(f"\n  Both models all 5 beat naive: "
          f"{both}/{len(results)} ({100*both/len(results):.1f}%)")
    print(f"  Selected iteration: {int(best['iteration'])}")


# ─────────────────────────────────────────────────────────────────────────────
# Article caption (clean)
# ─────────────────────────────────────────────────────────────────────────────

def _build_caption(results: pd.DataFrame, best: pd.Series, selected_iter: int) -> str:
    n_draws = len(results)
    return (
        f"Figure 5. Bootstrap validation across {n_draws} random draws, each "
        f"holding out five basins (one per field). Histograms show the pooled R² "
        f"(top) and the RMSE improvement over the naive baseline (bottom; "
        f"RMSE_naive − RMSE_model, cm h⁻¹) for Model 1 (within-segment decay; "
        f"left) and Model 2 (post-tillage recovery; right). Dashed grey lines mark "
        f"the medians; the dotted black line marks zero improvement; the gold star "
        f"marks the reported draw (draw {selected_iter}), selected as the "
        f"qualifying draw closest to the median Model 2 ΔRMSE. Distribution "
        f"medians and IQRs are reported in Sections 3.2–3.3."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  FIGURE 5 — Bootstrap validation (reads v2 outputs)")
    print("=" * 65)
    results, best, selected_iter = load_data()
    plot_figure(results, best, selected_iter)
    print_paper_summary(results, best)

    caption = _build_caption(results, best, selected_iter)
    print("\n" + "=" * 70)
    print("ARTICLE CAPTION (Figure 5):")
    print("=" * 70)
    print(caption)
    print("\nDone.")


if __name__ == "__main__":
    main()