"""
figures/fig6_model2_timeseries.py — Figure 6: Model 2 held-out time series + scatter
======================================================================================
Condition E (E-full): trained on non-held-out basins (disjoint train/val),
tested on 5 completely unseen held-out basins.

Imports and calls model2_reset.{load_data, prepare_splits, train_model}, so its
numbers are GUARANTEED identical to `python -m models.model2_reset`. Per-basin
metrics are NOT drawn on the figure — they live in Table 1. The figure shows
only the IRD_reset time series (col 0) and the actual-vs-predicted scatter
(col 1, Model 2 vs naive).

Held-out basins are read at runtime from config.SELECTED_BASINS_CSV.

Usage
-----
  python -m figures.fig6_model2_timeseries
  Save manually as PNG (300 DPI) + TIFF.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT, FIELD_COLORS, add_season_bands
from config import FIELD_NAMES
from pipeline.features import MODEL2_FEATURES, TARGET_M2
from models.model2_reset import (
    load_selected_basins, load_data, prepare_splits, train_model,
    _metrics, _back,
)

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  TUNING PANEL  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════
LAYOUT = dict(
    fig_width   = 12,
    row_height  = 3.2,
    col_ratios  = [1.0, 1.0],     # time series : scatter (metric column removed)
    hspace      = 0.38,
    wspace      = 0.22,
    scatter_pad = 0.08,
    fs_title    = 16,
    fs_ylabel   = 16,
    fs_xlabel   = 16,
    fs_tick     = 12,
    fs_legend   = 9,
    ts_s        = 35,
    ts_alpha    = 0.80,
    sc_s_model  = 30,
    sc_s_naive  = 24,
    sc_alpha    = 0.75,
    season_alpha = 0.8,
    connector_lw    = 0.5,
    connector_alpha = 0.30,
)
# ▲▲▲  end of tuning panel  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════════

L = LAYOUT


# ─────────────────────────────────────────────────────────────────────────────
# Predict on held-out test set + per-basin metrics (caption / console only)
# ─────────────────────────────────────────────────────────────────────────────

def predict_held_out(model, sc, feat_cols, test, held_out):
    Xte               = sc.transform(test[feat_cols].values)
    prev_ird          = test["prev_IRD_at_reset_raw"].values.astype(float)
    test              = test.copy()
    test["ird_pred"]  = _back(model.predict(Xte), prev_ird)
    test["ird_true"]  = test["IRD_at_reset"].values.astype(float)
    test["ird_naive"] = prev_ird

    print("\n  Held-out basin metrics (Model 2, Condition E):")
    print(f"  {'Basin':<8} {'Field':<12} {'n':>5} "
          f"{'R²':>8} {'RMSE':>8} {'MAPE%':>9} {'Nv RMSE':>8}  Beat?")
    print(f"  {'-'*68}")

    basin_metrics: dict = {}
    all_t, all_p, all_n = [], [], []
    for bn in held_out:
        bdf   = test[test["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        m2    = _metrics(ird_t, ird_p, verbose=False) or {}
        mn    = _metrics(ird_t, ird_n, verbose=False) or {}
        beat  = "✓" if m2.get("rmse", np.inf) < mn.get("rmse", np.inf) else "✗"
        basin_metrics[bn] = {
            **m2, "field": field,
            "rmse_naive": mn.get("rmse", np.nan),
            "r2_naive":   mn.get("r2",   np.nan),
        }
        all_t.append(ird_t); all_p.append(ird_p); all_n.append(ird_n)
        print(f"  {bn:<8} {field:<12} {m2.get('n', 0):>5} "
              f"{m2.get('r2', np.nan):>+8.3f} {m2.get('rmse', np.nan):>8.3f} "
              f"{m2.get('mape', np.nan):>8.1f}% {mn.get('rmse', np.nan):>8.3f}  {beat}")

    ird_t_all = np.concatenate(all_t)
    ird_p_all = np.concatenate(all_p)
    ird_n_all = np.concatenate(all_n)
    m_all   = _metrics(ird_t_all, ird_p_all, verbose=False) or {}
    m_naive = _metrics(ird_t_all, ird_n_all, verbose=False) or {}
    delta   = m_naive.get("rmse", np.nan) - m_all.get("rmse", np.nan)
    basin_metrics["POOLED"] = {
        **m_all, "field": "all",
        "rmse_naive": m_naive.get("rmse", np.nan),
        "r2_naive":   m_naive.get("r2",   np.nan),
        "delta":      delta,
    }
    print(f"  {'POOLED':<8} {'all':<12} {m_all.get('n', 0):>5} "
          f"{m_all.get('r2', np.nan):>+8.3f} {m_all.get('rmse', np.nan):>8.3f} "
          f"{m_all.get('mape', np.nan):>8.1f}% {m_naive.get('rmse', np.nan):>8.3f}")
    print(f"\n  RMSE improvement over naive: {delta:.3f} cm/h "
          f"({100 * delta / m_naive.get('rmse', 1):.1f}%)")
    return test, basin_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plot — 2 columns (time series + scatter); metrics live in Table 1
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(test, basin_metrics, held_out):
    apply_style()
    n   = len(held_out)
    fig = plt.figure(figsize=(L["fig_width"], L["row_height"] * n))
    gs  = GridSpec(n, 2, figure=fig, width_ratios=L["col_ratios"],
                   hspace=L["hspace"], wspace=L["wspace"])

    for row_idx, bn in enumerate(held_out):
        ax_ts = fig.add_subplot(gs[row_idx, 0])
        ax_sc = fig.add_subplot(gs[row_idx, 1])

        bdf     = test[test["basin_number"] == bn].sort_values("reset_date")
        field   = FIELD_NAMES.get(int(str(bn)[0]), "")
        color   = FIELD_COLORS.get(field, COLORS["deep_blue"])
        is_last = (row_idx == n - 1)
        is_mid  = (row_idx == n // 2)

        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)

        # ── TIME SERIES ───────────────────────────────────────────────────────
        import plot_style as _ps
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, L["season_alpha"])
        add_season_bands(ax_ts, bdf["reset_date"].min(), bdf["reset_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        dates  = bdf.loc[mask, "reset_date"]
        yt_msk = ird_t[mask]; yp_msk = ird_p[mask]; yn_msk = ird_n[mask]
        ax_ts.plot(dates, yn_msk, color="lightgrey", linewidth=1.0,
                   linestyle="-", alpha=0.9, zorder=2)
        ax_ts.scatter(dates, yt_msk, s=L["ts_s"], alpha=L["ts_alpha"],
                      color=COLORS["deep_blue"], marker="o", zorder=4)
        ax_ts.scatter(dates, yp_msk, s=L["ts_s"], alpha=L["ts_alpha"],
                      color=COLORS["orange"], marker="x", linewidths=1.5, zorder=5)
        # for dt, yt_i, yp_i in zip(dates, yt_msk, yp_msk):
        #     ax_ts.plot([dt, dt], [yt_i, yp_i], color="gray",
        #                linewidth=L["connector_lw"], alpha=L["connector_alpha"], zorder=3)
        # ax_ts.plot(dates, yt_msk, color=COLORS["deep_blue"],
        #            linewidth=0.6, alpha=0.25, linestyle="-", zorder=3)

        if is_mid:
            ax_ts.set_ylabel(f"IRD_reset (cm/h)\n{bn}", fontsize=L["fs_ylabel"])
        else:
            ax_ts.set_ylabel(f"{bn}", fontsize=L["fs_ylabel"])
        ax_ts.tick_params(axis="both", labelsize=L["fs_tick"])
        ax_ts.tick_params(axis="x", rotation=25)
        if not is_last:
            ax_ts.set_xticklabels([])
        if is_last:
            ax_ts.set_xlabel("", fontsize=L["fs_xlabel"])
        if row_idx == 0:
            ax_ts.legend(
                handles=[
                    Line2D([0], [0], marker="o", color=COLORS["deep_blue"],
                           markersize=5, linestyle="None", label="Actual IRD_reset"),
                    Line2D([0], [0], marker="x", color=COLORS["orange"],
                           markersize=5, linestyle="None",
                           markeredgewidth=1.5, label="Predicted IRD_reset"),
                    Line2D([0], [0], color="lightgrey", linewidth=1.5,
                           label="Naive baseline"),
                ],
                fontsize=L["fs_legend"], loc="upper right", ncol=3, framealpha=0.85)
            ax_ts.set_title("IRD_reset Time Series",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

        # ── SCATTER (Model 2 vs naive) ────────────────────────────────────────
        yt = ird_t[mask]; yp = ird_p[mask]; yn = ird_n[mask]
        pad      = L["scatter_pad"]
        all_vals = np.concatenate([yt, yp, yn])
        lo, hi   = all_vals.min(), all_vals.max()
        rng      = hi - lo
        lim_lo   = lo - pad * rng; lim_hi = hi + pad * rng
        ax_sc.scatter(yt, yn, s=L["sc_s_naive"], alpha=0.35,
                      color="grey", marker="^", zorder=2)
        ax_sc.scatter(yt, yp, s=L["sc_s_model"], alpha=L["sc_alpha"],
                      color=color, edgecolors="none", zorder=3)
        ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                   color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
        ax_sc.set_xlim(lim_lo, lim_hi); ax_sc.set_ylim(lim_lo, lim_hi)
        # ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.tick_params(axis="both", labelsize=L["fs_tick"])
        if is_last:
            ax_sc.set_xlabel("Actual IRD_reset (cm/h)", fontsize=L["fs_xlabel"])
        if row_idx == 2:
            ax_sc.set_ylabel("Predicted (cm/h)", fontsize=L["fs_ylabel"])
        if row_idx == 0:
            ax_sc.legend(
                handles=[
                    Line2D([0], [0], marker="o", color=color, markersize=5,
                           linestyle="None", label="Model 2"),
                    Line2D([0], [0], marker="^", color="grey", markersize=5,
                           linestyle="None", label="Naive"),
                ],
                fontsize=L["fs_legend"] - 1, loc="lower right", framealpha=0.85)
            ax_sc.set_title("Actual vs Predicted",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Caption (clean) + text-routed numbers (separate block)
# ─────────────────────────────────────────────────────────────────────────────

def _print_caption_and_numbers(held_out, basin_metrics):
    pooled = basin_metrics.get("POOLED", {})
    p_rmse  = pooled.get("rmse", np.nan)
    p_naive = pooled.get("rmse_naive", np.nan)
    delta   = pooled.get("delta", np.nan)

    caption = (
        "Figure 6. Model 2 (post-tillage recovery) on five completely unseen "
        "held-out basins, one per field (Condition E: trained on all non-held-out "
        "basins, no site-specific recalibration). Left: measured (circles) and "
        "predicted (crosses) post-tillage IRD (IRD_reset, cm h⁻¹) at each tillage "
        "event; the grey line is the naive baseline (no change from the previous "
        "reset); background shading denotes season (as in Figure 4). Right: "
        "predicted versus measured IRD_reset (coloured circles), with the naive "
        "baseline (grey triangles) shown for comparison; dashed line is 1:1. "
        "Per-basin and pooled metrics are given in Table 1."
    )

    print("\n" + "=" * 70)
    print("ARTICLE CAPTION (Figure 6):")
    print("=" * 70)
    print(caption)

    print("\n" + "=" * 70)
    print("TEXT NUMBERS (not for the caption — route to §3.3 / Table 1):")
    print("=" * 70)
    print(f"  Pooled (n={pooled.get('n', 0):,}): "
          f"R²={pooled.get('r2', np.nan):+.3f}, RMSE={p_rmse:.3f} cm/h, "
          f"MAPE={pooled.get('mape', np.nan):.1f}%")
    print(f"  Naive pooled RMSE={p_naive:.3f} cm/h; "
          f"Model 2 ΔRMSE={delta:.3f} cm/h "
          f"({100*delta/p_naive:.1f}%) over naive")
    print("  Per-basin → Table 1 (already locked).")


def main():
    print("=" * 65)
    print("  FIGURE 6 — Model 2 | Time Series + Scatter  (Condition E)")
    print("  Reproduces models.model2_reset exactly (shared code path)")
    print("=" * 65)

    held_out  = load_selected_basins()
    df        = load_data()
    feat_cols = [f for f in MODEL2_FEATURES if f in df.columns]
    train, val, test = prepare_splits(df, held_out, feat_cols)
    model, sc        = train_model(train, val, feat_cols)

    test_df, basin_metrics = predict_held_out(model, sc, feat_cols, test, held_out)

    print("\n  Rendering figure...")
    plot_figure(test_df, basin_metrics, held_out)
    _print_caption_and_numbers(held_out, basin_metrics)


if __name__ == "__main__":
    main()