"""
figures/fig4_held_out_timeseries.py — Figure 4: Model 1 held-out time series + scatter
========================================================================================
Condition E (E-full): trained on non-held-out basins (disjoint train/val),
tested on 5 completely unseen held-out basins.

Imports and calls model1_decay.{load_data, prepare_splits, train_model} so its
numbers are GUARANTEED identical to `python -m models.model1_decay`. Per-basin
metrics are NOT drawn on the figure — they live in Table 1. The figure shows
only the time series (col 0) and the actual-vs-predicted scatter (col 1).

Held-out basins are read at runtime from config.SELECTED_BASINS_CSV.

Usage
-----
  python -m figures.fig4_held_out_timeseries
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
from pipeline.features import TARGET_M1
from models.utils import back_transform, metrics_ird, predict
from models.model1_decay import (
    load_selected_basins, load_data, prepare_splits, train_model,
    _linear_extrap_baseline,
)

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  TUNING PANEL  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════
LAYOUT = dict(
    fig_width   = 12,
    row_height  = 3.2,
    col_ratios  = [1.0, 1.0],     # time series : scatter (metric column removed)
    hspace      = 0.22,
    wspace      = 0.22,
    scatter_pad = 0.08,
    fs_title    = 11,
    fs_ylabel   = 10,
    fs_xlabel   = 10,
    fs_tick     = 9,
    fs_legend   = 9,
    ts_s        = 10,
    ts_alpha    = 0.70,
    sc_s        = 16,
    sc_alpha    = 0.65,
    season_alpha = 0.8,
    reset_lw    = 0.5,
    reset_alpha = 0.20,
    connector_lw    = 0.35,
    connector_alpha = 0.25,
)
# ▲▲▲  end of tuning panel  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════════

L = LAYOUT


# ─────────────────────────────────────────────────────────────────────────────
# Predict on held-out test set + per-basin metrics (for caption / console only)
# ─────────────────────────────────────────────────────────────────────────────

def predict_held_out(model, sc, feat_cols, test, held_out):
    ho = test.copy()
    ird_reset        = pd.to_numeric(ho["IRD_at_reset"], errors="coerce").values
    pred_norm        = predict(model, sc, feat_cols, ho)
    ho["ird_pred"]   = back_transform(ird_reset, pred_norm)
    ho["ird_true"]   = back_transform(
        ird_reset, pd.to_numeric(ho[TARGET_M1], errors="coerce").values)
    ho["ird_naive"]  = ird_reset
    ho["ird_linear"] = _linear_extrap_baseline(ho)

    print("\n  Held-out basin metrics (Model 1, Condition E — all events):")
    print(f"  {'Basin':<8} {'Field':<12} {'n':>6} {'R²':>8} {'RMSE':>8} "
          f"{'Naive RMSE':>11} {'Linear RMSE':>12} {'MAPE%':>7}")
    print(f"  {'-'*72}")

    basin_metrics: dict = {}
    all_true, all_pred = [], []
    for bn in held_out:
        bdf   = ho[ho["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        ird_l = bdf["ird_linear"].values.astype(float)
        mask   = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        mask_l = mask & np.isfinite(ird_l) & (ird_l > 0)
        all_true.append(ird_t[mask]); all_pred.append(ird_p[mask])
        m  = metrics_ird(ird_t[mask],   ird_p[mask],   verbose=False) or {}
        mn = metrics_ird(ird_t[mask],   ird_n[mask],   verbose=False) or {}
        ml = metrics_ird(ird_t[mask_l], ird_l[mask_l], verbose=False) or {}
        basin_metrics[bn] = {
            **m, "field": field,
            "rmse_naive":  mn.get("rmse", np.nan),
            "rmse_linear": ml.get("rmse", np.nan),
            "n_linear":    ml.get("n", 0),
        }
        print(f"  {bn:<8} {field:<12} {m.get('n', 0):>6} "
              f"{m.get('r2', np.nan):>+8.3f} {m.get('rmse', np.nan):>8.3f} "
              f"{mn.get('rmse', np.nan):>11.3f} {ml.get('rmse', np.nan):>12.3f} "
              f"{m.get('mape', np.nan):>6.1f}%")

    ird_t_all = np.concatenate(all_true); ird_p_all = np.concatenate(all_pred)
    m_all = metrics_ird(ird_t_all, ird_p_all, verbose=False) or {}
    basin_metrics["POOLED"] = {**m_all, "field": "all"}
    print(f"  {'POOLED':<8} {'all':<12} {m_all.get('n', 0):>6} "
          f"{m_all.get('r2', np.nan):>+8.3f} {m_all.get('rmse', np.nan):>8.3f} "
          f"{'':>11} {'':>12} {m_all.get('mape', np.nan):>6.1f}%")
    return ho, basin_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plot — 2 columns now (time series + scatter); metrics live in Table 1
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(ho, basin_metrics, held_out):
    apply_style()
    n   = len(held_out)
    fig = plt.figure(figsize=(L["fig_width"], L["row_height"] * n))
    gs  = GridSpec(n, 2, figure=fig, width_ratios=L["col_ratios"],
                   hspace=L["hspace"], wspace=L["wspace"])

    for row_idx, bn in enumerate(held_out):
        ax_ts = fig.add_subplot(gs[row_idx, 0])
        ax_sc = fig.add_subplot(gs[row_idx, 1])

        bdf     = ho[ho["basin_number"] == bn].sort_values("opening_valve_date")
        field   = FIELD_NAMES.get(int(str(bn)[0]), "")
        color   = FIELD_COLORS.get(field, COLORS["deep_blue"])
        is_last = (row_idx == n - 1)
        is_mid  = (row_idx == n // 2)

        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)

        # ── TIME SERIES ───────────────────────────────────────────────────────
        import plot_style as _ps
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, L["season_alpha"])
        add_season_bands(ax_ts, bdf["opening_valve_date"].min(),
                         bdf["opening_valve_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        for sid in sorted(bdf["segment_id"].dropna().unique()):
            seg = bdf[bdf["segment_id"] == sid].sort_values("opening_valve_date")
            if not seg.empty:
                ax_ts.axvline(seg["opening_valve_date"].iloc[0],
                              color="black", linewidth=L["reset_lw"],
                              linestyle="--", alpha=L["reset_alpha"], zorder=1)

        valid_t = bdf["ird_true"].notna() & (bdf["ird_true"] > 0)
        valid_p = bdf["ird_pred"].notna() & (bdf["ird_pred"] > 0)
        both    = valid_t & valid_p
        ax_ts.scatter(bdf.loc[valid_t, "opening_valve_date"], bdf.loc[valid_t, "ird_true"],
                      s=L["ts_s"], alpha=L["ts_alpha"], color=COLORS["deep_blue"],
                      marker="o", zorder=4)
        ax_ts.scatter(bdf.loc[valid_p, "opening_valve_date"], bdf.loc[valid_p, "ird_pred"],
                      s=L["ts_s"], alpha=L["ts_alpha"], color=COLORS["orange"],
                      marker="x", linewidths=1.2, zorder=5)
        for _, row in bdf[both].iterrows():
            ax_ts.plot([row["opening_valve_date"]] * 2,
                       [row["ird_true"], row["ird_pred"]],
                       color="gray", linewidth=L["connector_lw"],
                       alpha=L["connector_alpha"], zorder=2)

        if is_mid:
            ax_ts.set_ylabel(f"IRD (cm/h)\n{bn}", fontsize=L["fs_ylabel"])
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
                           markersize=5, linestyle="None", label="Actual IRD"),
                    Line2D([0], [0], marker="x", color=COLORS["orange"],
                           markersize=5, linestyle="None",
                           markeredgewidth=1.2, label="Predicted IRD"),
                    Line2D([0], [0], color="black", linewidth=0.7,
                           linestyle="--", alpha=0.4, label="Tillage reset"),
                ],
                fontsize=L["fs_legend"], loc="upper right", ncol=3, framealpha=0.85)
            ax_ts.set_title("IRD Time Series",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

        # ── SCATTER ───────────────────────────────────────────────────────────
        yt = ird_t[mask]; yp = ird_p[mask]
        pad = L["scatter_pad"]
        lo  = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        rng = hi - lo
        lim_lo = lo - pad * rng; lim_hi = hi + pad * rng
        ax_sc.scatter(yt, yp, s=L["sc_s"], alpha=L["sc_alpha"],
                      color=color, edgecolors="none", zorder=3)
        ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                   color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
        ax_sc.set_xlim(lim_lo, lim_hi); ax_sc.set_ylim(lim_lo, lim_hi)
        # ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.tick_params(axis="both", labelsize=L["fs_tick"])
        if is_last:
            ax_sc.set_xlabel("Actual IRD (cm/h)", fontsize=L["fs_xlabel"])
        if row_idx == 2:
            ax_sc.set_ylabel("Predicted (cm/h)", fontsize=L["fs_ylabel"])
        if row_idx == 0:
            ax_sc.set_title("Actual vs Predicted",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Caption (clean) + text-routed numbers (separate block)
# ─────────────────────────────────────────────────────────────────────────────

def _print_caption_and_numbers(held_out, basin_metrics):
    pooled = basin_metrics.get("POOLED", {})

    caption = (
        "Figure 4. Model 1 (within-segment decay) on five completely unseen "
        "held-out basins, one per field (Condition E: trained on all non-held-out "
        "basins, all sensor-passing events, no site-specific recalibration). "
        "Left: measured (circles) and predicted (crosses) IRD (cm h⁻¹) over time; "
        "dashed vertical lines mark tillage events; background shading denotes "
        "season (winter blue, spring peach, summer yellow, autumn green). "
        "Right: predicted versus measured IRD; dashed line is 1:1. Per-basin and "
        "pooled metrics are given in Table 1."
    )

    print("\n" + "=" * 70)
    print("ARTICLE CAPTION (Figure 4):")
    print("=" * 70)
    print(caption)

    print("\n" + "=" * 70)
    print("TEXT / TABLE 1 NUMBERS (not for the caption — route to §3.2 / Table 1):")
    print("=" * 70)
    print(f"  Pooled (n={pooled.get('n', 0):,}): "
          f"R²={pooled.get('r2', np.nan):+.3f}, "
          f"RMSE={pooled.get('rmse', np.nan):.3f} cm/h, "
          f"MAPE={pooled.get('mape', np.nan):.1f}%")
    print("  Per-basin → Table 1 (already locked):")
    for bn in held_out:
        m = basin_metrics.get(bn, {})
        print(f"    {bn} {m.get('field',''):<9} "
              f"R²={m.get('r2', np.nan):+.3f}  RMSE={m.get('rmse', np.nan):.3f}  "
              f"naive={m.get('rmse_naive', np.nan):.3f}  "
              f"linear={m.get('rmse_linear', np.nan):.3f}  "
              f"MAPE={m.get('mape', np.nan):.1f}%")
    print("  (Linear-baseline column is for §3.2 / SI, not Table 1 or caption.)")


def main():
    print("=" * 65)
    print("  FIGURE 4 — Model 1 | Time Series + Scatter  (Condition E)")
    print("  Reproduces models.model1_decay exactly (shared code path)")
    print("=" * 65)

    held_out      = load_selected_basins()
    df, feat_cols = load_data()
    train, val, test = prepare_splits(df, held_out, feat_cols)
    model, sc        = train_model(train, val, feat_cols)

    ho, basin_metrics = predict_held_out(model, sc, feat_cols, test, held_out)

    print("\n  Rendering figure...")
    plot_figure(ho, basin_metrics, held_out)
    _print_caption_and_numbers(held_out, basin_metrics)


if __name__ == "__main__":
    main()