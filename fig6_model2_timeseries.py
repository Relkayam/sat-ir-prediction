"""
figures/fig6_model2_timeseries.py — Figure 6: Model 2 held-out time series + scatter
======================================================================================
Condition E: trained on 45 basins (all resets including outliers),
tested on 5 completely unseen held-out basins.

Layout: 5 rows × 3 columns
  Col 0 (wide)   : IRD_reset time series — actual / predicted / naive, season bands
  Col 1 (medium) : Actual vs predicted scatter — Model 2 circles + naive triangles
  Col 2 (narrow) : Per-basin metrics text — right of scatter, no axis

TUNING
------
  All layout, spacing, and size parameters live in the LAYOUT dict below.
  Edit only that dict — nothing else needs changing for appearance.

Runtime: ~1 minute (reset dataset is small — 4,163 rows)

Usage
-----
  python fig6_model2_timeseries.py
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
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT, FIELD_COLORS, add_season_bands
from config import RESET_CSV, RANDOM_SEED, FIELD_NAMES
from pipeline.features import MODEL2_FEATURES, TARGET_M2

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  TUNING PANEL — edit everything here, nowhere else  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════
LAYOUT = dict(

    # ── Figure dimensions ─────────────────────────────────────────────────────
    fig_width   = 18,    # total figure width (inches)
    row_height  = 3.2,   # height of each basin row (inches)

    # ── Column width ratios: [timeseries, scatter, blank_for_metrics_text] ───
    col_ratios  = [3.5, 1.6, 0.7],

    # ── Spacing ───────────────────────────────────────────────────────────────
    hspace      = 0.22,   # vertical gap between rows
    wspace      = 0.28,   # horizontal gap between columns

    # ── Scatter axis padding (fraction of data range added each side) ─────────
    scatter_pad = 0.08,

    # ── Font sizes ────────────────────────────────────────────────────────────
    fs_title    = 11,
    fs_ylabel   = 10,
    fs_xlabel   = 10,
    fs_tick     = 9,
    fs_legend   = 9,
    fs_metrics  = 8.5,

    # ── Time series markers ───────────────────────────────────────────────────
    ts_s        = 35,    # larger than fig5 — fewer points per panel
    ts_alpha    = 0.80,

    # ── Scatter markers ───────────────────────────────────────────────────────
    sc_s_model  = 28,    # Model 2 circles
    sc_s_naive  = 22,    # naive triangles (slightly smaller, behind)
    sc_alpha    = 0.75,

    # ── Season bands ──────────────────────────────────────────────────────────
    season_alpha = 0.8,

    # ── Connecting lines actual→predicted in time series ─────────────────────
    connector_lw    = 0.5,
    connector_alpha = 0.30,

    # ── Metrics text position ─────────────────────────────────────────────────
    metrics_gap  = 0.012,  # horizontal gap right of scatter (figure fraction)
    metrics_vtop = 0.82,   # vertical anchor (fraction from row bottom, upward)
    metrics_ls   = 1.6,    # line spacing
)
# ▲▲▲  end of tuning panel  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════════

HELD_OUT_BASINS = [3203, 4104, 5102, 6303, 7201]
L = LAYOUT


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if mask.sum() < 2:
        return {}
    yt, yp = y_true[mask], y_pred[mask]
    return dict(
        r2   = round(float(r2_score(yt, yp)), 3),
        rmse = round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
        mape = round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 1),
        n    = int(mask.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & training  (logic unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_train():
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.preprocessing import StandardScaler

    print("  Loading reset dataset...")
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    avail = [f for f in MODEL2_FEATURES if f in df.columns]
    print(f"  Features available: {len(avail)}  |  Total rows: {len(df)}")

    # Condition E: reassign outlier rows from "excluded" → chrono splits
    df_e = df.copy()
    outlier_mask = (df_e["basin_role"] == "outlier") & (df_e["split_held_out"] == "excluded")
    df_e.loc[outlier_mask, "split_held_out"] = df_e.loc[outlier_mask, "split_chrono"]
    print(f"  Condition E: {int(outlier_mask.sum())} outlier rows reassigned to chrono splits")

    required = avail + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    train = df_e[df_e["split_held_out"] == "train"].dropna(subset=required).reset_index(drop=True)
    val   = df_e[df_e["split_held_out"] == "val"].dropna(subset=required).reset_index(drop=True)
    test  = df_e[df_e["split_held_out"] == "held_out_test"].dropna(subset=required).reset_index(drop=True)
    print(f"  Split: train={len(train)}  val={len(val)}  test={len(test)}")

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[avail].values)
    Xva = sc.transform(val[avail].values)
    Xte = sc.transform(test[avail].values)

    print("  Training LightGBM Model 2 Condition E...")
    model = LGBMRegressor(
        n_estimators=1000, max_depth=-1, num_leaves=31,
        learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(
        Xtr, train[TARGET_M2].values,
        eval_set=[(Xva, val[TARGET_M2].values)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
    )
    print(f"  best_iter={model.best_iteration_}")

    prev_ird         = test["prev_IRD_at_reset_raw"].values.astype(float)
    test             = test.copy()
    test["ird_pred"] = prev_ird * np.exp(model.predict(Xte))
    test["ird_true"] = test["IRD_at_reset"].values.astype(float)
    test["ird_naive"]= prev_ird

    # Console metrics
    print(f"\n  Held-out basin metrics (Model 2, Condition E):")
    print(f"  {'Basin':<8} {'Field':<12} {'n':>5} "
          f"{'M2 R²':>8} {'M2 RMSE':>8} {'M2 MAPE%':>9} "
          f"{'Nv R²':>8} {'Nv RMSE':>8}")
    print(f"  {'-'*72}")

    basin_metrics = {}
    all_t, all_p, all_n = [], [], []
    for bn in HELD_OUT_BASINS:
        bdf   = test[test["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        m2    = _metrics(ird_t, ird_p)
        mn    = _metrics(ird_t, ird_n)
        basin_metrics[bn] = {**m2, "field": field,
                             "rmse_naive": mn.get("rmse", np.nan),
                             "r2_naive":   mn.get("r2",   np.nan)}
        all_t.append(ird_t); all_p.append(ird_p); all_n.append(ird_n)
        print(f"  {bn:<8} {field:<12} {m2.get('n',0):>5} "
              f"{m2.get('r2',np.nan):>+8.3f} "
              f"{m2.get('rmse',np.nan):>8.3f} "
              f"{m2.get('mape',np.nan):>8.1f}% "
              f"{mn.get('r2',np.nan):>+8.3f} "
              f"{mn.get('rmse',np.nan):>8.3f}")

    ird_t_all = np.concatenate(all_t)
    ird_p_all = np.concatenate(all_p)
    ird_n_all = np.concatenate(all_n)
    m_all   = _metrics(ird_t_all, ird_p_all)
    m_naive = _metrics(ird_t_all, ird_n_all)
    delta   = m_naive.get("rmse", np.nan) - m_all.get("rmse", np.nan)
    basin_metrics["POOLED"] = {**m_all, "field": "all",
                               "rmse_naive": m_naive.get("rmse", np.nan),
                               "r2_naive":   m_naive.get("r2",   np.nan)}
    print(f"  {'POOLED':<8} {'all':<12} {m_all.get('n',0):>5} "
          f"{m_all.get('r2',np.nan):>+8.3f} "
          f"{m_all.get('rmse',np.nan):>8.3f} "
          f"{m_all.get('mape',np.nan):>8.1f}% "
          f"{m_naive.get('r2',np.nan):>+8.3f} "
          f"{m_naive.get('rmse',np.nan):>8.3f}")
    print(f"\n  RMSE improvement over naive: {delta:.3f} cm/h "
          f"({100*delta/m_naive.get('rmse',1):.1f}%)")

    return test, basin_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(test: pd.DataFrame, basin_metrics: dict) -> None:
    apply_style()

    n   = len(HELD_OUT_BASINS)
    fig = plt.figure(figsize=(L["fig_width"], L["row_height"] * n))

    gs = GridSpec(
        n, 3,
        figure=fig,
        width_ratios=L["col_ratios"],
        hspace=L["hspace"],
        wspace=L["wspace"],
    )

    for row_idx, bn in enumerate(HELD_OUT_BASINS):
        ax_ts = fig.add_subplot(gs[row_idx, 0])
        ax_sc = fig.add_subplot(gs[row_idx, 1])
        # gs[row_idx, 2]: intentionally no axis — used only for fig.text()

        bdf    = test[test["basin_number"] == bn].sort_values("reset_date")
        field  = FIELD_NAMES.get(int(str(bn)[0]), "")
        color  = FIELD_COLORS.get(field, COLORS["deep_blue"])
        is_last = (row_idx == n - 1)
        is_mid  = (row_idx == n // 2)

        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        m     = basin_metrics.get(bn, {})

        # ── TIME SERIES ───────────────────────────────────────────────────────
        import plot_style as _ps
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, L["season_alpha"])
        add_season_bands(ax_ts, bdf["reset_date"].min(), bdf["reset_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        dates  = bdf.loc[mask, "reset_date"]
        yt_msk = ird_t[mask];  yp_msk = ird_p[mask];  yn_msk = ird_n[mask]

        # Naive — light grey step line
        ax_ts.plot(dates, yn_msk,
                   color="lightgrey", linewidth=1.0, linestyle="-",
                   alpha=0.9, zorder=2)

        # Actual — blue circles
        ax_ts.scatter(dates, yt_msk,
                      s=L["ts_s"], alpha=L["ts_alpha"],
                      color=COLORS["deep_blue"], marker="o", zorder=4)

        # Predicted — orange crosses
        ax_ts.scatter(dates, yp_msk,
                      s=L["ts_s"], alpha=L["ts_alpha"],
                      color=COLORS["orange"], marker="x",
                      linewidths=1.5, zorder=5)

        # Connecting lines actual → predicted
        for dt, yt_i, yp_i in zip(dates, yt_msk, yp_msk):
            ax_ts.plot([dt, dt], [yt_i, yp_i],
                       color="gray", linewidth=L["connector_lw"],
                       alpha=L["connector_alpha"], zorder=3)

        # Thin chronological line through actuals
        ax_ts.plot(dates, yt_msk,
                   color=COLORS["deep_blue"], linewidth=0.6,
                   alpha=0.25, linestyle="-", zorder=3)

        if is_mid:
            ax_ts.set_ylabel(f"IRD_reset (cm/h) \n {bn} ",
                             fontsize=L["fs_ylabel"])
        else:
            ax_ts.set_ylabel(f"{bn} ", fontsize=L["fs_ylabel"])

        ax_ts.tick_params(axis="both", labelsize=L["fs_tick"])
        ax_ts.tick_params(axis="x", rotation=25)
        if not is_last:
            ax_ts.set_xticklabels([])
        if is_last:
            ax_ts.set_xlabel("Date", fontsize=L["fs_xlabel"])

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
                fontsize=L["fs_legend"], loc="upper right",
                ncol=3, framealpha=0.85,
            )
            ax_ts.set_title("IRD_reset Time Series",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

        # ── SCATTER ───────────────────────────────────────────────────────────
        yt = ird_t[mask];  yp = ird_p[mask];  yn = ird_n[mask]
        pad = L["scatter_pad"]
        all_vals = np.concatenate([yt, yp, yn])
        lo = all_vals.min();  hi = all_vals.max();  rng = hi - lo
        lim_lo = lo - pad * rng;  lim_hi = hi + pad * rng

        # Naive — grey triangles behind
        ax_sc.scatter(yt, yn,
                      s=L["sc_s_naive"], alpha=0.35,
                      color="grey", marker="^", zorder=2)
        # Model 2 — field color circles in front
        ax_sc.scatter(yt, yp,
                      s=L["sc_s_model"], alpha=L["sc_alpha"],
                      color=color, edgecolors="none", zorder=3)
        # 1:1 line
        ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                   color="black", linewidth=0.8, linestyle="--",
                   alpha=0.6, zorder=1)
        ax_sc.set_xlim(lim_lo, lim_hi)
        ax_sc.set_ylim(lim_lo, lim_hi)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.tick_params(axis="both", labelsize=L["fs_tick"])

        if is_last:
            ax_sc.set_xlabel("Actual IRD_reset (cm/h)", fontsize=L["fs_xlabel"])
        if is_mid:
            ax_sc.set_ylabel("Predicted (cm/h)", fontsize=L["fs_ylabel"])
        if row_idx == 0:
            ax_sc.legend(
                handles=[
                    Line2D([0], [0], marker="o", color=color,
                           markersize=5, linestyle="None", label="Model 2"),
                    Line2D([0], [0], marker="^", color="grey",
                           markersize=5, linestyle="None", label="Naive"),
                ],
                fontsize=L["fs_legend"] - 1, loc="lower right", framealpha=0.85,
            )
            ax_sc.set_title("Actual vs Predicted",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

        # ── METRICS TEXT — right of scatter using fig.text() ──────────────────
        fig.canvas.draw()
        bb    = ax_sc.get_position()
        txt_x = bb.x1 + L["metrics_gap"]
        txt_y = bb.y0 + bb.height * L["metrics_vtop"]

        r2      = m.get("r2",         np.nan)
        rmse    = m.get("rmse",       np.nan)
        mape    = m.get("mape",       np.nan)
        n_ev    = m.get("n",          0)
        rmse_nv = m.get("rmse_naive", np.nan)

        fig.text(
            txt_x, txt_y,
            f"R²  = {r2:+.3f}\n"
            f"RMSE = {rmse:.3f} cm/h\n"
            f"Naive RMSE = {rmse_nv:.3f}\n"
            f"MAPE = {mape:.1f}%\n"
            f"n = {n_ev:,}",
            fontsize=L["fs_metrics"],
            verticalalignment="top",
            linespacing=L["metrics_ls"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.88, edgecolor="lightgrey", linewidth=0.6),
        )

    plt.tight_layout(rect=[0, 0, 0.87, 1])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Caption template  (update XX from console output after running)
# ─────────────────────────────────────────────────────────────────────────────
CAPTION = (
    "Figure 6. Model 2 generalizability to completely unseen basins "
    "(Condition E: trained on 45 basins including operational outliers). "
    "Left: actual (circles) and predicted (crosses) post-tillage IRD "
    "(IRD_reset, cm/h) at each tillage event; grey line = naive baseline "
    "(predict no change from previous reset). "
    "Centre: actual vs predicted scatter; grey triangles = naive baseline, "
    "coloured circles = Model 2; dashed line = 1:1. "
    "Right: R², RMSE, naive RMSE, and MAPE per basin. "
    "No site-specific recalibration was applied. "
    "Per-basin metrics (update after running): "
    "Basin 3203 (Soreq 2): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 4104 (Yavne 1): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 5102 (Yavne 2): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 6303 (Yavne 3): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 7201 (Yavne 4): R²=XX, RMSE=XX cm/h, MAPE=XX%. "
    "Pooled: R²=XX, RMSE=XX cm/h, MAPE=XX% (n=XX tillage events). "
    "Naive baseline pooled: RMSE=XX cm/h. "
    "Model 2 reduces RMSE by XX cm/h (XX%) over naive."
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FIGURE 6 — Model 2 | Time Series + Scatter  (Condition E)")
    print("=" * 60)
    print("  Tuning: edit the LAYOUT dict at the top of this file.\n")

    test, basin_metrics = load_and_train()

    print("\n  Rendering figure...")
    plot_figure(test, basin_metrics)

    print("\n" + "─" * 60)
    print("CAPTION TEMPLATE (replace XX with console values above):")
    print("─" * 60)
    print(CAPTION)


if __name__ == "__main__":
    main()