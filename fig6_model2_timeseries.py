"""
figures/fig6_model2_timeseries.py — Figure 6: Model 2 held-out time series
===========================================================================
Condition E: trained on 45 basins (all resets including outliers),
tested on 5 completely unseen held-out basins.

Each point = one tillage event (IRD_at_reset, cm/h).
Layout: 5 rows × 1 column — one panel per held-out basin.

Runtime: ~1 minute (reset dataset is small — 4,163 rows)

Usage
-----
  python fig6_model2_timeseries.py
  Save manually as PNG (300 DPI) + TIFF.

FONT SIZE CONTROL
-----------------
  Edit FONT_OVERRIDE below.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT, add_season_bands
from config import RESET_CSV, RANDOM_SEED, FIELD_NAMES
from pipeline.features import MODEL2_FEATURES, TARGET_M2

# ── Config ────────────────────────────────────────────────────────────────────
HELD_OUT_BASINS = [3203, 4104, 5102, 6303, 7201]

FONT_OVERRIDE = {
    "title"      : 11,
    "axis_label" : 10,
    "tick"       : 9,
    "legend"     : 9,
    "annotation" : 9,
}

SEASON_ALPHA  = 0.60
MARKER_SIZE   = 35       # larger than Fig 5 — fewer points, can afford it
MARKER_ALPHA  = 0.80
RESET_LINE_KW = dict(color="black", linewidth=0.5,
                     linestyle="--", alpha=0.20, zorder=1)

def _fs(key):
    return FONT_OVERRIDE.get(key, FONT.get(key, 11))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    if mask.sum() < 2:
        return {}
    yt, yp = y_true[mask], y_pred[mask]
    rmse   = float(np.sqrt(mean_squared_error(yt, yp)))
    sr, _  = sp_stats.spearmanr(yt, yp)
    return dict(
        r2   = round(float(r2_score(yt, yp)), 3),
        rmse = round(rmse, 3),
        mape = round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 1),
        n    = int(mask.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load and train — Condition E for Model 2
# ─────────────────────────────────────────────────────────────────────────────

def load_and_train():
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.preprocessing import StandardScaler

    print("  Loading reset dataset...")
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    avail = [f for f in MODEL2_FEATURES if f in df.columns]
    print(f"  Features: {len(avail)}  Rows: {len(df)}")

    # ── Condition E: reassign outlier basins from excluded → chrono splits ────
    df_e = df.copy()
    outlier_mask = (
        (df_e["basin_role"] == "outlier") &
        (df_e["split_held_out"] == "excluded")
    )
    df_e.loc[outlier_mask, "split_held_out"] = df_e.loc[
        outlier_mask, "split_chrono"
    ]
    n_reassigned = int(outlier_mask.sum())
    print(f"  Condition E: {n_reassigned} outlier rows reassigned to chrono splits")

    required = avail + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]

    train = df_e[df_e["split_held_out"] == "train"].dropna(
        subset=required).reset_index(drop=True)
    val   = df_e[df_e["split_held_out"] == "val"].dropna(
        subset=required).reset_index(drop=True)
    test  = df_e[df_e["split_held_out"] == "held_out_test"].dropna(
        subset=required).reset_index(drop=True)

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
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"  best_iter={model.best_iteration_}")

    # Predict on held-out test
    y_pred_log = model.predict(Xte)
    prev_ird   = test["prev_IRD_at_reset_raw"].values.astype(float)
    test["ird_pred"] = prev_ird * np.exp(y_pred_log)
    test["ird_true"] = test["IRD_at_reset"].values.astype(float)

    # Also compute naive baseline (predict no change)
    test["ird_naive"] = prev_ird

    # Print per-basin metrics
    print(f"\n  Held-out basin metrics (Model 2, Condition E):")
    print(f"  {'Basin':<8} {'Field':<12} {'n':>5} "
          f"{'R²':>8} {'RMSE':>8} {'MAPE%':>7}")
    print(f"  {'-'*55}")

    all_true, all_pred = [], []
    for bn in HELD_OUT_BASINS:
        bdf   = test[test["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        m     = _metrics(ird_t, ird_p)
        all_true.append(ird_t)
        all_pred.append(ird_p)
        print(f"  {bn:<8} {field:<12} {m.get('n',0):>5} "
              f"{m.get('r2',np.nan):>+8.3f} "
              f"{m.get('rmse',np.nan):>8.3f} "
              f"{m.get('mape',np.nan):>6.1f}%")

    # Pooled
    ird_t_all = np.concatenate(all_true)
    ird_p_all = np.concatenate(all_pred)
    m_all = _metrics(ird_t_all, ird_p_all)
    print(f"  {'POOLED':<8} {'all':<12} {m_all.get('n',0):>5} "
          f"{m_all.get('r2',np.nan):>+8.3f} "
          f"{m_all.get('rmse',np.nan):>8.3f} "
          f"{m_all.get('mape',np.nan):>6.1f}%")

    # Naive pooled
    m_naive = _metrics(ird_t_all,
                       np.concatenate([test[test["basin_number"] == bn]
                                       ["ird_naive"].values
                                       for bn in HELD_OUT_BASINS]))
    print(f"\n  Naive baseline (pooled): "
          f"R²={m_naive.get('r2',np.nan):+.3f}  "
          f"RMSE={m_naive.get('rmse',np.nan):.3f} cm/h  "
          f"MAPE={m_naive.get('mape',np.nan):.1f}%")

    return test


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(test: pd.DataFrame) -> None:
    apply_style()

    n_panels = len(HELD_OUT_BASINS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 4.0 * n_panels),
        gridspec_kw={"hspace": 0.40},
    )

    for ax, bn in zip(axes, HELD_OUT_BASINS):
        bdf   = test[test["basin_number"] == bn].sort_values("reset_date")
        field = FIELD_NAMES.get(int(str(bn)[0]), "")

        is_last   = (bn == HELD_OUT_BASINS[-1])
        is_mid    = (bn == HELD_OUT_BASINS[len(HELD_OUT_BASINS) // 2])
        panel_idx = HELD_OUT_BASINS.index(bn)

        # ── Season bands ──────────────────────────────────────────────────────
        import plot_style as _ps
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, SEASON_ALPHA)
        add_season_bands(ax,
                         bdf["reset_date"].min(),
                         bdf["reset_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        # ── Actual IRD_at_reset — blue circles ────────────────────────────────
        valid = (bdf["ird_true"].notna() & bdf["ird_pred"].notna() &
                 (bdf["ird_true"] > 0) & (bdf["ird_pred"] > 0))

        ax.scatter(
            bdf.loc[valid, "reset_date"],
            bdf.loc[valid, "ird_true"],
            s=MARKER_SIZE, alpha=MARKER_ALPHA,
            color=COLORS["deep_blue"], marker="o",
            zorder=4, label="Actual IRD_reset",
        )

        # ── Predicted IRD_at_reset — green crosses ────────────────────────────
        ax.scatter(
            bdf.loc[valid, "reset_date"],
            bdf.loc[valid, "ird_pred"],
            s=MARKER_SIZE, alpha=MARKER_ALPHA,
            color=COLORS["green"], marker="x", linewidths=1.5,
            zorder=5, label="Predicted IRD_reset",
        )

        # ── Connect actual to predicted ───────────────────────────────────────
        for _, row in bdf[valid].iterrows():
            ax.plot(
                [row["reset_date"]] * 2,
                [row["ird_true"], row["ird_pred"]],
                color="gray", linewidth=0.5, alpha=0.30, zorder=2,
            )

        # ── Connect actual points chronologically (thin line) ─────────────────
        ax.plot(
            bdf.loc[valid, "reset_date"],
            bdf.loc[valid, "ird_true"],
            color=COLORS["deep_blue"], linewidth=0.6,
            alpha=0.25, linestyle="-", zorder=3,
        )

        # ── Y-axis: basin number, units on middle panel only ──────────────────
        if is_mid:
            ax.set_ylabel(f"Basin {bn}\nIRD_reset (cm/h)",
                          fontsize=_fs("axis_label"))
        else:
            ax.set_ylabel(f"Basin {bn}",
                          fontsize=_fs("axis_label"))

        ax.tick_params(axis="both", labelsize=_fs("tick"))
        ax.tick_params(axis="x", rotation=25)

        if not is_last:
            ax.set_xticklabels([])

        # ── Legend — first panel only ─────────────────────────────────────────
        if panel_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color=COLORS["deep_blue"],
                       markersize=5, linestyle="None",
                       label="Actual IRD_reset"),
                Line2D([0], [0], marker="x", color=COLORS["green"],
                       markersize=5, linestyle="None",
                       markeredgewidth=1.5, label="Predicted IRD_reset"),
            ]
            ax.legend(
                handles=legend_elements,
                fontsize=_fs("legend"),
                loc="upper right", ncol=2,
                framealpha=0.85,
            )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Caption — updated after running with actual metrics
# ─────────────────────────────────────────────────────────────────────────────

CAPTION = (
    "Figure 6. Model 2 generalizability to completely unseen basins "
    "(Condition E: trained on 45 basins including operational outliers). "
    "Each panel shows the actual (circles) and predicted (crosses) "
    "post-tillage IRD (IRD_reset, cm/h) at each tillage event "
    "for one of five held-out basins. "
    "Each point represents one tillage event. "
    "Thin grey lines connect each actual–predicted pair. "
    "Seasonal background shading indicates winter (blue), spring (peach), "
    "summer (yellow), and autumn (green). "
    "The model was applied without any site-specific recalibration. "
    "Per-basin metrics — "
    "Basin 3203 (Soreq 2): R²=+0.422, RMSE=1.784 cm/h, MAPE=13.5%; "
    "Basin 4104 (Yavne 1): R²=+0.418, RMSE=0.333 cm/h, MAPE=13.7%; "
    "Basin 5102 (Yavne 2): R²=+0.557, RMSE=0.904 cm/h, MAPE=16.2%; "
    "Basin 6303 (Yavne 3): R²=+0.354, RMSE=0.585 cm/h, MAPE=15.4%; "
    "Basin 7201 (Yavne 4): R²=+0.697, RMSE=0.231 cm/h, MAPE=15.6%. "
    "Pooled across all five basins: R²=+0.884, RMSE=0.983 cm/h, MAPE=14.8% "
    "(n=454 tillage events). "
    "The naive baseline (predicting no change from previous reset) achieves "
    "R²=+0.867, RMSE=1.053 cm/h on the same test set — "
    "Model 2 reduces RMSE by 0.070 cm/h (6.6%) over naive. "
    "Note: per-basin R² values are lower than the pooled figure because "
    "pooled R² captures between-basin variance in IRD_reset levels, "
    "which the model recovers through the autocorrelation features "
    "(prev_IRD_at_reset, prev_prev_IRD_at_reset)."
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FIGURE 6 — Model 2 Held-out Time Series  (Condition E)")
    print("=" * 60)

    test = load_and_train()

    print("\n  Rendering figure...")
    print("  → Save manually: PNG (300 DPI) + TIFF\n")
    plot_figure(test)

    print("\n" + "─" * 60)
    print("CAPTION TEMPLATE (update metrics after running):")
    print("─" * 60)
    print(CAPTION)
    print("─" * 60)
    print("\n  → Copy per-basin metrics from console output above")
    print("  → Update CAPTION string in this file with actual numbers")


if __name__ == "__main__":
    main()