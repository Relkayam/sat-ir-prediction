"""
models/model2_reset.py — Model 2: IRD recovery prediction after tillage (V2)
=============================================================================
V2 changes over V1:
  - Three evaluation conditions mirroring Model 1:
      Chrono     : clean basins, chronological split (primary within-sample)
      Held-out D : clean basins train, 5 unseen basins test (split_held_out)
      Held-out E : all 45 non-held-out basins train, 5 unseen basins test
  - Time series plots for 5 held-out basins
  - Scatter plots with per-basin metrics for held-out test
  - SHAP on chrono condition
  - Per-held-out-basin summary table

Target
------
  IRD_norm_log_reset = log(IRD_at_reset[i] / IRD_at_reset[i-1])
  Back-transform: IRD_at_reset[i] = prev_IRD_at_reset * exp(predicted)

Usage
-----
  python -m models.model2_reset
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shap

from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RESET_CSV, TABLES_DIR, RANDOM_SEED, FIELD_NAMES,
)
from pipeline.features import MODEL2_FEATURES, TARGET_M2


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_raw(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "",
    verbose: bool = True,
) -> dict:
    mask = (
        np.isfinite(y_true) & np.isfinite(y_pred) &
        (y_true > 0) & (y_pred > 0)
    )
    if mask.sum() < 2:
        return {}
    yt, yp     = y_true[mask], y_pred[mask]
    rmse       = float(np.sqrt(mean_squared_error(yt, yp)))
    ird_mean   = float(np.mean(yt))
    spear_r, _ = sp_stats.spearmanr(yt, yp)
    m = dict(
        r2         = round(float(r2_score(yt, yp)),               4),
        rmse       = round(rmse,                                    4),
        mae        = round(float(mean_absolute_error(yt, yp)),     4),
        mape       = round(float(np.mean(np.abs((yt-yp)/yt))*100), 2),
        spearman_r = round(float(spear_r),                         4),
        rel_rmse   = round(rmse / ird_mean if ird_mean > 0 else np.nan, 4),
        n          = int(mask.sum()),
    )
    if verbose and label:
        print(
            f"  {label:<50}  "
            f"R²={m['r2']:+.4f}  "
            f"RMSE={m['rmse']:.4f} cm/h  "
            f"MAPE={m['mape']:.1f}%  "
            f"n={m['n']}"
        )
    return m


def _metrics_log(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "",
    verbose: bool = True,
) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {}
    yt, yp     = y_true[mask], y_pred[mask]
    spear_r, _ = sp_stats.spearmanr(yt, yp)
    m = dict(
        r2         = round(float(r2_score(yt, yp)),           4),
        rmse       = round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        mae        = round(float(mean_absolute_error(yt, yp)), 4),
        spearman_r = round(float(spear_r),                     4),
        n          = int(mask.sum()),
    )
    if verbose and label:
        print(
            f"  {label:<50}  "
            f"R²={m['r2']:+.4f}  "
            f"RMSE={m['rmse']:.4f}  "
            f"n={m['n']}"
        )
    return m


def _back_transform(
    y_pred_log: np.ndarray,
    prev_ird:   np.ndarray,
) -> np.ndarray:
    return prev_ird * np.exp(y_pred_log)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not RESET_CSV.exists():
        raise FileNotFoundError(
            f"{RESET_CSV} not found.\n"
            "Run: python -m pipeline.build_reset_dataset"
        )
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [f for f in MODEL2_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: missing features: {missing}")

    avail = [f for f in MODEL2_FEATURES if f in df.columns]
    print(f"  Reset dataset : {len(df)} rows  "
          f"{df['basin_number'].nunique()} basins")
    print(f"  Features      : {len(avail)} / {len(MODEL2_FEATURES)} available")

    # Split counts
    for col in ["split_chrono", "split_held_out"]:
        if col in df.columns:
            counts = df[col].value_counts().to_dict()
            print(f"  {col}: {counts}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate one condition
# ─────────────────────────────────────────────────────────────────────────────


def train_and_evaluate(
    df:        pd.DataFrame,
    split_col: str,
    test_val:  str,
    label:     str,
) -> tuple[dict, dict, object, StandardScaler, list[str]]:
    """
    Train LightGBM, evaluate on test split.

    Parameters
    ----------
    df        : full reset dataset
    split_col : column to use for splitting ('split_chrono' or 'split_held_out')
    test_val  : value in split_col that defines the test set
                ('test' for chrono, 'held_out_test' for held-out conditions)
    label     : descriptive label for printing

    Returns
    -------
    test_raw, test_log, model, scaler, feat_cols
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    avail = [f for f in MODEL2_FEATURES if f in df.columns]
    required = avail + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]

    train = df[df[split_col] == "train"].dropna(
        subset=required
    ).reset_index(drop=True)
    val   = df[df[split_col] == "val"].dropna(
        subset=required
    ).reset_index(drop=True)
    test  = df[df[split_col] == test_val].dropna(
        subset=required
    ).reset_index(drop=True)

    print(f"\n  [{label}]  "
          f"train={len(train)}  val={len(val)}  test={len(test)}")

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        print(f"  WARNING: insufficient data — skipping")
        return {}, {}, None, None, avail

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[avail].values)
    Xva = sc.transform(val[avail].values)
    Xte = sc.transform(test[avail].values)

    model = LGBMRegressor(
        n_estimators      = 1000,
        max_depth         = -1,
        num_leaves        = 31,
        learning_rate     = 0.05,
        subsample         = 0.8,
        feature_fraction  = 0.8,
        min_child_samples = 10,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
        verbose           = -1,
    )
    model.fit(
        Xtr, train[TARGET_M2].values,
        eval_set=[(Xva, val[TARGET_M2].values)],
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"  LightGBM best_iter={model.best_iteration_}  "
          f"features={len(avail)}")

    # Log-ratio metrics
    y_pred_log = model.predict(Xte)
    test_log   = _metrics_log(
        test[TARGET_M2].values, y_pred_log,
        label=f"  {label} [log]", verbose=True,
    )

    # Raw IRD metrics
    prev_ird   = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_pred   = _back_transform(y_pred_log, prev_ird)
    ird_true   = test["IRD_at_reset"].values.astype(float)
    test_raw   = _metrics_raw(
        ird_true, ird_pred,
        label=f"  {label} [IRD cm/h]", verbose=True,
    )

    return test_raw, test_log, model, sc, avail


# ─────────────────────────────────────────────────────────────────────────────
# Naive baseline
# ─────────────────────────────────────────────────────────────────────────────

def compute_naive(df: pd.DataFrame, split_col: str, test_val: str) -> dict:
    test = df[df[split_col] == test_val].copy()
    valid = (
        test["IRD_at_reset"].notna() &
        test["prev_IRD_at_reset_raw"].notna() &
        (test["IRD_at_reset"] > 0) &
        (test["prev_IRD_at_reset_raw"] > 0)
    )
    if valid.sum() < 2:
        return {}
    return _metrics_raw(
        test.loc[valid, "IRD_at_reset"].values.astype(float),
        test.loc[valid, "prev_IRD_at_reset_raw"].values.astype(float),
        label=f"  Naive [{split_col}/{test_val}]",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Held-out basin summary
# ─────────────────────────────────────────────────────────────────────────────

def print_held_out_summary(
    df:        pd.DataFrame,
    model,
    scaler:    StandardScaler,
    feat_cols: list[str],
    split_col: str,
    test_val:  str,
    label:     str,
) -> pd.DataFrame:
    """Per-basin metrics table for held-out test basins."""
    test = df[df[split_col] == test_val].dropna(
        subset=feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    ).copy()

    if test.empty or model is None:
        return pd.DataFrame()

    Xte = scaler.transform(test[feat_cols].values)
    test["_pred_log"] = model.predict(Xte)
    test["_ird_pred"] = _back_transform(
        test["_pred_log"].values,
        test["prev_IRD_at_reset_raw"].values.astype(float),
    )

    held_out_basins = sorted(test["basin_number"].unique().tolist())

    print(f"\n{'='*70}")
    print(f"  {label} — Per-basin held-out test metrics")
    print(f"{'='*70}")
    print(f"  {'Basin':>7}  {'Field':<12}  {'n':>5}  "
          f"{'R²':>8}  {'RMSE':>8}  {'MAPE%':>7}")
    print(f"  {'-'*55}")

    rows = []
    all_true, all_pred = [], []

    for bn in held_out_basins:
        bdf = test[test["basin_number"] == bn]
        ird_t = bdf["IRD_at_reset"].values.astype(float)
        ird_p = bdf["_ird_pred"].values.astype(float)
        mask  = (
            np.isfinite(ird_t) & np.isfinite(ird_p) &
            (ird_t > 0) & (ird_p > 0)
        )
        ird_t = ird_t[mask]; ird_p = ird_p[mask]
        all_true.append(ird_t); all_pred.append(ird_p)

        m     = _metrics_raw(ird_t, ird_p, verbose=False) or {}
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        print(
            f"  {bn:>7}  {field:<12}  "
            f"{m.get('n', 0):>5}  "
            f"{m.get('r2', np.nan):>+8.3f}  "
            f"{m.get('rmse', np.nan):>8.3f}  "
            f"{m.get('mape', np.nan):>6.1f}%"
        )
        rows.append(dict(basin=bn, field=field, **m))

    # Pooled
    if all_true:
        ird_t_all = np.concatenate(all_true)
        ird_p_all = np.concatenate(all_pred)
        m_all = _metrics_raw(ird_t_all, ird_p_all, verbose=False) or {}
        print(f"  {'-'*55}")
        print(
            f"  {'POOLED':>7}  {'all held-out':<12}  "
            f"{m_all.get('n', 0):>5}  "
            f"{m_all.get('r2', np.nan):>+8.3f}  "
            f"{m_all.get('rmse', np.nan):>8.3f}  "
            f"{m_all.get('mape', np.nan):>6.1f}%"
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: scatter actual vs predicted for held-out basins
# ─────────────────────────────────────────────────────────────────────────────

def plot_held_out_scatter(
    df:        pd.DataFrame,
    model,
    scaler:    StandardScaler,
    feat_cols: list[str],
    split_col: str,
    test_val:  str,
    label:     str,
) -> None:
    """
    One scatter panel per held-out basin.
    Actual vs predicted IRD_at_reset (cm/h).
    Metrics annotated on each panel.
    """
    test = df[df[split_col] == test_val].dropna(
        subset=feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    ).copy()

    if test.empty or model is None:
        return

    Xte = scaler.transform(test[feat_cols].values)
    test["_pred_log"] = model.predict(Xte)
    test["_ird_pred"] = _back_transform(
        test["_pred_log"].values,
        test["prev_IRD_at_reset_raw"].values.astype(float),
    )

    held_out_basins = sorted(test["basin_number"].unique().tolist())
    n_basins        = len(held_out_basins)
    fig, axes       = plt.subplots(1, n_basins, figsize=(4.5 * n_basins, 5),
                                   squeeze=False)
    axes = axes[0]

    rng    = np.random.default_rng(RANDOM_SEED)
    colors = ["tomato", "steelblue", "seagreen", "mediumpurple", "orange"]

    for ax, bn, color in zip(axes, held_out_basins, colors):
        bdf   = test[test["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))

        ird_t = bdf["IRD_at_reset"].values.astype(float)
        ird_p = bdf["_ird_pred"].values.astype(float)
        mask  = (
            np.isfinite(ird_t) & np.isfinite(ird_p) &
            (ird_t > 0) & (ird_p > 0)
        )
        ird_t = ird_t[mask]; ird_p = ird_p[mask]

        if len(ird_t) < 3:
            ax.set_title(f"Basin {bn}\n({field})\nInsufficient data")
            ax.axis("off")
            continue

        ax.scatter(ird_t, ird_p, s=40, alpha=0.7, color=color, zorder=3)

        lo = np.percentile(ird_t, 1) * 0.85
        hi = np.percentile(ird_t, 99) * 1.15
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        m = _metrics_raw(ird_t, ird_p, verbose=False)
        if m:
            ax.annotate(
                f"R²={m['r2']:+.3f}\n"
                f"RMSE={m['rmse']:.3f} cm/h\n"
                f"MAPE={m['mape']:.1f}%\n"
                f"n={m['n']}",
                xy=(0.05, 0.97), xycoords="axes fraction",
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.90),
            )

        ax.set_title(f"Basin {bn}\n({field})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual IRD_at_reset (cm/h)",    fontsize=8)
        ax.set_ylabel("Predicted IRD_at_reset (cm/h)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Model 2 — {label}\n"
        "Scatter: actual vs predicted IRD_at_reset (cm/h)\n"
        "These 5 basins were NEVER seen during training",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot: time series for held-out basins
# ─────────────────────────────────────────────────────────────────────────────

def plot_held_out_timeseries(
    df:        pd.DataFrame,
    model,
    scaler:    StandardScaler,
    feat_cols: list[str],
    split_col: str,
    test_val:  str,
    label:     str,
) -> None:
    """
    Time series of IRD_at_reset over the full 10-year record.
    One panel per held-out basin.
    Shows actual (filled circles) and predicted (crosses) reset values.
    Each point = one tillage event.
    """
    test = df[df[split_col] == test_val].dropna(
        subset=feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    ).copy()

    if test.empty or model is None:
        return

    Xte = scaler.transform(test[feat_cols].values)
    test["_pred_log"] = model.predict(Xte)
    test["_ird_pred"] = _back_transform(
        test["_pred_log"].values,
        test["prev_IRD_at_reset_raw"].values.astype(float),
    )

    held_out_basins = sorted(test["basin_number"].unique().tolist())
    n_basins        = len(held_out_basins)

    fig, axes = plt.subplots(n_basins, 1,
                             figsize=(14, 4 * n_basins),
                             squeeze=False)

    colors = ["tomato", "steelblue", "seagreen", "mediumpurple", "orange"]

    for ax, bn, color in zip(axes[:, 0], held_out_basins, colors):
        bdf = test[test["basin_number"] == bn].sort_values("reset_date")
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))

        ird_t = bdf["IRD_at_reset"].values.astype(float)
        ird_p = bdf["_ird_pred"].values.astype(float)
        dates = bdf["reset_date"].values

        mask = (
            np.isfinite(ird_t) & np.isfinite(ird_p) &
            (ird_t > 0) & (ird_p > 0)
        )

        # Actual — filled circles
        ax.scatter(
            dates[mask], ird_t[mask],
            s=35, alpha=0.85, color=color,
            marker="o", zorder=4, label="Actual IRD_at_reset",
        )
        # Predicted — crosses
        ax.scatter(
            dates[mask], ird_p[mask],
            s=35, alpha=0.85, color=color,
            marker="x", linewidths=1.8, zorder=5,
            label="Predicted IRD_at_reset",
        )
        # Connect actual to predicted with thin lines
        for d, yt, yp in zip(dates[mask], ird_t[mask], ird_p[mask]):
            ax.plot(
                [d, d], [yt, yp],
                color="gray", linewidth=0.5, alpha=0.35, zorder=2,
            )

        # Connect actual points chronologically
        ax.plot(
            dates[mask], ird_t[mask],
            color=color, linewidth=0.6, alpha=0.3,
            linestyle="-", zorder=3,
        )

        m = _metrics_raw(ird_t[mask], ird_p[mask], verbose=False) or {}

        ax.set_title(
            f"Basin {bn}  ({field})  —  HELD-OUT (never seen in training)\n"
            f"R²={m.get('r2', np.nan):+.3f}  "
            f"RMSE={m.get('rmse', np.nan):.3f} cm/h  "
            f"MAPE={m.get('mape', np.nan):.1f}%  "
            f"n_resets={int(mask.sum())}",
            fontsize=9,
        )
        ax.set_ylabel("IRD_at_reset (cm/h)", fontsize=8)
        ax.set_xlabel("Reset date",          fontsize=8)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.grid(True, alpha=0.2)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color=color,
                   markersize=6, linestyle="None", label="Actual"),
            Line2D([0], [0], marker="x", color=color,
                   markersize=6, linestyle="None",
                   markeredgewidth=1.8, label="Predicted"),
        ]
        ax.legend(handles=legend_elements, fontsize=7,
                  loc="upper right", ncol=2)

    fig.suptitle(
        f"Model 2 — {label}\n"
        "Time series: IRD_at_reset at each tillage event\n"
        "Circles = actual  |  Crosses = predicted  |  "
        "These basins were never seen during training",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot: chrono scatter (actual vs predicted, all basins, coloured by field)
# ─────────────────────────────────────────────────────────────────────────────

def plot_chrono_scatter(
    df:        pd.DataFrame,
    model,
    scaler:    StandardScaler,
    feat_cols: list[str],
    naive_m:   dict,
) -> None:
    """Two-panel scatter: Model 2 vs naive baseline — chrono test split."""
    test = df[df["split_chrono"] == "test"].dropna(
        subset=feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    ).copy()

    if test.empty or model is None:
        return

    Xte = scaler.transform(test[feat_cols].values)
    test["_pred_log"] = model.predict(Xte)
    test["_ird_pred"] = _back_transform(
        test["_pred_log"].values,
        test["prev_IRD_at_reset_raw"].values.astype(float),
    )
    test["_ird_naive"] = test["prev_IRD_at_reset_raw"].values.astype(float)
    test["_ird_true"]  = test["IRD_at_reset"].values.astype(float)

    field_colors = {
        name: color for name, color in zip(
            sorted(test["field_name"].dropna().unique()),
            ["steelblue", "seagreen", "tomato", "orange", "mediumpurple"],
        )
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Model 2 — Chrono test split: actual vs predicted IRD_at_reset (cm/h)\n"
        "Left: LightGBM  |  Right: Naive (predict no change)",
        fontsize=10, fontweight="bold",
    )

    for ax, pred_col, panel_title in [
        (axes[0], "_ird_pred",  "LightGBM"),
        (axes[1], "_ird_naive", "Naive baseline"),
    ]:
        valid = (
            test["_ird_true"].notna() & test[pred_col].notna() &
            (test["_ird_true"] > 0)   & (test[pred_col] > 0)
        )
        yt = test.loc[valid, "_ird_true"].values.astype(float)
        yp = test.loc[valid, pred_col].values.astype(float)
        m  = _metrics_raw(yt, yp, verbose=False)

        for field, color in field_colors.items():
            mask = (test.loc[valid, "field_name"] == field)
            if mask.any():
                ax.scatter(
                    yt[mask.values], yp[mask.values],
                    s=40, alpha=0.7, color=color,
                    label=field, zorder=3,
                )

        all_v = np.concatenate([yt, yp])
        lo = np.percentile(all_v[all_v > 0], 1) * 0.85
        hi = np.percentile(all_v[all_v > 0], 99) * 1.15
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_xlabel("Actual IRD_at_reset (cm/h)",    fontsize=9)
        ax.set_ylabel("Predicted IRD_at_reset (cm/h)", fontsize=9)
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, title="Field")

        if m:
            ax.annotate(
                f"R²={m['r2']:+.3f}\n"
                f"RMSE={m['rmse']:.3f} cm/h\n"
                f"MAPE={m['mape']:.1f}%\n"
                f"n={m['n']}",
                xy=(0.05, 0.97), xycoords="axes fraction",
                fontsize=9, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", alpha=0.90),
            )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot: SHAP beeswarm
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap(
    df:        pd.DataFrame,
    model,
    scaler:    StandardScaler,
    feat_cols: list[str],
    label:     str,
) -> None:
    """SHAP beeswarm for LightGBM on chrono test split."""
    test = df[df["split_chrono"] == "test"].dropna(
        subset=feat_cols + [TARGET_M2]
    )
    if len(test) < 5 or model is None:
        return

    Xte = scaler.transform(test[feat_cols].values)
    print(f"  Computing SHAP values — {label}...")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xte)
    except Exception as e:
        print(f"  ERROR computing SHAP: {e}")
        return

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":       feat_cols,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap":     shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n  SHAP feature importance — {label}:")
    print(f"  {'Rank':<5}  {'Feature':<30}  "
          f"{'Mean |SHAP|':>12}  {'Mean SHAP':>10}  Direction")
    print(f"  {'-'*75}")
    for i, row in shap_df.iterrows():
        direction = "→ higher recovery" if row["mean_shap"] > 0 \
                    else "→ lower recovery"
        print(
            f"  {i+1:<5}  {row['feature']:<30}  "
            f"{row['mean_abs_shap']:>12.4f}  "
            f"{row['mean_shap']:>10.4f}  {direction}"
        )

    n_sample = min(2000, len(Xte))
    idx      = np.random.default_rng(RANDOM_SEED).choice(
        len(Xte), n_sample, replace=False
    )
    shap_plot_df = pd.DataFrame(Xte[idx], columns=feat_cols)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[idx], shap_plot_df,
        show=False, plot_size=None,
    )
    plt.title(
        f"SHAP summary — LightGBM Model 2 ({label})\n"
        f"Target: {TARGET_M2} = log(IRD_at_reset[i] / IRD_at_reset[i-1])\n"
        "Each dot = one reset event  |  Red = high value, Blue = low\n"
        "Positive SHAP → higher recovery ratio vs previous reset",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results: dict) -> None:
    rows = []
    for condition, res in all_results.items():
        rows.append(dict(
            condition = condition,
            r2_ird    = res.get("r2",   np.nan),
            rmse_ird  = res.get("rmse", np.nan),
            mape      = res.get("mape", np.nan),
            n         = res.get("n",    np.nan),
        ))
    out = pd.DataFrame(rows)
    path = TABLES_DIR / "model2_results_v2.xlsx"
    out.to_excel(path, index=False)
    print(f"\n  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MODEL 2 V2 — IRD recovery prediction after tillage")
    print(f"  Target: {TARGET_M2} = log(IRD_at_reset[i] / IRD_at_reset[i-1])")
    print("=" * 65)

    df = load_data()

    all_results = {}

    # ── Condition Chrono ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CONDITION CHRONO — clean basins, chronological split")
    print(f"{'='*65}")

    naive_chrono = compute_naive(df, "split_chrono", "test")
    test_raw_c, test_log_c, model_c, scaler_c, cols_c = train_and_evaluate(
        df, "split_chrono", "test", "Chrono"
    )
    all_results["chrono"] = test_raw_c

    # ── Condition Held-out D ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CONDITION HELD-OUT D — clean basins train, 5 unseen basins test")
    print(f"{'='*65}")

    # For condition D: use split_held_out with clean basins only
    df_d = df[df["basin_role"].isin(["clean", "held_out"])].copy()
    naive_d = compute_naive(df_d, "split_held_out", "held_out_test")
    test_raw_d, test_log_d, model_d, scaler_d, cols_d = train_and_evaluate(
        df_d, "split_held_out", "held_out_test", "Held-out D"
    )
    all_results["held_out_D"] = test_raw_d

    # ── Condition Held-out E ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CONDITION HELD-OUT E — all 45 basins train, 5 unseen basins test")
    print(f"{'='*65}")

    # For condition E: use split_held_out with all non-excluded basins
    # For condition E: all 45 non-held-out basins including outliers
    # Outlier basins were tagged 'excluded' in split_held_out —
    # reassign them to their chrono splits so they contribute to training
    df_e = df.copy()
    outlier_mask = (
            (df_e["basin_role"] == "outlier") &
            (df_e["split_held_out"] == "excluded")
    )
    df_e.loc[outlier_mask, "split_held_out"] = df_e.loc[
        outlier_mask, "split_chrono"
    ]
    # Verify
    n_outlier_train = int(
        (df_e["basin_role"] == "outlier").sum()
    )
    print(f"  Condition E: {n_outlier_train} outlier basin rows "
          f"reassigned from excluded → chrono splits")
    naive_e = compute_naive(df_e, "split_held_out", "held_out_test")
    test_raw_e, test_log_e, model_e, scaler_e, cols_e = train_and_evaluate(
        df_e, "split_held_out", "held_out_test", "Held-out E"
    )
    all_results["held_out_E"] = test_raw_e

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SUMMARY — all conditions vs naive baseline")
    print(f"{'='*65}")
    print(f"  {'Condition':<20}  {'R²(IRD)':>8}  "
          f"{'RMSE':>8}  {'MAPE%':>7}  {'n':>6}")
    print(f"  {'-'*55}")

    for lbl, m in [
        ("Naive (chrono)",    naive_chrono),
        ("Chrono",            test_raw_c),
        ("Naive (D)",         naive_d),
        ("Held-out D",        test_raw_d),
        ("Naive (E)",         naive_e),
        ("Held-out E",        test_raw_e),
    ]:
        if not m:
            continue
        print(
            f"  {lbl:<20}  "
            f"{m.get('r2',   np.nan):>+8.4f}  "
            f"{m.get('rmse', np.nan):>8.4f}  "
            f"{m.get('mape', np.nan):>6.1f}%  "
            f"{m.get('n',    0):>6}"
        )

    save_results(all_results)

    # ── Per-basin held-out tables ────────────────────────────────────────────
    if model_d is not None:
        print_held_out_summary(
            df_d, model_d, scaler_d, cols_d,
            "split_held_out", "held_out_test", "Condition D"
        )
    if model_e is not None:
        print_held_out_summary(
            df_e, model_e, scaler_e, cols_e,
            "split_held_out", "held_out_test", "Condition E"
        )

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\n--- Figure: chrono scatter ---")
    if model_c is not None:
        plot_chrono_scatter(df, model_c, scaler_c, cols_c, naive_chrono)

    print("\n--- Figure: held-out D scatter ---")
    if model_d is not None:
        plot_held_out_scatter(
            df_d, model_d, scaler_d, cols_d,
            "split_held_out", "held_out_test",
            "Condition D (clean train, 5 unseen basins test)",
        )

    print("\n--- Figure: held-out D time series ---")
    if model_d is not None:
        plot_held_out_timeseries(
            df_d, model_d, scaler_d, cols_d,
            "split_held_out", "held_out_test",
            "Condition D (clean train, 5 unseen basins test)",
        )

    print("\n--- Figure: held-out E scatter ---")
    if model_e is not None:
        plot_held_out_scatter(
            df_e, model_e, scaler_e, cols_e,
            "split_held_out", "held_out_test",
            "Condition E (all-data train, 5 unseen basins test)",
        )

    print("\n--- Figure: held-out E time series ---")
    if model_e is not None:
        plot_held_out_timeseries(
            df_e, model_e, scaler_e, cols_e,
            "split_held_out", "held_out_test",
            "Condition E (all-data train, 5 unseen basins test)",
        )

    print("\n--- Figure: SHAP (chrono model) ---")
    if model_c is not None:
        plot_shap(df, model_c, scaler_c, cols_c, "Chrono")

    print("\nDone.")
    print(f"\n  PRIMARY RESULT (Chrono test, raw IRD cm/h):")
    print(f"    R²   = {test_raw_c.get('r2',   np.nan):+.4f}")
    print(f"    RMSE = {test_raw_c.get('rmse', np.nan):.4f} cm/h")
    print(f"    MAPE = {test_raw_c.get('mape', np.nan):.1f}%")
    print(f"    vs Naive R² = {naive_chrono.get('r2', np.nan):+.4f}")

    return model_c, scaler_c, cols_c


if __name__ == "__main__":
    main()