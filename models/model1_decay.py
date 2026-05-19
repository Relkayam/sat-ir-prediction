"""
models/model1_decay.py — Model 1: within-segment IRD decay prediction
======================================================================
Reads held-out basins from data/selected_basins.csv (written by bootstrap).
Trains E-full configuration and evaluates on those 5 basins.

Run experiments/run_bootstrap.py first to generate selected_basins.csv.

Usage
-----
  python -m models.model1_decay
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, OUTLIER_CSV, SELECTED_BASINS_CSV, TABLES_DIR,
    FIELD_NAMES, BOOSTING_PARAMS_M1, EARLY_STOPPING_ROUNDS_M1,
)
from pipeline.features import prepare_features, TARGET_M1
from models.utils import metrics_ird, back_transform, predict, per_basin_median_r2

FIELD_COLORS = {
    "Soreq 2": "#065A82", "Yavne 1": "#1C7293",
    "Yavne 2": "#E07B39", "Yavne 3": "#27AE60", "Yavne 4": "#7D3C98",
}
DEFAULT_COLOR = "#065A82"
PRED_COLOR    = "#AAAAAA"


# ─────────────────────────────────────────────────────────────────────────────
# Load selected basins
# ─────────────────────────────────────────────────────────────────────────────

def load_selected_basins() -> list[int]:
    """Read held-out basin list from data/selected_basins.csv."""
    if not SELECTED_BASINS_CSV.exists():
        raise FileNotFoundError(
            f"{SELECTED_BASINS_CSV} not found.\n"
            "Run: python -m experiments.run_bootstrap"
        )
    basins = []
    with open(SELECTED_BASINS_CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("basin"):
                continue
            try:
                basins.append(int(line.split(",")[0].strip()))
            except ValueError:
                pass
    print(f"  Selected held-out basins ({len(basins)}): {basins}")
    return basins


def load_outlier_basins() -> set[int]:
    if not OUTLIER_CSV.exists():
        print(f"  WARNING: {OUTLIER_CSV} not found.")
        return set()
    excluded = set()
    with open(OUTLIER_CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("basin"):
                continue
            try:
                excluded.add(int(line.split(",")[0].strip()))
            except ValueError:
                pass
    print(f"  Outlier basins ({len(excluded)}): {sorted(excluded)}")
    return excluded


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    if not EVENT_CSV.exists():
        raise FileNotFoundError(f"{EVENT_CSV} not found.")
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df, feat_cols = prepare_features(df)
    print(f"  Event dataset : {len(df):,} rows  "
          f"{df['basin_number'].nunique()} basins")
    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Prepare splits
# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(
    df:             pd.DataFrame,
    held_out:       list[int],
    feat_cols:      list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    E-full: train on ALL non-held-out basins (includes outliers).
    Val: good segments, split=='val', non-held-out.
    Test: held-out basins, good segments.
    """
    held_set = set(held_out)
    required = feat_cols + [TARGET_M1]

    test = df[
        df["basin_number"].isin(held_set) &
        (df["row_type"]        == "event") &
        (df["is_good_segment"] == True)
    ].dropna(subset=required).reset_index(drop=True)

    non_ho = df[
        ~df["basin_number"].isin(held_set) &
        (df["row_type"] == "event")
    ]

    train = non_ho.dropna(subset=required).reset_index(drop=True)
    val   = non_ho[
        (non_ho["split"]          == "val") &
        (non_ho["is_good_segment"] == True)
    ].dropna(subset=required).reset_index(drop=True)

    print(f"  train={len(train):,} ({non_ho['basin_number'].nunique()} basins)  "
          f"val={len(val):,}  test={len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    train: pd.DataFrame, val: pd.DataFrame, feat_cols: list[str]
) -> tuple[object, object]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.preprocessing import StandardScaler

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)

    model = LGBMRegressor(**BOOSTING_PARAMS_M1)
    model.fit(
        Xtr, train[TARGET_M1].values,
        eval_set=[(Xva, val[TARGET_M1].values)],
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS_M1, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"  best_iter={model.best_iteration_}")
    return model, sc


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_held_out(
    test:      pd.DataFrame,
    model,
    scaler,
    feat_cols: list[str],
    held_out:  list[int],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    test      = test.copy()
    ird_reset = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    pred_norm = predict(model, scaler, feat_cols, test)
    test["ird_pred"] = back_transform(ird_reset, pred_norm)
    test["ird_true"] = back_transform(
        ird_reset, pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    test["ird_naive"]= back_transform(ird_reset, np.zeros(len(test)))

    print(f"\n{'='*65}")
    print("  HELD-OUT RESULTS — Model 1 (E-full)")
    print(f"{'='*65}")
    print(f"  {'Basin':>7}  {'Field':<12}  {'n':>6}  "
          f"{'R²':>8}  {'RMSE':>8}  {'MAPE%':>7}  "
          f"{'Naive RMSE':>11}  Beat?")
    print(f"  {'-'*65}")

    rows = []; all_true = []; all_pred = []
    for bn in held_out:
        bdf   = test[test["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        mask  = (np.isfinite(ird_t) & np.isfinite(ird_p) &
                 (ird_t > 0) & (ird_p > 0))
        all_true.append(ird_t[mask]); all_pred.append(ird_p[mask])
        m  = metrics_ird(ird_t[mask], ird_p[mask], verbose=False) or {}
        mn = metrics_ird(ird_t[mask], ird_n[mask], verbose=False) or {}
        beat = "✓" if m.get("rmse", np.inf) < mn.get("rmse", np.inf) else "✗"
        print(f"  {bn:>7}  {field:<12}  "
              f"{m.get('n',0):>6}  {m.get('r2',np.nan):>+8.3f}  "
              f"{m.get('rmse',np.nan):>8.3f}  {m.get('mape',np.nan):>6.1f}%  "
              f"{mn.get('rmse',np.nan):>10.3f}  {beat}")
        rows.append(dict(basin=bn, field=field, n=m.get("n",0),
                         r2=m.get("r2",np.nan), rmse=m.get("rmse",np.nan),
                         mape=m.get("mape",np.nan),
                         rmse_naive=mn.get("rmse",np.nan), beats_naive=beat))

    ird_t_all = np.concatenate(all_true)
    ird_p_all = np.concatenate(all_pred)
    m_all = metrics_ird(ird_t_all, ird_p_all, verbose=False) or {}
    print(f"  {'-'*65}")
    print(f"  {'POOLED':>7}  {'all':12}  "
          f"{m_all.get('n',0):>6}  {m_all.get('r2',np.nan):>+8.3f}  "
          f"{m_all.get('rmse',np.nan):>8.3f}  {m_all.get('mape',np.nan):>6.1f}%")

    basin_med = per_basin_median_r2(
        test, predict(model, scaler, feat_cols, test), TARGET_M1)
    print(f"\n  Per-basin median R²: {basin_med:+.4f}")

    return m_all, pd.DataFrame(rows), test


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(test: pd.DataFrame, held_out: list[int]) -> None:
    from plot_style import add_season_bands
    import plot_style as _ps

    n   = len(held_out)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8*n),
                             gridspec_kw={"hspace": 0.35})
    if n == 1: axes = [axes]

    fig.suptitle(
        "Model 1 — Held-out time series (E-full)\n"
        "Coloured = actual  |  Grey crosses = predicted",
        fontsize=10, fontweight="bold",
    )
    for ax, bn in zip(axes, held_out):
        bdf   = test[test["basin_number"]==bn].sort_values("opening_valve_date")
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        color = FIELD_COLORS.get(field, DEFAULT_COLOR)
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        dates = bdf["opening_valve_date"].values
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t>0) & (ird_p>0)
        m     = metrics_ird(ird_t[mask], ird_p[mask], verbose=False) or {}

        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, 0.55)
        add_season_bands(ax, bdf["opening_valve_date"].min(),
                         bdf["opening_valve_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        for sid in sorted(bdf["segment_id"].dropna().unique()):
            seg = bdf[bdf["segment_id"]==sid].sort_values("opening_valve_date")
            if not seg.empty:
                ax.axvline(seg["opening_valve_date"].iloc[0], color="black",
                           linewidth=0.5, linestyle="--", alpha=0.20, zorder=1)

        ax.scatter(dates[mask], ird_t[mask], s=10, alpha=0.70,
                   color=color, marker="o", zorder=4)
        ax.scatter(dates[mask], ird_p[mask], s=10, alpha=0.60,
                   color=PRED_COLOR, marker="x", linewidths=1.2, zorder=5)
        for d, yt, yp in zip(dates[mask], ird_t[mask], ird_p[mask]):
            ax.plot([d,d],[yt,yp], color="gray", linewidth=0.35, alpha=0.25, zorder=2)

        ax.set_ylabel(f"Basin {bn} ({field})\nIRD (cm/h)", fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(True, alpha=0.15)
        ax.set_title(
            f"R²={m.get('r2',np.nan):+.3f}  "
            f"RMSE={m.get('rmse',np.nan):.3f} cm/h  "
            f"MAPE={m.get('mape',np.nan):.1f}%  n={m.get('n',0):,}",
            fontsize=8)
        if bn == held_out[0]:
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0],[0], marker="o", color=color, markersize=5,
                       linestyle="None", label="Actual IRD"),
                Line2D([0],[0], marker="x", color=PRED_COLOR, markersize=5,
                       linestyle="None", markeredgewidth=1.2, label="Predicted IRD"),
                Line2D([0],[0], color="black", linewidth=0.7,
                       linestyle="--", alpha=0.4, label="Tillage reset"),
            ], fontsize=8, loc="upper right", ncol=3, framealpha=0.85)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


def plot_scatter(test: pd.DataFrame, held_out: list[int]) -> None:
    n = len(held_out)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 5), squeeze=False)
    axes = axes[0]
    fig.suptitle("Model 1 — Actual vs predicted IRD (E-full)\nDashed = 1:1",
                 fontsize=10, fontweight="bold")
    for ax, bn in zip(axes, held_out):
        bdf   = test[test["basin_number"]==bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        color = FIELD_COLORS.get(field, DEFAULT_COLOR)
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t>0) & (ird_p>0)
        yt = ird_t[mask]; yp = ird_p[mask]
        m  = metrics_ird(yt, yp, verbose=False) or {}
        ax.scatter(yt, yp, s=12, alpha=0.65, color=color, edgecolors="none", zorder=3)
        lo = min(yt.min(), yp.min())*0.90; hi = max(yt.max(), yp.max())*1.05
        ax.plot([lo,hi],[lo,hi], "k--", linewidth=0.9, alpha=0.6)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Basin {bn} ({field})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Actual IRD (cm/h)", fontsize=8)
        ax.set_ylabel("Predicted (cm/h)", fontsize=8)
        ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)
        ax.annotate(
            f"R²={m.get('r2',np.nan):+.3f}\n"
            f"RMSE={m.get('rmse',np.nan):.3f}\nMAPE={m.get('mape',np.nan):.1f}%",
            xy=(0.05,0.97), xycoords="axes fraction", fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.90, edgecolor="lightgrey"))
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_results(pooled: dict, basin_df: pd.DataFrame) -> None:
    path = TABLES_DIR / "model1_results_condition_e.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([pooled]).to_excel(writer, sheet_name="pooled", index=False)
        basin_df.to_excel(writer, sheet_name="per_basin", index=False)
    print(f"\n  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  MODEL 1 — Within-segment IRD decay prediction")
    print(f"  Target  : {TARGET_M1}")
    print("  Config  : E-full (all non-held-out basins including outliers)")
    print("="*65)

    held_out = load_selected_basins()
    df, feat_cols = load_data()

    train, val, test = prepare_splits(df, held_out, feat_cols)
    model, scaler    = train_model(train, val, feat_cols)
    pooled, basin_df, test_df = evaluate_held_out(
        test, model, scaler, feat_cols, held_out)

    save_results(pooled, basin_df)

    print("\n--- Plots ---")
    plot_timeseries(test_df, held_out)
    plot_scatter(test_df, held_out)

    print("\nDone.")
    return model, scaler, feat_cols


if __name__ == "__main__":
    main()