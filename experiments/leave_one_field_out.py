"""
experiments/leave_one_field_out.py — Leave-one-field-out cross-validation
==========================================================================
Trains on 4 fields, tests on the 5th. Repeated for all 5 fields.

Results are printed to console and shown with plt.show().
Nothing is saved to disk — this is a diagnostic/exploration script.

For each held-out field:
  - Model 1 and Model 2 are trained on the 4 remaining fields
  - Results printed per basin within the held-out field
  - For each basin: 2×2 figure
      [0,0] Model 1 time series   [0,1] Model 1 scatter
      [1,0] Model 2 time series   [1,1] Model 2 scatter

Usage
-----
  python -m experiments.leave_one_field_out

CONFIG
------
  Edit the CONFIG block below.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, RESET_CSV, OUTLIER_CSV,
    BOOSTING_PARAMS_M1, EARLY_STOPPING_ROUNDS_M1,
    BOOSTING_PARAMS_M2, EARLY_STOPPING_ROUNDS_M2,
    FIELD_NAMES, FIELD_COLORS,
)
from pipeline.features import prepare_features, TARGET_M1, MODEL2_FEATURES, TARGET_M2

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here
# ══════════════════════════════════════════════════════════════════════════════
SHOW_PLOTS   = False    # False = skip all plt.show() calls (faster, console only)
FIELDS_ORDER = [3, 4, 5, 6, 7]   # order to run leave-one-field-out
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    if mask.sum() < 2:
        return {}
    yt, yp = y_true[mask], y_pred[mask]
    rmse   = float(np.sqrt(mean_squared_error(yt, yp)))
    return dict(
        r2   = round(float(r2_score(yt, yp)), 3),
        rmse = round(rmse, 3),
        mape = round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 1),
        n    = int(mask.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_outlier_basins() -> set[int]:
    if not OUTLIER_CSV.exists():
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
    return excluded


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # Event dataset
    ev = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    ev = ev.loc[:, ~ev.columns.duplicated()]
    if TARGET_M1 not in ev.columns and "IRD_norm" in ev.columns:
        ev[TARGET_M1] = ev["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in ev.columns:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")
    ev, feat_cols_m1 = prepare_features(ev)

    # Reset dataset
    rs = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    rs = rs.loc[:, ~rs.columns.duplicated()]

    print(f"  Event dataset : {len(ev):,} rows  {ev['basin_number'].nunique()} basins")
    print(f"  Reset dataset : {len(rs)} rows  {rs['basin_number'].nunique()} basins")
    return ev, rs, feat_cols_m1


# ─────────────────────────────────────────────────────────────────────────────
# Train Model 1
# ─────────────────────────────────────────────────────────────────────────────

def train_m1(
    event_df:      pd.DataFrame,
    feat_cols:     list[str],
    train_field:   set[int],   # field digits in training set
    test_field:    int,        # field digit held out
) -> tuple[object, StandardScaler, list[str]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    required = feat_cols + [TARGET_M1]

    train_df = event_df[
        event_df["field_digit"].isin(train_field) &
        (event_df["row_type"] == "event")
    ].dropna(subset=required).reset_index(drop=True)

    val_df = event_df[
        event_df["field_digit"].isin(train_field) &
        (event_df["row_type"] == "event") &
        (event_df["split"] == "val")
    ].dropna(subset=required).reset_index(drop=True)

    if len(val_df) < 10:
        # fallback: use last 15% of train chronologically as val
        n_val  = max(10, int(len(train_df) * 0.15))
        val_df = train_df.tail(n_val).copy()

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train_df[feat_cols].values)
    Xva = sc.transform(val_df[feat_cols].values)

    model = LGBMRegressor(**BOOSTING_PARAMS_M1)
    model.fit(
        Xtr, train_df[TARGET_M1].values,
        eval_set=[(Xva, val_df[TARGET_M1].values)],
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS_M1, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"    M1 best_iter={model.best_iteration_}  "
          f"train={len(train_df):,}  val={len(val_df):,}")
    return model, sc, feat_cols


def predict_m1(
    model, sc: StandardScaler, feat_cols: list[str],
    event_df: pd.DataFrame, test_field: int,
) -> pd.DataFrame:
    test = event_df[
        (event_df["field_digit"] == test_field) &
        (event_df["row_type"] == "event")
    ].copy()
    test, _ = prepare_features(test)

    valid      = test[feat_cols].notna().all(axis=1)
    pred_norm  = np.full(len(test), np.nan)
    if valid.sum() > 0:
        pred_norm[valid.values] = model.predict(
            sc.transform(test.loc[valid, feat_cols].values))

    ird_reset        = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    test["ird_pred"] = ird_reset * np.exp(pred_norm)
    test["ird_true"] = ird_reset * np.exp(
        pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    test["ird_naive"] = ird_reset
    return test


# ─────────────────────────────────────────────────────────────────────────────
# Train Model 2
# ─────────────────────────────────────────────────────────────────────────────

def train_m2(
    reset_df:    pd.DataFrame,
    feat_cols:   list[str],
    train_field: set[int],
) -> tuple[object, StandardScaler]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    required = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]

    train_df = reset_df[
        reset_df["field_digit"].isin(train_field)
    ].dropna(subset=required).reset_index(drop=True)

    val_df = reset_df[
        reset_df["field_digit"].isin(train_field) &
        (reset_df["split_chrono"] == "val")
    ].dropna(subset=required).reset_index(drop=True)

    if len(val_df) < 5:
        n_val  = max(5, int(len(train_df) * 0.15))
        val_df = train_df.tail(n_val).copy()

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train_df[feat_cols].values)
    Xva = sc.transform(val_df[feat_cols].values)

    model = LGBMRegressor(**BOOSTING_PARAMS_M2)
    model.fit(
        Xtr, train_df[TARGET_M2].values,
        eval_set=[(Xva, val_df[TARGET_M2].values)],
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS_M2, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"    M2 best_iter={model.best_iteration_}  "
          f"train={len(train_df):,}  val={len(val_df):,}")
    return model, sc


def predict_m2(
    model, sc: StandardScaler, feat_cols: list[str],
    reset_df: pd.DataFrame, test_field: int,
) -> pd.DataFrame:
    test     = reset_df[reset_df["field_digit"] == test_field].copy()
    required = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    test     = test.dropna(subset=required).reset_index(drop=True)

    Xte              = sc.transform(test[feat_cols].values)
    prev_ird         = test["prev_IRD_at_reset_raw"].values.astype(float)
    test["ird_pred"] = prev_ird * np.exp(model.predict(Xte))
    test["ird_true"] = test["IRD_at_reset"].values.astype(float)
    test["ird_naive"]= prev_ird
    return test


# ─────────────────────────────────────────────────────────────────────────────
# Add field_digit column
# ─────────────────────────────────────────────────────────────────────────────

def add_field_digit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["field_digit"] = df["basin_number"].apply(
        lambda bn: int(str(int(bn))[0]) if pd.notna(bn) else np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Print results for one field
# ─────────────────────────────────────────────────────────────────────────────

def print_field_results(
    field:    int,
    m1_test:  pd.DataFrame,
    m2_test:  pd.DataFrame,
) -> dict:
    field_name = FIELD_NAMES.get(field, str(field))
    basins     = sorted(m1_test["basin_number"].dropna().unique().astype(int))

    print(f"\n  {'='*65}")
    print(f"  HELD-OUT FIELD: {field_name}  ({len(basins)} basins)")
    print(f"  {'='*65}")

    # Model 1
    print(f"\n  Model 1:")
    print(f"  {'Basin':>7}  {'n':>6}  {'R²':>8}  {'RMSE':>8}  "
          f"{'MAPE%':>7}  {'Naive RMSE':>11}  Beat?")
    print(f"  {'-'*60}")
    m1_all_t, m1_all_p, m1_all_n = [], [], []
    m1_basin_metrics   = {}
    for bn in basins:
        bdf   = m1_test[m1_test["basin_number"] == bn]
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        m     = _metrics(ird_t[mask], ird_p[mask])
        mn    = _metrics(ird_t[mask], ird_n[mask])
        beat  = "✓" if m.get("rmse", np.inf) < mn.get("rmse", np.inf) else "✗"
        m1_basin_metrics[bn] = {**m, "rmse_naive": mn.get("rmse", np.nan)}
        m1_all_t.append(ird_t[mask])
        m1_all_p.append(ird_p[mask])
        m1_all_n.append(ird_n[mask])   # same mask — lengths always match
        print(f"  {bn:>7}  {m.get('n',0):>6}  {m.get('r2',np.nan):>+8.3f}  "
              f"{m.get('rmse',np.nan):>8.3f}  {m.get('mape',np.nan):>6.1f}%  "
              f"{mn.get('rmse',np.nan):>10.3f}  {beat}")
    if m1_all_t:
        m1_pool       = _metrics(np.concatenate(m1_all_t), np.concatenate(m1_all_p))
        m1_naive_pool = _metrics(np.concatenate(m1_all_t), np.concatenate(m1_all_n))
        m1_delta      = m1_naive_pool.get("rmse", np.nan) - m1_pool.get("rmse", np.nan)
        m1_pool["rmse_naive"] = m1_naive_pool.get("rmse", np.nan)
        m1_pool["delta_rmse"] = m1_delta
        beats_m1 = sum(1 for v in m1_basin_metrics.values()
                       if v.get("rmse", np.inf) < v.get("rmse_naive", np.inf))
        print(f"  {'POOLED':>7}  {m1_pool.get('n',0):>6}  "
              f"{m1_pool.get('r2',np.nan):>+8.3f}  "
              f"{m1_pool.get('rmse',np.nan):>8.3f}  "
              f"{m1_pool.get('mape',np.nan):>6.1f}%  "
              f"{m1_naive_pool.get('rmse',np.nan):>10.3f}")
        print(f"  RMSE improvement over naive: {m1_delta:+.3f} cm/h  "
              f"| Beats naive: {beats_m1}/{len(basins)} basins")
    else:
        m1_pool = {}

    # Model 2
    basins_m2 = sorted(m2_test["basin_number"].dropna().unique().astype(int))
    print(f"\n  Model 2:")
    print(f"  {'Basin':>7}  {'n':>5}  {'R²':>8}  {'RMSE':>8}  "
          f"{'MAPE%':>7}  {'Naive RMSE':>11}  Beat?")
    print(f"  {'-'*60}")
    m2_all_t, m2_all_p, m2_all_n = [], [], []
    m2_basin_metrics = {}
    for bn in basins_m2:
        bdf   = m2_test[m2_test["basin_number"] == bn]
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        m     = _metrics(ird_t[mask], ird_p[mask])
        mn    = _metrics(ird_t[mask], ird_n[mask])
        beat  = "✓" if m.get("rmse", np.inf) < mn.get("rmse", np.inf) else "✗"
        m2_basin_metrics[bn] = {**m, "rmse_naive": mn.get("rmse", np.nan)}
        m2_all_t.append(ird_t[mask])
        m2_all_p.append(ird_p[mask])
        m2_all_n.append(ird_n[mask])
        print(f"  {bn:>7}  {m.get('n',0):>5}  {m.get('r2',np.nan):>+8.3f}  "
              f"{m.get('rmse',np.nan):>8.3f}  {m.get('mape',np.nan):>6.1f}%  "
              f"{mn.get('rmse',np.nan):>10.3f}  {beat}")
    if m2_all_t:
        m2_pool       = _metrics(np.concatenate(m2_all_t), np.concatenate(m2_all_p))
        m2_naive_pool = _metrics(np.concatenate(m2_all_t), np.concatenate(m2_all_n))
        delta = m2_naive_pool.get("rmse", np.nan) - m2_pool.get("rmse", np.nan)
        m2_pool["rmse_naive"] = m2_naive_pool.get("rmse", np.nan)
        m2_pool["delta_rmse"] = delta
        beats_m2 = sum(1 for v in m2_basin_metrics.values()
                       if v.get("rmse", np.inf) < v.get("rmse_naive", np.inf))
        print(f"  {'POOLED':>7}  {m2_pool.get('n',0):>5}  "
              f"{m2_pool.get('r2',np.nan):>+8.3f}  "
              f"{m2_pool.get('rmse',np.nan):>8.3f}  "
              f"{m2_pool.get('mape',np.nan):>6.1f}%  "
              f"{m2_naive_pool.get('rmse',np.nan):>10.3f}")
        print(f"  RMSE improvement over naive: {delta:+.3f} cm/h  "
              f"| Beats naive: {beats_m2}/{len(basins_m2)} basins")
    else:
        m2_pool = {}

    return {
        "field": field, "field_name": field_name,
        "m1_pool": m1_pool, "m2_pool": m2_pool,
        "m1_basin": m1_basin_metrics, "m2_basin": m2_basin_metrics,
        "basins_m1": basins, "basins_m2": basins_m2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot: 2×2 grid per basin — called once per basin
# ─────────────────────────────────────────────────────────────────────────────

def plot_basin(
    bn:       int,
    field:    int,
    m1_test:  pd.DataFrame,
    m2_test:  pd.DataFrame,
    m1_metrics: dict,
    m2_metrics: dict,
) -> None:
    from plot_style import add_season_bands
    import plot_style as _ps

    field_name = FIELD_NAMES.get(field, str(field))
    color      = FIELD_COLORS.get(field_name, "#065A82")

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Leave-one-field-out | Field: {field_name} | Basin {bn}",
        fontsize=11, fontweight="bold",
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)
    ax_m1_ts = fig.add_subplot(gs[0, 0])
    ax_m1_sc = fig.add_subplot(gs[0, 1])
    ax_m2_ts = fig.add_subplot(gs[1, 0])
    ax_m2_sc = fig.add_subplot(gs[1, 1])

    # ── Model 1 time series ───────────────────────────────────────────────────
    bdf1  = m1_test[m1_test["basin_number"] == bn].sort_values("opening_valve_date")
    ird_t = bdf1["ird_true"].values.astype(float)
    ird_p = bdf1["ird_pred"].values.astype(float)
    mask1 = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
    m1    = m1_metrics.get(bn, {})

    if len(bdf1) > 0:
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, 0.55)
        add_season_bands(ax_m1_ts,
                         bdf1["opening_valve_date"].min(),
                         bdf1["opening_valve_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        # Reset lines
        for sid in sorted(bdf1["segment_id"].dropna().unique()):
            seg = bdf1[bdf1["segment_id"] == sid]
            if not seg.empty:
                ax_m1_ts.axvline(seg["opening_valve_date"].iloc[0],
                                 color="black", linewidth=0.5,
                                 linestyle="--", alpha=0.20, zorder=1)

        dates = bdf1["opening_valve_date"].values
        ax_m1_ts.scatter(dates[mask1], ird_t[mask1],
                         s=10, alpha=0.70, color="#065A82",
                         marker="o", zorder=4)
        ax_m1_ts.scatter(dates[mask1], ird_p[mask1],
                         s=10, alpha=0.60, color="#E07B39",
                         marker="x", linewidths=1.2, zorder=5)
        for d, yt, yp in zip(dates[mask1], ird_t[mask1], ird_p[mask1]):
            ax_m1_ts.plot([d, d], [yt, yp], color="gray",
                          linewidth=0.35, alpha=0.25, zorder=2)

    ax_m1_ts.set_title(
        f"Model 1 — IRD time series\n"
        f"R²={m1.get('r2',np.nan):+.3f}  "
        f"RMSE={m1.get('rmse',np.nan):.3f} cm/h  "
        f"MAPE={m1.get('mape',np.nan):.1f}%  n={m1.get('n',0):,}",
        fontsize=9,
    )
    ax_m1_ts.set_ylabel("IRD (cm/h)", fontsize=9)
    ax_m1_ts.set_xlabel("Date", fontsize=9)
    ax_m1_ts.tick_params(labelsize=8)
    ax_m1_ts.tick_params(axis="x", rotation=25)
    ax_m1_ts.legend(
        handles=[
            Line2D([0],[0], marker="o", color="#065A82", markersize=5,
                   linestyle="None", label="Actual"),
            Line2D([0],[0], marker="x", color="#E07B39", markersize=5,
                   linestyle="None", markeredgewidth=1.2, label="Predicted"),
        ],
        fontsize=8, loc="upper right", framealpha=0.85,
    )

    # ── Model 1 scatter ───────────────────────────────────────────────────────
    yt1 = ird_t[mask1]
    yp1 = ird_p[mask1]
    if len(yt1) >= 2:
        lo = min(yt1.min(), yp1.min()) * 0.90
        hi = max(yt1.max(), yp1.max()) * 1.05
        ax_m1_sc.scatter(yt1, yp1, s=12, alpha=0.65,
                         color=color, edgecolors="none", zorder=3)
        ax_m1_sc.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.6)
        ax_m1_sc.set_xlim(lo, hi); ax_m1_sc.set_ylim(lo, hi)
        ax_m1_sc.set_aspect("equal", adjustable="box")
    ax_m1_sc.set_title("Model 1 — Actual vs Predicted", fontsize=9)
    ax_m1_sc.set_xlabel("Actual IRD (cm/h)", fontsize=9)
    ax_m1_sc.set_ylabel("Predicted (cm/h)", fontsize=9)
    ax_m1_sc.tick_params(labelsize=8)
    ax_m1_sc.grid(True, alpha=0.2)

    # ── Model 2 time series ───────────────────────────────────────────────────
    bdf2   = m2_test[m2_test["basin_number"] == bn].sort_values("reset_date")
    m2     = m2_metrics.get(bn, {})
    ird_t2 = bdf2["ird_true"].values.astype(float)
    ird_p2 = bdf2["ird_pred"].values.astype(float)
    ird_n2 = bdf2["ird_naive"].values.astype(float)
    mask2  = (np.isfinite(ird_t2) & np.isfinite(ird_p2) &
              (ird_t2 > 0) & (ird_p2 > 0))

    if len(bdf2) > 0:
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, 0.55)
        add_season_bands(ax_m2_ts,
                         bdf2["reset_date"].min(),
                         bdf2["reset_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        dates2 = bdf2["reset_date"].values
        ax_m2_ts.plot(dates2[mask2], ird_n2[mask2],
                      color="lightgrey", linewidth=1.0, alpha=0.9, zorder=2)
        ax_m2_ts.scatter(dates2[mask2], ird_t2[mask2],
                         s=35, alpha=0.80, color="#065A82",
                         marker="o", zorder=4)
        ax_m2_ts.scatter(dates2[mask2], ird_p2[mask2],
                         s=35, alpha=0.80, color="#E07B39",
                         marker="x", linewidths=1.5, zorder=5)
        for d, yt, yp in zip(dates2[mask2], ird_t2[mask2], ird_p2[mask2]):
            ax_m2_ts.plot([d, d], [yt, yp], color="gray",
                          linewidth=0.5, alpha=0.30, zorder=3)

    rmse_nv = m2.get("rmse_naive", np.nan)
    ax_m2_ts.set_title(
        f"Model 2 — IRD_reset time series\n"
        f"R²={m2.get('r2',np.nan):+.3f}  "
        f"RMSE={m2.get('rmse',np.nan):.3f} cm/h  "
        f"Naive={rmse_nv:.3f}  "
        f"MAPE={m2.get('mape',np.nan):.1f}%  n={m2.get('n',0)}",
        fontsize=9,
    )
    ax_m2_ts.set_ylabel("IRD_reset (cm/h)", fontsize=9)
    ax_m2_ts.set_xlabel("Date", fontsize=9)
    ax_m2_ts.tick_params(labelsize=8)
    ax_m2_ts.tick_params(axis="x", rotation=25)
    ax_m2_ts.legend(
        handles=[
            Line2D([0],[0], marker="o", color="#065A82", markersize=5,
                   linestyle="None", label="Actual"),
            Line2D([0],[0], marker="x", color="#E07B39", markersize=5,
                   linestyle="None", markeredgewidth=1.5, label="Predicted"),
            Line2D([0],[0], color="lightgrey", linewidth=1.5, label="Naive"),
        ],
        fontsize=8, loc="upper right", framealpha=0.85,
    )

    # ── Model 2 scatter ───────────────────────────────────────────────────────
    yt2 = ird_t2[mask2]
    yp2 = ird_p2[mask2]
    yn2 = ird_n2[mask2]
    if len(yt2) >= 2:
        all_v = np.concatenate([yt2, yp2, yn2])
        lo = all_v.min() * 0.88; hi = all_v.max() * 1.08
        ax_m2_sc.scatter(yt2, yn2, s=22, alpha=0.35,
                         color="grey", marker="^", zorder=2, label="Naive")
        ax_m2_sc.scatter(yt2, yp2, s=28, alpha=0.75,
                         color=color, edgecolors="none", zorder=3, label="Model 2")
        ax_m2_sc.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.6)
        ax_m2_sc.set_xlim(lo, hi); ax_m2_sc.set_ylim(lo, hi)
        ax_m2_sc.set_aspect("equal", adjustable="box")
        ax_m2_sc.legend(fontsize=8, loc="lower right")
    ax_m2_sc.set_title("Model 2 — Actual vs Predicted", fontsize=9)
    ax_m2_sc.set_xlabel("Actual IRD_reset (cm/h)", fontsize=9)
    ax_m2_sc.set_ylabel("Predicted (cm/h)", fontsize=9)
    ax_m2_sc.tick_params(labelsize=8)
    ax_m2_sc.grid(True, alpha=0.2)

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary across all fields
# ─────────────────────────────────────────────────────────────────────────────

def print_overall_summary(all_field_results: list[dict]) -> None:
    print(f"\n{'='*80}")
    print("  LEAVE-ONE-FIELD-OUT SUMMARY")
    print(f"{'='*80}")

    # ── Model 1 ───────────────────────────────────────────────────────────────
    print(f"\n  Model 1 — pooled per held-out field:")
    print(f"  {'Field':<12}  {'n':>6}  {'R²':>8}  {'RMSE':>8}  "
          f"{'Naive RMSE':>11}  {'ΔRMSE':>8}  {'MAPE%':>7}")
    print(f"  {'-'*68}")
    m1_all_rmse, m1_all_delta = [], []
    for r in all_field_results:
        m     = r["m1_pool"]
        rmse  = m.get("rmse",       np.nan)
        naive = m.get("rmse_naive", np.nan)
        delta = m.get("delta_rmse", np.nan)
        m1_all_rmse.append(rmse)
        m1_all_delta.append(delta)
        print(f"  {r['field_name']:<12}  {m.get('n',0):>6}  "
              f"{m.get('r2',np.nan):>+8.3f}  {rmse:>8.3f}  "
              f"{naive:>11.3f}  {delta:>+8.3f}  "
              f"{m.get('mape',np.nan):>6.1f}%")
    print(f"  {'-'*68}")
    print(f"  {'Mean':<12}  {'':>6}  {'':>8}  "
          f"{np.nanmean(m1_all_rmse):>8.3f}  "
          f"{'':>11}  {np.nanmean(m1_all_delta):>+8.3f}")

    # ── Model 2 ───────────────────────────────────────────────────────────────
    print(f"\n  Model 2 — pooled per held-out field:")
    print(f"  {'Field':<12}  {'n':>6}  {'R²':>8}  {'RMSE':>8}  "
          f"{'Naive RMSE':>11}  {'ΔRMSE':>8}  {'MAPE%':>7}")
    print(f"  {'-'*68}")
    m2_all_rmse, m2_all_delta = [], []
    for r in all_field_results:
        m     = r["m2_pool"]
        rmse  = m.get("rmse",       np.nan)
        naive = m.get("rmse_naive", np.nan)
        delta = m.get("delta_rmse", np.nan)
        m2_all_rmse.append(rmse)
        m2_all_delta.append(delta)
        print(f"  {r['field_name']:<12}  {m.get('n',0):>6}  "
              f"{m.get('r2',np.nan):>+8.3f}  {rmse:>8.3f}  "
              f"{naive:>11.3f}  {delta:>+8.3f}  "
              f"{m.get('mape',np.nan):>6.1f}%")
    print(f"  {'-'*68}")
    print(f"  {'Mean':<12}  {'':>6}  {'':>8}  "
          f"{np.nanmean(m2_all_rmse):>8.3f}  "
          f"{'':>11}  {np.nanmean(m2_all_delta):>+8.3f}")
    print()
    # Quick read: how many fields did M2 beat naive?
    m2_beats = sum(1 for r in all_field_results
                   if r["m2_pool"].get("delta_rmse", -np.inf) > 0)
    m1_beats = sum(1 for r in all_field_results
                   if r["m1_pool"].get("delta_rmse", -np.inf) > 0)
    print(f"  Model 1 beats naive in {m1_beats}/5 fields")
    print(f"  Model 2 beats naive in {m2_beats}/5 fields")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  LEAVE-ONE-FIELD-OUT CROSS-VALIDATION")
    print(f"  Fields order : {FIELDS_ORDER}")
    print(f"  Show plots   : {SHOW_PLOTS}")
    print("  Nothing saved to disk.")
    print("=" * 70)

    print("\n--- Loading data ---")
    outlier_basins       = load_outlier_basins()
    event_df, reset_df, feat_cols_m1 = load_data()
    feat_cols_m2         = [f for f in MODEL2_FEATURES if f in reset_df.columns]

    # Add field_digit to both datasets
    event_df = add_field_digit(event_df)
    reset_df = add_field_digit(reset_df)

    all_field_results = []

    for test_field in FIELDS_ORDER:
        train_fields = set(FIELDS_ORDER) - {test_field}
        field_name   = FIELD_NAMES.get(test_field, str(test_field))

        print(f"\n{'='*70}")
        print(f"  FOLD: hold out {field_name}  |  "
              f"train on {[FIELD_NAMES.get(f,str(f)) for f in sorted(train_fields)]}")
        print(f"{'='*70}")

        # Train
        print(f"\n  Training...")
        m1_model, m1_sc, used_m1 = train_m1(
            event_df, feat_cols_m1, train_fields, test_field)
        m2_model, m2_sc           = train_m2(
            reset_df, feat_cols_m2, train_fields)

        # Predict
        m1_test = predict_m1(m1_model, m1_sc, used_m1, event_df, test_field)
        m2_test = predict_m2(m2_model, m2_sc, feat_cols_m2, reset_df, test_field)

        # Print results
        field_result = print_field_results(test_field, m1_test, m2_test)
        all_field_results.append(field_result)

        # Per-basin plots
        if SHOW_PLOTS:
            basins = sorted(
                set(field_result["basins_m1"]) | set(field_result["basins_m2"]))
            print(f"\n  Plotting {len(basins)} basins "
                  f"(one 2×2 figure per basin)...")
            for bn in basins:
                print(f"    Basin {bn} ...", end=" ", flush=True)
                plot_basin(
                    bn          = bn,
                    field       = test_field,
                    m1_test     = m1_test,
                    m2_test     = m2_test,
                    m1_metrics  = field_result["m1_basin"],
                    m2_metrics  = field_result["m2_basin"],
                )
                print("done")

    print_overall_summary(all_field_results)
    print("\nDone.")


if __name__ == "__main__":
    main()