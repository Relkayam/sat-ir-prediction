"""
models/model1_decay.py — Model 1: within-segment IRD decay prediction
======================================================================
Condition E (E-full):
  Train : non-held-out basins, event rows with split == 'train'.
          Events pass the four sensor-quality filters applied at
          build_dataset time (IRD_R² < 0.94, CIV < 3000, Ct < 20, AL < 5).
          No segment-level (is_good_segment) filter — that concept no
          longer exists in the pipeline.
  Val   : non-held-out basins, event rows with split == 'val'.
          DISJOINT from train; used solely for LightGBM early stopping.
  Test  : held-out basins, all event rows.

Baselines
---------
  Naive   : predict ρᵢ (IRD_at_reset) for every event — no decay assumed.
  Linear  : causal linear extrapolation on η (normalized IRD) vs LCT, fitted
            on prior events in the same segment, back-transformed. Undefined
            for the first two events of a segment (needs ≥2 priors); those
            events return NaN and are excluded from the linear-baseline metric.

Held-out basins read from data/selected_basins.csv (written by the bootstrap).
Run experiments/run_bootstrap_v2.py first, then point config.SELECTED_BASINS_CSV
at the bootstrap output.

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
    EVENT_CSV, SELECTED_BASINS_CSV, TABLES_DIR,
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
    """Read held-out basin list from config.SELECTED_BASINS_CSV."""
    if not SELECTED_BASINS_CSV.exists():
        raise FileNotFoundError(
            f"{SELECTED_BASINS_CSV} not found.\n"
            "Run: python -m experiments.run_bootstrap_v2  (then point "
            "config.SELECTED_BASINS_CSV at its output)"
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
    if "segment_id" in df.columns:
        df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce")
    df, feat_cols = prepare_features(df)
    print(f"  Event dataset : {len(df):,} rows  "
          f"{df['basin_number'].nunique()} basins")
    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Prepare splits — Condition E-full
# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(
    df:        pd.DataFrame,
    held_out:  list[int],
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Condition E-full — no is_good_segment filter anywhere.

    Train : non-held-out basins, event rows with split == 'train'
    Val   : non-held-out basins, event rows with split == 'val'  (DISJOINT)
    Test  : held-out basins, all event rows

    The val set shares no rows with train; it is used only for early stopping.
    """
    held_set = set(held_out)
    required = feat_cols + [TARGET_M1]

    test = df[
        df["basin_number"].isin(held_set) &
        (df["row_type"] == "event")
    ].dropna(subset=required).reset_index(drop=True)

    non_ho = df[
        ~df["basin_number"].isin(held_set) &
        (df["row_type"] == "event")
    ]

    train = non_ho[
        non_ho["split"] == "train"
    ].dropna(subset=required).reset_index(drop=True)

    val = non_ho[
        non_ho["split"] == "val"
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
# Linear extrapolation baseline
# ─────────────────────────────────────────────────────────────────────────────

def _linear_extrap_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Causal linear extrapolation baseline in η (normalized IRD) space.

    For each event at position k within a segment:
      - If ≥ 2 prior events exist in the segment:
          Fit η = α·LCT + β on the prior events, extrapolate to LCT_k,
          back-transform IRD̂ = ρᵢ · exp(η̂).
      - If < 2 prior events:
          Prediction = NaN (excluded from metric computation).

    Metrics are computed only where the baseline is defined (position ≥ 3
    within a segment). All valid predictions clipped to [0.01, 50] cm/h.
    """
    df = df.copy().sort_values(["basin_number", "segment_id", "opening_valve_date"])
    df = df.reset_index(drop=True)

    n         = len(df)
    bl        = np.full(n, np.nan)
    lct_all   = pd.to_numeric(df["LCT"],          errors="coerce").values
    eta_all   = pd.to_numeric(df[TARGET_M1],      errors="coerce").values
    reset_all = pd.to_numeric(df["IRD_at_reset"], errors="coerce").values

    for (bn, sid), grp in df.groupby(["basin_number", "segment_id"]):
        if sid < 0:
            continue

        idx   = grp.index.tolist()
        lct_g = lct_all[idx]
        eta_g = eta_all[idx]
        rho_i = float(reset_all[idx[0]]) if np.isfinite(reset_all[idx[0]]) else np.nan

        if not np.isfinite(rho_i) or rho_i <= 0:
            continue

        for pos in range(len(idx)):
            lct_k = lct_g[pos]
            if not np.isfinite(lct_k):
                continue   # leave as NaN

            prior_lct = lct_g[:pos]
            prior_eta = eta_g[:pos]
            valid = np.isfinite(prior_lct) & np.isfinite(prior_eta)

            if valid.sum() < 2:
                continue   # fewer than 2 prior valid events — leave as NaN

            coeffs  = np.polyfit(prior_lct[valid], prior_eta[valid], deg=1)
            eta_hat = np.polyval(coeffs, lct_k)
            ird_hat = rho_i * np.exp(eta_hat)
            bl[idx[pos]] = float(np.clip(ird_hat, 0.01, 50.0))

    return bl


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
    test["ird_pred"]   = back_transform(ird_reset, pred_norm)
    test["ird_true"]   = back_transform(
        ird_reset, pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    test["ird_naive"]  = ird_reset          # predict ρᵢ — no decay
    test["ird_linear"] = _linear_extrap_baseline(test)

    print(f"\n{'='*75}")
    print("  HELD-OUT RESULTS — Model 1 (E-full)")
    print(f"{'='*75}")
    print(f"  {'Basin':>7}  {'Field':<12}  {'n':>6}  "
          f"{'R²':>8}  {'RMSE':>8}  {'MAPE%':>7}  "
          f"{'Naive RMSE':>11}  {'Linear RMSE':>12}  Beat?")
    print(f"  {'-'*75}")

    rows = []; all_true = []; all_pred = []; all_lin = []; all_lin_true = []
    for bn in held_out:
        bdf   = test[test["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        ird_l = bdf["ird_linear"].values.astype(float)

        mask = (np.isfinite(ird_t) & np.isfinite(ird_p) &
                (ird_t > 0) & (ird_p > 0))
        mask_l = mask & np.isfinite(ird_l) & (ird_l > 0)

        all_true.append(ird_t[mask])
        all_pred.append(ird_p[mask])
        all_lin.append(ird_l[mask_l])
        all_lin_true.append(ird_t[mask_l])

        m  = metrics_ird(ird_t[mask],   ird_p[mask],   verbose=False) or {}
        mn = metrics_ird(ird_t[mask],   ird_n[mask],   verbose=False) or {}
        ml = metrics_ird(ird_t[mask_l], ird_l[mask_l], verbose=False) or {}
        beat = "✓" if m.get("rmse", np.inf) < mn.get("rmse", np.inf) else "✗"

        print(f"  {bn:>7}  {field:<12}  "
              f"{m.get('n',0):>6}  {m.get('r2',np.nan):>+8.3f}  "
              f"{m.get('rmse',np.nan):>8.3f}  {m.get('mape',np.nan):>6.1f}%  "
              f"{mn.get('rmse',np.nan):>10.3f}  "
              f"{ml.get('rmse',np.nan):>11.3f}  {beat}")

        rows.append(dict(
            basin=bn, field=field, n=m.get("n",0),
            r2=m.get("r2",np.nan), rmse=m.get("rmse",np.nan),
            mape=m.get("mape",np.nan),
            rmse_naive=mn.get("rmse",np.nan),
            rmse_linear=ml.get("rmse",np.nan),
            n_linear=ml.get("n",0),
            beats_naive=beat,
        ))

    ird_t_all = np.concatenate(all_true)
    ird_p_all = np.concatenate(all_pred)
    m_all = metrics_ird(ird_t_all, ird_p_all, verbose=False) or {}
    print(f"  {'-'*75}")
    print(f"  {'POOLED':>7}  {'all':12}  "
          f"{m_all.get('n',0):>6}  {m_all.get('r2',np.nan):>+8.3f}  "
          f"{m_all.get('rmse',np.nan):>8.3f}  {m_all.get('mape',np.nan):>6.1f}%")

    # Pooled linear baseline metrics — only where the baseline is defined
    ird_l_all = np.concatenate(all_lin)
    ird_t_lin = np.concatenate(all_lin_true)
    if len(ird_l_all) > 1:
        ml_all = metrics_ird(ird_t_lin, ird_l_all, verbose=False) or {}
        print(f"\n  Linear baseline (n={ml_all.get('n',0)} events where defined):")
        print(f"    RMSE={ml_all.get('rmse',np.nan):.3f} cm/h  "
              f"R²={ml_all.get('r2',np.nan):+.3f}  "
              f"MAPE={ml_all.get('mape',np.nan):.1f}%")

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
        "Coloured = actual  |  Grey crosses = predicted  |  Orange = linear baseline",
        fontsize=10, fontweight="bold",
    )
    for ax, bn in zip(axes, held_out):
        bdf   = test[test["basin_number"]==bn].sort_values("opening_valve_date")
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        color = FIELD_COLORS.get(field, DEFAULT_COLOR)
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_l = bdf["ird_linear"].values.astype(float)
        dates = bdf["opening_valve_date"].values
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t>0) & (ird_p>0)
        mask_l = mask & np.isfinite(ird_l) & (ird_l>0)
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

        ax.scatter(dates[mask],   ird_t[mask], s=10, alpha=0.70,
                   color=color,     marker="o", zorder=4)
        ax.scatter(dates[mask],   ird_p[mask], s=10, alpha=0.60,
                   color=PRED_COLOR, marker="x", linewidths=1.2, zorder=5)
        ax.scatter(dates[mask_l], ird_l[mask_l], s=8, alpha=0.50,
                   color="#E07B39",  marker="^", linewidths=0.8, zorder=3,
                   label="Linear baseline" if bn == held_out[0] else None)

        for d, yt, yp in zip(dates[mask], ird_t[mask], ird_p[mask]):
            ax.plot([d,d],[yt,yp], color="gray", linewidth=0.35,
                    alpha=0.25, zorder=2)

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
                Line2D([0],[0], marker="^", color="#E07B39", markersize=5,
                       linestyle="None", label="Linear baseline"),
                Line2D([0],[0], color="black", linewidth=0.7,
                       linestyle="--", alpha=0.4, label="Tillage reset"),
            ], fontsize=8, loc="upper right", ncol=4, framealpha=0.85)

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
        ird_l = bdf["ird_linear"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t>0) & (ird_p>0)
        mask_l = mask & np.isfinite(ird_l) & (ird_l>0)
        yt = ird_t[mask]; yp = ird_p[mask]
        yl = ird_l[mask_l]; yt_l = ird_t[mask_l]
        m  = metrics_ird(yt, yp, verbose=False) or {}

        if len(yl) > 0:
            ax.scatter(yt_l, yl, s=10, alpha=0.35, color="lightgrey",
                       marker="^", zorder=2, label="Linear baseline")
        ax.scatter(yt, yp, s=12, alpha=0.65, color=color,
                   edgecolors="none", zorder=3, label="Model 1")
        lo = min(yt.min(), yp.min())*0.90
        hi = max(yt.max(), yp.max())*1.05
        ax.plot([lo,hi],[lo,hi], "k--", linewidth=0.9, alpha=0.6)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Basin {bn} ({field})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Actual IRD (cm/h)", fontsize=8)
        ax.set_ylabel("Predicted (cm/h)", fontsize=8)
        ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)
        ax.annotate(
            f"R²={m.get('r2',np.nan):+.3f}\n"
            f"RMSE={m.get('rmse',np.nan):.3f}\n"
            f"MAPE={m.get('mape',np.nan):.1f}%",
            xy=(0.05,0.97), xycoords="axes fraction", fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.90, edgecolor="lightgrey"))
        if bn == held_out[0]:
            ax.legend(fontsize=7, loc="lower right")
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
    print("  Config  : E-full — all sensor-passing events, disjoint train/val")
    print("="*65)

    held_out      = load_selected_basins()
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