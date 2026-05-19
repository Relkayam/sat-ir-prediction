"""
models/model2_reset.py — Model 2: post-tillage IRD recovery prediction
=======================================================================
Reads held-out basins from data/selected_basins.csv (written by bootstrap).
Trains E-full configuration and evaluates on those 5 basins.

Features: Set-D (12 features, bootstrap-validated).
Run experiments/run_bootstrap.py first to generate selected_basins.csv.

Usage
-----
  python -m models.model2_reset
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RESET_CSV, OUTLIER_CSV, SELECTED_BASINS_CSV, TABLES_DIR,
    FIELD_NAMES, BOOSTING_PARAMS_M2, EARLY_STOPPING_ROUNDS_M2,
)
from pipeline.features import MODEL2_FEATURES, TARGET_M2

FIELD_COLORS = {
    "Soreq 2": "#065A82", "Yavne 1": "#1C7293",
    "Yavne 2": "#E07B39", "Yavne 3": "#27AE60", "Yavne 4": "#7D3C98",
}
DEFAULT_COLOR = "#065A82"


# ─────────────────────────────────────────────────────────────────────────────
# Load selected basins
# ─────────────────────────────────────────────────────────────────────────────

def load_selected_basins() -> list[int]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, label="", verbose=True):
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    if mask.sum() < 2: return {}
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
        rel_rmse   = round(rmse/ird_mean if ird_mean > 0 else np.nan, 4),
        n          = int(mask.sum()),
    )
    if verbose and label:
        print(f"  {label:<50}  "
              f"R²={m['r2']:+.4f}  RMSE={m['rmse']:.4f} cm/h  "
              f"MAPE={m['mape']:.1f}%  n={m['n']}")
    return m


def _back(y_log, prev_ird):
    return prev_ird * np.exp(y_log)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not RESET_CSV.exists():
        raise FileNotFoundError(f"{RESET_CSV} not found.")
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    avail   = [f for f in MODEL2_FEATURES if f in df.columns]
    missing = [f for f in MODEL2_FEATURES if f not in df.columns]
    print(f"  Reset dataset  : {len(df)} rows  {df['basin_number'].nunique()} basins")
    print(f"  Features       : {avail}")
    if missing: print(f"  WARNING missing: {missing}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Naive baseline
# ─────────────────────────────────────────────────────────────────────────────

def compute_naive(df: pd.DataFrame, held_out: list[int]) -> dict:
    held_set = set(held_out)
    test  = df[df["basin_number"].isin(held_set)].copy()
    valid = (test["IRD_at_reset"].notna() &
             test["prev_IRD_at_reset_raw"].notna() &
             (test["IRD_at_reset"] > 0) &
             (test["prev_IRD_at_reset_raw"] > 0))
    return _metrics(
        test.loc[valid, "IRD_at_reset"].values.astype(float),
        test.loc[valid, "prev_IRD_at_reset_raw"].values.astype(float),
        label="Naive baseline [held-out]",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prepare splits
# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(
    df:       pd.DataFrame,
    held_out: list[int],
    feat_cols:list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """E-full: train on all non-held-out basins."""
    held_set  = set(held_out)
    required  = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    avail_req = [c for c in required if c in df.columns]

    test  = df[df["basin_number"].isin(held_set)].dropna(
        subset=avail_req).reset_index(drop=True)
    non_ho= df[~df["basin_number"].isin(held_set)]
    train = non_ho.dropna(subset=avail_req).reset_index(drop=True)
    val   = non_ho[non_ho["split_chrono"]=="val"].dropna(
        subset=avail_req).reset_index(drop=True)

    print(f"  train={len(train)} ({non_ho['basin_number'].nunique()} basins)  "
          f"val={len(val)}  test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    train: pd.DataFrame, val: pd.DataFrame, feat_cols: list[str]
) -> tuple[object, StandardScaler]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    model = LGBMRegressor(**BOOSTING_PARAMS_M2)
    model.fit(
        Xtr, train[TARGET_M2].values,
        eval_set=[(Xva, val[TARGET_M2].values)],
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS_M2, verbose=False),
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
    scaler:    StandardScaler,
    feat_cols: list[str],
    held_out:  list[int],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    Xte              = scaler.transform(test[feat_cols].values)
    prev_ird         = test["prev_IRD_at_reset_raw"].values.astype(float)
    test             = test.copy()
    test["ird_pred"] = _back(model.predict(Xte), prev_ird)
    test["ird_true"] = test["IRD_at_reset"].values.astype(float)
    test["ird_naive"]= prev_ird

    print(f"\n{'='*70}")
    print("  HELD-OUT RESULTS — Model 2 (E-full, Set-D)")
    print(f"{'='*70}")
    print(f"  {'Basin':>7}  {'Field':<12}  {'n':>5}  "
          f"{'R²':>8}  {'RMSE':>8}  {'MAPE%':>7}  "
          f"{'Naive RMSE':>11}  Beat?")
    print(f"  {'-'*70}")

    rows = []; all_true = []; all_pred = []; all_naive = []
    for bn in held_out:
        bdf   = test[test["basin_number"]==bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t>0) & (ird_p>0)
        all_true.append(ird_t[mask]); all_pred.append(ird_p[mask])
        all_naive.append(ird_n[mask])
        m  = _metrics(ird_t[mask], ird_p[mask], verbose=False) or {}
        mn = _metrics(ird_t[mask], ird_n[mask], verbose=False) or {}
        beat = "✓" if m.get("rmse", np.inf) < mn.get("rmse", np.inf) else "✗"
        print(f"  {bn:>7}  {field:<12}  {m.get('n',0):>5}  "
              f"{m.get('r2',np.nan):>+8.3f}  {m.get('rmse',np.nan):>8.3f}  "
              f"{m.get('mape',np.nan):>6.1f}%  {mn.get('rmse',np.nan):>10.3f}  {beat}")
        rows.append(dict(basin=bn, field=field, n=m.get("n",0),
                         r2=m.get("r2",np.nan), rmse=m.get("rmse",np.nan),
                         mape=m.get("mape",np.nan),
                         rmse_naive=mn.get("rmse",np.nan), beats_naive=beat))

    ird_t_all = np.concatenate(all_true)
    ird_p_all = np.concatenate(all_pred)
    ird_n_all = np.concatenate(all_naive)
    m_all   = _metrics(ird_t_all, ird_p_all, verbose=False) or {}
    m_naive = _metrics(ird_t_all, ird_n_all, verbose=False) or {}
    delta   = m_naive.get("rmse",np.nan) - m_all.get("rmse",np.nan)
    wins    = sum(1 for r in rows if r["beats_naive"]=="✓")

    print(f"  {'-'*70}")
    print(f"  {'POOLED':>7}  {'all':12}  {m_all.get('n',0):>5}  "
          f"{m_all.get('r2',np.nan):>+8.3f}  {m_all.get('rmse',np.nan):>8.3f}  "
          f"{m_all.get('mape',np.nan):>6.1f}%  {m_naive.get('rmse',np.nan):>10.3f}")
    print(f"\n  RMSE improvement over naive: "
          f"{delta:.3f} cm/h ({100*delta/m_naive.get('rmse',1):.1f}%)")
    print(f"  Beats naive: {wins}/{len(held_out)} basins")

    return m_all, pd.DataFrame(rows), test


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(test: pd.DataFrame, held_out: list[int]) -> None:
    from plot_style import add_season_bands
    import plot_style as _ps

    n = len(held_out)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8*n),
                             gridspec_kw={"hspace": 0.35})
    if n == 1: axes = [axes]
    fig.suptitle(
        "Model 2 — Held-out time series (E-full, Set-D)\n"
        "Circles = actual  |  Crosses = predicted  |  Grey = naive",
        fontsize=10, fontweight="bold")

    for ax, bn in zip(axes, held_out):
        bdf   = test[test["basin_number"]==bn].sort_values("reset_date")
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        color = FIELD_COLORS.get(field, DEFAULT_COLOR)
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        dates = bdf["reset_date"].values
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t>0) & (ird_p>0)
        m  = _metrics(ird_t[mask], ird_p[mask], verbose=False) or {}
        mn = _metrics(ird_t[mask], ird_n[mask], verbose=False) or {}

        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, 0.55)
        add_season_bands(ax, bdf["reset_date"].min(), bdf["reset_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        ax.plot(dates[mask], ird_n[mask], color="lightgrey",
                linewidth=1.0, alpha=0.9, zorder=2)
        ax.scatter(dates[mask], ird_t[mask], s=35, alpha=0.85,
                   color=color, marker="o", zorder=4)
        ax.scatter(dates[mask], ird_p[mask], s=35, alpha=0.85,
                   color=color, marker="x", linewidths=1.8, zorder=5)
        for d, yt, yp in zip(dates[mask], ird_t[mask], ird_p[mask]):
            ax.plot([d,d],[yt,yp], color="gray", linewidth=0.5, alpha=0.30, zorder=3)
        ax.plot(dates[mask], ird_t[mask], color=color,
                linewidth=0.6, alpha=0.25, zorder=3)

        beat = "✓" if m.get("rmse",np.inf) < mn.get("rmse",np.inf) else "✗"
        ax.set_ylabel(f"Basin {bn} ({field})\nIRD_reset (cm/h)", fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(True, alpha=0.15)
        ax.set_title(
            f"R²={m.get('r2',np.nan):+.3f}  "
            f"RMSE={m.get('rmse',np.nan):.3f} cm/h  "
            f"MAPE={m.get('mape',np.nan):.1f}%  "
            f"Naive RMSE={mn.get('rmse',np.nan):.3f}  {beat}", fontsize=8)

        if bn == held_out[0]:
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0],[0], marker="o", color=color, markersize=5,
                       linestyle="None", label="Actual"),
                Line2D([0],[0], marker="x", color=color, markersize=5,
                       linestyle="None", markeredgewidth=1.8, label="Predicted"),
                Line2D([0],[0], color="lightgrey", linewidth=1.5, label="Naive"),
            ], fontsize=8, loc="upper right", ncol=3, framealpha=0.85)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


def plot_scatter(test: pd.DataFrame, held_out: list[int]) -> None:
    n = len(held_out)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 5), squeeze=False)
    axes = axes[0]
    fig.suptitle(
        "Model 2 — Actual vs predicted IRD_reset (E-full, Set-D)\n"
        "Circles = Model 2  |  Triangles = Naive  |  Dashed = 1:1",
        fontsize=10, fontweight="bold")

    for ax, bn in zip(axes, held_out):
        bdf   = test[test["basin_number"]==bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        color = FIELD_COLORS.get(field, DEFAULT_COLOR)
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        ird_n = bdf["ird_naive"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t>0) & (ird_p>0)
        yt = ird_t[mask]; yp = ird_p[mask]; yn = ird_n[mask]
        m  = _metrics(yt, yp, verbose=False) or {}
        mn = _metrics(yt, yn, verbose=False) or {}
        ax.scatter(yt, yn, s=28, alpha=0.35, color="grey",
                   marker="^", zorder=2, label="Naive")
        ax.scatter(yt, yp, s=32, alpha=0.75, color=color,
                   edgecolors="none", zorder=3, label="Model 2")
        all_v = np.concatenate([yt, yp, yn])
        lo = all_v.min()*0.88; hi = all_v.max()*1.08
        ax.plot([lo,hi],[lo,hi], "k--", linewidth=0.9, alpha=0.6)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_aspect("equal", adjustable="box")
        beat = "✓" if m.get("rmse",np.inf) < mn.get("rmse",np.inf) else "✗"
        ax.set_title(f"Basin {bn} ({field})\n{beat} beats naive",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Actual IRD_reset (cm/h)", fontsize=8)
        ax.set_ylabel("Predicted (cm/h)", fontsize=8)
        ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)
        ax.annotate(
            f"R²={m.get('r2',np.nan):+.3f}\n"
            f"RMSE={m.get('rmse',np.nan):.3f}\n"
            f"Naive={mn.get('rmse',np.nan):.3f}",
            xy=(0.05,0.97), xycoords="axes fraction", fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.90, edgecolor="lightgrey"))
        if bn == held_out[0]:
            ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout(rect=[0,0,1,0.90])
    plt.show()


def plot_shap(
    df: pd.DataFrame, model, scaler: StandardScaler,
    feat_cols: list[str], held_out: list[int]
) -> None:
    held_set = set(held_out)
    test = df[df["basin_number"].isin(held_set)].dropna(
        subset=feat_cols + [TARGET_M2])
    if len(test) < 5:
        print("  SHAP: insufficient data")
        return
    Xte = scaler.transform(test[feat_cols].values)
    print(f"\n  Computing SHAP (n={len(test)})...")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xte)
    except Exception as e:
        print(f"  ERROR: {e}"); return

    mean_abs   = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(-mean_abs)
    print(f"\n  SHAP feature importance:")
    print(f"  {'Rank':<5}  {'Feature':<22}  {'Mean |SHAP|':>12}  Direction")
    print(f"  {'-'*58}")
    for rank, i in enumerate(sorted_idx):
        direction = "→ higher recovery" if shap_values[:,i].mean() > 0 else "→ lower recovery"
        print(f"  {rank+1:<5}  {feat_cols[i]:<22}  "
              f"{mean_abs[i]:>12.4f}  {direction}")

    shap_df = pd.DataFrame(Xte, columns=feat_cols)
    plt.figure(figsize=(10, max(4, len(feat_cols)*0.9)))
    shap.summary_plot(shap_values, shap_df, show=False, plot_size=None)
    plt.title(f"SHAP beeswarm — Model 2 (E-full, Set-D)\n"
              f"n={len(test)}  |  Positive SHAP → higher recovery",
              fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_results(pooled: dict, basin_df: pd.DataFrame) -> None:
    path = TABLES_DIR / "model2_results_condition_e.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([pooled]).to_excel(writer, sheet_name="pooled", index=False)
        basin_df.to_excel(writer, sheet_name="per_basin", index=False)
    print(f"\n  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  MODEL 2 — Post-tillage IRD recovery prediction")
    print(f"  Target   : {TARGET_M2}")
    print(f"  Features : {MODEL2_FEATURES}")
    print("  Config   : E-full (all non-held-out basins including outliers)")
    print("="*65)

    held_out  = load_selected_basins()
    df        = load_data()
    feat_cols = [f for f in MODEL2_FEATURES if f in df.columns]

    naive = compute_naive(df, held_out)

    train, val, test = prepare_splits(df, held_out, feat_cols)
    model, scaler    = train_model(train, val, feat_cols)
    pooled, basin_df, test_df = evaluate_held_out(
        test, model, scaler, feat_cols, held_out)

    save_results(pooled, basin_df)

    print("\n--- Plots ---")
    plot_timeseries(test_df, held_out)
    plot_scatter(test_df, held_out)
    plot_shap(df, model, scaler, feat_cols, held_out)

    print("\nDone.")
    return model, scaler, feat_cols


if __name__ == "__main__":
    main()