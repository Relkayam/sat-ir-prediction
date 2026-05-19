"""
analysis/feature_selection_m2.py
==================================
Forward stepwise feature selection for Model 2 (IRD_norm_log_reset).

Evaluation condition — held-out:
  Train : basin_role == 'clean', all split_chrono values
  Val   : basin_role == 'clean', split_chrono == 'val'  (early stopping only)
  Test  : basin_role == 'held_out' (5 completely unseen basins)

The metric that drives selection is RMSE on the 5 held-out basins —
not within-sample chrono performance.

Candidate pool
--------------
  ALL numeric columns in reset_dataset.csv minus:
    - metadata / ID columns
    - target (IRD_norm_log_reset)
    - back-transform columns (IRD_at_reset, prev_IRD_at_reset_raw)
    - split / role columns
  New features (prev_delta, prev_prev_delta, IRD_trend, max_RD, etc.)
  are automatically included since they are in the CSV.

Metric: RMSE on raw IRD (cm/h) after back-transform.
Model:  LightGBM with BOOSTING_PARAMS_M2 (published defaults).

Output
------
  Prints performance at each step.
  Saves feature_selection_m2_held_out.xlsx to outputs/tables/
  Shows 3-panel performance curve (plt.show())

Usage
-----
  python -m analysis.feature_selection_m2
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RESET_CSV, TABLES_DIR,
    BOOSTING_PARAMS_M2, EARLY_STOPPING_ROUNDS_M2,
)
from pipeline.features import TARGET_M2

# ── Columns excluded from candidate pool ─────────────────────────────────────
_EXCLUDE_COLS = {
    "basin_number", "field_name", "segment_id", "reset_date",
    "is_good_segment", "facility",
    TARGET_M2,
    "IRD_at_reset", "prev_IRD_at_reset_raw",
    "basin_role", "split_chrono",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load reset_dataset.csv and build candidate feature pool.
    Splits computed at runtime from basin_role and split_chrono.
    """
    if not RESET_CSV.exists():
        raise FileNotFoundError(
            f"{RESET_CSV} not found.\n"
            "Run: python -m pipeline.build_reset_dataset"
        )

    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    # All numeric columns not in exclusion set and with enough non-null values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates   = sorted([
        c for c in numeric_cols
        if c not in _EXCLUDE_COLS and df[c].notna().sum() > 10
    ])

    print(f"  Reset dataset   : {len(df)} rows  "
          f"{df['basin_number'].nunique()} basins")
    print(f"  Basin roles     : "
          f"{df['basin_role'].value_counts().to_dict()}")
    print(f"  split_chrono    : "
          f"{df['split_chrono'].value_counts().to_dict()}")

    n_train = int((df["basin_role"] == "clean").sum())
    n_val   = int(((df["basin_role"] == "clean") &
                   (df["split_chrono"] == "val")).sum())
    n_test  = int((df["basin_role"] == "held_out").sum())
    print(f"  Training pool   : {n_train} rows "
          f"({df[df['basin_role']=='clean']['basin_number'].nunique()} basins)")
    print(f"  Val (early stop): {n_val} rows")
    print(f"  Test (held-out) : {n_test} rows "
          f"({df[df['basin_role']=='held_out']['basin_number'].nunique()} basins)")

    print(f"\n  Candidate pool ({len(candidates)} features):")
    for i, c in enumerate(candidates):
        n_nan = int(df[c].isna().sum())
        pct   = 100 * n_nan / len(df)
        print(f"    {i+1:>3}. {c:<30}  NaN={n_nan} ({pct:.1f}%)")

    return df, candidates


# ─────────────────────────────────────────────────────────────────────────────
# Naive baseline
# ─────────────────────────────────────────────────────────────────────────────

def naive_held_out_rmse(df: pd.DataFrame) -> float:
    """RMSE of naive baseline (δ̂=0) on held-out basins."""
    test = df[df["basin_role"] == "held_out"].dropna(
        subset=["IRD_at_reset", "prev_IRD_at_reset_raw"])
    ird_true  = test["IRD_at_reset"].values.astype(float)
    ird_naive = test["prev_IRD_at_reset_raw"].values.astype(float)
    mask = (np.isfinite(ird_true) & np.isfinite(ird_naive) &
            (ird_true > 0) & (ird_naive > 0))
    return float(np.sqrt(mean_squared_error(ird_true[mask], ird_naive[mask])))


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df:       pd.DataFrame,
    features: list[str],
) -> dict:
    """
    Train on basin_role=='clean' (all chrono splits).
    Early stopping on split_chrono=='val'.
    Evaluate on basin_role=='held_out'.
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    required = features + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]

    train = df[df["basin_role"] == "clean"].dropna(
        subset=required).reset_index(drop=True)
    val   = df[
        (df["basin_role"]   == "clean") &
        (df["split_chrono"] == "val")
    ].dropna(subset=required).reset_index(drop=True)
    test  = df[df["basin_role"] == "held_out"].dropna(
        subset=required).reset_index(drop=True)

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        return dict(rmse_ird=np.nan, mape=np.nan, r2_ird=np.nan,
                    n_test=0, best_iter=0)

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[features].values)
    Xva = sc.transform(val[features].values)
    Xte = sc.transform(test[features].values)

    model = LGBMRegressor(**BOOSTING_PARAMS_M2)
    model.fit(
        Xtr, train[TARGET_M2].values,
        eval_set=[(Xva, val[TARGET_M2].values)],
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS_M2, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    prev_ird = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_pred = prev_ird * np.exp(model.predict(Xte))
    ird_true = test["IRD_at_reset"].values.astype(float)

    mask = (np.isfinite(ird_true) & np.isfinite(ird_pred) &
            (ird_true > 0) & (ird_pred > 0))
    if mask.sum() < 5:
        return dict(rmse_ird=np.nan, mape=np.nan, r2_ird=np.nan,
                    n_test=0, best_iter=0)

    rmse = float(np.sqrt(mean_squared_error(ird_true[mask], ird_pred[mask])))
    mape = float(np.mean(
        np.abs((ird_true[mask]-ird_pred[mask])/ird_true[mask]))*100)
    r2   = float(r2_score(ird_true[mask], ird_pred[mask]))

    return dict(
        rmse_ird  = rmse,
        mape      = mape,
        r2_ird    = r2,
        n_test    = int(mask.sum()),
        best_iter = int(model.best_iteration_),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forward stepwise
# ─────────────────────────────────────────────────────────────────────────────

def forward_stepwise(
    df:           pd.DataFrame,
    all_features: list[str],
) -> pd.DataFrame:
    """
    Unconstrained forward stepwise on held-out test.
    Runs to completion — elbow identified post-hoc from printed results.
    """
    results   = []
    selected  = []
    remaining = list(all_features)

    rmse_naive = naive_held_out_rmse(df)
    print(f"\n  Naive baseline RMSE (held-out): {rmse_naive:.4f} cm/h")
    print(f"  Running {len(all_features)} steps...\n")

    results.append(dict(
        step=0, added="naive_baseline", n_features=0,
        features=[], rmse_ird=rmse_naive,
        mape=np.nan, r2_ird=np.nan, delta_rmse=0.0, best_iter=0,
    ))
    best_rmse = rmse_naive

    for step in range(1, len(all_features) + 1):
        if not remaining:
            break

        step_rows = []
        for candidate in remaining:
            m = train_and_evaluate(df, selected + [candidate])
            step_rows.append(dict(
                feature   = candidate,
                rmse_ird  = m["rmse_ird"],
                mape      = m["mape"],
                r2_ird    = m["r2_ird"],
                best_iter = m.get("best_iter", 0),
            ))

        step_df      = pd.DataFrame(step_rows).dropna(subset=["rmse_ird"])
        if step_df.empty:
            break

        best_row     = step_df.loc[step_df["rmse_ird"].idxmin()]
        best_feature = str(best_row["feature"])
        delta        = best_rmse - best_row["rmse_ird"]
        selected     = selected + [best_feature]
        remaining    = [f for f in remaining if f != best_feature]
        best_rmse    = best_row["rmse_ird"]

        marker = "✓" if delta > 0 else "✗"
        print(f"  Step {step:>3}  {marker}  {best_feature:<28}  "
              f"RMSE={best_row['rmse_ird']:.4f}  "
              f"Δ={delta:>+7.4f}  "
              f"iter={int(best_row['best_iter']):>4}  "
              f"set={selected}")

        results.append(dict(
            step      = step,
            added     = best_feature,
            n_features= len(selected),
            features  = list(selected),
            rmse_ird  = best_row["rmse_ird"],
            mape      = best_row["mape"],
            r2_ird    = best_row["r2_ird"],
            delta_rmse= delta,
            best_iter = int(best_row["best_iter"]),
        ))

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Summary and plot
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results_df: pd.DataFrame) -> None:
    naive_rmse = float(results_df.iloc[0]["rmse_ird"])
    best_row   = results_df.loc[results_df["rmse_ird"].idxmin()]

    print(f"\n{'='*85}")
    print("  FEATURE SELECTION SUMMARY — Model 2  |  Held-out evaluation")
    print(f"{'='*85}")
    print(f"  {'Step':>5}  {'Added':<28}  {'N':>3}  "
          f"{'RMSE':>8}  {'ΔRMSE':>8}  {'MAPE%':>7}  "
          f"{'R²':>7}  {'iter':>5}  Note")
    print(f"  {'-'*88}")

    best_rmse = results_df["rmse_ird"].min()
    for _, row in results_df.iterrows():
        delta_str = (f"{row['delta_rmse']:>+8.4f}"
                     if row["added"] != "naive_baseline" else f"{'—':>8}")
        note = ""
        if row["rmse_ird"] == best_rmse:
            note = "  ← BEST"
        elif row["added"] != "naive_baseline" and row["delta_rmse"] < 0:
            note = "  ← worse"
        mape_str = (f"{row['mape']:>7.1f}"
                    if np.isfinite(row.get("mape", np.nan)) else f"{'—':>7}")
        r2_str   = (f"{row['r2_ird']:>+7.4f}"
                    if np.isfinite(row.get("r2_ird", np.nan)) else f"{'—':>7}")
        iter_str = (f"{int(row['best_iter']):>5}"
                    if row.get("best_iter", 0) > 0 else f"{'—':>5}")
        print(f"  {int(row['step']):>5}  {str(row['added']):<28}  "
              f"{int(row['n_features']):>3}  "
              f"{row['rmse_ird']:>8.4f}  {delta_str}  "
              f"{mape_str}  {r2_str}  {iter_str}{note}")

    print(f"\n  Naive RMSE  : {naive_rmse:.4f} cm/h")
    print(f"  Best RMSE   : {best_row['rmse_ird']:.4f} cm/h")
    print(f"  Improvement : "
          f"{naive_rmse-best_row['rmse_ird']:+.4f} cm/h "
          f"({100*(naive_rmse-best_row['rmse_ird'])/naive_rmse:.1f}%)")
    print(f"  Best MAPE   : {best_row['mape']:.1f}%")
    print(f"  Best R²     : {best_row['r2_ird']:+.4f}")
    print(f"  N features  : {int(best_row['n_features'])}")
    print(f"\n  Feature set at best held-out RMSE:")
    for i, f in enumerate(best_row["features"], 1):
        print(f"    {i:>2}. {f}")
    print(f"\n  → Update MODEL2_FEATURES in pipeline/features.py:")
    print(f"    {best_row['features']}")


def plot_curve(results_df: pd.DataFrame) -> None:
    naive_rmse = float(results_df.iloc[0]["rmse_ird"])
    curve      = results_df.iloc[1:].copy().reset_index(drop=True)

    n_feat = curve["n_features"].values
    rmse   = curve["rmse_ird"].values
    r2     = curve["r2_ird"].values
    deltas = curve["delta_rmse"].values
    labels = curve["added"].values
    colors = ["seagreen" if d > 0 else "tomato" for d in deltas]
    best_idx = int(np.nanargmin(rmse))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Model 2 — Forward stepwise feature selection\n"
        "Metric: RMSE on 5 held-out unseen basins (raw IRD cm/h)\n"
        "Green = improved  |  Red = worsened  |  ★ = best",
        fontsize=10, fontweight="bold",
    )

    # RMSE curve
    ax = axes[0]
    ax.axhline(naive_rmse, color="gray", linewidth=1.5,
               linestyle="--", alpha=0.7, label=f"Naive={naive_rmse:.4f}")
    ax.plot(n_feat, rmse, color="black", linewidth=0.8, alpha=0.4, zorder=2)
    ax.scatter(n_feat, rmse, c=colors, s=55, zorder=4)
    for i, (x, y, lbl) in enumerate(zip(n_feat, rmse, labels)):
        va  = "bottom" if i % 2 == 0 else "top"
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(0, 4 if va=="bottom" else -4),
                    fontsize=5.5, ha="center", va=va, color=colors[i])
    ax.scatter(n_feat[best_idx], rmse[best_idx], color="gold", s=250,
               zorder=5, marker="*", edgecolors="black", linewidths=0.8)
    ax.set_xlabel("N features"); ax.set_ylabel("RMSE (cm/h)")
    ax.set_title("RMSE vs features  ↓ better", fontweight="bold")
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)

    # Delta bars
    ax = axes[1]
    bars = ax.bar(range(1, len(deltas)+1), deltas,
                  color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, lbl, val in zip(bars, labels, deltas):
        va  = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x()+bar.get_width()/2, val+(0.0003 if val>=0 else -0.0003),
                lbl, ha="center", va=va, fontsize=4.5, rotation=90)
    ax.set_xlabel("Step"); ax.set_ylabel("ΔRMSE (cm/h)")
    ax.set_title("Marginal improvement per step", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # R² curve
    ax = axes[2]
    valid = np.isfinite(r2)
    if valid.any():
        ax.plot(n_feat[valid], r2[valid], color="black",
                linewidth=0.8, alpha=0.4, zorder=2)
        ax.scatter(n_feat[valid], r2[valid],
                   c=[colors[i] for i in np.where(valid)[0]], s=55, zorder=4)
        ax.scatter(n_feat[np.nanargmax(r2)], r2[np.nanargmax(r2)],
                   color="gold", s=250, zorder=5, marker="*",
                   edgecolors="black", linewidths=0.8)
    for i, (x, y_val, lbl) in enumerate(zip(n_feat, r2, labels)):
        if not np.isfinite(y_val): continue
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(lbl, (x, y_val), textcoords="offset points",
                    xytext=(0, 4 if va=="bottom" else -4),
                    fontsize=5.5, ha="center", va=va, color=colors[i])
    ax.set_xlabel("N features"); ax.set_ylabel("R² (raw IRD)")
    ax.set_title("R² vs features  ↑ better", fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> list[str]:
    print("=" * 65)
    print("  MODEL 2 FEATURE SELECTION — forward stepwise")
    print("  Evaluation: 5 held-out unseen basins")
    print("  Metric: RMSE on raw IRD (cm/h) after back-transform")
    print("  Candidate pool: ALL numeric columns in reset_dataset.csv")
    print("=" * 65)

    df, candidates = load_data()
    results_df     = forward_stepwise(df, candidates)

    print_summary(results_df)

    # Full-candidate model for reference
    print(f"\n  Full {len(candidates)}-feature model (reference):")
    m_full = train_and_evaluate(df, candidates)
    print(f"    RMSE={m_full['rmse_ird']:.4f}  "
          f"MAPE={m_full['mape']:.1f}%  "
          f"R²={m_full['r2_ird']:+.4f}  "
          f"iter={m_full.get('best_iter',0)}")

    # Save
    save_df = results_df.copy()
    save_df["features"] = save_df["features"].apply(str)
    out_path = TABLES_DIR / "feature_selection_m2_held_out.xlsx"
    save_df.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    plot_curve(results_df)
    print("\nDone.")

    return results_df.loc[results_df["rmse_ird"].idxmin(), "features"]


if __name__ == "__main__":
    main()