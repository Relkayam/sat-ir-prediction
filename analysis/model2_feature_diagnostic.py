"""
analysis/model2_feature_diagnostic.py
======================================
Per-basin held-out comparison for Model 2 under three configurations:
  1. Naive baseline        (predict no change from previous reset)
  2. 3-feature model       (month_sin, total_LCT, month_cos)
  3. 11-feature model      (original full feature set)

Evaluation: 5 held-out unseen basins (Condition E)
Metric: RMSE and R² on raw IRD (cm/h) after back-transform.

Purpose: determine whether the 3-feature model improves on basins
4104 and 7201 where the 11-feature model lost to naive.

NOTE: Feature sets are defined locally here — this script does NOT
import MODEL2_FEATURES from features.py, to avoid the sub-list
consistency assertion while we are still experimenting.
Only TARGET_M2 is imported from features.py.

Usage
-----
  python -m analysis.model2_feature_diagnostic
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESET_CSV, RANDOM_SEED, FIELD_NAMES

# Import only TARGET_M2 — avoids triggering the sub-list assertion
TARGET_M2 = "IRD_norm_log_reset"

# ─────────────────────────────────────────────────────────────────────────────
# Feature sets to compare — defined locally, not from features.py
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_3 = [
    "month_sin",
    "total_LCT",
    "month_cos",
]

FEATURES_11 = [
    "month_sin",
    "month_cos",
    "prev_IRD_at_reset",
    "prev_prev_IRD_at_reset",
    "mean_ALPHA",
    "total_LCT",
    "sum_DrT",
    "sum_FT",
    "last_DrT",
    "last_RD",
    "DAR",
]

FEATURE_SETS = {
    "3-feature  (month_sin, total_LCT, month_cos)": FEATURES_3,
    "11-feature (original full set)":               FEATURES_11,
}

HELD_OUT_BASINS = [3203, 4104, 5102, 6303, 7201]

LGBM_PARAMS = dict(
    n_estimators=1000, max_depth=-1, num_leaves=31,
    learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Load data — Condition E
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    # Condition E: reassign outlier rows to chrono splits
    outlier_mask = (
        (df["basin_role"] == "outlier") &
        (df["split_held_out"] == "excluded")
    )
    df.loc[outlier_mask, "split_held_out"] = df.loc[outlier_mask, "split_chrono"]
    print(f"  Loaded {len(df)} rows | "
          f"{int(outlier_mask.sum())} outlier rows reassigned (Condition E)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Train and predict
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    required = features + [TARGET_M2, "prev_IRD_at_reset_raw",
                           "IRD_at_reset", "basin_number"]

    train = df[df["split_held_out"] == "train"].dropna(
        subset=required).reset_index(drop=True)
    val   = df[df["split_held_out"] == "val"].dropna(
        subset=required).reset_index(drop=True)
    test  = df[df["split_held_out"] == "held_out_test"].dropna(
        subset=required).reset_index(drop=True)

    print(f"    train={len(train)}  val={len(val)}  test={len(test)}")

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[features].values)
    Xva = sc.transform(val[features].values)
    Xte = sc.transform(test[features].values)

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        Xtr, train[TARGET_M2].values,
        eval_set=[(Xva, val[TARGET_M2].values)],
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"    best_iter={model.best_iteration_}")

    prev_ird          = test["prev_IRD_at_reset_raw"].values.astype(float)
    test              = test.copy()
    test["ird_pred"]  = prev_ird * np.exp(model.predict(Xte))
    test["ird_true"]  = test["IRD_at_reset"].values.astype(float)
    test["ird_naive"] = prev_ird

    return test[["basin_number", "ird_true", "ird_pred", "ird_naive"]]


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin and pooled metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, pred_col: str) -> dict:
    rows = {}
    all_true, all_pred = [], []

    for bn in HELD_OUT_BASINS:
        bdf  = results[results["basin_number"] == bn]
        yt   = bdf["ird_true"].values.astype(float)
        yp   = bdf[pred_col].values.astype(float)
        mask = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
        all_true.append(yt[mask])
        all_pred.append(yp[mask])

        if mask.sum() < 2:
            rows[bn] = dict(r2=np.nan, rmse=np.nan, n=0)
            continue

        rows[bn] = dict(
            r2   = round(float(r2_score(yt[mask], yp[mask])), 3),
            rmse = round(float(np.sqrt(mean_squared_error(
                yt[mask], yp[mask]))), 3),
            n    = int(mask.sum()),
        )

    yt_all = np.concatenate(all_true)
    yp_all = np.concatenate(all_pred)
    rows["POOLED"] = dict(
        r2   = round(float(r2_score(yt_all, yp_all)), 3),
        rmse = round(float(np.sqrt(mean_squared_error(yt_all, yp_all))), 3),
        n    = len(yt_all),
    )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(all_results: dict, naive_metrics: dict) -> None:
    model_names  = list(all_results.keys())
    basins_show  = HELD_OUT_BASINS + ["POOLED"]

    print(f"\n{'='*95}")
    print("  PER-BASIN COMPARISON — Model 2  |  Condition E  |  Held-out test")
    print("  RMSE (cm/h) and R²  |  ✓ = beats naive on RMSE  |  ✗ = naive wins")
    print(f"{'='*95}")

    # Header row
    hdr  = f"  {'Basin':<8} {'Field':<12}  {'Naive':^16}"
    for name in model_names:
        short = name.split("(")[0].strip()
        hdr  += f"  {short:^18}"
    print(hdr)

    sub  = f"  {'':<8} {'':<12}  {'RMSE':>7} {'R²':>7}"
    for _ in model_names:
        sub += f"  {'RMSE':>8} {'':1} {'R²':>7}"
    print(sub)
    print(f"  {'-'*90}")

    for bn in basins_show:
        field = (FIELD_NAMES.get(int(str(bn)[0]), "")
                 if bn != "POOLED" else "all")
        nm    = naive_metrics.get(bn, {})
        row   = (f"  {str(bn):<8} {field:<12}  "
                 f"{nm.get('rmse', np.nan):>7.3f} "
                 f"{nm.get('r2', np.nan):>+7.3f}")

        for name in model_names:
            m    = all_results[name].get(bn, {})
            rmse = m.get("rmse", np.nan)
            r2   = m.get("r2",   np.nan)
            flag = "✓" if rmse < nm.get("rmse", np.inf) else "✗"
            row += f"  {rmse:>8.3f} {flag} {r2:>+7.3f}"

        print(row)

    # Summary
    print(f"\n  Basins where model beats naive (RMSE):")
    naive_pooled = naive_metrics.get("POOLED", {}).get("rmse", np.nan)
    for name in model_names:
        short = name.split("(")[0].strip()
        wins  = sum(
            1 for bn in HELD_OUT_BASINS
            if all_results[name].get(bn, {}).get("rmse", np.inf)
            < naive_metrics.get(bn, {}).get("rmse", np.inf)
        )
        pooled_rmse = all_results[name].get("POOLED", {}).get("rmse", np.nan)
        delta = naive_pooled - pooled_rmse
        print(f"    {short}: {wins}/5 basins  |  "
              f"Pooled RMSE={pooled_rmse:.3f} cm/h  "
              f"Δ vs naive={delta:+.3f} cm/h")


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

def print_verdict(all_results: dict, naive_metrics: dict) -> None:
    print(f"\n{'='*65}")
    print("  VERDICT")
    print(f"{'='*65}")

    for name, results in all_results.items():
        short = name.split("(")[0].strip()
        wins  = sum(
            1 for bn in HELD_OUT_BASINS
            if results.get(bn, {}).get("rmse", np.inf)
            < naive_metrics.get(bn, {}).get("rmse", np.inf)
        )
        pooled = results.get("POOLED", {}).get("rmse", np.nan)
        naive_p = naive_metrics.get("POOLED", {}).get("rmse", np.nan)
        delta   = naive_p - pooled

        print(f"\n  {short}:")
        print(f"    Beats naive on {wins}/5 basins")
        print(f"    Pooled RMSE improvement: {delta:+.3f} cm/h "
              f"({100*delta/naive_p:.1f}%)")

        if wins == 5:
            print(f"    → RECOMMENDED: beats naive universally")
        elif wins >= 3:
            print(f"    → PARTIAL: investigate losing basins")
        else:
            print(f"    → PROBLEMATIC: naive wins on majority of basins")

    print(f"\n  Naive pooled RMSE: "
          f"{naive_metrics.get('POOLED', {}).get('rmse', np.nan):.3f} cm/h")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MODEL 2 FEATURE DIAGNOSTIC")
    print("  naive vs 3-feature vs 11-feature on held-out basins")
    print("=" * 65)

    df = load_data()

    all_results  = {}
    last_test_df = None

    for name, features in FEATURE_SETS.items():
        print(f"\n  Training: {name}")
        test_df             = train_and_predict(df, features)
        all_results[name]   = compute_metrics(test_df, "ird_pred")
        last_test_df        = test_df   # naive is the same for all runs

    naive_m = compute_metrics(last_test_df, "ird_naive")

    print_table(all_results, naive_m)
    print_verdict(all_results, naive_m)


if __name__ == "__main__":
    main()