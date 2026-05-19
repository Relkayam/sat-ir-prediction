"""
a_bootstrap_smoke.py — Bootstrap validation comparing Model 2 feature sets
===========================================================================
200 iterations: each iteration randomly selects 5 held-out basins from the
eligible pool (non-outlier, non-fixed), trains each feature set, evaluates
pooled RMSE on those 5 unseen basins.

Feature sets compared:
  Old-3  : original 3-feature model (month_sin, month_cos, total_LCT)
  New-16 : full 16-feature selection result
  Set C  : 10-feature set (no seasonality, no delta features)
  Set D  : 12-feature set (Set C + prev_delta + prev_prev_delta)

Run: python a_bootstrap_smoke.py
"""
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\user\PycharmProjects\sat-ir-prediction")

from config import (
    RESET_CSV, OUTLIER_CSV, HELD_OUT_BASIN_LIST,
    BOOSTING_PARAMS_M2, EARLY_STOPPING_ROUNDS_M2, RANDOM_SEED,
)
from pipeline.features import TARGET_M2

# ── Feature sets to compare ───────────────────────────────────────────────────
FEATURE_SETS = {
    "Old-3": [
        'month_sin', 'month_cos', 'total_LCT',
    ],
    "New-16": [
        'month_sin', 'frac_zero_DrT', 'n_events', 'sum_RW', 'total_LCT',
        'max_RD', 'max_TD', 'max_DrT', 'last_RD', 'min_TD', 'DAT',
        'mean_RW', 'mean_DrT', 'mean_TD', 'DAR', 'max_TW',
    ],
    "Set-C": [
        'frac_zero_DrT', 'n_events', 'sum_RW', 'total_LCT',
        'max_RD', 'max_DrT', 'last_RD',
        'mean_RW', 'mean_DrT', 'DAR',
    ],
    "Set-D": [
        'frac_zero_DrT', 'n_events', 'sum_RW', 'total_LCT',
        'max_RD', 'max_DrT', 'last_RD',
        'mean_RW', 'mean_DrT', 'DAR',
        'prev_delta', 'prev_prev_delta',
    ],
}

N_ITERATIONS = 200
N_HELD_OUT   = 5


# ─────────────────────────────────────────────────────────────────────────────
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    if mask.sum() < 2:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def run_one(
    df:        pd.DataFrame,
    held_out:  list[int],
    feat_cols: list[str],
) -> tuple[float, float]:
    """
    Train on all non-held-out basins, evaluate on held_out.
    Returns (rmse_model, rmse_naive).
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    required = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    held_set = set(held_out)

    train = df[~df["basin_number"].isin(held_set)].dropna(
        subset=required).reset_index(drop=True)
    val   = df[
        (~df["basin_number"].isin(held_set)) &
        (df["split_chrono"] == "val")
    ].dropna(subset=required).reset_index(drop=True)
    test  = df[df["basin_number"].isin(held_set)].dropna(
        subset=required).copy()

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        return np.nan, np.nan

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)

    model = LGBMRegressor(**BOOSTING_PARAMS_M2)
    model.fit(
        Xtr, train[TARGET_M2].values,
        eval_set=[(Xva, val[TARGET_M2].values)],
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS_M2, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    prev_ird  = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_pred  = prev_ird * np.exp(model.predict(Xte))
    ird_true  = test["IRD_at_reset"].values.astype(float)
    ird_naive = prev_ird

    return _rmse(ird_true, ird_pred), _rmse(ird_true, ird_naive)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  BOOTSTRAP VALIDATION — Model 2 feature set comparison")
    print(f"  {N_ITERATIONS} random held-out selections  |  "
          f"{N_HELD_OUT} basins per iteration")
    print(f"  Same 5 random basins used for all feature sets per iteration")
    print("=" * 70)

    # Load data
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"\n  Dataset: {len(df)} rows  {df['basin_number'].nunique()} basins")

    # Load outlier basins
    outlier_basins: set[int] = set()
    if OUTLIER_CSV.exists():
        with open(OUTLIER_CSV) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("basin"):
                    continue
                try:
                    outlier_basins.add(int(line.split(",")[0].strip()))
                except ValueError:
                    pass
    print(f"  Outlier basins: {len(outlier_basins)}  {sorted(outlier_basins)}")

    # Eligible pool — non-outlier, non-fixed
    fixed_ho = set(HELD_OUT_BASIN_LIST)
    all_bas  = set(df["basin_number"].dropna().unique().astype(int))
    eligible = sorted(all_bas - outlier_basins - fixed_ho)
    print(f"  Eligible pool : {len(eligible)} basins  {eligible}")

    # Feature set availability check
    print(f"\n  Feature set summary:")
    for name, feats in FEATURE_SETS.items():
        missing = [f for f in feats if f not in df.columns]
        nan_any = [f for f in feats if f in df.columns and
                   df[f].isna().mean() > 0.10]
        status  = "OK" if not missing else f"MISSING: {missing}"
        print(f"    {name:<8} {len(feats):>3} features  {status}")
        if nan_any:
            print(f"             ⚠ high NaN: {nan_any}")

    # Storage
    results     = {name: [] for name in FEATURE_SETS}
    naive_rmses = []
    rng         = np.random.default_rng(RANDOM_SEED)

    # Header
    names = list(FEATURE_SETS.keys())
    print(f"\n  Running {N_ITERATIONS} iterations...\n")
    hdr = f"  {'Iter':>5}  {'Naive':>8}"
    for n in names:
        hdr += f"  {n:>8}"
    for n in names:
        hdr += f"  {'Δ'+n:>9}"
    print(hdr)
    print(f"  {'-'*80}")

    for i in range(N_ITERATIONS):
        held_out = list(rng.choice(eligible, size=N_HELD_OUT, replace=False))

        naive_r      = None
        iter_results = {}

        for name, feats in FEATURE_SETS.items():
            rmse_m, rmse_n = run_one(df, held_out, feats)
            iter_results[name] = rmse_m
            if naive_r is None and np.isfinite(rmse_n):
                naive_r = rmse_n

        naive_rmses.append(naive_r if naive_r is not None else np.nan)
        for name in names:
            results[name].append(iter_results.get(name, np.nan))

        # Print progress every 25 iterations and first 3
        if (i + 1) % 25 == 0 or i < 3:
            row = f"  {i+1:>5}  {naive_r:>8.4f}"
            for n in names:
                v = iter_results.get(n, np.nan)
                row += f"  {v:>8.4f}"
            for n in names:
                v = iter_results.get(n, np.nan)
                d = (naive_r - v) if (naive_r and np.isfinite(v)) else np.nan
                row += f"  {d:>+9.4f}"
            print(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {N_ITERATIONS} random bootstrap iterations")
    print(f"  Each iteration: {N_HELD_OUT} basins randomly held out from "
          f"{len(eligible)}-basin eligible pool")
    print(f"{'='*70}")

    naive_arr = np.array(naive_rmses, dtype=float)
    print(f"\n  Naive baseline:")
    print(f"    median={np.nanmedian(naive_arr):.4f}  "
          f"IQR=[{np.nanpercentile(naive_arr,25):.4f}, "
          f"{np.nanpercentile(naive_arr,75):.4f}]")

    print(f"\n  {'Set':<8}  {'Median RMSE':>12}  {'Median Δ':>10}  "
          f"{'Beats naive':>12}  {'vs Old-3':>10}")
    print(f"  {'-'*60}")

    medians = {}
    for name in names:
        arr   = np.array(results[name], dtype=float)
        delta = naive_arr - arr
        beats = int(np.sum(arr < naive_arr))
        pct   = 100 * beats / N_ITERATIONS
        med   = float(np.nanmedian(arr))
        medians[name] = med

        vs_old3 = ""
        if name != "Old-3":
            diff = med - medians.get("Old-3", np.nan)
            vs_old3 = f"{diff:>+10.4f}"

        print(f"  {name:<8}  {med:>12.4f}  "
              f"{np.nanmedian(delta):>+10.4f}  "
              f"{beats:>5}/{N_ITERATIONS} ({pct:>5.1f}%)  "
              f"{vs_old3}")

    # Detailed stats
    print(f"\n  Detailed breakdown:")
    for name in names:
        arr   = np.array(results[name], dtype=float)
        delta = naive_arr - arr
        beats = int(np.sum(arr < naive_arr))
        pct   = 100 * beats / N_ITERATIONS
        print(f"\n  {name} ({len(FEATURE_SETS[name])} features):")
        print(f"    RMSE  median={np.nanmedian(arr):.4f}  "
              f"IQR=[{np.nanpercentile(arr,25):.4f}, "
              f"{np.nanpercentile(arr,75):.4f}]  "
              f"range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
        print(f"    ΔRMSE median={np.nanmedian(delta):+.4f}  "
              f"IQR=[{np.nanpercentile(delta,25):+.4f}, "
              f"{np.nanpercentile(delta,75):+.4f}]")
        print(f"    Beats naive: {beats}/{N_ITERATIONS} ({pct:.1f}%)")

    # Head-to-head vs Old-3
    print(f"\n  Head-to-head vs Old-3:")
    old3 = np.array(results["Old-3"], dtype=float)
    for name in names:
        if name == "Old-3":
            continue
        arr     = np.array(results[name], dtype=float)
        better  = int(np.sum(arr < old3))
        pct     = 100 * better / N_ITERATIONS
        med_diff= float(np.nanmedian(arr) - np.nanmedian(old3))
        print(f"    {name} beats Old-3: {better}/{N_ITERATIONS} "
              f"({pct:.1f}%)  median diff={med_diff:+.4f} cm/h")

    print("\nDone.")


if __name__ == "__main__":
    main()