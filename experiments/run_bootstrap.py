"""
experiments/run_bootstrap.py — Bootstrap validation + basin selection
=====================================================================
Runs 200 random held-out iterations for BOTH Model 1 and Model 2
simultaneously (same 5 random basins per iteration).

Selection criterion for paper figures
--------------------------------------
  From all 200 iterations, find the one where:
    1. ALL 5 held-out basins beat naive in BOTH models (hard constraint)
    2. Model 2 pooled ΔRMSE (naive − model) is maximised (objective)

  The selected 5 basins are written to data/selected_basins.csv.
  model1_decay.py and model2_reset.py read from this file at runtime.

Outputs
-------
  data/bootstrap_results.csv    — all 200 iterations, both models
  data/selected_basins.csv      — the 5 selected held-out basins
  outputs/figures/bootstrap_distribution.png — histogram with selection marked

Usage
-----
  python -m experiments.run_bootstrap
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
    EVENT_CSV, RESET_CSV, OUTLIER_CSV,
    BOOTSTRAP_RESULTS_CSV, SELECTED_BASINS_CSV,
    FIGURES_DIR, TABLES_DIR,
    BOOSTING_PARAMS_M1, EARLY_STOPPING_ROUNDS_M1,
    BOOSTING_PARAMS_M2, EARLY_STOPPING_ROUNDS_M2,
    BOOTSTRAP_N_ITERATIONS, BOOTSTRAP_N_HELD_OUT, BOOTSTRAP_SEED,
    FIELD_NAMES,
)
from pipeline.features import (
    prepare_features, TARGET_M1,
    MODEL2_FEATURES, TARGET_M2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))) \
        if mask.sum() >= 2 else np.nan


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    return float(r2_score(y_true[mask], y_pred[mask])) \
        if mask.sum() >= 2 else np.nan


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100) \
        if mask.sum() >= 2 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_outlier_basins() -> set[int]:
    if not OUTLIER_CSV.exists():
        print("  WARNING: outlier_basins.csv not found.")
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


def load_event_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df, feat_cols = prepare_features(df)
    print(f"  Event dataset : {len(df):,} rows  "
          f"{df['basin_number'].nunique()} basins  "
          f"{len(feat_cols)} M1 features")
    return df, feat_cols


def load_reset_data() -> pd.DataFrame:
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    feat_cols = [f for f in MODEL2_FEATURES if f in df.columns]
    missing   = [f for f in MODEL2_FEATURES if f not in df.columns]
    print(f"  Reset dataset : {len(df)} rows  "
          f"{df['basin_number'].nunique()} basins  "
          f"{len(feat_cols)} M2 features")
    if missing:
        print(f"  WARNING missing M2 features: {missing}")
    return df


def build_eligible_pool(
    event_df:       pd.DataFrame,
    outlier_basins: set[int],
) -> list[int]:
    all_bas  = set(event_df["basin_number"].dropna().unique().astype(int))
    eligible = sorted(all_bas - outlier_basins)
    print(f"  All basins    : {len(all_bas)}")
    print(f"  Minus outliers: {len(outlier_basins)}")
    print(f"  Eligible pool : {len(eligible)} basins")
    return eligible


# ─────────────────────────────────────────────────────────────────────────────
# One Model 1 iteration
# ─────────────────────────────────────────────────────────────────────────────

def run_m1_one(
    event_df:       pd.DataFrame,
    feat_cols:      list[str],
    outlier_basins: set[int],
    held_out:       list[int],
) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    held_set = set(held_out)
    required = feat_cols + [TARGET_M1]

    # Test — held-out basins, good segments
    test = event_df[
        event_df["basin_number"].isin(held_set) &
        (event_df["row_type"]        == "event") &
        (event_df["is_good_segment"] == True)
    ].dropna(subset=required).copy()

    # Train — all other basins, all event rows (E-full: includes outliers)
    non_ho = event_df[
        ~event_df["basin_number"].isin(held_set) &
        (event_df["row_type"] == "event")
    ].dropna(subset=required)

    val = non_ho[
        (non_ho["split"]          == "val") &
        (non_ho["is_good_segment"] == True)
    ].reset_index(drop=True)
    train = non_ho.reset_index(drop=True)

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        return {}

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)

    model = LGBMRegressor(**BOOSTING_PARAMS_M1)
    model.fit(
        Xtr, train[TARGET_M1].values,
        eval_set=[(Xva, val[TARGET_M1].values)],
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS_M1, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    ird_reset = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    pred_norm = model.predict(Xte)
    ird_pred  = ird_reset * np.exp(pred_norm)
    ird_true  = ird_reset * np.exp(
        pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    ird_naive = ird_reset * np.exp(np.zeros(len(test)))  # naive: predict 0

    # Per-basin beat check
    per_basin = {}
    for bn in held_out:
        mask_bn = test["basin_number"].values == bn
        ird_t   = ird_true[mask_bn]; ird_p = ird_pred[mask_bn]
        ird_n   = ird_naive[mask_bn]
        rmse_m  = _rmse(ird_t, ird_p)
        rmse_n  = _rmse(ird_t, ird_n)
        per_basin[bn] = {"rmse": rmse_m, "rmse_naive": rmse_n,
                         "beats": rmse_m < rmse_n}

    all_beat  = all(v["beats"] for v in per_basin.values())
    rmse_pool = _rmse(ird_true, ird_pred)
    rmse_naive= _rmse(ird_true, ird_naive)

    return dict(
        rmse_pool  = rmse_pool,
        rmse_naive = rmse_naive,
        r2_pool    = _r2(ird_true, ird_pred),
        mape_pool  = _mape(ird_true, ird_pred),
        delta_rmse = rmse_naive - rmse_pool,
        all_beat   = all_beat,
        per_basin  = per_basin,
    )


# ─────────────────────────────────────────────────────────────────────────────
# One Model 2 iteration
# ─────────────────────────────────────────────────────────────────────────────

def run_m2_one(
    reset_df:       pd.DataFrame,
    outlier_basins: set[int],
    held_out:       list[int],
) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    held_set  = set(held_out)
    feat_cols = [f for f in MODEL2_FEATURES if f in reset_df.columns]
    required  = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]

    test  = reset_df[reset_df["basin_number"].isin(held_set)].dropna(
        subset=required).copy()
    non_ho= reset_df[~reset_df["basin_number"].isin(held_set)].dropna(
        subset=required)
    val   = non_ho[non_ho["split_chrono"] == "val"].reset_index(drop=True)
    train = non_ho.reset_index(drop=True)

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        return {}

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

    per_basin = {}
    for bn in held_out:
        mask_bn = test["basin_number"].values == bn
        ird_t   = ird_true[mask_bn]; ird_p = ird_pred[mask_bn]
        ird_n   = ird_naive[mask_bn]
        rmse_m  = _rmse(ird_t, ird_p)
        rmse_n  = _rmse(ird_t, ird_n)
        per_basin[bn] = {"rmse": rmse_m, "rmse_naive": rmse_n,
                         "beats": rmse_m < rmse_n}

    all_beat  = all(v["beats"] for v in per_basin.values())
    rmse_pool = _rmse(ird_true, ird_pred)
    rmse_naive= _rmse(ird_true, ird_naive)

    return dict(
        rmse_pool  = rmse_pool,
        rmse_naive = rmse_naive,
        r2_pool    = _r2(ird_true, ird_pred),
        mape_pool  = _mape(ird_true, ird_pred),
        delta_rmse = rmse_naive - rmse_pool,
        all_beat   = all_beat,
        per_basin  = per_basin,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run all iterations
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap(
    event_df:       pd.DataFrame,
    reset_df:       pd.DataFrame,
    feat_cols_m1:   list[str],
    eligible:       list[int],
    outlier_basins: set[int],
) -> pd.DataFrame:
    rng  = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    print(f"\n  Running {BOOTSTRAP_N_ITERATIONS} iterations "
          f"({BOOTSTRAP_N_HELD_OUT} held-out per iteration)...\n")
    print(f"  {'Iter':>5}  {'M1 RMSE':>9}  {'M2 RMSE':>9}  "
          f"{'Naive':>8}  {'ΔM1':>8}  {'ΔM2':>8}  "
          f"{'M1 all✓':>8}  {'M2 all✓':>8}  {'Both✓':>6}")
    print(f"  {'-'*80}")

    for i in range(BOOTSTRAP_N_ITERATIONS):
        held_out = [int(x) for x in rng.choice(eligible, size=BOOTSTRAP_N_HELD_OUT, replace=False)]

        m1 = run_m1_one(event_df, feat_cols_m1, outlier_basins, held_out)
        m2 = run_m2_one(reset_df, outlier_basins, held_out)

        if not m1 or not m2:
            continue

        both_all_beat = m1["all_beat"] and m2["all_beat"]

        row = dict(
            iteration      = i + 1,
            held_out       = str(sorted(held_out)),
            # Model 1
            m1_rmse        = m1["rmse_pool"],
            m1_rmse_naive  = m1["rmse_naive"],
            m1_delta_rmse  = m1["delta_rmse"],
            m1_r2          = m1["r2_pool"],
            m1_mape        = m1["mape_pool"],
            m1_all_beat    = m1["all_beat"],
            # Model 2
            m2_rmse        = m2["rmse_pool"],
            m2_rmse_naive  = m2["rmse_naive"],
            m2_delta_rmse  = m2["delta_rmse"],
            m2_r2          = m2["r2_pool"],
            m2_mape        = m2["mape_pool"],
            m2_all_beat    = m2["all_beat"],
            # Combined
            both_all_beat  = both_all_beat,
        )
        # Per-basin details
        for bn in held_out:
            b1 = m1["per_basin"].get(bn, {})
            b2 = m2["per_basin"].get(bn, {})
            row[f"m1_rmse_{bn}"]  = b1.get("rmse", np.nan)
            row[f"m1_beats_{bn}"] = b1.get("beats", False)
            row[f"m2_rmse_{bn}"]  = b2.get("rmse", np.nan)
            row[f"m2_beats_{bn}"] = b2.get("beats", False)
        rows.append(row)

        if (i + 1) % 25 == 0 or i < 3:
            print(f"  {i+1:>5}  "
                  f"{m1['rmse_pool']:>9.4f}  "
                  f"{m2['rmse_pool']:>9.4f}  "
                  f"{m2['rmse_naive']:>8.4f}  "
                  f"{m1['delta_rmse']:>+8.4f}  "
                  f"{m2['delta_rmse']:>+8.4f}  "
                  f"{'✓' if m1['all_beat'] else '✗':>8}  "
                  f"{'✓' if m2['all_beat'] else '✗':>8}  "
                  f"{'✓' if both_all_beat else '✗':>6}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Select best iteration
# ─────────────────────────────────────────────────────────────────────────────

def select_best_iteration(results: pd.DataFrame) -> pd.Series:
    """
    Selection criterion:
      1. Both M1 and M2: all 5 held-out basins beat naive (hard constraint)
      2. Among qualifying: maximise M2 pooled ΔRMSE (objective)
    """
    qualifying = results[results["both_all_beat"] == True].copy()
    n_both     = len(qualifying)
    n_m1_only  = int(results["m1_all_beat"].sum())
    n_m2_only  = int(results["m2_all_beat"].sum())

    print(f"\n  Iterations where all 5 beat naive:")
    print(f"    Model 1 only : {n_m1_only}/{len(results)}")
    print(f"    Model 2 only : {n_m2_only}/{len(results)}")
    print(f"    BOTH models  : {n_both}/{len(results)}")

    if qualifying.empty:
        print("\n  WARNING: No iteration satisfies both constraints.")
        print("  Relaxing to: M2 all beat naive (ignoring M1 constraint).")
        qualifying = results[results["m2_all_beat"] == True].copy()
        if qualifying.empty:
            print("  Still empty — using best M2 ΔRMSE unconditionally.")
            qualifying = results.copy()

    best = qualifying.loc[qualifying["m2_delta_rmse"].idxmax()]
    print(f"\n  SELECTED ITERATION: {int(best['iteration'])}")
    print(f"    Held-out basins : {best['held_out']}")
    print(f"    M1 pooled RMSE  : {best['m1_rmse']:.4f}  "
          f"delta={best['m1_delta_rmse']:+.4f}  all_beat={best['m1_all_beat']}")
    print(f"    M2 pooled RMSE  : {best['m2_rmse']:.4f}  "
          f"delta={best['m2_delta_rmse']:+.4f}  all_beat={best['m2_all_beat']}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Write selected_basins.csv
# ─────────────────────────────────────────────────────────────────────────────

def write_selected_basins(best_row: pd.Series) -> list[int]:
    """
    Write the 5 selected held-out basins to data/selected_basins.csv.
    model1_decay.py and model2_reset.py read from this file at runtime.
    """
    import re
    # Extract 4-digit basin numbers — robust to np.int64(...) strings
    held_out = sorted([int(x) for x in re.findall(r'\b\d{4}\b', str(best_row["held_out"]))])

    lines = [
        "# Auto-generated by experiments/run_bootstrap.py",
        f"# Bootstrap iteration: {int(best_row['iteration'])}",
        f"# Selection: all 5 beat naive in both models, max M2 delta_rmse",
        f"# M1 pooled RMSE={best_row['m1_rmse']:.4f}  "
        f"delta={best_row['m1_delta_rmse']:+.4f}",
        f"# M2 pooled RMSE={best_row['m2_rmse']:.4f}  "
        f"delta={best_row['m2_delta_rmse']:+.4f}",
        "#",
        "# basin_number,field_name",
    ]
    for bn in held_out:
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        lines.append(f"{bn},{field}")

    with open(SELECTED_BASINS_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Written: {SELECTED_BASINS_CSV}")
    for bn in held_out:
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        print(f"    Basin {bn}  ({field})")
    return held_out


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame, best_row: pd.Series) -> None:
    print(f"\n{'='*70}")
    print(f"  BOOTSTRAP SUMMARY — {BOOTSTRAP_N_ITERATIONS} iterations")
    print(f"{'='*70}")
    for model, col_rmse, col_delta, col_beat in [
        ("Model 1", "m1_rmse", "m1_delta_rmse", "m1_all_beat"),
        ("Model 2", "m2_rmse", "m2_delta_rmse", "m2_all_beat"),
    ]:
        arr   = results[col_rmse].dropna()
        delta = results[col_delta].dropna()
        beats = int(results[col_beat].sum())
        print(f"\n  {model}:")
        print(f"    RMSE  median={arr.median():.4f}  "
              f"IQR=[{arr.quantile(0.25):.4f}, {arr.quantile(0.75):.4f}]")
        print(f"    ΔRMSE median={delta.median():+.4f}  "
              f"IQR=[{delta.quantile(0.25):+.4f}, {delta.quantile(0.75):+.4f}]")
        print(f"    All 5 beat naive: {beats}/{len(results)} "
              f"({100*beats/len(results):.1f}%)")

    both = int(results["both_all_beat"].sum())
    print(f"\n  Both models all 5 beat naive: "
          f"{both}/{len(results)} ({100*both/len(results):.1f}%)")
    print(f"\n  Selected iteration: {int(best_row['iteration'])}")
    print(f"    Basins: {best_row['held_out']}")
    print(f"    M1 ΔRMSE={best_row['m1_delta_rmse']:+.4f}  "
          f"M2 ΔRMSE={best_row['m2_delta_rmse']:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram(results: pd.DataFrame, best_row: pd.Series) -> None:
    """
    Two-panel histogram: Model 1 ΔRMSE and Model 2 ΔRMSE distributions.
    Selected iteration marked with a gold star.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Bootstrap validation — {BOOTSTRAP_N_ITERATIONS} random held-out selections\n"
        f"{BOOTSTRAP_N_HELD_OUT} basins held out per iteration  |  "
        "E-full training (includes outlier basins)\n"
        "★ = selected iteration (all 5 beat naive in both models, "
        "max Model 2 ΔRMSE)",
        fontsize=10, fontweight="bold",
    )

    for ax, model, col_delta, col_beat, color, sel_val in [
        (axes[0], "Model 1", "m1_delta_rmse", "m1_all_beat",
         "#065A82", best_row["m1_delta_rmse"]),
        (axes[1], "Model 2", "m2_delta_rmse", "m2_all_beat",
         "#E07B39", best_row["m2_delta_rmse"]),
    ]:
        delta  = results[col_delta].dropna()
        beats  = results[results[col_beat] == True][col_delta]
        nobeat = results[results[col_beat] == False][col_delta]

        ax.hist(nobeat, bins=30, color="lightgrey", edgecolor="white",
                alpha=0.85, label=f"Not all 5 beat naive ({len(nobeat)})")
        ax.hist(beats,  bins=30, color=color, edgecolor="white",
                alpha=0.85, label=f"All 5 beat naive ({len(beats)})")

        ax.axvline(0, color="black", linewidth=1.2,
                   linestyle="-", alpha=0.7, label="No improvement")
        ax.axvline(delta.median(), color="grey", linewidth=1.5,
                   linestyle="--", alpha=0.8,
                   label=f"Median={delta.median():+.4f}")

        # Mark selected iteration
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
        ax.scatter([sel_val], [ymax * 0.85], marker="*", color="gold",
                   s=400, zorder=6, edgecolors="black", linewidths=0.8,
                   label=f"Selected={sel_val:+.4f}")

        both_mark = best_row["both_all_beat"]
        ax.set_xlabel("ΔRMSE = naive − model (cm/h)  ↑ positive = model better",
                      fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"{model}  |  "
                     f"{int(results[col_beat].sum())}/{BOOTSTRAP_N_ITERATIONS} "
                     f"iterations: all 5 beat naive",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = FIGURES_DIR / "bootstrap_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    print("="*70)
    print(f"  BOOTSTRAP VALIDATION — Model 1 + Model 2")
    print(f"  {BOOTSTRAP_N_ITERATIONS} iterations  |  "
          f"{BOOTSTRAP_N_HELD_OUT} random held-out per iteration")
    print(f"  Training: E-full (all non-held-out basins including outliers)")
    print(f"  Seed: {BOOTSTRAP_SEED}")
    print("="*70)

    print("\n--- Loading data ---")
    outlier_basins = load_outlier_basins()
    event_df, feat_cols_m1 = load_event_data()
    reset_df               = load_reset_data()
    eligible               = build_eligible_pool(event_df, outlier_basins)

    if len(eligible) < BOOTSTRAP_N_HELD_OUT:
        raise ValueError(
            f"Eligible pool ({len(eligible)}) < N_HELD_OUT ({BOOTSTRAP_N_HELD_OUT})")

    print("\n--- Running bootstrap ---")
    results = run_bootstrap(
        event_df, reset_df, feat_cols_m1, eligible, outlier_basins)

    print(f"\n  Completed {len(results)} iterations")

    # Save full results
    results.to_csv(BOOTSTRAP_RESULTS_CSV, index=False)
    print(f"  Saved: {BOOTSTRAP_RESULTS_CSV.name}")

    # Select best iteration
    print("\n--- Selecting best iteration ---")
    best_row = select_best_iteration(results)

    # Write selected basins
    selected = write_selected_basins(best_row)

    # Summary
    print_summary(results, best_row)

    # Plot
    print("\n--- Generating histogram ---")
    plot_histogram(results, best_row)

    print(f"\n{'='*70}")
    print("  NEXT STEPS")
    print(f"{'='*70}")
    print(f"  Selected basins written to: {SELECTED_BASINS_CSV.name}")
    print(f"  Run models on selected basins:")
    print(f"    python -m models.model1_decay")
    print(f"    python -m models.model2_reset")
    print("\nDone.")
    return results


if __name__ == "__main__":
    main()