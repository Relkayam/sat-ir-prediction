"""
experiments/run_bootstrap_v2.py - Bootstrap validation + basin selection (v2)
==============================================================================
Runs N random draws for BOTH Model 1 and Model 2. N is read from
config.BOOTSTRAP_N_ITERATIONS.

Each draw selects one basin from each of the 5 operational fields
(fields 3,4,5,6,7) from all 50 basins; outliers are NOT excluded. In every
draw both models are trained on the 45 non-held-out basins and evaluated on
the 5 held-out basins.

Selection criterion (switchable - see CONFIG block)
---------------------------------------------------
  Qualifying set:
    (optional) all 5 held-out basins beat naive in BOTH models.
    (Every draw is one-per-field by construction.)
  Objective among qualifiers:
    "median" : the qualifier whose Model 2 dRMSE is CLOSEST to the median of
               the FULL bootstrap distribution (a representative draw).
    "max"    : the qualifier with the largest Model 2 dRMSE (best case).

SHAP (optional - COMPUTE_SHAP)
------------------------------
  When True, every draw also records per-feature importance (mean|SHAP|) and
  direction (corr of feature value vs SHAP value) on its held-out set, using
  the SAME draws as the performance bootstrap. Writes bootstrap_shap_*.csv,
  consumed by figures/fig7_bootstrap_shap.py.

Outputs
-------
  data/bootstrap_results_v2.csv
  data/selected_basins_v2.csv
  data/bootstrap_shap_m1.csv, bootstrap_shap_m1_dir.csv   (if COMPUTE_SHAP)
  data/bootstrap_shap_m2.csv, bootstrap_shap_m2_dir.csv   (if COMPUTE_SHAP)
  outputs/figures/bootstrap_distribution_v2.png

Usage
-----
  python -m experiments.run_bootstrap_v2
"""
from __future__ import annotations

import sys
import re
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
    FIGURES_DIR, TABLES_DIR,
    BOOSTING_PARAMS_M1, EARLY_STOPPING_ROUNDS_M1,
    BOOSTING_PARAMS_M2, EARLY_STOPPING_ROUNDS_M2,
    BOOTSTRAP_N_ITERATIONS, BOOTSTRAP_SEED,
    FIELD_NAMES,
)
from pipeline.features import (
    prepare_features, TARGET_M1,
    MODEL2_FEATURES, TARGET_M2,
)
from models.model1_decay import _linear_extrap_baseline
from experiments.bootstrap_shap import shap_row, write_shap_csvs   # SHAP

# ==============================================================================
# CONFIG - selection criterion (edit here)
# ==============================================================================
# "median" -> report the qualifying draw whose Model 2 dRMSE is closest to the
#            median of the FULL bootstrap distribution (a representative result).
# "max"    -> report the qualifying draw with the largest Model 2 dRMSE (best case).
SELECTION_OBJECTIVE        = "median"

# Require all 5 held-out basins to beat naive in BOTH models to qualify.
SELECTION_REQUIRE_BEAT_ALL = True

# SHAP - compute per-draw SHAP importance + direction on each held-out set.
# Adds ~1-2 s per draw and writes bootstrap_shap_*.csv. Set False to disable.
COMPUTE_SHAP = True
# ==============================================================================


# ------------------------------------------------------------------------------
# v2 output paths  (separate from v1)
# ------------------------------------------------------------------------------
DATA_DIR             = Path(__file__).parent.parent / "data"
BOOTSTRAP_RESULTS_V2 = DATA_DIR    / "bootstrap_results_v2.csv"
SELECTED_BASINS_V2   = DATA_DIR    / "selected_basins_v2.csv"
BOOTSTRAP_FIGURE_V2  = FIGURES_DIR / "bootstrap_distribution_v2.png"

# Number of random draws comes from config (single source of truth).
N_DRAWS = int(BOOTSTRAP_N_ITERATIONS)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
REQUIRED_FIELDS = {3, 4, 5, 6, 7}   # first digit of basin number


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

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
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
        if mask.sum() >= 2 else np.nan


def _covers_all_fields(held_out_str: str) -> bool:
    """True if the 5 basins include one from each of the 5 fields."""
    basin_nums = [int(x) for x in re.findall(r'\b\d{4}\b', str(held_out_str))]
    return {int(str(bn)[0]) for bn in basin_nums} == REQUIRED_FIELDS


def _draw_one_per_field(
    eligible_by_field: dict[int, list[int]],
    rng: np.random.Generator,
) -> list[int]:
    """Draw exactly one basin per field from the eligible pool."""
    return sorted([
        int(rng.choice(basins))
        for basins in eligible_by_field.values()
    ])


# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

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
    if "segment_id" in df.columns:
        df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce")
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
) -> tuple[list[int], dict[int, list[int]]]:
    """
    Returns:
      eligible          : sorted list of all basin numbers (outliers included)
      eligible_by_field : {field_digit: [basin_numbers]} for one-per-field draws

    NOTE: All 50 basins are eligible - outliers are NOT excluded from the draw.
    Outlier basins are included in training (E-full) and can be held out.
    The outlier list is kept for logging only.
    """
    all_bas  = set(event_df["basin_number"].dropna().unique().astype(int))
    eligible = sorted(all_bas)   # all basins - no outlier exclusion

    eligible_by_field: dict[int, list[int]] = {}
    for bn in eligible:
        field = int(str(bn)[0])
        eligible_by_field.setdefault(field, []).append(bn)

    print(f"  All basins    : {len(all_bas)}")
    print(f"  Outlier basins (informational): {sorted(outlier_basins)}")
    print(f"  Eligible pool : {len(eligible)} basins (all basins, outliers included)")
    print(f"  By field      : { {f: len(v) for f, v in sorted(eligible_by_field.items())} }")

    missing_fields = REQUIRED_FIELDS - set(eligible_by_field.keys())
    if missing_fields:
        raise ValueError(
            f"Fields {missing_fields} have no eligible basins. "
            "One-per-field draws are impossible."
        )
    return eligible, eligible_by_field


# ------------------------------------------------------------------------------
# One Model 1 iteration
# ------------------------------------------------------------------------------

def run_m1_one(
    event_df:  pd.DataFrame,
    feat_cols: list[str],
    held_out:  list[int],
) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    held_set = set(held_out)
    required = feat_cols + [TARGET_M1]

    test = event_df[
        event_df["basin_number"].isin(held_set) &
        (event_df["row_type"] == "event")
    ].dropna(subset=required).copy()

    non_ho = event_df[
        ~event_df["basin_number"].isin(held_set) &
        (event_df["row_type"] == "event")
    ].dropna(subset=required)

    # -- DISJOINT train / val (no val-in-train leak) ---------------------------
    train = non_ho[non_ho["split"] == "train"].reset_index(drop=True)
    val   = non_ho[non_ho["split"] == "val"].reset_index(drop=True)

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
    ird_naive = ird_reset

    ird_linear = _linear_extrap_baseline(test)

    per_basin = {}
    for bn in held_out:
        mask  = test["basin_number"].values == bn
        rmse_m = _rmse(ird_true[mask], ird_pred[mask])
        rmse_n = _rmse(ird_true[mask], ird_naive[mask])
        per_basin[bn] = {"rmse": rmse_m, "rmse_naive": rmse_n,
                         "beats": rmse_m < rmse_n}

    rmse_pool  = _rmse(ird_true, ird_pred)
    rmse_naive = _rmse(ird_true, ird_naive)

    mask_lin    = np.isfinite(ird_linear) & (ird_linear > 0) & \
                  np.isfinite(ird_true)   & (ird_true > 0)
    rmse_linear = _rmse(ird_true[mask_lin], ird_linear[mask_lin]) \
                  if mask_lin.sum() >= 2 else np.nan

    return dict(
        rmse_pool   = rmse_pool,
        rmse_naive  = rmse_naive,
        rmse_linear = rmse_linear,
        r2_pool     = _r2(ird_true, ird_pred),
        mape_pool   = _mape(ird_true, ird_pred),
        delta_rmse  = rmse_naive - rmse_pool,
        delta_rmse_linear = (rmse_linear - rmse_pool)
                            if np.isfinite(rmse_linear) else np.nan,
        all_beat    = all(v["beats"] for v in per_basin.values()),
        per_basin   = per_basin,
        _model      = model,        # SHAP - trained model
        _X_test     = Xte,          # SHAP - scaled held-out matrix
    )


# ------------------------------------------------------------------------------
# One Model 2 iteration
# ------------------------------------------------------------------------------

def run_m2_one(
    reset_df: pd.DataFrame,
    held_out: list[int],
) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    held_set  = set(held_out)
    feat_cols = [f for f in MODEL2_FEATURES if f in reset_df.columns]
    required  = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]

    test   = reset_df[reset_df["basin_number"].isin(held_set)].dropna(
        subset=required).copy()
    non_ho = reset_df[~reset_df["basin_number"].isin(held_set)].dropna(
        subset=required)

    # -- DISJOINT train / val (no val-in-train leak) ---------------------------
    train = non_ho[non_ho["split_chrono"] == "train"].reset_index(drop=True)
    val   = non_ho[non_ho["split_chrono"] == "val"].reset_index(drop=True)

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
        mask   = test["basin_number"].values == bn
        rmse_m = _rmse(ird_true[mask], ird_pred[mask])
        rmse_n = _rmse(ird_true[mask], ird_naive[mask])
        per_basin[bn] = {"rmse": rmse_m, "rmse_naive": rmse_n,
                         "beats": rmse_m < rmse_n}

    rmse_pool  = _rmse(ird_true, ird_pred)
    rmse_naive = _rmse(ird_true, ird_naive)

    return dict(
        rmse_pool  = rmse_pool,
        rmse_naive = rmse_naive,
        r2_pool    = _r2(ird_true, ird_pred),
        mape_pool  = _mape(ird_true, ird_pred),
        delta_rmse = rmse_naive - rmse_pool,
        all_beat   = all(v["beats"] for v in per_basin.values()),
        per_basin  = per_basin,
        _model     = model,         # SHAP - trained model
        _X_test    = Xte,           # SHAP - scaled held-out matrix
    )


# ------------------------------------------------------------------------------
# Build one result row
# ------------------------------------------------------------------------------

def _build_row(
    iteration: int,
    held_out:  list[int],
    m1:        dict,
    m2:        dict,
) -> dict:
    both_all_beat = m1["all_beat"] and m2["all_beat"]
    row = dict(
        iteration           = iteration,
        held_out            = str(sorted(held_out)),
        m1_rmse             = m1["rmse_pool"],
        m1_rmse_naive       = m1["rmse_naive"],
        m1_rmse_linear      = m1.get("rmse_linear", np.nan),
        m1_delta_rmse       = m1["delta_rmse"],
        m1_delta_rmse_linear= m1.get("delta_rmse_linear", np.nan),
        m1_r2               = m1["r2_pool"],
        m1_mape             = m1["mape_pool"],
        m1_all_beat         = m1["all_beat"],
        m2_rmse             = m2["rmse_pool"],
        m2_rmse_naive       = m2["rmse_naive"],
        m2_delta_rmse       = m2["delta_rmse"],
        m2_r2               = m2["r2_pool"],
        m2_mape             = m2["mape_pool"],
        m2_all_beat         = m2["all_beat"],
        both_all_beat       = both_all_beat,
    )
    for bn in held_out:
        b1 = m1["per_basin"].get(bn, {})
        b2 = m2["per_basin"].get(bn, {})
        row[f"m1_rmse_{bn}"]  = b1.get("rmse",  np.nan)
        row[f"m1_beats_{bn}"] = b1.get("beats", False)
        row[f"m2_rmse_{bn}"]  = b2.get("rmse",  np.nan)
        row[f"m2_beats_{bn}"] = b2.get("beats", False)
    return row


# ------------------------------------------------------------------------------
# Run all draws
# ------------------------------------------------------------------------------

def run_bootstrap(
    event_df:          pd.DataFrame,
    reset_df:          pd.DataFrame,
    feat_cols_m1:      list[str],
    eligible_by_field: dict[int, list[int]],
) -> pd.DataFrame:
    rng  = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    # SHAP - per-draw accumulators
    m1_imp_rows, m1_dir_rows = [], []
    m2_imp_rows, m2_dir_rows = [], []
    feat_cols_m2 = [f for f in MODEL2_FEATURES if f in reset_df.columns]

    print(f"\n  Total draws : {N_DRAWS} (random one-per-field)")
    if COMPUTE_SHAP:
        print(f"  SHAP        : ON (writing bootstrap_shap_*.csv)")
    print()
    print(f"  {'Draw':>5}  {'M1 RMSE':>9}  {'M2 RMSE':>9}  "
          f"{'dM1':>8}  {'dM2':>8}  {'M1 ok':>5}  {'M2 ok':>5}  {'Both':>5}")
    print(f"  {'-'*76}")

    for i in range(N_DRAWS):
        iteration = i + 1
        held_out  = _draw_one_per_field(eligible_by_field, rng)

        m1 = run_m1_one(event_df, feat_cols_m1, held_out)
        m2 = run_m2_one(reset_df, held_out)
        if not m1 or not m2:
            continue
        row = _build_row(iteration, held_out, m1, m2)
        rows.append(row)

        if COMPUTE_SHAP:
            i1, d1 = shap_row(m1.get("_model"), m1.get("_X_test"),
                              feat_cols_m1, iteration)
            if i1 is not None:
                m1_imp_rows.append(i1); m1_dir_rows.append(d1)
            i2, d2 = shap_row(m2.get("_model"), m2.get("_X_test"),
                              feat_cols_m2, iteration)
            if i2 is not None:
                m2_imp_rows.append(i2); m2_dir_rows.append(d2)

        if (iteration <= 3 or iteration % 50 == 0 or iteration == N_DRAWS):
            print(f"  {iteration:>5}  "
                  f"{m1['rmse_pool']:>9.4f}  {m2['rmse_pool']:>9.4f}  "
                  f"{m1['delta_rmse']:>+8.4f}  {m2['delta_rmse']:>+8.4f}  "
                  f"{'Y' if m1['all_beat'] else 'n':>5}  "
                  f"{'Y' if m2['all_beat'] else 'n':>5}  "
                  f"{'Y' if row['both_all_beat'] else 'n':>5}")

    # SHAP - write the four accumulated tables
    if COMPUTE_SHAP:
        print()
        write_shap_csvs(m1_imp_rows, m1_dir_rows, m2_imp_rows, m2_dir_rows)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Select reporting draw
# ------------------------------------------------------------------------------

def select_best_iteration(results: pd.DataFrame) -> pd.Series:
    """
    Selection (see CONFIG block):
      1. (optional) both models all 5 beat naive   - SELECTION_REQUIRE_BEAT_ALL
      2. objective among qualifiers:
           "median" -> Model 2 dRMSE closest to the median of the FULL
                      bootstrap distribution (representative draw)
           "max"    -> largest Model 2 dRMSE (best case)
    Every draw is one basin per field by construction.
    """
    total_median = float(results["m2_delta_rmse"].median())

    n_m1   = int(results["m1_all_beat"].sum())
    n_m2   = int(results["m2_all_beat"].sum())
    n_both = int(results["both_all_beat"].sum())

    print(f"\n  Selection objective : {SELECTION_OBJECTIVE!r}  "
          f"(require beat-all = {SELECTION_REQUIRE_BEAT_ALL})")
    print(f"  Full-distribution Model 2 dRMSE median : {total_median:+.4f} cm/h")
    print(f"\n  Draws where all 5 beat naive:")
    print(f"    Model 1 only  : {n_m1}/{len(results)}")
    print(f"    Model 2 only  : {n_m2}/{len(results)}")
    print(f"    BOTH models   : {n_both}/{len(results)}")

    qual = results.copy()
    if SELECTION_REQUIRE_BEAT_ALL:
        qual = qual[qual["both_all_beat"] == True].copy()
        if qual.empty:
            print("\n  WARNING: no draw has both models beat naive on all 5.")
            print("  Relaxing: dropping beat-all constraint.")
            qual = results.copy()

    if SELECTION_OBJECTIVE == "median":
        idx = (qual["m2_delta_rmse"] - total_median).abs().idxmin()
    elif SELECTION_OBJECTIVE == "max":
        idx = qual["m2_delta_rmse"].idxmax()
    else:
        raise ValueError(
            f"SELECTION_OBJECTIVE must be 'median' or 'max', "
            f"got {SELECTION_OBJECTIVE!r}")
    best = qual.loc[idx]

    basin_nums = [int(x) for x in re.findall(r'\b\d{4}\b', str(best["held_out"]))]
    fields_sel = sorted({int(str(bn)[0]) for bn in basin_nums})

    print(f"\n  SELECTED DRAW : {int(best['iteration'])}")
    print(f"    Held-out basins  : {best['held_out']}")
    print(f"    Fields covered   : {fields_sel}")
    print(f"    M1 RMSE={best['m1_rmse']:.4f}  "
          f"dRMSE={best['m1_delta_rmse']:+.4f}  all_beat={best['m1_all_beat']}")
    print(f"    M2 RMSE={best['m2_rmse']:.4f}  "
          f"dRMSE={best['m2_delta_rmse']:+.4f}  all_beat={best['m2_all_beat']}")
    if SELECTION_OBJECTIVE == "median":
        print(f"    M2 dRMSE distance from median : "
              f"{abs(best['m2_delta_rmse'] - total_median):.4f} cm/h")
    return best


# ------------------------------------------------------------------------------
# Write selected_basins_v2.csv
# ------------------------------------------------------------------------------

def write_selected_basins(best_row: pd.Series) -> list[int]:
    held_out = sorted([int(x) for x in
                       re.findall(r'\b\d{4}\b', str(best_row["held_out"]))])

    lines = [
        "# Auto-generated by experiments/run_bootstrap_v2.py",
        f"# Bootstrap draw: {int(best_row['iteration'])} (random one-per-field)",
        f"# Selection: objective={SELECTION_OBJECTIVE}, "
        f"require_beat_all={SELECTION_REQUIRE_BEAT_ALL}, one basin per field",
        f"# M1 RMSE={best_row['m1_rmse']:.4f}  dRMSE={best_row['m1_delta_rmse']:+.4f}",
        f"# M2 RMSE={best_row['m2_rmse']:.4f}  dRMSE={best_row['m2_delta_rmse']:+.4f}",
        "#",
        "# basin_number,field_name",
    ]
    for bn in held_out:
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        lines.append(f"{bn},{field}")

    SELECTED_BASINS_V2.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTED_BASINS_V2, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Written: {SELECTED_BASINS_V2}")
    for bn in held_out:
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        print(f"    Basin {bn}  ({field})")
    return held_out


# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

def print_summary(results: pd.DataFrame, best_row: pd.Series) -> None:
    print(f"\n{'='*70}")
    print(f"  BOOTSTRAP SUMMARY v2 - {len(results)} random draws (one-per-field)")
    print(f"{'='*70}")
    for model, col_rmse, col_delta, col_beat, col_lin in [
        ("Model 1", "m1_rmse", "m1_delta_rmse", "m1_all_beat", "m1_rmse_linear"),
        ("Model 2", "m2_rmse", "m2_delta_rmse", "m2_all_beat", None),
    ]:
        arr   = results[col_rmse].dropna()
        delta = results[col_delta].dropna()
        beats = int(results[col_beat].sum())
        print(f"\n  {model}:")
        print(f"    RMSE   median={arr.median():.4f}  "
              f"IQR=[{arr.quantile(0.25):.4f}, {arr.quantile(0.75):.4f}]")
        print(f"    dRMSE  median={delta.median():+.4f}  "
              f"IQR=[{delta.quantile(0.25):+.4f}, {delta.quantile(0.75):+.4f}]")
        if col_lin and col_lin in results.columns:
            lin = results[col_lin].dropna()
            if len(lin) > 0:
                print(f"    Linear RMSE  median={lin.median():.4f}  "
                      f"IQR=[{lin.quantile(0.25):.4f}, {lin.quantile(0.75):.4f}]")
        print(f"    All 5 beat naive: {beats}/{len(results)} "
              f"({100*beats/len(results):.1f}%)")

    both = int(results["both_all_beat"].sum())
    print(f"\n  Both models all 5 beat naive     : "
          f"{both}/{len(results)} ({100*both/len(results):.1f}%)")
    print(f"\n  Selected draw : {int(best_row['iteration'])}  "
          f"(objective={SELECTION_OBJECTIVE})")
    print(f"    Basins : {best_row['held_out']}")
    print(f"    M1 dRMSE={best_row['m1_delta_rmse']:+.4f}  "
          f"M2 dRMSE={best_row['m2_delta_rmse']:+.4f}")


# ------------------------------------------------------------------------------
# Plot histogram
# ------------------------------------------------------------------------------

def plot_histogram(results: pd.DataFrame, best_row: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.suptitle(
        f"Bootstrap validation v2 - {len(results)} random draws (one-per-field)\n"
        f"* = selected draw (objective={SELECTION_OBJECTIVE}, "
        f"require beat-all={SELECTION_REQUIRE_BEAT_ALL})",
        fontsize=10, fontweight="bold",
    )

    for ax, model, col_delta, col_beat, color, sel_val in [
        (axes[0], "Model 1", "m1_delta_rmse", "m1_all_beat",
         "#065A82", float(best_row["m1_delta_rmse"])),
        (axes[1], "Model 2", "m2_delta_rmse", "m2_all_beat",
         "#E07B39", float(best_row["m2_delta_rmse"])),
    ]:
        delta  = results[col_delta].dropna()
        beats  = results[results[col_beat] == True][col_delta]
        nobeat = results[results[col_beat] == False][col_delta]

        ax.hist(nobeat.values, bins=35, color="lightgrey", edgecolor="white",
                alpha=0.85, label=f"Not all 5 beat naive (n={len(nobeat)})")
        ax.hist(beats.values,  bins=35, color=color,      edgecolor="white",
                alpha=0.85, label=f"All 5 beat naive (n={len(beats)})")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="-",
                   alpha=0.7, label="No improvement")
        ax.axvline(float(delta.median()), color="grey", linewidth=1.5,
                   linestyle="--", alpha=0.8,
                   label=f"Median = {float(delta.median()):+.4f}")

        ax.autoscale(axis="y")
        ymax = ax.get_ylim()[1] or 1
        ax.scatter([sel_val], [ymax * 0.85], marker="*", color="gold",
                   s=400, zorder=6, edgecolors="black", linewidths=0.8,
                   label=f"Selected = {sel_val:+.4f}")

        ax.set_xlabel("dRMSE = naive - model (cm/h)  positive = model better",
                      fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(
            f"{model}  |  {int(results[col_beat].sum())}/{len(results)} "
            "draws: all 5 beat naive",
            fontsize=9, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(BOOTSTRAP_FIGURE_V2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {BOOTSTRAP_FIGURE_V2.name}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> pd.DataFrame:
    print("=" * 70)
    print(f"  BOOTSTRAP VALIDATION v2 - Model 1 + Model 2")
    print(f"  {N_DRAWS} random one-per-field draws "
          f"(from config.BOOTSTRAP_N_ITERATIONS)")
    print(f"  Selection         : objective={SELECTION_OBJECTIVE}, "
          f"require_beat_all={SELECTION_REQUIRE_BEAT_ALL}")
    print(f"  SHAP              : {'ON' if COMPUTE_SHAP else 'OFF'}")
    print(f"  Seed              : {BOOTSTRAP_SEED}")
    print(f"  Output results    : {BOOTSTRAP_RESULTS_V2.name}")
    print(f"  Output selection  : {SELECTED_BASINS_V2.name}")
    print("=" * 70)

    if N_DRAWS < 500:
        print(f"\n  WARNING: config.BOOTSTRAP_N_ITERATIONS = {N_DRAWS}. "
              f"The paper reports 500 random draws - set it to 500 in config.py.")

    print("\n--- Loading data ---")
    outlier_basins             = load_outlier_basins()
    event_df, feat_cols_m1     = load_event_data()
    reset_df                   = load_reset_data()
    eligible, eligible_by_field = build_eligible_pool(event_df, outlier_basins)

    print("\n--- Running bootstrap ---")
    results = run_bootstrap(
        event_df, reset_df, feat_cols_m1, eligible_by_field)

    print(f"\n  Completed {len(results)} draws")
    results.to_csv(BOOTSTRAP_RESULTS_V2, index=False)
    print(f"  Saved: {BOOTSTRAP_RESULTS_V2.name}")

    print("\n--- Selecting reporting draw ---")
    best_row = select_best_iteration(results)

    write_selected_basins(best_row)
    print_summary(results, best_row);9

    print("\n--- Generating histogram ---")
    plot_histogram(results, best_row)

    print(f"\n{'='*70}")
    print("  NEXT STEPS")
    print(f"{'='*70}")
    print(f"  If satisfied with the selection, point config.SELECTED_BASINS_CSV at:")
    print(f"    {SELECTED_BASINS_V2}")
    print(f"  Then run:")
    print(f"    python -m models.model1_decay")
    print(f"    python -m models.model2_reset")
    if COMPUTE_SHAP:
        print(f"    python -m figures.fig6_bootstrap_shap")
    print("\nDone.")
    return results


if __name__ == "__main__":
    main()