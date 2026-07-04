"""
experiments/tune_hyperparams.py — Leak-free hyperparameter grid search
======================================================================
ONE-TIME grid search over LightGBM hyperparameters for Model 1 and Model 2,
run on ALL 50 basins with GROUPED cross-validation (no held-out test concept
here — this step CHOOSES hyperparameters; held-out evaluation happens later in
run_bootstrap_v2.py with the params FROZEN).

Why this is leak-free with respect to the reported results
----------------------------------------------------------
  * The search optimizes the validation-fold RMSE on the SCALE-FREE TARGET
    (eta = ln(IRD/rho) for Model 1, delta = ln(rho_i/rho_{i-1}) for Model 2).
    Scoring in target space (not raw cm/h) keeps the several-fold between-basin
    scale from dominating selection — consistent with the Section 2.3 argument
    that the log-ratio target is what makes basins comparable. The PAPER still
    reports raw-IRD metrics; only hyperparameter SELECTION is in target space.
  * Grouped folds prevent within-group leakage:
        Model 1 — folds grouped by segment_id  (a segment is never split)
        Model 2 — folds grouped by basin_number (a basin is never split)
  * The five reported held-out basins are selected AFTERWARDS by the bootstrap,
    and are scored as TEST only after these params are frozen. Tuning never
    sees a held-out basin's test metric, so there is no test-set leakage.
    (Design choice (i): tune on the full pool with grouped CV, then freeze.)

What it writes
--------------
  data/best_params_m1.json
  data/best_params_m2.json
  outputs/tables/hyperparam_search_m1.xlsx   (full grid, sorted by mean CV RMSE)
  outputs/tables/hyperparam_search_m2.xlsx

config.py loads the JSONs into BOOSTING_PARAMS_M1/M2 if present, else falls
back to its in-file defaults (so the repo still runs before the search is run).

Usage
-----
  python -m experiments.tune_hyperparams
  python -m experiments.tune_hyperparams --quick   # tiny grid, smoke test
"""
from __future__ import annotations

import sys
import json
import argparse
import warnings
import itertools
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, RESET_CSV, DATA_DIR, TABLES_DIR,
    EARLY_STOPPING_ROUNDS_M1, EARLY_STOPPING_ROUNDS_M2,
    RANDOM_SEED,
)
from pipeline.features import (
    prepare_features, TARGET_M1,
    MODEL2_FEATURES, TARGET_M2,
)

# ──────────────────────────────────────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────────────────────────────────────
BEST_PARAMS_M1 = DATA_DIR / "best_params_m1.json"
BEST_PARAMS_M2 = DATA_DIR / "best_params_m2.json"

# ──────────────────────────────────────────────────────────────────────────────
# Search space (54 points per model = 3 × 3 × 3 × 2)
# ──────────────────────────────────────────────────────────────────────────────
SEARCH_GRID = {
    "num_leaves":        [15, 31, 63],
    "learning_rate":     [0.02, 0.05, 0.1],
    "reg_lambda":        [0.0, 1.0, 5.0],
    "min_child_samples": [10, 20],
}

# Fixed params shared by every grid point (knobs NOT searched).
FIXED_PARAMS = dict(
    n_estimators     = 1000,
    max_depth        = -1,
    subsample        = 0.8,
    feature_fraction = 0.8,
    reg_alpha        = 0.1,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
    verbose          = -1,
)

N_FOLDS = 5


def _grid_points() -> list[dict]:
    keys = list(SEARCH_GRID.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*SEARCH_GRID.values())]


# ──────────────────────────────────────────────────────────────────────────────
# One grouped-CV evaluation of a single parameter set
# ──────────────────────────────────────────────────────────────────────────────

def _cv_rmse_target(
    df:        pd.DataFrame,
    feat_cols: list[str],
    target:    str,
    groups:    np.ndarray,
    params:    dict,
    es_rounds: int,
) -> float:
    """
    Mean RMSE on the SCALE-FREE TARGET (eta for Model 1, delta for Model 2)
    across GroupKFold folds for one parameter set.

    Rationale: the log-ratio target removes the several-fold between-basin
    scale (Section 2.3), so scoring the search on it avoids the high-IRD
    basins dominating a raw-cm/h RMSE and biasing selection toward flat,
    over-regularized predictors. The PAPER still reports raw-IRD metrics;
    only hyperparameter SELECTION is done in target space.

    Within each fold the inner split uses one grouped slice for early stopping,
    so the fit mirrors production (n_estimators=1000 + early stopping).
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    gkf   = GroupKFold(n_splits=N_FOLDS)
    rmses = []

    for tr_idx, va_idx in gkf.split(df, groups=groups):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        # Inner early-stopping split: carve a small grouped slice out of train.
        # Use the LAST 15% of train groups as the ES val (grouped, disjoint).
        tr_groups = groups[tr_idx]
        uniq      = pd.unique(tr_groups)
        n_es      = max(1, int(round(len(uniq) * 0.15)))
        es_groups = set(uniq[-n_es:])
        es_mask   = np.isin(tr_groups, list(es_groups))

        fit_tr = tr[~es_mask]
        fit_es = tr[es_mask]
        if len(fit_tr) < 20 or len(fit_es) < 5 or len(va) < 2:
            continue

        sc  = StandardScaler()
        Xtr = sc.fit_transform(fit_tr[feat_cols].values)
        Xes = sc.transform(fit_es[feat_cols].values)
        Xva = sc.transform(va[feat_cols].values)

        model = LGBMRegressor(**{**FIXED_PARAMS, **params})
        model.fit(
            Xtr, fit_tr[target].values,
            eval_set=[(Xes, fit_es[target].values)],
            callbacks=[early_stopping(es_rounds, verbose=False),
                       log_evaluation(period=-1)],
        )

        y_true = pd.to_numeric(va[target], errors="coerce").values
        y_pred = model.predict(Xva)
        mask   = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(
            y_true[mask], y_pred[mask]))))

    return float(np.mean(rmses)) if rmses else np.inf


# ──────────────────────────────────────────────────────────────────────────────
# Model 1 search
# ──────────────────────────────────────────────────────────────────────────────

def search_m1(quick: bool = False) -> dict:
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    df, feat_cols = prepare_features(df)

    df = df[(df["row_type"] == "event")].dropna(
        subset=feat_cols + [TARGET_M1, "IRD_at_reset"]).reset_index(drop=True)
    groups = df["segment_id"].values

    print(f"  Model 1 search pool: {len(df):,} events  "
          f"{df['basin_number'].nunique()} basins  "
          f"{df['segment_id'].nunique()} segments (grouped by segment)")
    return _run_search(df, feat_cols, TARGET_M1, groups,
                       EARLY_STOPPING_ROUNDS_M1,
                       BEST_PARAMS_M1, TABLES_DIR / "hyperparam_search_m1.xlsx",
                       "Model 1", quick)


# ──────────────────────────────────────────────────────────────────────────────
# Model 2 search
# ──────────────────────────────────────────────────────────────────────────────

def search_m2(quick: bool = False) -> dict:
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    feat_cols = [f for f in MODEL2_FEATURES if f in df.columns]

    df = df.dropna(subset=feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw",
                                       "IRD_at_reset"]).reset_index(drop=True)
    groups = df["basin_number"].values

    print(f"  Model 2 search pool: {len(df):,} resets  "
          f"{df['basin_number'].nunique()} basins (grouped by basin)")
    return _run_search(df, feat_cols, TARGET_M2, groups,
                       EARLY_STOPPING_ROUNDS_M2,
                       BEST_PARAMS_M2, TABLES_DIR / "hyperparam_search_m2.xlsx",
                       "Model 2", quick)


# ──────────────────────────────────────────────────────────────────────────────
# Shared search driver
# ──────────────────────────────────────────────────────────────────────────────

def _run_search(df, feat_cols, target, groups, es_rounds,
                out_json: Path, out_xlsx: Path, label: str,
                quick: bool) -> dict:
    grid = _grid_points()
    if quick:
        grid = grid[:3]
        print(f"  [QUICK] evaluating {len(grid)} of {len(_grid_points())} points")

    rows = []
    for k, params in enumerate(grid, 1):
        rmse = _cv_rmse_target(df, feat_cols, target, groups,
                               params, es_rounds)
        rows.append({**params, "cv_rmse_target": rmse})
        print(f"    [{label}] {k:>2}/{len(grid)}  "
              f"{params}  ->  CV RMSE (target) = {rmse:.4f}")

    res = pd.DataFrame(rows).sort_values("cv_rmse_target").reset_index(drop=True)
    res.to_excel(out_xlsx, index=False)
    print(f"  Saved: {out_xlsx.name}")

    best = res.iloc[0].to_dict()
    best_params = {k: (int(best[k]) if k in ("num_leaves", "min_child_samples")
                       else float(best[k]))
                   for k in SEARCH_GRID}
    frozen = {**FIXED_PARAMS, **best_params}

    with open(out_json, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"  Best {label}: {best_params}  (CV RMSE target = {best['cv_rmse_target']:.4f})")
    print(f"  Wrote: {out_json.name}")
    return frozen


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(quick: bool = False) -> None:
    print("=" * 70)
    print("  HYPERPARAMETER GRID SEARCH (leak-free, grouped CV, one-time)")
    print(f"  Grid points per model : {len(_grid_points())}")
    print(f"  Folds                 : {N_FOLDS} (GroupKFold)")
    print(f"  Scoring               : RMSE on scale-free target (eta / delta)")
    print(f"  M1 grouped by         : segment_id")
    print(f"  M2 grouped by         : basin_number")
    print("=" * 70)

    if not EVENT_CSV.exists() or not RESET_CSV.exists():
        raise FileNotFoundError(
            "Run pipeline.build_dataset and pipeline.build_reset_dataset first.")

    print("\n--- Model 1 ---")
    search_m1(quick)
    print("\n--- Model 2 ---")
    search_m2(quick)

    print("\nDone. best_params_m1.json / best_params_m2.json written.")
    print("config.py will load them automatically on next import.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="tiny grid for a smoke test")
    args = parser.parse_args()
    main(quick=args.quick)