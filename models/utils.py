"""
models/utils.py — Shared utilities for Model 1 and Model 2
===========================================================
Contains everything that both models need:
  - Metrics computation (on log-ratio target and on raw IRD cm/h)
  - Generic train/predict helpers for XGBoost and LightGBM
  - A thin ModelResult dataclass that standardises what a trained model returns

Nothing in this file knows about features or targets — those are defined in
pipeline/features.py and passed in as arguments.

Design principle
----------------
Every function takes explicit arguments (no global state).
Import from config only for hyperparameters — never for paths or features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BOOSTING_PARAMS, BOOSTING_PARAMS_M2,
    EARLY_STOPPING_ROUNDS, EARLY_STOPPING_ROUNDS_M2,
    RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# ModelResult — standardised output of any train function
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """
    Standardised container returned by all train_* functions.

    Attributes
    ----------
    model       : fitted sklearn-compatible model
    scaler      : fitted StandardScaler (applied before predict)
    feat_cols   : ordered list of feature column names the model was trained on
    val_metrics : metrics dict on the validation split
    test_metrics: metrics dict on the test split
    model_name  : human-readable label (e.g. "XGBoost", "LightGBM")
    extra       : any additional info (e.g. best_iteration, basin_count)
    """
    model:        object
    scaler:       StandardScaler
    feat_cols:    list[str]
    val_metrics:  dict
    test_metrics: dict
    model_name:   str = ""
    extra:        dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics on IRD_norm_log (log-ratio target)
# ─────────────────────────────────────────────────────────────────────────────

def metrics_norm(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    label:   str = "",
    verbose: bool = True,
) -> Optional[dict]:
    """
    Compute metrics on IRD_norm_log = log(IRD / IRD_at_reset).

    Returns dict with: r2, rmse, mae, spearman_r, n.
    Returns None if fewer than 2 finite pairs.

    Note: MAPE is NOT computed here — the log-ratio target can be near zero,
    making percentage error numerically unstable. MAPE is computed on the
    back-transformed raw IRD in metrics_ird().
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return None

    yt, yp = y_true[mask], y_pred[mask]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spear_r, _ = sp_stats.spearmanr(yt, yp)
    spear_r = float(spear_r) if np.isfinite(spear_r) else np.nan

    m = dict(
        r2         = float(r2_score(yt, yp)),
        rmse       = float(np.sqrt(mean_squared_error(yt, yp))),
        mae        = float(mean_absolute_error(yt, yp)),
        spearman_r = float(spear_r),
        n          = int(mask.sum()),
    )

    if verbose and label:
        print(
            f"  {label:<35}  "
            f"R²={m['r2']:+.4f}  "
            f"RMSE={m['rmse']:.4f}  "
            f"MAE={m['mae']:.4f}  "
            f"ρ={m['spearman_r']:+.4f}  "
            f"n={m['n']}"
        )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Metrics on raw IRD (cm/h) — back-transformed from log-ratio
# ─────────────────────────────────────────────────────────────────────────────

def metrics_ird(
    ird_actual: np.ndarray,
    ird_pred:   np.ndarray,
    label:      str = "",
    verbose:    bool = True,
) -> Optional[dict]:
    """
    Compute metrics on back-transformed raw IRD (cm/h).

    Back-transform: IRD_pred = IRD_at_reset * exp(IRD_norm_log_pred)
    Includes MAPE (valid here since IRD > 0 always) and
    rel_RMSE = RMSE / mean(IRD_actual) — useful for comparing error
    magnitude across basins with different IRD scales.

    Returns None if fewer than 2 finite, positive pairs.
    """
    mask = (
        np.isfinite(ird_actual) & np.isfinite(ird_pred) &
        (ird_actual > 0)        & (ird_pred > 0)
    )
    if mask.sum() < 2:
        return None

    yt, yp     = ird_actual[mask], ird_pred[mask]
    spear_r, _ = sp_stats.spearmanr(yt, yp)
    rmse       = float(np.sqrt(mean_squared_error(yt, yp)))
    ird_mean   = float(np.mean(yt))

    m = dict(
        r2         = round(float(r2_score(yt, yp)),              4),
        rmse       = round(rmse,                                  4),
        mae        = round(float(mean_absolute_error(yt, yp)),    4),
        mape       = round(float(np.mean(np.abs((yt-yp)/yt))*100), 2),
        spearman_r = round(float(spear_r),                        4),
        rel_rmse   = round(rmse / ird_mean if ird_mean > 0 else np.nan, 4),
        ird_mean   = round(ird_mean,                              4),
        ird_std    = round(float(np.std(yt)),                     4),
        n          = int(mask.sum()),
    )

    if verbose and label:
        print(
            f"  {label:<35}  "
            f"R²={m['r2']:+.4f}  "
            f"RMSE={m['rmse']:.4f} cm/h  "
            f"MAPE={m['mape']:.1f}%  "
            f"rel_RMSE={m['rel_rmse']:.4f}  "
            f"n={m['n']}"
        )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Back-transform: IRD_norm_log → raw IRD (cm/h)
# ─────────────────────────────────────────────────────────────────────────────

def back_transform(
    ird_at_reset: np.ndarray,
    norm_log_pred: np.ndarray,
) -> np.ndarray:
    """
    Convert IRD_norm_log predictions back to raw IRD (cm/h).
    IRD = IRD_at_reset * exp(IRD_norm_log)
    """
    return ird_at_reset * np.exp(norm_log_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Generic predict function (works for any fitted sklearn-compatible model)
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model:     object,
    scaler:    StandardScaler,
    feat_cols: list[str],
    df:        pd.DataFrame,
) -> np.ndarray:
    """
    Apply a trained model to df. Returns predictions aligned to df's index.
    Rows with any missing feature values return NaN.

    Parameters
    ----------
    model     : fitted model with a .predict() method
    scaler    : fitted StandardScaler (applied before predict)
    feat_cols : ordered feature column list (must match training order)
    df        : DataFrame containing feat_cols

    Returns
    -------
    np.ndarray of shape (len(df),) — NaN where features were missing
    """
    result = np.full(len(df), np.nan)
    if not all(c in df.columns for c in feat_cols):
        missing = [c for c in feat_cols if c not in df.columns]
        print(f"  WARNING: missing feature columns: {missing}")
        return result

    valid = df[feat_cols].notna().all(axis=1)
    if valid.sum() == 0:
        return result

    X = scaler.transform(df.loc[valid, feat_cols].values)
    result[valid.values] = model.predict(X)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    train:      pd.DataFrame,
    val:        pd.DataFrame,
    feat_cols:  list[str],
    target:     str,
    model2:     bool = False,
    verbose:    bool = True,
) -> tuple[object, StandardScaler, list[str]]:
    """
    Fit XGBRegressor on train, early-stop on val.

    Parameters
    ----------
    train, val  : DataFrames with feat_cols and target columns
    feat_cols   : feature column names
    target      : target column name
    model2      : if True, uses BOOSTING_PARAMS_M2 (smaller dataset params)
    verbose     : print training summary

    Returns
    -------
    (model, scaler, used_cols)
    used_cols = feat_cols minus any column entirely missing from train
    """
    from xgboost import XGBRegressor

    params     = BOOSTING_PARAMS_M2 if model2 else BOOSTING_PARAMS
    stop_rnds  = EARLY_STOPPING_ROUNDS_M2 if model2 else EARLY_STOPPING_ROUNDS

    df_tr = train[feat_cols + [target]].dropna()
    df_va = val[feat_cols   + [target]].dropna()

    used_cols = [c for c in feat_cols if c in df_tr.columns
                 and df_tr[c].notna().any()]

    sc  = StandardScaler()
    Xtr = sc.fit_transform(df_tr[used_cols].values)
    Xva = sc.transform(df_va[used_cols].values)

    model = XGBRegressor(
        **{k: v for k, v in params.items()
           if k not in ("random_state",)},
        random_state          = RANDOM_SEED,
        early_stopping_rounds = stop_rnds,
        eval_metric           = "rmse",
        verbosity             = 0,
        n_jobs                = -1,
    )
    model.fit(
        Xtr, df_tr[target].values,
        eval_set=[(Xva, df_va[target].values)],
        verbose=False,
    )

    if verbose:
        print(
            f"  XGBoost  best_iter={model.best_iteration:>4}  "
            f"features={len(used_cols)}  "
            f"train_n={len(df_tr)}  val_n={len(df_va)}"
        )
    return model, sc, used_cols


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(
    train:      pd.DataFrame,
    val:        pd.DataFrame,
    feat_cols:  list[str],
    target:     str,
    model2:     bool = False,
    verbose:    bool = True,
) -> tuple[object, StandardScaler, list[str]]:
    """
    Fit LGBMRegressor on train, early-stop on val.
    Same interface as train_xgboost for easy swapping.
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = BOOSTING_PARAMS_M2 if model2 else BOOSTING_PARAMS

    df_tr = train[feat_cols + [target]].dropna()
    df_va = val[feat_cols   + [target]].dropna()

    used_cols = [c for c in feat_cols if c in df_tr.columns
                 and df_tr[c].notna().any()]

    sc  = StandardScaler()
    Xtr = sc.fit_transform(df_tr[used_cols].values)
    Xva = sc.transform(df_va[used_cols].values)

    stop_rnds = EARLY_STOPPING_ROUNDS_M2 if model2 else EARLY_STOPPING_ROUNDS

    model = LGBMRegressor(
        **{k: v for k, v in params.items()
           if k not in ("colsample_bytree", "random_state", "n_jobs")},
        feature_fraction = params.get("colsample_bytree", 0.8),
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbose          = -1,
    )
    model.fit(
        Xtr, df_tr[target].values,
        eval_set=[(Xva, df_va[target].values)],
        callbacks=[
            early_stopping(stop_rnds, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    if verbose:
        print(
            f"  LightGBM best_iter={model.best_iteration_:>4}  "
            f"features={len(used_cols)}  "
            f"train_n={len(df_tr)}  val_n={len(df_va)}"
        )
    return model, sc, used_cols


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin R² (median across basins — used to report within-basin performance)
# ─────────────────────────────────────────────────────────────────────────────

def per_basin_median_r2(
    df:        pd.DataFrame,
    y_pred:    np.ndarray,
    target:    str,
    min_rows:  int = 5,
) -> float:
    """
    Compute median R² across basins.

    This metric captures within-basin predictive power — it is lower than
    global R² because it excludes the between-basin variance that the
    global model captures from basin-scale differences in IRD levels.

    Parameters
    ----------
    df       : DataFrame with 'basin_number' and target columns
    y_pred   : predictions aligned to df's index (NaN where not predicted)
    target   : target column name
    min_rows : minimum rows needed per basin to include in median

    Returns
    -------
    float — median R² across basins, or NaN if no basins qualify
    """
    df = df.copy()
    df["_pred"] = y_pred
    r2s = []

    for _, bdf in df.groupby("basin_number"):
        valid = bdf[target].notna() & bdf["_pred"].notna()
        if valid.sum() < min_rows:
            continue
        try:
            r2 = r2_score(
                bdf.loc[valid, target].values,
                bdf.loc[valid, "_pred"].values,
            )
            r2s.append(float(r2))
        except Exception:
            pass

    return float(np.median(r2s)) if r2s else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Prepare splits from a DataFrame (convenience helper)
# ─────────────────────────────────────────────────────────────────────────────

def get_splits(
    df:        pd.DataFrame,
    feat_cols: list[str],
    target:    str,
    split_col: str = "split",
    train_val: str = "train",
    val_val:   str = "val",
    test_val:  str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract train / val / test DataFrames from a pooled DataFrame.
    Drops rows with missing features or target.

    Parameters
    ----------
    df        : pooled DataFrame with split_col column
    feat_cols : feature columns (used for dropna)
    target    : target column (used for dropna)
    split_col : column containing split labels
    *_val     : label values for each split

    Returns
    -------
    (train, val, test) — each a clean DataFrame ready for training
    """
    needed = list(dict.fromkeys(feat_cols + [target, split_col]))
    needed = [c for c in needed if c in df.columns]
    df_clean = df[needed].dropna(subset=feat_cols + [target]).copy()

    train = df_clean[df_clean[split_col] == train_val].reset_index(drop=True)
    val   = df_clean[df_clean[split_col] == val_val].reset_index(drop=True)
    test  = df_clean[df_clean[split_col] == test_val].reset_index(drop=True)

    print(
        f"  Splits — train={len(train)}  val={len(val)}  test={len(test)}  "
        f"(dropped {len(df) - len(df_clean)} rows with missing data)"
    )
    return train, val, test