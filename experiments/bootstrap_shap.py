"""
experiments/bootstrap_shap.py — SHAP accumulation for the v2 bootstrap
======================================================================
Helpers that let run_bootstrap_v2.py record, for every RANDOM iteration,
two per-feature quantities on that iteration's held-out test set, using the
SAME 500 draws as the performance bootstrap (the hardcoded iteration is
skipped — it is excluded from the distribution):

  1. IMPORTANCE : mean|SHAP|  (how much the feature moves predictions)
  2. DIRECTION  : corr(feature_value, SHAP_value)  across the held-out events
                  (the slope of the beeswarm — does higher feature value push
                  the prediction up or down). This is the CORRECT directional
                  summary for tree models: unlike the dataset-mean signed SHAP,
                  it does NOT collapse to zero for strong CONDITIONAL effects,
                  so it stays meaningful for the dominant (and often nonlinear)
                  features such as α and ρᵢ.

Four CSVs are written:
  data/bootstrap_shap_m1.csv       (importance per Model 1 feature, per iter)
  data/bootstrap_shap_m1_dir.csv   (direction  per Model 1 feature, per iter)
  data/bootstrap_shap_m2.csv       (importance per Model 2 feature, per iter)
  data/bootstrap_shap_m2_dir.csv   (direction  per Model 2 feature, per iter)

fig7_bootstrap_shap.py consumes these to plot per-feature median importance,
IQR, and DIRECTIONAL sign-consistency across the bootstrap.

NOTE on interpretation of DIRECTION (η and δ are higher = less decay / more
recovery): corr > 0 means higher feature value pushes the prediction UP
(toward less decay for Model 1, toward higher recovery for Model 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

SHAP_M1_CSV     = DATA_DIR / "bootstrap_shap_m1.csv"
SHAP_M2_CSV     = DATA_DIR / "bootstrap_shap_m2.csv"
SHAP_M1_DIR_CSV = DATA_DIR / "bootstrap_shap_m1_dir.csv"
SHAP_M2_DIR_CSV = DATA_DIR / "bootstrap_shap_m2_dir.csv"


def _safe_corr(x: np.ndarray, s: np.ndarray) -> float:
    """corr(feature value, SHAP value); nan if either is (near-)constant."""
    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-9 or np.std(s) < 1e-9:
        return np.nan
    try:
        c = float(np.corrcoef(x, s)[0, 1])
        return c if np.isfinite(c) else np.nan
    except Exception:
        return np.nan


def shap_row(model, X_test: np.ndarray, feat_cols: list[str],
             iteration: int) -> tuple[dict, dict] | tuple[None, None]:
    """
    Compute per-feature IMPORTANCE (mean|SHAP|) and DIRECTION
    (corr(feature_value, SHAP_value)) on the held-out test matrix for one
    iteration.

    Returns (importance_row, direction_row), each a dict keyed by feature name
    plus 'iteration'. Returns (None, None) on any failure (kept robust so a
    single bad iteration never aborts the bootstrap).
    """
    try:
        import shap
        if X_test is None or len(X_test) < 3:
            return None, None
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X_test)      # (n_test, n_feat)
        mean_abs  = np.abs(sv).mean(axis=0)
        imp = {"iteration": iteration}
        dir_= {"iteration": iteration}
        for j, f in enumerate(feat_cols):
            imp[f]  = float(mean_abs[j])
            dir_[f] = _safe_corr(X_test[:, j], sv[:, j])
        return imp, dir_
    except Exception as e:
        print(f"    [shap] iteration {iteration}: skipped ({e})")
        return None, None


def write_shap_csvs(m1_imp: list[dict], m1_dir: list[dict],
                    m2_imp: list[dict], m2_dir: list[dict]) -> None:
    """Write the four accumulated SHAP tables (only if non-empty)."""
    pairs = [
        (m1_imp, SHAP_M1_CSV),     (m1_dir, SHAP_M1_DIR_CSV),
        (m2_imp, SHAP_M2_CSV),     (m2_dir, SHAP_M2_DIR_CSV),
    ]
    for rows, path in pairs:
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  Saved: {path.name}  ({len(rows)} iterations)")
        else:
            print(f"  (no SHAP rows to write for {path.name})")