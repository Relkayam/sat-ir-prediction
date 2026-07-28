"""
analysis/depth_baseline.py — Depth-stratified skill vs naive-at-lag baseline
============================================================================
Purpose (per reviewer "forecasting-span" comment):
  Demonstrate that Model 1 predicts the WHOLE within-segment trajectory from
  the reset anchor, not just one step ahead. We compare, as a function of
  within-segment "depth" n:

    - MODEL       : Model 1 prediction of IRD at event position k
                    (built from features known in advance at the reset anchor)
    - NAIVE(n)    : predict IRD(k) using the OBSERVED IRD n events earlier,
                    i.e. IRD_hat(k) = IRD(k - n)     [naive-at-lag-n]

  Prediction: NAIVE(n) degrades as n grows (staler information), while the
  MODEL stays roughly flat because every event is predicted from the anchor.
  => ΔRMSE = RMSE_naive(n) - RMSE_model  should WIDEN with depth n.

This module contains ONLY the analysis logic + a synthetic smoke test.
It does not touch real data unless you call run_depth_analysis() with your
real event dataframe in PyCharm.

Expected columns in the event dataframe (matching build_dataset output):
    basin_number : int
    segment_id   : int / str   (unique per segment; reset + its followers)
    row_type     : str         ("reset" for the anchor, else in-segment)
    LCT          : float        time since last tillage (event ordering axis)
    IRD          : float        measured infiltration rate (cm/h)
    IRD_pred     : float        Model 1 predicted IRD (cm/h)  [see note]

NOTE on IRD_pred:
    Supply your Model 1 predictions in raw cm/h (after back-transform
    IRD_hat = rho_i * exp(eta_hat)). If you don't have a per-event prediction
    column yet, run_depth_analysis(..., model_pred_col=None) will still
    compute the naive-at-lag curve alone so you can see the baseline shape.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def depth_table(
    df: pd.DataFrame,
    *,
    max_depth: int = 6,
    basin_col: str = "basin_number",
    segment_col: str = "segment_id",
    order_col: str = "LCT",
    ird_col: str = "IRD",
    model_pred_col: str | None = "IRD_pred",
    row_type_col: str = "row_type",
) -> pd.DataFrame:
    """
    Build the depth-stratified RMSE table.

    For each segment, order events by `order_col`. Position k within the
    segment (k = 0 is the reset anchor). For depth n, the naive-at-lag-n
    prediction of event k is the OBSERVED IRD at event k-n (same segment).

    Returns one row per depth n with:
        depth, n_events, rmse_naive, rmse_model (if preds given), delta_rmse
    """
    needed = {basin_col, segment_col, order_col, ird_col}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"depth_table: missing columns {sorted(missing)}")

    have_model = model_pred_col is not None and model_pred_col in df.columns

    # collect per (depth) arrays of (truth, naive_pred, model_pred)
    rows = []
    for depth in range(1, max_depth + 1):
        truth, naive_pred, model_pred = [], [], []
        for _, seg in df.groupby([basin_col, segment_col], sort=False):
            seg = seg.sort_values(order_col).reset_index(drop=True)
            ird = pd.to_numeric(seg[ird_col], errors="coerce").to_numpy()
            if len(seg) <= depth:
                continue
            # event k predicted from observed event k-depth
            for k in range(depth, len(seg)):
                truth.append(ird[k])
                naive_pred.append(ird[k - depth])
                if have_model:
                    model_pred.append(
                        pd.to_numeric(seg[model_pred_col], errors="coerce").to_numpy()[k]
                    )
        truth = np.asarray(truth, float)
        naive_pred = np.asarray(naive_pred, float)
        r_naive = _rmse(truth, naive_pred)
        row = {"depth": depth, "n_events": int(np.isfinite(truth).sum()),
               "rmse_naive": r_naive}
        if have_model:
            model_pred = np.asarray(model_pred, float)
            r_model = _rmse(truth, model_pred)
            row["rmse_model"] = r_model
            row["delta_rmse"] = r_naive - r_model  # positive = model better
        rows.append(row)

    return pd.DataFrame(rows)


def run_depth_analysis(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Thin wrapper so you can call this from PyCharm on the real dataframe."""
    tbl = depth_table(df, **kw)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\n=== Depth-stratified baseline (naive-at-lag-n vs model) ===")
        print(tbl.to_string(index=False))
    return tbl

# This module is a library only — no __main__, no synthetic data.
# Call depth_table() / run_depth_analysis() on a REAL held-out event frame,
# e.g. from analysis/run_depth_on_heldout.py.