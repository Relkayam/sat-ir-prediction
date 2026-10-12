"""
analysis/lagged_eta_test.py
===========================
Reviewer C1 (most important revision): "add lagged eta(t-1) as a feature to
show operational features add skill beyond the autoregressive term."

Logic: if Model 1 were merely doing autoregression (predicting eta(t) from
eta(t-1)), then adding eta(t-1) as a feature would absorb most of the skill and
the operational/weather features (alpha, radiation, ...) would lose importance.
If instead the operational features add genuine skill, then:
  (a) adding eta(t-1) improves RMSE only modestly, and
  (b) alpha / radiation retain their SHAP importance even with eta(t-1) present.

This trains Model 1 on the held-out draw in TWO configurations:
  BASELINE  : current feature set
  +LAG      : current features + eta_prev (eta of the previous surviving event
              in the same segment)
and reports held-out RMSE/R2 for both, plus the top SHAP features of +LAG so you
can see whether alpha survives.

eta_prev is the previous event's IRD_norm_log within the segment (shift(1)),
computed on quality-passing events only — same convention as prev_* features.

Run: python analysis/lagged_eta_test.py
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\user\PycharmProjects\sat-ir-prediction")

from models.model1_decay import load_selected_basins, load_data, prepare_splits
from models.utils import predict, back_transform
from pipeline.features import TARGET_M1
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
from config import BOOSTING_PARAMS_M1, EARLY_STOPPING_ROUNDS_M1


def add_eta_prev(df: pd.DataFrame) -> pd.DataFrame:
    """eta of the previous event within each segment (shift within segment_id)."""
    df = df.copy()
    df["eta_prev"] = np.nan
    for sid, grp in df.groupby("segment_id"):
        if sid < 0:
            continue
        df.loc[grp.index, "eta_prev"] = (
            pd.to_numeric(grp[TARGET_M1], errors="coerce").shift(1).values)
    return df


def _train(train, val, feat_cols):
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    m = LGBMRegressor(**BOOSTING_PARAMS_M1)
    m.fit(Xtr, train[TARGET_M1].values,
          eval_set=[(Xva, val[TARGET_M1].values)],
          callbacks=[early_stopping(EARLY_STOPPING_ROUNDS_M1, verbose=False),
                     log_evaluation(period=-1)])
    return m, sc


def _metrics(model, sc, feat_cols, test):
    ird_reset = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    pred_norm = predict(model, sc, feat_cols, test)
    ird_pred = back_transform(ird_reset, pred_norm)
    ird_true = pd.to_numeric(test["IRD"], errors="coerce").values
    m = np.isfinite(ird_pred) & np.isfinite(ird_true)
    rmse = float(np.sqrt(np.mean((ird_pred[m] - ird_true[m]) ** 2)))
    ss_res = float(np.sum((ird_true[m] - ird_pred[m]) ** 2))
    ss_tot = float(np.sum((ird_true[m] - ird_true[m].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, r2, int(m.sum())


def _top_shap(model, sc, feat_cols, test, k=6):
    try:
        import shap
    except ImportError:
        return None
    X = sc.transform(test[feat_cols].dropna().values)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:k]
    return [(feat_cols[i], float(imp[i])) for i in order]


def main():
    print("=" * 70)
    print("  LAGGED-ETA TEST — do operational features add skill beyond eta(t-1)?")
    print("=" * 70)

    held_out = load_selected_basins()
    df, feat_cols = load_data()
    df = add_eta_prev(df)

    # rows must have eta_prev for a fair head-to-head (drop first-in-segment)
    df_lag = df.dropna(subset=["eta_prev"]).copy()

    # BASELINE (current features) on the SAME rows as +LAG, for fairness
    tr_b, va_b, te_b = prepare_splits(df_lag, held_out, feat_cols)
    mb, sb = _train(tr_b, va_b, feat_cols)
    rmse_b, r2_b, n_b = _metrics(mb, sb, feat_cols, te_b)

    # +LAG (current features + eta_prev)
    feat_lag = feat_cols + ["eta_prev"]
    tr_l, va_l, te_l = prepare_splits(df_lag, held_out, feat_lag)
    ml, sl = _train(tr_l, va_l, feat_lag)
    rmse_l, r2_l, n_l = _metrics(ml, sl, feat_lag, te_l)

    print(f"\n  (rows with eta_prev defined; held-out n≈{n_b})")
    print(f"  BASELINE (no eta_prev):  RMSE={rmse_b:.3f}  R2={r2_b:+.3f}")
    print(f"  +LAG  (with eta_prev) :  RMSE={rmse_l:.3f}  R2={r2_l:+.3f}")
    print(f"  ΔRMSE from adding eta_prev: {rmse_b - rmse_l:+.3f} cm/h "
          f"({100*(rmse_b-rmse_l)/rmse_b:+.1f}%)")

    top = _top_shap(ml, sl, feat_lag, te_l, k=6)
    if top:
        print("\n  Top SHAP features in the +LAG model (does alpha survive?):")
        for name, imp in top:
            print(f"    {name:<16} {imp:.4f}")
    print("\n  Read: if ΔRMSE from eta_prev is small AND alpha/radiation stay near")
    print("  the top, operational features add skill beyond autoregression.")


if __name__ == "__main__":
    main()