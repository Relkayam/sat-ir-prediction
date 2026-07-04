"""
make_table1.py — Assemble Table 1 (held-out per-basin metrics, M1 + M2)
=======================================================================
Trains Model 1 and Model 2 via their own module code paths, computes per-basin
metrics, and writes Table 1: one row per held-out basin (plus POOLED), M1 and
M2 side by side.

WHY RE-COMPUTE INSTEAD OF READING A FILE
  model1/2_results_condition_e.xlsx store only the POOLED summary row (no
  per-basin rows). So Table 1 reproduces per-basin metrics using the SAME
  functions the model + figure scripts use — guaranteeing it matches
  `python -m models.model1_decay` / `model2_reset`. Mirrors the figure scripts'
  predict_held_out logic, without plotting.

  Linear baseline (M1) intentionally excluded — belongs in §3.2 / SI.
  Output: outputs/tables/table1_held_out_metrics.xlsx

Usage:  python -m analysis.make_table1
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TABLES_DIR, FIELD_NAMES
from pipeline.features import TARGET_M1, MODEL2_FEATURES, TARGET_M2
from models.utils import back_transform, metrics_ird, predict

import models.model1_decay as m1mod
import models.model2_reset as m2mod

OUT = TABLES_DIR / "table1_held_out_metrics.xlsx"


def m1_per_basin() -> dict:
    held_out         = m1mod.load_selected_basins()
    df, feat_cols    = m1mod.load_data()
    train, val, test = m1mod.prepare_splits(df, held_out, feat_cols)
    model, sc        = m1mod.train_model(train, val, feat_cols)

    ho = test.copy()
    ird_reset      = pd.to_numeric(ho["IRD_at_reset"], errors="coerce").values
    pred_norm      = predict(model, sc, feat_cols, ho)
    ho["ird_pred"] = back_transform(ird_reset, pred_norm)
    ho["ird_true"] = back_transform(
        ird_reset, pd.to_numeric(ho[TARGET_M1], errors="coerce").values)
    ho["ird_naive"] = ird_reset

    out: dict = {}
    all_t, all_p = [], []
    for bn in held_out:
        bdf   = ho[ho["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        t  = bdf["ird_true"].values.astype(float)
        p  = bdf["ird_pred"].values.astype(float)
        nv = bdf["ird_naive"].values.astype(float)
        mask = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
        all_t.append(t[mask]); all_p.append(p[mask])
        m  = metrics_ird(t[mask], p[mask],  verbose=False) or {}
        mn = metrics_ird(t[mask], nv[mask], verbose=False) or {}
        out[str(bn)] = dict(field=field, n=m.get("n", 0),
                            r2=m.get("r2", np.nan), rmse=m.get("rmse", np.nan),
                            rmse_naive=mn.get("rmse", np.nan),
                            mape=m.get("mape", np.nan))
    t_all = np.concatenate(all_t); p_all = np.concatenate(all_p)
    m_all = metrics_ird(t_all, p_all, verbose=False) or {}
    out["POOLED"] = dict(field="all", n=m_all.get("n", 0),
                         r2=m_all.get("r2", np.nan), rmse=m_all.get("rmse", np.nan),
                         rmse_naive=np.nan, mape=m_all.get("mape", np.nan))
    return out


def m2_per_basin() -> dict:
    held_out  = m2mod.load_selected_basins()
    df        = m2mod.load_data()
    feat_cols = [f for f in MODEL2_FEATURES if f in df.columns]
    train, val, test = m2mod.prepare_splits(df, held_out, feat_cols)
    model, sc        = m2mod.train_model(train, val, feat_cols)

    Xte      = sc.transform(test[feat_cols].values)
    prev_ird = test["prev_IRD_at_reset_raw"].values.astype(float)
    test     = test.copy()
    test["ird_pred"]  = m2mod._back(model.predict(Xte), prev_ird)
    test["ird_true"]  = test["IRD_at_reset"].values.astype(float)
    test["ird_naive"] = prev_ird

    out: dict = {}
    all_t, all_p, all_n = [], [], []
    for bn in held_out:
        bdf   = test[test["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        t  = bdf["ird_true"].values.astype(float)
        p  = bdf["ird_pred"].values.astype(float)
        nv = bdf["ird_naive"].values.astype(float)
        m  = m2mod._metrics(t, p,  verbose=False) or {}
        mn = m2mod._metrics(t, nv, verbose=False) or {}
        all_t.append(t); all_p.append(p); all_n.append(nv)
        out[str(bn)] = dict(field=field, n=m.get("n", 0),
                            r2=m.get("r2", np.nan), rmse=m.get("rmse", np.nan),
                            rmse_naive=mn.get("rmse", np.nan),
                            mape=m.get("mape", np.nan))
    t_all = np.concatenate(all_t); p_all = np.concatenate(all_p)
    n_all = np.concatenate(all_n)
    m_all  = m2mod._metrics(t_all, p_all, verbose=False) or {}
    mn_all = m2mod._metrics(t_all, n_all, verbose=False) or {}
    out["POOLED"] = dict(field="all", n=m_all.get("n", 0),
                         r2=m_all.get("r2", np.nan), rmse=m_all.get("rmse", np.nan),
                         rmse_naive=mn_all.get("rmse", np.nan),
                         mape=m_all.get("mape", np.nan))
    return out


def main() -> None:
    print("=" * 64)
    print("  BUILD TABLE 1 — held-out per-basin metrics (M1 + M2)")
    print("  (re-computes via model code paths; no plotting)")
    print("=" * 64)

    print("\n--- Model 1 ---")
    m1 = m1_per_basin()
    print("\n--- Model 2 ---")
    m2 = m2_per_basin()

    keys = sorted({k for k in m1 if k != "POOLED"} | {k for k in m2 if k != "POOLED"},
                  key=lambda b: int(b))
    keys.append("POOLED")

    rows = []
    for b in keys:
        a = m1.get(b, {}); c = m2.get(b, {})
        field = a.get("field") or c.get("field") or (
            "all" if b == "POOLED" else FIELD_NAMES.get(int(str(b)[0]), ""))
        rows.append(dict(
            basin=b, field=field,
            n_M1=a.get("n"), r2_M1=a.get("r2"), rmse_M1=a.get("rmse"),
            rmse_naive_M1=a.get("rmse_naive"), mape_M1=a.get("mape"),
            n_M2=c.get("n"), r2_M2=c.get("r2"), rmse_M2=c.get("rmse"),
            rmse_naive_M2=c.get("rmse_naive"), mape_M2=c.get("mape"),
        ))
    tab = pd.DataFrame(rows)

    for col, nd in {"r2_M1":3,"rmse_M1":3,"rmse_naive_M1":3,"mape_M1":1,
                    "r2_M2":3,"rmse_M2":3,"rmse_naive_M2":3,"mape_M2":1}.items():
        tab[col] = pd.to_numeric(tab[col], errors="coerce").round(nd)

    print("\n  TABLE 1 (preview):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(tab.to_string(index=False))

    tab.to_excel(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print("  NOTE: M1 POOLED naive RMSE blank (per-basin naive only); = 1.849 cm/h")
    print("        from the comparison table if needed.")
    print("  NOTE: linear baseline (M1) excluded by design — see §3.2 / SI.")


if __name__ == "__main__":
    main()