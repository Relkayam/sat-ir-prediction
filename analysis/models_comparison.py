"""
analysis/model_comparison.py — SI Table S2: algorithm comparison (M1 + M2)
==========================================================================
Compares the naive baseline and four ML algorithms on the SAME five
bootstrap-selected held-out basins used in the main text, for both models.

  Naive         — persistence (Model 1: no decay; Model 2: no change)
  Ridge         — linear ML baseline (on the standardized feature matrix)
  RandomForest  — bagged trees
  XGBoost       — gradient boosting
  LightGBM      — the primary model reported in the paper

Leak-free, matched to the main models
-------------------------------------
  * Held-out basins read from config.SELECTED_BASINS_CSV (selected_basins_v2.csv).
  * Model 1 train = split=='train', val = split=='val'  (disjoint; val only for
    early stopping). Model 2 train = split_chrono=='train', val =='val'.
  * NO is_good_segment gate anywhere — every sensor-passing event is used,
    exactly as in models/model1_decay.py and models/model2_reset.py.
  * Metric computed in raw IRD space (cm/h) after back-transform, identical to
    the main models, so the LightGBM row reproduces the headline numbers.

NOTE: the Ridge here is a plain ridge regression on the full feature matrix —
a *machine-learning* linear baseline. It is distinct from the causal linear
extrapolation of the main text (which uses only each segment's own prior
trajectory). They answer different questions and both are legitimate.

Output
------
  Printed per-basin + pooled tables (M1 and M2)
  outputs/tables/model_comparison.xlsx
  outputs/figures/model_comparison.png

Usage
-----
  python -m analysis.model_comparison
"""
from __future__ import annotations

import sys
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, RESET_CSV,
    SELECTED_BASINS_CSV, TABLES_DIR, FIGURES_DIR,
    RANDOM_SEED, FIELD_NAMES, FIGURE_DPI,
    BOOSTING_PARAMS_M1, EARLY_STOPPING_ROUNDS_M1,
    BOOSTING_PARAMS_M2, EARLY_STOPPING_ROUNDS_M2,
)
from pipeline.features import (
    prepare_features, TARGET_M1,
    MODEL2_FEATURES, TARGET_M2,
)

ALGO_ORDER  = ["Ridge", "RandomForest", "XGBoost", "LightGBM"]
ALGO_COLORS = {
    "Naive":        "#AAAAAA",
    "Ridge":        "#4BACC6",
    "RandomForest": "#70AD47",
    "XGBoost":      "#ED7D31",
    "LightGBM":     "#065A82",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics / collection
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if mask.sum() < 2:
        return {}
    yt, yp = y_true[mask], y_pred[mask]
    rmse   = float(np.sqrt(mean_squared_error(yt, yp)))
    return dict(
        r2   = round(float(r2_score(yt, yp)), 4),
        rmse = round(rmse, 4),
        mape = round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 2),
        n    = int(mask.sum()),
    )


def _collect(name: str, ird_true: np.ndarray, ird_pred: np.ndarray,
             test_df: pd.DataFrame, held_out: list[int]) -> dict:
    basins = test_df["basin_number"].values
    per_basin, all_t, all_p = {}, [], []
    for bn in held_out:
        m = basins == bn
        per_basin[bn] = _metrics(ird_true[m], ird_pred[m])
        v = np.isfinite(ird_true[m]) & np.isfinite(ird_pred[m]) & \
            (ird_true[m] > 0) & (ird_pred[m] > 0)
        all_t.append(ird_true[m][v]); all_p.append(ird_pred[m][v])
    pooled = _metrics(np.concatenate(all_t), np.concatenate(all_p))
    return dict(name=name, per_basin=per_basin, pooled=pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Selected basins
# ─────────────────────────────────────────────────────────────────────────────

def load_selected_basins() -> list[int]:
    if not SELECTED_BASINS_CSV.exists():
        raise FileNotFoundError(
            f"{SELECTED_BASINS_CSV} not found. Run experiments.run_bootstrap_v2.")
    basins = []
    with open(SELECTED_BASINS_CSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("basin"):
                continue
            try:
                basins.append(int(line.split(",")[0].strip()))
            except ValueError:
                pass
    print(f"  Selected held-out basins ({len(basins)}): {basins}")
    return basins


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 data + run
# ─────────────────────────────────────────────────────────────────────────────

def prepare_m1(held_out: list[int]):
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    if "segment_id" in df.columns:
        df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce")
    df, feat_cols = prepare_features(df)

    held = set(held_out)
    req  = feat_cols + [TARGET_M1]

    test = df[df["basin_number"].isin(held) & (df["row_type"] == "event")].dropna(
        subset=req).reset_index(drop=True)
    non_ho = df[~df["basin_number"].isin(held) & (df["row_type"] == "event")].dropna(
        subset=req)
    train = non_ho[non_ho["split"] == "train"].reset_index(drop=True)  # disjoint
    val   = non_ho[non_ho["split"] == "val"].reset_index(drop=True)

    print(f"  M1: train={len(train):,} ({non_ho['basin_number'].nunique()} basins)  "
          f"val={len(val):,}  test={len(test):,}  features={len(feat_cols)}")
    return train, val, test, feat_cols


def run_m1(name, model, train, val, test, feat_cols, held_out) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from xgboost import XGBRegressor

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)
    ytr, yva = train[TARGET_M1].values, val[TARGET_M1].values

    if isinstance(model, XGBRegressor):
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif isinstance(model, LGBMRegressor):
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[early_stopping(EARLY_STOPPING_ROUNDS_M1, verbose=False),
                             log_evaluation(period=-1)])
    else:
        model.fit(Xtr, ytr)

    ird_reset = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    ird_pred  = ird_reset * np.exp(model.predict(Xte))
    ird_true  = ird_reset * np.exp(pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    return _collect(name, ird_true, ird_pred, test, held_out)


def run_m1_naive(test, held_out) -> dict:
    ird_reset = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    ird_true  = ird_reset * np.exp(pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    return _collect("Naive", ird_true, ird_reset.copy(), test, held_out)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 data + run
# ─────────────────────────────────────────────────────────────────────────────

def prepare_m2(held_out: list[int]):
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    feat_cols = [f for f in MODEL2_FEATURES if f in df.columns]
    req = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]

    held = set(held_out)
    test = df[df["basin_number"].isin(held)].dropna(subset=req).reset_index(drop=True)
    non_ho = df[~df["basin_number"].isin(held)].dropna(subset=req)
    train = non_ho[non_ho["split_chrono"] == "train"].reset_index(drop=True)  # disjoint
    val   = non_ho[non_ho["split_chrono"] == "val"].reset_index(drop=True)

    print(f"  M2: train={len(train)} ({non_ho['basin_number'].nunique()} basins)  "
          f"val={len(val)}  test={len(test)}  features={len(feat_cols)}")
    return train, val, test, feat_cols


def run_m2(name, model, train, val, test, feat_cols, held_out) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from xgboost import XGBRegressor

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)
    ytr, yva = train[TARGET_M2].values, val[TARGET_M2].values

    if isinstance(model, XGBRegressor):
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif isinstance(model, LGBMRegressor):
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[early_stopping(EARLY_STOPPING_ROUNDS_M2, verbose=False),
                             log_evaluation(period=-1)])
    else:
        model.fit(Xtr, ytr)

    prev = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_pred = prev * np.exp(model.predict(Xte))
    ird_true = test["IRD_at_reset"].values.astype(float)
    return _collect(name, ird_true, ird_pred, test, held_out)


def run_m2_naive(test, held_out) -> dict:
    prev = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_true = test["IRD_at_reset"].values.astype(float)
    return _collect("Naive", ird_true, prev, test, held_out)


# ─────────────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list[dict], held_out: list[int], label: str) -> None:
    print(f"\n{'='*78}")
    print(f"  {label} — algorithm comparison  |  5 held-out basins  |  raw IRD (cm/h)")
    print(f"{'='*78}")
    naive = next(r for r in results if r["name"] == "Naive")
    print(f"  {'Algorithm':<14} {'pooled RMSE':>12} {'R2':>8} {'MAPE%':>7} "
          f"{'beats naive':>12}")
    print(f"  {'-'*72}")
    n_rmse = naive["pooled"].get("rmse", np.nan)
    print(f"  {'Naive':<14} {n_rmse:>12.4f} {naive['pooled'].get('r2',np.nan):>+8.3f} "
          f"{naive['pooled'].get('mape',np.nan):>6.1f}% {'(baseline)':>12}")
    for r in results:
        if r["name"] == "Naive":
            continue
        p = r["pooled"]
        wins = sum(1 for bn in held_out
                   if r["per_basin"].get(bn, {}).get("rmse", np.inf) <
                   naive["per_basin"].get(bn, {}).get("rmse", np.inf))
        print(f"  {r['name']:<14} {p.get('rmse',np.nan):>12.4f} "
              f"{p.get('r2',np.nan):>+8.3f} {p.get('mape',np.nan):>6.1f}% "
              f"{f'{wins}/5':>12}")
    # delta summary
    print(f"\n  pooled \u0394RMSE vs naive (cm/h):")
    for r in results:
        if r["name"] == "Naive":
            continue
        d = n_rmse - r["pooled"].get("rmse", np.nan)
        print(f"    {r['name']:<14} {d:>+8.4f}  "
              f"({100*d/n_rmse:>+5.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(m1: list[dict], m2: list[dict]) -> None:
    cfg = [("rmse", "RMSE (cm/h)", False), ("r2", "R\u00b2", True),
           ("mape", "MAPE %", False)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Algorithm comparison — Model 1 (top) and Model 2 (bottom)\n"
                 "5 held-out basins, raw IRD space; dashed = naive baseline",
                 fontsize=9, fontweight="bold")
    for row, (results, ylab) in enumerate([(m1, "Model 1"), (m2, "Model 2")]):
        naive = next(r for r in results if r["name"] == "Naive")
        algos = [r for r in results if r["name"] != "Naive"]
        names = [r["name"] for r in algos]
        colors = [ALGO_COLORS.get(n, "#888") for n in names]
        x = np.arange(len(names))
        for col, (metric, label, higher) in enumerate(cfg):
            ax = axes[row][col]
            vals = [r["pooled"].get(metric, np.nan) for r in algos]
            nv = naive["pooled"].get(metric, np.nan)
            bars = ax.bar(x, vals, color=colors, alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x()+b.get_width()/2,
                            b.get_height()+abs(b.get_height())*0.02,
                            f"{v:.3f}" if metric != "mape" else f"{v:.1f}",
                            ha="center", va="bottom", fontsize=7.5)
            if np.isfinite(nv):
                ax.axhline(nv, color="black", lw=1.4, ls="--", alpha=0.65,
                           label=f"Naive={nv:.3f}")
                ax.legend(fontsize=7, loc="best")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right",
                                                 fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
            if col == 0:
                ax.set_ylabel(ylab, fontsize=9, fontweight="bold")
            arrow = "\u2191 better" if higher else "\u2193 better"
            ax.set_title(f"{label} ({arrow})", fontsize=8, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {out.name}")


def save_results(m1, m2, held_out) -> None:
    rows = []
    for model_label, results in [("Model1", m1), ("Model2", m2)]:
        for r in results:
            p = r["pooled"]
            row = dict(model=model_label, algorithm=r["name"],
                       rmse_pool=p.get("rmse", np.nan), r2_pool=p.get("r2", np.nan),
                       mape_pool=p.get("mape", np.nan), n_pool=p.get("n", np.nan))
            for bn in held_out:
                m = r["per_basin"].get(bn, {})
                row[f"rmse_{bn}"] = m.get("rmse", np.nan)
                row[f"r2_{bn}"]   = m.get("r2", np.nan)
            rows.append(row)
    pd.DataFrame(rows).to_excel(TABLES_DIR / "model_comparison.xlsx", index=False)
    print(f"  Saved: model_comparison.xlsx")


# This is the new function to add to analysis/model_comparison.py
# (shown standalone here so you can paste it in; integration note below)

def _si_block(results, held_out, model_label):
    """Build one SI-formatted dataframe block (naive row first, then algos)."""
    import numpy as np
    import pandas as pd
    naive = next(r for r in results if r["name"] == "Naive")
    n_rmse = naive["pooled"].get("rmse", np.nan)

    def _row(r, is_naive=False):
        p = r["pooled"]
        if is_naive:
            return {
                "Algorithm": "Naive",
                "RMSE (cm/h)": round(p.get("rmse", np.nan), 3),
                "R2": round(p.get("r2", np.nan), 3),
                "MAPE (%)": round(p.get("mape", np.nan), 1),
                "dRMSE vs naive": "(baseline)",
                "beats naive": "-",
            }
        wins = sum(1 for bn in held_out
                   if r["per_basin"].get(bn, {}).get("rmse", np.inf) <
                   naive["per_basin"].get(bn, {}).get("rmse", np.inf))
        d = n_rmse - p.get("rmse", np.nan)
        pct = 100 * d / n_rmse if np.isfinite(n_rmse) and n_rmse else np.nan
        return {
            "Algorithm": r["name"],
            "RMSE (cm/h)": round(p.get("rmse", np.nan), 3),
            "R2": round(p.get("r2", np.nan), 3),
            "MAPE (%)": round(p.get("mape", np.nan), 1),
            "dRMSE vs naive": f"{d:+.3f} ({pct:+.1f}%)",
            "beats naive": f"{wins}/{len(held_out)}",
        }

    rows = [_row(naive, is_naive=True)]
    for r in results:
        if r["name"] != "Naive":
            rows.append(_row(r))
    return pd.DataFrame(rows)


def save_si_table(m1, m2, held_out):
    """Write SI Table S1 in paste-ready layout: two stacked blocks on one sheet,
    plus each block on its own sheet."""
    import pandas as pd
    from config import TABLES_DIR

    b1 = _si_block(m1, held_out, "Model 1")
    b2 = _si_block(m2, held_out, "Model 2")

    out = TABLES_DIR / "tableS1_algorithm_comparison.xlsx"
    with pd.ExcelWriter(out) as xl:
        # Combined sheet — what you paste into the SI (M1 block, gap, M2 block)
        startrow = 0
        header1 = pd.DataFrame({"Model 1 - within-segment decay": []})
        header1.to_excel(xl, sheet_name="SI_TableS1", index=False, startrow=startrow)
        b1.to_excel(xl, sheet_name="SI_TableS1", index=False, startrow=startrow + 1)
        startrow += len(b1) + 3
        header2 = pd.DataFrame({"Model 2 - post-tillage recovery": []})
        header2.to_excel(xl, sheet_name="SI_TableS1", index=False, startrow=startrow)
        b2.to_excel(xl, sheet_name="SI_TableS1", index=False, startrow=startrow + 1)
        # Separate sheets too, for convenience
        b1.to_excel(xl, sheet_name="Model_1", index=False)
        b2.to_excel(xl, sheet_name="Model_2", index=False)

    print(f"  Saved SI-formatted table: {out.name}  (sheet 'SI_TableS1' is paste-ready)")
    print("\n  --- Model 1 block ---")
    print(b1.to_string(index=False))
    print("\n  --- Model 2 block ---")
    print(b2.to_string(index=False))




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  MODEL COMPARISON (SI Table S2) — Model 1 + Model 2")
    print("  5 bootstrap-selected held-out basins; leak-free disjoint train/val")
    print("=" * 70)

    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    held_out = load_selected_basins()

    templates = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_SEED),
        "XGBoost": XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=50, eval_metric="rmse",
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
        "LightGBM": None,
    }

    print("\n--- Model 1 ---")
    tr1, va1, te1, f1 = prepare_m1(held_out)
    m1 = [run_m1_naive(te1, held_out)]
    for name in ALGO_ORDER:
        model = (LGBMRegressor(**BOOSTING_PARAMS_M1) if name == "LightGBM"
                 else copy.deepcopy(templates[name]))
        try:
            m1.append(run_m1(name, model, tr1, va1, te1, f1, held_out))
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    print_table(m1, held_out, "Model 1 (within-segment decay)")

    print("\n--- Model 2 ---")
    tr2, va2, te2, f2 = prepare_m2(held_out)
    m2 = [run_m2_naive(te2, held_out)]
    for name in ALGO_ORDER:
        model = (LGBMRegressor(**BOOSTING_PARAMS_M2) if name == "LightGBM"
                 else copy.deepcopy(templates[name]))
        try:
            m2.append(run_m2(name, model, tr2, va2, te2, f2, held_out))
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    print_table(m2, held_out, "Model 2 (post-tillage recovery)")

    print("\n--- Figure + save ---")
    plot_comparison(m1, m2)
    save_results(m1, m2, held_out)
    save_si_table(m1, m2, held_out)

    print("\n  >> Paste both tables here so SI Table S2 + the §2.4 claim get real numbers.")
    print("Done.")


if __name__ == "__main__":
    main()