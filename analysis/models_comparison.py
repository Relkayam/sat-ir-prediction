"""
analysis/model_comparison.py — Algorithm comparison for Model 1 and Model 2
============================================================================
Compares multiple ML algorithms against the naive baseline on the same
5 bootstrap-selected held-out basins (read from data/selected_basins.csv).

Training : Condition E-full — all non-held-out basins including outliers
Test     : 5 selected held-out basins (same for all algorithms)
Metric   : raw IRD (cm/h) after back-transform — same space for both models

Algorithms
----------
  Naive         — predict no change (delta = 0) — the baseline to beat
  Ridge         — linear baseline
  RandomForest  — bagged trees
  XGBoost       — gradient boosting
  LightGBM      — primary model (published paper result)

Both Model 1 (within-segment decay) and Model 2 (post-tillage recovery)
are evaluated in a single run using the same 5 held-out basins.

Output
------
  Printed per-basin and pooled tables for each model
  Bar chart figure: all algorithms x both models
  outputs/tables/model_comparison.xlsx

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

ALGO_COLORS = {
    "Naive":        "#AAAAAA",
    "Ridge":        "#4BACC6",
    "RandomForest": "#70AD47",
    "XGBoost":      "#ED7D31",
    "LightGBM":     "#065A82",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_true > 0) & (y_pred > 0))
    if mask.sum() < 2:
        return {}
    yt, yp   = y_true[mask], y_pred[mask]
    rmse     = float(np.sqrt(mean_squared_error(yt, yp)))
    ird_mean = float(np.mean(yt))
    return dict(
        r2       = round(float(r2_score(yt, yp)),                    4),
        rmse     = round(rmse,                                         4),
        mape     = round(float(np.mean(np.abs((yt-yp)/yt)) * 100),   2),
        rel_rmse = round(rmse / ird_mean if ird_mean > 0 else np.nan, 4),
        n        = int(mask.sum()),
    )


def _collect_results(
    name:      str,
    ird_true:  np.ndarray,
    ird_pred:  np.ndarray,
    test_df:   pd.DataFrame,
    held_out:  list[int],
    basin_col: str = "basin_number",
) -> dict:
    basins_arr   = test_df[basin_col].values
    per_basin: dict = {}
    all_t, all_p = [], []

    for bn in held_out:
        mask_bn = basins_arr == bn
        yt      = ird_true[mask_bn]
        yp      = ird_pred[mask_bn]
        m       = _metrics(yt, yp)
        per_basin[bn] = m
        valid = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
        all_t.append(yt[valid])
        all_p.append(yp[valid])

    pooled = _metrics(np.concatenate(all_t), np.concatenate(all_p))
    return dict(name=name, per_basin=per_basin, pooled=pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Load selected basins
# ─────────────────────────────────────────────────────────────────────────────

def load_selected_basins() -> list[int]:
    if not SELECTED_BASINS_CSV.exists():
        raise FileNotFoundError(
            f"{SELECTED_BASINS_CSV} not found.\n"
            "Run: python -m experiments.run_bootstrap"
        )
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
# Model 1
# ─────────────────────────────────────────────────────────────────────────────

def prepare_m1(held_out: list[int]) -> tuple[pd.DataFrame, pd.DataFrame,
                                              pd.DataFrame, list[str]]:
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df, feat_cols = prepare_features(df)

    held_set = set(held_out)
    required = feat_cols + [TARGET_M1]

    test = df[
        df["basin_number"].isin(held_set) &
        (df["row_type"]        == "event") &
        (df["is_good_segment"] == True)
    ].dropna(subset=required).reset_index(drop=True)

    non_ho = df[~df["basin_number"].isin(held_set) & (df["row_type"] == "event")]
    train  = non_ho.dropna(subset=required).reset_index(drop=True)
    val    = non_ho[
        (non_ho["split"] == "val") & (non_ho["is_good_segment"] == True)
    ].dropna(subset=required).reset_index(drop=True)

    print(f"  M1: {len(df):,} rows  train={len(train):,} "
          f"({non_ho['basin_number'].nunique()} basins)  "
          f"val={len(val):,}  test={len(test):,}  "
          f"features={len(feat_cols)}")
    return train, val, test, feat_cols


def run_m1(name: str, model, train: pd.DataFrame, val: pd.DataFrame,
           test: pd.DataFrame, feat_cols: list[str],
           held_out: list[int]) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from xgboost  import XGBRegressor

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)
    ytr = train[TARGET_M1].values
    yva = val[TARGET_M1].values

    try:
        if isinstance(model, XGBRegressor):
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        elif isinstance(model, LGBMRegressor):
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                      callbacks=[early_stopping(EARLY_STOPPING_ROUNDS_M1, verbose=False),
                                 log_evaluation(period=-1)])
        else:
            model.fit(Xtr, ytr)
    except Exception as e:
        print(f"    ERROR: {e}"); return {}

    ird_reset = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    ird_pred  = ird_reset * np.exp(model.predict(Xte))
    ird_true  = ird_reset * np.exp(
        pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    return _collect_results(name, ird_true, ird_pred, test, held_out)


def run_m1_naive(test: pd.DataFrame, held_out: list[int]) -> dict:
    ird_reset = pd.to_numeric(test["IRD_at_reset"], errors="coerce").values
    ird_true  = ird_reset * np.exp(
        pd.to_numeric(test[TARGET_M1], errors="coerce").values)
    ird_pred  = ird_reset.copy()   # predict: IRD stays at reset level (decay=0)
    return _collect_results("Naive", ird_true, ird_pred, test, held_out)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2
# ─────────────────────────────────────────────────────────────────────────────

def prepare_m2(held_out: list[int]) -> tuple[pd.DataFrame, pd.DataFrame,
                                              pd.DataFrame, list[str]]:
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    feat_cols = [f for f in MODEL2_FEATURES if f in df.columns]
    missing   = [f for f in MODEL2_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING missing M2 features: {missing}")

    held_set  = set(held_out)
    required  = feat_cols + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    avail_req = [c for c in required if c in df.columns]

    test   = df[df["basin_number"].isin(held_set)].dropna(
        subset=avail_req).reset_index(drop=True)
    non_ho = df[~df["basin_number"].isin(held_set)]
    train  = non_ho.dropna(subset=avail_req).reset_index(drop=True)
    val    = non_ho[non_ho["split_chrono"] == "val"].dropna(
        subset=avail_req).reset_index(drop=True)

    print(f"  M2: {len(df)} rows  train={len(train)} "
          f"({non_ho['basin_number'].nunique()} basins)  "
          f"val={len(val)}  test={len(test)}  "
          f"features={len(feat_cols)}")
    return train, val, test, feat_cols


def run_m2(name: str, model, train: pd.DataFrame, val: pd.DataFrame,
           test: pd.DataFrame, feat_cols: list[str],
           held_out: list[int]) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from xgboost  import XGBRegressor

    required = feat_cols + [TARGET_M2]
    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)
    ytr = train[TARGET_M2].values
    yva = val[TARGET_M2].values

    try:
        if isinstance(model, XGBRegressor):
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        elif isinstance(model, LGBMRegressor):
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                      callbacks=[early_stopping(EARLY_STOPPING_ROUNDS_M2, verbose=False),
                                 log_evaluation(period=-1)])
        else:
            model.fit(Xtr, ytr)
    except Exception as e:
        print(f"    ERROR: {e}"); return {}

    prev_ird = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_pred = prev_ird * np.exp(model.predict(Xte))
    ird_true = test["IRD_at_reset"].values.astype(float)
    return _collect_results("Naive" if name == "Naive" else name,
                            ird_true, ird_pred, test, held_out)


def run_m2_naive(test: pd.DataFrame, held_out: list[int]) -> dict:
    ird_pred = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_true = test["IRD_at_reset"].values.astype(float)
    return _collect_results("Naive", ird_true, ird_pred, test, held_out)


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list[dict], held_out: list[int], label: str) -> None:
    print(f"\n{'='*90}")
    print(f"  {label} — Algorithm comparison  |  E-full  |  5 held-out basins")
    print(f"  Metric: raw IRD (cm/h)  |  tick = beats naive RMSE")
    print(f"{'='*90}")

    naive = next((r for r in results if r["name"] == "Naive"), None)

    # Header
    print(f"  {'Basin':<7}  {'Field':<12}", end="")
    for r in results:
        print(f"  {r['name']:^21}", end="")
    print()
    print(f"  {'':<7}  {'':<12}", end="")
    for _ in results:
        print(f"  {'RMSE':>7} {'R2':>6} {'MAPE':>5}", end="")
    print()
    print(f"  {'-'*85}")

    for bn in held_out + ["POOLED"]:
        field = FIELD_NAMES.get(int(str(bn)[0]), "") if bn != "POOLED" else "all"
        print(f"  {str(bn):<7}  {field:<12}", end="")
        naive_rmse = (naive["per_basin"].get(bn, {}).get("rmse", np.nan)
                      if bn != "POOLED"
                      else naive["pooled"].get("rmse", np.nan))
        for r in results:
            m    = r["per_basin"].get(bn, {}) if bn != "POOLED" else r["pooled"]
            rmse = m.get("rmse", np.nan)
            r2   = m.get("r2",   np.nan)
            mape = m.get("mape", np.nan)
            beat = "" if r["name"] == "Naive" else \
                   ("+" if rmse < naive_rmse else "-")
            print(f"  {rmse:>7.3f} {r2:>+6.3f} {mape:>4.1f}%{beat}", end="")
        print()

    print(f"\n  Summary vs naive (RMSE):")
    for r in results:
        if r["name"] == "Naive": continue
        wins = sum(1 for bn in held_out
                   if r["per_basin"].get(bn, {}).get("rmse", np.inf) <
                   naive["per_basin"].get(bn, {}).get("rmse", np.inf))
        p_rmse = r["pooled"].get("rmse", np.nan)
        n_rmse = naive["pooled"].get("rmse", np.nan)
        delta  = n_rmse - p_rmse
        print(f"    {r['name']:<15} {wins}/5 basins  "
              f"pooled RMSE={p_rmse:.3f}  "
              f"delta={delta:+.3f} cm/h ({100*delta/n_rmse:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(m1_results: list[dict], m2_results: list[dict]) -> None:
    metrics_cfg = [
        ("r2",   "R\u00b2 (raw IRD)",  True),
        ("rmse", "RMSE (cm/h)",         False),
        ("mape", "MAPE %",              False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Algorithm comparison — Model 1 and Model 2\n"
        "Condition E-full  |  5 bootstrap-selected held-out basins  |  "
        "Raw IRD space (cm/h)  |  Dashed = naive baseline",
        fontsize=9, fontweight="bold",
    )

    for row_idx, (results, model_label) in enumerate([
        (m1_results, "Model 1\nWithin-segment decay"),
        (m2_results, "Model 2\nPost-tillage recovery"),
    ]):
        naive = next(r for r in results if r["name"] == "Naive")
        algos = [r for r in results if r["name"] != "Naive"]
        names  = [r["name"] for r in algos]
        colors = [ALGO_COLORS.get(n, "#888888") for n in names]
        x      = np.arange(len(names))

        for col_idx, (metric, label, higher_better) in enumerate(metrics_cfg):
            ax    = axes[row_idx][col_idx]
            vals  = [r["pooled"].get(metric, np.nan) for r in algos]
            naive_val = naive["pooled"].get(metric, np.nan)

            bars = ax.bar(x, vals, color=colors, alpha=0.85,
                          edgecolor="white", linewidth=0.5)

            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + abs(bar.get_height()) * 0.02,
                        f"{val:.3f}" if metric != "mape" else f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=7.5,
                    )

            if np.isfinite(naive_val):
                ax.axhline(naive_val, color="black", linewidth=1.4,
                           linestyle="--", alpha=0.65,
                           label=f"Naive={naive_val:.3f}")
                ax.legend(fontsize=7, loc="upper right")

            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")

            arrow = "\u2191 better" if higher_better else "\u2193 better"
            if col_idx == 0:
                ax.set_ylabel(model_label, fontsize=8, fontweight="bold")
            ax.set_title(f"{label}  ({arrow})", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_results(m1: list[dict], m2: list[dict], held_out: list[int]) -> None:
    rows = []
    for model_label, results in [("Model1", m1), ("Model2", m2)]:
        for r in results:
            p   = r["pooled"]
            row = dict(
                model_type = model_label,
                algorithm  = r["name"],
                rmse_pool  = round(p.get("rmse", np.nan), 4),
                r2_pool    = round(p.get("r2",   np.nan), 4),
                mape_pool  = round(p.get("mape", np.nan), 2),
                n_pool     = p.get("n", np.nan),
            )
            for bn in held_out:
                m     = r["per_basin"].get(bn, {})
                field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
                row[f"rmse_{bn}"] = round(m.get("rmse", np.nan), 4)
                row[f"r2_{bn}"]   = round(m.get("r2",   np.nan), 4)
            rows.append(row)

    pd.DataFrame(rows).to_excel(TABLES_DIR / "model_comparison.xlsx", index=False)
    print(f"  Saved: model_comparison.xlsx")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("="*65)
    print("  MODEL COMPARISON — Model 1 and Model 2")
    print("  Condition E-full  |  5 bootstrap-selected held-out basins")
    print("  Metric: raw IRD (cm/h) after back-transform")
    print("="*65)

    from lightgbm import LGBMRegressor
    from xgboost  import XGBRegressor

    held_out = load_selected_basins()

    algo_templates = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED),
        "XGBoost": XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=50, eval_metric="rmse",
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
        "LightGBM": None,
    }

    # ── Model 1 ───────────────────────────────────────────────────────────────
    print("\n--- Model 1 ---")
    m1_train, m1_val, m1_test, feat_m1 = prepare_m1(held_out)
    m1_results: list[dict] = [run_m1_naive(m1_test, held_out)]

    for name, template in algo_templates.items():
        print(f"  {name}:")
        model = (LGBMRegressor(**BOOSTING_PARAMS_M1)
                 if name == "LightGBM" else copy.deepcopy(template))
        try:
            res = run_m1(name, model, m1_train, m1_val, m1_test,
                         feat_m1, held_out)
            if res: m1_results.append(res)
        except Exception as e:
            print(f"    ERROR: {e}")

    print_table(m1_results, held_out, "Model 1 — Within-segment decay")

    # ── Model 2 ───────────────────────────────────────────────────────────────
    print("\n--- Model 2 ---")
    m2_train, m2_val, m2_test, feat_m2 = prepare_m2(held_out)
    m2_results: list[dict] = [run_m2_naive(m2_test, held_out)]

    for name, template in algo_templates.items():
        print(f"  {name}:")
        model = (LGBMRegressor(**BOOSTING_PARAMS_M2)
                 if name == "LightGBM" else copy.deepcopy(template))
        try:
            res = run_m2(name, model, m2_train, m2_val, m2_test,
                         feat_m2, held_out)
            if res: m2_results.append(res)
        except Exception as e:
            print(f"    ERROR: {e}")

    print_table(m2_results, held_out, "Model 2 — Post-tillage recovery")

    # ── Figure + save ─────────────────────────────────────────────────────────
    print("\n--- Figure ---")
    plot_comparison(m1_results, m2_results)
    save_results(m1_results, m2_results, held_out)
    print("\nDone.")


if __name__ == "__main__":
    main()