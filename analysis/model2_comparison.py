"""
analysis/model2_comparison.py — 5-algorithm comparison for Model 2 (V2)
========================================================================
Compares 5 regression models across 3 conditions:
  Chrono     : clean basins chronological split
  Held-out D : clean train, 5 unseen basins test
  Held-out E : all 45 basins train, 5 unseen basins test

All metrics on raw IRD (cm/h) after back-transform.
SHAP on LightGBM for all 3 conditions.

Usage
-----
  python -m analysis.model2_comparison
"""
from __future__ import annotations

import sys
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESET_CSV, TABLES_DIR, RANDOM_SEED
from pipeline.features import MODEL2_FEATURES, TARGET_M2


# ─────────────────────────────────────────────────────────────────────────────
# Metrics and back-transform
# ─────────────────────────────────────────────────────────────────────────────

def _back_transform(y_pred_log: np.ndarray, prev_ird: np.ndarray) -> np.ndarray:
    return prev_ird * np.exp(y_pred_log)


def _metrics_raw(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "",
    verbose: bool = True,
) -> dict:
    mask = (
        np.isfinite(y_true) & np.isfinite(y_pred) &
        (y_true > 0) & (y_pred > 0)
    )
    if mask.sum() < 2:
        return {}
    yt, yp     = y_true[mask], y_pred[mask]
    rmse       = float(np.sqrt(mean_squared_error(yt, yp)))
    ird_mean   = float(np.mean(yt))
    spear_r, _ = sp_stats.spearmanr(yt, yp)
    m = dict(
        r2         = round(float(r2_score(yt, yp)),               4),
        rmse       = round(rmse,                                    4),
        mae        = round(float(mean_absolute_error(yt, yp)),     4),
        mape       = round(float(np.mean(np.abs((yt-yp)/yt))*100), 2),
        spearman_r = round(float(spear_r),                         4),
        rel_rmse   = round(rmse / ird_mean if ird_mean > 0 else np.nan, 4),
        n          = int(mask.sum()),
    )
    if verbose and label:
        print(
            f"  {label:<45}  "
            f"R²={m['r2']:+.4f}  "
            f"RMSE={m['rmse']:.4f} cm/h  "
            f"MAPE={m['mape']:.1f}%"
        )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and condition preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not RESET_CSV.exists():
        raise FileNotFoundError(f"{RESET_CSV} not found.")
    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"  Reset dataset : {len(df)} rows  "
          f"{df['basin_number'].nunique()} basins")
    return df


def prepare_condition(
    df:       pd.DataFrame,
    condition: str,
) -> tuple[pd.DataFrame, str, str]:
    """
    Returns (df_condition, split_col, test_val) for a given condition.

    Conditions:
      chrono     : split_chrono, test='test'
      held_out_D : split_held_out, test='held_out_test',
                   only clean + held_out basins
      held_out_E : split_held_out, test='held_out_test',
                   all non-excluded basins
    """
    if condition == "chrono":
        return df.copy(), "split_chrono", "test"
    elif condition == "held_out_D":
        df_d = df[df["basin_role"].isin(["clean", "held_out"])].copy()
        return df_d, "split_held_out", "held_out_test"
    elif condition == "held_out_E":
        df_e = df.copy()
        df_e = _add_outlier_train_splits(df_e)
        # Now outlier basins have train/val rows, held_out basins have held_out_test
        return df_e, "split_held_out", "held_out_test"
    else:
        raise ValueError(f"Unknown condition: {condition}")


def compute_naive(
    df: pd.DataFrame, split_col: str, test_val: str
) -> dict:
    test = df[df[split_col] == test_val].copy()
    valid = (
        test["IRD_at_reset"].notna() &
        test["prev_IRD_at_reset_raw"].notna() &
        (test["IRD_at_reset"] > 0) &
        (test["prev_IRD_at_reset_raw"] > 0)
    )
    if valid.sum() < 2:
        return {}
    return _metrics_raw(
        test.loc[valid, "IRD_at_reset"].values.astype(float),
        test.loc[valid, "prev_IRD_at_reset_raw"].values.astype(float),
        verbose=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm registry
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    from xgboost  import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    return {
        "Ridge": (
            Ridge(alpha=1.0, random_state=RANDOM_SEED),
            True,
        ),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=500, max_depth=None,
                min_samples_leaf=2, max_features="sqrt",
                n_jobs=-1, random_state=RANDOM_SEED,
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=1000, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                early_stopping_rounds=50, eval_metric="rmse",
                random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
            ),
            False,
        ),
        "LightGBM": (
            LGBMRegressor(
                n_estimators=1000, max_depth=-1, num_leaves=31,
                learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
                min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            ),
            False,
        ),
        "CatBoost": (
            CatBoostRegressor(
                iterations=1000, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, min_data_in_leaf=5,
                early_stopping_rounds=50,
                random_seed=RANDOM_SEED, verbose=0,
            ),
            False,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate one algorithm on one condition
# ─────────────────────────────────────────────────────────────────────────────

def _add_outlier_train_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    For condition E: reassign outlier basin rows from 'excluded'
    to chrono-style train/val splits so they contribute to training.
    Outlier basins are never in the held-out test set.
    """
    df = df.copy()
    outlier_mask = (
        (df["basin_role"] == "outlier") &
        (df["split_held_out"] == "excluded")
    )
    # Use their split_chrono assignment as the training split
    df.loc[outlier_mask, "split_held_out"] = df.loc[
        outlier_mask, "split_chrono"
    ]
    return df


def run_one_model(
    name:      str,
    model,
    scale:     bool,
    df:        pd.DataFrame,
    feat_cols: list[str],
    split_col: str,
    test_val:  str,
) -> dict:
    from xgboost  import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    required = feat_cols + [TARGET_M2, "IRD_at_reset", "prev_IRD_at_reset_raw"]
    train = df[df[split_col] == "train"].dropna(subset=required).reset_index(drop=True)
    val   = df[df[split_col] == "val"].dropna(  subset=required).reset_index(drop=True)
    test  = df[df[split_col] == test_val].dropna(subset=required).reset_index(drop=True)

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        print(f"    {name}: insufficient data — skipping")
        return {}

    sc = StandardScaler()
    if scale:
        Xtr = sc.fit_transform(train[feat_cols].values)
        Xva = sc.transform(val[feat_cols].values)
        Xte = sc.transform(test[feat_cols].values)
    else:
        Xtr = sc.fit_transform(train[feat_cols].values)
        Xva = sc.transform(val[feat_cols].values)
        Xte = sc.transform(test[feat_cols].values)

    try:
        if isinstance(model, XGBRegressor):
            model.fit(
                Xtr, train[TARGET_M2].values,
                eval_set=[(Xva, val[TARGET_M2].values)],
                verbose=False,
            )
        elif isinstance(model, LGBMRegressor):
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                Xtr, train[TARGET_M2].values,
                eval_set=[(Xva, val[TARGET_M2].values)],
                callbacks=[
                    early_stopping(50, verbose=False),
                    log_evaluation(period=-1),
                ],
            )
        elif isinstance(model, CatBoostRegressor):
            from catboost import Pool
            model.fit(
                Pool(Xtr, train[TARGET_M2].values),
                eval_set=Pool(Xva, val[TARGET_M2].values),
                verbose=False,
            )
        else:
            model.fit(Xtr, train[TARGET_M2].values)
    except Exception as e:
        print(f"    ERROR fitting {name}: {e}")
        return {}

    y_pred_log = model.predict(Xte)
    prev_ird   = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_pred   = _back_transform(y_pred_log, prev_ird)
    ird_true   = test["IRD_at_reset"].values.astype(float)

    m = _metrics_raw(ird_true, ird_pred, verbose=False)
    if not m:
        return {}

    m["model"]      = name
    m["_model"]     = model
    m["_scaler"]    = sc
    m["_Xte"]       = Xte
    m["_feat_cols"] = feat_cols

    print(
        f"    {name:<15}  "
        f"R²={m['r2']:+.4f}  "
        f"RMSE={m['rmse']:.4f} cm/h  "
        f"MAPE={m['mape']:.1f}%"
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(
    cond_results: dict,
    condition:    str,
    feat_cols:    list[str],
) -> pd.DataFrame | None:
    if "LightGBM" not in cond_results:
        return None
    res   = cond_results["LightGBM"]
    model = res.get("_model")
    Xte   = res.get("_Xte")
    if model is None or Xte is None:
        return None

    feature_names = feat_cols[:Xte.shape[1]]
    print(f"  Computing SHAP — LightGBM, {condition}...")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xte)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap":     shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n  SHAP — {condition}:")
    print(f"  {'Rank':<5}  {'Feature':<30}  "
          f"{'|SHAP|':>10}  {'SHAP':>10}  Direction")
    print(f"  {'-'*65}")
    for i, row in shap_df.iterrows():
        direction = "→ higher recovery" if row["mean_shap"] > 0 \
                    else "→ lower recovery"
        print(
            f"  {i+1:<5}  {row['feature']:<30}  "
            f"{row['mean_abs_shap']:>10.4f}  "
            f"{row['mean_shap']:>10.4f}  {direction}"
        )

    n_sample     = min(2000, len(Xte))
    idx          = np.random.default_rng(RANDOM_SEED).choice(
        len(Xte), n_sample, replace=False
    )
    shap_plot_df = pd.DataFrame(Xte[idx], columns=feature_names)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[idx], shap_plot_df,
        show=False, plot_size=None,
    )
    plt.title(
        f"SHAP — LightGBM Model 2 — {condition}\n"
        f"Target: {TARGET_M2}\n"
        "Positive SHAP → higher recovery ratio",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return shap_df


def plot_shap_comparison(shap_results: dict) -> None:
    """Side-by-side SHAP bar charts across conditions."""
    if len(shap_results) < 2:
        return

    conditions = list(shap_results.keys())
    ref_cond   = conditions[0]
    ref_order  = shap_results[ref_cond]["feature"].tolist()

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(6 * len(conditions), 8),
                             sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    colors = {"chrono": "tomato", "held_out_D": "steelblue",
               "held_out_E": "seagreen"}

    for ax, cond in zip(axes, conditions):
        sr    = shap_results[cond].set_index("feature")
        vals  = [sr.loc[f, "mean_abs_shap"] if f in sr.index else 0.0
                 for f in ref_order]
        color = colors.get(cond, "gray")

        ax.barh(range(len(ref_order)), vals,
                color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(ref_order)))
        ax.set_yticklabels(ref_order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP|", fontsize=9)
        ax.set_title(cond, fontsize=10, fontweight="bold", color=color)
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle(
        "Model 2 — SHAP comparison across conditions\n"
        "Stable rankings → robust physical interpretation",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    all_results: dict,
    naive_dict:  dict,
) -> None:
    """Bar chart: R², RMSE, MAPE for all algorithms × all conditions."""
    conditions  = list(all_results.keys())
    metrics_cfg = [
        ("r2",   "R² (raw IRD)",    True),
        ("rmse", "RMSE (cm/h)",     False),
        ("mape", "MAPE %",          False),
    ]

    fig, axes = plt.subplots(
        len(metrics_cfg), len(conditions),
        figsize=(5 * len(conditions), 4 * len(metrics_cfg)),
        squeeze=False,
    )
    fig.suptitle(
        "Model 2 — algorithm comparison across all conditions\n"
        "Metrics on raw IRD (cm/h) after back-transform  |  "
        "LightGBM = primary model",
        fontsize=11, fontweight="bold",
    )

    for col, condition in enumerate(conditions):
        cond_res    = all_results[condition]
        model_names = list(cond_res.keys())
        x           = np.arange(len(model_names))
        colors      = [
            "tomato" if n == "LightGBM" else "steelblue"
            for n in model_names
        ]

        for row, (metric, label, higher_better) in enumerate(metrics_cfg):
            ax   = axes[row][col]
            vals = [cond_res[m].get(metric, np.nan) for m in model_names]

            bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.02,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=6, rotation=90,
                    )

            # Naive reference line
            naive_val = naive_dict.get(condition, {}).get(metric, np.nan)
            if np.isfinite(naive_val):
                ax.axhline(
                    naive_val, color="black", linewidth=1.5,
                    linestyle="--", alpha=0.6,
                    label=f"Naive={naive_val:.3f}",
                )
                ax.legend(fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=40,
                               ha="right", fontsize=7)
            ax.grid(True, alpha=0.2, axis="y")

            if col == 0:
                arrow = "↑" if higher_better else "↓"
                ax.set_ylabel(f"{label} {arrow}", fontsize=8)
            if row == 0:
                ax.set_title(condition, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_lgbm_condition_summary(
    all_results: dict,
    naive_dict:  dict,
) -> None:
    """LightGBM only — all metrics across conditions."""
    conditions  = list(all_results.keys())
    metrics_cfg = [
        ("r2",   "R² (raw IRD)",  True),
        ("rmse", "RMSE (cm/h)",   False),
        ("mape", "MAPE %",        False),
    ]

    x      = np.arange(len(conditions))
    colors = ["tomato", "steelblue", "seagreen"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Model 2 LightGBM — performance across conditions\n"
        "Chrono=within-sample  |  D=clean held-out  |  E=all-data held-out\n"
        "Dashed = naive baseline per condition",
        fontsize=10, fontweight="bold",
    )

    for ax, (metric, label, higher_better) in zip(axes, metrics_cfg):
        vals  = [
            (all_results[c].get("LightGBM") or {}).get(metric, np.nan)
            for c in conditions
        ]
        naive = [
            naive_dict.get(c, {}).get(metric, np.nan)
            for c in conditions
        ]

        bars = ax.bar(x, vals, color=colors[:len(conditions)],
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}" if metric != "mape" else f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )

        # Naive reference markers
        for xi, nv in zip(x, naive):
            if np.isfinite(nv):
                ax.plot([xi - 0.4, xi + 0.4], [nv, nv],
                        color="black", linewidth=1.5,
                        linestyle="--", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=9)
        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{label}\n({arrow})", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    all_results: dict,
    naive_dict:  dict,
) -> None:
    rows = []
    for condition, cond_res in all_results.items():
        naive_m = naive_dict.get(condition, {})
        rows.append(dict(
            condition = condition,
            model     = "Naive",
            r2        = round(naive_m.get("r2",   np.nan), 4),
            rmse      = round(naive_m.get("rmse", np.nan), 4),
            mape      = round(naive_m.get("mape", np.nan), 2),
            n         = naive_m.get("n", np.nan),
        ))
        for name, m in cond_res.items():
            if not m:
                continue
            clean = {k: v for k, v in m.items() if not k.startswith("_")}
            rows.append(dict(
                condition = condition,
                model     = name,
                r2        = round(clean.get("r2",   np.nan), 4),
                rmse      = round(clean.get("rmse", np.nan), 4),
                mape      = round(clean.get("mape", np.nan), 2),
                n         = clean.get("n", np.nan),
            ))

    df_out   = pd.DataFrame(rows)
    out_path = TABLES_DIR / "model2_comparison_v2.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("  MODEL 2 COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Condition':<15}  {'Model':<15}  "
          f"{'R²(IRD)':>8}  {'RMSE':>8}  {'MAPE%':>7}")
    print(f"  {'-'*60}")
    for _, row in df_out.iterrows():
        tag = "  ← primary" if row["model"] == "LightGBM" else ""
        print(
            f"  {row['condition']:<15}  "
            f"{row['model']:<15}  "
            f"{row['r2']:>+8.4f}  "
            f"{row['rmse']:>8.4f}  "
            f"{row['mape']:>6.1f}%"
            f"{tag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  MODEL 2 COMPARISON V2 — analysis/model2_comparison.py")
    print(f"  Target: {TARGET_M2}")
    print("=" * 65)

    df     = load_data()
    models = get_models()
    avail  = [f for f in MODEL2_FEATURES if f in df.columns]
    print(f"  Features: {len(avail)}  {avail}")

    conditions = ["chrono", "held_out_D", "held_out_E"]
    all_results: dict = {}
    naive_dict:  dict = {}
    shap_results: dict = {}

    for condition in conditions:
        print(f"\n{'='*65}")
        print(f"  CONDITION: {condition}")
        print(f"{'='*65}")

        df_cond, split_col, test_val = prepare_condition(df, condition)
        naive_dict[condition] = compute_naive(df_cond, split_col, test_val)
        print(
            f"  Naive: R²={naive_dict[condition].get('r2', np.nan):+.4f}  "
            f"RMSE={naive_dict[condition].get('rmse', np.nan):.4f}"
        )

        cond_results: dict = {}
        for name, (model_template, scale) in models.items():
            print(f"\n  [{condition}] {name}")
            fresh = copy.deepcopy(model_template)
            try:
                result = run_one_model(
                    name, fresh, scale,
                    df_cond, avail, split_col, test_val,
                )
                if result:
                    cond_results[name] = result
            except Exception as e:
                print(f"    ERROR: {e}")

        if cond_results:
            all_results[condition] = cond_results

        # SHAP for LightGBM on all conditions
        if "LightGBM" in cond_results:
            sr = compute_shap(cond_results, condition, avail)
            if sr is not None:
                shap_results[condition] = sr

    # SHAP comparison
    if len(shap_results) >= 2:
        print("\n--- SHAP comparison ---")
        plot_shap_comparison(shap_results)

    # Figures
    if all_results:
        print("\n--- Figure: algorithm comparison ---")
        plot_comparison(all_results, naive_dict)

        print("\n--- Figure: LightGBM across conditions ---")
        plot_lgbm_condition_summary(all_results, naive_dict)

        save_results(all_results, naive_dict)

    print("\nDone.")




if __name__ == "__main__":
    main()