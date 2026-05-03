"""
analysis/model_comparison.py — Multi-condition model comparison + SHAP (V2)
============================================================================
V2 changes over V1:
  - Runs all 5 evaluation conditions (A, B, C, D, E)
  - SHAP produced for conditions A, D, E (paper model + generalisability)
  - Basin sets resolved at runtime from outlier_basins.csv + basin_role
  - All algorithms use well-established defaults, early stopping where supported
  - Single comparison table across all conditions and algorithms

Algorithms
----------
  Ridge         — linear baseline
  RandomForest  — bagged trees
  XGBoost       — gradient boosting
  LightGBM      — gradient boosting (primary Model 1)
  CatBoost      — gradient boosting

SHAP conditions
---------------
  A — clean model      (paper model, within-sample test)
  D — held-out basins  (clean training, unseen basin test)
  E — all-data held-out (all 45 basins including outliers, unseen basin test)

  Comparing A vs D vs E SHAP rankings tests whether physical interpretation
  is stable across data quality choices.

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
import matplotlib.gridspec as gridspec
import shap

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, OUTLIER_CSV, TABLES_DIR,
    RANDOM_SEED, TRAIN_FRAC, VAL_FRAC,
)
from pipeline.features import prepare_features, TARGET_M1
from models.utils import (
    metrics_norm, metrics_ird, back_transform,
    per_basin_median_r2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Basin set resolution — identical logic to model1_decay.py
# ─────────────────────────────────────────────────────────────────────────────

def load_outlier_basins() -> set[int]:
    """Load outlier_basins.csv. Returns empty set if not found."""
    if not OUTLIER_CSV.exists():
        print(f"  WARNING: {OUTLIER_CSV} not found — no basin exclusions")
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
    print(f"  Outlier basins: {sorted(excluded)}")
    return excluded


def resolve_basin_sets(
    df:             pd.DataFrame,
    outlier_basins: set[int],
) -> dict:
    """Resolve all basin sets from data at runtime."""
    all_basins = set(df["basin_number"].dropna().unique().astype(int).tolist())

    if "basin_role" in df.columns:
        held_out = set(
            df.loc[df["basin_role"] == "held_out", "basin_number"]
            .dropna().unique().astype(int).tolist()
        )
    else:
        held_out = set()

    clean_basins       = all_basins - outlier_basins - held_out
    held_out_train     = clean_basins
    all_held_out_train = all_basins - held_out

    print(f"  Basin sets: all={len(all_basins)}  "
          f"outlier={len(outlier_basins)}  "
          f"held_out={len(held_out)}  "
          f"clean={len(clean_basins)}")

    return dict(
        all              = all_basins,
        outlier          = outlier_basins,
        held_out         = held_out,
        clean            = clean_basins,
        held_out_train   = held_out_train,
        all_held_out_train = all_held_out_train,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Split reassignment
# ─────────────────────────────────────────────────────────────────────────────

def _reassign_splits(df: pd.DataFrame) -> pd.Series:
    """Random 70/15/15 splits by segment."""
    split_col = pd.Series("excluded", index=df.index)
    seg_ids   = sorted([
        int(s) for s in df["segment_id"].dropna().unique() if s >= 0
    ])
    if len(seg_ids) < 5:
        return split_col

    rng      = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(seg_ids)
    n        = len(shuffled)
    n_tr     = max(1, round(n * TRAIN_FRAC))
    n_va     = max(1, round(n * VAL_FRAC))
    train_s  = set(shuffled[:n_tr].tolist())
    val_s    = set(shuffled[n_tr:n_tr + n_va].tolist())

    def _label(s):
        if pd.isna(s) or int(s) < 0: return "excluded"
        s = int(s)
        if s in train_s: return "train"
        if s in val_s:   return "val"
        return "test"

    return df["segment_id"].apply(_label)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading per condition
# ─────────────────────────────────────────────────────────────────────────────

def load_condition(
    df_full:    pd.DataFrame,
    condition:  str,
    basin_sets: dict,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    """
    Filter and prepare data for one evaluation condition.
    Returns (df_ready, feat_cols) or (None, None) if condition cannot run.
    """
    df = df_full.copy()

    if condition == "A":
        df = df[
            (df["basin_number"].isin(basin_sets["clean"])) &
            (df["row_type"]         == "event") &
            (df["is_good_segment"]  == True)
        ].copy()

    elif condition == "B":
        df = df[
            (df["basin_number"].isin(basin_sets["clean"])) &
            (df["row_type"] == "event")
        ].copy()
        df["split"] = _reassign_splits(df)

    elif condition == "C":
        df = df[df["row_type"] == "event"].copy()
        df["split"] = _reassign_splits(df)

    elif condition == "D":
        train_b  = basin_sets["held_out_train"]
        held_out = basin_sets["held_out"]
        if not held_out:
            print(f"  SKIP D — no held-out basins")
            return None, None
        df = df[
            (df["basin_number"].isin(train_b | held_out)) &
            (df["row_type"]        == "event") &
            (df["is_good_segment"] == True)
        ].copy()
        if "split_held_out" not in df.columns:
            print(f"  SKIP D — split_held_out column missing")
            return None, None
        df["split"] = df["split_held_out"].replace({"held_out_test": "test"})

    elif condition == "E":
        train_b  = basin_sets["all_held_out_train"]
        held_out = basin_sets["held_out"]
        if not held_out:
            print(f"  SKIP E — no held-out basins")
            return None, None
        df = df[
            (df["basin_number"].isin(train_b | held_out)) &
            (df["row_type"] == "event")
        ].copy()
        if "split_held_out" not in df.columns:
            print(f"  SKIP E — split_held_out column missing")
            return None, None
        df["split"] = df["split_held_out"].replace({"held_out_test": "test"})
        needs_split = ~df["split"].isin(["train", "val", "test"])
        df.loc[needs_split, "split"] = _reassign_splits(
            df[needs_split]
        ).values

    else:
        raise ValueError(f"Unknown condition: {condition}")

    df = df[df["split"].isin(["train", "val", "test"])].copy()

    if len(df) < 100:
        print(f"  SKIP {condition} — insufficient events ({len(df)})")
        return None, None

    df, feat_cols = prepare_features(df)
    n_tr = int((df["split"] == "train").sum())
    n_va = int((df["split"] == "val").sum())
    n_te = int((df["split"] == "test").sum())
    print(f"  Condition {condition}: {len(df)} events  "
          f"{df['basin_number'].nunique()} basins  "
          f"train={n_tr}  val={n_va}  test={n_te}")
    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm registry — well-established defaults, early stopping where supported
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict[str, tuple]:
    """
    Returns {name: (model_instance, needs_scaling)}.

    Hyperparameters: well-established defaults for tabular regression.
    No grid search — consistent with the paper's claim that the framework
    works without extensive tuning (supporting transferability).

    Early stopping applied where natively supported (XGBoost, LightGBM).
    CatBoost uses its own overfitting detector.
    RandomForest and Ridge have no early stopping equivalent.
    """
    from xgboost  import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    return {
        "Ridge": (
            Ridge(alpha=1.0, random_state=RANDOM_SEED),
            True,   # needs scaling — Ridge is scale-sensitive
        ),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators    = 500,
                max_depth       = None,  # fully grown trees
                min_samples_leaf= 2,
                max_features    = "sqrt",
                n_jobs          = -1,
                random_state    = RANDOM_SEED,
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators          = 1000,
                max_depth             = 6,
                learning_rate         = 0.05,
                subsample             = 0.8,
                colsample_bytree      = 0.8,
                min_child_weight      = 5,
                gamma                 = 0.1,
                reg_alpha             = 0.1,
                reg_lambda            = 1.0,
                early_stopping_rounds = 50,
                eval_metric           = "rmse",
                random_state          = RANDOM_SEED,
                n_jobs                = -1,
                verbosity             = 0,
            ),
            False,
        ),
        "LightGBM": (
            LGBMRegressor(
                n_estimators     = 1000,
                max_depth        = -1,    # no limit — controlled by num_leaves
                num_leaves       = 63,
                learning_rate    = 0.05,
                subsample        = 0.8,
                feature_fraction = 0.8,
                min_child_samples= 20,
                reg_alpha        = 0.1,
                reg_lambda       = 1.0,
                random_state     = RANDOM_SEED,
                n_jobs           = -1,
                verbose          = -1,
            ),
            False,
        ),
        "CatBoost": (
            CatBoostRegressor(
                iterations          = 1000,
                depth               = 6,
                learning_rate       = 0.05,
                l2_leaf_reg         = 3.0,
                min_data_in_leaf    = 5,
                early_stopping_rounds = 50,
                random_seed         = RANDOM_SEED,
                verbose             = 0,
            ),
            False,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate one algorithm on one condition
# ─────────────────────────────────────────────────────────────────────────────

def run_one_model(
    name:      str,
    model,
    scale:     bool,
    df:        pd.DataFrame,
    feat_cols: list[str],
) -> dict | None:
    """
    Train on train split, evaluate on test split.
    Returns metrics dict or None on error.
    """
    from xgboost  import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    train = df[df["split"] == "train"].dropna(
        subset=feat_cols + [TARGET_M1]
    ).reset_index(drop=True)
    val   = df[df["split"] == "val"].dropna(
        subset=feat_cols + [TARGET_M1]
    ).reset_index(drop=True)
    test  = df[df["split"] == "test"].dropna(
        subset=feat_cols + [TARGET_M1]
    ).reset_index(drop=True)
    eval_df = df[df["split"].isin(["val", "test"])].dropna(
        subset=feat_cols + [TARGET_M1]
    ).reset_index(drop=True)

    if len(train) < 50 or len(val) < 10 or len(test) < 10:
        print(f"    SKIP — insufficient data")
        return None

    sc = StandardScaler()
    if scale:
        Xtr    = sc.fit_transform(train[feat_cols].values)
        Xva    = sc.transform(val[feat_cols].values)
        Xte    = sc.transform(test[feat_cols].values)
        X_eval = sc.transform(eval_df[feat_cols].values)
    else:
        Xtr    = sc.fit_transform(train[feat_cols].values)
        Xva    = sc.transform(val[feat_cols].values)
        Xte    = sc.transform(test[feat_cols].values)
        X_eval = sc.transform(eval_df[feat_cols].values)

    ytr = train[TARGET_M1].values
    yva = val[TARGET_M1].values
    yte = test[TARGET_M1].values

    # ── Fit ──────────────────────────────────────────────────────────────────
    try:
        if isinstance(model, XGBRegressor):
            model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                verbose=False,
            )
        elif isinstance(model, LGBMRegressor):
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                callbacks=[
                    early_stopping(50, verbose=False),
                    log_evaluation(period=-1),
                ],
            )
        elif isinstance(model, CatBoostRegressor):
            from catboost import Pool
            model.fit(
                Pool(Xtr, ytr),
                eval_set=Pool(Xva, yva),
                verbose=False,
            )
        else:
            # Ridge, RandomForest — no early stopping
            model.fit(Xtr, ytr)
    except Exception as e:
        print(f"    ERROR fitting {name}: {e}")
        return None

    # ── Metrics: IRD_norm_log ─────────────────────────────────────────────────
    y_pred = model.predict(Xte)
    m = metrics_norm(yte, y_pred, verbose=False) or {}

    # ── Metrics: raw IRD (cm/h) ───────────────────────────────────────────────
    if "IRD_at_reset" in test.columns:
        ird_reset = pd.to_numeric(
            test["IRD_at_reset"], errors="coerce"
        ).values.ravel()
        ird_true = back_transform(ird_reset, yte.ravel())
        ird_pred = back_transform(ird_reset, y_pred.ravel())
        m_ird    = metrics_ird(ird_true, ird_pred, verbose=False) or {}
        m["mape"]     = m_ird.get("mape",  np.nan)
        m["rmse_ird"] = m_ird.get("rmse",  np.nan)
        m["r2_ird"]   = m_ird.get("r2",    np.nan)

    # ── Per-basin median R² ───────────────────────────────────────────────────
    y_pred_eval       = model.predict(X_eval)
    m["basin_med_r2"] = per_basin_median_r2(eval_df, y_pred_eval, TARGET_M1)
    m["model"]        = name
    m["n_train"]      = len(train)
    m["n_test"]       = len(test)

    # Store for SHAP
    m["_model"]     = model
    m["_scaler"]    = sc
    m["_Xte"]       = Xte
    m["_Xtr"]       = Xtr
    m["_feat_cols"] = feat_cols
    m["_yte"]       = yte

    print(
        f"    {name:<15}  "
        f"R²={m.get('r2',           np.nan):+.4f}  "
        f"RMSE={m.get('rmse',       np.nan):.4f}  "
        f"MAPE={m.get('mape',       np.nan):.1f}%  "
        f"R²(IRD)={m.get('r2_ird',  np.nan):+.4f}  "
        f"BasMedR²={m.get('basin_med_r2', np.nan):+.4f}"
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_plot_shap(
    condition_results: dict[str, dict],
    condition:         str,
    feat_cols:         list[str],
) -> None:
    """
    Compute and plot SHAP beeswarm for LightGBM on one condition.
    Also prints ranked feature importance table.
    """
    if "LightGBM" not in condition_results:
        print(f"  LightGBM not in condition {condition} results — skipping SHAP")
        return

    res   = condition_results["LightGBM"]
    model = res.get("_model")
    Xte   = res.get("_Xte")

    if model is None or Xte is None:
        print(f"  SHAP skipped — model or test data not available")
        return

    n_features    = Xte.shape[1]
    feature_names = feat_cols[:n_features]

    print(f"  Computing SHAP values — LightGBM, Condition {condition}...")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xte)
    except Exception as e:
        print(f"  ERROR computing SHAP: {e}")
        return

    # Subsample for plot readability
    n_sample = min(2000, len(Xte))
    idx      = np.random.default_rng(RANDOM_SEED).choice(
        len(Xte), n_sample, replace=False
    )
    shap_df = pd.DataFrame(Xte[idx], columns=feature_names)

    plt.figure(figsize=(10, 9))
    shap.summary_plot(
        shap_values[idx], shap_df,
        show=False, plot_size=None,
    )
    plt.title(
        f"SHAP summary — LightGBM — Condition {condition}\n"
        f"Each dot = one flooding event  |  "
        f"Red = high feature value, Blue = low\n"
        f"Positive SHAP = less decay  |  Negative SHAP = more decay",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    # Ranked feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_summary  = pd.DataFrame({
        "feature":        feature_names,
        "mean_abs_shap":  mean_abs_shap,
        "mean_shap":      shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n  SHAP feature importance — LightGBM, Condition {condition}:")
    print(f"  {'Rank':<5}  {'Feature':<25}  "
          f"{'Mean |SHAP|':>12}  {'Mean SHAP':>10}  Direction")
    print(f"  {'-'*65}")
    for i, row in shap_summary.iterrows():
        direction = "→ less decay" if row["mean_shap"] > 0 else "→ more decay"
        print(
            f"  {i+1:<5}  {row['feature']:<25}  "
            f"{row['mean_abs_shap']:>12.4f}  "
            f"{row['mean_shap']:>10.4f}  {direction}"
        )

    return shap_summary


def plot_shap_comparison(
    shap_results: dict[str, pd.DataFrame],
) -> None:
    """
    Side-by-side SHAP mean |SHAP| bar charts for conditions A, D, E.
    Shows whether feature importance ranking is stable across data quality choices.

    Physical interpretation: if SHAP rankings are consistent across A/D/E,
    the model's learned physics is robust regardless of whether we include
    outlier basins and noisy segments in training.
    """
    if len(shap_results) < 2:
        return

    conditions = list(shap_results.keys())
    n_cond     = len(conditions)

    # Union of all features, sorted by condition A importance
    ref_cond   = "A" if "A" in shap_results else conditions[0]
    ref_order  = shap_results[ref_cond]["feature"].tolist()

    fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 8),
                             sharey=True)
    if n_cond == 1:
        axes = [axes]

    colors = {"A": "tomato", "D": "steelblue", "E": "seagreen"}

    for ax, cond in zip(axes, conditions):
        sr    = shap_results[cond].set_index("feature")
        vals  = [sr.loc[f, "mean_abs_shap"]
                 if f in sr.index else 0.0
                 for f in ref_order]
        color = colors.get(cond, "gray")

        bars = ax.barh(
            range(len(ref_order)), vals,
            color=color, alpha=0.8, edgecolor="white",
        )
        ax.set_yticks(range(len(ref_order)))
        ax.set_yticklabels(ref_order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        ax.set_title(
            f"Condition {cond}",
            fontsize=10, fontweight="bold", color=color,
        )
        ax.grid(True, alpha=0.2, axis="x")

        # Value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7,
                )

    fig.suptitle(
        "SHAP feature importance comparison — LightGBM\n"
        "Conditions: A=Clean  D=Held-out  E=All-data held-out\n"
        "Stable rankings across conditions → robust physical interpretation",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_all_conditions(
    all_results: dict[str, dict[str, dict]],
) -> None:
    """
    Multi-panel bar chart: one column per condition, rows = metrics.
    Shows R²(log), R²(IRD), MAPE, Basin med R² for all algorithms × conditions.
    LightGBM bars highlighted in each panel.
    """
    conditions  = list(all_results.keys())
    metrics_cfg = [
        ("r2",          "R² (log-ratio)",   True),
        ("r2_ird",      "R² (raw IRD)",     True),
        ("mape",        "MAPE % (raw IRD)", False),
        ("basin_med_r2","Basin median R²",  True),
    ]

    n_metrics    = len(metrics_cfg)
    n_conditions = len(conditions)

    fig, axes = plt.subplots(
        n_metrics, n_conditions,
        figsize=(4 * n_conditions, 3.5 * n_metrics),
        squeeze=False,
    )

    fig.suptitle(
        "Model comparison — all algorithms × all conditions (test split)\n"
        "LightGBM highlighted as primary model",
        fontsize=12, fontweight="bold",
    )

    for col, condition in enumerate(conditions):
        cond_results = all_results[condition]
        model_names  = list(cond_results.keys())
        x            = np.arange(len(model_names))
        colors = [
            "tomato" if n == "LightGBM" else "steelblue"
            for n in model_names
        ]

        for row, (metric, label, higher_better) in enumerate(metrics_cfg):
            ax   = axes[row][col]
            vals = [cond_results[m].get(metric, np.nan)
                    for m in model_names]

            bars = ax.bar(x, vals, color=colors, alpha=0.85,
                          edgecolor="white")

            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.02,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=6, rotation=90,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=40,
                               ha="right", fontsize=7)
            ax.grid(True, alpha=0.2, axis="y")
            ax.axhline(0, color="black", linewidth=0.5,
                       linestyle="--", alpha=0.4)

            if col == 0:
                arrow = "↑" if higher_better else "↓"
                ax.set_ylabel(f"{label} {arrow}", fontsize=8)
            if row == 0:
                ax.set_title(f"Condition {condition}", fontsize=9,
                             fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_lgbm_condition_summary(
    all_results: dict[str, dict[str, dict]],
) -> None:
    """
    LightGBM only — all metrics across all conditions.
    Clean comparison for the paper: how does the primary model
    perform across data quality choices?
    """
    conditions = list(all_results.keys())
    metrics_cfg = [
        ("r2",          "R² (log-ratio)",   True),
        ("r2_ird",      "R² (raw IRD)",     True),
        ("mape",        "MAPE % (raw IRD)", False),
        ("basin_med_r2","Basin median R²",  True),
    ]

    x      = np.arange(len(conditions))
    colors = ["tomato", "steelblue", "seagreen", "mediumpurple", "orange"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "LightGBM — performance across evaluation conditions (test split)\n"
        "A=Clean  B=All segs  C=All data  D=Held-out  E=All-data held-out",
        fontsize=11, fontweight="bold",
    )

    for ax, (metric, label, higher_better) in zip(axes, metrics_cfg):
        vals = [
            (all_results[c].get("LightGBM") or {}).get(metric, np.nan)
            for c in conditions
        ]
        bars = ax.bar(x, vals,
                      color=colors[:len(conditions)],
                      alpha=0.85, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}" if metric != "mape" else f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Cond. {c}" for c in conditions], fontsize=9)
        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{label}\n({arrow})", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results: dict[str, dict[str, dict]]) -> None:
    """Save all condition × algorithm results to XLSX."""
    rows = []
    for condition, cond_results in all_results.items():
        for model_name, m in cond_results.items():
            rows.append(dict(
                condition    = condition,
                model        = model_name,
                r2           = round(m.get("r2",           np.nan), 4),
                rmse         = round(m.get("rmse",         np.nan), 4),
                mae          = round(m.get("mae",          np.nan), 4),
                mape         = round(m.get("mape",         np.nan), 2),
                r2_ird       = round(m.get("r2_ird",       np.nan), 4),
                rmse_ird     = round(m.get("rmse_ird",     np.nan), 4),
                basin_med_r2 = round(m.get("basin_med_r2", np.nan), 4),
                spearman_r   = round(m.get("spearman_r",   np.nan), 4),
                n_train      = m.get("n_train", np.nan),
                n_test       = m.get("n_test",  np.nan),
            ))

    df_out   = pd.DataFrame(rows)
    out_path = TABLES_DIR / "model_comparison_v2.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Print clean summary table
    print(f"\n{'='*90}")
    print("  MODEL COMPARISON SUMMARY — LightGBM (primary) across all conditions")
    print(f"{'='*90}")
    print(f"  {'Cond':<6}  {'R²(log)':>8}  {'RMSE(log)':>10}  "
          f"{'MAPE%':>7}  {'R²(IRD)':>8}  {'BasMedR²':>9}")
    print(f"  {'-'*55}")
    for condition, cond_results in all_results.items():
        m = cond_results.get("LightGBM", {})
        print(
            f"  {condition:<6}  "
            f"{m.get('r2',           np.nan):>+8.4f}  "
            f"{m.get('rmse',         np.nan):>10.4f}  "
            f"{m.get('mape',         np.nan):>6.1f}%  "
            f"{m.get('r2_ird',       np.nan):>+8.4f}  "
            f"{m.get('basin_med_r2', np.nan):>+9.4f}"
        )

    print(f"\n{'='*90}")
    print("  FULL ALGORITHM COMPARISON — Condition A (clean model, paper result)")
    print(f"{'='*90}")
    cond_a = all_results.get("A", {})
    sorted_models = sorted(
        cond_a.keys(),
        key=lambda n: cond_a[n].get("r2", -np.inf),
        reverse=True,
    )
    print(f"  {'Model':<15}  {'R²(log)':>8}  {'RMSE':>8}  "
          f"{'MAPE%':>7}  {'R²(IRD)':>8}  {'BasMedR²':>9}")
    print(f"  {'-'*65}")
    for name in sorted_models:
        m   = cond_a[name]
        tag = "  ← primary" if name == "LightGBM" else ""
        print(
            f"  {name:<15}  "
            f"{m.get('r2',           np.nan):>+8.4f}  "
            f"{m.get('rmse',         np.nan):>8.4f}  "
            f"{m.get('mape',         np.nan):>7.1f}  "
            f"{m.get('r2_ird',       np.nan):>+8.4f}  "
            f"{m.get('basin_med_r2', np.nan):>+9.4f}"
            f"{tag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  MODEL COMPARISON V2 — analysis/model_comparison.py")
    print("  All conditions × all algorithms + SHAP on A, D, E")
    print("=" * 65)

    # Load full dataset once
    print("\n--- Loading full event dataset ---")
    df_full = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]
    if TARGET_M1 not in df_full.columns and "IRD_norm" in df_full.columns:
        df_full[TARGET_M1] = df_full["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce")
    print(f"  Loaded {len(df_full)} rows  "
          f"{df_full['basin_number'].nunique()} basins")

    # Resolve basin sets
    outlier_basins = load_outlier_basins()
    basin_sets     = resolve_basin_sets(df_full, outlier_basins)

    # Algorithm registry
    models = get_models()
    print(f"\n  Algorithms: {list(models.keys())}")
    print(f"  Conditions: A, B, C, D, E")

    # Run all conditions × all algorithms
    all_results:  dict[str, dict[str, dict]] = {}
    shap_results: dict[str, pd.DataFrame]    = {}

    for condition in ["A", "B", "C", "D", "E"]:
        print(f"\n{'='*65}")
        print(f"  CONDITION {condition}")
        print(f"{'='*65}")

        df_cond, feat_cols = load_condition(df_full, condition, basin_sets)
        if df_cond is None:
            continue

        cond_results: dict[str, dict] = {}

        for name, (model_template, scale) in models.items():
            print(f"\n  [{condition}] {name}")
            fresh = copy.deepcopy(model_template)
            try:
                result = run_one_model(
                    name, fresh, scale, df_cond, feat_cols
                )
                if result is not None:
                    cond_results[name] = result
            except Exception as e:
                print(f"    ERROR: {e}")

        if cond_results:
            all_results[condition] = cond_results

        # SHAP for conditions A, D, E on LightGBM
        if condition in ("A", "D", "E") and "LightGBM" in cond_results:
            print(f"\n--- SHAP analysis — Condition {condition} ---")
            sr = compute_and_plot_shap(cond_results, condition, feat_cols)
            if sr is not None:
                shap_results[condition] = sr

    # Cross-condition SHAP comparison
    if len(shap_results) >= 2:
        print("\n--- SHAP comparison across conditions A, D, E ---")
        plot_shap_comparison(shap_results)

    # Figures
    if all_results:
        print("\n--- Figure: full model comparison ---")
        plot_model_comparison_all_conditions(all_results)

        print("\n--- Figure: LightGBM across conditions ---")
        plot_lgbm_condition_summary(all_results)

        save_results(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()