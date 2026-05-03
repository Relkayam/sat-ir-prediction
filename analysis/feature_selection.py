"""
analysis/feature_selection.py — Forward stepwise feature selection for Model 1
===============================================================================
Finds the best 10-feature subset using forward stepwise selection,
evaluated on condition E (all-data held-out test, 5 unseen basins).

Strategy
--------
  Step 0: Start with 5 mandatory features (top SHAP across A/D/E)
  Steps 1-5: At each step, try adding each remaining candidate feature
             one at a time, pick the one that minimises RMSE on the
             condition E held-out test set, add it permanently.
  Result: ordered list of 10 features + performance curve

Mandatory base (5 features)
----------------------------
  IRD_at_reset   — basin/segment scale anchor
  prev_ALPHA     — drying fraction (primary recovery driver)
  log1p_prev_HL  — hydraulic load (log-transformed)
  prev_DrT       — drying time
  LCT            — time since reset (primary decay axis)

Candidate pool (16 remaining features)
---------------------------------------
  All other MODEL1_FEATURES not in the mandatory base.
  cum_RD and cum_RW are seeded as the first two additions
  (physically motivated: cumulative photodegradation and
  biofilm activity since reset). The search then picks the
  remaining 3 freely from the full candidate pool.

Evaluation
----------
  Condition E: all 45 non-held-out basins for training
               (including outlier basins, all segments),
               5 held-out basins for test only.
  Metric: RMSE on raw IRD (cm/h) on held-out test events.
  Model: LightGBM with well-established defaults.

Output
------
  Prints performance at each step.
  Saves feature_selection_results.xlsx to outputs/tables/
  Shows performance curve (plt.show())

Usage
-----
  python -m analysis.feature_selection
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, OUTLIER_CSV, TABLES_DIR,
    RANDOM_SEED, TRAIN_FRAC, VAL_FRAC,
)
from pipeline.features import (
    prepare_features, TARGET_M1, MODEL1_FEATURES,
    LOG_TRANSFORM,
)
from models.utils import back_transform, metrics_ird


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Mandatory base — locked in, never varied
MANDATORY_FEATURES = [
    "IRD_at_reset",
    "prev_ALPHA",
    "log1p_prev_HL",
    "prev_DrT",
    "LCT",
]

# Seeded additions — added first before free search
# Physically motivated: cumulative radiation drives photodegradation (dry)
# and biofilm activity (wet) since reset
SEEDED_FEATURES = [
    "cum_RD",   # cumulative radiation during drying since reset
    "cum_RW",   # cumulative radiation during wetting since reset
]

# Target total features
TARGET_N_FEATURES = 10

# LightGBM hyperparameters — well-established defaults
LGBM_PARAMS = dict(
    n_estimators     = 1000,
    max_depth        = -1,
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
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and condition E preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_outlier_basins() -> set[int]:
    if not OUTLIER_CSV.exists():
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
    return excluded


def _reassign_splits(df: pd.DataFrame) -> pd.Series:
    """Random 70/15/15 splits by segment for non-held-out basins."""
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


def load_condition_e(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare condition E dataset:
      - Train: all 45 non-held-out basins, ALL segments (no quality filter)
      - Test:  5 held-out basins, good segments only
    Returns df with 'split' column set.
    """
    df = df_full.copy()

    # Identify held-out basins from basin_role
    if "basin_role" not in df.columns:
        raise ValueError(
            "basin_role column not found — rebuild from V2 build_dataset.py"
        )

    held_out = set(
        df.loc[df["basin_role"] == "held_out", "basin_number"]
        .dropna().unique().astype(int).tolist()
    )

    # Training set: all non-held-out basins, all event rows
    train_mask = (
        (~df["basin_number"].isin(held_out)) &
        (df["row_type"] == "event")
    )
    # Test set: held-out basins, good segments only
    test_mask = (
        (df["basin_number"].isin(held_out)) &
        (df["row_type"]        == "event") &
        (df["is_good_segment"] == True)
    )

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    # Assign splits to training basins
    df_train["split"] = _reassign_splits(df_train)
    df_train = df_train[
        df_train["split"].isin(["train", "val"])
    ].copy()

    # All held-out events are test
    df_test["split"] = "test"

    df_out = pd.concat([df_train, df_test], ignore_index=True)

    n_tr = int((df_out["split"] == "train").sum())
    n_va = int((df_out["split"] == "val").sum())
    n_te = int((df_out["split"] == "test").sum())

    print(f"  Condition E: train={n_tr}  val={n_va}  "
          f"test={n_te} (held-out basins)  "
          f"total={len(df_out)}")
    print(f"  Held-out basins: {sorted(held_out)}")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation — apply log transforms to any HL feature present
# ─────────────────────────────────────────────────────────────────────────────

def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transforms defined in LOG_TRANSFORM."""
    df = df.copy()
    for raw_col, new_col in LOG_TRANSFORM.items():
        if raw_col in df.columns:
            df[new_col] = np.log1p(
                pd.to_numeric(df[raw_col], errors="coerce")
            )
    return df


def resolve_feature_name(feature: str, df: pd.DataFrame) -> str | None:
    """
    Resolve a feature name — if it needs a log transform, return
    the transformed name. Returns None if feature not in df.
    """
    # Check if this raw feature has a log-transformed version
    transformed = LOG_TRANSFORM.get(feature)
    if transformed and transformed in df.columns:
        return transformed
    if feature in df.columns:
        return feature
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Model training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df:        pd.DataFrame,
    features:  list[str],
    verbose:   bool = False,
) -> dict:
    """
    Train LightGBM on train split, evaluate RMSE on held-out test events.

    Parameters
    ----------
    df       : full condition E DataFrame with 'split' column
    features : list of feature column names to use
    verbose  : print training progress

    Returns
    -------
    dict with rmse_ird, mape, r2_ird, n_test
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    # Verify all features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    train = df[df["split"] == "train"].dropna(
        subset=features + [TARGET_M1]
    ).reset_index(drop=True)
    val   = df[df["split"] == "val"].dropna(
        subset=features + [TARGET_M1]
    ).reset_index(drop=True)
    test  = df[df["split"] == "test"].dropna(
        subset=features + [TARGET_M1]
    ).reset_index(drop=True)

    if len(train) < 50 or len(val) < 10 or len(test) < 10:
        return dict(rmse_ird=np.nan, mape=np.nan, r2_ird=np.nan, n_test=0)

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[features].values)
    Xva = sc.transform(val[features].values)
    Xte = sc.transform(test[features].values)

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        Xtr, train[TARGET_M1].values,
        eval_set=[(Xva, val[TARGET_M1].values)],
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    y_pred = model.predict(Xte)
    y_true = test[TARGET_M1].values

    # Back-transform to raw IRD (cm/h)
    ird_reset = pd.to_numeric(
        test["IRD_at_reset"], errors="coerce"
    ).values.ravel()
    ird_pred = back_transform(ird_reset, y_pred.ravel())
    ird_true = back_transform(ird_reset, y_true.ravel())

    m = metrics_ird(ird_true, ird_pred, verbose=False) or {}

    if verbose:
        print(
            f"    RMSE={m.get('rmse', np.nan):.4f} cm/h  "
            f"MAPE={m.get('mape', np.nan):.1f}%  "
            f"R²(IRD)={m.get('r2', np.nan):+.4f}  "
            f"n_test={m.get('n', 0)}"
        )

    return dict(
        rmse_ird = m.get("rmse", np.nan),
        mape     = m.get("mape", np.nan),
        r2_ird   = m.get("r2",   np.nan),
        n_test   = m.get("n",    0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forward stepwise selection
# ─────────────────────────────────────────────────────────────────────────────

def forward_stepwise_selection(
    df:         pd.DataFrame,
    all_features: list[str],
) -> pd.DataFrame:
    """
    Forward stepwise feature selection.

    Phase 0: Evaluate mandatory base (5 features)
    Phase 1: Add seeded features (cum_RD, cum_RW) — physically motivated
    Phase 2: Free search — add best remaining feature at each step
             until TARGET_N_FEATURES is reached

    At each step, all remaining candidate features are tried.
    The one that minimises RMSE on held-out test is added permanently.

    Returns DataFrame with one row per step showing selected feature
    and resulting performance.
    """
    # Build full candidate pool from MODEL1_FEATURES
    # Resolve any log-transformed names
    resolved_all = []
    for f in all_features:
        resolved = resolve_feature_name(f, df)
        if resolved and resolved not in resolved_all:
            resolved_all.append(resolved)

    # Resolve mandatory features
    mandatory = []
    for f in MANDATORY_FEATURES:
        resolved = resolve_feature_name(f, df)
        if resolved:
            mandatory.append(resolved)
        else:
            print(f"  WARNING: mandatory feature '{f}' not found in data")

    # Resolve seeded features
    seeded = []
    for f in SEEDED_FEATURES:
        resolved = resolve_feature_name(f, df)
        if resolved:
            seeded.append(resolved)
        else:
            print(f"  WARNING: seeded feature '{f}' not found in data")

    # Candidate pool = all features minus mandatory and seeded
    candidates = [
        f for f in resolved_all
        if f not in mandatory and f not in seeded
    ]

    print(f"\n  Mandatory base ({len(mandatory)}): {mandatory}")
    print(f"  Seeded additions ({len(seeded)}): {seeded}")
    print(f"  Free candidates ({len(candidates)}): {candidates}")
    print(f"  Target: {TARGET_N_FEATURES} features total")

    results = []
    selected = list(mandatory)

    # ── Phase 0: evaluate mandatory base ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 0 — Mandatory base ({len(selected)} features)")
    print(f"{'='*65}")
    print(f"  Features: {selected}")

    m = train_and_evaluate(df, selected, verbose=True)
    results.append(dict(
        step        = 0,
        phase       = "mandatory_base",
        added       = "base",
        n_features  = len(selected),
        features    = str(selected),
        rmse_ird    = m["rmse_ird"],
        mape        = m["mape"],
        r2_ird      = m["r2_ird"],
        n_test      = m["n_test"],
    ))

    # ── Phase 1: add seeded features ─────────────────────────────────────────
    for seed_feat in seeded:
        if len(selected) >= TARGET_N_FEATURES:
            break

        selected = selected + [seed_feat]
        step_num = len(selected) - len(mandatory)

        print(f"\n{'='*65}")
        print(f"  STEP {step_num} — Seeded addition: '{seed_feat}' "
              f"({len(selected)} features total)")
        print(f"{'='*65}")
        print(f"  Features: {selected}")

        m = train_and_evaluate(df, selected, verbose=True)
        results.append(dict(
            step        = step_num,
            phase       = "seeded",
            added       = seed_feat,
            n_features  = len(selected),
            features    = str(selected),
            rmse_ird    = m["rmse_ird"],
            mape        = m["mape"],
            r2_ird      = m["r2_ird"],
            n_test      = m["n_test"],
        ))

    # ── Phase 2: free search ──────────────────────────────────────────────────
    remaining = [f for f in candidates if f not in selected]

    while len(selected) < TARGET_N_FEATURES and remaining:
        step_num = len(selected) - len(mandatory) + 1
        print(f"\n{'='*65}")
        print(f"  STEP {step_num} — Free search "
              f"({len(remaining)} candidates, "
              f"{len(selected)} features so far)")
        print(f"{'='*65}")

        step_results = []
        for candidate in remaining:
            trial_features = selected + [candidate]
            m = train_and_evaluate(df, trial_features, verbose=False)
            step_results.append(dict(
                feature  = candidate,
                rmse_ird = m["rmse_ird"],
                mape     = m["mape"],
                r2_ird   = m["r2_ird"],
            ))
            print(
                f"    + {candidate:<25}  "
                f"RMSE={m['rmse_ird']:.4f} cm/h  "
                f"MAPE={m['mape']:.1f}%  "
                f"R²={m['r2_ird']:+.4f}"
            )

        # Pick best by RMSE
        step_df  = pd.DataFrame(step_results).dropna(subset=["rmse_ird"])
        if step_df.empty:
            print("  No valid candidates — stopping")
            break

        best_row     = step_df.loc[step_df["rmse_ird"].idxmin()]
        best_feature = best_row["feature"]
        selected     = selected + [best_feature]
        remaining    = [f for f in remaining if f != best_feature]

        print(f"\n  ✓ Best addition: '{best_feature}'  "
              f"RMSE={best_row['rmse_ird']:.4f} cm/h")
        print(f"  Selected so far ({len(selected)}): {selected}")

        results.append(dict(
            step        = step_num,
            phase       = "free_search",
            added       = best_feature,
            n_features  = len(selected),
            features    = str(selected),
            rmse_ird    = best_row["rmse_ird"],
            mape        = best_row["mape"],
            r2_ird      = best_row["r2_ird"],
            n_test      = m["n_test"],
        ))

    return pd.DataFrame(results), selected


# ─────────────────────────────────────────────────────────────────────────────
# Plot performance curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_selection_curve(results_df: pd.DataFrame) -> None:
    """
    Two-panel plot:
    Left:  RMSE vs number of features
    Right: R²(IRD) vs number of features
    Phase boundaries marked (mandatory / seeded / free search).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Forward stepwise feature selection — Condition E (held-out test)\n"
        "LightGBM  |  Metric: RMSE on raw IRD (cm/h)  |  "
        "5 held-out basins never seen during training",
        fontsize=10, fontweight="bold",
    )

    colors = {
        "mandatory_base": "gray",
        "seeded":         "steelblue",
        "free_search":    "tomato",
    }
    labels = {
        "mandatory_base": "Mandatory base",
        "seeded":         "Seeded (cum_RD, cum_RW)",
        "free_search":    "Free search",
    }

    for ax, metric, ylabel, title in [
        (axes[0], "rmse_ird", "RMSE (cm/h)",  "RMSE on held-out test ↓ better"),
        (axes[1], "r2_ird",   "R² (IRD)",     "R²(IRD) on held-out test ↑ better"),
    ]:
        seen_phases = set()
        for _, row in results_df.iterrows():
            phase  = row["phase"]
            color  = colors.get(phase, "black")
            label  = labels.get(phase) if phase not in seen_phases else None
            seen_phases.add(phase)

            ax.scatter(
                row["n_features"], row[metric],
                color=color, s=80, zorder=4,
                label=label,
            )
            ax.annotate(
                row["added"] if row["added"] != "base" else "base",
                (row["n_features"], row[metric]),
                textcoords="offset points",
                xytext=(5, 3), fontsize=7, color=color,
            )

        # Connect points with line
        ax.plot(
            results_df["n_features"],
            results_df[metric],
            color="black", linewidth=0.8, alpha=0.4, zorder=3,
        )

        # Mark best point
        if metric == "rmse_ird":
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()

        best_row = results_df.loc[best_idx]
        ax.scatter(
            best_row["n_features"], best_row[metric],
            color="gold", s=200, zorder=5,
            marker="*", edgecolors="black", linewidths=0.8,
            label=f"Best ({int(best_row['n_features'])} features)",
        )

        ax.set_xlabel("Number of features", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(results_df["n_features"].tolist())

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> list[str]:
    """
    Run forward stepwise feature selection on condition E.
    Returns the final selected feature list.
    """
    print("=" * 65)
    print("  FEATURE SELECTION — analysis/feature_selection.py")
    print("  Strategy: Forward stepwise, Condition E, Metric: RMSE(IRD)")
    print("=" * 65)

    # Load full dataset
    print("\n--- Loading full event dataset ---")
    df_full = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]
    if TARGET_M1 not in df_full.columns and "IRD_norm" in df_full.columns:
        df_full[TARGET_M1] = df_full["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce")

    # Apply log transforms
    df_full = apply_log_transforms(df_full)

    print(f"  Loaded {len(df_full)} rows  "
          f"{df_full['basin_number'].nunique()} basins")

    # Build condition E
    print("\n--- Building condition E dataset ---")
    df_e = load_condition_e(df_full)

    # Resolve full candidate feature list from MODEL1_FEATURES
    all_features = []
    for f in MODEL1_FEATURES:
        if f in df_e.columns:
            all_features.append(f)
    print(f"\n  Full feature pool ({len(all_features)}): {all_features}")

    # Run forward stepwise selection
    print("\n--- Forward stepwise selection ---")
    results_df, final_features = forward_stepwise_selection(df_e, all_features)

    # Summary
    print(f"\n{'='*65}")
    print(f"  SELECTION COMPLETE")
    print(f"{'='*65}")
    print(f"\n  Final feature set ({len(final_features)}):")
    for i, f in enumerate(final_features, 1):
        print(f"    {i:>2}. {f}")

    best_row = results_df.loc[results_df["rmse_ird"].idxmin()]
    print(f"\n  Best performance:")
    print(f"    At {int(best_row['n_features'])} features:")
    print(f"    RMSE = {best_row['rmse_ird']:.4f} cm/h")
    print(f"    MAPE = {best_row['mape']:.1f}%")
    print(f"    R²   = {best_row['r2_ird']:+.4f}")

    # Baseline: all 21 features
    print(f"\n--- Baseline: all {len(all_features)} features ---")
    m_full = train_and_evaluate(df_e, all_features, verbose=True)
    print(
        f"  Full model ({len(all_features)} features): "
        f"RMSE={m_full['rmse_ird']:.4f} cm/h  "
        f"MAPE={m_full['mape']:.1f}%  "
        f"R²={m_full['r2_ird']:+.4f}"
    )

    delta_rmse = m_full["rmse_ird"] - best_row["rmse_ird"]
    print(f"\n  ΔRMSE (full vs best subset): {delta_rmse:+.4f} cm/h")
    if delta_rmse > 0:
        print(f"  → Reduced feature set is BETTER than full model by "
              f"{delta_rmse:.4f} cm/h RMSE")
    else:
        print(f"  → Full model is better by {-delta_rmse:.4f} cm/h RMSE")

    # Save results
    out_path = TABLES_DIR / "feature_selection_results.xlsx"
    results_df.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Plot
    plot_selection_curve(results_df)

    print("\nDone.")
    print(f"\n  → Update MODEL1_FEATURES in pipeline/features.py with:")
    print(f"    {final_features}")

    return final_features


if __name__ == "__main__":
    main()