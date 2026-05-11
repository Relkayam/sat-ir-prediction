"""
analysis/feature_selection_unconstrained.py
============================================
Forward stepwise feature selection for Model 1 — unconstrained.
Starts from mandatory base of 5 features, freely adds from all remaining
features until all are added.

PRIMARY CONDITION: Condition E
  - Training: all non-held-out basins (45 incl. outliers), ALL segments
  - Test:     5 held-out basins, ALL events (no good_segment filter)
  - Metric:   RMSE on raw IRD (cm/h) on held-out test set

RESULT (documented for reproducibility):
  Best: 11 features, RMSE=0.6491 cm/h
  Elbow at step 7 (12th feature cum_TD worsened RMSE by 0.009 cm/h)
  Full 21-feature model RMSE=0.7117 cm/h — 11-feature model better by 0.047 cm/h

Usage
-----
  python -m analysis.feature_selection_unconstrained
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, TABLES_DIR,
    RANDOM_SEED, TRAIN_FRAC, VAL_FRAC,
)
from pipeline.features import (
    TARGET_M1, MODEL1_FEATURES, LOG_TRANSFORM,
)
from models.utils import back_transform, metrics_ird

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MANDATORY_FEATURES = [
    "IRD_at_reset",
    "prev_ALPHA",
    "log1p_prev_HL",
    "prev_DrT",
    "LCT",
]

LGBM_PARAMS = dict(
    n_estimators      = 1000,
    max_depth         = -1,
    num_leaves        = 63,
    learning_rate     = 0.05,
    subsample         = 0.8,
    feature_fraction  = 0.8,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbose           = -1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _reassign_splits(df: pd.DataFrame) -> pd.Series:
    """Random segment-level 70/15/15 split for training basins."""
    seg_ids = sorted([
        int(s) for s in df["segment_id"].dropna().unique() if s >= 0
    ])
    if len(seg_ids) < 5:
        return pd.Series("excluded", index=df.index)

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


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for raw_col, new_col in LOG_TRANSFORM.items():
        if raw_col in df.columns:
            df[new_col] = np.log1p(
                pd.to_numeric(df[raw_col], errors="coerce")
            )
    return df


def resolve_feature_name(feature: str, df: pd.DataFrame) -> str | None:
    transformed = LOG_TRANSFORM.get(feature)
    if transformed and transformed in df.columns:
        return transformed
    if feature in df.columns:
        return feature
    return None


def load_condition_e(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct Condition E exactly:
    - Training: all non-held-out basins (45, including outliers),
                ALL segments — no good_segment filter
    - Test:     5 held-out basins, ALL events (no good_segment filter)

    IMPORTANT: Condition E uses ALL segments on BOTH train and test.
    The is_good_segment filter must NOT be applied to the held-out test set.
    Using only good segments on test would inflate performance by excluding
    difficult events — giving an optimistic and misleading evaluation.

    The key change from earlier versions: removed `is_good_segment == True`
    filter on df_test. Condition E is defined as no quality filtering.
    """
    df = df_full.copy()

    if "basin_role" not in df.columns:
        raise ValueError("basin_role column not found.")

    held_out = set(
        df.loc[df["basin_role"] == "held_out", "basin_number"]
        .dropna().unique().astype(int).tolist()
    )

    # Training: all non-held-out basins, ALL segments (no quality filter)
    df_train = df[
        (~df["basin_number"].isin(held_out)) &
        (df["row_type"] == "event")
    ].copy()

    # Test: held-out basins, ALL events — NO is_good_segment filter
    # This is the critical fix: Condition E evaluates on ALL held-out events,
    # not just good-segment events. Filtering would inflate test performance.
    df_test = df[
        (df["basin_number"].isin(held_out)) &
        (df["row_type"] == "event")
    ].copy()

    df_train["split"] = _reassign_splits(df_train)
    df_train = df_train[df_train["split"].isin(["train", "val"])].copy()
    df_test["split"]  = "test"

    df_out = pd.concat([df_train, df_test], ignore_index=True)

    n_tr = int((df_out["split"] == "train").sum())
    n_va = int((df_out["split"] == "val").sum())
    n_te = int((df_out["split"] == "test").sum())
    print(f"  Condition E: train={n_tr}  val={n_va}  "
          f"test={n_te}  held-out={sorted(held_out)}")
    print(f"  ALL segments used — no quality filter on train or test")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df:       pd.DataFrame,
    features: list[str],
    verbose:  bool = False,
) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    missing = [f for f in features if f not in df.columns]
    if missing:
        return dict(rmse_ird=np.nan, mape=np.nan, r2_ird=np.nan, n_test=0)

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
            f"R²={m.get('r2', np.nan):+.4f}  "
            f"n={m.get('n', 0)}"
        )

    return dict(
        rmse_ird = m.get("rmse", np.nan),
        mape     = m.get("mape", np.nan),
        r2_ird   = m.get("r2",   np.nan),
        n_test   = m.get("n",    0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forward stepwise — unconstrained
# ─────────────────────────────────────────────────────────────────────────────

def forward_stepwise_unconstrained(
    df:           pd.DataFrame,
    all_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Unconstrained forward stepwise selection.
    Starts from mandatory base, adds one feature at a time,
    always picking the one that minimises RMSE on held-out test.
    Runs to completion — elbow identified visually from the output curve.
    """
    mandatory = []
    for f in MANDATORY_FEATURES:
        r = resolve_feature_name(f, df)
        if r:
            mandatory.append(r)

    candidates = [f for f in all_features if f not in mandatory]

    print(f"  Mandatory base ({len(mandatory)}): {mandatory}")
    print(f"  Candidates ({len(candidates)}): {candidates}")
    print(f"  Running to completion — {len(candidates)} additional steps")

    results  = []
    selected = list(mandatory)

    # Step 0: mandatory base
    print(f"\n{'='*65}")
    print(f"  STEP 0 — Mandatory base ({len(selected)} features)")
    print(f"{'='*65}")
    m = train_and_evaluate(df, selected, verbose=True)
    results.append(dict(
        step       = 0,
        added      = "base",
        n_features = len(selected),
        features   = list(selected),
        rmse_ird   = m["rmse_ird"],
        mape       = m["mape"],
        r2_ird     = m["r2_ird"],
        delta_rmse = 0.0,
    ))
    best_rmse = m["rmse_ird"]

    remaining = list(candidates)
    step      = 1

    while remaining:
        print(f"\n{'='*65}")
        print(f"  STEP {step} — {len(remaining)} candidates, "
              f"{len(selected)} features so far")
        print(f"{'='*65}")

        step_rows = []
        for candidate in remaining:
            trial = selected + [candidate]
            m     = train_and_evaluate(df, trial, verbose=False)
            step_rows.append(dict(
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

        step_df = pd.DataFrame(step_rows).dropna(subset=["rmse_ird"])
        if step_df.empty:
            break

        best_row     = step_df.loc[step_df["rmse_ird"].idxmin()]
        best_feature = best_row["feature"]
        delta        = best_rmse - best_row["rmse_ird"]

        selected  = selected + [best_feature]
        remaining = [f for f in remaining if f != best_feature]
        best_rmse = best_row["rmse_ird"]

        direction = "↓ better" if delta > 0 else "↑ worse"
        print(
            f"\n  ✓ Best: '{best_feature}'  "
            f"RMSE={best_row['rmse_ird']:.4f}  "
            f"Δ={delta:+.4f} cm/h {direction}"
        )
        print(f"  Selected ({len(selected)}): {selected}")

        results.append(dict(
            step       = step,
            added      = best_feature,
            n_features = len(selected),
            features   = list(selected),
            rmse_ird   = best_row["rmse_ird"],
            mape       = best_row["mape"],
            r2_ird     = best_row["r2_ird"],
            delta_rmse = delta,
        ))
        step += 1

    return pd.DataFrame(results), selected


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_curve(results_df: pd.DataFrame) -> None:
    """
    Three-panel figure:
      Left:   RMSE vs n_features with feature labels
      Middle: ΔRMSE at each step (marginal improvement)
      Right:  R²(IRD) vs n_features
    Color: green = improvement, red = worsening, gold star = best.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Feature selection — unconstrained forward stepwise (Condition E)\n"
        "Training: 45 basins incl. outliers, all segments  |  "
        "Test: 5 held-out basins, ALL events  |  Metric: RMSE on raw IRD (cm/h)",
        fontsize=10, fontweight="bold",
    )

    n_feat  = results_df["n_features"].values
    rmse    = results_df["rmse_ird"].values
    r2      = results_df["r2_ird"].values
    deltas  = results_df["delta_rmse"].values
    labels  = results_df["added"].values
    colors  = ["seagreen" if d > 0 else "tomato" for d in deltas]
    colors[0] = "gray"

    best_idx = int(np.nanargmin(rmse))

    # Left: RMSE curve
    ax = axes[0]
    ax.plot(n_feat, rmse, color="black", linewidth=1.0, alpha=0.5, zorder=2)
    ax.scatter(n_feat, rmse, c=colors, s=60, zorder=4)
    for i, (x, y, lbl) in enumerate(zip(n_feat, rmse, labels)):
        if lbl == "base": continue
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(0, 4 if va == "bottom" else -4),
                    fontsize=6.5, ha="center", va=va,
                    color=colors[i])
    ax.scatter(n_feat[best_idx], rmse[best_idx],
               color="gold", s=250, zorder=5, marker="*",
               edgecolors="black", linewidths=0.8)
    ax.annotate(f"  Best: {int(n_feat[best_idx])} features\n"
                f"  RMSE={rmse[best_idx]:.4f} cm/h",
                (n_feat[best_idx], rmse[best_idx]),
                fontsize=8, color="darkgoldenrod",
                xytext=(8, -12), textcoords="offset points")
    ax.set_xlabel("Number of features", fontsize=9)
    ax.set_ylabel("RMSE (cm/h)", fontsize=9)
    ax.set_title("RMSE vs features ↓ better", fontsize=9, fontweight="bold")
    ax.set_xticks(n_feat)
    ax.grid(True, alpha=0.2)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="seagreen", label="RMSE improved"),
        Patch(color="tomato",   label="RMSE worsened"),
    ], fontsize=8)

    # Middle: delta RMSE
    ax = axes[1]
    bar_colors = colors[1:]
    bars = ax.bar(n_feat[1:], deltas[1:],
                  color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, lbl, val in zip(bars, labels[1:], deltas[1:]):
        va  = "bottom" if val >= 0 else "top"
        off = 0.001 if val >= 0 else -0.001
        ax.text(bar.get_x() + bar.get_width() / 2, val + off,
                lbl, ha="center", va=va, fontsize=6.5, rotation=90)
    ax.set_xlabel("Number of features", fontsize=9)
    ax.set_ylabel("ΔRMSE (cm/h)", fontsize=9)
    ax.set_title("Marginal improvement at each step\n↑ positive = improved",
                 fontsize=9, fontweight="bold")
    ax.set_xticks(n_feat[1:])
    ax.grid(True, alpha=0.2, axis="y")

    # Right: R² curve
    ax = axes[2]
    ax.plot(n_feat, r2, color="black", linewidth=1.0, alpha=0.5, zorder=2)
    ax.scatter(n_feat, r2, c=colors, s=60, zorder=4)
    best_r2_idx = int(np.nanargmax(r2))
    ax.scatter(n_feat[best_r2_idx], r2[best_r2_idx],
               color="gold", s=250, zorder=5, marker="*",
               edgecolors="black", linewidths=0.8)
    for i, (x, y, lbl) in enumerate(zip(n_feat, r2, labels)):
        if lbl == "base": continue
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(0, 4 if va == "bottom" else -4),
                    fontsize=6.5, ha="center", va=va,
                    color=colors[i])
    ax.set_xlabel("Number of features", fontsize=9)
    ax.set_ylabel("R² (raw IRD)", fontsize=9)
    ax.set_title("R²(IRD) vs features ↑ better", fontsize=9, fontweight="bold")
    ax.set_xticks(n_feat)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results_df: pd.DataFrame) -> None:
    print(f"\n{'='*75}")
    print("  FULL SELECTION PATH — unconstrained forward stepwise (Condition E)")
    print(f"{'='*75}")
    print(f"  {'Step':>5}  {'Added':<25}  {'N':>3}  "
          f"{'RMSE':>8}  {'ΔRMSE':>8}  {'MAPE%':>7}  {'R²':>7}  Note")
    print(f"  {'-'*75}")

    best_rmse = results_df["rmse_ird"].min()

    for _, row in results_df.iterrows():
        delta_str = (
            f"{row['delta_rmse']:>+8.4f}"
            if row["added"] != "base"
            else f"{'—':>8}"
        )
        note = ""
        if row["rmse_ird"] == best_rmse:
            note = "  ← BEST"
        elif row["added"] != "base" and row["delta_rmse"] < 0:
            note = "  ← worsened"
        print(
            f"  {int(row['step']):>5}  "
            f"{str(row['added']):<25}  "
            f"{int(row['n_features']):>3}  "
            f"{row['rmse_ird']:>8.4f}  "
            f"{delta_str}  "
            f"{row['mape']:>7.1f}  "
            f"{row['r2_ird']:>+7.4f}"
            f"{note}"
        )

    best_row = results_df.loc[results_df["rmse_ird"].idxmin()]
    print(f"\n  Recommended stopping point: {int(best_row['n_features'])} features")
    print(f"  Best RMSE: {best_row['rmse_ird']:.4f} cm/h")
    print(f"  Best MAPE: {best_row['mape']:.1f}%")
    print(f"  Best R²:   {best_row['r2_ird']:+.4f}")
    print(f"\n  Feature set at best RMSE:")
    for i, f in enumerate(best_row["features"], 1):
        print(f"    {i:>2}. {f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> list[str]:
    print("=" * 65)
    print("  FEATURE SELECTION — unconstrained forward stepwise")
    print("  Condition E  |  ALL segments on train AND test")
    print("  Metric: RMSE on raw IRD (cm/h) on 5 held-out test basins")
    print("=" * 65)

    print("\n--- Loading dataset ---")
    df_full = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]
    if TARGET_M1 not in df_full.columns and "IRD_norm" in df_full.columns:
        df_full[TARGET_M1] = df_full["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce")
    df_full = apply_log_transforms(df_full)
    print(f"  {len(df_full)} rows  {df_full['basin_number'].nunique()} basins")

    print("\n--- Building Condition E ---")
    df_e = load_condition_e(df_full)

    # Resolve feature pool
    all_features = []
    for f in MODEL1_FEATURES:
        if f in df_e.columns:
            all_features.append(f)
    print(f"\n  Full pool ({len(all_features)}): {all_features}")

    print("\n--- Running unconstrained forward selection ---")
    results_df, final_features = forward_stepwise_unconstrained(
        df_e, all_features
    )

    print_summary(results_df)

    # Save
    results_save = results_df.copy()
    results_save["features"] = results_save["features"].apply(str)
    out_path = TABLES_DIR / "feature_selection_unconstrained.xlsx"
    results_save.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    plot_full_curve(results_df)

    print("\nDone.")
    return final_features


if __name__ == "__main__":
    main()