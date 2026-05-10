"""
analysis/feature_selection_m2.py
=================================
Forward stepwise feature selection for Model 2 (IRD_norm_log_reset).
Pure unconstrained search — no mandatory base, no seeding.
Runs to completion across all features.

Evaluation: chronological split (split_chrono)
Metric: RMSE on raw IRD (cm/h) after back-transform on test split.
Model: LightGBM with well-established defaults.

Output
------
  Prints performance at each step with delta.
  Saves feature_selection_m2.xlsx to outputs/tables/
  Shows 3-panel performance curve (plt.show())

Usage
-----
  python -m analysis.feature_selection_m2
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

from config import RESET_CSV, TABLES_DIR, RANDOM_SEED
from pipeline.features import MODEL2_FEATURES, TARGET_M2

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LGBM_PARAMS = dict(
    n_estimators      = 500,
    max_depth         = -1,
    num_leaves        = 31,
    learning_rate     = 0.05,
    subsample         = 0.8,
    feature_fraction  = 0.8,
    min_child_samples = 10,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbose           = -1,
)

SPLIT_COL = "split_chrono"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load reset_dataset.csv and resolve available features."""
    if not RESET_CSV.exists():
        raise FileNotFoundError(
            f"{RESET_CSV} not found.\n"
            "Run: python -m pipeline.build_reset_dataset"
        )

    df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    avail = [f for f in MODEL2_FEATURES if f in df.columns]
    missing = [f for f in MODEL2_FEATURES if f not in df.columns]

    print(f"  Reset dataset : {len(df)} rows  "
          f"{df['basin_number'].nunique()} basins")
    print(f"  Split column  : {SPLIT_COL}")
    print(f"  Features      : {len(avail)} available")
    if missing:
        print(f"  Missing       : {missing}")

    # Split counts
    for split in ["train", "val", "test"]:
        n = int((df[SPLIT_COL] == split).sum())
        print(f"    {split}: {n} resets")

    return df, avail


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df:       pd.DataFrame,
    features: list[str],
    verbose:  bool = False,
) -> dict:
    """
    Train LightGBM on chrono train split.
    Evaluate RMSE on raw IRD (cm/h) after back-transform on test split.
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    required = features + [TARGET_M2, "prev_IRD_at_reset_raw", "IRD_at_reset"]
    train = df[df[SPLIT_COL] == "train"].dropna(
        subset=required
    ).reset_index(drop=True)
    val   = df[df[SPLIT_COL] == "val"].dropna(
        subset=required
    ).reset_index(drop=True)
    test  = df[df[SPLIT_COL] == "test"].dropna(
        subset=required
    ).reset_index(drop=True)

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        return dict(rmse_ird=np.nan, mape=np.nan, r2_ird=np.nan, n_test=0)

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[features].values)
    Xva = sc.transform(val[features].values)
    Xte = sc.transform(test[features].values)

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        Xtr, train[TARGET_M2].values,
        eval_set=[(Xva, val[TARGET_M2].values)],
        callbacks=[
            early_stopping(30, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    # Back-transform predictions to raw IRD
    y_pred_log = model.predict(Xte)
    prev_ird   = test["prev_IRD_at_reset_raw"].values.astype(float)
    ird_pred   = prev_ird * np.exp(y_pred_log)
    ird_true   = test["IRD_at_reset"].values.astype(float)

    mask = (
        np.isfinite(ird_true) & np.isfinite(ird_pred) &
        (ird_true > 0) & (ird_pred > 0)
    )

    if mask.sum() < 5:
        return dict(rmse_ird=np.nan, mape=np.nan, r2_ird=np.nan, n_test=0)

    from sklearn.metrics import r2_score
    rmse = float(np.sqrt(mean_squared_error(ird_true[mask], ird_pred[mask])))
    mape = float(np.mean(
        np.abs((ird_true[mask] - ird_pred[mask]) / ird_true[mask])
    ) * 100)
    r2   = float(r2_score(ird_true[mask], ird_pred[mask]))

    if verbose:
        print(
            f"    RMSE={rmse:.4f} cm/h  "
            f"MAPE={mape:.1f}%  "
            f"R²={r2:+.4f}  "
            f"n={int(mask.sum())}"
        )

    return dict(
        rmse_ird = rmse,
        mape     = mape,
        r2_ird   = r2,
        n_test   = int(mask.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure forward stepwise — no constraints
# ─────────────────────────────────────────────────────────────────────────────

def forward_stepwise(
    df:           pd.DataFrame,
    all_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Unconstrained forward stepwise selection.
    At each step: try adding each remaining feature, pick the one
    that minimises RMSE on chrono test split.
    Runs until all features are added.
    """
    results   = []
    selected  = []
    remaining = list(all_features)

    print(f"  Full candidate pool ({len(all_features)}): {all_features}")
    print(f"  Will run to completion — {len(all_features)} steps")

    # Step 0: empty model baseline — naive predicts log-ratio = 0
    # (IRD_at_reset[i] = IRD_at_reset[i-1])
    test_df = df[df[SPLIT_COL] == "test"].dropna(
        subset=["IRD_at_reset", "prev_IRD_at_reset_raw"]
    )
    ird_true  = test_df["IRD_at_reset"].values.astype(float)
    ird_naive = test_df["prev_IRD_at_reset_raw"].values.astype(float)
    mask0     = (np.isfinite(ird_true) & np.isfinite(ird_naive) &
                 (ird_true > 0) & (ird_naive > 0))
    rmse_naive = float(np.sqrt(mean_squared_error(
        ird_true[mask0], ird_naive[mask0]
    )))

    print(f"\n{'='*65}")
    print(f"  STEP 0 — Naive baseline (predict no change)")
    print(f"{'='*65}")
    print(f"    RMSE={rmse_naive:.4f} cm/h  (naive: IRD_at_reset[i] = IRD_at_reset[i-1])")

    results.append(dict(
        step       = 0,
        added      = "naive_baseline",
        n_features = 0,
        features   = [],
        rmse_ird   = rmse_naive,
        mape       = np.nan,
        r2_ird     = np.nan,
        delta_rmse = 0.0,
    ))
    best_rmse = rmse_naive

    # Steps 1..N: free search
    step = 1
    while remaining:
        print(f"\n{'='*65}")
        print(f"  STEP {step} — "
              f"{len(remaining)} candidates  "
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
                f"    + {candidate:<30}  "
                f"RMSE={m['rmse_ird']:.4f} cm/h  "
                f"MAPE={m['mape']:.1f}%  "
                f"R²={m['r2_ird']:+.4f}"
            )

        step_df  = pd.DataFrame(step_rows).dropna(subset=["rmse_ird"])
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
# Print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results_df: pd.DataFrame) -> None:
    """Print step-by-step summary with delta annotations."""
    print(f"\n{'='*80}")
    print("  FULL SELECTION PATH — Model 2 unconstrained forward stepwise")
    print(f"{'='*80}")
    print(f"  {'Step':>5}  {'Added':<32}  {'N':>3}  "
          f"{'RMSE':>8}  {'ΔRMSE':>8}  {'MAPE%':>7}  "
          f"{'R²':>7}  Note")
    print(f"  {'-'*85}")

    best_rmse = results_df["rmse_ird"].min()

    for _, row in results_df.iterrows():
        delta_str = (
            f"{row['delta_rmse']:>+8.4f}"
            if row["added"] != "naive_baseline"
            else f"{'—':>8}"
        )
        note = ""
        if row["rmse_ird"] == best_rmse:
            note = "  ← BEST"
        elif row["added"] != "naive_baseline" and row["delta_rmse"] < 0:
            note = "  ← worsened"

        mape_str = f"{row['mape']:>7.1f}" if np.isfinite(row.get("mape", np.nan)) else f"{'—':>7}"
        r2_str   = f"{row['r2_ird']:>+7.4f}" if np.isfinite(row.get("r2_ird", np.nan)) else f"{'—':>7}"

        print(
            f"  {int(row['step']):>5}  "
            f"{str(row['added']):<32}  "
            f"{int(row['n_features']):>3}  "
            f"{row['rmse_ird']:>8.4f}  "
            f"{delta_str}  "
            f"{mape_str}  "
            f"{r2_str}"
            f"{note}"
        )

    best_row = results_df.loc[results_df["rmse_ird"].idxmin()]
    print(f"\n  Naive baseline RMSE : {results_df.iloc[0]['rmse_ird']:.4f} cm/h")
    print(f"  Best model RMSE     : {best_row['rmse_ird']:.4f} cm/h")
    print(f"  Improvement vs naive: "
          f"{results_df.iloc[0]['rmse_ird'] - best_row['rmse_ird']:+.4f} cm/h")
    print(f"\n  Recommended stopping point: {int(best_row['n_features'])} features")
    print(f"  Best RMSE  : {best_row['rmse_ird']:.4f} cm/h")
    print(f"  Best MAPE  : {best_row['mape']:.1f}%")
    print(f"  Best R²    : {best_row['r2_ird']:+.4f}")
    print(f"\n  Feature set at best RMSE:")
    for i, f in enumerate(best_row["features"], 1):
        print(f"    {i:>2}. {f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_curve(results_df: pd.DataFrame) -> None:
    """
    Three-panel figure identical to Model 1 analysis:
    Left:   RMSE vs n_features with feature labels
    Middle: ΔRMSE at each step (marginal improvement)
    Right:  R²(IRD) vs n_features

    Step 0 = naive baseline shown as horizontal reference line.
    Gold star = global minimum RMSE.
    Green points = improvement, Red points = worsening.
    """
    # Exclude step 0 (naive baseline) from the curve
    # but show it as a reference line
    naive_rmse = float(results_df.iloc[0]["rmse_ird"])
    curve_df   = results_df.iloc[1:].copy().reset_index(drop=True)

    n_feat  = curve_df["n_features"].values
    rmse    = curve_df["rmse_ird"].values
    r2      = curve_df["r2_ird"].values
    deltas  = curve_df["delta_rmse"].values
    labels  = curve_df["added"].values
    colors  = ["seagreen" if d > 0 else "tomato" for d in deltas]

    best_idx = int(np.nanargmin(rmse))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Model 2 — Feature selection: unconstrained forward stepwise\n"
        "Chronological split  |  Metric: RMSE on raw IRD (cm/h) after back-transform\n"
        "Dashed line = naive baseline (predict no change from previous reset)",
        fontsize=10, fontweight="bold",
    )

    # ── Left: RMSE curve ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(
        naive_rmse, color="gray", linewidth=1.2,
        linestyle="--", alpha=0.7, label=f"Naive={naive_rmse:.4f}",
    )
    ax.plot(n_feat, rmse, color="black",
            linewidth=0.8, alpha=0.5, zorder=2)
    ax.scatter(n_feat, rmse, c=colors, s=60, zorder=4)

    for i, (x, y, lbl) in enumerate(zip(n_feat, rmse, labels)):
        va     = "bottom" if i % 2 == 0 else "top"
        offset = 4 if va == "bottom" else -4
        ax.annotate(
            lbl, (x, y),
            textcoords="offset points",
            xytext=(0, offset),
            fontsize=6, ha="center", va=va,
            color=colors[i],
        )

    ax.scatter(
        n_feat[best_idx], rmse[best_idx],
        color="gold", s=250, zorder=5,
        marker="*", edgecolors="black", linewidths=0.8,
    )
    ax.annotate(
        f"  Best: {int(n_feat[best_idx])} features\n"
        f"  RMSE={rmse[best_idx]:.4f}",
        (n_feat[best_idx], rmse[best_idx]),
        fontsize=8, color="darkgoldenrod",
        xytext=(8, -12), textcoords="offset points",
    )

    ax.set_xlabel("Number of features", fontsize=9)
    ax.set_ylabel("RMSE (cm/h)",        fontsize=9)
    ax.set_title("RMSE vs features ↓ better", fontsize=9, fontweight="bold")
    ax.set_xticks(n_feat)
    ax.grid(True, alpha=0.2)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="seagreen", label="Improved"),
        Patch(color="tomato",   label="Worsened"),
        plt.scatter([], [], marker="*", color="gold",
                    edgecolors="black", s=150, label="Best"),
        plt.plot([], [], color="gray", linestyle="--",
                 label=f"Naive={naive_rmse:.4f}")[0],
    ], fontsize=7)

    # ── Middle: delta RMSE ───────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(
        n_feat, deltas,
        color=colors, alpha=0.85, edgecolor="white",
    )
    ax.axhline(0, color="black", linewidth=0.8,
               linestyle="--", alpha=0.5)

    for bar, lbl, val in zip(bars, labels, deltas):
        va  = "bottom" if val >= 0 else "top"
        off = 0.0005 if val >= 0 else -0.0005
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + off,
            lbl, ha="center", va=va,
            fontsize=5.5, rotation=90,
        )

    ax.set_xlabel("Number of features", fontsize=9)
    ax.set_ylabel("ΔRMSE (cm/h)",       fontsize=9)
    ax.set_title("Marginal improvement\n↑ positive = RMSE improved",
                 fontsize=9, fontweight="bold")
    ax.set_xticks(n_feat)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Right: R² curve ───────────────────────────────────────────────────────
    ax = axes[2]
    valid_r2 = np.isfinite(r2)
    ax.plot(n_feat[valid_r2], r2[valid_r2],
            color="black", linewidth=0.8, alpha=0.5, zorder=2)
    ax.scatter(n_feat[valid_r2], r2[valid_r2],
               c=[colors[i] for i in np.where(valid_r2)[0]],
               s=60, zorder=4)

    if valid_r2.any():
        best_r2_idx = int(np.nanargmax(r2))
        ax.scatter(
            n_feat[best_r2_idx], r2[best_r2_idx],
            color="gold", s=250, zorder=5,
            marker="*", edgecolors="black", linewidths=0.8,
        )

    for i, (x, y_val, lbl) in enumerate(zip(n_feat, r2, labels)):
        if not np.isfinite(y_val):
            continue
        va     = "bottom" if i % 2 == 0 else "top"
        offset = 4 if va == "bottom" else -4
        ax.annotate(
            lbl, (x, y_val),
            textcoords="offset points",
            xytext=(0, offset),
            fontsize=6, ha="center", va=va,
            color=colors[i],
        )

    ax.set_xlabel("Number of features", fontsize=9)
    ax.set_ylabel("R² (raw IRD)",       fontsize=9)
    ax.set_title("R²(IRD) vs features ↑ better",
                 fontsize=9, fontweight="bold")
    ax.set_xticks(n_feat)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> list[str]:
    print("=" * 65)
    print("  MODEL 2 FEATURE SELECTION — unconstrained forward stepwise")
    print(f"  Split: {SPLIT_COL}  |  Metric: RMSE on raw IRD (cm/h)")
    print("  Runs to completion — all features evaluated")
    print("=" * 65)

    print("\n--- Loading reset dataset ---")
    df, all_features = load_data()

    print(f"\n--- Running unconstrained forward selection ---")
    print(f"  {len(all_features)} total features to evaluate")
    results_df, final_features = forward_stepwise(df, all_features)

    print_summary(results_df)

    # Also run baseline with ALL features for comparison
    print(f"\n--- Baseline: all {len(all_features)} features ---")
    m_full = train_and_evaluate(df, all_features, verbose=True)
    print(
        f"  Full model ({len(all_features)} features): "
        f"RMSE={m_full['rmse_ird']:.4f} cm/h  "
        f"MAPE={m_full['mape']:.1f}%  "
        f"R²={m_full['r2_ird']:+.4f}"
    )

    best_row  = results_df.loc[results_df["rmse_ird"].idxmin()]
    delta_vs_full = m_full["rmse_ird"] - best_row["rmse_ird"]
    print(f"\n  ΔRMSE (full vs best subset): {delta_vs_full:+.4f} cm/h")
    if delta_vs_full > 0:
        print(f"  → Reduced feature set BETTER than full model")
    else:
        print(f"  → Full model is better by {-delta_vs_full:.4f} cm/h")

    # Save
    results_save = results_df.copy()
    results_save["features"] = results_save["features"].apply(str)
    out_path = TABLES_DIR / "feature_selection_m2.xlsx"
    results_save.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Plot
    plot_curve(results_df)

    print("\nDone.")
    print(f"\n  → Update MODEL2_FEATURES in pipeline/features.py with:")
    print(f"    {final_features}")

    return final_features


if __name__ == "__main__":
    main()