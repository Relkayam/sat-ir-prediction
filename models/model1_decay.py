"""
models/model1_decay.py — Model 1: within-segment IRD decay prediction (V2)
===========================================================================
V2 changes over V1:
  - Four evaluation conditions (A, B, C, D)
  - Basin sets resolved at runtime from outlier_basins.csv and basin_role
  - No hardcoded basin counts or basin numbers anywhere
  - All existing V1 plots preserved, produced per condition
  - Condition D: held-out basin test (basins never seen during training)

Execution order:
  1. python -m pipeline.build_dataset --rebuild
  2. python -m pipeline.build_reset_dataset
  3. python -m models.model1_decay          (pass 1 — skip: run basin_analysis first)
  4. python -m analysis.basin_analysis      (produces outlier_basins.csv)
  5. python -m models.model1_decay          (pass 2 — reads outlier_basins.csv)

Usage
-----
  python -m models.model1_decay
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, OUTLIER_CSV, TABLES_DIR,
    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED,
)
from pipeline.features import prepare_features, TARGET_M1
from models.utils import (
    ModelResult,
    metrics_norm, metrics_ird, back_transform,
    train_lightgbm, predict,
    per_basin_median_r2, get_splits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Load outlier basins from CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_outlier_basins() -> set[int]:
    """
    Load outlier_basins.csv produced by basin_analysis.py.
    Returns empty set if file does not exist — triggers pass-1 behavior
    where all basins are included (condition A = all 50 basins).
    """
    if not OUTLIER_CSV.exists():
        print(f"  INFO: {OUTLIER_CSV} not found.")
        print(f"  Running in pass-1 mode — all basins included.")
        print(f"  After this run: python -m analysis.basin_analysis")
        print(f"  Then re-run this script for pass-2 with clean conditions.")
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

    print(f"  Outlier basins loaded ({len(excluded)}): {sorted(excluded)}")
    return excluded


# ─────────────────────────────────────────────────────────────────────────────
# Resolve basin sets at runtime
# ─────────────────────────────────────────────────────────────────────────────

def resolve_basin_sets(
    df:             pd.DataFrame,
    outlier_basins: set[int],
) -> dict:
    """
    Derive all basin sets from the data at runtime.
    No hardcoded basin numbers or counts.

    Returns dict with keys:
      all            — all basins in the dataset
      outlier        — basins flagged by basin_analysis.py
      held_out       — basins reserved for condition D test only
      clean          — all - outlier - held_out  (condition A/B train+test)
      held_out_train — clean basins used for training in condition D
    """
    all_basins = set(
        df["basin_number"].dropna().unique().astype(int).tolist()
    )

    # held_out comes from basin_role column in the CSV
    if "basin_role" in df.columns:
        held_out = set(
            df.loc[df["basin_role"] == "held_out", "basin_number"]
            .dropna().unique().astype(int).tolist()
        )
    else:
        print("  WARNING: basin_role column not found — no held-out basins.")
        held_out = set()

    clean_basins   = all_basins - outlier_basins - held_out
    held_out_train = clean_basins  # condition D trains on clean minus held-out

    print(f"\n  Basin sets resolved at runtime:")
    print(f"    All basins       : {len(all_basins)}")
    print(f"    Outlier basins   : {len(outlier_basins)}  "
          f"{sorted(outlier_basins) if outlier_basins else '(none — pass-1 mode)'}")
    print(f"    Held-out basins  : {len(held_out)}  {sorted(held_out)}")
    print(f"    Clean basins     : {len(clean_basins)}  "
          f"(used for conditions A, B, D training)")
    print(f"    Condition D train: {len(held_out_train)} basins  "
          f"Condition D test: {len(held_out)} held-out basins")

    all_held_out_train = all_basins - held_out

    print(f"    All held-out train: {len(all_held_out_train)} basins  "
          f"(condition E — includes outlier basins)")


    return dict(
        all            = all_basins,
        outlier        = outlier_basins,
        held_out       = held_out,
        clean          = clean_basins,
        held_out_train = held_out_train,
        all_held_out_train = all_basins - held_out  # 45 basins including outliers
    )


# ─────────────────────────────────────────────────────────────────────────────
# Split reassignment for conditions B and C
# ─────────────────────────────────────────────────────────────────────────────

def _reassign_splits(df: pd.DataFrame) -> pd.Series:
    """
    Reassign random 70/15/15 splits by segment.
    Used for conditions B and C where original splits cover good
    segments only — bad segments need new split assignments.
    """
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

    train_s = set(shuffled[:n_tr].tolist())
    val_s   = set(shuffled[n_tr:n_tr + n_va].tolist())

    def _label(s):
        if pd.isna(s) or int(s) < 0: return "excluded"
        s = int(s)
        if s in train_s: return "train"
        if s in val_s:   return "val"
        return "test"

    return df["segment_id"].apply(_label)


# ─────────────────────────────────────────────────────────────────────────────
# Load events for a given condition
# ─────────────────────────────────────────────────────────────────────────────

def load_events_for_condition(
    df_full:    pd.DataFrame,
    condition:  str,
    basin_sets: dict,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    """
    Filter the full event dataset for a given evaluation condition
    and apply feature preparation.

    Condition logic
    ---------------
    A — clean basins, good segments only,     pre-assigned random split
    B — clean basins, ALL segments,           freshly reassigned splits
    C — all basins,   ALL segments,           freshly reassigned splits
    D — clean basins train + held-out test,   split_held_out column
    """
    df = df_full.copy()

    if condition == "A":
        df = df[
            (df["basin_number"].isin(basin_sets["clean"])) &
            (df["row_type"]         == "event") &
            (df["is_good_segment"]  == True)
        ].copy()
        # Use pre-assigned splits from build_dataset.py
        # These are segment-level random splits on good segments only

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
        train_basins = basin_sets["held_out_train"]
        held_out     = basin_sets["held_out"]
        all_d        = train_basins | held_out

        if not held_out:
            print(f"  SKIP condition D — no held-out basins defined")
            return None, None

        df = df[
            (df["basin_number"].isin(all_d)) &
            (df["row_type"]        == "event") &
            (df["is_good_segment"] == True)
        ].copy()

        if "split_held_out" not in df.columns:
            print(f"  SKIP condition D — split_held_out column not found. "
                  f"Rebuild from V2 build_dataset.py")
            return None, None

        df["split"] = df["split_held_out"].replace({"held_out_test": "test"})

    elif condition == "E":
        train_basins = basin_sets["all_held_out_train"]
        held_out = basin_sets["held_out"]
        all_e = train_basins | held_out

        if not held_out:
            print(f"  SKIP condition E — no held-out basins defined")
            return None, None

        df = df_full[
            (df_full["basin_number"].isin(all_e)) &
            (df_full["row_type"] == "event")
            ].copy()

        if "split_held_out" not in df.columns:
            print(f"  SKIP condition E — split_held_out column not found.")
            return None, None

        df["split"] = df["split_held_out"].replace({"held_out_test": "test"})
        # For training basins with no split_held_out assignment,
        # reassign fresh splits
        needs_split = df["split"].isin(["excluded", ""]) | df["split"].isna()
        df.loc[needs_split, "split"] = _reassign_splits(
            df[needs_split]
        ).values


    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Keep only rows with valid split assignments
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    if len(df) < 100:
        print(f"  WARNING: condition {condition} has only {len(df)} events — skipping")
        return None, None

    df, feat_cols = prepare_features(df)

    n_train = int((df["split"] == "train").sum())
    n_val   = int((df["split"] == "val").sum())
    n_test  = int((df["split"] == "test").sum())

    print(f"  Condition {condition}: {len(df)} events  "
          f"{df['basin_number'].nunique()} basins  "
          f"{len(feat_cols)} features")
    print(f"  Split: train={n_train}  val={n_val}  test={n_test}")

    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate — identical to V1
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df:        pd.DataFrame,
    feat_cols: list[str],
    label:     str,
) -> ModelResult:
    """Train LightGBM and evaluate on val and test splits."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    train, val, test = get_splits(df, feat_cols, TARGET_M1, split_col="split")

    model, scaler, used_cols = train_lightgbm(
        train, val, feat_cols, TARGET_M1, model2=False
    )

    # Metrics: IRD_norm_log
    print(f"\n  Metrics on IRD_norm_log:")
    val_metrics  = None
    test_metrics = None

    for split_name, split_df in [("val", val), ("test", test)]:
        y_pred = predict(model, scaler, used_cols, split_df)
        m = metrics_norm(
            split_df[TARGET_M1].values, y_pred,
            label=f"  {label} [{split_name}]",
        )
        if split_name == "val":
            val_metrics = m
        else:
            test_metrics = m

    # Metrics: raw IRD (cm/h)
    print(f"\n  Metrics on raw IRD (cm/h):")
    for split_name, split_df in [("val", val), ("test", test)]:
        orig = df[df["split"] == split_name].copy()
        orig["_pred_norm"] = predict(model, scaler, used_cols, orig)
        ird_reset = pd.to_numeric(
            orig["IRD_at_reset"], errors="coerce"
        ).values.ravel()
        ird_pred = back_transform(ird_reset, orig["_pred_norm"].values.ravel())
        ird_true = back_transform(
            ird_reset,
            pd.to_numeric(orig[TARGET_M1], errors="coerce").values.ravel(),
        )
        metrics_ird(ird_true, ird_pred,
                    label=f"  {label} IRD [{split_name}]")

    # Per-basin median R²
    eval_df      = df[df["split"].isin(["val", "test"])].copy()
    y_pred_eval  = predict(model, scaler, used_cols, eval_df)
    basin_med_r2 = per_basin_median_r2(eval_df, y_pred_eval, TARGET_M1)
    print(f"\n  Per-basin median R² (val+test): {basin_med_r2:+.4f}")

    return ModelResult(
        model        = model,
        scaler       = scaler,
        feat_cols    = used_cols,
        val_metrics  = val_metrics  or {},
        test_metrics = test_metrics or {},
        model_name   = f"LightGBM_{label}",
        extra        = dict(
            basin_median_r2 = basin_med_r2,
            n_basins        = df["basin_number"].nunique(),
            n_train         = len(train),
            n_val           = len(val),
            n_test          = len(test),
            label           = label,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baselines — identical to V1
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute naive and oracle baselines on the test split."""
    test = df[df["split"] == "test"].copy()

    # Naive: prev_IRD_norm_log
    test["pred_naive"] = np.nan
    for sid, grp in test.groupby("segment_id"):
        shifted = pd.to_numeric(grp[TARGET_M1], errors="coerce").shift(1)
        test.loc[grp.index, "pred_naive"] = shifted.values

    # Oracle: fitted exponential decay
    for col in ["seg_a", "seg_b", "seg_lambda"]:
        if col in test.columns:
            test[col] = pd.to_numeric(test[col], errors="coerce")

    has_fit = (
        test["seg_a"].notna()      &
        test["seg_b"].notna()      &
        test["seg_lambda"].notna() &
        test["LCT"].notna()
    )
    test["pred_decay_fit"] = np.nan
    if has_fit.any():
        lct = test.loc[has_fit, "LCT"].values
        a   = test.loc[has_fit, "seg_a"].values
        b   = test.loc[has_fit, "seg_b"].values
        lam = test.loc[has_fit, "seg_lambda"].values
        test.loc[has_fit, "pred_decay_fit"] = a * np.exp(-lam * lct) + b

    return test


# ─────────────────────────────────────────────────────────────────────────────
# Plots — all from V1, condition label added to titles
# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline_comparison(
    test_df:      pd.DataFrame,
    model_result: ModelResult,
    condition:    str,
    n_sample:     int = 3000,
) -> None:
    """Four-panel scatter + residuals in IRD_norm_log space."""
    valid_mask = test_df["pred_naive"].notna()
    test_valid = test_df[valid_mask].copy()

    y_true      = pd.to_numeric(test_valid[TARGET_M1], errors="coerce").values
    y_naive     = test_valid["pred_naive"].values
    y_decay_fit = test_valid["pred_decay_fit"].values
    y_model     = predict(
        model_result.model, model_result.scaler,
        model_result.feat_cols, test_valid,
    )

    rng = np.random.default_rng(42)
    predictors = [
        ("Naive (prev IRD_norm_log)", y_naive,     "steelblue"),
        ("Oracle decay fit",          y_decay_fit, "seagreen"),
        ("Model 1 (LightGBM)",        y_model,     "tomato"),
    ]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    finite_true = y_true[np.isfinite(y_true)]
    lo = np.percentile(finite_true, 1) - 0.1
    hi = np.percentile(finite_true, 99) + 0.1

    for i, (lbl, y_pred, color) in enumerate(predictors):
        ax = fig.add_subplot(gs[0, i])
        m  = metrics_norm(y_true, y_pred, verbose=False)
        finite_idx = np.where(np.isfinite(y_true) & np.isfinite(y_pred))[0]
        s = rng.choice(finite_idx,
                       size=min(n_sample, len(finite_idx)),
                       replace=False)
        ax.scatter(y_true[s], y_pred[s], s=5, alpha=0.3, color=color)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual IRD_norm_log", fontsize=8)
        ax.set_ylabel("Predicted",           fontsize=8)
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
        if m:
            ax.annotate(
                f"R²={m['r2']:+.3f}\nRMSE={m['rmse']:.3f}\n"
                f"rho={m['spearman_r']:+.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.85),
            )

    ax_res = fig.add_subplot(gs[0, 3])
    for lbl, y_pred, color in predictors:
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        ax_res.hist(y_true[valid] - y_pred[valid],
                    bins=60, alpha=0.55, color=color,
                    label=lbl, density=True)
    ax_res.axvline(0, color="black", linewidth=1,
                   linestyle="--", alpha=0.6)
    ax_res.set_xlabel("Residual", fontsize=8)
    ax_res.set_ylabel("Density",  fontsize=8)
    ax_res.set_title("Residuals", fontsize=9, fontweight="bold")
    ax_res.legend(fontsize=7); ax_res.tick_params(labelsize=7)
    ax_res.grid(True, alpha=0.2)

    fig.suptitle(
        f"Model 1 — Condition {condition} — Baseline comparison (IRD_norm_log)\n"
        f"{model_result.extra.get('label','')}  |  "
        f"n_test={valid_mask.sum()}  "
        f"n_basins={model_result.extra.get('n_basins')}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_baseline_comparison_ird(
    test_df:      pd.DataFrame,
    model_result: ModelResult,
    condition:    str,
    n_sample:     int = 3000,
) -> None:
    """Four-panel scatter + residuals in raw IRD (cm/h) space."""
    valid_mask = test_df["pred_naive"].notna()
    test_valid = test_df[valid_mask].copy()

    ird_reset  = pd.to_numeric(
        test_valid["IRD_at_reset"], errors="coerce"
    ).values.ravel()
    ird_true   = back_transform(
        ird_reset,
        pd.to_numeric(test_valid[TARGET_M1], errors="coerce").values.ravel(),
    )
    ird_naive  = back_transform(ird_reset,
                                test_valid["pred_naive"].values.ravel())
    ird_oracle = np.clip(
        back_transform(ird_reset,
                       test_valid["pred_decay_fit"].values.ravel()),
        0, 15.0,
    )
    ird_model = back_transform(
        ird_reset,
        predict(model_result.model, model_result.scaler,
                model_result.feat_cols, test_valid).ravel(),
    )

    rng = np.random.default_rng(42)
    predictors = [
        ("Naive (prev IRD_norm_log)", ird_naive,  "steelblue"),
        ("Oracle decay fit",          ird_oracle, "seagreen"),
        ("Model 1 (LightGBM)",        ird_model,  "tomato"),
    ]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    finite_true = ird_true[np.isfinite(ird_true) & (ird_true > 0)]
    lo = np.percentile(finite_true, 1) * 0.9
    hi = np.percentile(finite_true, 99) * 1.1

    for i, (lbl, ird_pred, color) in enumerate(predictors):
        ax = fig.add_subplot(gs[0, i])
        m  = metrics_ird(ird_true, ird_pred, verbose=False)
        finite_idx = np.where(
            np.isfinite(ird_true) & np.isfinite(ird_pred) &
            (ird_true > 0) & (ird_pred > 0)
        )[0]
        s = rng.choice(finite_idx,
                       size=min(n_sample, len(finite_idx)),
                       replace=False)
        ax.scatter(ird_true[s], ird_pred[s], s=5, alpha=0.3, color=color)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual IRD (cm/h)",    fontsize=8)
        ax.set_ylabel("Predicted IRD (cm/h)", fontsize=8)
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
        if m:
            ax.annotate(
                f"R²={m['r2']:+.3f}\n"
                f"RMSE={m['rmse']:.3f} cm/h\n"
                f"MAPE={m['mape']:.1f}%",
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.85),
            )

    ax_res = fig.add_subplot(gs[0, 3])
    for lbl, ird_pred, color in predictors:
        valid = (
            np.isfinite(ird_true) & np.isfinite(ird_pred) &
            (ird_true > 0) & (ird_pred > 0)
        )
        ax_res.hist(ird_true[valid] - ird_pred[valid],
                    bins=60, alpha=0.55, color=color,
                    label=lbl, density=True)
    ax_res.axvline(0, color="black", linewidth=1,
                   linestyle="--", alpha=0.6)
    ax_res.set_xlabel("Residual (cm/h)", fontsize=8)
    ax_res.set_ylabel("Density",         fontsize=8)
    ax_res.set_title("Residuals",        fontsize=9, fontweight="bold")
    ax_res.legend(fontsize=7); ax_res.tick_params(labelsize=7)
    ax_res.grid(True, alpha=0.2)

    fig.suptitle(
        f"Model 1 — Condition {condition} — Baseline comparison (raw IRD cm/h)\n"
        f"{model_result.extra.get('label','')}  |  "
        f"n_test={valid_mask.sum()}  "
        f"n_basins={model_result.extra.get('n_basins')}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def _plot_scatter_only(
    df:        pd.DataFrame,
    result:    ModelResult,
    condition: str,
    n_sample:  int = 3000,
) -> None:
    """
    Simple two-panel scatter for conditions B and C
    where oracle baseline is unavailable.
    """
    test  = df[df["split"] == "test"].copy()
    y_pred = predict(result.model, result.scaler, result.feat_cols, test)
    y_true = pd.to_numeric(test[TARGET_M1], errors="coerce").values.ravel()

    ird_reset = pd.to_numeric(
        test["IRD_at_reset"], errors="coerce"
    ).values.ravel()
    ird_pred = back_transform(ird_reset, y_pred.ravel())
    ird_true = back_transform(ird_reset, y_true.ravel())

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, yt, yp, xlabel, ylabel in [
        (axes[0], y_true,   y_pred,
         "Actual IRD_norm_log", "Predicted IRD_norm_log"),
        (axes[1], ird_true, ird_pred,
         "Actual IRD (cm/h)", "Predicted IRD (cm/h)"),
    ]:
        is_ird = "cm/h" in xlabel
        finite = (
            np.isfinite(yt) & np.isfinite(yp) &
            ((yt > 0) if is_ird else True)
        )
        idx = np.where(finite)[0]
        s   = rng.choice(idx, size=min(n_sample, len(idx)), replace=False)
        ax.scatter(yt[s], yp[s], s=5, alpha=0.3, color="tomato")
        lo = np.percentile(yt[finite], 1)
        hi = np.percentile(yt[finite], 99)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2)
        m = (metrics_ird(yt, yp, verbose=False)
             if is_ird
             else metrics_norm(yt, yp, verbose=False))
        if m:
            txt = f"R²={m.get('r2',np.nan):+.3f}\nRMSE={m.get('rmse',np.nan):.3f}"
            if is_ird:
                txt += f"\nMAPE={m.get('mape',np.nan):.1f}%"
            ax.annotate(txt, xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.85))

    fig.suptitle(
        f"Model 1 — Condition {condition} — {result.extra.get('label','')}\n"
        f"n_test={int(np.isfinite(y_true).sum())}  "
        f"n_basins={result.extra.get('n_basins')}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_condition_summary(results: dict) -> None:
    """
    Bar chart comparing R²(log), R²(IRD), MAPE across all conditions.
    This is the key SI figure for the filtering justification.
    """
    conditions = list(results.keys())
    r2_log = [results[c]["r2_log"]  for c in conditions]
    r2_ird = [results[c]["r2_ird"]  for c in conditions]
    mape   = [results[c]["mape"]    for c in conditions]
    labels = [results[c]["label"]   for c in conditions]

    x = np.arange(len(conditions))
    colors = ["tomato", "steelblue", "seagreen", "mediumpurple"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, vals, title, ylabel in zip(
        axes,
        [r2_log, r2_ird, mape],
        ["R² (log-ratio space)", "R² (raw IRD cm/h)", "MAPE (%)"],
        ["R²", "R²", "MAPE (%)"],
    ):
        bars = ax.bar(x, vals,
                      color=colors[:len(conditions)],
                      alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cond. {c}" for c in conditions], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.005,
                    f"{val:.3f}" if ylabel != "MAPE (%)" else f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )

        # Condition description below bars
        y_min = ax.get_ylim()[0]
        y_rng = ax.get_ylim()[1] - y_min
        for xi, lbl in zip(x, labels):
            ax.text(xi, y_min - 0.03 * y_rng,
                    lbl, ha="center", va="top",
                    fontsize=6, rotation=10)

    fig.suptitle(
        "Model 1 — Evaluation condition comparison (test split)\n"
        "A=Clean  B=All segments  C=All basins  D=Held-out basins",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Print and save
# ─────────────────────────────────────────────────────────────────────────────

def print_baseline_comparison(
    test_df:      pd.DataFrame,
    model_result: ModelResult,
    condition:    str,
) -> None:
    """Print baseline vs model table in IRD_norm_log space."""
    valid_mask  = test_df["pred_naive"].notna()
    test_valid  = test_df[valid_mask].copy()
    y_true      = pd.to_numeric(test_valid[TARGET_M1], errors="coerce").values
    y_naive     = test_valid["pred_naive"].values
    y_decay_fit = test_valid["pred_decay_fit"].values
    y_model     = predict(model_result.model, model_result.scaler,
                          model_result.feat_cols, test_valid)

    print(f"\n{'='*68}")
    print(f"  BASELINE COMPARISON — Condition {condition} [IRD_norm_log]")
    print(f"  n={valid_mask.sum()} events with valid prev_IRD_norm_log")
    print(f"{'='*68}")
    print(f"  {'Predictor':<35} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'rho':>8}")
    print(f"  {'-'*63}")

    r2s = {}
    for lbl, y_pred in [
        ("Naive (prev_IRD_norm_log)", y_naive),
        ("Oracle decay fit",          y_decay_fit),
        ("Model 1 (LightGBM)",        y_model),
    ]:
        m = metrics_norm(y_true, y_pred, verbose=False)
        r2s[lbl] = (m or {}).get("r2", np.nan)
        if m:
            print(f"  {lbl:<35} "
                  f"{m['r2']:>+8.4f} {m['rmse']:>8.4f} "
                  f"{m['mae']:>8.4f} {m['spearman_r']:>+8.4f}")

    naive_r2  = r2s.get("Naive (prev_IRD_norm_log)", np.nan)
    oracle_r2 = r2s.get("Oracle decay fit",          np.nan)
    model_r2  = r2s.get("Model 1 (LightGBM)",        np.nan)
    pct_ceil  = (model_r2 / oracle_r2 * 100
                 if np.isfinite(oracle_r2) and oracle_r2 > 0
                 else np.nan)

    print(f"\n  Naive R²={naive_r2:+.3f}  "
          f"Oracle R²={oracle_r2:+.3f}  "
          f"Model R²={model_r2:+.3f}  "
          f"({pct_ceil:.0f}% of oracle ceiling)")


def print_baseline_comparison_ird(
    test_df:      pd.DataFrame,
    model_result: ModelResult,
    condition:    str,
) -> None:
    """Print baseline vs model table in raw IRD (cm/h) space."""
    valid_mask = test_df["pred_naive"].notna()
    test_valid = test_df[valid_mask].copy()

    ird_reset = pd.to_numeric(
        test_valid["IRD_at_reset"], errors="coerce"
    ).values.ravel()
    ird_true  = back_transform(
        ird_reset,
        pd.to_numeric(test_valid[TARGET_M1], errors="coerce").values.ravel(),
    )
    ird_naive  = back_transform(ird_reset,
                                test_valid["pred_naive"].values.ravel())
    ird_oracle = np.clip(
        back_transform(ird_reset,
                       test_valid["pred_decay_fit"].values.ravel()),
        0, 15.0,
    )
    ird_model = back_transform(
        ird_reset,
        predict(model_result.model, model_result.scaler,
                model_result.feat_cols, test_valid).ravel(),
    )

    print(f"\n{'='*68}")
    print(f"  BASELINE COMPARISON — Condition {condition} [IRD cm/h]")
    print(f"{'='*68}")
    print(f"  {'Predictor':<35} {'R²':>8} {'RMSE':>8} {'MAPE%':>8} {'rel_RMSE':>9}")
    print(f"  {'-'*70}")

    for lbl, ird_pred in [
        ("Naive (prev_IRD_norm_log)", ird_naive),
        ("Oracle decay fit",          ird_oracle),
        ("Model 1 (LightGBM)",        ird_model),
    ]:
        m = metrics_ird(ird_true, ird_pred, verbose=False)
        if m:
            print(f"  {lbl:<35} "
                  f"{m['r2']:>+8.4f} {m['rmse']:>8.3f} "
                  f"{m['mape']:>7.1f}% {m['rel_rmse']:>9.4f}")


def save_results(all_results: dict) -> None:
    """Save all condition results to XLSX."""
    rows = []
    for condition, res in all_results.items():
        rows.append(dict(
            condition    = condition,
            label        = res.get("label",        ""),
            r2_log       = res.get("r2_log",        np.nan),
            rmse_log     = res.get("rmse_log",      np.nan),
            r2_ird       = res.get("r2_ird",        np.nan),
            mape         = res.get("mape",          np.nan),
            basin_med_r2 = res.get("basin_med_r2",  np.nan),
            n_basins     = res.get("n_basins",      np.nan),
            n_train      = res.get("n_train",       np.nan),
            n_test       = res.get("n_test",        np.nan),
        ))
    out  = pd.DataFrame(rows)
    path = TABLES_DIR / "model1_results_v2.xlsx"
    out.to_excel(path, index=False)
    print(f"\n  Saved: {path}")



def plot_held_out_basins_timeseries(
    df_full:    pd.DataFrame,
    result:     ModelResult,
    basin_sets: dict,
) -> None:
    """
    Time series plots for the 5 held-out basins.
    Each panel shows actual IRD (dots) and predicted IRD (crosses)
    over the full operational history of the basin.
    Colored by segment — each segment a different color.
    Vertical dashed lines mark segment resets.

    Layout: 5 rows × 1 column (one row per held-out basin)

    Parameters
    ----------
    df_full    : full event dataset
    result     : ModelResult from condition D
    basin_sets : resolved basin sets
    """
    from config import FIELD_NAMES

    held_out = sorted(basin_sets["held_out"])
    if not held_out:
        print("  No held-out basins — skipping time series plot")
        return

    # Get held-out basin events — good segments only
    ho_df = df_full[
        (df_full["basin_number"].isin(held_out)) &
        (df_full["row_type"]        == "event") &
        (df_full["is_good_segment"] == True)
    ].copy()

    if ho_df.empty:
        print("  WARNING: no good events for held-out basins")
        return

    ho_df, _ = prepare_features(ho_df)

    # Predict
    ho_df["_pred_norm"] = predict(
        result.model, result.scaler, result.feat_cols, ho_df
    )
    ird_reset = pd.to_numeric(
        ho_df["IRD_at_reset"], errors="coerce"
    ).values.ravel()
    ho_df["ird_pred"] = back_transform(
        ird_reset, ho_df["_pred_norm"].values.ravel()
    )
    ho_df["ird_true"] = back_transform(
        ird_reset,
        pd.to_numeric(ho_df[TARGET_M1], errors="coerce").values.ravel(),
    )

    import matplotlib.cm as cm

    n_basins = len(held_out)
    fig, axes = plt.subplots(
        n_basins, 1,
        figsize=(14, 4 * n_basins),
        squeeze=False,
    )

    for ax, bn in zip(axes[:, 0], held_out):
        bdf   = ho_df[ho_df["basin_number"] == bn].sort_values(
            "opening_valve_date"
        )
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))

        segs  = sorted(bdf["segment_id"].dropna().unique())
        cmap  = cm.get_cmap("tab20", max(len(segs), 1))

        reset_dates = []

        for i, sid in enumerate(segs):
            seg   = bdf[bdf["segment_id"] == sid].sort_values(
                "opening_valve_date"
            )
            color = cmap(i)

            # Actual IRD — filled dots
            valid_true = (
                seg["ird_true"].notna() & (seg["ird_true"] > 0)
            )
            ax.scatter(
                seg.loc[valid_true, "opening_valve_date"],
                seg.loc[valid_true, "ird_true"],
                s=18, alpha=0.80, color=color,
                marker="o", zorder=3,
                label=f"seg {int(sid)} actual" if i < 3 else None,
            )

            # Predicted IRD — crosses
            valid_pred = (
                seg["ird_pred"].notna() & (seg["ird_pred"] > 0)
            )
            ax.scatter(
                seg.loc[valid_pred, "opening_valve_date"],
                seg.loc[valid_pred, "ird_pred"],
                s=18, alpha=0.80, color=color,
                marker="x", linewidths=1.5, zorder=4,
            )

            # Connect actual to predicted with thin gray lines
            both = valid_true & valid_pred
            for _, row in seg[both].iterrows():
                ax.plot(
                    [row["opening_valve_date"]] * 2,
                    [row["ird_true"], row["ird_pred"]],
                    color="gray", linewidth=0.4, alpha=0.3, zorder=2,
                )

            # Record reset date for vertical line
            if not seg.empty:
                reset_dates.append(seg["opening_valve_date"].iloc[0])

        # Reset lines
        for j, rd in enumerate(reset_dates):
            ax.axvline(
                rd, color="black", linewidth=0.6,
                linestyle="--", alpha=0.3,
                label="reset" if j == 0 else None,
            )

        # Per-basin metrics
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = (
            np.isfinite(ird_t) & np.isfinite(ird_p) &
            (ird_t > 0) & (ird_p > 0)
        )
        m = metrics_ird(ird_t[mask], ird_p[mask], verbose=False) or {}

        ax.set_title(
            f"Basin {bn}  ({field})  —  HELD-OUT (never seen in training)\n"
            f"R²={m.get('r2', np.nan):+.3f}  "
            f"RMSE={m.get('rmse', np.nan):.3f} cm/h  "
            f"MAPE={m.get('mape', np.nan):.1f}%  "
            f"n_segments={len(segs)}",
            fontsize=9,
        )
        ax.set_ylabel("IRD (cm/h)", fontsize=8)
        ax.set_xlabel("Date",       fontsize=8)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.grid(True, alpha=0.2)

        # Legend: actual vs predicted only (not per segment)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="gray",
                   markersize=6, linestyle="None", label="Actual IRD"),
            Line2D([0], [0], marker="x", color="gray",
                   markersize=6, linestyle="None",
                   markeredgewidth=1.5, label="Predicted IRD"),
            Line2D([0], [0], color="black", linewidth=0.8,
                   linestyle="--", alpha=0.5, label="Reset"),
        ]
        ax.legend(handles=legend_elements, fontsize=7,
                  loc="upper right", ncol=3)

    fig.suptitle(
        "Model 1 — Held-out basin time series\n"
        "Condition D: model trained on clean basins — "
        "these basins never seen during training\n"
        "Dots = actual IRD  |  Crosses = predicted IRD  |  "
        "Each color = one segment",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_held_out_basins(
    df_full:    pd.DataFrame,
    result:     ModelResult,
    basin_sets: dict,
    n_sample:   int = 500,
) -> None:
    """
    Scatter plots for the 5 held-out basins — actual vs predicted IRD.

    The model was trained on condition D (clean basins only, held-out
    basins excluded entirely). These plots show how well the model
    generalises to basins it has never seen during training.

    Layout: 1 row × 5 columns (one panel per held-out basin)
    Each panel shows:
      - Actual vs predicted IRD (cm/h) scatter
      - 1:1 line
      - R², RMSE, MAPE annotated
      - Basin number and field in title

    This figure demonstrates out-of-sample generalisability and
    is a candidate for the paper SI.

    Parameters
    ----------
    df_full    : full event dataset (all basins)
    result     : ModelResult from condition D training
    basin_sets : resolved basin sets (needs 'held_out' key)
    n_sample   : max points to plot per basin (for readability)
    """
    from config import FIELD_NAMES, FIELD_COLORS

    held_out = sorted(basin_sets["held_out"])
    if not held_out:
        print("  No held-out basins — skipping held-out scatter plot")
        return

    # Get held-out basin events — good segments only
    ho_df = df_full[
        (df_full["basin_number"].isin(held_out)) &
        (df_full["row_type"]        == "event") &
        (df_full["is_good_segment"] == True)
    ].copy()

    if ho_df.empty:
        print("  WARNING: no good events found for held-out basins")
        return

    ho_df, _ = prepare_features(ho_df)

    # Predict using condition D model
    ho_df["_pred_norm"] = predict(
        result.model, result.scaler, result.feat_cols, ho_df
    )

    ird_reset = pd.to_numeric(
        ho_df["IRD_at_reset"], errors="coerce"
    ).values.ravel()
    ho_df["ird_pred"] = back_transform(ird_reset, ho_df["_pred_norm"].values.ravel())
    ho_df["ird_true"] = back_transform(
        ird_reset,
        pd.to_numeric(ho_df[TARGET_M1], errors="coerce").values.ravel(),
    )

    # One panel per held-out basin
    n_basins = len(held_out)
    fig, axes = plt.subplots(
        1, n_basins,
        figsize=(4 * n_basins, 5),
        squeeze=False,
    )
    axes = axes[0]

    rng = np.random.default_rng(42)

    for ax, bn in zip(axes, held_out):
        bdf = ho_df[ho_df["basin_number"] == bn].copy()
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        color = FIELD_COLORS.get(field, "tomato")

        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)

        mask = (
            np.isfinite(ird_t) & np.isfinite(ird_p) &
            (ird_t > 0) & (ird_p > 0)
        )
        ird_t = ird_t[mask]
        ird_p = ird_p[mask]

        if len(ird_t) < 5:
            ax.set_title(f"Basin {bn}\n({field})\nInsufficient data")
            ax.axis("off")
            continue

        # Subsample for readability
        idx = rng.choice(
            len(ird_t),
            size=min(n_sample, len(ird_t)),
            replace=False,
        )

        ax.scatter(
            ird_t[idx], ird_p[idx],
            s=12, alpha=0.5, color=color, zorder=3,
        )

        lo = np.percentile(ird_t, 1) * 0.9
        hi = np.percentile(ird_t, 99) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        m = metrics_ird(ird_t, ird_p, verbose=False)
        if m:
            ax.annotate(
                f"R²={m['r2']:+.3f}\n"
                f"RMSE={m['rmse']:.3f} cm/h\n"
                f"MAPE={m['mape']:.1f}%\n"
                f"n={m['n']}",
                xy=(0.05, 0.97), xycoords="axes fraction",
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.90),
            )

        ax.set_title(
            f"Basin {bn}\n({field})",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Actual IRD (cm/h)",    fontsize=9)
        ax.set_ylabel("Predicted IRD (cm/h)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Model 1 — Held-out basin generalisation test\n"
        "Condition D: model trained on clean basins only — "
        "these 5 basins were never seen during training",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    # Print summary table
    print(f"\n{'='*65}")
    print("  HELD-OUT BASIN GENERALISATION — Condition D test")
    print(f"  Model trained on {len(basin_sets['held_out_train'])} basins, "
          f"tested on {len(held_out)} unseen basins")
    print(f"{'='*65}")
    print(f"  {'Basin':>7}  {'Field':<10}  {'n':>6}  "
          f"{'R²':>8}  {'RMSE':>8}  {'MAPE%':>7}")
    print(f"  {'-'*55}")

    all_true = []
    all_pred = []
    for bn in held_out:
        bdf   = ho_df[ho_df["basin_number"] == bn]
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = (
            np.isfinite(ird_t) & np.isfinite(ird_p) &
            (ird_t > 0) & (ird_p > 0)
        )
        ird_t = ird_t[mask]; ird_p = ird_p[mask]
        all_true.append(ird_t); all_pred.append(ird_p)

        m     = metrics_ird(ird_t, ird_p, verbose=False) or {}
        field = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        print(
            f"  {bn:>7}  {field:<10}  "
            f"{m.get('n', 0):>6}  "
            f"{m.get('r2', np.nan):>+8.3f}  "
            f"{m.get('rmse', np.nan):>8.3f}  "
            f"{m.get('mape', np.nan):>6.1f}%"
        )

    # Pooled across all held-out basins
    if all_true:
        ird_t_all = np.concatenate(all_true)
        ird_p_all = np.concatenate(all_pred)
        m_all     = metrics_ird(ird_t_all, ird_p_all, verbose=False) or {}
        print(f"  {'-'*55}")
        print(
            f"  {'POOLED':>7}  {'all held-out':<10}  "
            f"{m_all.get('n', 0):>6}  "
            f"{m_all.get('r2', np.nan):>+8.3f}  "
            f"{m_all.get('rmse', np.nan):>8.3f}  "
            f"{m_all.get('mape', np.nan):>6.1f}%"
        )



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  MODEL 1 V2 — Within-segment IRD decay — Four conditions")
    print(f"  Target: {TARGET_M1} = log(IRD / IRD_at_reset)")
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

    # Load outlier basins from CSV
    outlier_basins = load_outlier_basins()

    # Resolve basin sets at runtime — no hardcoded numbers
    basin_sets = resolve_basin_sets(df_full, outlier_basins)

    # Condition labels derived from resolved sets
    condition_labels = {
        "A": (f"Clean model "
              f"({len(basin_sets['clean'])} basins, good segs)"),
        "B": (f"All segments "
              f"({len(basin_sets['clean'])} basins)"),
        "C": (f"All data "
              f"({len(basin_sets['all'])} basins)"),
        "D": (f"Held-out test "
              f"({len(basin_sets['held_out_train'])} train / "
              f"{len(basin_sets['held_out'])} test basins)"),

        "E": (f"All-data held-out "
              f"({len(basin_sets['all_held_out_train'])} train / "
              f"{len(basin_sets['held_out'])} test basins)"),
    }

    all_results   = {}
    model_results = {}

    for condition in ["A", "B", "C", "D", "E"]:
        print(f"\n{'='*65}")
        print(f"  CONDITION {condition} — {condition_labels[condition]}")
        print(f"{'='*65}")

        df_cond, feat_cols = load_events_for_condition(
            df_full, condition, basin_sets
        )

        if df_cond is None:
            print(f"  Condition {condition} skipped.")
            continue

        result = train_and_evaluate(
            df_cond, feat_cols,
            label=f"Condition {condition} — {condition_labels[condition]}",
        )
        model_results[condition] = (result, df_cond)

        # IRD metrics on test split
        test_df = df_cond[df_cond["split"] == "test"].copy()
        test_df["_pred"] = predict(
            result.model, result.scaler, result.feat_cols, test_df
        )
        ird_reset = pd.to_numeric(
            test_df["IRD_at_reset"], errors="coerce"
        ).values.ravel()
        ird_pred = back_transform(ird_reset, test_df["_pred"].values.ravel())
        ird_true = back_transform(
            ird_reset,
            pd.to_numeric(test_df[TARGET_M1],
                          errors="coerce").values.ravel(),
        )
        m_ird = metrics_ird(ird_true, ird_pred, verbose=False) or {}

        all_results[condition] = dict(
            label        = condition_labels[condition],
            r2_log       = result.test_metrics.get("r2",   np.nan),
            rmse_log     = result.test_metrics.get("rmse",  np.nan),
            r2_ird       = m_ird.get("r2",   np.nan),
            mape         = m_ird.get("mape",  np.nan),
            basin_med_r2 = result.extra.get("basin_median_r2", np.nan),
            n_basins     = result.extra.get("n_basins"),
            n_train      = result.extra.get("n_train"),
            n_test       = result.extra.get("n_test"),
        )

        # Baselines and plots
        has_decay = (
            all(c in df_cond.columns
                for c in ["seg_a", "seg_b", "seg_lambda"])
            and condition in ("A", "D")
        )

        if has_decay:
            test_with_baselines = compute_baselines(df_cond)
            print_baseline_comparison(
                test_with_baselines, result, condition
            )
            print_baseline_comparison_ird(
                test_with_baselines, result, condition
            )
            plot_baseline_comparison(
                test_with_baselines, result, condition
            )
            plot_baseline_comparison_ird(
                test_with_baselines, result, condition
            )
        else:
            _plot_scatter_only(df_cond, result, condition)

    # Cross-condition summary
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print("  SUMMARY — All conditions (test split)")
        print(f"{'='*65}")
        print(f"  {'Cond':<6} {'Label':<48} {'R²(log)':>8} "
              f"{'R²(IRD)':>8} {'MAPE%':>7} {'BasMedR²':>9}")
        print(f"  {'-'*85}")
        for c, res in all_results.items():
            print(
                f"  {c:<6} {res['label']:<48} "
                f"{res['r2_log']:>+8.4f} "
                f"{res['r2_ird']:>+8.4f} "
                f"{res['mape']:>6.1f}% "
                f"{res['basin_med_r2']:>+9.4f}"
            )

        plot_condition_summary(all_results)
        save_results(all_results)

        # Held-out scatter — only for condition D
        if condition == ("D", "E") and basin_sets["held_out"]:
            plot_held_out_basins(df_full, result, basin_sets)

    if condition in ("D", "E") and basin_sets["held_out"]:
        plot_held_out_basins(df_full, result, basin_sets)
        plot_held_out_basins_timeseries(df_full, result, basin_sets)

    print("\nDone.")


if __name__ == "__main__":
    main()