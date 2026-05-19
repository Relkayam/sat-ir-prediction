"""
analysis/basin_analysis.py — Per-basin evaluation and outlier detection
========================================================================
Trains a pass-1 global LightGBM on ALL basins (including held-out),
evaluates per-basin performance, classifies outlier types, and writes
outlier_basins.csv.

This is the ONLY place outlier basins are defined. The CSV it produces
is the single source of truth — no outlier basin numbers are hardcoded
anywhere else in the project.

Execution order
---------------
  1. python -m pipeline.build_dataset --rebuild
  2. python -m pipeline.build_reset_dataset
  *** THIS SCRIPT RUNS HERE ***
  * TBA python -m run_bootstrap - to find the best candidate to present in paper
  3. python -m models.model1_decay
  4. python -m models.model2_reset
  5. python -m experiments.run_bootstrap

Pass-1 design
-------------
  - Trains on ALL 50 basins including held-out
  - Purpose: identify anomalous basins across the full system
  - Held-out basins ARE in training for this diagnostic only
  - Their outlier status affects Condition E training (which basins
    are included), not the held-out test set

Usage
-----
  python -m analysis.basin_analysis
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
    EVENT_CSV, OUTLIER_CSV, TABLES_DIR, FIGURES_DIR,
    OUTLIER_R2_THRESHOLD, OUTLIER_REL_RMSE_THRESHOLD,
    FIELD_NAMES,
)
from pipeline.features import prepare_features, TARGET_M1
from models.utils import (
    metrics_ird, back_transform,
    train_lightgbm, predict,
    get_splits,
)

# ── Output directories ────────────────────────────────────────────────────────
BASIN_METRICS_XLSX = TABLES_DIR / "basin_metrics.xlsx"
BASIN_PLOT_DIR     = FIGURES_DIR / "basin_plots"
BASIN_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Outlier type thresholds (data-derived in classify_outlier_types) ──────────
LOW_RANGE_PERCENTILE = 20
REGIME_SHIFT_FACTOR  = 1.5

# ── Plot style — local constants (not imported from config) ───────────────────
SPLIT_COLORS  = {"train": "#555555", "val": "#1C7293", "test": "#E07B39"}
SPLIT_MARKERS = {"train": "o",       "val": "s",       "test": "^"}
SPLIT_ALPHA   = {"train": 0.35,      "val": 0.85,      "test": 0.85}
SPLIT_SIZE    = {"train": 15,        "val": 40,        "test": 40}


# ─────────────────────────────────────────────────────────────────────────────
# Load data — ALL basins, good segments only
# ─────────────────────────────────────────────────────────────────────────────

def load_events() -> tuple[pd.DataFrame, list[str]]:
    """
    Load good events from ALL 50 basins — no exclusions.
    Pass-1 model uses the full dataset to identify anomalous basins.
    basin_role column preserved for annotation in plots.
    """
    if not EVENT_CSV.exists():
        raise FileNotFoundError(
            f"{EVENT_CSV} not found.\n"
            "Run: python -m pipeline.build_dataset --rebuild"
        )

    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    if TARGET_M1 not in df.columns:
        if "IRD_norm" in df.columns:
            df[TARGET_M1] = df["IRD_norm"]
        else:
            raise ValueError(f"Target '{TARGET_M1}' not found in CSV.")

    # Good events only — all 50 basins
    df = df[
        (df["row_type"]        == "event") &
        (df["is_good_segment"] == True)
    ].copy()

    if "basin_role" not in df.columns:
        print("  WARNING: basin_role not found — defaulting to 'clean'.")
        print("  Rebuild CSV with: python -m pipeline.build_dataset --rebuild")
        df["basin_role"] = "clean"

    df, feat_cols = prepare_features(df)

    role_counts = df.groupby("basin_role")["basin_number"].nunique()
    print(f"  Loaded {len(df):,} good events  "
          f"{df['basin_number'].nunique()} basins")
    for role, n in role_counts.items():
        print(f"    {role:<12}: {n} basins")

    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_basin_metrics(
    df:        pd.DataFrame,
    feat_cols: list[str],
    model,
    scaler,
    used_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the pass-1 global model on each basin's val+test events.
    Back-transforms predictions to raw IRD (cm/h).
    Returns (metrics_df, eval_df).
    """
    eval_df = df[df["split"].isin(["val", "test"])].copy()
    eval_df["_pred_norm"] = predict(model, scaler, used_cols, eval_df)

    ird_reset             = pd.to_numeric(eval_df["IRD_at_reset"], errors="coerce")
    eval_df["ird_actual"] = back_transform(
        ird_reset.values,
        pd.to_numeric(eval_df[TARGET_M1], errors="coerce").values,
    )
    eval_df["ird_pred"] = back_transform(
        ird_reset.values,
        eval_df["_pred_norm"].values,
    )

    rows = []
    for bn, bdf in eval_df.groupby("basin_number"):
        n_train    = int((df[df["basin_number"] == bn]["split"] == "train").sum())
        basin_role = str(
            df.loc[df["basin_number"] == bn, "basin_role"].iloc[0]
            if len(df[df["basin_number"] == bn]) > 0 else "clean"
        )
        m = metrics_ird(
            bdf["ird_actual"].values.astype(float),
            bdf["ird_pred"].values.astype(float),
            verbose=False,
        )
        if m is None:
            continue

        m["basin_number"] = int(bn)
        m["field_name"]   = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
        m["n_train"]      = n_train
        m["basin_role"]   = basin_role
        m["auto_flag"]    = bool(
            m["r2"]       < OUTLIER_R2_THRESHOLD or
            m["rel_rmse"] > OUTLIER_REL_RMSE_THRESHOLD
        )
        rows.append(m)

    metrics_df = pd.DataFrame(rows).sort_values("r2").reset_index(drop=True)
    return metrics_df, eval_df


# ─────────────────────────────────────────────────────────────────────────────
# Outlier type classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_outlier_types(
    metrics_df:  pd.DataFrame,
    good_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify each auto-flagged basin into Type 1, Type 2, or Type 3.

    Type 1 — Low dynamic range: IRD IQR below 20th system percentile
    Type 2 — Regime shift:      large IRD step change mid-record
    Type 3 — Non-stationary:    neither Type 1 nor Type 2
    """
    metrics_df = metrics_df.copy()
    metrics_df["outlier_type"]        = ""
    metrics_df["outlier_type_label"]  = ""
    metrics_df["outlier_type_reason"] = ""

    basin_chars = {}
    for bn, bdf in good_events.groupby("basin_number"):
        ird = pd.to_numeric(bdf["IRD"], errors="coerce").dropna()
        if len(ird) < 10:
            continue
        iqr     = float(np.percentile(ird, 75) - np.percentile(ird, 25))
        mid     = len(ird) // 2
        ird_srt = pd.to_numeric(
            bdf.sort_values("opening_valve_date")["IRD"], errors="coerce"
        ).dropna()
        mean_first  = float(ird_srt.iloc[:mid].mean())
        mean_second = float(ird_srt.iloc[mid:].mean())
        reset_ird   = pd.to_numeric(
            bdf.groupby("segment_id")["IRD_at_reset"].first(), errors="coerce"
        ).dropna()
        basin_chars[int(bn)] = dict(
            iqr        = iqr,
            half_diff  = abs(mean_first - mean_second),
            mean_first = mean_first,
            mean_second= mean_second,
            reset_std  = float(reset_ird.std()) if len(reset_ird) > 1 else np.nan,
        )

    all_iqrs   = [v["iqr"]       for v in basin_chars.values() if np.isfinite(v["iqr"])]
    all_diffs  = [v["half_diff"] for v in basin_chars.values() if np.isfinite(v["half_diff"])]
    all_rst    = [v["reset_std"] for v in basin_chars.values() if np.isfinite(v["reset_std"])]

    t1 = float(np.percentile(all_iqrs,  LOW_RANGE_PERCENTILE))
    t2 = float(np.median(all_diffs) * REGIME_SHIFT_FACTOR)
    t3 = float(np.median(all_rst))

    print(f"\n  Outlier type thresholds (data-derived):")
    print(f"    Type 1 — IQR < {t1:.3f} cm/h  (bottom {LOW_RANGE_PERCENTILE}th pct)")
    print(f"    Type 2 — half-diff > {t2:.3f} cm/h  (median × {REGIME_SHIFT_FACTOR})")
    print(f"    Type 3 — residual  (reset_std median = {t3:.3f} cm/h)")

    for idx, row in metrics_df.iterrows():
        if not row["auto_flag"]:
            continue
        bn = int(row["basin_number"])
        if bn not in basin_chars:
            metrics_df.at[idx, "outlier_type"]        = "Type3"
            metrics_df.at[idx, "outlier_type_label"]  = "Non-stationary operations"
            metrics_df.at[idx, "outlier_type_reason"] = "Insufficient data"
            continue

        c = basin_chars[bn]

        if c["iqr"] <= t1:
            metrics_df.at[idx, "outlier_type"]        = "Type1"
            metrics_df.at[idx, "outlier_type_label"]  = "Low dynamic range"
            metrics_df.at[idx, "outlier_type_reason"] = (
                f"IQR={c['iqr']:.3f} <= {t1:.3f} cm/h")
        elif c["half_diff"] >= t2:
            direction = "decreasing" if c["mean_second"] < c["mean_first"] else "increasing"
            metrics_df.at[idx, "outlier_type"]        = "Type2"
            metrics_df.at[idx, "outlier_type_label"]  = "Regime shift"
            metrics_df.at[idx, "outlier_type_reason"] = (
                f"half_diff={c['half_diff']:.3f} >= {t2:.3f} cm/h  IRD {direction}")
        else:
            metrics_df.at[idx, "outlier_type"]        = "Type3"
            metrics_df.at[idx, "outlier_type_label"]  = "Non-stationary operations"
            metrics_df.at[idx, "outlier_type_reason"] = (
                f"Not Type1 (IQR={c['iqr']:.3f}) or Type2 "
                f"(half_diff={c['half_diff']:.3f})")

    return metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    print(f"\n  {'Basin':>7}  {'Field':<10}  {'Role':<10}  {'n_eval':>6}  "
          f"{'R²':>7}  {'RMSE':>7}  {'rel_RMSE':>9}  {'MAPE%':>7}  "
          f"{'Flag':>5}  {'Type':<8}")
    print(f"  {'-'*90}")

    for _, row in metrics_df.iterrows():
        flag  = "FLAG" if row["auto_flag"] else ""
        otype = row.get("outlier_type", "")
        role  = row.get("basin_role", "")
        print(f"  {int(row['basin_number']):>7}  "
              f"{str(row['field_name']):<10}  {role:<10}  "
              f"{int(row['n']):>6}  {row['r2']:>+7.3f}  "
              f"{row['rmse']:>7.3f}  {row['rel_rmse']:>9.3f}  "
              f"{row['mape']:>7.1f}  {flag:>5}  {otype:<8}")

    flagged = metrics_df[metrics_df["auto_flag"]]
    print(f"\n  Auto-flagged: {len(flagged)} basins")

    if "outlier_type" in metrics_df.columns:
        print(f"\n  Type breakdown:")
        for t, label in [("Type1", "Low dynamic range"),
                         ("Type2", "Regime shift"),
                         ("Type3", "Non-stationary operations")]:
            n      = int((metrics_df["outlier_type"] == t).sum())
            basins = metrics_df.loc[
                metrics_df["outlier_type"] == t, "basin_number"
            ].astype(int).tolist()
            if n > 0:
                print(f"    {t} — {label:<30}: {n} basin(s)  {basins}")

    # Held-out basin performance note
    held_out = metrics_df[metrics_df.get("basin_role", pd.Series()) == "held_out"]
    if len(held_out) > 0:
        print(f"\n  Held-out basins (in pass-1 training — for reference only):")
        for _, row in held_out.iterrows():
            flag = "FLAG" if row["auto_flag"] else "ok"
            print(f"    Basin {int(row['basin_number']):>6}  "
                  f"{str(row['field_name']):<10}  "
                  f"R²={row['r2']:>+7.3f}  {flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Write outlier_basins.csv
# ─────────────────────────────────────────────────────────────────────────────

def write_outlier_csv(metrics_df: pd.DataFrame) -> None:
    """
    Write auto-flagged basins to outlier_basins.csv.
    This CSV is the single source of truth for outlier basin numbers.
    No outlier basin numbers are hardcoded anywhere else.
    """
    flagged = metrics_df[metrics_df["auto_flag"]].sort_values("r2").copy()

    lines = [
        "# Auto-generated by analysis/basin_analysis.py",
        f"# Criteria: R² < {OUTLIER_R2_THRESHOLD} OR "
        f"rel_RMSE > {OUTLIER_REL_RMSE_THRESHOLD}",
        "#",
        "# Type1 = Low dynamic range",
        "# Type2 = Regime shift",
        "# Type3 = Non-stationary operations",
        "#",
        "# To add manually identified outliers, append rows below.",
        "# Format: basin_number,outlier_type,outlier_type_label,r2,rel_rmse,reason",
        "basin_number,outlier_type,outlier_type_label,r2,rel_rmse,reason",
    ]

    for _, row in flagged.iterrows():
        bn     = int(row["basin_number"])
        otype  = row.get("outlier_type",       "Type3")
        olabel = row.get("outlier_type_label", "Non-stationary operations")
        oreason= row.get("outlier_type_reason","")
        lines.append(
            f"{bn},{otype},{olabel},"
            f"{row['r2']:.3f},{row['rel_rmse']:.3f},"
            f"R²={row['r2']:.3f} | {oreason}"
        )

    OUTLIER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTLIER_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Written: {OUTLIER_CSV}")
    print(f"  {len(flagged)} basins flagged:")
    for _, row in flagged.iterrows():
        role_note = "  [held_out]" if row.get("basin_role") == "held_out" else ""
        print(f"    Basin {int(row['basin_number']):>6}  "
              f"R²={row['r2']:+.3f}  rel_RMSE={row['rel_rmse']:.3f}  "
              f"→ {row.get('outlier_type','?')} "
              f"({row.get('outlier_type_label','')})"
              f"{role_note}")

    print("\n  → Review plots:    outputs/figures/basin_plots/")
    print(  "  → Edit if needed: data/outlier_basins.csv")
    print(  "  → Then run:       python -m models.model1_decay")


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_basin(
    bn:      int,
    bdf:     pd.DataFrame,
    metrics: dict,
    flagged: bool,
) -> None:
    """
    Two-panel diagnostic plot per basin.
    Top:    IRD time series — actual vs predicted, coloured by split
    Bottom: scatter actual vs predicted (val + test)
    Saved to outputs/figures/basin_plots/basin_{bn}.png
    """
    field      = FIELD_NAMES.get(int(str(bn)[0]), str(bn))
    otype      = metrics.get("outlier_type", "")
    olabel     = metrics.get("outlier_type_label", "")
    basin_role = metrics.get("basin_role", "clean")

    if basin_role == "held_out":
        flag_str = "  [HELD-OUT]"
    elif flagged and otype:
        flag_str = f"  ⚠ {otype} — {olabel}"
    else:
        flag_str = ""

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f"Basin {bn}  ({field}){flag_str}\n"
        f"R²={metrics.get('r2', np.nan):+.3f}  "
        f"RMSE={metrics.get('rmse', np.nan):.3f} cm/h  "
        f"rel_RMSE={metrics.get('rel_rmse', np.nan):.3f}  "
        f"MAPE={metrics.get('mape', np.nan):.1f}%  "
        f"n_eval={metrics.get('n', 0)}  role={basin_role}",
        fontsize=10,
    )
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

    # Time series
    ax_ts = fig.add_subplot(gs[0])
    for split in ["train", "val", "test"]:
        sub = bdf[bdf["split"] == split].sort_values("opening_valve_date")
        if sub.empty:
            continue
        valid_act = sub["ird_actual"].notna() & (sub["ird_actual"] > 0)
        ax_ts.scatter(
            sub.loc[valid_act, "opening_valve_date"],
            sub.loc[valid_act, "ird_actual"],
            s=SPLIT_SIZE[split], alpha=SPLIT_ALPHA[split],
            color=SPLIT_COLORS[split], marker=SPLIT_MARKERS[split],
            label=f"{split} actual", zorder=3,
        )
        if split in ["val", "test"]:
            valid_pred = sub["ird_pred"].notna() & (sub["ird_pred"] > 0)
            if valid_pred.any():
                ax_ts.scatter(
                    sub.loc[valid_pred, "opening_valve_date"],
                    sub.loc[valid_pred, "ird_pred"],
                    s=SPLIT_SIZE[split], alpha=SPLIT_ALPHA[split],
                    color=SPLIT_COLORS[split], marker="x",
                    linewidths=1.5, label=f"{split} pred", zorder=4,
                )
                both = valid_pred & valid_act
                for _, row in sub[both].iterrows():
                    ax_ts.plot(
                        [row["opening_valve_date"]] * 2,
                        [row["ird_actual"], row["ird_pred"]],
                        color="gray", linewidth=0.5, alpha=0.3, zorder=2,
                    )

    ax_ts.set_xlabel("Date"); ax_ts.set_ylabel("IRD (cm/h)")
    ax_ts.set_title("Time series — actual vs predicted (pass-1 global model)")
    ax_ts.tick_params(axis="x", rotation=30, labelsize=7)
    ax_ts.legend(fontsize=7, ncol=3); ax_ts.grid(True, alpha=0.2)

    # Scatter
    ax_sc = fig.add_subplot(gs[1])
    eval_df = bdf[
        bdf["split"].isin(["val", "test"]) &
        bdf["ird_actual"].notna() & bdf["ird_pred"].notna() &
        (bdf["ird_actual"] > 0)  & (bdf["ird_pred"] > 0)
    ]
    if len(eval_df) >= 2:
        for split in ["val", "test"]:
            sub = eval_df[eval_df["split"] == split]
            if not sub.empty:
                ax_sc.scatter(
                    sub["ird_actual"], sub["ird_pred"],
                    s=SPLIT_SIZE[split], alpha=0.85,
                    color=SPLIT_COLORS[split],
                    marker=SPLIT_MARKERS[split], label=split, zorder=3,
                )
        all_v = np.concatenate(
            [eval_df["ird_actual"].values, eval_df["ird_pred"].values])
        all_v = all_v[np.isfinite(all_v) & (all_v > 0)]
        if len(all_v):
            lo = np.percentile(all_v, 1) * 0.9
            hi = np.percentile(all_v, 99) * 1.1
            ax_sc.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
            ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
        ax_sc.annotate(
            f"R²={metrics.get('r2', np.nan):+.3f}\n"
            f"RMSE={metrics.get('rmse', np.nan):.3f} cm/h\n"
            f"rel_RMSE={metrics.get('rel_rmse', np.nan):.3f}\n"
            f"MAPE={metrics.get('mape', np.nan):.1f}%\n"
            f"n={metrics.get('n', 0)}",
            xy=(0.05, 0.97), xycoords="axes fraction",
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    ax_sc.set_xlabel("IRD actual (cm/h)"); ax_sc.set_ylabel("IRD predicted (cm/h)")
    ax_sc.set_title("Scatter — val + test")
    ax_sc.legend(fontsize=8); ax_sc.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = BASIN_PLOT_DIR / f"basin_{bn}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_histograms(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Per-basin metric distributions — {len(metrics_df)} basins\n"
        "Pass-1 global LightGBM (all 50 basins)",
        fontsize=11,
    )

    specs = [
        (axes[0], "r2",       "steelblue",    "R²",        OUTLIER_R2_THRESHOLD),
        (axes[1], "rel_rmse", "seagreen",     "rel_RMSE",  OUTLIER_REL_RMSE_THRESHOLD),
        (axes[2], "mape",     "mediumpurple", "MAPE (%)",  None),
    ]

    for ax, col, color, xlabel, threshold in specs:
        vals         = metrics_df[col].dropna().values
        flagged_vals = metrics_df.loc[metrics_df["auto_flag"], col].dropna().values
        ax.hist(vals, bins=20, color=color, edgecolor="white",
                alpha=0.75, label="all basins")
        if len(flagged_vals):
            ax.hist(flagged_vals, bins=20, color="tomato", edgecolor="white",
                    alpha=0.85, label="flagged")
        ax.axvline(np.median(vals), color="black", linewidth=1.5,
                   linestyle="--", label=f"Median={np.median(vals):.3f}")
        if threshold is not None:
            ax.axvline(threshold, color="tomato", linewidth=1.2,
                       linestyle=":", label=f"Threshold={threshold}")
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        ax.set_title(f"{col} distribution")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = FIGURES_DIR / "basin_metric_histograms.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    print("=" * 65)
    print("  BASIN ANALYSIS — analysis/basin_analysis.py")
    print("  Pass-1: global LightGBM on ALL 50 basins")
    print("  Output: outlier_basins.csv — single source of truth")
    print("=" * 65)

    df, feat_cols = load_events()

    train, val, test = get_splits(df, feat_cols, TARGET_M1, split_col="split")

    print("\n--- Training pass-1 global LightGBM ---")
    model, scaler, used_cols = train_lightgbm(
        train, val, feat_cols, TARGET_M1, model2=False
    )

    print("\n--- Computing per-basin metrics ---")
    metrics_df, eval_df = compute_per_basin_metrics(
        df, feat_cols, model, scaler, used_cols
    )

    print("\n--- Classifying outlier types ---")
    metrics_df = classify_outlier_types(metrics_df, df)

    print_metrics_table(metrics_df)

    # Save metrics table
    col_order = [
        "basin_number", "field_name", "basin_role", "n", "n_train",
        "r2", "spearman_r", "rmse", "mae", "mape",
        "rel_rmse", "ird_mean", "ird_std", "auto_flag",
        "outlier_type", "outlier_type_label", "outlier_type_reason",
    ]
    col_order = [c for c in col_order if c in metrics_df.columns]
    metrics_df[col_order].to_excel(BASIN_METRICS_XLSX, index=False)
    print(f"\n  Saved: {BASIN_METRICS_XLSX.name}")

    write_outlier_csv(metrics_df)

    # Per-basin diagnostic plots
    print(f"\n--- Generating per-basin plots ({df['basin_number'].nunique()} basins) ---")
    metrics_lookup = {
        int(row["basin_number"]): row.to_dict()
        for _, row in metrics_df.iterrows()
    }

    for bn in sorted(df["basin_number"].unique()):
        bdf = eval_df[eval_df["basin_number"] == bn].copy()
        train_rows = df[
            (df["basin_number"] == bn) & (df["split"] == "train")
        ].copy()
        train_rows["ird_actual"] = back_transform(
            pd.to_numeric(train_rows["IRD_at_reset"], errors="coerce").values,
            pd.to_numeric(train_rows[TARGET_M1],      errors="coerce").values,
        )
        train_rows["ird_pred"] = np.nan
        bdf = pd.concat([bdf, train_rows], ignore_index=True)
        m   = metrics_lookup.get(int(bn), {})
        plot_basin(int(bn), bdf, m, flagged=m.get("auto_flag", False))

    print(f"  Saved {df['basin_number'].nunique()} plots → {BASIN_PLOT_DIR}")
    plot_metric_histograms(metrics_df)

    print("\n" + "=" * 65)
    print("  NEXT STEPS")
    print("=" * 65)
    print(f"  1. Review: {BASIN_METRICS_XLSX.name}")
    print(f"  2. Review plots: {BASIN_PLOT_DIR}")
    print(f"  3. Edit if needed: {OUTLIER_CSV}")
    print(f"  4. Run python -m experiment.run_bootstrap")
    print(f"  5. Run models: python -m models.model1_decay")
    print(f"                 python -m models.model2_reset")
    print("\nDone.")

    return metrics_df


if __name__ == "__main__":
    main()