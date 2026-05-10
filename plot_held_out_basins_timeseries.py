"""
figures/fig4_shap_beeswarm.py — Figure 4: SHAP beeswarm Model 1 Condition E
=============================================================================
Retrains LightGBM on Condition E (45 basins, all segments, 5 held-out test)
and computes SHAP values on the held-out test set.

Produces a single paper-quality SHAP beeswarm figure styled with plot_style.py.

Runtime: ~5 minutes (LightGBM training + SHAP computation)

Usage
-----
  python fig4_shap_beeswarm.py
  Save manually as PNG (300 DPI) + TIFF when satisfied.

FONT SIZE CONTROL
-----------------
  Edit FONT_OVERRIDE below. All sizes in points.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap

from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT
from config import (
    EVENT_CSV, OUTLIER_CSV, RANDOM_SEED,
    TRAIN_FRAC, VAL_FRAC,
)
from pipeline.features import prepare_features, TARGET_M1

# ── Font size control ─────────────────────────────────────────────────────────
FONT_OVERRIDE = {
    "axis_label"  : 13,
    "tick"        : 12,
    "annotation"  : 11,
    "legend"      : 11,
    "title"       : 13,
}

def _fs(key):
    return FONT_OVERRIDE.get(key, FONT.get(key, 11))

# ── Feature display names ─────────────────────────────────────────────────────
# Maps raw feature names → readable labels for the y-axis
FEATURE_LABELS = {
    "prev_ALPHA"        : "Prev. drying fraction (ALPHA)",
    "IRD_at_reset"      : "Post-tillage IRD (ρᵢ)",
    "prev_DrT"          : "Prev. drying time (DrT, h)",
    "log1p_prev_HL"     : "Prev. hydraulic load (log HL)",
    "LCT"               : "Time since tillage (LCT, h)",
    "prev_TD"           : "Prev. drying temperature (°C)",
    "prev_FT"           : "Prev. flooding time (FT, h)",
    "cum_TW"            : "Cumul. wetting temperature (°C·h)",
    "cum_FT"            : "Cumul. flooding time (h)",
    "prev_RD"           : "Prev. drying radiation (W/m²)",
    "prev_RW"           : "Prev. wetting radiation (W/m²)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load and prepare Condition E
# ─────────────────────────────────────────────────────────────────────────────

def load_outlier_basins() -> set[int]:
    if not OUTLIER_CSV.exists():
        print("  WARNING: outlier_basins.csv not found — no exclusions")
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


def _reassign_splits(df: pd.DataFrame) -> pd.Series:
    """Random 70/15/15 splits by segment — matches model_comparison.py exactly."""
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


def load_condition_e() -> tuple[pd.DataFrame, list[str]]:
    """
    Reconstruct Condition E exactly as model_comparison.py does:
    - 45 basins (all except 5 held-out)
    - All segments (no is_good_segment filter)
    - split_held_out for held-out basins, reassigned splits for others
    """
    print("  Loading event dataset...")
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    outlier_basins = load_outlier_basins()

    all_basins = set(df["basin_number"].dropna().unique().astype(int))
    held_out   = set(
        df.loc[df["basin_role"] == "held_out", "basin_number"]
        .dropna().unique().astype(int)
    ) if "basin_role" in df.columns else set()

    train_basins = all_basins - held_out   # 45 basins
    all_e        = train_basins | held_out

    print(f"  Condition E: {len(train_basins)} train basins, "
          f"{len(held_out)} held-out test basins")

    df_e = df[
        (df["basin_number"].isin(all_e)) &
        (df["row_type"] == "event")
    ].copy()

    if "split_held_out" not in df_e.columns:
        raise ValueError("split_held_out column missing — rebuild from V2 build_dataset.py")

    df_e["split"] = df_e["split_held_out"].replace({"held_out_test": "test"})

    # Reassign splits for rows without a valid split assignment
    needs_split = df_e["split"].isin(["excluded", ""]) | df_e["split"].isna()
    df_e.loc[needs_split, "split"] = _reassign_splits(df_e[needs_split]).values

    df_e = df_e[df_e["split"].isin(["train", "val", "test"])].copy()
    df_e, feat_cols = prepare_features(df_e)

    n_tr = int((df_e["split"] == "train").sum())
    n_va = int((df_e["split"] == "val").sum())
    n_te = int((df_e["split"] == "test").sum())
    print(f"  Events: train={n_tr}  val={n_va}  test={n_te}")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")

    return df_e, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train LightGBM Condition E
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(
    df: pd.DataFrame, feat_cols: list[str]
):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.preprocessing import StandardScaler

    train = df[df["split"] == "train"].dropna(
        subset=feat_cols + [TARGET_M1]).reset_index(drop=True)
    val   = df[df["split"] == "val"].dropna(
        subset=feat_cols + [TARGET_M1]).reset_index(drop=True)
    test  = df[df["split"] == "test"].dropna(
        subset=feat_cols + [TARGET_M1]).reset_index(drop=True)

    print(f"\n  Training LightGBM Condition E...")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)

    model = LGBMRegressor(
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
    model.fit(
        Xtr, train[TARGET_M1].values,
        eval_set=[(Xva, val[TARGET_M1].values)],
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"  best_iter={model.best_iteration_}  "
          f"train_n={len(train)}  test_n={len(test)}")

    return model, sc, Xte, test, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Compute SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model, Xte: np.ndarray, feat_cols: list[str]):
    print("\n  Computing SHAP values (this takes ~2 min)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xte)

    # Subsample for plot readability
    n_sample = min(3000, len(Xte))
    rng      = np.random.default_rng(RANDOM_SEED)
    idx      = rng.choice(len(Xte), n_sample, replace=False)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]   # descending

    print(f"\n  SHAP feature importance — Condition E (n={len(Xte)}):")
    print(f"  {'Rank':<5}  {'Feature':<35}  {'Mean |SHAP|':>12}  Direction")
    print(f"  {'-'*65}")
    for rank, i in enumerate(order):
        direction = "→ less decay" if shap_values[:, i].mean() > 0 \
                    else "→ more decay"
        label = FEATURE_LABELS.get(feat_cols[i], feat_cols[i])
        print(f"  {rank+1:<5}  {label:<35}  "
              f"{mean_abs[i]:>12.4f}  {direction}")

    return shap_values, idx, order


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Paper-quality beeswarm figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_beeswarm(
    shap_values: np.ndarray,
    Xte:         np.ndarray,
    feat_cols:   list[str],
    idx:         np.ndarray,
    order:       np.ndarray,
) -> None:
    """
    Paper-quality SHAP beeswarm.
    Uses shap.summary_plot() with show=False, then applies plot_style.py styling.
    Features sorted by mean |SHAP|, y-axis shows readable labels.
    Colorbar: feature value (blue=low, red=high).
    """
    apply_style()

    # Build display names in sorted order
    feat_labels_sorted = [
        FEATURE_LABELS.get(feat_cols[i], feat_cols[i])
        for i in order
    ]

    # SHAP data for plotting — subsample, sorted feature order
    shap_plot  = shap_values[idx][:, order]
    x_plot     = Xte[idx][:, order]
    feat_names = feat_labels_sorted

    # ── Call shap.summary_plot with show=False ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    shap.summary_plot(
        shap_plot,
        x_plot,
        feature_names=feat_names,
        show=False,
        plot_size=None,
        color_bar=False,   # we add our own colorbar
        plot_type="dot",
        max_display=len(feat_cols),
        alpha=0.4,
    )

    # ── Restyle the axes shap produced ───────────────────────────────────────
    ax = plt.gca()
    ax.set_xlabel(
        "SHAP value  (impact on IRD decay prediction)",
        fontsize=_fs("axis_label"),
    )
    ax.tick_params(axis="y", labelsize=_fs("tick"))
    ax.tick_params(axis="x", labelsize=_fs("tick"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.20, linestyle="--", linewidth=0.6)
    ax.axvline(0, color=COLORS["dark_gray"], linewidth=0.8,
               linestyle="-", alpha=0.5)

    # ── Direction annotation ──────────────────────────────────────────────────
    ax.annotate(
        "← more decay",
        xy=(0.02, -0.09), xycoords="axes fraction",
        fontsize=_fs("annotation") - 1,
        color=COLORS["teal"], ha="left",
    )
    ax.annotate(
        "less decay →",
        xy=(0.98, -0.09), xycoords="axes fraction",
        fontsize=_fs("annotation") - 1,
        color=COLORS["orange"], ha="right",
    )

    # ── Colorbar — feature value ──────────────────────────────────────────────
    cmap = plt.cm.RdBu_r
    sm   = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.set_label("Feature value", fontsize=_fs("legend"))
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"], fontsize=_fs("legend"))

    # ── Top-4 bracket annotation ──────────────────────────────────────────────
    # Highlight that top 4 are stable across conditions
    n_feat = len(feat_cols)
    ax.annotate(
        "Top 4: stable\nacross all conditions",
        xy=(ax.get_xlim()[1] * 0.98, n_feat - 4.5),
        fontsize=_fs("annotation") - 2,
        va="center", ha="right",
        color=COLORS["mid_gray"],
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=COLORS["near_white"],
                  edgecolor=COLORS["light_gray"],
                  alpha=0.9),
    )
    # Bracket line
    ax.annotate(
        "",
        xy=(ax.get_xlim()[1] * 0.96, n_feat - 1.0),
        xytext=(ax.get_xlim()[1] * 0.96, n_feat - 4.0),
        arrowprops=dict(
            arrowstyle="-",
            color=COLORS["mid_gray"],
            lw=1.2,
        ),
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Caption
# ─────────────────────────────────────────────────────────────────────────────

CAPTION = (
    "Figure 4. SHAP feature importance for Model 1 "
    "(LightGBM, Condition E: 45 basins including operational outliers, "
    "all segments, tested on 5 held-out basins). "
    "Each dot represents one flooding–drying event from the held-out test set "
    "(n = 5 unseen basins). "
    "Color indicates the feature value (red = high, blue = low). "
    "The x-axis shows the SHAP value: positive values push the prediction "
    "toward slower decay (higher η); negative values push toward faster decay. "
    "Features are ranked by mean absolute SHAP value. "
    "The four dominant features — previous drying fraction (ALPHA), "
    "post-tillage IRD (ρᵢ), previous drying time (DrT), and log-hydraulic load — "
    "are consistent across all evaluation conditions (Conditions A, D, E), "
    "demonstrating that the physical interpretation is robust to data quality choices. "
    "The operational implication is direct: increasing the drying fraction "
    "(more drying relative to cycle time) is the strongest lever available "
    "to field managers for slowing within-segment IRD decay."
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FIGURE 4 — SHAP Beeswarm  (Model 1, Condition E)")
    print("=" * 60)

    # Load and prepare
    df_e, feat_cols = load_condition_e()

    # Train
    model, sc, Xte, test_df, feat_cols = train_lgbm(df_e, feat_cols)

    # SHAP
    shap_values, idx, order = compute_shap(model, Xte, feat_cols)

    # Figure
    print("\n  Rendering figure...")
    print("  → Save manually: PNG (300 DPI) + TIFF\n")
    plot_beeswarm(shap_values, Xte, feat_cols, idx, order)

    print("\n" + "─" * 60)
    print("CAPTION (copy to PPT / paper):")
    print("─" * 60)
    print(CAPTION)
    print("─" * 60)


if __name__ == "__main__":
    main()