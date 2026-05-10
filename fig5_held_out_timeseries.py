"""
figures/fig5_held_out_timeseries.py — Figure 5: Model 1 held-out time series
=============================================================================
Condition E: trained on 45 basins (all segments including outliers),
tested on 5 completely unseen held-out basins.

Layout: 5 rows × 1 column — one panel per held-out basin.
Each panel:
  - Season bands background (10-year record)
  - Actual IRD (filled circles, colored by field)
  - Predicted IRD (crosses, same color)
  - Thin grey connecting lines between actual and predicted
  - Vertical dashed lines at segment resets
  - Per-basin metrics annotated

Runtime: ~5 minutes (retrains LightGBM Condition E)

Usage
-----
  python fig5_held_out_timeseries.py
  Save manually as PNG (300 DPI) + TIFF.

FONT SIZE CONTROL
-----------------
  Edit FONT_OVERRIDE below.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT, FIELD_COLORS, add_season_bands
from config import EVENT_CSV, OUTLIER_CSV, RANDOM_SEED, TRAIN_FRAC, VAL_FRAC, FIELD_NAMES
from pipeline.features import prepare_features, TARGET_M1
from models.utils import back_transform, metrics_ird, predict

# ── Config ────────────────────────────────────────────────────────────────────
HELD_OUT_BASINS = [3203, 4104, 5102, 6303, 7201]

FONT_OVERRIDE = {
    "title"      : 11,
    "axis_label" : 10,
    "tick"       : 9,
    "legend"     : 9,
    "annotation" : 9,
}

SEASON_ALPHA   = 0.60
SCATTER_SIZE   = 10
SCATTER_ALPHA  = 0.70
RESET_LINE_KW  = dict(color="black", linewidth=0.5,
                      linestyle="--", alpha=0.20, zorder=1)

def _fs(key):
    return FONT_OVERRIDE.get(key, FONT.get(key, 11))

# ─────────────────────────────────────────────────────────────────────────────
# Load and train — identical to fig4, Condition E
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
    split_col = pd.Series("excluded", index=df.index)
    seg_ids   = sorted([int(s) for s in df["segment_id"].dropna().unique()
                        if s >= 0])
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


def load_and_train():
    """Load full dataset, train Condition E, return model + full event data."""
    print("  Loading event dataset...")
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_basins = set(df["basin_number"].dropna().unique().astype(int))
    held_out   = set(
        df.loc[df["basin_role"] == "held_out", "basin_number"]
        .dropna().unique().astype(int)
    ) if "basin_role" in df.columns else set()
    train_basins = all_basins - held_out
    all_e        = train_basins | held_out

    print(f"  Condition E: {len(train_basins)} train, {len(held_out)} held-out")

    df_e = df[
        (df["basin_number"].isin(all_e)) &
        (df["row_type"] == "event")
    ].copy()

    df_e["split"] = df_e["split_held_out"].replace({"held_out_test": "test"})
    needs_split   = df_e["split"].isin(["excluded", ""]) | df_e["split"].isna()
    df_e.loc[needs_split, "split"] = _reassign_splits(df_e[needs_split]).values
    df_e = df_e[df_e["split"].isin(["train", "val", "test"])].copy()
    df_e, feat_cols = prepare_features(df_e)

    # Train
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.preprocessing import StandardScaler

    train = df_e[df_e["split"] == "train"].dropna(
        subset=feat_cols + [TARGET_M1]).reset_index(drop=True)
    val   = df_e[df_e["split"] == "val"].dropna(
        subset=feat_cols + [TARGET_M1]).reset_index(drop=True)

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xva = sc.transform(val[feat_cols].values)

    print("  Training LightGBM Condition E...")
    model = LGBMRegressor(
        n_estimators=1000, max_depth=-1, num_leaves=63,
        learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(
        Xtr, train[TARGET_M1].values,
        eval_set=[(Xva, val[TARGET_M1].values)],
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    print(f"  best_iter={model.best_iteration_}")

    return model, sc, feat_cols, df


# ─────────────────────────────────────────────────────────────────────────────
# Prepare held-out predictions
# ─────────────────────────────────────────────────────────────────────────────

def prepare_held_out(model, sc, feat_cols, df_full):
    """Get actual + predicted IRD for all held-out basin events."""
    ho = df_full[
        (df_full["basin_number"].isin(HELD_OUT_BASINS)) &
        (df_full["row_type"] == "event")
    ].copy()

    ho, _ = prepare_features(ho)

    ird_reset      = pd.to_numeric(ho["IRD_at_reset"], errors="coerce").values
    pred_norm      = predict(model, sc, feat_cols, ho)
    ho["ird_pred"] = back_transform(ird_reset, pred_norm)
    ho["ird_true"] = back_transform(
        ird_reset,
        pd.to_numeric(ho[TARGET_M1], errors="coerce").values,
    )

    print("\n  Held-out basin metrics (Condition E — ALL segments):")
    print(f"  {'Basin':<8} {'Field':<12} {'n':>6} "
          f"{'R²':>8} {'RMSE':>8} {'MAPE%':>7}")
    print(f"  {'-'*55}")

    all_true, all_pred = [], []
    for bn in HELD_OUT_BASINS:
        bdf   = ho[ho["basin_number"] == bn]
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        all_true.append(ird_t[mask])
        all_pred.append(ird_p[mask])
        m     = metrics_ird(ird_t[mask], ird_p[mask], verbose=False) or {}
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        print(f"  {bn:<8} {field:<12} {m.get('n',0):>6} "
              f"{m.get('r2',np.nan):>+8.3f} "
              f"{m.get('rmse',np.nan):>8.3f} "
              f"{m.get('mape',np.nan):>6.1f}%")

    # Pooled
    ird_t_all = np.concatenate(all_true)
    ird_p_all = np.concatenate(all_pred)
    m_all = metrics_ird(ird_t_all, ird_p_all, verbose=False) or {}
    print(f"  {'POOLED':<8} {'all':<12} {m_all.get('n',0):>6} "
          f"{m_all.get('r2',np.nan):>+8.3f} "
          f"{m_all.get('rmse',np.nan):>8.3f} "
          f"{m_all.get('mape',np.nan):>6.1f}%")

    return ho


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(ho: pd.DataFrame) -> None:
    apply_style()

    n_panels = len(HELD_OUT_BASINS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 4.2 * n_panels),
        gridspec_kw={"hspace": 0.45},
    )

    for ax, bn in zip(axes, HELD_OUT_BASINS):
        bdf   = ho[ho["basin_number"] == bn].sort_values("opening_valve_date")
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        color = FIELD_COLORS.get(field, COLORS["deep_blue"])

        # ── Season bands ──────────────────────────────────────────────────────
        import plot_style as _ps
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, SEASON_ALPHA)
        add_season_bands(ax,
                         bdf["opening_valve_date"].min(),
                         bdf["opening_valve_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        # ── Reset lines ───────────────────────────────────────────────────────
        segs = sorted(bdf["segment_id"].dropna().unique())
        reset_dates = []
        for sid in segs:
            seg = bdf[bdf["segment_id"] == sid].sort_values("opening_valve_date")
            if not seg.empty:
                reset_dates.append(seg["opening_valve_date"].iloc[0])

        for rd in reset_dates:
            ax.axvline(rd, **RESET_LINE_KW)

        # ── Actual IRD — blue circles ─────────────────────────────────────────
        valid_true = bdf["ird_true"].notna() & (bdf["ird_true"] > 0)
        ax.scatter(
            bdf.loc[valid_true, "opening_valve_date"],
            bdf.loc[valid_true, "ird_true"],
            s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
            color=COLORS["deep_blue"], marker="o", zorder=4,
            label="Actual IRD",
        )

        # ── Predicted IRD — green crosses ─────────────────────────────────────
        valid_pred = bdf["ird_pred"].notna() & (bdf["ird_pred"] > 0)
        ax.scatter(
            bdf.loc[valid_pred, "opening_valve_date"],
            bdf.loc[valid_pred, "ird_pred"],
            s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
            color=COLORS["green"], marker="x", linewidths=1.2, zorder=5,
            label="Predicted IRD",
        )

        # ── Connecting lines actual → predicted ───────────────────────────────
        both = valid_true & valid_pred
        for _, row in bdf[both].iterrows():
            ax.plot(
                [row["opening_valve_date"]] * 2,
                [row["ird_true"], row["ird_pred"]],
                color="gray", linewidth=0.35, alpha=0.25, zorder=2,
            )

        # ── Metrics — computed for caption, not plotted ───────────────────────
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        m     = metrics_ird(ird_t[mask], ird_p[mask], verbose=False) or {}

        # ── Panel label — basin number only on y-axis ────────────────────────
        is_last   = (bn == HELD_OUT_BASINS[-1])
        is_mid    = (bn == HELD_OUT_BASINS[len(HELD_OUT_BASINS) // 2])
        panel_idx = HELD_OUT_BASINS.index(bn)

        if is_mid:
            ax.set_ylabel(f"Basin {bn}\nIRD (cm/h)", fontsize=_fs("axis_label"))
        else:
            ax.set_ylabel(f"Basin {bn}", fontsize=_fs("axis_label"))

        ax.tick_params(axis="both", labelsize=_fs("tick"))
        ax.tick_params(axis="x", rotation=25)

        # No x label on any panel
        if not is_last:
            ax.set_xticklabels([])
        if panel_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color=COLORS["deep_blue"],
                       markersize=5, linestyle="None",
                       label="Actual IRD"),
                Line2D([0], [0], marker="x", color=COLORS["green"],
                       markersize=5, linestyle="None",
                       markeredgewidth=1.2, label="Predicted IRD"),
                Line2D([0], [0], color="black", linewidth=0.7,
                       linestyle="--", alpha=0.4,
                       label="Segment reset"),
            ]
            ax.legend(
                handles=legend_elements,
                fontsize=_fs("legend"),
                loc="upper right", ncol=3,
                framealpha=0.85,
            )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Caption
# ─────────────────────────────────────────────────────────────────────────────

CAPTION = (
    "Figure 5. Model 1 generalizability to completely unseen basins "
    "(Condition E: trained on 45 basins including operational outliers, "
    "all segments without quality filtering). "
    "Each panel shows the actual (circles) and predicted (crosses) IRD (cm/h) "
    "time series for one of five held-out basins excluded from all training, "
    "validation, and feature selection. "
    "Thin grey lines connect each actual–predicted pair. "
    "Dashed vertical lines mark segment reset events (tillage). "
    "Seasonal background shading indicates winter (blue), spring (peach), "
    "summer (yellow), and autumn (green). "
    "The model was applied without any site-specific recalibration. "
    "Per-basin metrics — "
    "Basin 3203 (Soreq 2): R²=+0.707, RMSE=0.884 cm/h, MAPE=12.4%; "
    "Basin 4104 (Yavne 1): R²=+0.718, RMSE=0.208 cm/h, MAPE=12.0%; "
    "Basin 5102 (Yavne 2): R²=+0.725, RMSE=0.550 cm/h, MAPE=14.1%; "
    "Basin 6303 (Yavne 3): R²=+0.731, RMSE=0.423 cm/h, MAPE=13.3%; "
    "Basin 7201 (Yavne 4): R²=+0.862, RMSE=0.139 cm/h, MAPE=14.2%. "
    "Pooled across all five basins: R²=+0.898, RMSE=0.607 cm/h, MAPE=13.1% "
    "(n=4,386 flooding-drying events)."
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FIGURE 5 — Held-out Time Series  (Model 1, Condition E)")
    print("=" * 60)

    model, sc, feat_cols, df_full = load_and_train()
    ho = prepare_held_out(model, sc, feat_cols, df_full)

    print("\n  Rendering figure...")
    print("  → Save manually: PNG (300 DPI) + TIFF\n")
    plot_figure(ho)

    print("\n" + "─" * 60)
    print("CAPTION (copy to PPT / paper):")
    print("─" * 60)
    print(CAPTION)
    print("─" * 60)


if __name__ == "__main__":
    main()