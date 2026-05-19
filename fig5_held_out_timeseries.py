"""
figures/fig5_held_out_timeseries.py — Figure 5: Model 1 held-out time series + scatter
========================================================================================
Condition E: trained on 45 basins (all segments including outliers),
tested on 5 completely unseen held-out basins.

Layout: 5 rows × 3 columns
  Col 0 (wide)   : IRD time series — actual vs predicted, season bands, reset lines
  Col 1 (medium) : Actual vs predicted scatter with 1:1 line
  Col 2 (narrow) : Per-basin metrics text — right of scatter, no axis

TUNING
------
  All layout, spacing, and size parameters live in the LAYOUT dict below.
  Edit only that dict — nothing else needs changing for appearance.

Runtime: ~5 minutes (retrains LightGBM Condition E)

Usage
-----
  python fig5_held_out_timeseries.py
  Save manually as PNG (300 DPI) + TIFF.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT, FIELD_COLORS, add_season_bands
from config import EVENT_CSV, OUTLIER_CSV, RANDOM_SEED, TRAIN_FRAC, VAL_FRAC, FIELD_NAMES
from pipeline.features import prepare_features, TARGET_M1
from models.utils import back_transform, metrics_ird, predict

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  TUNING PANEL — edit everything here, nowhere else  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════
LAYOUT = dict(

    # ── Figure dimensions ─────────────────────────────────────────────────────
    fig_width   = 18,    # total figure width (inches)
    row_height  = 3.2,   # height of each basin row (inches) — reduce to compress

    # ── Column width ratios: [timeseries, scatter, right_margin_for_text] ─────
    # Increase col[1] to widen scatter; col[2] is blank space for fig.text()
    col_ratios  = [3.5, 1.6, 0.7],

    # ── Spacing ───────────────────────────────────────────────────────────────
    hspace      = 0.18,   # vertical gap between rows — main fix for "too much space"
    wspace      = 0.28,   # horizontal gap between columns

    # ── Scatter axis padding (fraction of data range added each side) ─────────
    scatter_pad = 0.08,

    # ── Font sizes ────────────────────────────────────────────────────────────
    fs_title    = 11,
    fs_ylabel   = 10,
    fs_xlabel   = 10,
    fs_tick     = 9,
    fs_legend   = 9,
    fs_metrics  = 8.5,   # right-column metrics text

    # ── Time series markers ───────────────────────────────────────────────────
    ts_s        = 10,    # marker size
    ts_alpha    = 0.70,

    # ── Scatter markers ───────────────────────────────────────────────────────
    sc_s        = 14,    # marker size
    sc_alpha    = 0.65,

    # ── Season bands ──────────────────────────────────────────────────────────
    season_alpha = 0.8,

    # ── Reset vertical lines ──────────────────────────────────────────────────
    reset_lw    = 0.5,
    reset_alpha = 0.20,

    # ── Connecting lines actual→predicted in time series ─────────────────────
    connector_lw    = 0.35,
    connector_alpha = 0.25,

    # ── Metrics text position ─────────────────────────────────────────────────
    # Horizontal: fraction of figure width to add RIGHT of the scatter bbox
    metrics_gap  = 0.012,
    # Vertical: fraction from row top (0=top of row, 1=bottom)
    metrics_vtop = 0.82,
    # Line spacing multiplier for the metrics text block
    metrics_ls   = 1.6,
)
# ▲▲▲  end of tuning panel  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════════

HELD_OUT_BASINS = [3203, 4104, 5102, 6303, 7201]
L = LAYOUT   # short alias throughout


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & training  (logic unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _reassign_splits(df: pd.DataFrame) -> pd.Series:
    seg_ids  = sorted([int(s) for s in df["segment_id"].dropna().unique() if s >= 0])
    if len(seg_ids) < 5:
        return pd.Series("excluded", index=df.index)
    rng      = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(seg_ids)
    n_tr     = max(1, round(len(shuffled) * TRAIN_FRAC))
    n_va     = max(1, round(len(shuffled) * VAL_FRAC))
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
    print("  Loading event dataset...")
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if TARGET_M1 not in df.columns and "IRD_norm" in df.columns:
        df[TARGET_M1] = df["IRD_norm"]
    for col in ["is_good_segment", "segment_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_basins   = set(df["basin_number"].dropna().unique().astype(int))
    held_out     = (set(df.loc[df["basin_role"] == "held_out", "basin_number"]
                        .dropna().unique().astype(int))
                    if "basin_role" in df.columns else set())
    train_basins = all_basins - held_out

    print(f"  Condition E: {len(train_basins)} train, {len(held_out)} held-out")

    df_e = df[
        (df["basin_number"].isin(train_basins | held_out)) &
        (df["row_type"] == "event")
    ].copy()

    df_e["split"] = df_e["split_held_out"].replace({"held_out_test": "test"})
    needs = df_e["split"].isin(["excluded", ""]) | df_e["split"].isna()
    df_e.loc[needs, "split"] = _reassign_splits(df_e[needs]).values
    df_e = df_e[df_e["split"].isin(["train", "val", "test"])].copy()
    df_e, feat_cols = prepare_features(df_e)

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
        callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
    )
    print(f"  best_iter={model.best_iteration_}")
    return model, sc, feat_cols, df


def prepare_held_out(model, sc, feat_cols, df_full):
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
    print(f"  {'Basin':<8} {'Field':<12} {'n':>6} {'R²':>8} {'RMSE':>8} {'MAPE%':>7}")
    print(f"  {'-'*55}")

    basin_metrics = {}
    all_true, all_pred = [], []
    for bn in HELD_OUT_BASINS:
        bdf   = ho[ho["basin_number"] == bn]
        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        all_true.append(ird_t[mask]);  all_pred.append(ird_p[mask])
        m     = metrics_ird(ird_t[mask], ird_p[mask], verbose=False) or {}
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        basin_metrics[bn] = {**m, "field": field}
        print(f"  {bn:<8} {field:<12} {m.get('n',0):>6} "
              f"{m.get('r2',np.nan):>+8.3f} "
              f"{m.get('rmse',np.nan):>8.3f} "
              f"{m.get('mape',np.nan):>6.1f}%")

    ird_t_all = np.concatenate(all_true)
    ird_p_all = np.concatenate(all_pred)
    m_all = metrics_ird(ird_t_all, ird_p_all, verbose=False) or {}
    basin_metrics["POOLED"] = {**m_all, "field": "all"}
    print(f"  {'POOLED':<8} {'all':<12} {m_all.get('n',0):>6} "
          f"{m_all.get('r2',np.nan):>+8.3f} "
          f"{m_all.get('rmse',np.nan):>8.3f} "
          f"{m_all.get('mape',np.nan):>6.1f}%")
    return ho, basin_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(ho: pd.DataFrame, basin_metrics: dict) -> None:
    apply_style()

    n   = len(HELD_OUT_BASINS)
    fig = plt.figure(figsize=(L["fig_width"], L["row_height"] * n))

    # 3-column GridSpec: timeseries | scatter | blank (for fig.text metrics)
    gs = GridSpec(
        n, 3,
        figure=fig,
        width_ratios=L["col_ratios"],
        hspace=L["hspace"],
        wspace=L["wspace"],
    )

    for row_idx, bn in enumerate(HELD_OUT_BASINS):
        ax_ts = fig.add_subplot(gs[row_idx, 0])
        ax_sc = fig.add_subplot(gs[row_idx, 1])
        # gs[row_idx, 2] intentionally unused as an axis

        bdf    = ho[ho["basin_number"] == bn].sort_values("opening_valve_date")
        field  = FIELD_NAMES.get(int(str(bn)[0]), "")
        color  = FIELD_COLORS.get(field, COLORS["deep_blue"])
        is_last = (row_idx == n - 1)
        is_mid  = (row_idx == n // 2)

        ird_t = bdf["ird_true"].values.astype(float)
        ird_p = bdf["ird_pred"].values.astype(float)
        mask  = np.isfinite(ird_t) & np.isfinite(ird_p) & (ird_t > 0) & (ird_p > 0)
        m     = basin_metrics.get(bn, {})

        # ── TIME SERIES ───────────────────────────────────────────────────────
        import plot_style as _ps
        _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
        for k, (col, _) in _orig.items():
            _ps.SEASON_COLORS[k] = (col, L["season_alpha"])
        add_season_bands(ax_ts,
                         bdf["opening_valve_date"].min(),
                         bdf["opening_valve_date"].max())
        for k, v in _orig.items():
            _ps.SEASON_COLORS[k] = v

        # Reset lines
        for sid in sorted(bdf["segment_id"].dropna().unique()):
            seg = bdf[bdf["segment_id"] == sid].sort_values("opening_valve_date")
            if not seg.empty:
                ax_ts.axvline(seg["opening_valve_date"].iloc[0],
                              color="black", linewidth=L["reset_lw"],
                              linestyle="--", alpha=L["reset_alpha"], zorder=1)

        valid_t = bdf["ird_true"].notna() & (bdf["ird_true"] > 0)
        valid_p = bdf["ird_pred"].notna() & (bdf["ird_pred"] > 0)
        both    = valid_t & valid_p

        ax_ts.scatter(bdf.loc[valid_t, "opening_valve_date"],
                      bdf.loc[valid_t, "ird_true"],
                      s=L["ts_s"], alpha=L["ts_alpha"],
                      color=COLORS["deep_blue"], marker="o", zorder=4)
        ax_ts.scatter(bdf.loc[valid_p, "opening_valve_date"],
                      bdf.loc[valid_p, "ird_pred"],
                      s=L["ts_s"], alpha=L["ts_alpha"],
                      color=COLORS["orange"], marker="x",
                      linewidths=1.2, zorder=5)

        for _, row in bdf[both].iterrows():
            ax_ts.plot([row["opening_valve_date"]] * 2,
                       [row["ird_true"], row["ird_pred"]],
                       color="gray", linewidth=L["connector_lw"],
                       alpha=L["connector_alpha"], zorder=2)

        if is_mid:
            ax_ts.set_ylabel(f"IRD (cm/h) \n {bn}",
                             fontsize=L["fs_ylabel"])
        else:
            ax_ts.set_ylabel(f" {bn} ", fontsize=L["fs_ylabel"])

        ax_ts.tick_params(axis="both", labelsize=L["fs_tick"])
        ax_ts.tick_params(axis="x", rotation=25)
        if not is_last:
            ax_ts.set_xticklabels([])
        if is_last:
            ax_ts.set_xlabel("Date", fontsize=L["fs_xlabel"])

        if row_idx == 0:
            ax_ts.legend(
                handles=[
                    Line2D([0], [0], marker="o", color=COLORS["deep_blue"],
                           markersize=5, linestyle="None", label="Actual IRD"),
                    Line2D([0], [0], marker="x", color=COLORS["orange"],
                           markersize=5, linestyle="None",
                           markeredgewidth=1.2, label="Predicted IRD"),
                    Line2D([0], [0], color="black", linewidth=0.7,
                           linestyle="--", alpha=0.4, label="Tillage reset"),
                ],
                fontsize=L["fs_legend"], loc="upper right",
                ncol=3, framealpha=0.85,
            )
            ax_ts.set_title("IRD Time Series",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

        # ── SCATTER ───────────────────────────────────────────────────────────
        yt = ird_t[mask];  yp = ird_p[mask]
        pad = L["scatter_pad"]
        lo  = min(yt.min(), yp.min());  hi = max(yt.max(), yp.max())
        rng = hi - lo
        lim_lo = lo - pad * rng;  lim_hi = hi + pad * rng

        ax_sc.scatter(yt, yp,
                      s=L["sc_s"], alpha=L["sc_alpha"],
                      color=color, edgecolors="none", zorder=3)
        ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                   color="black", linewidth=0.8, linestyle="--",
                   alpha=0.6, zorder=2)
        ax_sc.set_xlim(lim_lo, lim_hi)
        ax_sc.set_ylim(lim_lo, lim_hi)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.tick_params(axis="both", labelsize=L["fs_tick"])

        if is_last:
            ax_sc.set_xlabel("Actual IRD (cm/h)", fontsize=L["fs_xlabel"])
        if is_mid:
            ax_sc.set_ylabel("Predicted (cm/h)", fontsize=L["fs_ylabel"])
        if row_idx == 0:
            ax_sc.set_title("Actual vs Predicted",
                            fontsize=L["fs_title"], fontweight="bold", pad=4)

        # ── METRICS TEXT — right of scatter using fig.text() ──────────────────
        # Must draw first to get accurate axis positions in figure coordinates
        fig.canvas.draw()
        bb     = ax_sc.get_position()          # in figure fraction
        txt_x  = bb.x1 + L["metrics_gap"]     # just right of scatter
        txt_y  = bb.y0 + bb.height * L["metrics_vtop"]

        r2   = m.get("r2",   np.nan)
        rmse = m.get("rmse", np.nan)
        mape = m.get("mape", np.nan)
        n_ev = m.get("n",    0)

        fig.text(
            txt_x, txt_y,
            f"R²  = {r2:+.3f}\n"
            f"RMSE = {rmse:.3f} cm/h\n"
            f"MAPE = {mape:.1f}%\n"
            f"n = {n_ev:,}",
            fontsize=L["fs_metrics"],
            verticalalignment="top",
            linespacing=L["metrics_ls"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.88, edgecolor="lightgrey", linewidth=0.6),
        )

    # Leave right margin so metrics text is not clipped
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Caption template  (update XX from console output after running)
# ─────────────────────────────────────────────────────────────────────────────
CAPTION = (
    "Figure 5. Model 1 generalizability to completely unseen basins "
    "(Condition E: trained on 45 basins including operational outliers, "
    "all segments without quality filtering). "
    "Left: actual (circles) and predicted (crosses) IRD (cm/h) time series "
    "for each held-out basin; dashed vertical lines mark tillage events; "
    "seasonal shading: winter (blue), spring (peach), summer (yellow), autumn (green). "
    "Centre: actual vs predicted scatter; dashed line = 1:1. "
    "Right: R², RMSE and MAPE per basin. "
    "No site-specific recalibration was applied. "
    "Per-basin metrics (update after running): "
    "Basin 3203 (Soreq 2): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 4104 (Yavne 1): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 5102 (Yavne 2): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 6303 (Yavne 3): R²=XX, RMSE=XX cm/h, MAPE=XX%; "
    "Basin 7201 (Yavne 4): R²=XX, RMSE=XX cm/h, MAPE=XX%. "
    "Pooled: R²=XX, RMSE=XX cm/h, MAPE=XX% (n=XX events)."
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FIGURE 5 — Model 1 | Time Series + Scatter  (Condition E)")
    print("=" * 60)
    print("  Tuning: edit the LAYOUT dict at the top of this file.\n")

    model, sc, feat_cols, df_full = load_and_train()
    ho, basin_metrics = prepare_held_out(model, sc, feat_cols, df_full)

    print("\n  Rendering figure...")
    plot_figure(ho, basin_metrics)

    print("\n" + "─" * 60)
    print("CAPTION TEMPLATE (replace XX with console values above):")
    print("─" * 60)
    print(CAPTION)


if __name__ == "__main__":
    main()