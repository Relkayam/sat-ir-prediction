"""
figures/fig2_decay_evidence.py — Figure 2 (main) + Figure Sx (SI)
==================================================================
One file produces BOTH figures by toggling SHOW_ALL_SEGMENTS.

SHOW_ALL_SEGMENTS = False  →  Figure 2  (main paper)
    Only good segments (is_good_segment == True, ~48% of events).
    Confirms the decay hypothesis on clean data.

SHOW_ALL_SEGMENTS = True   →  Figure Sx (SI)
    All segments including those with no clear decay signal.
    Shows why the hypothesis is not always visually obvious in raw data,
    and sets up the narrative that the model works even on this full dataset.

Layout: 3 rows × 2 columns
  Column 1 (a1, b1, c1): Raw IRD time series, season bands, reset lines
  Column 2 (a2, b2, c2): IRD_norm_log vs LCT, decay fits (good segs only)

Basins: 3204 (Soreq 2), 5201 (Yavne 2), 7401 (Yavne 4)

Usage
-----
  python fig2_decay_evidence.py
  → Toggle SHOW_ALL_SEGMENTS below to switch between main and SI figure
  → Save manually: PNG (300 DPI) + TIFF

KEY FONT SIZE CONTROL
---------------------
  Edit FONT_OVERRIDE below. Set any value to None to inherit plot_style.py.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT, add_season_bands
from config import EVENT_CSV, FIELD_NAMES

# ═════════════════════════════════════════════════════════════════════════════
# ★ USER CONFIG — edit here
# ═════════════════════════════════════════════════════════════════════════════

BASINS       = [3204, 5201, 7401]
PANEL_LABELS = ["a", "b", "c"]
CMAP_NAME    = "viridis"

# ── MAIN SWITCH ───────────────────────────────────────────────────────────────
# False → Figure 2  (main paper, good segments only)
# True  → Figure Sx (SI, all segments)
# SHOW_ALL_SEGMENTS = False
SHOW_ALL_SEGMENTS = True

# ── Date range clip ───────────────────────────────────────────────────────────
DATE_FROM = "2021-01-01"    # e.g. "2018-01-01" | None
DATE_TO   = None             # e.g. "2023-12-31" | None

# ── FONT SIZES ────────────────────────────────────────────────────────────────
# ★ Reviewer font-size fix: bump these up if comments received.
# All sizes in points. None = inherit from plot_style.py.
FONT_OVERRIDE = {
    "title"       : 9,     # panel title  (a1 Basin 3204…)
    "panel_label" : 9,     # bold panel label
    "axis_label"  : 8,     # x/y axis labels
    "tick"        : 9,     # axis tick labels
    "legend"      : 9,     # legend + colorbar
    "annotation"  : 5,     # summary box text
}

# ── Season band opacity ───────────────────────────────────────────────────────
SEASON_ALPHA = 0.7         # 0=invisible, 1=opaque. 0.15–0.25 recommended.

# ── Reset line appearance ─────────────────────────────────────────────────────
RESET_LINE_KW  = dict(color="black", linewidth=0.7,
                       linestyle="--", alpha=0.30, zorder=1)
RESET_LABEL_KW = dict(fontsize=8, color="black", alpha=0.55,
                       rotation=90, va="top", ha="right")
MAX_RESET_LABELS = 6        # max "T" labels per time-series panel

# ═════════════════════════════════════════════════════════════════════════════


def _fs(key: str) -> int:
    v = FONT_OVERRIDE.get(key)
    return v if v is not None else FONT.get(key, 11)


def _decay_curve(lct, a, b, lam):
    return a * np.exp(-lam * lct) + b


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (good_events, all_events) — both filtered to date range.
    good_events : is_good_segment == True  (~48% of data)
    all_events  : all event rows including bad segments and outlier basins
    """
    if not EVENT_CSV.exists():
        raise FileNotFoundError(
            f"{EVENT_CSV} not found.\nRun: python -m pipeline.build_dataset --rebuild"
        )
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]

    if "IRD_norm_log" not in df.columns and "IRD_norm" in df.columns:
        df["IRD_norm_log"] = df["IRD_norm"]

    for col in ["LCT", "IRD", "IRD_at_reset", "IRD_norm_log",
                "seg_lambda", "seg_a", "seg_b", "seg_r2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if DATE_FROM:
        df = df[df["opening_valve_date"] >= pd.Timestamp(DATE_FROM)]
    if DATE_TO:
        df = df[df["opening_valve_date"] <= pd.Timestamp(DATE_TO)]

    # All events (row_type == "event", any segment quality)
    all_ev = df[df["row_type"] == "event"].copy()

    # Good segments only
    good_ev = all_ev[all_ev["is_good_segment"] == True].copy()

    dr = f"{DATE_FROM or 'start'} → {DATE_TO or 'end'}"
    print(f"  Good events : {len(good_ev):>6,}  |  All events: {len(all_ev):>6,}  "
          f"|  date range: {dr}")
    print(f"  Good segment coverage: {100*len(good_ev)/len(all_ev):.1f}% of all events")

    return good_ev, all_ev


# ─────────────────────────────────────────────────────────────────────────────
# Segment color map — consistent across both panels per row
# ─────────────────────────────────────────────────────────────────────────────

def _seg_colormap(segs: list) -> dict:
    """Return {segment_id: rgba} using viridis, chronological order."""
    n    = len(segs)
    cmap = plt.colormaps[CMAP_NAME].resampled(max(n, 2))
    return {sid: cmap(i / max(n - 1, 1)) for i, sid in enumerate(sorted(segs))}


# ─────────────────────────────────────────────────────────────────────────────
# Left panel — IRD time series
# ─────────────────────────────────────────────────────────────────────────────

def _plot_timeseries(ax, bdf: pd.DataFrame, seg_colors: dict,
                     panel_label: str, show_xlabel: bool):
    segs = sorted(bdf["segment_id"].unique())

    # Season bands
    import plot_style as _ps
    _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
    for k, (col, _) in _orig.items():
        _ps.SEASON_COLORS[k] = (col, SEASON_ALPHA)
    date_min = bdf["opening_valve_date"].min()
    date_max = bdf["opening_valve_date"].max()
    season_patches = add_season_bands(ax, date_min, date_max)
    for k, v in _orig.items():
        _ps.SEASON_COLORS[k] = v

    # Per-segment scatter + line
    reset_dates = []
    for sid in segs:
        seg = bdf[bdf["segment_id"] == sid].sort_values("opening_valve_date")
        valid = seg["IRD"].notna() & (seg["IRD"] > 0)
        if not valid.any():
            continue
        color = seg_colors[sid]
        ax.scatter(seg.loc[valid, "opening_valve_date"], seg.loc[valid, "IRD"],
                   s=12, alpha=0.75, color=color, zorder=3, linewidths=0)
        ax.plot(seg.loc[valid, "opening_valve_date"], seg.loc[valid, "IRD"],
                color=color, linewidth=0.5, alpha=0.30, zorder=2)
        reset_dates.append(seg["opening_valve_date"].iloc[0])

    # Reset lines + "T" labels
    n_resets = len(reset_dates)
    if n_resets <= MAX_RESET_LABELS:
        label_idx = set(range(n_resets))
    else:
        step = max(1, n_resets // MAX_RESET_LABELS)
        label_idx = set(range(0, n_resets, step))

    for i, rd in enumerate(reset_dates):
        ax.axvline(rd, **RESET_LINE_KW)
        if i in label_idx:
            ax.annotate("T", xy=(rd, 1.0),
                        xycoords=("data", "axes fraction"),
                        **RESET_LABEL_KW)

    ax.set_title(panel_label, fontsize=_fs("title"), loc="left", pad=5)
    ax.set_ylabel("IRD (cm/h)", fontsize=_fs("axis_label"))
    ax.tick_params(axis="both", labelsize=_fs("tick"))
    ax.tick_params(axis="x", rotation=25)
    if show_xlabel:
        ax.set_xlabel("Date", fontsize=_fs("axis_label"))
    else:
        ax.set_xticklabels([])

    return season_patches


# ─────────────────────────────────────────────────────────────────────────────
# Right panel — decay scatter
# ─────────────────────────────────────────────────────────────────────────────

def _plot_decay(ax, bdf: pd.DataFrame, seg_colors: dict,
                panel_label: str, show_xlabel: bool, fig,
                good_segs_only: bool) -> None:
    """
    Scatter IRD_norm_log vs LCT.
    good_segs_only: if True, decay fits shown only for good segments.
                    if False, all segments plotted (fits only where available).
    """
    segs = sorted(bdf["segment_id"].unique())

    for sid in segs:
        seg = (bdf[bdf["segment_id"] == sid]
               .dropna(subset=["LCT", "IRD_norm_log"])
               .sort_values("LCT"))
        if len(seg) < 2:
            continue
        color = seg_colors[sid]
        lct_h = seg["LCT"].values
        lct_d = lct_h / 24
        inorm = seg["IRD_norm_log"].values

        ax.scatter(lct_d, inorm, s=12, alpha=0.70, color=color,
                   zorder=3, linewidths=0)

        # Draw fit line only for good segments (where fit params exist)
        sa  = float(seg["seg_a"].iloc[0])   if "seg_a"   in seg.columns else np.nan
        sb  = float(seg["seg_b"].iloc[0])   if "seg_b"   in seg.columns else np.nan
        slm = float(seg["seg_lambda"].iloc[0])
        if all(np.isfinite([sa, sb, slm])):
            lh = np.linspace(lct_h.min(), lct_h.max(), 300)
            ax.plot(lh / 24, _decay_curve(lh, sa, sb, slm),
                    color=color, linewidth=1.1, alpha=0.75, zorder=4)

    ax.axhline(0, color=COLORS["dark_gray"], linewidth=0.9,
               linestyle="-", alpha=0.40, zorder=2)

    # Summary annotation — counts differ between good-only and all
    good_mask  = bdf["is_good_segment"] == True
    n_good     = bdf[good_mask]["segment_id"].nunique()
    n_total    = len(segs)

    med_r2  = bdf[good_mask].groupby("segment_id")["seg_r2"].first().median()
    med_lam = bdf[good_mask].groupby("segment_id")["seg_lambda"].first().median()
    med_lam_d = med_lam * 24 if np.isfinite(med_lam) else np.nan

    if good_segs_only:
        ann = (f"$n_{{seg}}$ = {n_good}\n"
               f"Median $R^2$ = {med_r2:.3f}\n"
               f"Median $\\lambda$ = {med_lam_d:.3f} d$^{{-1}}$")
    else:
        ann = (f"$n_{{total}}$ = {n_total}  ($n_{{good}}$ = {n_good})\n"
               f"Good seg median $R^2$ = {med_r2:.3f}\n"
               f"Good seg median $\\lambda$ = {med_lam_d:.3f} d$^{{-1}}$")

    ax.annotate(ann,
                xy=(0.97, 0.05), xycoords="axes fraction",
                fontsize=_fs("annotation"), va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=COLORS["light_gray"], alpha=0.92),
                zorder=10)

    # Secondary y-axis
    # median_reset = bdf["IRD_at_reset"].median()
    # if np.isfinite(median_reset) and median_reset > 0:
    #     ax2 = ax.twinx()
    #     ax2.set_ylabel(f"IRD (cm/h)  [ref = {median_reset:.1f}]",
    #                    fontsize=_fs("legend"), color=COLORS["mid_gray"])
    #     ax2.tick_params(axis="y", labelcolor=COLORS["mid_gray"],
    #                     labelsize=_fs("legend"), color=COLORS["mid_gray"])
    #     for spine in ["top", "left", "bottom"]:
    #         ax2.spines[spine].set_visible(False)
    #     ax2.spines["right"].set_alpha(0.35)
    #     ax2.spines["right"].set_color(COLORS["mid_gray"])
    #     ax._ax2          = ax2
    #     ax._median_reset = median_reset

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps[CMAP_NAME],
                                norm=plt.Normalize(vmin=1, vmax=n_total))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.20)
    cbar.set_label("Segment #", fontsize=_fs("legend"))
    cbar.ax.tick_params(labelsize=_fs("legend"))

    ax.set_title(panel_label, fontsize=_fs("title"), loc="left", pad=5)
    ax.set_ylabel(
        r"$\ln\!\left(\mathrm{IRD}/\mathrm{IRD}_\mathrm{reset}\right)$",
        fontsize=_fs("axis_label"))
    ax.tick_params(axis="both", labelsize=_fs("tick"))
    if show_xlabel:
        ax.set_xlabel("Loading Cycle Time (days)", fontsize=_fs("axis_label"))
    else:
        ax.set_xticklabels([])


# ─────────────────────────────────────────────────────────────────────────────
# Assemble
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(plot_df: pd.DataFrame, full_df: pd.DataFrame,
                all_segs: bool) -> None:
    """
    plot_df  : data to plot (good only, or all)
    full_df  : full dataset for the same basins (for annotation counts)
    all_segs : True = SI figure, False = main figure
    """
    apply_style()
    n_rows = len(BASINS)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(14, 4.5 * n_rows),
        gridspec_kw={"wspace": 0.30, "hspace": 0.40},
    )

    season_patches = None

    for row_i, (bn, plabel) in enumerate(zip(BASINS, PANEL_LABELS)):
        bdf   = plot_df[plot_df["basin_number"] == bn].copy()
        # For annotation: always pass full basin data
        bdf_full = full_df[full_df["basin_number"] == bn].copy()
        segs  = sorted(bdf["segment_id"].unique())
        # Color map spans ALL segments so colors are consistent if comparing figs
        all_segs_basin = sorted(
            full_df[full_df["basin_number"] == bn]["segment_id"].unique()
        )
        seg_c   = _seg_colormap(all_segs_basin)
        field   = FIELD_NAMES.get(int(str(bn)[0]), "")
        is_last = (row_i == n_rows - 1)

        label_ts  = f"$\\mathbf{{{plabel}1}}$   Basin {bn} — {field}"
        label_dec = f"$\\mathbf{{{plabel}2}}$   Basin {bn} — {field}"

        ax_ts  = axes[row_i, 0]
        ax_dec = axes[row_i, 1]

        patches = _plot_timeseries(ax_ts, bdf, seg_c,
                                   panel_label=label_ts,
                                   show_xlabel=is_last)
        if row_i == 0:
            season_patches = patches

        _plot_decay(ax_dec, bdf_full if all_segs else bdf,
                    seg_c,
                    panel_label=label_dec,
                    show_xlabel=is_last,
                    fig=fig,
                    good_segs_only=not all_segs)

    # Sync secondary y-axes
    for row_i in range(n_rows):
        ax_dec = axes[row_i, 1]
        if hasattr(ax_dec, "_ax2") and hasattr(ax_dec, "_median_reset"):
            ax2 = ax_dec._ax2
            mr  = ax_dec._median_reset
            y_lo, y_hi = ax_dec.get_ylim()
            ird_lo = mr * np.exp(y_lo)
            ird_hi = mr * np.exp(y_hi)
            ax2.set_ylim(ird_lo, ird_hi)
            step = max(0.5, round((ird_hi - ird_lo) / 5, 1))
            ticks = np.arange(np.ceil(ird_lo / step) * step,
                               ird_hi + step * 0.1, step)
            ax2.set_yticks(ticks)
            ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Captions
# ─────────────────────────────────────────────────────────────────────────────

def _caption_main() -> str:
    basins_str = ", ".join(str(b) for b in BASINS)
    return (
        f"Figure S2. Within-segment IRD decay in clean flooding segments across "
        f"three representative basins ({basins_str}), one per field "
        f"(Soreq 2, Yavne 2, Yavne 4). "
        f"Only segments satisfying all quality criteria are shown "
    )


def _caption_si() -> str:
    basins_str = ", ".join(str(b) for b in BASINS)
    return (
        f"Figure 2. Within-segment IRD dynamics across three representative basins "
f"({basins_str}), one per field (Soreq 2, Yavne 2, Yavne 4), showing all "
f"recorded flooding–drying events over the study period. "
f"Left column (a1–c1): Raw infiltration rate (IRD, cm/h) over time. "
f"Each inter-tillage segment (sequence of flooding–drying cycles between two "
f"consecutive tillage events) is rendered in a distinct color "
f"(viridis, chronological order). "
f"Dashed vertical lines mark tillage events (T); seasonal background shading "
f"indicates winter (blue), spring (peach), summer (yellow), and autumn (green). "
f"A clear seasonal modulation is visible across all three basins, with IRD "
f"tending to peak in summer and decline in winter, consistent with "
f"temperature- and radiation-driven biofilm dynamics. "
f"Right column (a2–c2): Normalized IRD decay within each inter-tillage segment. "
f"Each point represents one flooding–drying event. "
f"Y-axis: IRD_norm = ln(IRD / IRD_reset), where 0 is the post-tillage baseline "
f"and negative values indicate clogging-driven reduction. "
f"X-axis: time since last tillage event (LCT, days). "
f"Exponential decay fit lines are shown for segments that passed quality "
f"screening (Pearson r < −0.05; fit R² ≥ 0.10; ≥ 4 events per segment); "
f"segments without a clear decay signal appear as scatter only. "
f"The annotation box reports total segment count, good segment count, "
f"and decay statistics computed from quality-screened segments. "
f"While the decay signal is evident in many segments, substantial variability "
f"exists — some segments show no monotonic decline or even IRD increase, "
f"consistent with mechanisms such as gas expulsion or biofilm sloughing. "
f"This heterogeneity motivates the machine learning approach: rather than "
f"requiring a clean exponential signal, Model 1 learns the decay rate λ as "
f"a function of operational and environmental features, achieving R² = +0.798 "
f"and MAPE = 18.1%% on held-out basins even when trained on the full noisy dataset "
f"(Condition E)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    mode = "SI — all segments" if SHOW_ALL_SEGMENTS else "Main paper — good segments only"
    print("=" * 60)
    print(f"  FIGURE 2 / Sx — IRD Decay Evidence")
    print(f"  Mode: {mode}")
    print("=" * 60)

    good_ev, all_ev = load_data()

    # Select data to plot based on switch
    plot_df = all_ev if SHOW_ALL_SEGMENTS else good_ev
    # full_df always has all events (for annotation counts in SI figure)
    full_df = all_ev

    print(f"\n  Basins (mode: {mode}):")
    for bn in BASINS:
        bdf_g = good_ev[good_ev["basin_number"] == bn]
        bdf_a = all_ev[all_ev["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        print(f"    [{bn}]  {field:<10}  "
              f"good_events={len(bdf_g):>4}  all_events={len(bdf_a):>4}  "
              f"good_segs={bdf_g['segment_id'].nunique():>3}  "
              f"all_segs={bdf_a['segment_id'].nunique():>3}  "
              f"median_R²={bdf_g.groupby('segment_id')['seg_r2'].first().median():.3f}")

    print(f"\n  Rendering ({mode}) ...")
    print("  → Save manually: PNG (300 DPI) + TIFF\n")

    plot_figure(plot_df, full_df, all_segs=SHOW_ALL_SEGMENTS)

    caption = _caption_si() if SHOW_ALL_SEGMENTS else _caption_main()
    print("\n" + "─" * 60)
    print("CAPTION (copy to PPT / paper):")
    print("─" * 60)
    print(caption)
    print("─" * 60)


if __name__ == "__main__":
    main()