# smoke_5_normalization_2x2.py
# 2x2 grid:
#   [0,0] Basin 5201 — IRD time series, viridis by segment, season bands
#   [0,1] Basin 4104 — IRD time series, viridis by segment, season bands
#   [1,0] IRD vs LCT (hours) — both basins overlaid
#   [1,1] η(t) vs LCT (hours) — both basins overlaid, fit lines
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT, add_season_bands
from config import EVENT_CSV, FIELD_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
BASINS        = [5201, 4104]
BASIN_MARKERS = {5201: "o", 4104: "^"}
BASIN_MS      = {5201: 14,  4104: 16}
BASIN_ALPHA   = {5201: 0.75, 4104: 0.75}
BASIN_LABELS  = {
    5201: "Basin 5201 — Yavne 2  (median ρ = 6.3 cm/h)",
    4104: "Basin 4104 — Yavne 1  (median ρ = 1.7 cm/h)",
}
# Distinct colors for bottom panels (basin identity, not segment)
BASIN_COLOR   = {5201: "#065A82", 4104: "#E07B39"}

CMAP_NAME    = "viridis"
SEASON_ALPHA = 0.6
DATE_FROM    = "2015-01-01"
DATE_TO      = None

FONT_OVERRIDE = {
    "title"      : 11,
    "axis_label" : 10,
    "tick"       : 9,
    "legend"     : 8,
    "annotation" : 9,
}

def _fs(key):
    return FONT_OVERRIDE.get(key, FONT.get(key, 11))

def _decay_curve(lct, a, b, lam):
    return a * np.exp(-lam * lct) + b

# ── Load ──────────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
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
    return df[
        (df["row_type"] == "event") &
        (df["basin_number"].isin(BASINS))
    ].copy()

def _seg_colormap(segs: list) -> dict:
    n    = len(segs)
    cmap = plt.colormaps[CMAP_NAME].resampled(max(n, 2))
    return {sid: cmap(i / max(n - 1, 1)) for i, sid in enumerate(sorted(segs))}

# ── Panel helpers ─────────────────────────────────────────────────────────────

def _plot_timeseries(ax, bdf: pd.DataFrame, bn: int,
                     seg_c: dict, show_season_legend: bool) -> list:
    """Top panels: IRD vs date, viridis by segment, season bands."""
    import plot_style as _ps
    _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
    for k, (col, _) in _orig.items():
        _ps.SEASON_COLORS[k] = (col, SEASON_ALPHA)
    patches = add_season_bands(
        ax,
        bdf["opening_valve_date"].min(),
        bdf["opening_valve_date"].max(),
    )
    for k, v in _orig.items():
        _ps.SEASON_COLORS[k] = v

    segs = sorted(bdf["segment_id"].unique())
    for sid in segs:
        seg   = bdf[bdf["segment_id"] == sid].sort_values("opening_valve_date")
        valid = seg["IRD"].notna() & (seg["IRD"] > 0)
        if not valid.any():
            continue
        color = seg_c[sid]
        ax.scatter(seg.loc[valid, "opening_valve_date"], seg.loc[valid, "IRD"],
                   s=12, alpha=0.80, color=color, zorder=3, linewidths=0)
        ax.plot(seg.loc[valid, "opening_valve_date"], seg.loc[valid, "IRD"],
                color=color, linewidth=0.5, alpha=0.25, zorder=2)

    field = FIELD_NAMES.get(int(str(bn)[0]), "")
    n_segs = len(segs)
    ax.set_title(f"Basin {bn} — {field}  |  {n_segs} segments",
                 fontsize=_fs("title"), loc="left", pad=4)
    ax.set_xlabel("Date", fontsize=_fs("axis_label"))
    ax.set_ylabel("IRD (cm/h)", fontsize=_fs("axis_label"))
    ax.tick_params(axis="both", labelsize=_fs("tick"))
    ax.tick_params(axis="x", rotation=25)
    return patches


def _plot_ird_lct(ax, good: pd.DataFrame) -> None:
    """Bottom-left: raw IRD vs LCT (hours), both basins overlaid."""
    for bn in BASINS:
        bdf   = good[good["basin_number"] == bn]
        color = BASIN_COLOR[bn]
        ax.scatter(bdf["LCT"], bdf["IRD"],
                   s=BASIN_MS[bn], alpha=0.55,
                   color=color, marker=BASIN_MARKERS[bn],
                   zorder=3, linewidths=0,
                   label=BASIN_LABELS[bn])

    ax.set_title("IRD vs LCT — raw IRD space",
                 fontsize=_fs("title"), loc="left", pad=4)
    ax.set_xlabel("Time since last tillage, LCT (hours)",
                  fontsize=_fs("axis_label"))
    ax.set_ylabel("IRD (cm/h)", fontsize=_fs("axis_label"))
    ax.tick_params(axis="both", labelsize=_fs("tick"))
    ax.legend(fontsize=_fs("legend"), loc="upper right")


def _plot_eta_lct(ax, good: pd.DataFrame) -> None:
    """Bottom-right: η(t) vs LCT (hours), both basins overlaid, fit lines."""
    for bn in BASINS:
        bdf  = good[good["basin_number"] == bn]
        segs = sorted(bdf["segment_id"].unique())
        seg_c = _seg_colormap(segs)
        first = True

        for sid in segs:
            seg = (bdf[bdf["segment_id"] == sid]
                   .dropna(subset=["LCT", "IRD_norm_log"])
                   .sort_values("LCT"))
            if len(seg) < 2:
                continue
            color  = seg_c[sid]
            lct_h  = seg["LCT"].values
            inorm  = seg["IRD_norm_log"].values

            ax.scatter(lct_h, inorm,
                       s=BASIN_MS[bn], alpha=0.60,
                       color=color, marker=BASIN_MARKERS[bn],
                       zorder=3, linewidths=0,
                       label=BASIN_LABELS[bn] if first else None)

            # Fit line
            sa  = float(seg["seg_a"].iloc[0])      if "seg_a"      in seg.columns else np.nan
            sb  = float(seg["seg_b"].iloc[0])      if "seg_b"      in seg.columns else np.nan
            slm = float(seg["seg_lambda"].iloc[0])
            if all(np.isfinite([sa, sb, slm])):
                lh = np.linspace(lct_h.min(), lct_h.max(), 300)
                ax.plot(lh, _decay_curve(lh, sa, sb, slm),
                        color=color, linewidth=1.0, alpha=0.65, zorder=4)
            first = False

    ax.axhline(0, color=COLORS["dark_gray"], lw=0.9,
               linestyle="-", alpha=0.40, zorder=2,
               label="η = 0  (post-tillage baseline)")

    ax.set_title("η(t) vs LCT — normalized space",
                 fontsize=_fs("title"), loc="left", pad=4)
    ax.set_xlabel("Time since last tillage, LCT (hours)",
                  fontsize=_fs("axis_label"))
    ax.set_ylabel(
        r"$\eta(t) = \ln\!\left(\mathrm{IRD}(t)\,/\,\mathrm{IRD}_\mathrm{reset}\right)$",
        fontsize=_fs("axis_label"),
    )
    ax.tick_params(axis="both", labelsize=_fs("tick"))

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(),
              fontsize=_fs("legend"), loc="lower left")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    apply_style()
    good = load()

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9),
        gridspec_kw={"hspace": 0.42, "wspace": 0.30},
    )

    # Top-left: Basin 5201 time series
    segs_5201 = sorted(good[good["basin_number"] == 5201]["segment_id"].unique())
    seg_c_5201 = _seg_colormap(segs_5201)
    patches = _plot_timeseries(axes[0, 0],
                               good[good["basin_number"] == 5201],
                               5201, seg_c_5201,
                               show_season_legend=True)

    # Top-right: Basin 4104 time series
    segs_4104 = sorted(good[good["basin_number"] == 4104]["segment_id"].unique())
    seg_c_4104 = _seg_colormap(segs_4104)
    _plot_timeseries(axes[0, 1],
                     good[good["basin_number"] == 4104],
                     4104, seg_c_4104,
                     show_season_legend=False)

    # Bottom-left: IRD vs LCT
    _plot_ird_lct(axes[1, 0], good)

    # Bottom-right: η vs LCT
    _plot_eta_lct(axes[1, 1], good)

    # Panel labels
    for ax, lbl in zip(axes.flat, ["a1", "a2", "b1", "b2"]):
        ax.set_title(
            f"$\\mathbf{{{lbl}}}$   " + ax.get_title(),
            fontsize=_fs("title"), loc="left", pad=4,
        )

    # Season legend — top center, above top panels
    if patches:
        fig.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            fontsize=_fs("legend"),
            title="Season",
            title_fontsize=_fs("legend"),
            framealpha=0.85,
        )

    plt.tight_layout()
    plt.show()

    # Summary
    print("\nSummary (good segments, date-clipped):")
    for bn in BASINS:
        bdf = good[good["basin_number"] == bn]
        print(f"\n  Basin {bn} ({FIELD_NAMES.get(int(str(bn)[0]), '')})")
        print(f"    n events     : {len(bdf)}")
        print(f"    n segments   : {bdf['segment_id'].nunique()}")
        print(f"    IRD range    : {bdf['IRD'].min():.2f}–{bdf['IRD'].max():.2f} cm/h")
        print(f"    LCT range    : {bdf['LCT'].min():.0f}–{bdf['LCT'].max():.0f} hours")
        print(f"    η range      : {bdf['IRD_norm_log'].min():.3f}–"
              f"{bdf['IRD_norm_log'].max():.3f}")
        print(f"    Median seg R²: "
              f"{bdf.groupby('segment_id')['seg_r2'].first().median():.3f}")
        print(f"    Median λ     : "
              f"{bdf.groupby('segment_id')['seg_lambda'].first().median():.5f} h⁻¹  "
              f"({bdf.groupby('segment_id')['seg_lambda'].first().median()*24:.4f} d⁻¹)")


    print("Figure 1. The normalization argument. Panels a1–a2: raw IRD (cm/h) time series for Basin 5201 (Yavne 2, median IRDreset = 6.3 cm/h) and Basin 4104 (Yavne 1, median IRDreset = 1.7 cm/h). Each color represents one inter-tillage segment (viridis, chronological order); seasonal background shading indicates winter (blue), spring (peach), summer (yellow), and autumn (green). Panel b1: raw IRD vs time since last tillage (LCT, hours) for both basins overlaid — the two basins occupy separate regions of the y-axis, reflecting the ~4× difference in saturated hydraulic conductivity. Panel b2: normalized IRD, η(t) = ln(IRD(t)/IRDreset), vs LCT for both basins overlaid — the two clouds collapse onto the same decay structure, demonstrating that log-ratio normalization removes between-basin scale differences and enables a single global model.")

if __name__ == "__main__":
    main()