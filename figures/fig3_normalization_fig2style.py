"""
fig3_normalization.py — Figure 3: the normalization argument (scale removal)
============================================================================
2x2 grid:
  [0,0] Basin 5201 — IRD time series, house palette by segment, season bands
  [0,1] Basin 4104 — IRD time series, house palette by segment, season bands
  [1,0] IRD vs TST (hours) — both basins overlaid (raw: separated by scale)
  [1,1] η(t) vs TST (hours) — both basins overlaid (normalized: level equalized)

Claim (scale removal only — see collapse_metric results): log-ratio
normalization removes the between-basin SCALE. It does NOT make basins follow a
common decay path (the global TST-only model explains ~0.6% of η variance;
basins still decay at different rates). The 'rate not shared' point is made in
the §3.1 text bridge to Model 1, not in this figure.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import (apply_style, COLORS, FONT, add_season_bands,
                        segment_colors as _house_segment_colors)
from config import EVENT_CSV, FIELD_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
BASINS        = [5201, 4104]
BASIN_MARKERS = {5201: "o", 4104: "^"}
BASIN_MS      = {5201: 14,  4104: 16}
BASIN_ALPHA   = {5201: 0.75, 4104: 0.75}
# Basin labels — no hardcoded ρ values (the real numbers are printed by the
# variance-share summary and reported in the text, not baked into the legend).
BASIN_LABELS  = {
    5201: "Basin 5201 — Yavne 2",
    4104: "Basin 4104 — Yavne 1",
}
# Distinct colors for bottom panels (basin identity, not segment)
BASIN_COLOR   = {5201: "#065A82", 4104: "#E07B39"}

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
    """Segment → color from the shared house palette (matches Figure 2)."""
    segs_sorted = sorted(segs)
    cols = _house_segment_colors(len(segs_sorted))
    return {sid: cols[i] for i, sid in enumerate(segs_sorted)}


# ── Variance-share statistic (Option C) — all 50 basins ───────────────────────

def _between_share(values: np.ndarray, group: np.ndarray) -> float:
    s = pd.DataFrame({"v": values, "g": group}).dropna()
    grand  = s["v"].mean()
    ss_tot = float(np.sum((s["v"] - grand) ** 2))
    if ss_tot <= 0:
        return np.nan
    gmean = s.groupby("g")["v"].transform("mean")
    ss_bet = float(np.sum((gmean - grand) ** 2))
    return ss_bet / ss_tot


def variance_share_summary() -> None:
    """
    Print the between-basin variance share, raw IRD vs η, over ALL 50 basins,
    plus the basin median-ρ range. This is Figure 3's quantitative claim:
    normalization removes the between-basin SCALE (definitional in part —
    framed as 'how much of raw variance was scale'), not the decay rate.
    """
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if "IRD_norm_log" not in df.columns and "IRD_norm" in df.columns:
        df["IRD_norm_log"] = df["IRD_norm"]
    for c in ["IRD", "IRD_norm_log", "IRD_at_reset"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    ev = df[df["row_type"] == "event"].dropna(subset=["IRD", "IRD_norm_log"]).copy()
    ev = ev[ev["IRD"] > 0]
    ev["logIRD"] = np.log(ev["IRD"])

    raw   = _between_share(ev["IRD"].to_numpy(float),       ev["basin_number"].to_numpy())
    logr  = _between_share(ev["logIRD"].to_numpy(float),    ev["basin_number"].to_numpy())
    etash = _between_share(ev["IRD_norm_log"].to_numpy(float), ev["basin_number"].to_numpy())

    print("\n" + "=" * 64)
    print("  NORMALIZATION SCALE-REMOVAL (Option C) — all 50 basins")
    print("=" * 64)
    print(f"  Events: {len(ev):,}   Basins: {ev['basin_number'].nunique()}")
    print(f"  Between-basin share of variance:")
    print(f"    raw IRD   : {100*raw:5.1f}%")
    print(f"    log(IRD)  : {100*logr:5.1f}%   <- fairer denominator (IRD skewed)")
    print(f"    η         : {100*etash:5.1f}%")
    print(f"  Scale removed: {100*logr:.1f}% (log basis) -> {100*etash:.1f}% (η)")

    if "IRD_at_reset" in df.columns:
        rho  = pd.to_numeric(df["IRD_at_reset"], errors="coerce")
        bmed = df.assign(rho=rho).groupby("basin_number")["rho"].median().dropna()
        print(f"  Basin median ρ: {bmed.min():.2f}–{bmed.max():.2f} cm/h  "
              f"({bmed.max()/bmed.min():.1f}-fold range)")
    print("  NOTE: η drop is partly definitional; report as 'fraction of raw")
    print("        variance that was scale', not as a discovery.")


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _plot_timeseries(ax, bdf: pd.DataFrame, bn: int,
                     seg_c: dict, show_season_legend: bool) -> list:
    """Top panels: IRD vs date, house palette by segment, season bands."""
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
    """Bottom-left: raw IRD vs TST (hours), both basins overlaid."""
    for bn in BASINS:
        bdf   = good[good["basin_number"] == bn]
        color = BASIN_COLOR[bn]
        ax.scatter(bdf["LCT"], bdf["IRD"],
                   s=BASIN_MS[bn], alpha=0.55,
                   color=color, marker=BASIN_MARKERS[bn],
                   zorder=3, linewidths=0,
                   label=BASIN_LABELS[bn])

    ax.set_title("IRD vs TST — raw IRD space",
                 fontsize=_fs("title"), loc="left", pad=4)
    ax.set_xlabel("Time Since Tillage (hours)",
                  fontsize=_fs("axis_label"))
    ax.set_ylabel("IRD (cm/h)", fontsize=_fs("axis_label"))
    ax.tick_params(axis="both", labelsize=_fs("tick"))
    ax.legend(fontsize=_fs("legend"), loc="upper right")


def _plot_eta_lct(ax, good: pd.DataFrame) -> None:
    """Bottom-right: η(t) vs TST (hours), both basins overlaid, fit lines."""
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

            # Diagnostic fit line (descriptive only)
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

    ax.set_title("η(t) vs TST — normalized space",
                 fontsize=_fs("title"), loc="left", pad=4)
    ax.set_xlabel("Time Since Tillage (hours)",
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

    # Bottom-left: IRD vs TST
    _plot_ird_lct(axes[1, 0], good)

    # Bottom-right: η vs TST
    _plot_eta_lct(axes[1, 1], good)

    # Panel labels
    for ax, lbl in zip(axes.flat, ["a1 -5201", "a2 -4104", "b1", "b2"]):
        ax.set_title(
            f"$\\mathrm{{{lbl}}}$   " + ax.get_title(),
            fontsize=_fs("title"), loc="left", pad=4,
        )

    plt.tight_layout()
    plt.show()

    # ── Summary (per-basin, the 2 shown basins) ──────────────────────────────
    print("\nSummary (good segments, date-clipped):")
    for bn in BASINS:
        bdf = good[good["basin_number"] == bn]
        print(f"\n  Basin {bn} ({FIELD_NAMES.get(int(str(bn)[0]), '')})")
        print(f"    n events     : {len(bdf)}")
        print(f"    n segments   : {bdf['segment_id'].nunique()}")
        print(f"    IRD range    : {bdf['IRD'].min():.2f}–{bdf['IRD'].max():.2f} cm/h")
        print(f"    TST range    : {bdf['LCT'].min():.0f}–{bdf['LCT'].max():.0f} hours")
        print(f"    η range      : {bdf['IRD_norm_log'].min():.3f}–"
              f"{bdf['IRD_norm_log'].max():.3f}")
        med_rho = pd.to_numeric(bdf['IRD_at_reset'], errors='coerce').median()
        print(f"    median ρ     : {med_rho:.2f} cm/h")

    # ── All-50-basin scale-removal statistic (Figure 3's claim) ──────────────
    variance_share_summary()


if __name__ == "__main__":
    main()