# """
# figures/fig2_decay_evidence.py — Figure 2 (within-segment decay evidence)
# ==========================================================================
# Shows within-segment IRD decay across three representative basins. ALL
# segments are plotted; a per-segment exponential trend line is drawn on every
# segment that has a (diagnostic) fit in the dataset. There is NO display screen
# — the figure is purely descriptive and gates nothing.
#
# The per-segment exponential fit (seg_a / seg_b / seg_lambda / seg_r2) is read
# from the dataset, where it is computed as a NON-GATING diagnostic by
# build_dataset (any segment with >= MIN_EVENTS_PER_SEGMENT events). A trend line
# is drawn iff those fit parameters are finite — the only reason a segment lacks
# a line is that a curve cannot be fit to too few points, not any quality screen.
#
# Layout: 3 rows x 2 columns
#   Column 1 (a1, b1, c1): Raw IRD time series, season bands, reset lines
#   Column 2 (a2, b2, c2): IRD_norm_log vs TST (hours), per-segment decay fits
#
# Basins: 3204 (Soreq 2), 5201 (Yavne 2), 7401 (Yavne 4)
#
# Usage
# -----
#   python fig2_decay_evidence.py
#   → Toggle VIRIDIS below to switch colormaps
#   → Save manually (plt.show): PNG (300 DPI) + TIFF
# """
# from __future__ import annotations
#
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from pathlib import Path
#
# PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
# sys.path.insert(0, str(PROJECT_ROOT))
#
# from plot_style import (apply_style, COLORS, FONT, add_season_bands,
#                         segment_colors as _house_segment_colors)
# from config import EVENT_CSV, FIELD_NAMES, TABLES_DIR
#
# # ═════════════════════════════════════════════════════════════════════════════
# # ★ USER CONFIG — edit here
# # ═════════════════════════════════════════════════════════════════════════════
#
# BASINS       = [3204, 5201, 7401]
# PANEL_LABELS = ["a", "b", "c"]
#
# # ── Colormap toggle ───────────────────────────────────────────────────────────
# # True  → viridis (chronological order encoded as dark→light)
# # False → 100 distinct cycling colors (segment index mod 100, shuffled so
# #         adjacent segments contrast; wraps to the first color after 100)
# VIRIDIS = False
#
# # ── Date range clip ───────────────────────────────────────────────────────────
# DATE_FROM = "2021-01-01"    # e.g. "2018-01-01" | None
# DATE_TO   = None             # e.g. "2023-12-31" | None
#
# # ── FONT SIZES ────────────────────────────────────────────────────────────────
# FONT_OVERRIDE = {
#     "title"       : 9,
#     "panel_label" : 9,
#     "axis_label"  : 8,
#     "tick"        : 9,
#     "legend"      : 9,
# }
#
# # ── Season band opacity ───────────────────────────────────────────────────────
# SEASON_ALPHA = 0.7
#
# # ── Reset line appearance ─────────────────────────────────────────────────────
# RESET_LINE_KW  = dict(color="black", linewidth=0.7,
#                        linestyle="--", alpha=0.30, zorder=1)
# RESET_LABEL_KW = dict(fontsize=8, color="black", alpha=0.55,
#                        rotation=90, va="top", ha="right")
# MAX_RESET_LABELS = 6
#
# # ═════════════════════════════════════════════════════════════════════════════
#
#
# def _fs(key: str) -> int:
#     v = FONT_OVERRIDE.get(key)
#     return v if v is not None else FONT.get(key, 11)
#
#
# def _decay_curve(lct, a, b, lam):
#     return a * np.exp(-lam * lct) + b
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Load
# # ─────────────────────────────────────────────────────────────────────────────
#
# def load_data() -> pd.DataFrame:
#     """Return all event rows for the target basins, date-clipped."""
#     if not EVENT_CSV.exists():
#         raise FileNotFoundError(
#             f"{EVENT_CSV} not found.\nRun: python -m pipeline.build_dataset --rebuild"
#         )
#     df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
#     df = df.loc[:, ~df.columns.duplicated()]
#
#     if "IRD_norm_log" not in df.columns and "IRD_norm" in df.columns:
#         df["IRD_norm_log"] = df["IRD_norm"]
#
#     for col in ["LCT", "IRD", "IRD_at_reset", "IRD_norm_log",
#                 "seg_lambda", "seg_a", "seg_b", "seg_r2"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#
#     if DATE_FROM:
#         df = df[df["opening_valve_date"] >= pd.Timestamp(DATE_FROM)]
#     if DATE_TO:
#         df = df[df["opening_valve_date"] <= pd.Timestamp(DATE_TO)]
#
#     all_ev = df[
#         (df["row_type"] == "event") &
#         (df["basin_number"].isin(BASINS))
#     ].copy()
#
#     dr = f"{DATE_FROM or 'start'} → {DATE_TO or 'end'}"
#     print(f"  Event rows for target basins: {len(all_ev):,}  |  date range: {dr}")
#     return all_ev
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Segment color map — consistent across both panels per row
# # ─────────────────────────────────────────────────────────────────────────────
#
# def _seg_colormap(segs: list) -> dict:
#     """
#     Map each segment_id to a color.
#       VIRIDIS = True  → viridis spread across the segments (chronological).
#       VIRIDIS = False → shared house cycling palette from plot_style
#                         (segment_colors), assigned by chronological index;
#                         cycles after 100 segments.
#     """
#     segs_sorted = sorted(segs)
#     if VIRIDIS:
#         n    = len(segs_sorted)
#         cmap = plt.colormaps["viridis"].resampled(max(n, 2))
#         return {sid: cmap(i / max(n - 1, 1)) for i, sid in enumerate(segs_sorted)}
#     else:
#         cols = _house_segment_colors(len(segs_sorted))
#         return {sid: cols[i] for i, sid in enumerate(segs_sorted)}
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Left panel — IRD time series
# # ─────────────────────────────────────────────────────────────────────────────
#
# def _plot_timeseries(ax, bdf: pd.DataFrame, seg_colors: dict,
#                      panel_label: str, show_xlabel: bool):
#     segs = sorted(bdf["segment_id"].unique())
#
#     import plot_style as _ps
#     _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
#     for k, (col, _) in _orig.items():
#         _ps.SEASON_COLORS[k] = (col, SEASON_ALPHA)
#     season_patches = add_season_bands(ax, bdf["opening_valve_date"].min(),
#                                       bdf["opening_valve_date"].max())
#     for k, v in _orig.items():
#         _ps.SEASON_COLORS[k] = v
#
#     reset_dates = []
#     for sid in segs:
#         seg = bdf[bdf["segment_id"] == sid].sort_values("opening_valve_date")
#         valid = seg["IRD"].notna() & (seg["IRD"] > 0)
#         if not valid.any():
#             continue
#         color = seg_colors[sid]
#         ax.scatter(seg.loc[valid, "opening_valve_date"], seg.loc[valid, "IRD"],
#                    s=12, alpha=0.75, color=color, zorder=3, linewidths=0)
#         ax.plot(seg.loc[valid, "opening_valve_date"], seg.loc[valid, "IRD"],
#                 color=color, linewidth=0.5, alpha=0.30, zorder=2)
#         reset_dates.append(seg["opening_valve_date"].iloc[0])
#
#     n_resets = len(reset_dates)
#     if n_resets <= MAX_RESET_LABELS:
#         label_idx = set(range(n_resets))
#     else:
#         step = max(1, n_resets // MAX_RESET_LABELS)
#         label_idx = set(range(0, n_resets, step))
#
#     for i, rd in enumerate(reset_dates):
#         ax.axvline(rd, **RESET_LINE_KW)
#         if i in label_idx:
#             ax.annotate("T", xy=(rd, 1.0),
#                         xycoords=("data", "axes fraction"),
#                         **RESET_LABEL_KW)
#
#     ax.set_title(panel_label, fontsize=_fs("title"), loc="left", pad=5)
#     ax.set_ylabel("IRD (cm/h)", fontsize=_fs("axis_label"))
#     ax.tick_params(axis="both", labelsize=_fs("tick"))
#     ax.tick_params(axis="x", rotation=25)
#     if show_xlabel:
#         ax.set_xlabel("Date", fontsize=_fs("axis_label"))
#     else:
#         ax.set_xticklabels([])
#
#     return season_patches
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Right panel — decay scatter (ALL segments; trend line where a fit exists)
# # ─────────────────────────────────────────────────────────────────────────────
#
# def _plot_decay(ax, bdf: pd.DataFrame, seg_colors: dict,
#                 panel_label: str, show_xlabel: bool) -> None:
#     """
#     Scatter IRD_norm_log vs TST (hours) for ALL segments. Draw the per-segment
#     exponential trend line wherever the (diagnostic) fit parameters in the
#     dataset are finite — no display screen.
#     """
#     segs = sorted(bdf["segment_id"].unique())
#
#     for sid in segs:
#         seg = (bdf[bdf["segment_id"] == sid]
#                .dropna(subset=["LCT", "IRD_norm_log"])
#                .sort_values("LCT"))
#         if len(seg) < 2:
#             continue
#         color = seg_colors[sid]
#         lct_h = seg["LCT"].values
#         inorm = seg["IRD_norm_log"].values
#
#         ax.scatter(lct_h, inorm, s=12, alpha=0.70, color=color,
#                    zorder=3, linewidths=0)
#
#         # Trend line wherever the dataset's diagnostic fit is finite.
#         sa  = float(seg["seg_a"].iloc[0])      if "seg_a"      in seg.columns else np.nan
#         sb  = float(seg["seg_b"].iloc[0])      if "seg_b"      in seg.columns else np.nan
#         slm = float(seg["seg_lambda"].iloc[0]) if "seg_lambda" in seg.columns else np.nan
#         if all(np.isfinite([sa, sb, slm])):
#             lh = np.linspace(lct_h.min(), lct_h.max(), 300)
#             ax.plot(lh, _decay_curve(lh, sa, sb, slm),
#                     color=color, linewidth=1.1, alpha=0.75, zorder=4)
#
#     ax.axhline(0, color=COLORS["dark_gray"], linewidth=0.9,
#                linestyle="-", alpha=0.40, zorder=2)
#
#     ax.set_title(panel_label, fontsize=_fs("title"), loc="left", pad=5)
#     ax.set_ylabel(
#         r"$\ln\!\left(\mathrm{IRD}/\mathrm{IRD}_\mathrm{reset}\right)$",
#         fontsize=_fs("axis_label"))
#     ax.tick_params(axis="both", labelsize=_fs("tick"))
#     if show_xlabel:
#         ax.set_xlabel("Time Since Tillage (hours)", fontsize=_fs("axis_label"))
#     else:
#         ax.set_xticklabels([])
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Assemble
# # ─────────────────────────────────────────────────────────────────────────────
#
# def plot_figure(all_ev: pd.DataFrame) -> None:
#     apply_style()
#     n_rows = len(BASINS)
#     fig, axes = plt.subplots(
#         n_rows, 2,
#         figsize=(14, 4.5 * n_rows),
#         gridspec_kw={"wspace": 0.30, "hspace": 0.40},
#     )
#
#     for row_i, (bn, plabel) in enumerate(zip(BASINS, PANEL_LABELS)):
#         bdf = all_ev[all_ev["basin_number"] == bn].copy()
#
#         all_basin_segs = sorted(bdf["segment_id"].dropna().unique())
#         seg_c   = _seg_colormap(all_basin_segs)
#         field   = FIELD_NAMES.get(int(str(bn)[0]), "")
#         is_last = (row_i == n_rows - 1)
#
#         label_ts  = f"$\\mathrm{{{plabel}1}}$   Basin {bn} — {field}"
#         label_dec = f"$\\mathrm{{{plabel}2}}$   Basin {bn} — {field}"
#
#         _plot_timeseries(axes[row_i, 0], bdf, seg_c,
#                          panel_label=label_ts, show_xlabel=is_last)
#         _plot_decay(axes[row_i, 1], bdf, seg_c,
#                     panel_label=label_dec, show_xlabel=is_last)
#
#     plt.tight_layout()
#     plt.show()
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Segment-decay statistics over ALL 50 basins (Option 2 — distributions only)
# # ─────────────────────────────────────────────────────────────────────────────
#
# def _iqr(s: pd.Series) -> tuple[float, float]:
#     return float(s.quantile(0.25)), float(s.quantile(0.75))
#
#
# def segment_decay_statistics() -> pd.DataFrame:
#     """
#     Quantify within-segment decay across ALL 50 basins, with NO threshold/screen
#     (Option 2). For every segment that has a diagnostic exponential fit in the
#     dataset (seg_lambda / seg_r2 finite — built for any segment with
#     >= MIN_EVENTS_PER_SEGMENT events), collect the fit R² and the decay rate λ.
#
#     The DISTRIBUTION of these (median + IQR) is the quantitative form of the
#     'detectable but variable' claim: the spread is the variability, and the low
#     tail of fit R² (including poor / near-zero fits) is the 'not everything
#     follows the pattern' evidence. No segment is classified as following / not
#     following — there is no cutoff.
#
#     Transparency: per basin we report BOTH n_segments (all segments) and
#     n_with_fit (segments where curve_fit produced a finite fit), so the
#     fittability attrition is visible rather than hidden.
#
#     Reads the FULL record (no DATE_FROM clip — the figure's date clip is a
#     display choice and must not bias the reported statistic).
#
#     Returns the per-basin table and prints the pooled dataset-level summary.
#     """
#     df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
#     df = df.loc[:, ~df.columns.duplicated()]
#     if "IRD_norm_log" not in df.columns and "IRD_norm" in df.columns:
#         df["IRD_norm_log"] = df["IRD_norm"]
#     for col in ["seg_lambda", "seg_r2"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#
#     ev = df[df["row_type"] == "event"].copy()
#
#     # One row per segment: its fit R² and λ (first value; broadcast per segment)
#     seg = (ev.groupby(["basin_number", "segment_id"])
#               .agg(seg_r2=("seg_r2", "first"),
#                    seg_lambda=("seg_lambda", "first"))
#               .reset_index())
#     seg = seg[seg["segment_id"] >= 0]
#
#     # λ in d⁻¹ for reporting (dataset stores h⁻¹)
#     seg["lambda_d"] = seg["seg_lambda"] * 24.0
#     has_fit = seg["seg_lambda"].notna() & seg["seg_r2"].notna()
#
#     # ── Per-basin table ──────────────────────────────────────────────────────
#     rows = []
#     for bn, g in seg.groupby("basin_number"):
#         field = FIELD_NAMES.get(int(str(int(bn))[0]), str(int(bn)))
#         gfit  = g[g["seg_lambda"].notna() & g["seg_r2"].notna()]
#         if len(gfit):
#             r2_med = float(gfit["seg_r2"].median())
#             r2_lo, r2_hi = _iqr(gfit["seg_r2"])
#             l_med  = float(gfit["lambda_d"].median())
#             l_lo, l_hi = _iqr(gfit["lambda_d"])
#         else:
#             r2_med = r2_lo = r2_hi = l_med = l_lo = l_hi = np.nan
#         rows.append(dict(
#             basin=int(bn), field=field,
#             n_segments=int(len(g)), n_with_fit=int(len(gfit)),
#             fitR2_median=r2_med, fitR2_q25=r2_lo, fitR2_q75=r2_hi,
#             lambda_d_median=l_med, lambda_d_q25=l_lo, lambda_d_q75=l_hi,
#         ))
#     table = pd.DataFrame(rows).sort_values("basin").reset_index(drop=True)
#
#     # ── Pooled dataset-level summary (the §3.1 numbers) ──────────────────────
#     pooled = seg[has_fit]
#     r2_med = float(pooled["seg_r2"].median())
#     r2_lo, r2_hi = _iqr(pooled["seg_r2"])
#     l_med  = float(pooled["lambda_d"].median())
#     l_lo, l_hi = _iqr(pooled["lambda_d"])
#     n_seg_total = int(len(seg))
#     n_seg_fit   = int(len(pooled))
#
#     print("\n" + "=" * 64)
#     print("  SEGMENT-DECAY STATISTICS — all 50 basins (no threshold)")
#     print("=" * 64)
#     print(f"  Segments total (>=1 event)      : {n_seg_total}")
#     print(f"  Segments with a finite fit      : {n_seg_fit} "
#           f"({100*n_seg_fit/n_seg_total:.1f}% of all segments)")
#     print(f"  Fit R²  : median={r2_med:.3f}  IQR=[{r2_lo:.3f}, {r2_hi:.3f}]")
#     print(f"  λ (d⁻¹) : median={l_med:.4f}  IQR=[{l_lo:.4f}, {l_hi:.4f}]")
#     print("  (Fit R² distribution INCLUDES poor/near-zero fits — no screen.)")
#
#     # ── Richer distribution detail for the text ──────────────────────────────
#     r2 = pooled["seg_r2"]
#     print("\n  Fit-R² distribution detail (all fitted segments):")
#     for q in (0.10, 0.25, 0.50, 0.75, 0.90):
#         print(f"    p{int(q * 100):>2} = {r2.quantile(q):.3f}")
#     # Share of fitted segments in interpretable bands (descriptive, not a gate)
#     for lo, hi in [(-np.inf, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
#         share = float(((r2 >= lo) & (r2 < hi)).mean()) * 100
#         lbl = f"R²<0" if hi == 0 else f"{lo:.2f}-{hi:.2f}"
#         print(f"    {lbl:>10}: {share:4.1f}% of fitted segments")
#     # Fraction with essentially no decay structure (descriptive only)
#     print(f"  Fitted segments with R² < 0.10 : "
#           f"{float((r2 < 0.10).mean()) * 100:.1f}%")
#
#
#     out = TABLES_DIR / "segment_decay_stats_SI.xlsx"
#     table.to_excel(out, index=False)
#     print(f"\n  Saved SI table ({len(table)} basins): {out.name}")
#     return table
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────
#
# def main():
#     cmap_mode = "viridis (chronological)" if VIRIDIS else "house cycling palette"
#     print("=" * 60)
#     print(f"  FIGURE 2 — IRD Decay Evidence (all segments, no screen)")
#     print(f"  Colormap: {cmap_mode}")
#     print("=" * 60)
#
#     all_ev = load_data()
#
#     print(f"\n  Per-basin summary — the 3 figure basins (all segments):")
#     for bn in BASINS:
#         bdf   = all_ev[all_ev["basin_number"] == bn]
#         field = FIELD_NAMES.get(int(str(bn)[0]), "")
#         gseg  = bdf.groupby("segment_id").agg(
#             seg_r2=("seg_r2", "first"), seg_lambda=("seg_lambda", "first"))
#         gfit  = gseg[gseg["seg_lambda"].notna() & gseg["seg_r2"].notna()]
#         if len(gfit):
#             r2_med = float(gfit["seg_r2"].median())
#             r2_lo, r2_hi = _iqr(gfit["seg_r2"])
#             ld = gfit["seg_lambda"] * 24.0
#             l_med = float(ld.median()); l_lo, l_hi = _iqr(ld)
#         else:
#             r2_med = r2_lo = r2_hi = l_med = l_lo = l_hi = np.nan
#         print(f"    [{bn}]  {field:<10}  "
#               f"segments={gseg.shape[0]:>3}  with_fit={len(gfit):>3}  "
#               f"fitR²={r2_med:.3f} [{r2_lo:.3f},{r2_hi:.3f}]  "
#               f"λ={l_med:.4f} [{l_lo:.4f},{l_hi:.4f}] d⁻¹")
#
#     # All-50-basin statistics + SI table
#     segment_decay_statistics()
#
#     print(f"\n  Rendering ({cmap_mode}) ...")
#     print("  → Save manually: PNG (300 DPI) + TIFF\n")
#
#     plot_figure(all_ev)
#
#
# if __name__ == "__main__":
#     main()



"""
figures/fig2_decay_evidence.py — Figure 2 (within-segment decay evidence)
==========================================================================
Shows within-segment IRD decay across three representative basins. ALL
segments are plotted; a per-segment exponential trend line is drawn on every
segment that has a (diagnostic) fit in the dataset. There is NO display screen
— the figure is purely descriptive and gates nothing.

The per-segment exponential fit (seg_a / seg_b / seg_lambda / seg_r2) is read
from the dataset, where it is computed as a NON-GATING diagnostic by
build_dataset (any segment with >= MIN_EVENTS_PER_SEGMENT events). A trend line
is drawn iff those fit parameters are finite — the only reason a segment lacks
a line is that a curve cannot be fit to too few points, not any quality screen.

Layout: 3 rows x 2 columns
  Column 1 (a1, b1, c1): Raw IRD time series, season bands, reset lines
  Column 2 (a2, b2, c2): IRD_norm_log vs TST (hours), per-segment decay fits

Basins: 3204 (Soreq 2), 5201 (Yavne 2), 7401 (Yavne 4)

Usage
-----
  python fig2_decay_evidence.py
  → Toggle VIRIDIS below to switch colormaps
  → Save manually (plt.show): PNG (300 DPI) + TIFF
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import (apply_style, COLORS, FONT, add_season_bands,
                        segment_colors as _house_segment_colors)
from config import EVENT_CSV, FIELD_NAMES, TABLES_DIR

# ═════════════════════════════════════════════════════════════════════════════
# ★ USER CONFIG — edit here
# ═════════════════════════════════════════════════════════════════════════════

BASINS       = [3204, 5201, 7401]
PANEL_LABELS = ["a", "b", "c"]

# ── Colormap toggle ───────────────────────────────────────────────────────────
# True  → viridis (chronological order encoded as dark→light)
# False → 100 distinct cycling colors (segment index mod 100, shuffled so
#         adjacent segments contrast; wraps to the first color after 100)
VIRIDIS = False

# ── Date range clip ───────────────────────────────────────────────────────────
DATE_FROM = "2021-01-01"    # e.g. "2018-01-01" | None
DATE_TO   = None             # e.g. "2023-12-31" | None

# ── FONT SIZES ────────────────────────────────────────────────────────────────
FONT_OVERRIDE = {
    "title"       : 13,
    "panel_label" : 13,
    "axis_label"  : 12,
    "tick"        : 11,
    "legend"      : 11,
}

# ── Season band opacity ───────────────────────────────────────────────────────
SEASON_ALPHA = 0.7

# ── Reset line appearance ─────────────────────────────────────────────────────
RESET_LINE_KW  = dict(color="black", linewidth=0.7,
                       linestyle="--", alpha=0.30, zorder=1)
RESET_LABEL_KW = dict(fontsize=8, color="black", alpha=0.55,
                       rotation=90, va="top", ha="right")
MAX_RESET_LABELS = 6

# ═════════════════════════════════════════════════════════════════════════════


def _fs(key: str) -> int:
    v = FONT_OVERRIDE.get(key)
    return v if v is not None else FONT.get(key, 11)


def _decay_curve(lct, a, b, lam):
    return a * np.exp(-lam * lct) + b


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Return all event rows for the target basins, date-clipped."""
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

    all_ev = df[
        (df["row_type"] == "event") &
        (df["basin_number"].isin(BASINS))
    ].copy()

    dr = f"{DATE_FROM or 'start'} → {DATE_TO or 'end'}"
    print(f"  Event rows for target basins: {len(all_ev):,}  |  date range: {dr}")
    return all_ev


# ─────────────────────────────────────────────────────────────────────────────
# Segment color map — consistent across both panels per row
# ─────────────────────────────────────────────────────────────────────────────

def _seg_colormap(segs: list) -> dict:
    """
    Map each segment_id to a color.
      VIRIDIS = True  → viridis spread across the segments (chronological).
      VIRIDIS = False → shared house cycling palette from plot_style
                        (segment_colors), assigned by chronological index;
                        cycles after 100 segments.
    """
    segs_sorted = sorted(segs)
    if VIRIDIS:
        n    = len(segs_sorted)
        cmap = plt.colormaps["viridis"].resampled(max(n, 2))
        return {sid: cmap(i / max(n - 1, 1)) for i, sid in enumerate(segs_sorted)}
    else:
        cols = _house_segment_colors(len(segs_sorted))
        return {sid: cols[i] for i, sid in enumerate(segs_sorted)}


# ─────────────────────────────────────────────────────────────────────────────
# Left panel — IRD time series
# ─────────────────────────────────────────────────────────────────────────────

def _plot_timeseries(ax, bdf: pd.DataFrame, seg_colors: dict,
                     panel_label: str, show_xlabel: bool):
    segs = sorted(bdf["segment_id"].unique())

    import plot_style as _ps
    _orig = {k: v for k, v in _ps.SEASON_COLORS.items()}
    for k, (col, _) in _orig.items():
        _ps.SEASON_COLORS[k] = (col, SEASON_ALPHA)
    season_patches = add_season_bands(ax, bdf["opening_valve_date"].min(),
                                      bdf["opening_valve_date"].max())
    for k, v in _orig.items():
        _ps.SEASON_COLORS[k] = v

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
# Right panel — decay scatter (ALL segments; trend line where a fit exists)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_decay(ax, bdf: pd.DataFrame, seg_colors: dict,
                panel_label: str, show_xlabel: bool) -> None:
    """
    Scatter IRD_norm_log vs TST (hours) for ALL segments. Draw the per-segment
    exponential trend line wherever the (diagnostic) fit parameters in the
    dataset are finite — no display screen.
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
        inorm = seg["IRD_norm_log"].values

        ax.scatter(lct_h, inorm, s=12, alpha=0.70, color=color,
                   zorder=3, linewidths=0)

        # Trend line wherever the dataset's diagnostic fit is finite.
        sa  = float(seg["seg_a"].iloc[0])      if "seg_a"      in seg.columns else np.nan
        sb  = float(seg["seg_b"].iloc[0])      if "seg_b"      in seg.columns else np.nan
        slm = float(seg["seg_lambda"].iloc[0]) if "seg_lambda" in seg.columns else np.nan
        if all(np.isfinite([sa, sb, slm])):
            lh = np.linspace(lct_h.min(), lct_h.max(), 300)
            ax.plot(lh, _decay_curve(lh, sa, sb, slm),
                    color=color, linewidth=1.1, alpha=0.75, zorder=4)

    ax.axhline(0, color=COLORS["dark_gray"], linewidth=0.9,
               linestyle="-", alpha=0.40, zorder=2)

    ax.set_title(panel_label, fontsize=_fs("title"), loc="left", pad=5)
    ax.set_ylabel(
        r"$\ln\!\left(\mathrm{IRD}/\mathrm{IRD}_\mathrm{reset}\right)$",
        fontsize=_fs("axis_label"))
    ax.tick_params(axis="both", labelsize=_fs("tick"))
    if show_xlabel:
        ax.set_xlabel("Time Since Tillage (hours)", fontsize=_fs("axis_label"))
    else:
        ax.set_xticklabels([])


# ─────────────────────────────────────────────────────────────────────────────
# Assemble
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(all_ev: pd.DataFrame) -> None:
    apply_style()
    n_rows = len(BASINS)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(14, 4.5 * n_rows),
        gridspec_kw={"wspace": 0.30, "hspace": 0.40},
    )

    for row_i, (bn, plabel) in enumerate(zip(BASINS, PANEL_LABELS)):
        bdf = all_ev[all_ev["basin_number"] == bn].copy()

        all_basin_segs = sorted(bdf["segment_id"].dropna().unique())
        seg_c   = _seg_colormap(all_basin_segs)
        field   = FIELD_NAMES.get(int(str(bn)[0]), "")
        is_last = (row_i == n_rows - 1)

        label_ts  = f"$\\mathrm{{{plabel}1}}$   Basin {bn} — {field}"
        label_dec = f"$\\mathrm{{{plabel}2}}$   Basin {bn} — {field}"

        _plot_timeseries(axes[row_i, 0], bdf, seg_c,
                         panel_label=label_ts, show_xlabel=is_last)
        _plot_decay(axes[row_i, 1], bdf, seg_c,
                    panel_label=label_dec, show_xlabel=is_last)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Segment-decay statistics over ALL 50 basins (Option 2 — distributions only)
# ─────────────────────────────────────────────────────────────────────────────

def _iqr(s: pd.Series) -> tuple[float, float]:
    return float(s.quantile(0.25)), float(s.quantile(0.75))


def segment_decay_statistics() -> pd.DataFrame:
    """
    Quantify within-segment decay across ALL 50 basins, with NO threshold/screen
    (Option 2). For every segment that has a diagnostic exponential fit in the
    dataset (seg_lambda / seg_r2 finite — built for any segment with
    >= MIN_EVENTS_PER_SEGMENT events), collect the fit R² and the decay rate λ.

    The DISTRIBUTION of these (median + IQR) is the quantitative form of the
    'detectable but variable' claim: the spread is the variability, and the low
    tail of fit R² (including poor / near-zero fits) is the 'not everything
    follows the pattern' evidence. No segment is classified as following / not
    following — there is no cutoff.

    Transparency: per basin we report BOTH n_segments (all segments) and
    n_with_fit (segments where curve_fit produced a finite fit), so the
    fittability attrition is visible rather than hidden.

    Reads the FULL record (no DATE_FROM clip — the figure's date clip is a
    display choice and must not bias the reported statistic).

    Returns the per-basin table and prints the pooled dataset-level summary.
    """
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    if "IRD_norm_log" not in df.columns and "IRD_norm" in df.columns:
        df["IRD_norm_log"] = df["IRD_norm"]
    for col in ["seg_lambda", "seg_r2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ev = df[df["row_type"] == "event"].copy()

    # One row per segment: its fit R² and λ (first value; broadcast per segment)
    seg = (ev.groupby(["basin_number", "segment_id"])
              .agg(seg_r2=("seg_r2", "first"),
                   seg_lambda=("seg_lambda", "first"))
              .reset_index())
    seg = seg[seg["segment_id"] >= 0]

    # λ in d⁻¹ for reporting (dataset stores h⁻¹)
    seg["lambda_d"] = seg["seg_lambda"] * 24.0
    has_fit = seg["seg_lambda"].notna() & seg["seg_r2"].notna()

    # ── Per-basin table ──────────────────────────────────────────────────────
    rows = []
    for bn, g in seg.groupby("basin_number"):
        field = FIELD_NAMES.get(int(str(int(bn))[0]), str(int(bn)))
        gfit  = g[g["seg_lambda"].notna() & g["seg_r2"].notna()]
        if len(gfit):
            r2_med = float(gfit["seg_r2"].median())
            r2_lo, r2_hi = _iqr(gfit["seg_r2"])
            l_med  = float(gfit["lambda_d"].median())
            l_lo, l_hi = _iqr(gfit["lambda_d"])
        else:
            r2_med = r2_lo = r2_hi = l_med = l_lo = l_hi = np.nan
        rows.append(dict(
            basin=int(bn), field=field,
            n_segments=int(len(g)), n_with_fit=int(len(gfit)),
            fitR2_median=r2_med, fitR2_q25=r2_lo, fitR2_q75=r2_hi,
            lambda_d_median=l_med, lambda_d_q25=l_lo, lambda_d_q75=l_hi,
        ))
    table = pd.DataFrame(rows).sort_values("basin").reset_index(drop=True)

    # ── Pooled dataset-level summary (the §3.1 numbers) ──────────────────────
    pooled = seg[has_fit]
    r2_med = float(pooled["seg_r2"].median())
    r2_lo, r2_hi = _iqr(pooled["seg_r2"])
    l_med  = float(pooled["lambda_d"].median())
    l_lo, l_hi = _iqr(pooled["lambda_d"])
    n_seg_total = int(len(seg))
    n_seg_fit   = int(len(pooled))

    print("\n" + "=" * 64)
    print("  SEGMENT-DECAY STATISTICS — all 50 basins (no threshold)")
    print("=" * 64)
    print(f"  Segments total (>=1 event)      : {n_seg_total}")
    print(f"  Segments with a finite fit      : {n_seg_fit} "
          f"({100*n_seg_fit/n_seg_total:.1f}% of all segments)")
    print(f"  Fit R²  : median={r2_med:.3f}  IQR=[{r2_lo:.3f}, {r2_hi:.3f}]")
    print(f"  λ (d⁻¹) : median={l_med:.4f}  IQR=[{l_lo:.4f}, {l_hi:.4f}]")
    print("  (Fit R² distribution INCLUDES poor/near-zero fits — no screen.)")

    # ── Richer distribution detail for the text ──────────────────────────────
    r2 = pooled["seg_r2"]
    print("\n  Fit-R² distribution detail (all fitted segments):")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"    p{int(q * 100):>2} = {r2.quantile(q):.3f}")
    # Share of fitted segments in interpretable bands (descriptive, not a gate)
    for lo, hi in [(-np.inf, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        share = float(((r2 >= lo) & (r2 < hi)).mean()) * 100
        lbl = f"R²<0" if hi == 0 else f"{lo:.2f}-{hi:.2f}"
        print(f"    {lbl:>10}: {share:4.1f}% of fitted segments")
    # Fraction with essentially no decay structure (descriptive only)
    print(f"  Fitted segments with R² < 0.10 : "
          f"{float((r2 < 0.10).mean()) * 100:.1f}%")


    out = TABLES_DIR / "segment_decay_stats_SI.xlsx"
    table.to_excel(out, index=False)
    print(f"\n  Saved SI table ({len(table)} basins): {out.name}")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cmap_mode = "viridis (chronological)" if VIRIDIS else "house cycling palette"
    print("=" * 60)
    print(f"  FIGURE 2 — IRD Decay Evidence (all segments, no screen)")
    print(f"  Colormap: {cmap_mode}")
    print("=" * 60)

    all_ev = load_data()

    print(f"\n  Per-basin summary — the 3 figure basins (all segments):")
    for bn in BASINS:
        bdf   = all_ev[all_ev["basin_number"] == bn]
        field = FIELD_NAMES.get(int(str(bn)[0]), "")
        gseg  = bdf.groupby("segment_id").agg(
            seg_r2=("seg_r2", "first"), seg_lambda=("seg_lambda", "first"))
        gfit  = gseg[gseg["seg_lambda"].notna() & gseg["seg_r2"].notna()]
        if len(gfit):
            r2_med = float(gfit["seg_r2"].median())
            r2_lo, r2_hi = _iqr(gfit["seg_r2"])
            ld = gfit["seg_lambda"] * 24.0
            l_med = float(ld.median()); l_lo, l_hi = _iqr(ld)
        else:
            r2_med = r2_lo = r2_hi = l_med = l_lo = l_hi = np.nan
        print(f"    [{bn}]  {field:<10}  "
              f"segments={gseg.shape[0]:>3}  with_fit={len(gfit):>3}  "
              f"fitR²={r2_med:.3f} [{r2_lo:.3f},{r2_hi:.3f}]  "
              f"λ={l_med:.4f} [{l_lo:.4f},{l_hi:.4f}] d⁻¹")

    # All-50-basin statistics + SI table
    segment_decay_statistics()

    print(f"\n  Rendering ({cmap_mode}) ...")
    print("  → Save manually: PNG (300 DPI) + TIFF\n")

    plot_figure(all_ev)


if __name__ == "__main__":
    main()