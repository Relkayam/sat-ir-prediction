# """
# figures/fig7_bootstrap_shap.py — Figure 7: SHAP stability across the bootstrap
# ==============================================================================
# Summarizes, per feature, mean|SHAP| importance and effect direction across the
# 500 random bootstrap draws (not a single draw). Two panels: Model 1 (left),
# Model 2 (right). Reads the per-iteration SHAP tables written by
# run_bootstrap_v2.py (COMPUTE_SHAP=True).
#
# Feature definitions are given in SI Table S[x] (cited from §2.4). This figure
# does NOT state feature counts in its caption — the count/definitions live in
# that SI table and §2.4.
#
# Usage
# -----
#   python -m figures.fig7_bootstrap_shap
# """
# from __future__ import annotations
#
# import sys
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from pathlib import Path
#
# warnings.filterwarnings("ignore")
#
# PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))
#
# from plot_style import apply_style, COLORS, FONT
#
# DATA_DIR = PROJECT_ROOT / "data"
# SHAP_M1     = DATA_DIR / "bootstrap_shap_m1.csv"
# SHAP_M2     = DATA_DIR / "bootstrap_shap_m2.csv"
# SHAP_M1_DIR = DATA_DIR / "bootstrap_shap_m1_dir.csv"
# SHAP_M2_DIR = DATA_DIR / "bootstrap_shap_m2_dir.csv"
#
# DIR_STABLE_THRESHOLD = 0.90
#
# M1_COLOR = "#065A82"
# M2_COLOR = "#E07B39"
#
# # ── Fonts — journal scale; feature labels (ytick) largest for readability ─────
# FONTS = dict(
#     feature_label = 13,   # y-tick feature names — the thing readers must read
#     axis_label    = 12,
#     title         = 13,
#     tick          = 11,
#     glyph         = 10,
#     legend        = 10,
# )
#
# FEATURE_LABELS = {
#     # Model 1
#     "prev_ALPHA"    : "α (prev. drying fraction)",
#     "IRD_at_reset"  : "ρᵢ (post-tillage IRD)",
#     "prev_DrT"      : "t_dry (prev.)",
#     "log1p_prev_HL" : "log HL (prev.)",
#     "LCT"           : "TST (time since tillage)",
#     "prev_TD"       : "T_dry (prev.)",
#     "prev_FT"       : "t_flood (prev.)",
#     "cum_TW"        : "cum. T_wet",
#     "cum_FT"        : "cum. t_flood",
#     "prev_RD"       : "R_dry (prev.)",
#     "prev_RW"       : "R_wet (prev.)",
#     # Model 2
#     "prev_delta"      : "δᵢ₋₁ (prev. reset)",
#     "prev_prev_delta" : "δᵢ₋₂",
#     "DAR"             : "R̄_day (tillage date)",
#     "mean_RW"         : "mean R_wet",
#     "mean_DrT"        : "mean t_dry",
#     "max_DrT"         : "max t_dry",
#     "last_RD"         : "last R_dry",
#     "max_RD"          : "max R_dry",
#     "sum_RW"          : "sum R_wet",
#     "total_LCT"       : "total segment duration",
#     "n_events"        : "event count",
#     "frac_zero_DrT"   : "frac. events no drying",
# }
#
#
# def _label(f: str) -> str:
#     return FEATURE_LABELS.get(f, f)
#
#
# def _summarize(imp_csv: Path, dir_csv: Path) -> pd.DataFrame:
#     imp = pd.read_csv(imp_csv)
#     dir_ = pd.read_csv(dir_csv)
#     feats = [c for c in imp.columns if c != "iteration"]
#     recs = []
#     for f in feats:
#         v = imp[f].dropna().values
#         d = dir_[f].dropna().values if f in dir_.columns else np.array([])
#         if len(v) == 0:
#             continue
#         if len(d):
#             med_dir = float(np.median(d))
#             dom     = 1.0 if med_dir >= 0 else -1.0
#             consist = float(np.mean(np.sign(d) == dom))
#         else:
#             med_dir, dom, consist = np.nan, 0.0, np.nan
#         recs.append(dict(
#             feature=f, label=_label(f),
#             median_imp=float(np.median(v)),
#             q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
#             median_dir=med_dir, dir_consistency=consist, dom_dir=dom,
#             n_iter=int(len(v)),
#         ))
#     return pd.DataFrame(recs).sort_values("median_imp", ascending=True).reset_index(drop=True)
#
#
# def _panel(ax, summ: pd.DataFrame, color: str, title: str) -> None:
#     y   = np.arange(len(summ))
#     med = summ["median_imp"].values
#     q25 = summ["q25"].values
#     q75 = summ["q75"].values
#     stable = summ["dir_consistency"].values >= DIR_STABLE_THRESHOLD
#
#     for i in range(len(summ)):
#         if stable[i]:
#             ax.barh(y[i], med[i], color=color, alpha=0.85, zorder=3,
#                     edgecolor="white", linewidth=0.4)
#         else:
#             ax.barh(y[i], med[i], color="none", zorder=3,
#                     edgecolor=COLORS["mid_gray"], linewidth=1.0, hatch="////")
#
#     xerr = np.vstack([med - q25, q75 - med])
#     ax.errorbar(med, y, xerr=xerr, fmt="none", ecolor=COLORS["dark_gray"],
#                 elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4, alpha=0.8)
#
#     for i in range(len(summ)):
#         if stable[i]:
#             glyph = "↑less" if summ["dom_dir"].iloc[i] > 0 else "↑more"
#             ax.text(q75[i] + max(med) * 0.02, y[i], glyph,
#                     va="center", ha="left", fontsize=FONTS["glyph"],
#                     color=COLORS["dark_gray"])
#
#     ax.set_yticks(y)
#     ax.set_yticklabels(summ["label"].values, fontsize=FONTS["feature_label"])
#     ax.set_xlabel("Median mean|SHAP|  (IQR whiskers)", fontsize=FONTS["axis_label"])
#     ax.set_title(title, fontsize=FONTS["title"], fontweight="bold", pad=6)
#     ax.tick_params(axis="x", labelsize=FONTS["tick"])
#     ax.grid(True, axis="x", alpha=0.18, linestyle="--", linewidth=0.6)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.set_xlim(left=0)
#
#
# def _print_table(name: str, summ: pd.DataFrame) -> None:
#     print(f"  {name} — features by median importance:")
#     print(f"  {'Feature':<28} {'median':>8} {'IQR':>16} "
#           f"{'med.dir':>8} {'dir-cons':>9}")
#     print(f"  {'-'*74}")
#     for _, r in summ.sort_values('median_imp', ascending=False).iterrows():
#         d = r['median_dir']
#         dstr = f"{d:+.2f}" if np.isfinite(d) else "  n/a"
#         print(f"  {r['label']:<28} {r['median_imp']:>8.4f} "
#               f"[{r['q25']:>6.4f},{r['q75']:>6.4f}] "
#               f"{dstr:>8} {r['dir_consistency']:>8.0%}")
#     print()
#
#
# def _build_caption(m1, m2, n_iter):
#     def topk(summ, k=2):
#         s = summ.sort_values("median_imp", ascending=False).head(k)
#         return ", ".join(s["label"].tolist())
#     m1_top = topk(m1, 2)
#     m2_max = m2.sort_values("median_imp", ascending=False)["median_imp"].iloc[0]
#     return (
#         f"Figure 7. SHAP feature importance and effect direction across {n_iter} "
#         f"bootstrap held-out configurations. Bars show the median mean absolute "
#         f"SHAP value across draws (whiskers: interquartile range); a bar is solid "
#         f"where the sign of the feature's effect — corr(feature value, SHAP "
#         f"value) — agrees in at least {DIR_STABLE_THRESHOLD:.0%} of draws "
#         f"(annotated ↑less or ↑more, i.e. whether a higher feature value reduces "
#         f"or increases decay / raises recovery), and hatched where the direction "
#         f"is unstable. Left: Model 1 (within-segment decay η); the two leading "
#         f"features ({m1_top}) are separated from the rest and stable across "
#         f"configurations. Right: Model 2 (post-tillage recovery δ); no feature "
#         f"dominates, consistent with recovery being weakly feature-predictable "
#         f"and dominated by the system's self-restoring tendency. Feature "
#         f"definitions are given in SI Table S[x]."
#     )
#
#
# def main():
#     print("=" * 65)
#     print("  FIGURE 7 — SHAP stability across the bootstrap (M1 + M2)")
#     print("=" * 65)
#
#     for p in [SHAP_M1, SHAP_M2, SHAP_M1_DIR, SHAP_M2_DIR]:
#         if not p.exists():
#             raise FileNotFoundError(
#                 f"{p} not found.\nRun run_bootstrap_v2.py with COMPUTE_SHAP=True first.")
#
#     m1 = _summarize(SHAP_M1, SHAP_M1_DIR)
#     m2 = _summarize(SHAP_M2, SHAP_M2_DIR)
#     n_iter = int(m1["n_iter"].max())
#
#     print(f"  Iterations with SHAP: M1={int(m1['n_iter'].max())}  "
#           f"M2={int(m2['n_iter'].max())}")
#     print(f"  M1 features shown: {len(m1)}   M2 features shown: {len(m2)}")
#     print(f"  Direction = corr(feature value, SHAP value); "
#           f"stable if ≥{DIR_STABLE_THRESHOLD:.0%} of draws agree on sign\n")
#
#     apply_style()
#     fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#     _panel(axes[0], m1, M1_COLOR, "Model 1 — within-segment decay η")
#     _panel(axes[1], m2, M2_COLOR, "Model 2 — post-tillage recovery δ")
#
#     legend_elems = [
#         Patch(facecolor=COLORS["mid_gray"], alpha=0.85,
#               label="direction stable (↑less / ↑more)"),
#         Patch(facecolor="none", edgecolor=COLORS["mid_gray"],
#               hatch="////", label="direction unstable across draws"),
#     ]
#     axes[1].legend(handles=legend_elems, fontsize=FONTS["legend"],
#                    loc="lower right", framealpha=0.85)
#
#     plt.tight_layout()
#     plt.show()
#
#     # ── text-routed numbers ──────────────────────────────────────────────────
#     print("\n" + "=" * 70)
#     print("TEXT NUMBERS (route to §3.4):")
#     print("=" * 70)
#     _print_table("Model 1", m1)
#     _print_table("Model 2", m2)
#
#     print("=" * 70)
#     print("ARTICLE CAPTION (Figure 7):")
#     print("=" * 70)
#     print(_build_caption(m1, m2, n_iter))
#
#
# if __name__ == "__main__":
#     main()
"""
figures/fig7_bootstrap_shap.py — Figure 7: SHAP stability across the bootstrap
==============================================================================
Summarizes, per feature, mean|SHAP| importance and effect direction across the
500 random bootstrap draws (not a single draw). Two panels: Model 1 (left),
Model 2 (right). Reads the per-iteration SHAP tables written by
run_bootstrap_v2.py (COMPUTE_SHAP=True).

Feature definitions are given in SI Table S[x] (cited from §2.4). This figure
does NOT state feature counts in its caption — the count/definitions live in
that SI table and §2.4.

Usage
-----
  python -m figures.fig7_bootstrap_shap
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_style, COLORS, FONT

DATA_DIR = PROJECT_ROOT / "data"
SHAP_M1     = DATA_DIR / "bootstrap_shap_m1.csv"
SHAP_M2     = DATA_DIR / "bootstrap_shap_m2.csv"
SHAP_M1_DIR = DATA_DIR / "bootstrap_shap_m1_dir.csv"
SHAP_M2_DIR = DATA_DIR / "bootstrap_shap_m2_dir.csv"

DIR_STABLE_THRESHOLD = 0.90

M1_COLOR = "#065A82"
M2_COLOR = "#E07B39"

# ── Fonts — journal scale; feature labels (ytick) largest for readability ─────
FONTS = dict(
    feature_label = 13,   # y-tick feature names — the thing readers must read
    axis_label    = 13,
    title         = 13,
    tick          = 13,
    glyph         = 10,
    legend        = 10,
)

FEATURE_LABELS = {
    # Model 1
    "prev_ALPHA"    : "α (prev. drying fraction)",
    "IRD_at_reset"  : "ρᵢ (post-tillage IRD)",
    "prev_DrT"      : "t_dry (prev.)",
    "log1p_prev_HL" : "log HL (prev.)",
    "LCT"           : "TST (time since tillage)",
    "prev_TD"       : "T_dry (prev.)",
    "prev_FT"       : "t_flood (prev.)",
    "cum_TW"        : "cum. T_wet",
    "cum_FT"        : "cum. t_flood",
    "prev_RD"       : "R_dry (prev.)",
    "prev_RW"       : "R_wet (prev.)",
    # Model 2
    "prev_delta"      : "δᵢ₋₁ (prev. reset)",
    "prev_prev_delta" : "δᵢ₋₂",
    "DAR"             : "R̄_day (tillage date)",
    "mean_RW"         : "mean R_wet",
    "mean_DrT"        : "mean t_dry",
    "max_DrT"         : "max t_dry",
    "last_RD"         : "last R_dry",
    "max_RD"          : "max R_dry",
    "sum_RW"          : "sum R_wet",
    "total_LCT"       : "total segment duration",
    "n_events"        : "event count",
    "frac_zero_DrT"   : "frac. events no drying",
}


def _label(f: str) -> str:
    return FEATURE_LABELS.get(f, f)


def _summarize(imp_csv: Path, dir_csv: Path) -> pd.DataFrame:
    imp = pd.read_csv(imp_csv)
    dir_ = pd.read_csv(dir_csv)
    feats = [c for c in imp.columns if c != "iteration"]
    recs = []
    for f in feats:
        v = imp[f].dropna().values
        d = dir_[f].dropna().values if f in dir_.columns else np.array([])
        if len(v) == 0:
            continue
        if len(d):
            med_dir = float(np.median(d))
            dom     = 1.0 if med_dir >= 0 else -1.0
            consist = float(np.mean(np.sign(d) == dom))
        else:
            med_dir, dom, consist = np.nan, 0.0, np.nan
        recs.append(dict(
            feature=f, label=_label(f),
            median_imp=float(np.median(v)),
            q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
            median_dir=med_dir, dir_consistency=consist, dom_dir=dom,
            n_iter=int(len(v)),
        ))
    return pd.DataFrame(recs).sort_values("median_imp", ascending=True).reset_index(drop=True)


def _panel(ax, summ: pd.DataFrame, color: str, title: str) -> None:
    y   = np.arange(len(summ))
    med = summ["median_imp"].values
    q25 = summ["q25"].values
    q75 = summ["q75"].values
    stable = summ["dir_consistency"].values >= DIR_STABLE_THRESHOLD

    for i in range(len(summ)):
        if stable[i]:
            ax.barh(y[i], med[i], color=color, alpha=0.85, zorder=3,
                    edgecolor="white", linewidth=0.4)
        else:
            ax.barh(y[i], med[i], color="none", zorder=3,
                    edgecolor=COLORS["mid_gray"], linewidth=1.0, hatch="////")

    xerr = np.vstack([med - q25, q75 - med])
    ax.errorbar(med, y, xerr=xerr, fmt="none", ecolor=COLORS["dark_gray"],
                elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4, alpha=0.8)

    for i in range(len(summ)):
        if stable[i]:
            # Arrow refers to the EFFECT ON INFILTRATION RATE, not the feature.
            # dom_dir = sign of corr(feature, SHAP-on-target); the target (η, δ)
            # increases with IR, so dom_dir>0 => higher feature raises IR (↑ IR),
            # dom_dir<0 => higher feature lowers IR (↓ IR). Unambiguous per panel.
            glyph = "↑ IR" if summ["dom_dir"].iloc[i] > 0 else "↓ IR"
            ax.text(q75[i] + max(med) * 0.02, y[i], glyph,
                    va="center", ha="left", fontsize=FONTS["glyph"],
                    color=COLORS["dark_gray"])

    ax.set_yticks(y)
    ax.set_yticklabels(summ["label"].values, fontsize=FONTS["feature_label"])
    ax.set_xlabel("Median mean|SHAP|  (IQR whiskers)", fontsize=FONTS["axis_label"])
    ax.set_title(title, fontsize=FONTS["title"], fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=FONTS["tick"])
    ax.grid(True, axis="x", alpha=0.18, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)


def _print_table(name: str, summ: pd.DataFrame) -> None:
    print(f"  {name} — features by median importance:")
    print(f"  {'Feature':<28} {'median':>8} {'IQR':>16} "
          f"{'med.dir':>8} {'dir-cons':>9}")
    print(f"  {'-'*74}")
    for _, r in summ.sort_values('median_imp', ascending=False).iterrows():
        d = r['median_dir']
        dstr = f"{d:+.2f}" if np.isfinite(d) else "  n/a"
        print(f"  {r['label']:<28} {r['median_imp']:>8.4f} "
              f"[{r['q25']:>6.4f},{r['q75']:>6.4f}] "
              f"{dstr:>8} {r['dir_consistency']:>8.0%}")
    print()


def _build_caption(m1, m2, n_iter):
    def topk(summ, k=2):
        s = summ.sort_values("median_imp", ascending=False).head(k)
        return ", ".join(s["label"].tolist())
    m1_top = topk(m1, 2)
    m2_max = m2.sort_values("median_imp", ascending=False)["median_imp"].iloc[0]
    return (
        f"Figure 7. SHAP feature importance and effect direction across {n_iter} "
        f"bootstrap held-out configurations. Bars show the median mean absolute "
        f"SHAP value across draws (whiskers: interquartile range); a bar is solid "
        f"where the sign of the feature's effect — corr(feature value, SHAP "
        f"value) — agrees in at least {DIR_STABLE_THRESHOLD:.0%} of draws "
        f"(annotated ↑ IR or ↓ IR, i.e. whether a higher feature value raises or "
        f"lowers the infiltration rate), and hatched where the direction "
        f"is unstable. Left: Model 1 (within-segment decay η); the two leading "
        f"features ({m1_top}) are separated from the rest and stable across "
        f"configurations. Right: Model 2 (post-tillage recovery δ); the two "
        f"preceding reset log-ratios lead, both with a stable negative sign "
        f"indicating mean reversion, while weather and operational descriptors "
        f"follow at a lower level. Feature definitions are given in SI Table S[x]."
    )


def main():
    print("=" * 65)
    print("  FIGURE 7 — SHAP stability across the bootstrap (M1 + M2)")
    print("=" * 65)

    for p in [SHAP_M1, SHAP_M2, SHAP_M1_DIR, SHAP_M2_DIR]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\nRun run_bootstrap_v2.py with COMPUTE_SHAP=True first.")

    m1 = _summarize(SHAP_M1, SHAP_M1_DIR)
    m2 = _summarize(SHAP_M2, SHAP_M2_DIR)
    n_iter = int(m1["n_iter"].max())

    print(f"  Iterations with SHAP: M1={int(m1['n_iter'].max())}  "
          f"M2={int(m2['n_iter'].max())}")
    print(f"  M1 features shown: {len(m1)}   M2 features shown: {len(m2)}")
    print(f"  Direction = corr(feature value, SHAP value); "
          f"stable if ≥{DIR_STABLE_THRESHOLD:.0%} of draws agree on sign\n")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _panel(axes[0], m1, M1_COLOR, "Model 1 — within-segment decay η")
    _panel(axes[1], m2, M2_COLOR, "Model 2 — post-tillage recovery δ")

    legend_elems = [
        Patch(facecolor=COLORS["mid_gray"], alpha=0.85,
              label="direction stable (↑ IR / ↓ IR)"),
        Patch(facecolor="none", edgecolor=COLORS["mid_gray"],
              hatch="////", label="direction unstable across draws"),
    ]
    axes[1].legend(handles=legend_elems, fontsize=FONTS["legend"],
                   loc="lower right", framealpha=0.85)

    plt.tight_layout()
    plt.show()

    # ── text-routed numbers ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEXT NUMBERS (route to §3.4):")
    print("=" * 70)
    _print_table("Model 1", m1)
    _print_table("Model 2", m2)

    print("=" * 70)
    print("ARTICLE CAPTION (Figure 7):")
    print("=" * 70)
    print(_build_caption(m1, m2, n_iter))


if __name__ == "__main__":
    main()