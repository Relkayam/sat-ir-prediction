"""
figures/figure_s1_filter_funnel.py — SI Figure S1: sensor-quality funnel
=========================================================================
Standalone. Reads data/event_dataset.csv (already built by
pipeline/build_dataset.py, which writes a `filter_reason` column on every
event) and renders the per-filter exclusion funnel for the four sensor-quality
thresholds.

The funnel is SEQUENTIAL (first-match-wins): each event is attributed to the
FIRST threshold it fails, in the order R² -> CIV -> t_cycle -> AL (the same
order applied in build_dataset._tag_quality_filter). Each bar therefore shows
the events REMOVED MARGINALLY by that filter — not the total number failing the
criterion in isolation. Events with row_type=='pre_segment' (before a basin's
first reset) are shown as a separate step. The final bar is the set of events
used by the models (filter_reason == "").

This matches the main-text statement that 39,565 of 41,959 events (94.3%) pass
all four thresholds.

No model training, no DuckDB — pure read + plot.

Usage
-----
  python -m figures.figure_s1_filter_funnel
  (saves PNG + prints the table; copy the numbers into SI Table / caption)
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EVENT_CSV, FIGURES_DIR, TABLES_DIR, DATA_CUTOFF

# filter_reason -> (display label using PAPER notation, bar colour)
# Order MUST match build_dataset._tag_quality_filter insertion order:
#   IRD_R_squared -> CIV -> Ct(=t_cycle) -> AL
FILTER_STEPS = [
    ("quality_filter_IRD_R_squared", "Drainage R\u00b2 < 0.94",      "#C0392B"),
    ("quality_filter_CIV",           "CIV < 3,000 m\u00b3",          "#E07B39"),
    ("quality_filter_Ct",            "t_cycle < 20 h",               "#E6B800"),
    ("quality_filter_AL",            "AL < 5 cm",                    "#7D9D3A"),
]
PASS_COLOR = "#065A82"
RAW_COLOR  = "#34495E"


def build_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funnel over the FLOODING-DRYING EVENT population only (row_type == "event"),
    which is the population the main text counts (41,959 events). Reset rows and
    pre-segment rows are NOT part of this universe and are excluded entirely.

    filter_reason is stored as NaN for passing events (not ""), so the pass set
    is row_type=="event" AND filter_reason is null. The four sensor filters are
    attributed sequentially (first-match-wins) in build_dataset order.
    """
    if "filter_reason" not in df.columns or "row_type" not in df.columns:
        raise KeyError(
            "event_dataset.csv needs 'filter_reason' and 'row_type' columns — "
            "rebuild with pipeline/build_dataset.py --rebuild."
        )
    # Restrict to the flooding-drying EVENT population (matches the paper).
    ev = df[df["row_type"] == "event"].copy()
    reason = ev["filter_reason"]
    # passing events carry NaN/empty; failures carry a quality_filter_* string
    is_fail = reason.notna() & (reason.astype(str).str.strip() != "")

    total_events = len(ev)
    counts = reason[is_fail].astype(str).value_counts()

    rows = [dict(step="Flooding–drying events", filter_reason="\u2014",
                 n_removed=np.nan, n_remaining=total_events, pct_of_raw=100.0)]
    remaining = total_events
    for reason_key, label, _c in FILTER_STEPS:
        if reason_key == "pre_segment":
            continue   # pre-segment rows are not in the event population
        n = int(counts.get(reason_key, 0))
        remaining -= n
        rows.append(dict(step=label, filter_reason=reason_key,
                         n_removed=n, n_remaining=remaining,
                         pct_of_raw=round(100.0 * remaining / total_events, 1)
                         if total_events else 0.0))

    n_pass = int((~is_fail).sum())
    rows.append(dict(step="Passed events (used by models)", filter_reason="",
                     n_removed=np.nan, n_remaining=n_pass,
                     pct_of_raw=round(100.0 * n_pass / total_events, 1)
                     if total_events else 0.0))
    return pd.DataFrame(rows)


def print_table(f: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  FILTER FUNNEL  (sequential, first-match-wins)")
    print("=" * 70)
    print(f"  {'Step':<34} {'removed':>8} {'remaining':>10} {'% raw':>7}")
    print(f"  {'-'*68}")
    for _, r in f.iterrows():
        rem = "" if not np.isfinite(r["n_removed"]) else f"{int(r['n_removed']):,}"
        print(f"  {r['step']:<34} {rem:>8} {int(r['n_remaining']):>10,} "
              f"{r['pct_of_raw']:>6.1f}%")
    print("=" * 70)


def plot_funnel(f: pd.DataFrame) -> Path:
    raw     = int(f.iloc[0]["n_remaining"])
    passed  = int(f.iloc[-1]["n_remaining"])
    labels  = [f.iloc[0]["step"]] + [s[1] for s in FILTER_STEPS] + [f.iloc[-1]["step"]]
    remain  = f["n_remaining"].tolist()
    colors  = [RAW_COLOR] + [s[2] for s in FILTER_STEPS] + [PASS_COLOR]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, remain, color=colors, edgecolor="white", linewidth=0.6, alpha=0.9)

    for yi, val, row in zip(y, remain, f.itertuples()):
        txt = f"{int(val):,}"
        if np.isfinite(row.n_removed):
            txt += f"   (\u2212{int(row.n_removed):,})"
        ax.text(val + raw * 0.01, yi, txt, va="center", ha="left", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Events remaining", fontsize=10)
    ax.set_xlim(0, raw * 1.18)
    pct = 100.0 * passed / raw if raw else 0
    ax.set_title(f"Sensor-quality funnel: {passed:,} of {raw:,} events "
                 f"({pct:.1f}%) pass all four thresholds",
                 fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.18, linestyle="--", linewidth=0.6)
    plt.tight_layout()

    out = FIGURES_DIR / "figure_s1_filter_funnel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    return out


def main() -> None:
    print("=" * 70)
    print("  SI FIGURE S1 — sensor-quality funnel")
    print("=" * 70)
    if not EVENT_CSV.exists():
        raise FileNotFoundError(f"{EVENT_CSV} not found — run build_dataset.py first.")
    df = pd.read_csv(EVENT_CSV, usecols=lambda c: c in
                     {"filter_reason", "row_type", "opening_valve_date"}
                     or True)
    print(f"  Loaded {len(df):,} event rows from {EVENT_CSV.name}")
    funnel = build_funnel(df)
    print_table(funnel)
    funnel.to_excel(TABLES_DIR / "figure_s1_filter_funnel.xlsx", index=False)
    print(f"  Saved table: figure_s1_filter_funnel.xlsx")
    out = plot_funnel(funnel)
    print(f"  Saved figure: {out.name}")
    print("\n  >> Paste the printed table here so the SI caption gets real numbers.")


if __name__ == "__main__":
    main()