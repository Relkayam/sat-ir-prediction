"""
analysis/filter_stats.py — distribution smoke for quality-filter columns
========================================================================
Goal: choose data-driven thresholds (esp. an upper Ct cap) instead of eyeballing.
Also tests the hypothesis: "a Ct-max cap removes the FT outliers too, because
FT is part of Ct" — so we may not need a separate FT filter.

Reads the already-built event_dataset.csv (no rebuild). Real data only.

Run:  python analysis/filter_stats.py
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\user\PycharmProjects\sat-ir-prediction")

EVENT_CSV = r"C:\Users\user\PycharmProjects\sat-ir-prediction\data\event_dataset.csv"

COLS = ["FT", "Ct", "CIV", "IRD_R_squared", "AL", "DrT", "DT", "WT"]
CT_CAPS = [200, 300, 500, 1000, 2000]   # candidate upper caps to test


def describe_col(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    q = s.quantile([0.50, 0.90, 0.95, 0.99, 0.999]).to_dict()
    return {
        "n": len(s), "min": s.min(), "median": q[0.50],
        "p90": q[0.90], "p95": q[0.95], "p99": q[0.99],
        "p99.9": q[0.999], "max": s.max(), "mean": s.mean(), "std": s.std(),
    }


def main():
    print("=" * 70)
    print("  QUALITY-FILTER DISTRIBUTION SMOKE (real event_dataset.csv)")
    print("=" * 70)
    df = pd.read_csv(EVENT_CSV, low_memory=False)
    print(f"  rows={len(df):,}\n")

    # 1) per-column distribution
    rows = []
    for c in COLS:
        if c not in df.columns:
            print(f"  [skip] {c} absent"); continue
        d = describe_col(df[c]); d["col"] = c; rows.append(d)
    prof = pd.DataFrame(rows).set_index("col")
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}",
                           "display.width", 160, "display.max_columns", None):
        print("=== distributions ===")
        print(prof.to_string())

    # 2) does FT sit inside Ct? correlation + how FT behaves at Ct extremes
    if {"FT", "Ct"}.issubset(df.columns):
        ft = pd.to_numeric(df["FT"], errors="coerce")
        ct = pd.to_numeric(df["Ct"], errors="coerce")
        m = ft.notna() & ct.notna()
        r = np.corrcoef(ft[m], ct[m])[0, 1]
        frac = (ft[m] <= ct[m] + 1e-6).mean()
        print(f"\n  corr(FT, Ct) = {r:.3f}   |   fraction with FT <= Ct = {frac:.3f}")

    # 3) KEY TEST: does a Ct cap remove the FT outliers?
    if {"FT", "Ct"}.issubset(df.columns):
        print("\n=== Does a Ct-max cap subsume FT outliers? ===")
        ft = pd.to_numeric(df["FT"], errors="coerce")
        ct = pd.to_numeric(df["Ct"], errors="coerce")
        n = len(df)
        print(f"  {'Ct_cap':>8} {'rows_dropped':>13} {'%dropped':>9} "
              f"{'max_FT_kept':>12} {'p99.9_FT_kept':>14}")
        for cap in CT_CAPS:
            keep = ct <= cap
            dropped = int((~keep).sum())
            ft_kept = ft[keep].dropna()
            print(f"  {cap:>8} {dropped:>13,} {100*dropped/n:>8.2f}% "
                  f"{ft_kept.max():>12,.1f} {ft_kept.quantile(0.999):>14,.1f}")
        # for reference: pure FT cap at 200
        keep_ft = ft <= 200
        print(f"\n  (reference) pure FT<=200 would drop "
              f"{int((~keep_ft).sum()):,} rows "
              f"({100*(~keep_ft).mean():.2f}%); max Ct among kept = "
              f"{ct[keep_ft].max():,.1f}")


if __name__ == "__main__":
    main()