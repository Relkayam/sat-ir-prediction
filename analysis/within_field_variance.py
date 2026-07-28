"""
analysis/within_field_variance.py
=================================
Colleague question (§3.1): the log-ratio normalization collapses the
BETWEEN-BASIN variance share from ~53% (raw log-IRD) to ~4.5% across all 50
basins. Does this also hold WITHIN a single field (e.g. Yavne 1 vs 2 vs 4),
where basins share lithology/field context?

This computes the between-basin variance share BOTH pooled over all basins AND
separately within each field, raw (log-IRD) vs normalized (eta). If the
within-field share is also small after normalization, the scale-removal claim
holds at the field level too — strengthening §3.1.

Reuses the exact between_share() definition from experiments/collapse_metric.py.

HONESTY NOTE (same as collapse_metric): eta subtracts each segment's own
ceiling, so between-basin scale is removed BY CONSTRUCTION. This QUANTIFIES how
much variance that scale represented within a field, it does not "discover" it.

Run: python analysis/within_field_variance.py
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))
from config import EVENT_CSV


def between_share(values: np.ndarray, group: np.ndarray) -> float:
    """Fraction of total variance attributable to between-group mean differences."""
    s = pd.DataFrame({"v": values, "g": group}).dropna()
    if s["g"].nunique() < 2:
        return np.nan  # need >=2 basins in the field to have a between-basin share
    grand = s["v"].mean()
    ss_tot = float(np.sum((s["v"] - grand) ** 2))
    if ss_tot <= 0:
        return np.nan
    gmean = s.groupby("g")["v"].transform("mean")
    ss_between = float(np.sum((gmean - grand) ** 2))
    return ss_between / ss_tot


def main():
    print("=" * 70)
    print("  WITHIN-FIELD variance-collapse check (raw log-IRD vs eta)")
    print("=" * 70)

    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"], low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()]
    if "IRD_norm_log" not in df.columns and "IRD_norm" in df.columns:
        df["IRD_norm_log"] = df["IRD_norm"]
    for c in ["IRD", "IRD_norm_log"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ev = df[df["row_type"] == "event"].dropna(subset=["IRD", "IRD_norm_log"]).copy()
    ev = ev[ev["IRD"] > 0]
    ev["logIRD"] = np.log(ev["IRD"])

    # pooled over all 50 basins (reproduces the §3.1 headline)
    raw_all = between_share(ev["logIRD"].to_numpy(float), ev["basin_number"].to_numpy())
    eta_all = between_share(ev["IRD_norm_log"].to_numpy(float), ev["basin_number"].to_numpy())
    print(f"\n  ALL 50 basins:   raw log-IRD between-basin share = {raw_all:6.1%}"
          f"   ->  eta = {eta_all:6.1%}")

    # within each field
    print("\n  === within each field (between-basin share among that field's basins) ===")
    print(f"  {'field':<12} {'n_basins':>9} {'n_events':>9} {'raw_share':>10} {'eta_share':>10}")
    rows = []
    for field, g in ev.groupby("field_name", sort=True):
        nb = g["basin_number"].nunique()
        raw = between_share(g["logIRD"].to_numpy(float), g["basin_number"].to_numpy())
        eta = between_share(g["IRD_norm_log"].to_numpy(float), g["basin_number"].to_numpy())
        rows.append((field, nb, len(g), raw, eta))
        raw_s = f"{raw:.1%}" if np.isfinite(raw) else "  n/a"
        eta_s = f"{eta:.1%}" if np.isfinite(eta) else "  n/a"
        print(f"  {str(field):<12} {nb:>9} {len(g):>9,} {raw_s:>10} {eta_s:>10}")

    print("\n  Read: if eta_share is small within fields too (as it is pooled),")
    print("  normalization removes between-basin scale at the field level as well,")
    print("  not only across the full 50-basin set.")


if __name__ == "__main__":
    main()