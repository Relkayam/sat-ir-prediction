"""
endogeneity_tests.py — Discussion evidence: feature endogeneity + zero-drying prevalence
========================================================================================
Two standalone tests that turn the 'counterintuitive feature directions' into
demonstrated structure rather than narrative.

TEST 1 — Operator response to capacity (the chicken-and-egg / endogeneity claim)
  Claim: operators apply more water (HL), flood longer (t_flood), and allow less
  drying on basins that ALREADY have high infiltration capacity. If true, the
  counterintuitive SHAP signs of HL / t_flood / t_dry are explained by reverse
  causation: these operational variables are RESPONSES to capacity, not exogenous
  drivers.
  Test: at the BASIN level (one point per basin, n=50 — avoids pseudoreplication),
  correlate basin-median capacity ρ (IRD_at_reset) against basin-median operational
  variables. Spearman (robust to skew). A positive ρ–HL / ρ–t_flood correlation and
  negative ρ–α / ρ–t_dry correlation is the endogeneity signature.

TEST 2 — Zero-drying prevalence (Model 2 frac_zero_DrT context)
  How many events have effectively no drying (t_dry ≈ 0)? These are the extreme
  low-α operating points. If rare, that explains why frac_zero_DrT is a weak,
  unstable Model 2 feature.

Reads only the existing CSVs (event_dataset, reset_dataset). Nothing is trained.
Prints numbers for the Discussion; writes a small SI table for Test 1.

Usage:  python -m analysis.endogeneity_tests
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EVENT_CSV, RESET_CSV, TABLES_DIR, FIELD_NAMES

ZERO_DRY_THRESHOLD_H = 1.0   # t_dry below this (hours) counts as "no drying"


def _load_events() -> pd.DataFrame:
    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ["IRD_at_reset", "HL", "FT", "DrT", "ALPHA", "Ct", "IRD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["row_type"] == "event"].copy()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — basin-level endogeneity
# ─────────────────────────────────────────────────────────────────────────────

def test_endogeneity(ev: pd.DataFrame) -> pd.DataFrame:
    # one row per basin: median capacity + median operational variables
    g = ev.groupby("basin_number")
    basin = pd.DataFrame({
        "rho_med":    g["IRD_at_reset"].median(),
        "HL_med":     g["HL"].median()  if "HL"    in ev.columns else np.nan,
        "tflood_med": g["FT"].median()  if "FT"    in ev.columns else np.nan,
        "tdry_med":   g["DrT"].median() if "DrT"   in ev.columns else np.nan,
        "alpha_med":  g["ALPHA"].median() if "ALPHA" in ev.columns else np.nan,
        "n_events":   g.size(),
    }).dropna(subset=["rho_med"]).reset_index()

    print("=" * 70)
    print("  TEST 1 — Operator response to capacity (basin level, n={})".format(len(basin)))
    print("=" * 70)
    print("  Spearman correlation of basin-median capacity ρ vs basin-median:")

    targets = [
        ("HL_med",     "hydraulic load HL",     "positive expected (more capacity → more water applied)"),
        ("tflood_med", "flooding time t_flood", "positive expected (more capacity → longer flooding)"),
        ("tdry_med",   "drying time t_dry",     "negative expected (more capacity → less drying allowed)"),
        ("alpha_med",  "drying fraction α",     "negative expected (more capacity → lower α)"),
    ]
    rows = []
    for col, name, expect in targets:
        sub = basin[["rho_med", col]].dropna()
        if len(sub) < 5:
            continue
        rho_s, p = spearmanr(sub["rho_med"], sub[col])
        rows.append(dict(variable=name, spearman_rho=round(float(rho_s), 3),
                         p_value=float(p), n=len(sub), expectation=expect))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    ρ vs {name:<22}: rs={rho_s:+.3f}  p={p:.2e} {sig:>3}   [{expect}]")

    out = pd.DataFrame(rows)
    path = TABLES_DIR / "tableS_endogeneity.xlsx"
    out.to_excel(path, index=False)
    print(f"\n  Saved SI table: {path.name}")
    print("  Interpretation: significant correlations in the expected directions")
    print("  show HL / t_flood / t_dry / α are operator RESPONSES to capacity —")
    print("  i.e. endogenous — so their SHAP signs are not causal drivers.")
    return basin


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — zero-drying prevalence
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_drying(ev: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 2 — Zero-drying event prevalence")
    print("=" * 70)
    if "DrT" not in ev.columns:
        print("  DrT not in event dataset — skipping.")
        return
    dr = pd.to_numeric(ev["DrT"], errors="coerce").dropna()
    n_total = len(dr)
    n_zero  = int((dr <= ZERO_DRY_THRESHOLD_H).sum())
    print(f"  Events with t_dry ≤ {ZERO_DRY_THRESHOLD_H:.0f} h (no drying): "
          f"{n_zero} / {n_total}  ({100*n_zero/n_total:.1f}%)")
    # alpha at those events (should be ~0)
    if "ALPHA" in ev.columns:
        a = pd.to_numeric(ev.loc[dr.index, "ALPHA"], errors="coerce")
        zmask = dr <= ZERO_DRY_THRESHOLD_H
        print(f"  α at zero-drying events: median={a[zmask].median():.3f}  "
              f"(vs all-events median={a.median():.3f})")

    # reset-dataset frac_zero_DrT, for the Model 2 feature context
    if RESET_CSV.exists():
        rd = pd.read_csv(RESET_CSV)
        if "frac_zero_DrT" in rd.columns:
            f = pd.to_numeric(rd["frac_zero_DrT"], errors="coerce").dropna()
            print(f"  Model 2 feature frac_zero_DrT (per segment): "
                  f"median={f.median():.3f}  mean={f.mean():.3f}  "
                  f"share of segments with 0: {100*(f==0).mean():.1f}%")
    print("  Interpretation: zero-drying events are the extreme-low-α operating")
    print("  points; if rare, frac_zero_DrT carries little signal — consistent")
    print("  with its low, unstable Model 2 importance.")


def main() -> None:
    ev = _load_events()
    test_endogeneity(ev)
    test_zero_drying(ev)


if __name__ == "__main__":
    main()