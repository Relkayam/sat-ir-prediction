"""
delta_stats_check.py — Confirm the Model 2 target (δ) descriptive statistics
============================================================================
Verifies the three numbers used in §2.3 / §3.3:
  - mean δ  (draft: -0.0003)
  - sd δ    (draft: 0.26)
  - share of resets within 10% of the previous ceiling, |δ| < 0.10 (draft: 62%)

δ = ln(ρᵢ / ρᵢ₋₁) is the Model 2 target. Reads the reset dataset only; trains
nothing. Reports n so we know the sample it's computed over.

Usage:  python -m analysis.delta_stats_check
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import RESET_CSV
from pipeline.features import TARGET_M2

rd = pd.read_csv(RESET_CSV)
rd = rd.loc[:, ~rd.columns.duplicated()]

# δ is the Model 2 target column
if TARGET_M2 in rd.columns:
    d = pd.to_numeric(rd[TARGET_M2], errors="coerce").dropna()
    src = TARGET_M2
else:
    # fallback: compute from consecutive ceilings if target column absent
    src = "(computed fallback)"
    d = None

print("=" * 60)
print("  δ (Model 2 target) descriptive statistics")
print("=" * 60)
print(f"  Source column : {src}")
if d is not None:
    n = len(d)
    within10 = float((d.abs() < 0.10).mean()) * 100
    print(f"  n resets      : {n}")
    print(f"  mean δ        : {d.mean():+.4f}   (draft: -0.0003)")
    print(f"  sd   δ        : {d.std():.4f}   (draft: 0.26)")
    print(f"  median δ      : {d.median():+.4f}")
    print(f"  |δ| < 0.10    : {within10:.1f}%   (draft: 62%)")
    print(f"  min / max     : {d.min():+.3f} / {d.max():+.3f}")
else:
    print("  TARGET_M2 column not found — check reset dataset columns:")
    print(" ", list(rd.columns))