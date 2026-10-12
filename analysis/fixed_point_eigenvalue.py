"""
analysis/fixed_point_eigenvalue.py
==================================
Reviewer C1 (opportunity): convert Section 4.2 from asserted framing to a
MEASURED result by estimating the Poincare-map slope / fixed-point eigenvalue
of the reset map.

The reset (Poincare) map sends one post-tillage ceiling to the next:
    log rho_i = c + a * log rho_{i-1} + noise
The slope `a` is the map's eigenvalue at the fixed point:
    |a| < 1  -> stable fixed point (perturbations decay; system self-restores)
    a = 0    -> full reset to a basin-typical level each tillage (no memory)
    a = 1    -> random walk (no restoring force)
Equivalently, in delta space (delta_i = log rho_i - log rho_{i-1}), a restoring
map shows negative dependence of delta_i on the previous level — the mean
reversion already visible in Model 2's SHAP (delta_{i-1} leads, negative sign).

We estimate `a` three ways for robustness:
  (1) pooled OLS of log rho_i on log rho_{i-1} (within basin, consecutive resets)
  (2) per-basin slopes, then median across basins (removes between-basin scale)
  (3) AR(1) on delta: delta_i ~ delta_{i-1}  (slope b; relates to mean reversion)

Uses reset_dataset.csv (has IRD_at_reset = rho_i and prev_IRD_at_reset = rho_{i-1}).

Run: python analysis/fixed_point_eigenvalue.py
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\user\PycharmProjects\sat-ir-prediction")
from config import RESET_CSV


def _ols_slope(x, y):
    """slope, intercept, r2 for y ~ x (finite pairs only)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())
    x, y = x[m], y[m]
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2), int(m.sum())


def main():
    print("=" * 70)
    print("  FIXED-POINT EIGENVALUE — reset (Poincare) map slope")
    print("=" * 70)
    df = pd.read_csv(RESET_CSV, low_memory=False)

    # rho_i and rho_{i-1}
    rho   = pd.to_numeric(df["IRD_at_reset"], errors="coerce")
    rho_p = pd.to_numeric(df["prev_IRD_at_reset"], errors="coerce")
    ok = (rho > 0) & (rho_p > 0)
    d = df.loc[ok].copy()
    d["logrho"]  = np.log(rho[ok].values)
    d["logrho_p"] = np.log(rho_p[ok].values)

    # (1) pooled OLS  log rho_i ~ log rho_{i-1}
    a1, c1, r2_1, n1 = _ols_slope(d["logrho_p"], d["logrho"])
    print(f"\n  (1) Pooled:      log rho_i = {c1:+.3f} + {a1:.3f}*log rho_(i-1)"
          f"   [R2={r2_1:.3f}, n={n1}]")
    print(f"      eigenvalue a = {a1:.3f}  ->  "
          f"{'STABLE (|a|<1)' if abs(a1) < 1 else 'NOT stable (|a|>=1)'}")

    # (2) per-basin slopes, median
    slopes = []
    for b, g in d.groupby("basin_number"):
        a, _, _, n = _ols_slope(g["logrho_p"], g["logrho"])
        if np.isfinite(a) and n >= 5:
            slopes.append(a)
    slopes = np.array(slopes)
    if len(slopes):
        print(f"\n  (2) Per-basin:   median a = {np.median(slopes):.3f}  "
              f"IQR [{np.percentile(slopes,25):.3f}, {np.percentile(slopes,75):.3f}]"
              f"  ({len(slopes)} basins)")

    # (3) AR(1) on delta: delta_i ~ delta_{i-1}
    if {"prev_delta", "prev_prev_delta"}.issubset(df.columns):
        # delta_i is IRD_norm_log_reset (= ln rho_i/rho_{i-1}); prev_delta = delta_{i-1}
        di   = pd.to_numeric(df["IRD_norm_log_reset"], errors="coerce")
        di_p = pd.to_numeric(df["prev_delta"], errors="coerce")
        b3, c3, r2_3, n3 = _ols_slope(di_p, di)
        print(f"\n  (3) AR(1) delta: delta_i = {c3:+.3f} + {b3:.3f}*delta_(i-1)"
              f"   [R2={r2_3:.3f}, n={n3}]")
        print(f"      slope {b3:+.3f} (negative => mean reversion, matches Model 2 SHAP)")

    print("\n  Interpretation: a stable fixed point (|a|<1) means each tillage")
    print("  pulls the ceiling back toward a basin-typical level; the closer a")
    print("  is to 0, the stronger the reset. This is the measured counterpart")
    print("  of the impulsive-dynamical-system framing in Section 4.2.")


if __name__ == "__main__":
    main()