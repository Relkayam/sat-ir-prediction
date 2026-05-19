# smoke_4_normalization_real_data.py
# Demonstrates the between-basin scale problem and its solution
# Uses real event_dataset.csv — no synthetic data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
df = pd.read_csv(PROJECT_ROOT / "data" / "event_dataset.csv",
                 parse_dates=["opening_valve_date"])
df = df.loc[:, ~df.columns.duplicated()]

if "IRD_norm_log" not in df.columns and "IRD_norm" in df.columns:
    df["IRD_norm_log"] = df["IRD_norm"]

for col in ["LCT", "IRD", "IRD_at_reset", "IRD_norm_log"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Good events only, two contrasting basins
BASINS = [5201, 4104]   # median rho: 6.29 vs 1.68 cm/h
COLORS = ["#065A82", "#E07B39"]
LABELS = ["Basin 5201 — Yavne 2\n(median IRD_reset = 6.3 cm/h)",
          "Basin 4104 — Yavne 1\n(median IRD_reset = 1.7 cm/h)"]

good = df[
    (df["row_type"] == "event") &
    (df["is_good_segment"] == True) &
    (df["basin_number"].isin(BASINS))
].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: raw IRD vs LCT ──────────────────────────────────────────────────────
ax = axes[0]
for bn, color, label in zip(BASINS, COLORS, LABELS):
    bdf = good[good["basin_number"] == bn]
    lct_d = bdf["LCT"] / 24
    ax.scatter(lct_d, bdf["IRD"], s=8, alpha=0.45,
               color=color, label=label, linewidths=0)

ax.set_xlabel("Time since last tillage, LCT (days)", fontsize=12)
ax.set_ylabel("IRD (cm/h)", fontsize=12)
ax.set_title("Raw IRD — same clogging physics,\ndifferent hydraulic conductivity scale",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.22)
ax.text(0.04, 0.06,
        "A model trained on raw IRD\nlearns basin identity,\nnot clogging physics.",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD", alpha=0.9))

# ── Right: normalized IRD_norm_log vs LCT ────────────────────────────────────
ax = axes[1]
for bn, color, label in zip(BASINS, COLORS, LABELS):
    bdf = good[good["basin_number"] == bn]
    lct_d = bdf["LCT"] / 24
    ax.scatter(lct_d, bdf["IRD_norm_log"], s=8, alpha=0.45,
               color=color, label=label, linewidths=0)

ax.axhline(0, color="black", lw=1.0, linestyle="-", alpha=0.4,
           label="η = 0  (post-tillage baseline)")

# Overlay median exponential fit across all good segments system-wide
all_good = df[
    (df["row_type"] == "event") &
    (df["is_good_segment"] == True)
].copy()
med_lam = all_good.groupby(
    ["basin_number","segment_id"])["seg_lambda"].first().median()
med_b   = all_good.groupby(
    ["basin_number","segment_id"])["seg_b"].first().median()
if np.isfinite(med_lam) and np.isfinite(med_b):
    t_h = np.linspace(0, 600, 400)
    eta_fit = (0 - med_b) * np.exp(-med_lam * t_h) + med_b
    ax.plot(t_h/24, eta_fit, "k--", lw=1.8, alpha=0.55,
            label=f"Median system fit\n(λ={med_lam*24:.3f} d⁻¹)")

ax.set_xlabel("Time since last tillage, LCT (days)", fontsize=12)
ax.set_ylabel("η(t) = ln(IRD / IRD_reset)", fontsize=12)
ax.set_title("Normalized IRD — both basins collapse\nto the same decay structure",
             fontsize=11)
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, alpha=0.22)
ax.text(0.04, 0.06,
        "After normalization, a single\nglobal model can learn\nthe physics across all basins.",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#D5E8D4", alpha=0.9))

plt.suptitle(
    "Figure M1. The normalization argument: log-ratio targets enable global modeling\n"
    "across basins with different saturated hydraulic conductivity",
    fontsize=11, y=1.02)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary statistics:")
for bn in BASINS:
    bdf = good[good["basin_number"] == bn]
    print(f"\n  Basin {bn}:")
    print(f"    n events          : {len(bdf)}")
    print(f"    IRD range         : {bdf['IRD'].min():.2f} – "
          f"{bdf['IRD'].max():.2f} cm/h")
    print(f"    IRD_reset range   : {bdf['IRD_at_reset'].min():.2f} – "
          f"{bdf['IRD_at_reset'].max():.2f} cm/h")
    print(f"    η range           : {bdf['IRD_norm_log'].min():.3f} – "
          f"{bdf['IRD_norm_log'].max():.3f}")
    print(f"    Median η at end   : "
          f"{bdf.groupby('segment_id')['IRD_norm_log'].last().median():.3f}")