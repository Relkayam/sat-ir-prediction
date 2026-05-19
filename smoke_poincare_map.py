# smoke_poincare_v2.py
# Normalized Poincaré map — removes between-basin Ks differences
# Shows within-basin attractor structure
# Includes 3D delay embedding (Takens reconstruction)

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESET_CSV, FIELD_NAMES
from plot_style import apply_style, COLORS, FONT, FIELD_COLORS

apply_style()

df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
df = df.loc[:, ~df.columns.duplicated()]
for col in ["IRD_at_reset","prev_IRD_at_reset_raw","DAR","mean_ALPHA"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Normalize per basin ───────────────────────────────────────────────────────
# rho_tilde = rho / median(rho_basin)
# Centers each basin at 1.0 — removes Ks scale difference
basin_medians = df.groupby("basin_number")["IRD_at_reset"].median()
df["rho_norm"] = df["IRD_at_reset"] / df["basin_number"].map(basin_medians)
df["rho_prev_norm"] = df["prev_IRD_at_reset_raw"] / \
                      df["basin_number"].map(basin_medians)

# Build 3-step delay embedding per basin
# rho_k, rho_{k-1}, rho_{k-2} — normalized
rows = []
for bn, bdf in df.groupby("basin_number"):
    bdf = bdf.sort_values("reset_date").copy()
    rho  = bdf["rho_norm"].values
    dar  = bdf["DAR"].values if "DAR" in bdf.columns else np.full(len(bdf), np.nan)
    field = FIELD_NAMES.get(int(str(int(bn))[0]), "")
    for i in range(2, len(bdf)):
        if (np.isfinite(rho[i]) and np.isfinite(rho[i-1]) and
                np.isfinite(rho[i-2])):
            rows.append(dict(
                basin=int(bn), field=field,
                r0=rho[i],     # current
                r1=rho[i-1],   # lag-1
                r2=rho[i-2],   # lag-2
                dar=dar[i],
                delta=np.log(rho[i]/rho[i-1]) if rho[i-1]>0 else np.nan,
            ))

emb = pd.DataFrame(rows).dropna(subset=["r0","r1","r2"])
print(f"Delay embedding: {len(emb)} reset triplets from "
      f"{emb['basin'].nunique()} basins")

# ── Statistics on normalized rho ─────────────────────────────────────────────
delta = emb["delta"].values
print(f"\nNormalized δ statistics:")
print(f"  mean   = {delta.mean():.4f}")
print(f"  std    = {delta.std():.4f}")
print(f"  % |δ| < 0.1  : {(np.abs(delta)<0.1).mean()*100:.1f}%")
print(f"  % |δ| < 0.2  : {(np.abs(delta)<0.2).mean()*100:.1f}%")
print(f"  % δ > +0.3   : {(delta>0.3).mean()*100:.1f}%  (strong recovery)")
print(f"  % δ < -0.3   : {(delta<-0.3).mean()*100:.1f}%  (strong degradation)")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))

# Panel 1: Normalized Poincaré map (2D)
ax1 = fig.add_subplot(131)
for field, color in FIELD_COLORS.items():
    mask = emb["field"] == field
    if not mask.any(): continue
    ax1.scatter(emb.loc[mask,"r1"], emb.loc[mask,"r0"],
                s=6, alpha=0.35, color=color,
                label=field, linewidths=0, zorder=3)

lo = max(0, np.percentile(emb["r1"],1)*0.85)
hi = np.percentile(emb["r1"],99)*1.15
ax1.plot([lo,hi],[lo,hi],"k--",lw=1.5,alpha=0.6,
         label="Fixed point (no change)")
ax1.set_xlim(lo,hi); ax1.set_ylim(lo,hi)
ax1.set_xlabel("ρ̃ₖ₋₁  (normalized)", fontsize=10)
ax1.set_ylabel("ρ̃ₖ  (normalized)", fontsize=10)
ax1.set_title("Normalized Poincaré map\n"
              "ρ̃ = ρ / median(ρ_basin)",
              fontsize=10, loc="left")
ax1.legend(fontsize=7, loc="upper left")
ax1.grid(True, alpha=0.20)
ax1.annotate(
    f"n={len(emb):,} reset pairs\n"
    f"{(np.abs(delta)<0.2).mean()*100:.0f}% within ±0.2 of fixed point",
    xy=(0.97,0.03), xycoords="axes fraction",
    fontsize=8, va="bottom", ha="right",
    bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
              edgecolor=COLORS["light_gray"],alpha=0.90))

# Panel 2: 3D delay embedding (Takens reconstruction)
ax2 = fig.add_subplot(132, projection="3d")
# Color by field
for field, color in FIELD_COLORS.items():
    mask = emb["field"] == field
    if not mask.any(): continue
    ax2.scatter(emb.loc[mask,"r2"],
                emb.loc[mask,"r1"],
                emb.loc[mask,"r0"],
                s=4, alpha=0.30, color=color,
                label=field, linewidths=0)

ax2.set_xlabel("ρ̃ₖ₋₂", fontsize=8, labelpad=2)
ax2.set_ylabel("ρ̃ₖ₋₁", fontsize=8, labelpad=2)
ax2.set_zlabel("ρ̃ₖ", fontsize=8, labelpad=2)
ax2.set_title("3D delay embedding\n(Takens reconstruction)",
              fontsize=10, loc="left")
ax2.tick_params(labelsize=7)

# Panel 3: δ vs DAR with running mean
ax3 = fig.add_subplot(133)
dar_v = emb["dar"].values
dlt_v = emb["delta"].values
valid = np.isfinite(dar_v) & np.isfinite(dlt_v)

sc = ax3.scatter(dar_v[valid], dlt_v[valid],
                 s=6, alpha=0.25,
                 c=dlt_v[valid], cmap="RdBu_r",
                 vmin=-0.5, vmax=0.5, linewidths=0, zorder=3)

# Running mean
idx_s  = np.argsort(dar_v[valid])
dar_s  = dar_v[valid][idx_s]
dlt_s  = dlt_v[valid][idx_s]
w = 300
if len(dar_s) > w:
    rm = np.convolve(dlt_s, np.ones(w)/w, mode="valid")
    ax3.plot(dar_s[w//2:-w//2+1], rm,
             color="black", lw=2.2, alpha=0.85,
             label=f"Running mean (n={w})", zorder=5)

ax3.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.5)
plt.colorbar(sc, ax=ax3, label="δ", fraction=0.030)

r_sp, _ = spearmanr(dar_v[valid], dlt_v[valid])
ax3.set_xlabel("DAR at tillage (W/m²)", fontsize=10)
ax3.set_ylabel("δ = ln(ρₖ/ρₖ₋₁)", fontsize=10)
ax3.set_title("δ vs DAR — Poincaré map driver\n"
              "High radiation → positive deviation from fixed point",
              fontsize=10, loc="left")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.20)
ax3.annotate(
    f"Spearman r = {r_sp:.3f}\n"
    "Operational insight:\ntill when radiation is high",
    xy=(0.97,0.03), xycoords="axes fraction",
    fontsize=8, va="bottom", ha="right",
    bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
              edgecolor=COLORS["light_gray"],alpha=0.90))

plt.suptitle(
    "Empirical Poincaré map of Shafdan SAT — normalized by basin median\n"
    "Each point = one tillage event  |  ρ̃ = ρ / median(ρ_basin) removes Ks differences",
    fontsize=11, y=1.02)
plt.tight_layout()
plt.show()