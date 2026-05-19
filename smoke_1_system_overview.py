# smoke_1_system_overview.py
# Run this locally — no data needed, purely synthetic
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Synthetic system parameters ──────────────────────────────────────────────
n_segments   = 4
lam          = 0.003          # decay rate h⁻¹ (realistic for Shafdan)
eta_inf      = -1.2           # asymptotic floor (log-ratio units)
reset_levels = [3.5, 4.1, 2.8, 3.9]   # IRD_at_reset per segment (cm/h)
seg_lengths  = [400, 350, 500, 300]    # LCT duration per segment (hours)
noise_sd     = 0.08           # event-level noise (log-ratio units)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ── Top panel: raw IRD time series ───────────────────────────────────────────
ax1 = axes[0]
t_global = 0
colors = plt.colormaps["viridis"].resampled(n_segments)

for i in range(n_segments):
    rho   = reset_levels[i]
    T     = seg_lengths[i]
    t_seg = np.linspace(0, T, 20)
    # IRD = rho * exp(lam*t_seg * (eta_inf stuff))  — full model below
    eta   = (0 - eta_inf) * np.exp(-lam * t_seg) + eta_inf
    ird   = rho * np.exp(eta + np.random.normal(0, noise_sd, len(t_seg)))

    t_abs = t_global + t_seg
    ax1.scatter(t_abs, ird, s=20, color=colors(i), zorder=3)
    ax1.plot(t_abs, ird, color=colors(i), lw=0.8, alpha=0.4)

    # Tillage line
    if i > 0:
        ax1.axvline(t_global, color="black", lw=1.2,
                    linestyle="--", alpha=0.6)
        ax1.annotate("Tillage", xy=(t_global, max(reset_levels)*0.95),
                    fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2",
                             facecolor="white", alpha=0.8))
    # IRD_reset marker
    ax1.scatter([t_global], [rho], s=80, color=colors(i),
               marker="*", zorder=5)

    t_global += T

ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("IRD (cm/h)")
ax1.set_title("Raw IRD time series — repeating decay-reset cycle\n"
              "★ = IRD_reset (post-tillage anchor)  |  "
              "dashed = tillage event")
ax1.grid(True, alpha=0.25)

# ── Bottom panel: normalized decay (eta vs LCT) ───────────────────────────────
ax2 = axes[1]

for i in range(n_segments):
    T     = seg_lengths[i]
    t_seg = np.linspace(0, T, 20)
    eta   = (0 - eta_inf) * np.exp(-lam * t_seg) + eta_inf
    eta_noisy = eta + np.random.normal(0, noise_sd, len(t_seg))

    # Smooth fit line
    t_smooth = np.linspace(0, T, 200)
    eta_smooth = (0 - eta_inf) * np.exp(-lam * t_smooth) + eta_inf

    ax2.scatter(t_seg/24, eta_noisy, s=20, color=colors(i), zorder=3,
               label=f"Segment {i+1}  (ρ={reset_levels[i]} cm/h)")
    ax2.plot(t_smooth/24, eta_smooth, color=colors(i), lw=1.5, alpha=0.8)

ax2.axhline(0,    color="black",  lw=1.0, linestyle="-",  alpha=0.5,
           label="η = 0  (reset baseline)")
ax2.axhline(eta_inf, color="red", lw=1.0, linestyle=":",  alpha=0.7,
           label=f"η∞ = {eta_inf}  (asymptotic floor)")
ax2.set_xlabel("LCT — time since last tillage (days)")
ax2.set_ylabel("η(t) = ln(IRD / IRD_reset)")
ax2.set_title("Normalized decay η(t) — all segments collapse to the same curve\n"
              "This is what Model 1 predicts")
ax2.legend(fontsize=8, loc="lower left")
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()