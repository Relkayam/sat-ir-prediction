# smoke_3_normalization_argument.py
# Two basins, same physics, different Ks — raw vs normalized
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

lam     = 0.003
eta_inf = -1.2
T       = 400
t       = np.linspace(0, T, 30)
noise   = 0.06

# Basin 1: high Ks (rho = 8 cm/h)
# Basin 2: low Ks  (rho = 1.5 cm/h)
rho1, rho2 = 8.0, 1.5

eta_true = (0 - eta_inf) * np.exp(-lam * t) + eta_inf

ird1 = rho1 * np.exp(eta_true + np.random.normal(0, noise, len(t)))
ird2 = rho2 * np.exp(eta_true + np.random.normal(0, noise, len(t)))

eta1 = np.log(ird1 / rho1)
eta2 = np.log(ird2 / rho2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: raw IRD — looks completely different
ax = axes[0]
ax.scatter(t/24, ird1, color="#065A82", s=25, label=f"Basin 1  (ρ = {rho1} cm/h)")
ax.scatter(t/24, ird2, color="#E07B39", s=25, label=f"Basin 2  (ρ = {rho2} cm/h)")
ax.set_xlabel("LCT (days)")
ax.set_ylabel("IRD (cm/h)")
ax.set_title("Raw IRD — same physics, different scale\n"
             "A model trained here learns basin identity, not physics")
ax.legend()
ax.grid(True, alpha=0.25)

# Right: normalized — collapse to same curve
ax = axes[1]
ax.scatter(t/24, eta1, color="#065A82", s=25, label=f"Basin 1  (ρ = {rho1} cm/h)")
ax.scatter(t/24, eta2, color="#E07B39", s=25, label=f"Basin 2  (ρ = {rho2} cm/h)")

# True curve
t_smooth = np.linspace(0, T, 300)
eta_smooth = (0 - eta_inf) * np.exp(-lam * t_smooth) + eta_inf
ax.plot(t_smooth/24, eta_smooth, "k--", lw=1.5, alpha=0.6,
        label="True decay curve (shared)")
ax.axhline(eta_inf, color="red", lw=1.0, linestyle=":",
           alpha=0.7, label=f"η∞ = {eta_inf}")
ax.axhline(0, color="gray", lw=0.8, linestyle="-", alpha=0.4)

ax.set_xlabel("LCT (days)")
ax.set_ylabel("η(t) = ln(IRD / IRD_reset)")
ax.set_title("Normalized η — both basins collapse to the same curve\n"
             "Now a single global model can learn the physics")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

plt.suptitle("The normalization argument: why log-ratio targets enable global modeling",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.show()