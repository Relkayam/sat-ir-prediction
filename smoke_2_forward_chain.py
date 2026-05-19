# smoke_2_forward_chain.py
# Shows how Model 2 output feeds Model 1 — no data needed
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# Simulate 3 consecutive segments
# True values
rho_true  = [3.5, 4.1, 2.9]
lam_true  = [0.003, 0.0025, 0.0035]
eta_inf   = -1.2
T         = 400   # hours per segment

# Model predictions (add small error)
rho_pred  = [r * np.exp(np.random.normal(0, 0.08)) for r in rho_true]
lam_pred  = [l * np.exp(np.random.normal(0, 0.10)) for l in lam_true]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig.suptitle("Forward chain: Model 2 → ρ̂ → Model 1 → IRD trajectory\n"
             "Blue = true  |  Orange dashed = predicted", fontsize=11)

t = np.linspace(0, T, 200)

for i, ax in enumerate(axes):
    # True trajectory
    eta_true = (0 - eta_inf) * np.exp(-lam_true[i] * t) + eta_inf
    ird_true = rho_true[i] * np.exp(eta_true)

    # Predicted trajectory (uses predicted rho AND predicted lambda)
    eta_pred = (0 - eta_inf) * np.exp(-lam_pred[i] * t) + eta_inf
    ird_pred = rho_pred[i] * np.exp(eta_pred)

    ax.plot(t/24, ird_true, color="#065A82", lw=2.0,
            label=f"True  (ρ={rho_true[i]:.1f})")
    ax.plot(t/24, ird_pred, color="#E07B39", lw=2.0,
            linestyle="--",
            label=f"Predicted  (ρ̂={rho_pred[i]:.2f})")

    # Mark IRD_reset levels
    ax.axhline(rho_true[i], color="#065A82", lw=0.8,
               linestyle=":", alpha=0.6)
    ax.axhline(rho_pred[i], color="#E07B39", lw=0.8,
               linestyle=":", alpha=0.6)

    ax.set_title(f"Segment {i+1}\n"
                 f"Model 2 error: {100*(rho_pred[i]/rho_true[i]-1):+.1f}%",
                 fontsize=10)
    ax.set_xlabel("LCT (days)")
    if i == 0:
        ax.set_ylabel("IRD (cm/h)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()