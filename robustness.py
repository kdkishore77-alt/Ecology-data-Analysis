import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Parameters
# ======================================================
np.random.seed(42)
T = 200        # total simulation time
dt = 0.01      # time step
steps = int(T/dt)
n_species = 3  # number of species
m_patches = 2  # number of patches

# Growth rates
r = np.array([0.6, 0.5, 0.55])

# Competition matrix with small heterogeneity
alpha_base = np.array([[1.0, 0.4, 0.3],
                       [0.5, 1.0, 0.35],
                       [0.45, 0.4, 1.0]])
alpha = alpha_base + 0.05 * np.random.randn(*alpha_base.shape)

# Dispersal rates
d = 0.05

# Noise parameters
sigma = 0.1
theta_ou = 0.1
mu_ou = 0.0

# ======================================================
# Ornstein-Uhlenbeck noise generator
# ======================================================
def generate_ou(T, dt, theta, mu, sigma, size):
    steps = int(T/dt)
    x = np.zeros((steps, size))
    for t in range(1, steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.randn(size)
        x[t] = x[t-1] + dx
    return x

# ======================================================
# Simulation function
# ======================================================
def simulate(noise_type="white"):
    # Initial populations
    N = np.ones((steps, n_species, m_patches)) * 0.5

    # OU noise pre-generated if needed
    if noise_type == "ou":
        eta = generate_ou(T, dt, theta_ou, mu_ou, sigma, n_species*m_patches)
        eta = eta.reshape(steps, n_species, m_patches)
    else:
        eta = np.zeros((steps, n_species, m_patches))

    for t in range(1, steps):
        for i in range(n_species):
            for j in range(m_patches):
                interaction = np.sum(alpha[i, :] * N[t-1, :, j])
                dispersal = d * (np.sum(N[t-1, i, :]) - m_patches * N[t-1, i, j])

                # Noise perturbs intrinsic growth rate
                if noise_type == "white":
                    noise = sigma * np.random.randn()
                else:  # OU noise (already scaled by sigma in generator)
                    noise = eta[t, i, j]

                effective_r = r[i] + noise
                growth = N[t-1, i, j] * (effective_r - interaction)

                N[t, i, j] = N[t-1, i, j] + (growth + dispersal) * dt
                N[t, i, j] = max(N[t, i, j], 1e-6)  # avoid extinction < 0
    return N

# ======================================================
# Run simulations
# ======================================================
N_white = simulate(noise_type="white")
N_ou    = simulate(noise_type="ou")

# ======================================================
# Robustness plots
# ======================================================
time = np.arange(steps)*dt
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Panel A: trajectories under white noise (averaged across patches)
for i in range(n_species):
    axs[0].plot(time, np.mean(N_white[:,i,:], axis=1), lw=2, label=f"Species {i+1}")
axs[0].set_title("White noise dynamics", fontsize=14)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Population size")
axs[0].legend(frameon=False)
axs[0].grid(alpha=0.3)

# Panel B: trajectories under OU noise (averaged across patches)
for i in range(n_species):
    axs[1].plot(time, np.mean(N_ou[:,i,:], axis=1), lw=2, label=f"Species {i+1}")
axs[1].set_title("Ornstein–Uhlenbeck noise dynamics", fontsize=14)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Population size")
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("robustness.png", dpi=300, bbox_inches="tight")
plt.show()
