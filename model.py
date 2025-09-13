import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import matplotlib.pyplot as plt

# ------------------------------
# Model Parameters
# ------------------------------
n_species = 5  # Number of species (can increase)
r = np.linspace(0.5, 1.0, n_species)  # Base growth rates

# Heterogeneous competition matrix (random but diagonally dominant for stability)
np.random.seed(42)
alpha = np.abs(np.random.rand(n_species, n_species))
np.fill_diagonal(alpha, 1.0)

# Ornstein-Uhlenbeck (OU) noise parameters
theta = 0.5  # Mean reversion speed
sigma = 0.2  # Noise intensity
dt = 0.05    # Small timestep for OU updates

def update_eta(eta, mu=0.0):
    """Ornstein-Uhlenbeck update for temporal heterogeneity."""
    dW = np.random.normal(0, np.sqrt(dt), size=len(eta))
    return eta + theta * (mu - eta) * dt + sigma * dW

# ------------------------------
# Ecological Equilibrium Equations
# ------------------------------
def equilibrium_eq(N, r, alpha, eta):
    """Equilibrium equations for all species."""
    return [N[i] * (r[i] - np.dot(alpha[i], N) + eta[i]) for i in range(len(N))]

def jacobian_matrix(N_star, r, alpha, eta):
    """Jacobian at equilibrium N_star."""
    n_species = len(N_star)
    J = np.zeros((n_species, n_species))
    for i in range(n_species):
        for j in range(n_species):
            if i == j:
                J[i][j] = r[i] - np.dot(alpha[i], N_star) + eta[i] - alpha[i][j] * N_star[i]
            else:
                J[i][j] = -alpha[i][j] * N_star[i]
    return J

# ------------------------------
# Parameter Sweep: spatial (alpha scaling) vs temporal (OU mean)
# ------------------------------
eta_means = np.linspace(0.0, 1.0, 40)   # Temporal heterogeneity scale
alpha_scales = np.linspace(0.5, 2.0, 40)  # Spatial heterogeneity scale

stability_map = np.zeros((len(eta_means), len(alpha_scales)))

# Initial guess for equilibrium
N_star_initial = np.ones(n_species) * 0.5

for i, mu_eta in enumerate(eta_means):
    for j, scale in enumerate(alpha_scales):
        # Scale the alpha matrix to vary spatial heterogeneity
        alpha_scaled = alpha * scale

        # Simulate OU process for temporal heterogeneity
        eta = np.zeros(n_species)
        for _ in range(10):  # few iterations to approximate stationary state
            eta = update_eta(eta, mu=mu_eta)

        # Solve for equilibrium
        N_star, info, ier, msg = fsolve(equilibrium_eq, N_star_initial, args=(r, alpha_scaled, eta), full_output=True)

        if ier != 1 or np.any(N_star < 0):
            stability_map[i, j] = np.nan
            continue

        N_star_initial = N_star  # continuation

        # Compute Jacobian and eigenvalues
        J = jacobian_matrix(N_star, r, alpha_scaled, eta)
        eigenvalues = eigvals(J)
        stability_map[i, j] = max(np.real(eigenvalues))

# ------------------------------
# Plotting Results
# ------------------------------
plt.figure(figsize=(10, 6))
plt.contourf(alpha_scales, eta_means, stability_map, levels=np.linspace(-1, 1, 20), cmap="coolwarm")
plt.colorbar(label="Max Real Part of Eigenvalues")
plt.xlabel("Spatial Heterogeneity (α scaling)", fontsize=18)
plt.ylabel("Temporal Heterogeneity (OU mean)", fontsize=18)
plt.title(f"Stability Landscape for {n_species}-Species Community", fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("multi-species-OU-bifurcation.png")
plt.show()
