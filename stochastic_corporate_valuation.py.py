# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation for the HJB Corporate Valuation Model.
Structure based on Oosterlee and Grzelak (2020), Ch. 4.

Model:
    dK(t) = (I* - delta*K(t)) dt + sigma*K(t) dW(t)

Estimation of V(K0) by Monte Carlo:
    V(K0) = E[int_0^T e^{-rho*t} h(K,I*) dt] + E[e^{-rho*T} V(K(T))]

Convergence: Comparison with reference solution (dt_ref = dt/10^-4)

Diego Sonsona Carrillo
Undergraduate Thesis (TFG) — University of Zaragoza
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────
pi    = 0.4
delta = 0.1
c     = 2.0
rho   = 0.05
sigma = 0.2
K0    = 1.0

A           = pi / (rho + delta)
B           = (A - 1.0)**2 / (2.0 * c * rho)
I_opt       = (A - 1.0) / c
V_analitica = A * K0 + B

print("=" * 55)
print("ANALYTICAL SOLUTION")
print("=" * 55)
print(f"  A     = {A:.6f}  (capital shadow price)")
print(f"  B     = {B:.6f}  (constant term)")
print(f"  I*    = {I_opt:.6f}  (optimal investment)")
print(f"  V(K0) = {V_analitica:.6f}  (firm value)")
print()


# ─────────────────────────────────────────────────────────────
# 2. CORE FUNCTIONS (Oosterlee Style)
# ─────────────────────────────────────────────────────────────
def GeneratePathsHJBEuler(NoOfPaths, NoOfSteps, T,
                           I_opt, delta, sigma, K_0,
                           W_ext=None):
    """
    Generates K(t) trajectories using Euler-Maruyama.
    If W_ext (external Brownian motion) is provided, it uses it
    to compare different time steps on identical paths (convergence analysis).
    """
    dt   = T / float(NoOfSteps)
    time = np.linspace(0, T, NoOfSteps + 1)

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])

    K = np.zeros([NoOfPaths, NoOfSteps + 1])
    K[:, 0] = K_0

    for i in range(NoOfSteps):
        if W_ext is None:
            # Oosterlee-Grzelak Normalization
            if NoOfPaths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:, i+1] = W[:, i] + np.sqrt(dt) * Z[:, i]
        else:
            # Use external Brownian paths (convergence testing)
            W[:, i+1] = W_ext[:, i+1]

        Ki = K[:, i]
        K[:, i+1] = np.maximum(
            Ki + (I_opt - delta * Ki) * dt
            + sigma * Ki * (W[:, i+1] - W[:, i]),
            1e-8
        )

    return {"time": time, "K": K, "W": W}


def EstimarValorEmpresa(Paths, rho, pi, I_opt, c, A, B, T):
    """
    Estimates V(K0) using the Monte Carlo estimator:
        V(K0) ≈ E[sum_i e^{-rho*t_i} h(K_i, I*) dt]
               + E[e^{-rho*T} V(K(T))]
    """
    time     = Paths["time"]
    K        = Paths["K"]
    dt       = time[1] - time[0]
    N        = len(time) - 1
    retornos = np.zeros(K.shape[0])

    for i in range(N):
        h = pi * K[:, i] - I_opt - 0.5 * c * I_opt**2
        retornos += np.exp(-rho * time[i]) * h * dt

    # Discounted terminal value
    retornos += np.exp(-rho * T) * (A * K[:, -1] + B)

    V_MC  = np.mean(retornos)
    error = np.std(retornos) / np.sqrt(len(retornos))
    return V_MC, error


# ─────────────────────────────────────────────────────────────
# 3. MAIN SIMULATION
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

NoOfPaths = 10000
NoOfSteps = 5000
T         = 50

Paths    = GeneratePathsHJBEuler(NoOfPaths, NoOfSteps, T,
                                  I_opt, delta, sigma, K0)
timeGrid = Paths["time"]
K_tray   = Paths["K"]

V_MC, error_std = EstimarValorEmpresa(
    Paths, rho, pi, I_opt, c, A, B, T)

print("=" * 55)
print("MONTE CARLO ESTIMATION")
print("=" * 55)
print(f"  V(K0) Analytical  = {V_analitica:.6f}")
print(f"  V(K0) Monte Carlo = {V_MC:.6f}")
print(f"  Standard Error    = {error_std:.6f}")
print(f"  95% CI            = [{V_MC-1.96*error_std:.6f},"
      f" {V_MC+1.96*error_std:.6f}]")
print(f"  Relative Error    = "
      f"{abs(V_MC - V_analitica)/V_analitica*100:.4f}%")
print()


# ─────────────────────────────────────────────────────────────
# 4. CONVERGENCE ANALYSIS (reference solution with dt_ref = dt/10^-4)
# ─────────────────────────────────────────────────────────────
print("Computing convergence...")
NoOfPaths_c   = 500
T_conv        = 1.0
NoOfSteps_ref = 10000
dt_values     = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
errores_conv  = []

# Generate reference Brownian paths once
np.random.seed(123)
Z_ref = np.random.normal(0.0, 1.0, [NoOfPaths_c, NoOfSteps_ref])
for j in range(NoOfSteps_ref):
    if NoOfPaths_c > 1:
        Z_ref[:,j] = ((Z_ref[:,j] - np.mean(Z_ref[:,j]))
                      / np.std(Z_ref[:,j]))
dt_ref = T_conv / NoOfSteps_ref
W_ref  = np.zeros([NoOfPaths_c, NoOfSteps_ref + 1])
for j in range(NoOfSteps_ref):
    W_ref[:, j+1] = W_ref[:, j] + np.sqrt(dt_ref) * Z_ref[:, j]

# Reference solution with dt_ref
K_ref = np.zeros([NoOfPaths_c, NoOfSteps_ref + 1])
K_ref[:, 0] = K0
for j in range(NoOfSteps_ref):
    Kr = K_ref[:, j]
    K_ref[:, j+1] = np.maximum(
        Kr + (I_opt - delta*Kr)*dt_ref
        + sigma*Kr*(W_ref[:,j+1]-W_ref[:,j]), 1e-8)
K_ref_T = K_ref[:, -1]

for dt_c in dt_values:
    NoOfSteps_c = int(T_conv / dt_c)
    idx = np.linspace(0, NoOfSteps_ref, NoOfSteps_c+1, dtype=int)
    W_coarse = W_ref[:, idx]
    K_c = np.zeros([NoOfPaths_c, NoOfSteps_c + 1])
    K_c[:, 0] = K0
    for j in range(NoOfSteps_c):
        Kc = K_c[:, j]
        K_c[:, j+1] = np.maximum(
            Kc + (I_opt - delta*Kc)*dt_c
            + sigma*Kc*(W_coarse[:,j+1]-W_coarse[:,j]), 1e-8)
    errores_conv.append(np.mean(np.abs(K_c[:,-1] - K_ref_T)))

errores_conv = np.array(errores_conv)
print("Convergence calculated.")


# ─────────────────────────────────────────────────────────────
# 5. SENSITIVITY ANALYSIS TO SIGMA
# ─────────────────────────────────────────────────────────────
sigmas  = [0.0, 0.1, 0.2, 0.3, 0.4]
colores = ['black', 'steelblue', 'forestgreen',
           'darkorange', 'crimson']
medias_sigma = {}

for s in sigmas:
    P_s = GeneratePathsHJBEuler(5000, 1000, T, I_opt, delta, s, K0)
    medias_sigma[s] = np.mean(P_s["K"], axis=0)

time_sigma = P_s["time"]


# ─────────────────────────────────────────────────────────────
# 6. PLOTS AND VISUALIZATION
# ─────────────────────────────────────────────────────────────
K_star = I_opt / delta

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    'Monte Carlo Simulation — HJB Model\n'
    f'$\\pi={pi}$, $\\delta={delta}$, $c={c}$, '
    f'$\\rho={rho}$, $\\sigma={sigma}$, $K_0={K0}$',
    fontsize=11
)

# ── Panel 1: Trajectories ────────────────────────────────────
ax = axes[0, 0]
for i in range(min(30, NoOfPaths)):
    ax.plot(timeGrid, K_tray[i], color='steelblue',
            alpha=0.12, linewidth=0.5)
ax.plot(timeGrid, np.mean(K_tray, axis=0), 'k-',
        linewidth=2, label='Sample mean')
ax.axhline(K_star, color='crimson', linestyle='--',
           linewidth=1.5, label=f'$K^*={K_star:.3f}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$K(t)$')
ax.set_title('Capital Trajectories $K(t)$')
ax.legend(fontsize=8)
ax.set_xlim([0, T])

# ── Panel 2: Convergence ────────────────────────────────────
ax = axes[0, 1]
ax.loglog(dt_values, errores_conv, 'o-', color='steelblue',
          linewidth=1.5, markersize=5, label='Numerical error')
ref_x = dt_values
ref_y = errores_conv[-1] * np.sqrt(dt_values / dt_values[-1])
ax.loglog(ref_x, ref_y, 'k--', linewidth=1,
          label=r'$O((\Delta t)^{1/2})$')
ax.set_xlabel('$\\Delta t$')
ax.set_ylabel('$e(∆t)$')
ax.set_title('Euler-Maruyama Scheme Convergence')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

# ── Panel 3: Distribution of K(T) ────────────────────────────
ax = axes[1, 0]
ax.hist(K_tray[:, -1], bins=60, color='steelblue',
        alpha=0.7, density=True, edgecolor='white',
        linewidth=0.3)
ax.axvline(np.mean(K_tray[:, -1]), color='black',
           linewidth=2,
           label=f'Mean = {np.mean(K_tray[:,-1]):.3f}')
ax.axvline(K_star, color='crimson', linestyle='--',
           linewidth=1.5, label=f'$K^* = {K_star:.3f}$')
ax.set_xlabel(f'$K({T})$')
ax.set_ylabel('Density')
ax.set_title(f'Distribution of $K(T)$, $T = {T}$')
ax.legend(fontsize=8)

# ── Panel 4: Effect of Sigma ──────────────────────────────────
ax = axes[1, 1]
for s, col in zip(sigmas, colores):
    ax.plot(time_sigma, medias_sigma[s],
            color=col, linewidth=1.5,
            label=f'$\\sigma = {s}$')
ax.axhline(K_star, color='gray', linestyle=':',
           linewidth=1, label=f'$K^* = {K_star:.3f}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$\\mathbb{E}[K(t)]$')
ax.set_title('Effect of $\\sigma$ on $\\mathbb{E}[K(t)]$')
ax.legend(fontsize=8)
ax.set_xlim([0, T])

plt.tight_layout()
plt.savefig('hjb_simulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plots saved successfully.")