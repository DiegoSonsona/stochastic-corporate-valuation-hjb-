# Dynamic Stochastic Corporate Valuation Engine (HJB & Monte Carlo)

An advanced quantitative corporate valuation framework solving optimal capital investment under uncertainty using Hamilton-Jacobi-Bellman (HJB) partial differential equations and Euler-Maruyama Monte Carlo path generation.

## Overview
This engine models firm valuation in continuous time under stochastic capital accumulation. It combines analytical benchmark solutions with numerical stochastic differential equation (SDE) solvers to estimate firm value $V(K_0)$ and evaluate convergence under stochastic disturbance.

## Key Features
- **Hamilton-Jacobi-Bellman (HJB) Formulation**: Models optimal investment decision rules and firm valuation under capital adjustment costs.
- **Euler-Maruyama Path Generation**: Simulates continuous-time stochastic capital accumulation paths $K(t)$ driven by Brownian motion.
- **Monte Carlo Valuation Estimator**: Computes discounted cash flow returns across 10,000 simulated firm trajectories.
- **Analytical vs Numerical Benchmark**: Compares simulated firm value against closed-form theoretical solutions.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the valuation engine**:
   ```bash
   python stochastic_corporate_valuation.py
   ```
##Generated Artifacts
  hjb_simulation.png: High-resolution 4-panel diagnostic dashboard featuring capital trajectories, Euler-Maruyama numerical convergence, terminal capital distribution, and volatility sensitivity analytics.
## Project Structure
- `stochastic_corporate_valuation.py` — Main Python simulation pipeline
- `requirements.txt` — Project dependencies
- `README.md` — Repository landing page
- `DOCUMENTATION.md` — Mathematical foundation and HJB formulation
