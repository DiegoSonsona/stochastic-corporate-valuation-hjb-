# Dynamic Stochastic Corporate Valuation Engine — Technical Documentation

## System Architecture

The pipeline models firm valuation under stochastic capital accumulation and optimal investment rules, consisting of 4 core components:

### 1. Capital Accumulation Dynamics
Firm capital $K(t)$ evolves according to the stochastic differential equation (SDE):
$$dK(t) = (I^* - \delta K(t)) dt + \sigma K(t) dW(t)$$

Where:
- $I^*$: Optimal investment rate derived from adjustment cost minimization.
- $\delta$: Capital depreciation rate.
- $\sigma$: Volatility of capital shocks.
- $W(t)$: Standard Wiener process (Brownian motion).

### 2. Hamilton-Jacobi-Bellman (HJB) Value Function
Continuous-time firm valuation solves the HJB partial differential equation:
$$\rho V(K) = \max_I \left\{ \pi K - I - \frac{c}{2} I^2 + (I - \delta K) V'(K) + \frac{1}{2} \sigma^2 K^2 V''(K) \right\}$$

With closed-form linear solution $V(K) = A K + B$, where:
$$A = \frac{\pi}{\rho + \delta}, \quad B = \frac{(A - 1)^2}{2 c \rho}, \quad I^* = \frac{A - 1}{c}$$

### 3. Numerical Scheme & Path Generation
Path trajectories are discretized using the **Euler-Maruyama scheme** over horizon $T = 50$ with $M = 10,000$ paths and $N = 5,000$ time steps.

### 4. Monte Carlo Valuation Estimator
Firm value is estimated by numerical integration of discounted operating cash flows:
$$\hat{V}_M(K_0) = \frac{1}{M} \sum_{j=1}^M \left[ \sum_{i=1}^N e^{-\rho t_i} h(K_i^j, I^*) \Delta t + e^{-\rho T} V(K_N^j) \right]$$

---

## Technical Limitations & Considerations

1. **Static Parameter Assumption**: Parameters ($\pi, \delta, \rho, \sigma$) are assumed constant over the entire simulation horizon.
2. **Quadratic Adjustment Friction**: Capital adjustment costs $c$ are quadratic; non-linear frictions or irreversible investment bounds are not modeled.
3. **Discretization Bias**: Euler-Maruyama introduces strong order $O((\Delta t)^{1/2})$ discretization error, requiring sufficiently small time steps $\Delta t$ for accurate convergence.