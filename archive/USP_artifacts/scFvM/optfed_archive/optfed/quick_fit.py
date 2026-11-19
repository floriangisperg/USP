"""
Fast parameter fitting using local optimizer with good initial guess
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Load data
df = pd.read_excel('examples/doe_example/data/traintest_compiled.xlsx')

# Model (same as diagnostic_v2)
def model(params, t_eval, feed_rate_func, X0, P0, V0, induction_time=None):
    Y_XS, Y_PS, alpha, beta = params

    def derivatives(t, y):
        X, P, V = y
        f = feed_rate_func(t)

        if V <= 0 or f <= 0:
            dXdt = -(f / V) * X if V > 0 else 0
            dPdt = -(f / V) * P if V > 0 else 0
            dVdt = f
            return [dXdt, dPdt, dVdt]

        F_S = f * 400.0

        if induction_time is not None and t >= induction_time:
            prod_fraction = alpha + (beta / max(alpha, 1e-6) * 0.1)
            prod_fraction = min(prod_fraction, 0.9)
        else:
            prod_fraction = 0.0

        dXdt = Y_XS * F_S * (1 - prod_fraction) - (f / V) * X
        dPdt = Y_PS * F_S * prod_fraction - (f / V) * P
        dVdt = f

        return [dXdt, dPdt, dVdt]

    y0 = [X0, P0, V0]
    sol = solve_ivp(derivatives, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

    V = sol.y[2, :]
    return {"t": sol.t, "X_conc": sol.y[0, :] / V, "P_conc": sol.y[1, :] / V, "V": V}

# Prepare data
runs = []
for run_id in ['ferm_01', 'ferm_02']:
    run_df = df[df['RunID'] == run_id].copy().sort_values('time').reset_index(drop=True)
    meas_mask = ~np.isnan(run_df['X'].values)
    induction_time = run_df[run_df['inductor_switch'] > 0]['time'].min() if len(run_df[run_df['inductor_switch'] > 0]) > 0 else None

    time = run_df['time'].values
    feed_rate = np.maximum(np.gradient(run_df['feed'].values / 1000.0, time), 0)

    def feed_func(t, time=time, rate=feed_rate):
        if t <= time[0]: return rate[0]
        elif t >= time[-1]: return rate[-1]
        else: return np.interp(t, time, rate)

    runs.append({
        'id': run_id,
        'time': time,
        'X': run_df['X'].values,
        'P': run_df['P'].values,
        'V': run_df['reactor_volume'].values / 1000.0,
        'feed': feed_func,
        'induction': induction_time,
        'meas_idx': np.where(meas_mask)[0],
    })

# Initial guess from diagnostic
initial_guess = [0.5, 0.1, 0.15, 0.02]

# Objective function
def objective(params):
    total_error = 0
    n_points = 0

    for run in runs:
        try:
            first = run['meas_idx'][0]
            X0 = run['X'][first] * run['V'][first]
            P0 = run['P'][first] * run['V'][first] if not np.isnan(run['P'][first]) else 0.0
            V0 = run['V'][first]

            result = model(params, run['time'], run['feed'], X0, P0, V0, run['induction'])

            for idx in run['meas_idx']:
                X_pred = result['X_conc'][idx]
                X_true = run['X'][idx]
                if np.isfinite(X_pred) and np.isfinite(X_true):
                    total_error += (X_pred - X_true) ** 2
                    n_points += 1

        except:
            total_error += 1e6
            n_points += 1

    return total_error / max(1, n_points)

print("Initial objective:", objective(initial_guess))
print("Optimizing...")

# Use local optimizer (faster than differential evolution)
result = minimize(
    objective,
    initial_guess,
    method='L-BFGS-B',
    bounds=[(0.2, 0.8), (0.05, 0.5), (0.01, 0.4), (0.001, 0.1)],
    options={'maxiter': 200, 'disp': True}
)

print("\n" + "=" * 60)
print("FITTING RESULTS")
print("=" * 60)
print(f"Success: {result.success}")
print(f"Final objective: {result.fun:.4f}")
print(f"Iterations: {result.nit}")
print("\nFitted parameters:")
names = ['Y_XS', 'Y_PS', 'alpha', 'beta']
for name, val in zip(names, result.x):
    print(f"  {name}: {val:.4g}")

# Generate plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, run in enumerate(runs):
    first = run['meas_idx'][0]
    X0 = run['X'][first] * run['V'][first]
    P0 = run['P'][first] * run['V'][first] if not np.isnan(run['P'][first]) else 0.0
    V0 = run['V'][first]

    sim = model(result.x, run['time'], run['feed'], X0, P0, V0, run['induction'])

    # Biomass
    ax = axes[0, 0] if i == 0 else axes[0, 1]
    ax.plot(run['time'], run['X'], 'ko', label='Data', markersize=6)
    ax.plot(sim['t'], sim['X_conc'], '-', label='Model', color='orange', linewidth=2)
    if run['induction']:
        ax.axvline(run['induction'], color='red', linestyle='--', alpha=0.5, label='Induction')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Biomass (g/L)')
    ax.set_title(f"{run['id']} - Biomass")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Product
    ax = axes[1, 0] if i == 0 else axes[1, 1]
    if np.any(run['P'] > 0):
        ax.plot(run['time'], run['P'], 'ko', label='Data', markersize=6)
    ax.plot(sim['t'], sim['P_conc'], '-', label='Model', color='crimson', linewidth=2)
    if run['induction']:
        ax.axvline(run['induction'], color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Product (g/L)')
    ax.set_title(f"{run['id']} - Product")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/doe_example/optfed/quick_fit_results.png', dpi=150)
print("\nSaved: examples/doe_example/optfed/quick_fit_results.png")
