"""
Quick diagnostic script to understand model behavior
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Load data
df = pd.read_excel('examples/doe_example/data/traintest_compiled.xlsx')

# Simple glucose-limited model
def simple_model(params, t_eval, feed_rate_func, X0, P0, V0, induction_time=None):
    """
    Glucose-limited fed-batch model.

    params: [mu_max, Y_XS, alpha, beta, Y_PS]
    """
    mu_max, Y_XS, alpha, beta, Y_PS = params

    def derivatives(t, y):
        X, P, V = y

        f = feed_rate_func(t)

        # Substrate required per unit biomass/product
        sigma_X = 1.0 / Y_XS
        sigma_P = 1.0 / Y_PS

        # Growth rate limited by feed
        if V > 0:
            mu_feed = (f / V) * 400.0 / (sigma_X + alpha * sigma_P)
            mu = min(mu_feed, mu_max)
        else:
            mu = 0

        # Product formation (only after induction)
        if induction_time is not None and t >= induction_time:
            pi = alpha * mu + beta
        else:
            pi = 0.0

        dXdt = mu * X - (f / V) * X if V > 0 else 0
        dPdt = pi * X - (f / V) * P if V > 0 else 0
        dVdt = f

        return [dXdt, dPdt, dVdt]

    y0 = [X0, P0, V0]

    sol = solve_ivp(
        fun=derivatives,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
    )

    V = sol.y[2, :]
    X_conc = sol.y[0, :] / V
    P_conc = sol.y[1, :] / V

    return {
        "t": sol.t,
        "X_conc": X_conc,
        "P_conc": P_conc,
        "V": V,
    }

# Test on ferm_01
run_id = 'ferm_01'
run_df = df[df['RunID'] == run_id].copy()
run_df = run_df.sort_values('time').reset_index(drop=True)

# Get measurements
meas_mask = ~np.isnan(run_df['X'].values)
times_meas = run_df['time'].values[meas_mask]
X_meas = run_df['X'].values[meas_mask]

# Get induction time
induction_time = run_df[run_df['inductor_switch'] > 0]['time'].min()

# Create feed function
time = run_df['time'].values
feed_cumulative = run_df['feed'].values / 1000.0
feed_rate = np.gradient(feed_cumulative, time)
feed_rate = np.maximum(feed_rate, 0)

def feed_func(t):
    if t <= time[0]:
        return feed_rate[0]
    elif t >= time[-1]:
        return feed_rate[-1]
    else:
        return np.interp(t, time, feed_rate)

# Initial conditions (from first measurement)
X0 = X_meas[0] * run_df['reactor_volume'].values[0] / 1000.0
P0 = 0.0
V0 = run_df['reactor_volume'].values[0] / 1000.0

print(f"Testing {run_id}:")
print(f"  Induction at: {induction_time:.2f} h")
print(f"  Feed rate range: {feed_rate.min():.4f} - {feed_rate.max():.4f} L/h")
print(f"  Initial conditions: X0={X0:.2f} g, V0={V0:.3f} L")
print()

# Test different parameter sets
test_params = [
    ([0.5, 0.5, 0.05, 0.005, 0.2], "Default"),
    ([0.3, 0.5, 0.1, 0.01, 0.2], "Higher alpha/beta"),
    ([0.4, 0.4, 0.05, 0.005, 0.3], "Lower Y_XS, higher Y_PS"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for params, label in test_params:
    result = simple_model(params, time, feed_func, X0, P0, V0, induction_time)

    # Plot biomass
    axes[0, 0].plot(result['t'], result['X_conc'], label=label, linewidth=2)
    axes[0, 1].plot(result['t'], result['P_conc'], label=label, linewidth=2)

    # Calculate error
    X_pred_at_meas = np.interp(times_meas, result['t'], result['X_conc'])
    rmse = np.sqrt(np.mean((X_pred_at_meas - X_meas)**2))

    print(f"{label}:")
    print(f"  Final X: {result['X_conc'][-1]:.2f} g/L")
    print(f"  Final P: {result['P_conc'][-1]:.2f} g/L")
    print(f"  RMSE: {rmse:.3f} g/L")
    print()

# Plot data
axes[0, 0].plot(times_meas, X_meas, 'ko', label='Data', markersize=8)
axes[0, 0].axvline(induction_time, color='red', linestyle='--', alpha=0.5, label='Induction')
axes[0, 0].set_xlabel('Time (h)')
axes[0, 0].set_ylabel('Biomass (g/L)')
axes[0, 0].set_title(f'{run_id} - Biomass')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].axvline(induction_time, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Time (h)')
axes[0, 1].set_ylabel('Product (g/L)')
axes[0, 1].set_title(f'{run_id} - Product')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot feed rate
axes[1, 0].plot(time, feed_rate, 'b-', linewidth=1.5)
axes[1, 0].axvline(induction_time, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Time (h)')
axes[1, 0].set_ylabel('Feed Rate (L/h)')
axes[1, 0].set_title('Feed Rate Profile')
axes[1, 0].grid(True, alpha=0.3)

# Plot volume
V_data = run_df['reactor_volume'].values / 1000.0
axes[1, 1].plot(time, V_data, 'purple', linewidth=1.5, label='Data')
for params, label in test_params:
    result = simple_model(params, time, feed_func, X0, P0, V0, induction_time)
    axes[1, 1].plot(result['t'], result['V'], '--', label=label)
axes[1, 1].axvline(induction_time, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Time (h)')
axes[1, 1].set_ylabel('Volume (L)')
axes[1, 1].set_title('Volume')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/doe_example/optfed/quick_diagnostic.png', dpi=150)
print("Saved: examples/doe_example/optfed/quick_diagnostic.png")
