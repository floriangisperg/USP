"""
Fixed diagnostic script - corrected glucose-limited model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Load data
df = pd.read_excel('examples/doe_example/data/traintest_compiled.xlsx')

# CORRECTED glucose-limited model
def corrected_model(params, t_eval, feed_rate_func, X0, P0, V0, induction_time=None):
    """
    CORRECTED Glucose-limited fed-batch model.

    The key insight: In glucose-limited conditions, the growth is limited by
    substrate supply, not by biomass amount.

    Correct equations:
    - dX/dt = Y_XS * (f/V) * Sf - (f/V) * X
    - dP/dt = Y_PS * (f/V) * Sf * prod_fraction - (f/V) * P
    - dV/dt = f

    Where prod_fraction is the fraction of substrate going to product.

    params: [Y_XS, Y_PS, alpha, beta]
    - Y_XS: Biomass yield on substrate (gX/gS)
    - Y_PS: Product yield on substrate (gP/gS)
    - alpha: Growth-associated product fraction (0-1)
    - beta: Non-growth-associated product formation rate (1/h)
    """
    Y_XS, Y_PS, alpha, beta = params

    def derivatives(t, y):
        X, P, V = y  # Total biomass (g), total product (g), volume (L)

        f = feed_rate_func(t)  # Feed rate (L/h)
        Sf = 400.0  # Feed substrate concentration (g/L)

        if V <= 0 or f <= 0:
            # No feed = no growth
            dXdt = -(f / V) * X if V > 0 else 0
            dPdt = -(f / V) * P if V > 0 else 0
            dVdt = f
            return [dXdt, dPdt, dVdt]

        # Substrate inflow rate (gS/h)
        F_S = f * Sf

        # Substrate distribution
        # Total substrate uptake goes to: biomass + product + maintenance
        # We assume a fraction (alpha) goes to growth-associated product
        # and (1-alpha) goes to biomass

        # Biomass production rate (gX/h)
        # dX/dt = Y_XS * F_S * (1 - prod_fraction) - dilution
        # where prod_fraction depends on induction state

        if induction_time is not None and t >= induction_time:
            # After induction: allocate some substrate to product
            # Fraction to product = f(alpha, mu)
            # Simplified: use alpha as base fraction + beta-dependent term
            prod_fraction = alpha + (beta / (alpha + 1e-6))
            prod_fraction = min(prod_fraction, 0.9)  # Cap at 90%
        else:
            # Before induction: all to biomass
            prod_fraction = 0.0

        # Biomass accumulation
        dXdt_growth = Y_XS * F_S * (1 - prod_fraction)
        dXdt_dilution = (f / V) * X
        dXdt = dXdt_growth - dXdt_dilution

        # Product accumulation
        dPdt_growth = Y_PS * F_S * prod_fraction
        dPdt_dilution = (f / V) * P
        dPdt = dPdt_growth - dPdt_dilution

        dVdt = f

        return [dXdt, dPdt, dVdt]

    y0 = [X0, P0, V0]

    sol = solve_ivp(
        fun=derivatives,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    V = sol.y[2, :]
    X_conc = sol.y[0, :] / V
    P_conc = sol.y[1, :] / V

    return {
        "t": sol.t,
        "X_conc": X_conc,
        "P_conc": P_conc,
        "V": V,
        "X_total": sol.y[0, :],
        "P_total": sol.y[1, :],
    }

# Test on ferm_01
run_id = 'ferm_01'
run_df = df[df['RunID'] == run_id].copy()
run_df = run_df.sort_values('time').reset_index(drop=True)

# Get measurements
meas_mask = ~np.isnan(run_df['X'].values)
times_meas = run_df['time'].values[meas_mask]
X_meas = run_df['X'].values[meas_mask]
P_meas = run_df['P'].values[meas_mask]

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
P0 = P_meas[0] * run_df['reactor_volume'].values[0] / 1000.0 if not np.isnan(P_meas[0]) else 0.0
V0 = run_df['reactor_volume'].values[0] / 1000.0

print(f"Testing {run_id}:")
print(f"  Induction at: {induction_time:.2f} h")
print(f"  Feed rate range: {feed_rate.min():.4f} - {feed_rate.max():.4f} L/h")
print(f"  Initial conditions: X0={X0:.2f} g, P0={P0:.2f} g, V0={V0:.3f} L")
print(f"  Target final X: {X_meas[-1]:.2f} g/L")
print()

# Test different parameter sets
test_params = [
    ([0.5, 0.2, 0.1, 0.01], "Baseline"),
    ([0.5, 0.1, 0.15, 0.02], "More product"),
    ([0.4, 0.15, 0.2, 0.01], "Lower Y_XS"),
    ([0.5, 0.2, 0.05, 0.005], "Less product"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

best_rmse = float('inf')
best_params = None
best_label = None

for params, label in test_params:
    result = corrected_model(params, time, feed_func, X0, P0, V0, induction_time)

    # Plot biomass
    axes[0, 0].plot(result['t'], result['X_conc'], label=label, linewidth=2)
    axes[0, 1].plot(result['t'], result['P_conc'], label=label, linewidth=2)

    # Calculate error
    X_pred_at_meas = np.interp(times_meas, result['t'], result['X_conc'])
    rmse = np.sqrt(np.mean((X_pred_at_meas - X_meas)**2))

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
        best_label = label

    print(f"{label} {params}:")
    print(f"  Final X: {result['X_conc'][-1]:.2f} g/L (target: {X_meas[-1]:.2f})")
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

axes[0, 1].plot(times_meas, P_meas, 'ko', label='Data', markersize=8)
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
    result = corrected_model(params, time, feed_func, X0, P0, V0, induction_time)
    axes[1, 1].plot(result['t'], result['V'], '--', label=label)
axes[1, 1].axvline(induction_time, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Time (h)')
axes[1, 1].set_ylabel('Volume (L)')
axes[1, 1].set_title('Volume')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/doe_example/optfed/diagnostic_v2.png', dpi=150)
print(f"\nSaved: examples/doe_example/optfed/diagnostic_v2.png")
print(f"Best params: {best_label} {best_params} with RMSE={best_rmse:.3f}")
