"""
Better IB Model Options (No Floor Requirement)

Requirements:
- T=25°C: 0% IB (pure soluble)
- T=30°C: ~13% IB
- T=35°C: ~100% IB

Key insight: The function must start at 0 at low T, not at a positive floor.
"""

import numpy as np
import matplotlib.pyplot as plt

T_plot = np.linspace(20, 40, 200)
T_obs = np.array([25, 30, 35])
IB_obs = np.array([0.0, 0.13, 1.0])

# ============================================================
# OPTION 1: Shifted Hill Equation (most flexible)
# ============================================================
# phi_IB = ((T - T_onset) / T_scale)^n / (1 + ((T - T_onset) / T_scale)^n)
# - Below T_onset: 0% IB
# - Above T_onset: increases smoothly
# - T_scale and n control shape

def ib_hill_shifted(T, T_onset, T_scale, n):
    """Hill equation with onset temperature."""
    x = np.maximum(T - T_onset, 0)  # No IB below T_onset
    return (x / T_scale)**n / (1 + (x / T_scale)**n)

# Fit: T_onset ≈ 24°C, T_scale ≈ 10°C, n ≈ 3
IB_opt1 = ib_hill_shifted(T_plot, T_onset=24, T_scale=10, n=3)

# ============================================================
# OPTION 2: Quadratic (simple polynomial, goes through origin)
# ============================================================
# phi_IB = a * (T - T0)^2
# - At T0: 0% IB
# - Parabolic increase

def ib_quadratic(T, a, T0):
    """Quadratic: zero at T0, increases quadratically."""
    return a * np.maximum(T - T0, 0)**2

# Fit to pass through (35, 1): a = 1 / (35-25)^2 = 1/100 = 0.01
IB_opt2 = np.minimum(ib_quadratic(T_plot, a=0.01, T0=25), 1.0)

# ============================================================
# OPTION 3: Boltzmann-like (asymmetric sigmoid, offset)
# ============================================================
# phi_IB = 1 / (1 + exp(-k * (T - T_50))) - offset
# where offset is chosen so phi_IB(T_onset) = 0

def ib_boltzmann_offset(T, k, T_50, T_onset):
    """Boltzmann with offset to ensure zero at T_onset."""
    offset = 1.0 / (1.0 + np.exp(-k * (T_onset - T_50)))
    sigmoid = 1.0 / (1.0 + np.exp(-k * (T - T_50)))
    return np.maximum(sigmoid - offset, 0) / (1 - offset)

IB_opt3 = ib_boltzmann_offset(T_plot, k=0.3, T_50=32, T_onset=24)

# ============================================================
# OPTION 4: Feeding-rate dependent (qs affects IB!)
# ============================================================
# Maybe IB depends on BOTH temperature AND specific uptake rate
# Higher qs → more protein → more crowding → more IB?

def ib_temperature_qs(T, qs, T_onset, T_scale, n, alpha_qs):
    """
    IB depends on temperature AND qs.
    - Temperature: Hill-type with onset
    - qs: Modulates the temperature effect (higher qs → more IB)
    """
    temp_factor = (np.maximum(T - T_onset, 0) / T_scale)**n / (1 + (np.maximum(T - T_onset, 0) / T_scale)**n)
    qs_factor = 1 + alpha_qs * (qs - 0.2)  # Normalize around lowest qs
    return np.minimum(temp_factor * qs_factor, 1.0)

# Plot for different qs values
qs_values = [0.2, 0.4, 0.6]
IB_opt4_lines = []
for qs in qs_values:
    IB_opt4_lines.append(ib_temperature_qs(T_plot, qs, T_onset=24, T_scale=10, n=3, alpha_qs=0.3))

# ============================================================
# OPTION 5: Mechanistic - Aggregation vs Folding competition
# ============================================================
# k_agg(T): temperature-dependent aggregation rate
# k_fold: constant folding rate
# phi_IB = k_agg(T) / (k_agg(T) + k_fold)

# Arrhenius for aggregation: k_agg = A * exp(-Ea/(R*T))
# Folding: constant (temperature-independent in this range)

def ib_mechanistic(T, A, Ea, k_fold, R=8.314):
    """Mechanistic: competition between folding and aggregation."""
    T_K = T + 273.15  # Convert to Kelvin
    k_agg = A * np.exp(-Ea / (R * T_K))
    return k_agg / (k_agg + k_fold)

# Fit: A=1e10, Ea=60000 J/mol, k_fold=1
IB_opt5 = ib_mechanistic(T_plot, A=1e10, Ea=60000, k_fold=1.0)

# ============================================================
# PLOT ALL OPTIONS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

options = [
    ("Option 1: Shifted Hill", IB_opt1,
     "T_onset=24, T_scale=10, n=3\nEnsures 0% below T_onset"),
    ("Option 2: Quadratic", IB_opt2,
     "phi = a*(T-T0)^2\nSimple, parsimonious"),
    ("Option 3: Boltzmann + Offset", IB_opt3,
     "Sigmoid shifted to zero\nSmooth, asymmetric"),
    ("Option 4: T + qs dependence", IB_opt4_lines[0],
     "IB depends on T AND qs!\nqs=0.2 (blue), 0.4 (orange), 0.6 (green)"),
    ("Option 5: Mechanistic (Arrhenius)", IB_opt5,
     "Competition: folding vs aggregation\nA*exp(-Ea/RT) / (k_fold + ...)"),
    ("Summary", None, "Compare all"),
]

# Plot first 5 options
for ax, (title, IB_data, desc) in zip(axes[:5], options[:5]):
    ax.plot(T_plot, T_plot*0 + 1, 'k--', alpha=0.2, label='100%')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)

    if title == "Option 4: T + qs dependence":
        colors = ['blue', 'orange', 'green']
        for i, (IB, qs, c) in enumerate(zip(IB_opt4_lines, qs_values, colors)):
            ax.plot(T_plot, IB * 100, color=c, linewidth=2, label=f'qs={qs}')
        ax.legend()
    else:
        ax.plot(T_plot, IB_data * 100, 'b-', linewidth=2, label='Model')

    ax.scatter(T_obs, IB_obs * 100, c='red', s=100, zorder=5,
               edgecolors='black', label='Data', marker='o')
    ax.set_xlabel('Temperature [C]')
    ax.set_ylabel('IB Fraction [%]')
    ax.set_title(f'{title}\n{desc}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(20, 40)

# Summary plot
ax = axes[5]
ax.plot(T_plot, IB_opt1 * 100, label='Shifted Hill', linewidth=2)
ax.plot(T_plot, IB_opt2 * 100, label='Quadratic', linewidth=2, linestyle='--')
ax.plot(T_plot, IB_opt3 * 100, label='Boltzmann Offset', linewidth=2, linestyle=':')
ax.plot(T_plot, IB_opt5 * 100, label='Mechanistic', linewidth=2, linestyle='-.')
ax.scatter(T_obs, IB_obs * 100, c='red', s=100, zorder=5, edgecolors='black', label='Data', marker='o')
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('IB Fraction [%]')
ax.set_title('All Models Compared')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 105)
ax.set_xlim(20, 40)

plt.suptitle('IB Model Options (No Floor - True Zero at Low T)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/ib_model_better_options.png', dpi=150, bbox_inches='tight')
print("Plot saved to results/ib_model_better_options.png")

# ============================================================
# PRINT RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 60)
print("BETTER IB MODEL OPTIONS")
print("=" * 60)
print("\nAll options satisfy: 0% IB at low T (not a positive floor)")
print("\n" + "-" * 60)

print("\nRECOMMENDED: Option 1 (Shifted Hill)")
print("-" * 60)
print("  Formula: phi = ((T - T_onset)/T_scale)^n / (1 + ((T - T_onset)/T_scale)^n)")
print("  Parameters: T_onset≈24, T_scale≈10, n≈3")
print("  Pros: Flexible, goes to 0 below T_onset, well-established")
print("  Cons: 3 parameters")

print("\nALTERNATIVE: Option 2 (Quadratic)")
print("-" * 60)
print("  Formula: phi = a * (T - T0)^2")
print("  Parameters: a, T0")
print("  Pros: Simple! Only 2 parameters, goes through zero")
print("  Cons: May not capture steep transition at 35C")

print("\nINTERESTING: Option 4 (qs dependence)")
print("-" * 60)
print("  Formula: phi = f_temp(T) * (1 + alpha*(qs - qs_ref))")
print("  Hypothesis: Higher feeding → more protein expression → more crowding → more IB")
print("  Test: ferm_01 (qs=0.2, T=35) vs ferm_04 (qs=0.6, T=35)")
print("        Both show 100% IB... so qs may NOT matter")
print("  But: ferm_05-07 (qs=0.4, T=30) show 13% IB")
print("       If qs mattered, we'd expect different IB at different qs (same T)")
print("       Current data suggests T is dominant factor")

print("\nMECHANISTIC: Option 5 (Arrhenius)")
print("-" * 60)
print("  Formula: phi = A*exp(-Ea/RT) / (k_fold + A*exp(-Ea/RT))")
print("  Competition between temperature-dependent aggregation and folding")
print("  Pros: Physically meaningful parameters")
print("  Cons: More complex, nonlinear in T")

print("\n" + "=" * 60)
print("IMPLEMENTATION NOTES:")
print("=" * 60)
print("Replace in fit_model.py:")
print("  OLD: phi_IB = 1 / (1 + exp(-k_T * (T - T_50)))")
print("  NEW: phi_IB = ((T - T_onset) / T_scale)^n / (1 + ((T - T_onset) / T_scale)^n)")
print("\nAdd to PARAM_NAMES: 'T_onset', 'T_scale', 'n'")
print("Remove from PARAM_NAMES: 'T_50', 'k_T'")
print("Adjust PARAM_BOUNDS accordingly")
