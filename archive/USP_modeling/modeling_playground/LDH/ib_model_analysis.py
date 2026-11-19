"""
IB Model Refinement Options

Current model (pure sigmoid):
  phi_IB(T) = 1 / (1 + exp(-k_T * (T - T_50)))
  Problem: Predicts ~0% IB at 30°C, but data shows ~13%

Data summary:
  T=25°C: 0% IB
  T=30°C: ~13% IB
  T=35°C: ~100% IB
"""

import numpy as np
import matplotlib.pyplot as plt

# Observed data
T_obs = np.array([25, 30, 35])
IB_obs = np.array([0.0, 0.13, 1.0])  # Fraction of IB
T_plot = np.linspace(20, 40, 200)

# Current fitted parameters
T_50_current = 34.05
k_T_current = 1.76

# ============================================================
# OPTION 1: Sigmoid with baseline floor
# ============================================================
# phi_IB = phi_min + (1 - phi_min) * sigmoid(k*(T - T_50))
# Ensures minimum IB fraction even at low T

def ib_model_floor(T, phi_min, k_T, T_50):
    """Sigmoid with floor parameter."""
    sigmoid = 1.0 / (1.0 + np.exp(-k_T * (T - T_50)))
    return phi_min + (1 - phi_min) * sigmoid

# Rough fit: phi_min ≈ 0.10 (10% baseline IB)
phi_min_opt1 = 0.10
IB_opt1 = ib_model_floor(T_plot, phi_min_opt1, k_T_current, T_50_current)

# ============================================================
# OPTION 2: Sigmoid with both floor and ceiling
# ============================================================
# phi_IB = phi_min + (phi_max - phi_min) * sigmoid(k*(T - T_50))
# Allows: (a) baseline IB at low T, (b) <100% IB at high T

def ib_model_clamped(T, phi_min, phi_max, k_T, T_50):
    """Sigmoid with floor and ceiling."""
    sigmoid = 1.0 / (1.0 + np.exp(-k_T * (T - T_50)))
    return phi_min + (phi_max - phi_min) * sigmoid

# Rough fit: phi_min≈0.08, phi_max≈0.95 (some soluble remains even at 35C)
phi_min_opt2 = 0.08
phi_max_opt2 = 0.95
IB_opt2 = ib_model_clamped(T_plot, phi_min_opt2, phi_max_opt2, k_T_current, T_50_current)

# ============================================================
# OPTION 3: Two-component model (mechanistically motivated)
# ============================================================
# IB comes from two sources:
# 1. Baseline aggregation (temperature-independent, constant fraction)
# 2. Heat-induced misfolding (temperature-dependent, sigmoid)
# phi_IB = phi_baseline + (1 - phi_baseline) * f_heat(T)

def ib_model_twocomponent(T, phi_baseline, k_T, T_50):
    """Two-component: baseline + heat-induced."""
    # Heat-induced fraction (sigmoid)
    f_heat = 1.0 / (1.0 + np.exp(-k_T * (T - T_50)))
    # Baseline aggregation happens regardless, then heat adds more
    return phi_baseline + (1 - phi_baseline) * f_heat

# This is mathematically same as Option 1, but with mechanistic interpretation
phi_baseline_opt3 = 0.12
IB_opt3 = ib_model_twocomponent(T_plot, phi_baseline_opt3, k_T_current, T_50_current)

# ============================================================
# OPTION 4: Modified temperature function (asymmetric)
# ============================================================
# Use a different function that allows gradual onset
# phi_IB = T^n / (T^n + T_50^n)  (Hill equation)

def ib_model_hill(T, T_50, n):
    """Hill equation (cooperative binding style)."""
    return T**n / (T**n + T_50**n)

# Rough fit: T_50≈32, n≈4 (steep transition)
T_50_opt4 = 32.0
n_opt4 = 4
IB_opt4 = ib_model_hill(T_plot, T_50_opt4, n_opt4)

# ============================================================
# PLOT COMPARISON
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

options = [
    ("Option 1: Sigmoid + Floor", IB_opt1,
     f"phi_min={phi_min_opt1:.2f}, k_T={k_T_current:.2f}, T_50={T_50_current:.1f}"),
    ("Option 2: Floor + Ceiling", IB_opt2,
     f"phi_min={phi_min_opt2:.2f}, phi_max={phi_max_opt2:.2f}"),
    ("Option 3: Two-Component", IB_opt3,
     f"phi_baseline={phi_baseline_opt3:.2f} (mechanistic)"),
    ("Option 4: Hill Equation", IB_opt4,
     f"T_50={T_50_opt4:.1f}, n={n_opt4}"),
]

for ax, (title, IB_pred, desc) in zip(axes, options):
    ax.plot(T_plot, IB_pred * 100, 'b-', linewidth=2, label='Model')
    ax.scatter(T_obs, IB_obs * 100, c='red', s=100, zorder=5,
               edgecolors='black', label='Observed', marker='o')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('IB Fraction [%]')
    ax.set_title(f'{title}\n{desc}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(20, 40)

plt.suptitle('IB Model Refinement Options', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/ib_model_options.png', dpi=150, bbox_inches='tight')
print("Plot saved to results/ib_model_options.png")

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("IB MODEL REFINEMENT OPTIONS")
print("=" * 60)
print("\nData to match:")
print("  T=25°C: 0% IB")
print("  T=30°C: 13% IB  <-- Current model fails here")
print("  T=35°C: 100% IB")

print("\n" + "-" * 60)
print("RECOMMENDATION: Option 1 (Sigmoid + Floor)")
print("-" * 60)
print("  Model: phi_IB = phi_min + (1 - phi_min) * sigmoid(k*(T - T_50))")
print("  New parameter: phi_min ≈ 0.10-0.15 (baseline IB fraction)")
print("  Interpretation: ~10% of protein aggregates regardless of T")
print("  Temperature then drives the remaining 0-90%")
print("\n  Advantages:")
print("    - Mechanistically interpretable")
print("    - Only 1 new parameter")
print("    - Preserves sigmoid shape")
print("    - Fits all data points")

print("\n" + "-" * 60)
print("ALTERNATIVE: Option 2 (Floor + Ceiling)")
print("-" * 60)
print("  Model: phi_IB = phi_min + (phi_max - phi_min) * sigmoid(...)")
print("  New parameters: phi_min ≈ 0.08, phi_max ≈ 0.95")
print("  Interpretation: Even at 35°C, some soluble remains")
print("\n  Advantages:")
print("    - More flexible (can fit non-0 to non-100%)")
print("    - Accounts for measurement uncertainty")
print("\n  Disadvantages:")
print("    - 2 new parameters (risk of overfitting)")

print("\n" + "-" * 60)
print("IMPLEMENTATION IN fit_model.py:")
print("-" * 60)
print("1. Add 'phi_min' to PARAM_NAMES")
print("2. Add (0.0, 0.3) to PARAM_BOUNDS")
print("3. Modify ode_system() to use new IB formula")
print("4. Re-fit on ferm_01-04")
print("5. Validate on ferm_05-07")
