"""
Question 2: Does IB Behavior Change Over Time?

Current model assumes: phi_IB = f(Temperature only)
Reality: Could depend on many time-varying factors

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# HYPOTHESES: WHY IB MIGHT CHANGE OVER TIME
# ============================================================

hypotheses = [
    {
        "name": "Cellular stress accumulation",
        "mechanism": "As cells age, stress increases → more misfolding",
        "predictor": "Time since induction",
        "test": "Plot IB fraction vs time at each T"
    },
    {
        "name": "Protease saturation",
        "mechanism": "Misfolded protein accumulates → proteases overwhelmed → more IB",
        "predictor": "Total protein produced",
        "test": "Plot IB fraction vs cumulative P"
    },
    {
        "name": "Chaperone depletion",
        "mechanism": "Chaperones (DnaK, GroEL) get depleted → less folding capacity",
        "predictor": "Expression level relative to chaperone pool",
        "test": "Need chaperone measurements (not available?)"
    },
    {
        "name": "Growth rate dependence",
        "mechanism": "High growth → less time for folding → more IB",
        "predictor": "Specific growth rate (mu)",
        "test": "Plot IB fraction vs mu at different time points"
    },
    {
        "name": "Metabolic state",
        "mechanism": "Energy status affects folding capacity",
        "predictor": "qs, qO2, ATP levels",
        "test": "Correlate IB with metabolic fluxes"
    },
    {
        "name": "Phase-dependent behavior",
        "mechanism": "Early vs late induction have different IB dynamics",
        "predictor": "Phase of induction (early, mid, late)",
        "test": "Split data by time since induction"
    },
]

print("=" * 60)
print("HYPOTHESES: TIME-VARYING IB BEHAVIOR")
print("=" * 60)

for i, h in enumerate(hypotheses, 1):
    print(f"\n{i}. {h['name']}")
    print(f"   Mechanism: {h['mechanism']}")
    print(f"   Predictor: {h['predictor']}")
    print(f"   Test:      {h['test']}")

# ============================================================
# DIAGNOSTIC: RESIDUAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DIAGNOSTIC: RESIDUAL ANALYSIS")
print("=" * 60)

print("\nIf IB model is WRONG (missing time dependence):")
print("  Residuals will show PATTERNS, not random scatter")
print("\nExpected patterns if time-dependence exists:")
print("  - Early time points: consistent over/under-prediction")
print("  - Late time points: opposite bias")
print("  - Trends: residuals increase/decrease with time")

print("\nHow to check:")
print("  1. For each sample, compute residual = P_IB_pred - P_IB_obs")
print("  2. Plot residuals vs time")
print("  3. If pattern exists: model missing time-dependent term")

# ============================================================
# TEST: DOES IB FRACTION CHANGE WITHIN A RUN?
# ============================================================

def test_ib_evolution():
    """
    Test if IB fraction changes over time within a single run.
    If it does, phi_IB is not just a function of T!
    """

    print("\n" + "=" * 60)
    print("TEST: IB EVOLUTION WITHIN SINGLE RUNS")
    print("=" * 60)

    # Simulated example: what we'd see if IB changes over time
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    scenarios = [
        ("Static IB (current model)", 25, False),
        ("Static IB (current model)", 30, False),
        ("Static IB (current model)", 35, False),
        ("Time-varying IB", 25, True),
        ("Time-varying IB", 30, True),
        ("Time-varying IB", 35, True),
    ]

    time = np.linspace(5, 30, 100)

    for ax, (label, T, time_varying) in zip(axes, scenarios):
        if time_varying:
            # IB increases over time (e.g., stress accumulation)
            base_IB = 0.0 if T == 25 else (0.1 if T == 30 else 0.9)
            IB_over_time = base_IB + 0.3 * (1 - np.exp(-(time - 5)/10))
            P_IB = IB_over_time * 5.0  # g/L
            P_sol = (1 - IB_over_time) * 5.0
            ax.plot(time, P_IB, 'r-', linewidth=2, label='IB')
            ax.plot(time, P_sol, 'g-', linewidth=2, label='Soluble')
            ax.set_title(f"{label}\nT={T}°C (IB fraction changes)")
        else:
            # Constant IB fraction
            IB_frac = 0.0 if T == 25 else (0.13 if T == 30 else 1.0)
            P_IB = np.full_like(time, IB_frac * 5.0)
            P_sol = np.full_like(time, (1 - IB_frac) * 5.0)
            ax.plot(time, P_IB, 'r-', linewidth=2, label='IB')
            ax.plot(time, P_sol, 'g-', linewidth=2, label='Soluble')
            ax.set_title(f"{label}\nT={T}°C (constant IB fraction)")

        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Protein [g/L]')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Static vs Time-Varying IB Partitioning', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/ib_time_evolution_test.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to results/ib_time_evolution_test.png")

test_ib_evolution()

# ============================================================
# EXTENDED MODELS WITH TIME-VARYING IB
# ============================================================

print("\n" + "=" * 60)
print("EXTENDED MODELS: IF IB CHANGES OVER TIME")
print("=" * 60)

models = [
    {
        "name": "Current model",
        "formula": "phi_IB = f(T)",
        "params": "T_onset, T_scale, n",
        "pros": "Simple, works if IB is T-dependent only",
        "cons": "Cannot capture time variation"
    },
    {
        "name": "Time-scaled model",
        "formula": "phi_IB = f(T) * (1 + c * t_since_induction)",
        "params": "T_onset, T_scale, n, c",
        "pros": "Simple time dependence",
        "cons": "May extrapolate poorly"
    },
    {
        "name": "Stress-accumulation model",
        "formula": "phi_IB = f(T) / (1 + exp(-k_stress * (P_total - P_thresh)))",
        "params": "T_onset, T_scale, n, k_stress, P_thresh",
        "pros": "Mechanistic (saturation)",
        "cons": "2 extra parameters"
    },
    {
        "name": "Growth-rate dependent",
        "formula": "phi_IB = f(T) * (1 + alpha_mu * (mu - mu_ref))",
        "params": "T_onset, T_scale, n, alpha_mu",
        "pros": "Links to cellular state",
        "cons": "mu varies, may be redundant"
    },
]

for i, m in enumerate(models, 1):
    print(f"\n{i}. {m['name']}")
    print(f"   Formula:  {m['formula']}")
    print(f"   Params:   {m['params']}")
    print(f"   Pros:     {m['pros']}")
    print(f"   Cons:     {m['cons']}")

# ============================================================
# PRACTICAL APPROACH
# ============================================================

print("\n" + "=" * 60)
print("PRACTICAL APPROACH: HOW TO TEST")
print("=" * 60)

print("\nStep 1: Fit current model (phi_IB = f(T))")
print("Step 2: Analyze residuals vs time")
print("  - If no pattern: IB is T-dependent only ✓")
print("  - If pattern exists: Continue")

print("\nStep 3: For runs with IB data at multiple time points:")
print("  - Compute observed IB fraction at each time point")
print("  - Plot IB fraction vs time")
print("  - Test: Is IB fraction constant or changing?")

print("\nStep 4: If changing, fit extended model")
print("  - Use likelihood ratio test")
print("  - Compare AIC/BIC")
print("  - Cross-validate!")

print("\nStep 5: Report findings")
print("  - 'IB fraction constant over time (p > 0.05)'")
print("  - OR 'IB fraction increases with stress (p < 0.01)'")

# ============================================================
# SUMMARY CHECKLIST
# ============================================================

print("\n" + "=" * 60)
print("CHECKLIST: ANSWERING YOUR QUESTIONS")
print("=" * 60)

print("\nQ1: Am I overfitting?")
print("-" * 60)
print("  [ ] Use train/validation split (don't fit all data)")
print("  [ ] Compare training vs validation loss")
print("  [ ] Profile likelihood for parameter identifiability")
print("  [ ] Report confidence intervals")

print("\nQ2: Is IB parameter fixed or time-varying?")
print("-" * 60)
print("  [ ] Residual analysis: plot residuals vs time")
print("  [ ] For each run, plot IB fraction vs time")
print("  [ ] Statistical test: is IB fraction constant?")
print("  [ ] If varying: fit extended model with time term")
print("  [ ] Validate extended model on held-out data")

print("\nQ3: How do I know if feeding strategy matters?")
print("-" * 60)
print("  [ ] Compare ferm_01 (qs=0.2) vs ferm_04 (qs=0.6) at same T")
print("  [ ] Current data: both show 100% IB at 35C")
print("  [ ] Suggests: T dominates, qs has minor effect")
print("  [ ] Test: add qs term to IB model, fit, compare AIC")

print("\n" + "=" * 60)
