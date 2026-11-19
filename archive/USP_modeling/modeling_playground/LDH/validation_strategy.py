"""
Validation Strategy: Avoid Overfitting

Your data structure:
- ferm_01-04: Corner points (training data)
- ferm_05-07: Triplicates at intermediate condition (validation data)

"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# VALIDATION APPROACHES
# ============================================================

print("=" * 60)
print("VALIDATION STRATEGIES TO AVOID OVERFITTING")
print("=" * 60)

print("\n1. TRAIN/TEST SPLIT (What you did)")
print("-" * 60)
print("   Training:   ferm_01-04 (corner points)")
print("   Validation: ferm_05-07 (intermediate condition)")
print("   Test:       (none - need more data)")
print("\n   ✓ Best approach with limited data")
print("   ✓ Tests interpolation (30C is between 25C and 35C)")
print("   ✗ Cannot test extrapolation")

print("\n2. LEAVE-ONE-OUT CROSS-VALIDATION")
print("-" * 60)
print("   For each run i=1..7:")
print("     - Train on 6 runs")
print("     - Validate on run i")
print("     - Record error")
print("   Report: mean ± std of errors")
print("\n   ✓ Uses all data for validation eventually")
print("   ✓ Shows which runs are outliers")
print("   ✗ Computationally expensive")

print("\n3. K-FOLD CROSS-VALIDATION")
print("-" * 60)
print("   Split data into K folds (e.g., K=3)")
print("   Fold 1: ferm_01, ferm_02 (qs=0.2)")
print("   Fold 2: ferm_03, ferm_04 (qs=0.6)")
print("   Fold 3: ferm_05, ferm_06, ferm_07 (qs=0.4)")
print("\n   ✓ Validates across different conditions")
print("   ✗ Small sample size per fold")

print("\n4. BOOTSTRAP VALIDATION")
print("-" * 60)
print("   Resample data with replacement B times")
print("   Fit B models, plot distribution of parameters")
print("   Check: Are parameters well-constrained?")
print("\n   ✓ Shows parameter uncertainty")
print("   ✓ Identifies overfitting (wide parameter distributions)")

print("\n5. PARAMETER IDENTIFIABILITY (Profile Likelihood)")
print("-" * 60)
print("   For each parameter theta:")
print("     - Fix theta at different values")
print("     - Fit all other parameters")
print("     - Plot loss vs theta")
print("\n   Well-identified: Clear minimum (parabolic)")
print("   Poorly-identified: Flat (can vary widely)")
print("\n   ✓ Shows which parameters are actually informed by data")
print("   ✓ Reveals overparameterization")

print("\n" + "=" * 60)
print("RECOMMENDED STRATEGY FOR YOUR DATA")
print("=" * 60)

print("\nPRIMARY: Train/Validation split (keep ferm_05-07 for validation)")
print("\nSECONDARY: Profile likelihood on trained parameters")
print("   - If parameters have flat profiles: model is overparameterized")
print("   - Consider reducing parameters or collecting more data")

print("\nDIAGNOSTIC: Compare training vs validation loss")
print("   Training loss << Validation loss: OVERFITTING")
print("   Similar losses: Good generalization")
print("   Both high: UNDERFITTING (model too simple)")

# ============================================================
# VISUAL: PARAMETER IDENTIFIABILITY CHECK
# ============================================================

def illustrate_identifiability():
    """Show what well vs poorly identified parameters look like."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    theta = np.linspace(0, 1, 100)

    # Well-identified
    loss_well = 10 * (theta - 0.5)**2 + 1
    axes[0].plot(theta, loss_well, 'b-', linewidth=2)
    axes[0].axvline(0.5, color='r', linestyle='--', label='Optimal')
    axes[0].set_xlabel('Parameter value')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Well-identified parameter\n(clear minimum)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Poorly-identified (flat)
    loss_poor = np.ones_like(theta) + 0.1 * np.random.randn(len(theta))
    loss_poor = np.maximum(loss_poor, 0.5)
    axes[1].plot(theta, loss_poor, 'b-', linewidth=2)
    axes[1].set_xlabel('Parameter value')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Poorly-identified parameter\n(any value works)')
    axes[1].grid(True, alpha=0.3)

    # Correlated parameters (valley)
    theta1, theta2 = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    loss_corr = 5 * ((theta1 + theta2)/2 - 0.5)**2 + 1
    axes[2].contour(theta1, theta2, loss_corr, levels=10)
    axes[2].plot([0, 1], [1, 0], 'r--', label='Correlation')
    axes[2].set_xlabel('Parameter 1')
    axes[2].set_ylabel('Parameter 2')
    axes[2].set_title('Correlated parameters\n(valley, not unique minimum)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Parameter Identifiability Diagnostics', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/parameter_identifiability.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to results/parameter_identifiability.png")

illustrate_identifiability()

print("\n" + "=" * 60)
print("BOTTOM LINE")
print("=" * 60)
print("With 7 runs total:")
print("  - Use 4 for training (ferm_01-04)")
print("  - Use 3 for validation (ferm_05-07)")
print("  - If validation loss ≈ training loss: NO overfitting")
print("  - Report confidence intervals on parameters")
print("\nDO NOT fit all 7 runs and claim 'good fit' - that's circular!")
