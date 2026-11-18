"""
Compare v1 (Additive) vs v2 (Multiplicative) IB Models
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)


def compare_structures():
    """
    Visualize the difference between additive and multiplicative structures.

    Additive (v1):     phi_IB = phi_syn + phi_agg
    Multiplicative (v2): phi_IB = 1 - (1 - phi_syn) * (1 - phi_agg)
    """

    # Example component values
    phi_syn_range = np.linspace(0, 0.5, 100)
    phi_agg_fixed = 0.3  # Example fixed aggregation value

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Additive structure
    ax = axes[0]
    phi_IB_additive = phi_syn_range + phi_agg_fixed
    phi_IB_additive = np.clip(phi_IB_additive, 0, 1)

    ax.plot(phi_syn_range * 100, phi_IB_additive * 100, 'b-', linewidth=3,
            label='Additive: phi_IB = phi_syn + phi_agg')
    ax.axhline(phi_agg_fixed * 100, color='r', linestyle='--',
               label=f'phi_agg = {phi_agg_fixed*100:.0f}% (baseline)')

    # Add example points
    ax.scatter([5, 10, 20], [(5/100 + phi_agg_fixed)*100, (10/100 + phi_agg_fixed)*100,
                              min((20/100 + phi_agg_fixed)*100, 100)],
               c='black', s=100, zorder=5, marker='o')

    ax.set_xlabel('Synthesis Overload (phi_syn) [%]', fontsize=12)
    ax.set_ylabel('IB Fraction (phi_IB) [%]', fontsize=12)
    ax.set_title('Additive Structure (v1)\nProblem: IB always > phi_agg', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)

    # Plot 2: Multiplicative structure
    ax = axes[1]
    phi_IB_multiplicative = 1 - (1 - phi_syn_range) * (1 - phi_agg_fixed)

    ax.plot(phi_syn_range * 100, phi_IB_multiplicative * 100, 'g-', linewidth=3,
            label='Multiplicative: phi_IB = 1 - (1-phi_syn)*(1-phi_agg)')

    # Add same example points
    ax.scatter([5, 10, 20], [1 - (1 - 0.05) * (1 - phi_agg_fixed),
                              1 - (1 - 0.10) * (1 - phi_agg_fixed),
                              1 - (1 - 0.20) * (1 - phi_agg_fixed)],
               c='black', s=100, zorder=5, marker='o')

    # Show regions
    ax.axhspan(0, phi_agg_fixed * 100, alpha=0.1, color='red',
               label=f'Region: phi_syn < {phi_agg_fixed*100:.0f}%')

    ax.set_xlabel('Synthesis Overload (phi_syn) [%]', fontsize=12)
    ax.set_ylabel('IB Fraction (phi_IB) [%]', fontsize=12)
    ax.set_title('Multiplicative Structure (v2)\nSolution: Both must contribute', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)

    plt.suptitle('Model Structure Comparison: Additive vs Multiplicative',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(RESULTS_PATH / 'model_structure_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved to {RESULTS_PATH / 'model_structure_comparison.png'}")
    plt.show()


def explain_difference():
    """Print explanation of the difference."""

    print("\n" + "=" * 70)
    print("ADDITIVE vs MULTIPLICATIVE IB MODELS")
    print("=" * 70)

    print("\nPROBLEM WITH ADDITIVE (v1):")
    print("-" * 70)
    print("  phi_IB = phi_synthesis + phi_aggregation")
    print("")
    print("  Example: If phi_agg = 30% (from thermal effects)")
    print("    - Even with phi_syn = 0%,  phi_IB = 30%  (wrong at low T!)")
    print("    - With    phi_syn = 5%,   phi_IB = 35%  (still too high)")
    print("    - With    phi_syn = 20%,  phi_IB = 50%  (excessive)")
    print("")
    print("  Result: Model always predicts IB > thermal aggregation baseline")
    print("  Issue: Cannot achieve 0% IB at low temperatures")

    print("\n\nSOLUTION WITH MULTIPLICATIVE (v2):")
    print("-" * 70)
    print("  phi_IB = 1 - (1 - phi_synthesis) * (1 - phi_aggregation)")
    print("")
    print("  Mechanistic interpretation:")
    print("    - (1 - phi_syn)     = probability of surviving synthesis overload")
    print("    - (1 - phi_agg)     = probability of avoiding thermal aggregation")
    print("    - Product           = probability protein remains soluble")
    print("    - 1 - product       = probability protein forms IB")
    print("")
    print("  Example: If phi_agg = 30% (from thermal effects)")
    print("    - With phi_syn = 0%,   phi_IB = 1 - (1.0) * (0.7) = 30%")
    print("    - With phi_syn = 5%,   phi_IB = 1 - (0.95) * (0.7) = 33.5%")
    print("    - With phi_syn = 20%,  phi_IB = 1 - (0.80) * (0.7) = 44%")
    print("")
    print("  Key insight: If phi_agg = 0%, then phi_IB = phi_syn")
    print("               If phi_syn = 0%, then phi_IB = phi_agg")
    print("               If both = 0%,      then phi_IB = 0%  [OK]")

    print("\n\nADVANTAGE OF MULTIPLICATIVE:")
    print("-" * 70)
    print("  Can achieve 0% IB when both mechanisms are inactive")
    print("  Mechanistically interpretable as failure probabilities")
    print("  Better matches biophysics of protein folding/aggregation")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    explain_difference()
    print("\nGenerating comparison plot...")
    compare_structures()
    print("\nDone!")
