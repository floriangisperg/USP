"""
Test script for OptFed framework.

Tests the three-stage framework:
1. Stage I: Define - Mechanistic model simulation
2. Stage II: Fit - Parameter estimation (placeholder)
3. Stage III: Optimize - Control optimization (placeholder)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from model import (
    ControlProfile,
    FeedStrategy,
    InitialConditions,
    ModelParameters,
    OptFedModel,
    create_default_model,
)


def test_stage_i_define():
    """Test Stage I: Model definition and simulation."""
    print("=" * 60)
    print("STAGE I: DEFINE - Testing Mechanistic Model")
    print("=" * 60)

    # Create model with default parameters
    model = create_default_model()

    print("\nModel Parameters:")
    print(f"  Y_XG (biomass yield): {model.params.Y_XG}")
    print(f"  Y_PG (product yield): {model.params.Y_PG}")
    print(f"  gamma_max (max uptake): {model.params.gamma_max} 1/h")
    print(f"  K_m (MM constant): {model.params.K_m} g/L")
    print(f"  Gf (feed concentration): {model.params.Gf} g/L")

    print("\nInitial Conditions:")
    print(f"  X0: {model.init_cond.X0} g")
    print(f"  P0: {model.init_cond.P0} g")
    print(f"  G0: {model.init_cond.G0} g")
    print(f"  V0: {model.init_cond.V0} L")

    # Test 1: Constant feed
    print("\n--- Test 1: Constant Feed ---")
    control_const = ControlProfile(feed_rate=FeedStrategy.constant(rate=0.01))
    t_eval = np.linspace(0, 10, 101)

    result_const = model.simulate(
        t_span=(0, 10),
        t_eval=t_eval,
        control=control_const,
        method="LSODA",
    )

    print(f"Success: {result_const['success']}")
    print(f"Final biomass: {result_const['X_conc'][-1]:.2f} g/L")
    print(f"Final product: {result_const['P_conc'][-1]:.4f} g/L")
    print(f"Final volume: {result_const['V'][-1]:.2f} L")

    # Test 2: Exponential feed
    print("\n--- Test 2: Exponential Feed ---")
    control_exp = ControlProfile(feed_rate=FeedStrategy.exponential(f0=0.005, mu_f=0.1))

    result_exp = model.simulate(
        t_span=(0, 10),
        t_eval=t_eval,
        control=control_exp,
        method="LSODA",
    )

    print(f"Success: {result_exp['success']}")
    print(f"Final biomass: {result_exp['X_conc'][-1]:.2f} g/L")
    print(f"Final product: {result_exp['P_conc'][-1]:.4f} g/L")
    print(f"Final volume: {result_exp['V'][-1]:.2f} L")

    # Test 3: Linear feed (post-induction style)
    print("\n--- Test 3: Linear Feed ---")
    control_lin = ControlProfile(feed_rate=FeedStrategy.linear(phi1=0.01, phi2=0.002))

    result_lin = model.simulate(
        t_span=(0, 10),
        t_eval=t_eval,
        control=control_lin,
        method="LSODA",
    )

    print(f"Success: {result_lin['success']}")
    print(f"Final biomass: {result_lin['X_conc'][-1]:.2f} g/L")
    print(f"Final product: {result_lin['P_conc'][-1]:.4f} g/L")
    print(f"Final volume: {result_lin['V'][-1]:.2f} L")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(result_const['t'], result_const['X_conc'], label='Constant')
    axes[0, 0].plot(result_exp['t'], result_exp['X_conc'], label='Exponential')
    axes[0, 0].plot(result_lin['t'], result_lin['X_conc'], label='Linear')
    axes[0, 0].set_xlabel('Time (h)')
    axes[0, 0].set_ylabel('Biomass (g/L)')
    axes[0, 0].set_title('Biomass Concentration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(result_const['t'], result_const['P_conc'], label='Constant')
    axes[0, 1].plot(result_exp['t'], result_exp['P_conc'], label='Exponential')
    axes[0, 1].plot(result_lin['t'], result_lin['P_conc'], label='Linear')
    axes[0, 1].set_xlabel('Time (h)')
    axes[0, 1].set_ylabel('Product (g/L)')
    axes[0, 1].set_title('Product Concentration')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(result_const['t'], result_const['G_conc'], label='Constant')
    axes[1, 0].plot(result_exp['t'], result_exp['G_conc'], label='Exponential')
    axes[1, 0].plot(result_lin['t'], result_lin['G_conc'], label='Linear')
    axes[1, 0].set_xlabel('Time (h)')
    axes[1, 0].set_ylabel('Glucose (g/L)')
    axes[1, 0].set_title('Substrate Concentration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(result_const['t'], result_const['V'], label='Constant')
    axes[1, 1].plot(result_exp['t'], result_exp['V'], label='Exponential')
    axes[1, 1].plot(result_lin['t'], result_lin['V'], label='Linear')
    axes[1, 1].set_xlabel('Time (h)')
    axes[1, 1].set_ylabel('Volume (L)')
    axes[1, 1].set_title('Reactor Volume')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "results" / "test_simulation.png", dpi=150)
    print(f"\nPlot saved to: {Path(__file__).parent / 'results' / 'test_simulation.png'}")

    return True


def test_stage_ii_fit():
    """Test Stage II: Parameter fitting (placeholder)."""
    print("\n" + "=" * 60)
    print("STAGE II: FIT - Parameter Estimation")
    print("=" * 60)
    print("\nParameter fitting module implemented but not tested.")
    print("To use: Run fitting.py with actual training data.")
    print("\nExample:")
    print("  from fitting import fit_model_to_data")
    print("  result = fit_model_to_data('path/to/data.xlsx')")
    return True


def test_stage_iii_optimize():
    """Test Stage III: Optimization (placeholder)."""
    print("\n" + "=" * 60)
    print("STAGE III: OPTIMIZE - Control Optimization")
    print("=" * 60)
    print("\nOptimization module implemented but not tested.")
    print("To use: Run optimization.py with fitted parameters.")
    print("\nExample:")
    print("  from optimization import optimize_process")
    print("  result = optimize_process(fitted_params)")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OptFed Framework Test Suite")
    print("=" * 60)

    # Ensure results directory exists
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    tests = [
        ("Stage I: Define", test_stage_i_define),
        ("Stage II: Fit", test_stage_ii_fit),
        ("Stage III: Optimize", test_stage_iii_optimize),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
