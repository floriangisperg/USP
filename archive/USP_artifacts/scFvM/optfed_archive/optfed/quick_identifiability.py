"""
Quick identifiability check WITHOUT full re-optimization.

Uses a simpler approach:
1. Calculate objective at literature parameters
2. Vary each parameter individually (holding others at literature)
3. Check sensitivity - flat profile = not identifiable

This is MUCH faster than full profile likelihood.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import TRAIN_RUN_IDS, DataLoader
from fitting import FitConfig, ParameterEstimator
from model import InitialConditions, ModelParameters, OptFedModel


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for one parameter."""
    param_name: str
    param_values: np.ndarray
    objective_values: np.ndarray
    sensitivity_range: float  # Max - min objective
    relative_sensitivity: float  # Relative to parameter range
    identifiable: bool


class QuickIdentifiabilityCheck:
    """
    Fast identifiability check using one-at-a-time sensitivity.

    Much faster than profile likelihood because we don't re-optimize.
    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the quick check.

        Args:
            estimator: ParameterEstimator with loaded data
            output_dir: Directory to save plots
        """
        self.estimator = estimator
        self.output_dir = output_dir or Path("results/identifiability")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get active parameters and bounds
        self.active_params = estimator._get_active_params()
        self.bounds = estimator._get_bounds()

        # Literature values
        self.literature_values = {
            "Y_XG": 0.627, "Y_PG": 0.652, "gamma_max": 0.49, "K_m": 0.001,
            "K_G_inhib": 89.0, "K_pi_m": 0.62, "gamma_alpha_min": 2.4e-5,
            "K_alpha_gamma": 1e-4, "K_alpha_n": 1.5, "K_alpha_PX": 0.11,
        }

    def check_parameter_sensitivity(
        self,
        param_idx: int,
        n_points: int = 15,
    ) -> SensitivityResult:
        """
        Check sensitivity of objective to one parameter.

        Varies the parameter while holding others at literature values.

        Args:
            param_idx: Index of parameter to check
            n_points: Number of parameter values to test

        Returns:
            SensitivityResult
        """
        param_name = self.active_params[param_idx]
        lower, upper = self.bounds[param_idx]

        print(f"\n=== Checking sensitivity of {param_name} ===")
        print(f"  Bounds: [{lower:.6g}, {upper:.6g}]")

        # Create parameter value grid (log spacing for wide ranges)
        lit_value = self.literature_values.get(param_name, (lower + upper) / 2)

        if lit_value > 0:
            # Log spacing around literature value
            log_min = np.log10(max(lower, lit_value / 10))
            log_max = np.log10(min(upper, lit_value * 10))
            param_values = np.logspace(log_min, log_max, n_points)
        else:
            # Linear spacing
            param_values = np.linspace(lower, upper, n_points)

        # Get base parameter vector (literature values)
        x_base = np.array([
            self.literature_values.get(p, (self.bounds[i][0] + self.bounds[i][1]) / 2)
            for i, p in enumerate(self.active_params)
        ])

        # Evaluate objective at each parameter value
        objective_values = []
        for i, val in enumerate(param_values):
            x = x_base.copy()
            x[param_idx] = val

            print(f"  Testing {param_name} = {val:.6g} ({i+1}/{n_points})...", end=" ", flush=True)
            obj = self.estimator._objective(x)
            objective_values.append(obj)
            print(f"obj = {obj:.6g}")

        objective_values = np.array(objective_values)

        # Calculate sensitivity metrics
        obj_min = objective_values.min()
        obj_max = objective_values.max()
        sensitivity_range = obj_max - obj_min
        relative_sensitivity = sensitivity_range / (obj_min + 1e-10)

        # Parameter is identifiable if objective varies significantly (>10%)
        identifiable = relative_sensitivity > 0.1

        print(f"\n  Sensitivity range: {sensitivity_range:.6g}")
        print(f"  Relative sensitivity: {relative_sensitivity:.3%}")
        print(f"  Status: {'IDENTIFIABLE' if identifiable else 'NOT IDENTIFIABLE'}")

        return SensitivityResult(
            param_name=param_name,
            param_values=param_values,
            objective_values=objective_values,
            sensitivity_range=sensitivity_range,
            relative_sensitivity=relative_sensitivity,
            identifiable=identifiable,
        )

    def plot_sensitivity(self, result: SensitivityResult):
        """Plot sensitivity for one parameter."""
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot sensitivity curve
        ax.plot(result.param_values, result.objective_values, 'o-', linewidth=2, markersize=6)

        # Mark literature value
        lit_value = self.literature_values.get(result.param_name)
        if lit_value is not None:
            lit_idx = np.argmin(np.abs(result.param_values - lit_value))
            lit_obj = result.objective_values[lit_idx]
            ax.axvline(lit_value, color='r', linestyle='--', alpha=0.5, label=f'Literature = {lit_value:.6g}')
            ax.plot(lit_value, lit_obj, 'r*', markersize=15, label=f'obj = {lit_obj:.6g}')

        # Labels
        ax.set_xlabel(f'{result.param_name} Value', fontsize=12)
        ax.set_ylabel('Objective Value (SSE)', fontsize=12)
        ax.set_title(f'Sensitivity Analysis: {result.param_name}', fontsize=14)

        # Identifiability status
        status = "IDENTIFIABLE" if result.identifiable else "NOT IDENTIFIABLE"
        ax.text(
            0.05, 0.95,
            f"Status: {status}\nSensitivity: {result.sensitivity_range:.6g}\nRelative: {result.relative_sensitivity:.1%}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11,
        )

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Save
        save_path = self.output_dir / f"sensitivity_{result.param_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved plot to {save_path}")

    def full_analysis(self) -> dict[str, SensitivityResult]:
        """
        Run quick sensitivity analysis for all parameters.

        Returns:
            Dictionary of SensitivityResults
        """
        print("=" * 60)
        print("QUICK IDENTIFIABILITY CHECK")
        print("=" * 60)
        print(f"Checking {len(self.active_params)} parameters...")
        print("Note: This varies each parameter individually (no re-optimization)")

        results = {}

        for idx in range(len(self.active_params)):
            result = self.check_parameter_sensitivity(idx, n_points=15)
            results[result.param_name] = result
            self.plot_sensitivity(result)

        # Summary
        self.print_summary(results)

        # Export results
        self.export_results(results)

        return results

    def print_summary(self, results: dict[str, SensitivityResult]):
        """Print summary of identifiability analysis."""
        print("\n" + "=" * 60)
        print("IDENTIFIABILITY SUMMARY")
        print("=" * 60)

        # Sort by sensitivity
        sorted_results = sorted(
            results.values(),
            key=lambda r: r.sensitivity_range,
            reverse=True,
        )

        print(f"\nParameter sensitivity ranking:")
        print("(Higher sensitivity = more identifiable)")
        print()

        for i, r in enumerate(sorted_results, 1):
            marker = "✓" if r.identifiable else "✗"
            print(f"{i:2d}. {marker} {r.param_name:15s}: sensitivity = {r.sensitivity_range:8.6g} ({r.relative_sensitivity:6.1%})")

        identifiable = [p for p in results.values() if p.identifiable]
        not_identifiable = [p for p in results.values() if not p.identifiable]

        print(f"\n{'=' * 60}")
        print("RECOMMENDATION")
        print("=" * 60)
        print(f"\nIdentifiable parameters ({len(identifiable)}):")
        for p in identifiable:
            print(f"  - {p.param_name}")

        print(f"\nNOT identifiable parameters ({len(not_identifiable)}):")
        print("  Fix these at literature values:")
        for p in not_identifiable:
            lit_val = self.literature_values.get(p.param_name, "N/A")
            print(f"    {p.param_name} = {lit_val}")

        print(f"\nSuggested FitConfig:")
        print(f"  active_params = {[p.param_name for p in identifiable]}")

    def export_results(self, results: dict[str, SensitivityResult]):
        """Export results to CSV."""
        rows = []
        for r in results.values():
            rows.append({
                'parameter': r.param_name,
                'sensitivity_range': r.sensitivity_range,
                'relative_sensitivity': r.relative_sensitivity,
                'identifiable': r.identifiable,
                'optimal_value': self.literature_values.get(r.param_name, 'N/A'),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('sensitivity_range', ascending=False)

        save_path = self.output_dir / "sensitivity_results.csv"
        df.to_csv(save_path, index=False)
        print(f"\nExported results to {save_path}")


if __name__ == "__main__":
    from model import ModelParameters

    data_path = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\data\traintest_compiled.xlsx"

    print("Loading data...")
    loader = DataLoader(data_path, TRAIN_RUN_IDS)
    data = loader.load_data()

    print("Creating model and estimator...")
    params = ModelParameters(Gf=400.0)
    init_cond = InitialConditions(X0=10.0, P0=0.0, G0=0.1, V0=1.1)
    model = OptFedModel(params, init_cond)

    config = FitConfig()
    estimator = ParameterEstimator(model, data, config)

    print("Running quick identifiability check...")
    checker = QuickIdentifiabilityCheck(estimator)

    results = checker.full_analysis()

    print("\nDone!")
