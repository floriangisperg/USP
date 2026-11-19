"""
Profile likelihood analysis to assess parameter identifiability.

For each parameter:
1. Fix parameter at different values
2. Optimize all other parameters
3. Check if objective has a unique minimum

If the profile is flat, parameter is NOT identifiable.
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
from scipy.optimize import minimize


@dataclass
class ProfileResult:
    """Result of profile likelihood for one parameter."""
    param_name: str
    param_values: np.ndarray
    objective_values: np.ndarray
    identifiable: bool
    optimal_value: float
    confidence_interval: Optional[tuple] = None


class IdentifiabilityChecker:
    """
    Assess parameter identifiability using profile likelihood.
    """

    def __init__(
        self,
        model: OptFedModel,
        estimator: ParameterEstimator,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the identifiability checker.

        Args:
            model: OptFed model
            estimator: ParameterEstimator with loaded data
            output_dir: Directory to save plots
        """
        self.model = model
        self.estimator = estimator
        self.output_dir = output_dir or Path("results/identifiability")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get active parameters and bounds
        self.active_params = estimator._get_active_params()
        self.bounds = estimator._get_bounds()

    def fit_other_params(
        self,
        fixed_param_idx: int,
        fixed_param_value: float,
        maxiter: int = 100,
    ) -> float:
        """
        Fit all parameters except one (which is fixed).

        Args:
            fixed_param_idx: Index of parameter to fix
            fixed_param_value: Value to fix it at
            maxiter: Max iterations for local optimization

        Returns:
            Objective value at optimum
        """
        # Get literature values as starting point
        literature_values = {
            "Y_XG": 0.627, "Y_PG": 0.652, "gamma_max": 0.49, "K_m": 0.001,
            "K_G_inhib": 89.0, "K_pi_m": 0.62, "gamma_alpha_min": 2.4e-5,
            "K_alpha_gamma": 1e-4, "K_alpha_n": 1.5, "K_alpha_PX": 0.11,
        }

        # Initial guess (literature values for active params)
        x0 = np.array([
            literature_values.get(p, 0.5) for p in self.active_params
        ])
        x0[fixed_param_idx] = fixed_param_value

        # Bounds (fix the parameter by setting narrow bounds)
        bounds = self.bounds.copy()
        bounds[fixed_param_idx] = (fixed_param_value, fixed_param_value)

        # Local optimization
        result = minimize(
            fun=self.estimator._objective,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={'maxiter': maxiter},
        )

        return result.fun if result.success else np.inf

    def profile_parameter(
        self,
        param_idx: int,
        n_points: int = 15,
        span: float = 3.0,
    ) -> ProfileResult:
        """
        Profile likelihood for a single parameter.

        Args:
            param_idx: Index of parameter to profile
            n_points: Number of parameter values to test
            span: Range as ±log10(span) from optimal value

        Returns:
            ProfileResult with identifiability assessment
        """
        param_name = self.active_params[param_idx]
        lower, upper = self.bounds[param_idx]

        print(f"\n=== Profiling {param_name} ===")

        # First, find optimal value (fit all params)
        print("Finding optimal value...")
        x0_lit = np.array([
            0.627 if p == "Y_XG" else
            0.652 if p == "Y_PG" else
            0.49 if p == "gamma_max" else
            0.001 if p == "K_m" else
            89.0 if p == "K_G_inhib" else
            0.62 if p == "K_pi_m" else
            2.4e-5 if p == "gamma_alpha_min" else
            1e-4 if p == "K_alpha_gamma" else
            1.5 if p == "K_alpha_n" else
            0.11 if p == "K_alpha_PX" else 0.5
            for p in self.active_params
        ])

        result_opt = minimize(
            fun=self.estimator._objective,
            x0=x0_lit,
            bounds=self.bounds,
            method="L-BFGS-B",
            options={'maxiter': 200},
        )

        optimal_value = result_opt.x[param_idx]
        optimal_obj = result_opt.fun
        print(f"  Optimal {param_name} = {optimal_value:.6g}")
        print(f"  Objective at optimum = {optimal_obj:.6g}")

        # Create parameter value grid (log spacing for wide ranges)
        if optimal_value > 0:
            log_range = np.log10(max(optimal_value, lower)) - np.log10(lower)
            param_values = np.logspace(
                np.log10(max(lower, optimal_value / span)),
                np.log10(min(upper, optimal_value * span)),
                n_points,
            )
        else:
            param_values = np.linspace(lower, upper, n_points)

        # Evaluate profile
        objective_values = []
        for i, val in enumerate(param_values):
            print(f"  Testing {param_name} = {val:.6g} ({i+1}/{n_points})...", end=" ")
            obj = self.fit_other_params(param_idx, val, maxiter=50)
            objective_values.append(obj)
            print(f"obj = {obj:.6g}")

        objective_values = np.array(objective_values)

        # Assess identifiability
        # Parameter is identifiable if objective has a clear minimum
        obj_min = objective_values.min()
        obj_max = objective_values.max()
        obj_range = obj_max - obj_min

        # Check if profile is flat (less than 10% variation)
        flat_threshold = 0.1 * obj_min
        identifiable = obj_range > flat_threshold

        # Estimate confidence interval (where obj < obj_min + chi2_95)
        # chi2_95 for 1 dof ≈ 3.84
        chi2_threshold = optimal_obj + 3.84
        below_threshold = objective_values < chi2_threshold

        if below_threshold.sum() > 1:
            ci_lower = param_values[below_threshold][0]
            ci_upper = param_values[below_threshold][-1]
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = None

        return ProfileResult(
            param_name=param_name,
            param_values=param_values,
            objective_values=objective_values,
            identifiable=identifiable,
            optimal_value=optimal_value,
            confidence_interval=confidence_interval,
        )

    def plot_profile(self, profile: ProfileResult):
        """Plot profile likelihood for one parameter."""
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot profile
        ax.plot(profile.param_values, profile.objective_values, 'o-', linewidth=2)
        ax.axvline(profile.optimal_value, color='r', linestyle='--', label=f'Optimal = {profile.optimal_value:.6g}')

        # Add confidence interval if available
        if profile.confidence_interval:
            ax.axvspan(
                profile.confidence_interval[0],
                profile.confidence_interval[1],
                alpha=0.2,
                color='green',
                label='95% CI',
            )

        # Labels
        ax.set_xlabel(f'{profile.param_name} Value', fontsize=12)
        ax.set_ylabel('Objective Value (SSE)', fontsize=12)
        ax.set_title(f'Profile Likelihood: {profile.param_name}', fontsize=14)

        # Identifiability status
        status = "IDENTIFIABLE" if profile.identifiable else "NOT IDENTIFIABLE"
        ax.text(
            0.05, 0.95, f"Status: {status}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11,
        )

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Save
        save_path = self.output_dir / f"profile_{profile.param_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved plot to {save_path}")

    def full_analysis(self, subset: Optional[list[str]] = None) -> dict:
        """
        Run profile likelihood analysis for selected parameters.

        Args:
            subset: List of parameter names to profile (default: all)

        Returns:
            Dictionary of ProfileResults
        """
        # Determine which parameters to profile
        if subset is None:
            param_indices = list(range(len(self.active_params)))
        else:
            param_indices = [
                i for i, p in enumerate(self.active_params)
                if p in subset
            ]

        results = {}
        for idx in param_indices:
            profile = self.profile_parameter(idx, n_points=11)
            results[profile.param_name] = profile
            self.plot_profile(profile)

        # Summary
        self.print_summary(results)

        return results

    def print_summary(self, results: dict[ProfileResult]):
        """Print summary of identifiability analysis."""
        print("\n" + "=" * 60)
        print("IDENTIFIABILITY SUMMARY")
        print("=" * 60)

        identifiable = [p for p in results.values() if p.identifiable]
        not_identifiable = [p for p in results.values() if not p.identifiable]

        print(f"\nIdentifiable parameters ({len(identifiable)}):")
        for p in identifiable:
            ci_str = f"CI: [{p.confidence_interval[0]:.3g}, {p.confidence_interval[1]:.3g}]" if p.confidence_interval else "N/A"
            print(f"  {p.param_name}: {p.optimal_value:.6g} ({ci_str})")

        print(f"\nNOT identifiable parameters ({len(not_identifiable)}):")
        for p in not_identifiable:
            print(f"  {p.param_name}: flat profile")

        # Recommendation
        print(f"\n{'=' * 60}")
        print("RECOMMENDATION")
        print("=" * 60)
        print(f"Fix these parameters at literature values:")
        for p in not_identifiable:
            print(f"  - {p.param_name}")
        print(f"\nFit only these {len(identifiable)} parameters:")
        for p in identifiable:
            print(f"  - {p.param_name}")


if __name__ == "__main__":
    # Quick test on a subset of parameters
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

    print("Running identifiability check...")
    checker = IdentifiabilityChecker(model, estimator)

    # Test on key parameters first
    key_params = ['Y_XG', 'Y_PG', 'gamma_max', 'K_m', 'K_G_inhib']
    results = checker.full_analysis(subset=key_params)
