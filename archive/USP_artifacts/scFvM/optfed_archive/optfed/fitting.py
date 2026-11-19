"""
Parameter fitting for OptFed mechanistic model (Stage II).

Implements model parameter estimation using differential evolution or CMA-ES,
with model selection based on significance testing.

Based on: Schlögel et al. (2024)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

# Try to import CMA-ES
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    cma = None
try:
    from sklearn.model_selection import KFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Create a dummy KFold for type hints
    class KFold:
        def __init__(self, n_splits, **kwargs):
            self.n_splits = n_splits
import pickle

from data_loader import TRAIN_RUN_IDS, DataLoader, FermentationRun, ProcessData
from model import (
    ControlProfile,
    FeedStrategy,
    InitialConditions,
    ModelParameters,
    OptFedModel,
)


class OptimizerType(Enum):
    """Types of optimizers for parameter fitting."""
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    CMA_ES = "cma_es"
    HYBRID = "hybrid"  # CMA-ES followed by local refinement


class ModelComplexity(Enum):
    """Model complexity levels based on paper's model selection."""
    FULL = "full"           # All 15 parameters (full maintenance model)
    REDUCED = "reduced"     # 10 parameters (paper's selected model)
    MINIMAL = "minimal"     # Even simpler model for debugging


@dataclass
class FitConfig:
    """Configuration for parameter fitting."""
    # Bounds for parameters (lower, upper)
    # Wider bounds based on paper Table 1 values
    # Yield coefficients
    Y_XG_bounds: tuple = (0.1, 1.0)       # Paper: 0.627
    Y_PG_bounds: tuple = (0.05, 1.0)       # Paper: 0.652 (was 0.5 max!)

    # Substrate uptake parameters
    gamma_max_bounds: tuple = (0.1, 2.0)   # Paper: 0.49
    K_m_bounds: tuple = (1e-6, 1.0)        # Paper: 0.001
    K_G_inhib_bounds: tuple = (1.0, 1000.0) # Paper: 89.0
    K_P_inhib_bounds: tuple = (1.0, 1e7)   # Paper: 1e6 (disabled)
    K_X_inhib_bounds: tuple = (1.0, 1e7)   # Paper: 1e6 (disabled)

    # Production uptake parameters
    gamma_pi_max_bounds: tuple = (1e-6, 1e7) # Paper: 1e6 (effectively disabled)
    K_pi_m_bounds: tuple = (0.001, 1.0)    # Paper: 0.62 (was 0.5 max!)

    # Maintenance parameters (full model with all activation terms)
    gamma_alpha_min_bounds: tuple = (1e-6, 0.1)  # Paper: 2.4e-5 (was 0.001 min!)
    K_alpha_gamma_bounds: tuple = (1e-6, 5.0)    # Paper: 1e-4
    K_alpha_G_bounds: tuple = (1e-6, 1e7)        # Paper: 1e6 (disabled)
    K_alpha_n_bounds: tuple = (0.1, 20.0)        # Paper: 1.5
    K_alpha_PX_bounds: tuple = (0.01, 10.0)      # Paper: 0.11
    K_alpha_X_bounds: tuple = (1.0, 1e7)         # Paper: 1e6 (disabled)

    # Optimization settings
    maxiter: int = 1000
    popsize: int = 15
    tol: float = 1e-6
    workers: int = -1  # -1 uses all CPUs

    # Cross-validation
    n_folds: int = 3
    test_size: float = 0.2

    # Optimizer selection
    optimizer: OptimizerType = OptimizerType.CMA_ES

    # Model complexity (which parameters to fit)
    model_complexity: ModelComplexity = ModelComplexity.REDUCED

    # CMA-ES specific settings
    cma_sigma0: Optional[float] = None  # Initial step size (None: auto)
    cma_popsize: Optional[int] = None  # Population size (None: auto, typically 4 + 3*ln(N))
    cma_maxfevals: int = 10000  # Maximum function evaluations
    cma_tolfun: float = 1e-11  # Tolerance on function value
    cma_tolx: float = 1e-11  # Tolerance on x changes
    cma_verbose: int = 1  # Verbosity level (0-3)

    # Local refinement (for hybrid mode)
    local_refinement: bool = True  # Use local optimization after global search
    local_method: str = "L-BFGS-B"  # Local optimization method
    local_maxiter: int = 1000  # Max iterations for local refinement


@dataclass
class FitResult:
    """Results from parameter fitting."""
    params: ModelParameters
    success: bool
    message: str
    objective_value: float
    n_iterations: int
    cv_scores: Optional[dict] = None
    parameter_history: Optional[list] = None


class ParameterEstimator:
    """
    Estimate model parameters from experimental data.

    Uses differential evolution to minimize the sum of squared errors
    between model predictions and experimental data.
    """

    def __init__(
        self,
        model: OptFedModel,
        data: ProcessData,
        config: Optional[FitConfig] = None,
    ):
        """
        Initialize the parameter estimator.

        Args:
            model: OptFed model to fit
            data: Process data with train and test runs
            config: Fitting configuration
        """
        self.model = model
        self.data = data
        self.config = config or FitConfig()

        # Prepare training data
        self._prepare_training_data()

    def _prepare_training_data(self) -> None:
        """Prepare training data for fitting."""
        # Load data and extract relevant columns
        train_df = self.data.get_train_dataframe()

        # Group by run_id
        self._run_data = {}
        self._run_data_full = {}  # Store full time-series data for feed interpolation
        for run_id, group in train_df.groupby("run_id"):
            # Store full data for feed rate calculation (all time points)
            self._run_data_full[run_id] = {
                "time": group["time"].values,
                "feed": group["feed"].values / 1000.0 if "feed" in group.columns else None,  # mL to L
                "temp": group["temp"].values if "temp" in group.columns else None,
                "V": group["V"].values / 1000.0 if "V" in group.columns else None,  # mL to L
            }

            # Filter to rows with biomass measurements (for fitting objective)
            group_valid = group[group["X"].notna()].copy()
            if len(group_valid) > 0:
                self._run_data[run_id] = {
                    "time": group_valid["time"].values,
                    "X": group_valid["X"].values,  # g/L
                    "P": group_valid["P"].values if "P" in group_valid.columns else None,
                    "V": group_valid["V"].values / 1000.0,  # mL to L
                    "feed": group_valid["feed"].values / 1000.0 if "feed" in group_valid.columns else None,  # mL to L
                    "temp": group_valid["temp"].values if "temp" in group_valid.columns else None,
                    "induction": group_valid["induction"].values if "induction" in group_valid.columns else None,
                }

    def _get_active_params(self) -> list:
        """Get list of active parameter names based on model complexity."""
        complexity = self.config.model_complexity

        # All 15 parameter names in order
        all_params = [
            "Y_XG", "Y_PG",
            "gamma_max", "K_m", "K_G_inhib", "K_P_inhib", "K_X_inhib",
            "gamma_pi_max", "K_pi_m",
            "gamma_alpha_min", "K_alpha_gamma", "K_alpha_G", "K_alpha_n", "K_alpha_PX", "K_alpha_X",
        ]

        if complexity == ModelComplexity.FULL:
            return all_params
        elif complexity == ModelComplexity.REDUCED:
            # Paper's reduced model - 10 parameters
            # Removed: K_P_inhib, K_X_inhib, gamma_pi_max, K_alpha_G, K_alpha_X
            return [
                "Y_XG", "Y_PG",
                "gamma_max", "K_m", "K_G_inhib",
                "K_pi_m",
                "gamma_alpha_min", "K_alpha_gamma", "K_alpha_n", "K_alpha_PX",
            ]
        elif complexity == ModelComplexity.MINIMAL:
            # Very simple model - 6 parameters
            return [
                "Y_XG", "Y_PG",
                "gamma_max", "K_m",
                "K_pi_m", "gamma_alpha_min",
            ]
        else:
            return all_params

    def _get_fixed_params(self) -> dict:
        """Get fixed parameter values for inactive parameters."""
        # Values from paper Table 1 (disabled = large value)
        return {
            "K_P_inhib": 1e6,      # Product inhibition disabled
            "K_X_inhib": 1e6,      # Biomass inhibition disabled
            "gamma_pi_max": 1e-6,  # Production essentially disabled
            "K_alpha_G": 1e6,      # G activation in maintenance disabled
            "K_alpha_X": 1e6,      # X activation in maintenance disabled
        }

    def _get_bounds(self) -> list:
        """Get parameter bounds for optimization (only active parameters)."""
        active_params = self._get_active_params()

        # Map parameter names to bounds
        param_to_bounds = {
            "Y_XG": self.config.Y_XG_bounds,
            "Y_PG": self.config.Y_PG_bounds,
            "gamma_max": self.config.gamma_max_bounds,
            "K_m": self.config.K_m_bounds,
            "K_G_inhib": self.config.K_G_inhib_bounds,
            "K_P_inhib": self.config.K_P_inhib_bounds,
            "K_X_inhib": self.config.K_X_inhib_bounds,
            "gamma_pi_max": self.config.gamma_pi_max_bounds,
            "K_pi_m": self.config.K_pi_m_bounds,
            "gamma_alpha_min": self.config.gamma_alpha_min_bounds,
            "K_alpha_gamma": self.config.K_alpha_gamma_bounds,
            "K_alpha_G": self.config.K_alpha_G_bounds,
            "K_alpha_n": self.config.K_alpha_n_bounds,
            "K_alpha_PX": self.config.K_alpha_PX_bounds,
            "K_alpha_X": self.config.K_alpha_X_bounds,
        }

        bounds = [param_to_bounds[p] for p in active_params]
        return bounds

    def _vector_to_params(self, x: np.ndarray) -> ModelParameters:
        """Convert optimization vector to ModelParameters."""
        active_params = self._get_active_params()
        fixed_params = self._get_fixed_params()

        # Start with fixed values for all parameters
        params_dict = {
            "Y_XG": 0.627,
            "Y_PG": 0.652,
            "gamma_max": 0.49,
            "K_m": 0.001,
            "K_G_inhib": 89.0,
            "K_P_inhib": fixed_params["K_P_inhib"],
            "K_X_inhib": fixed_params["K_X_inhib"],
            "gamma_pi_max": fixed_params["gamma_pi_max"],
            "K_pi_m": 0.62,
            "gamma_alpha_min": 2.4e-5,
            "K_alpha_gamma": 1e-4,
            "K_alpha_G": fixed_params["K_alpha_G"],
            "K_alpha_n": 1.5,
            "K_alpha_PX": 0.11,
            "K_alpha_X": fixed_params["K_alpha_X"],
            "Gf": 400.0,
        }

        # Overwrite with optimized values
        for i, param_name in enumerate(active_params):
            params_dict[param_name] = x[i]

        return ModelParameters(**params_dict)

    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function for optimization.

        Calculates sum of squared errors between model predictions
        and experimental data across all training runs.

        Args:
            x: Parameter vector

        Returns:
            Sum of squared errors
        """
        params = self._vector_to_params(x)
        self.model.params = params

        total_error = 0.0
        n_points = 0
        debug_mode = False  # Set to True for detailed debugging

        for run_id, run_data in self._run_data.items():
            try:
                # Create initial conditions from first data point
                # Note: X is in g/L, V is in L, so X*V gives total mass in g
                V0 = run_data["V"][0]
                X0 = run_data["X"][0] * V0
                P0 = run_data["P"][0] * V0 if run_data["P"] is not None else 0.0
                G0 = 0.1 * V0  # Small initial substrate (0.1 g/L * V) to kickstart metabolism

                init_cond = InitialConditions(
                    X0=X0,
                    P0=P0,
                    G0=G0,
                    V0=V0,
                )
                self.model.init_cond = init_cond

                # Estimate feed rate from cumulative feed (use FULL time-series data)
                if run_id in self._run_data_full and self._run_data_full[run_id]["feed"] is not None:
                    full_data = self._run_data_full[run_id]
                    feed_cumulative = full_data["feed"]
                    time_full = full_data["time"]

                    # Calculate feed rate from full data
                    feed_rate = np.gradient(feed_cumulative, time_full)
                    feed_rate = np.maximum(feed_rate, 0)  # Clip negative values

                    # Create callable feed rate with proper interpolation
                    def feed_func(t):
                        if t <= time_full[0]:
                            return feed_rate[0]
                        elif t >= time_full[-1]:
                            return feed_rate[-1]
                        else:
                            return np.interp(t, time_full, feed_rate)

                    # Also use temperature data if available
                    temp_data = full_data.get("temp")
                    if temp_data is not None and len(temp_data) > 0:
                        def temp_func(t):
                            if t <= time_full[0]:
                                return temp_data[0]
                            elif t >= time_full[-1]:
                                return temp_data[-1]
                            else:
                                return np.interp(t, time_full, temp_data)
                    else:
                        # No temperature data - use midpoint of typical range
                        # WARNING: This may affect model accuracy for temperature-dependent parameters
                        import warnings
                        warnings.warn(
                            f"No temperature data for run {run_id}. Using midpoint of 25-42°C range (33.5°C). "
                            "Model fitting may be inaccurate if temperature varied significantly."
                        )
                        temp_func = 33.5  # Midpoint of typical 25-42°C range

                    control = ControlProfile(feed_rate=feed_func, temperature=temp_func)
                else:
                    # No feed data - batch fermentation
                    control = ControlProfile(feed_rate=lambda t: 0.0, temperature=33.5)

                # Simulate with stiff solver (Radau) for numerical stability
                result = self.model.simulate(
                    t_span=(run_data["time"][0], run_data["time"][-1]),
                    t_eval=run_data["time"],
                    control=control,
                    method="Radau",  # Stiff solver for better numerical stability
                    rtol=1e-4,
                    atol=1e-6,
                )

                # Calculate errors with normalization
                X_pred = result["X_conc"]
                X_true = run_data["X"]

                # Mask out any NaN or inf values
                valid_mask = np.isfinite(X_pred) & np.isfinite(X_true)
                if valid_mask.sum() > 0:
                    X_error = np.sum((X_pred[valid_mask] - X_true[valid_mask]) ** 2)
                    total_error += X_error
                    n_points += valid_mask.sum()

                    if debug_mode and run_id == list(self._run_data.keys())[0]:
                        X_rmse = np.sqrt(X_error / valid_mask.sum())
                        print(f"  {run_id}: X RMSE = {X_rmse:.4f}, n_points = {valid_mask.sum()}")

                if run_data["P"] is not None:
                    P_valid_mask = run_data["P"].notna() & np.isfinite(result["P_conc"])
                    if P_valid_mask.sum() > 0:
                        P_error = np.sum((result["P_conc"][P_valid_mask] - run_data["P"][P_valid_mask]) ** 2)
                        total_error += P_error
                        n_points += P_valid_mask.sum()

                        if debug_mode and run_id == list(self._run_data.keys())[0]:
                            P_rmse = np.sqrt(P_error / P_valid_mask.sum())
                            print(f"  {run_id}: P RMSE = {P_rmse:.4f}")

            except Exception as e:
                # Penalize failed simulations heavily
                if debug_mode:
                    print(f"  {run_id}: Simulation failed - {e}")
                total_error += 1e6
                n_points += 1

        return total_error / max(1, n_points)

    def _fit_differential_evolution(self, bounds: list) -> FitResult:
        """
        Fit using differential evolution.

        Args:
            bounds: Parameter bounds

        Returns:
            FitResult
        """
        result = differential_evolution(
            func=self._objective,
            bounds=bounds,
            maxiter=self.config.maxiter,
            popsize=self.config.popsize,
            tol=self.config.tol,
            workers=self.config.workers,
            disp=True,
        )

        fitted_params = self._vector_to_params(result.x)

        return FitResult(
            params=fitted_params,
            success=result.success,
            message=result.message,
            objective_value=result.fun,
            n_iterations=result.nit,
        )

    def _fit_cma_es(self, bounds: list) -> FitResult:
        """
        Fit using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

        CMA-ES is a stochastic derivative-free optimization algorithm that
        adapts a covariance matrix to guide the search. It often performs
        well on high-dimensional, ill-conditioned, or multimodal problems.

        Uses parameter scaling to handle widely different parameter ranges.

        Args:
            bounds: Parameter bounds

        Returns:
            FitResult
        """
        if not HAS_CMA:
            raise ImportError(
                "CMA-ES optimizer selected but 'cma' package is not installed. "
                "Install it with: pip install cma"
            )

        # Get active parameters for this model complexity
        active_params = self._get_active_params()
        n_params = len(active_params)

        # Literature-based initial values (from paper Table 1)
        # Map for all 15 parameters
        literature_values = {
            "Y_XG": 0.627,
            "Y_PG": 0.652,
            "gamma_max": 0.49,
            "K_m": 1.0e-3,
            "K_G_inhib": 89.0,
            "K_P_inhib": 1e6,      # Disabled
            "K_X_inhib": 1e6,      # Disabled
            "gamma_pi_max": 1e6,   # Effectively disabled
            "K_pi_m": 0.62,
            "gamma_alpha_min": 2.4e-5,
            "K_alpha_gamma": 1e-4,
            "K_alpha_G": 1e6,      # Disabled
            "K_alpha_n": 1.5,
            "K_alpha_PX": 0.11,
            "K_alpha_X": 1e6,      # Disabled
        }

        # Extract literature values for active parameters only
        literature_guess = np.array([literature_values[p] for p in active_params])

        # Clip to bounds
        x0 = np.clip(literature_guess, [b[0] for b in bounds], [b[1] for b in bounds])

        print(f"\n=== CMA-ES Optimization ({n_params} parameters: {self.config.model_complexity.value}) ===")
        print(f"Active parameters: {active_params}")
        print(f"Initial parameters (literature guess):")
        for i, p in enumerate(active_params):
            print(f"  {p:15s} = {x0[i]:.6g}")
        print()

        # Parameter scaling: transform to [0,1] space for better CMA-ES performance
        bounds_array = np.array(bounds)
        bound_ranges = bounds_array[:, 1] - bounds_array[:, 0]
        bound_mins = bounds_array[:, 0]

        # Scaling functions
        def to_scaled(x):
            """Transform from parameter space to [0,1] scaled space."""
            return (x - bound_mins) / bound_ranges

        def from_scaled(x_scaled):
            """Transform from [0,1] scaled space to parameter space."""
            return x_scaled * bound_ranges + bound_mins

        # Initial point in scaled space
        x0_scaled = to_scaled(x0)

        # In scaled space, sigma is relative to [0,1] range
        # 0.2 is a reasonable default (20% of the scaled range)
        sigma0_scaled = self.config.cma_sigma0 if self.config.cma_sigma0 is not None else 0.2

        # Set up CMA-ES options (in scaled space, bounds are [0,1])
        cma_options = {
            'bounds': [0, 1],  # Simple bounds in scaled space
            'maxfevals': self.config.cma_maxfevals,
            'tolfun': self.config.cma_tolfun,
            'tolx': self.config.cma_tolx,
            'verbose': self.config.cma_verbose,
        }

        # Add optional population size
        if self.config.cma_popsize is not None:
            cma_options['popsize'] = self.config.cma_popsize

        print(f"Initial step size (scaled): {sigma0_scaled:.4f}")
        print("=" * 40)

        # Wrapped objective function that handles scaling
        def objective_scaled(x_scaled):
            x = from_scaled(x_scaled)
            return self._objective(x)

        # Run CMA-ES
        try:
            result = cma.fmin(
                objective_function=objective_scaled,
                x0=x0_scaled,
                sigma0=sigma0_scaled,
                options=cma_options,
            )

            # CMA-ES returns (x_best_scaled, f_best) tuple
            x_best_scaled, f_best = result[0], result[1]

            # Transform back to parameter space
            x_best = from_scaled(x_best_scaled)
            fitted_params = self._vector_to_params(x_best)

            # cma.fmin returns tuple: (x_best, f_best, evals, iter, ..., es, logger)
            # result[2] is the evaluation count (int), result[9] is CMAEvolutionStrategy
            try:
                if len(result) >= 3:
                    n_evals = int(result[2])  # Evaluation count is at index 2
                else:
                    n_evals = 0
            except:
                n_evals = 0

            print(f"\n=== CMA-ES Complete ===")
            print(f"Final objective value: {f_best:.6g}")
            print(f"Function evaluations: {n_evals if n_evals > 0 else 'N/A'}")

            return FitResult(
                params=fitted_params,
                success=True,
                message="CMA-ES completed successfully",
                objective_value=f_best,
                n_iterations=n_evals,
            )

        except Exception as e:
            return FitResult(
                params=self._vector_to_params(x0),
                success=False,
                message=f"CMA-ES failed: {str(e)}",
                objective_value=np.inf,
                n_iterations=0,
            )

    def _fit_hybrid(self, bounds: list) -> FitResult:
        """
        Fit using hybrid approach: global search (CMA-ES or DE) + local refinement.

        Args:
            bounds: Parameter bounds

        Returns:
            FitResult
        """
        print("\n=== Hybrid Optimization ===")
        print(f"Stage 1: Global search with {self.config.optimizer.value}")

        # Stage 1: Global optimization
        if self.config.optimizer == OptimizerType.HYBRID:
            # Use CMA-ES for global search (default for hybrid)
            global_result = self._fit_cma_es(bounds)
        else:
            global_result = self._fit_differential_evolution(bounds)

        if not global_result.success:
            print("Global optimization failed, skipping local refinement")
            return global_result

        print(f"Global search complete: objective = {global_result.objective_value:.6g}")

        # Stage 2: Local refinement
        if self.config.local_refinement:
            print(f"Stage 2: Local refinement with {self.config.local_method}")

            # Convert params back to vector for local optimization
            # Use active parameters only
            active_params = self._get_active_params()
            param_dict = global_result.params.__dict__
            x0 = np.array([param_dict[p] for p in active_params])

            # Local optimization
            local_result = minimize(
                fun=self._objective,
                x0=x0,
                bounds=bounds,
                method=self.config.local_method,
                options={'maxiter': self.config.local_maxiter},
            )

            if local_result.success:
                fitted_params = self._vector_to_params(local_result.x)
                final_objective = local_result.fun

                print(f"Local refinement complete: objective = {final_objective:.6g}")

                return FitResult(
                    params=fitted_params,
                    success=True,
                    message=f"Hybrid optimization: global + local refinement",
                    objective_value=final_objective,
                    n_iterations=global_result.n_iterations + local_result.nit,
                )
            else:
                print(f"Local refinement failed: {local_result.message}")
                print("Using global optimization result")

        return global_result

    def fit(self, optimizer: Optional[OptimizerType] = None) -> FitResult:
        """
        Fit model parameters using the configured optimizer.

        Supported optimizers:
        - DIFFERENTIAL_EVOLUTION: scipy's differential evolution (global, derivative-free)
        - CMA_ES: Covariance Matrix Adaptation Evolution Strategy (global, derivative-free)
        - HYBRID: Global search + local refinement (best of both worlds)

        Args:
            optimizer: Override the configured optimizer type

        Returns:
            FitResult with fitted parameters and metadata
        """
        # Determine which optimizer to use
        opt_type = optimizer or self.config.optimizer

        bounds = self._get_bounds()

        if opt_type == OptimizerType.CMA_ES:
            return self._fit_cma_es(bounds)
        elif opt_type == OptimizerType.HYBRID:
            return self._fit_hybrid(bounds)
        else:  # DIFFERENTIAL_EVOLUTION
            return self._fit_differential_evolution(bounds)

    def cross_validate(self, n_folds: Optional[int] = None) -> dict:
        """
        Perform k-fold cross-validation on the training data.

        Args:
            n_folds: Number of folds (default: from config)

        Returns:
            Dictionary with CV scores (mean and std of objective values)
        """
        n_folds = n_folds or self.config.n_folds

        run_ids = list(self._run_data.keys())
        if len(run_ids) < n_folds:
            n_folds = len(run_ids)

        fold_scores = []

        for i in range(n_folds):
            # Split runs into train and validation
            val_runs = [run_ids[j] for j in range(i, len(run_ids), n_folds)]
            train_runs = [r for r in run_ids if r not in val_runs]

            # Create training data for this fold
            fold_data = {k: v for k, v in self._run_data.items() if k in train_runs}

            # Temporarily replace training data
            original_data = self._run_data
            self._run_data = fold_data

            # Fit and evaluate
            try:
                result = self.fit()
                fold_scores.append(result.objective_value)
            except Exception as e:
                print(f"Fold {i+1} failed: {e}")
                fold_scores.append(np.inf)

            # Restore original data
            self._run_data = original_data

        return {
            "mean_score": np.mean(fold_scores),
            "std_score": np.std(fold_scores),
            "fold_scores": fold_scores,
        }


class ModelSelector:
    """
    Model selection based on significance testing.

    Implements the heuristic algorithm from the paper for removing
    insignificant model terms.
    """

    def __init__(
        self,
        model: OptFedModel,
        data: ProcessData,
        config: Optional[FitConfig] = None,
    ):
        """
        Initialize the model selector.

        Args:
            model: OptFed model
            data: Process data
            config: Fitting configuration
        """
        self.model = model
        self.data = data
        self.config = config or FitConfig()
        self.estimator = ParameterEstimator(model, data, config)

    def select_model(self, alpha: float = 0.05) -> ModelParameters:
        """
        Perform model selection by removing insignificant parameters.

        Uses iterative F-test based selection:
        1. Fit full model
        2. For each candidate term, test significance by comparing reduced model
        3. Remove least significant term if F < F_critical
        4. Repeat until all terms are significant

        Args:
            alpha: Significance level for F-test

        Returns:
            Selected model parameters
        """
        # Start with full model
        full_result = self.estimator.fit()
        current_params = full_result.params
        current_sse = full_result.objective_value

        # Degrees of freedom: n_samples - n_parameters
        n_samples = self._count_data_points()
        df_full = n_samples - 15  # 15 fitted parameters in full model

        print("=== Model Selection: F-test Based ===")
        print(f"Full model SSE: {current_sse:.6g}")
        print(f"Degrees of freedom: {df_full}")

        # Define candidate terms to test for removal
        # Format: (parameter_name, description, fixed_value_if_removed, type)
        candidate_terms = [
            ("K_G_inhib", "Substrate inhibition in gamma", 1e6, "inhibition"),
            ("K_P_inhib", "Product inhibition in gamma", 1e6, "inhibition"),
            ("K_X_inhib", "Biomass inhibition in gamma", 1e6, "inhibition"),
            ("K_alpha_gamma", "Gamma activation in maintenance", 1e-6, "maintenance"),
            ("K_alpha_G", "Substrate activation in maintenance", 1e-6, "maintenance"),
            ("K_alpha_n", "Generation activation in maintenance", 1e-6, "maintenance"),
            ("K_alpha_PX", "P/X activation in maintenance", 1e-6, "maintenance"),
            ("K_alpha_X", "Biomass activation in maintenance", 1e-6, "maintenance"),
            ("K_pi_m", "Michaelis constant for production", 0.5, "production"),
        ]

        removed_terms = []
        iteration = 0

        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Test each remaining candidate term
            f_statistics = {}
            
            for param_name, description, fixed_val, term_type in candidate_terms:
                if param_name in removed_terms:
                    continue

                # Estimate SSE for reduced model using sensitivity approximation
                # This avoids expensive re-fitting for each term
                reduced_sse = self._estimate_reduced_sse(
                    current_sse, current_params, param_name, fixed_val, term_type
                )

                # Calculate degrees of freedom for reduced model
                n_params_reduced = 15 - len(removed_terms) - 1
                df_reduced = n_samples - n_params_reduced

                # Calculate F-statistic
                sse_diff = reduced_sse - current_sse
                df_diff = df_reduced - df_full
                
                if df_diff > 0 and current_sse > 0:
                    F_stat = (sse_diff / df_diff) / (current_sse / df_full)
                else:
                    F_stat = 0

                f_statistics[param_name] = {
                    "F": F_stat,
                    "SSE_reduced": reduced_sse,
                    "SSE_diff": sse_diff,
                    "df_diff": df_diff,
                    "description": description,
                    "reduced_df": df_reduced,
                    "fixed_value": fixed_val
                }

            # Find least significant term (lowest F-statistic)
            if not f_statistics:
                print("No more terms to test.")
                break

            min_f_param = min(f_statistics.keys(), key=lambda k: f_statistics[k]["F"])
            min_f_value = f_statistics[min_f_param]["F"]

            print(f"Least significant: {min_f_param} ({f_statistics[min_f_param]['description']})")
            print(f"  F-statistic: {min_f_value:.4g}")
            print(f"  SSE increase: {f_statistics[min_f_param]['SSE_diff']:.6g}")

            # Critical F-value for alpha=0.05
            # F(1, df_large) ≈ 3.84 for alpha=0.05
            f_critical = 3.84

            if min_f_value < f_critical:
                # Remove the term
                removed_terms.append(min_f_param)
                current_params = self._fix_parameter(
                    current_params, min_f_param, 
                    f_statistics[min_f_param]["fixed_value"]
                )
                current_sse = f_statistics[min_f_param]["SSE_reduced"]
                df_full = f_statistics[min_f_param]["reduced_df"]
                print(f"  → REMOVED (F < {f_critical:.2f})")
            else:
                print(f"  → KEPT (F >= {f_critical:.2f})")
                print("\nAll remaining terms are significant.")
                break

        print(f"\n=== Selection Complete ===")
        print(f"Removed {len(removed_terms)} terms: {removed_terms}")
        print(f"Final SSE: {current_sse:.6g}")

        return current_params

    def _count_data_points(self) -> int:
        """Count total number of data points across all training runs."""
        count = 0
        for run_id, run_data in self.estimator._run_data.items():
            count += len(run_data["X"])
        return count

    def _estimate_reduced_sse(
        self, 
        base_sse: float,
        base_params: ModelParameters, 
        param_name: str, 
        fixed_value: float,
        term_type: str
    ) -> float:
        """
        Estimate SSE for reduced model using sensitivity-based approximation.
        
        Instead of full re-fitting, estimates the SSE increase from removing a term
        based on the parameter's current contribution to the model fit.
        """
        # Get current parameter value
        param_value = getattr(base_params, param_name)
        
        if term_type == "inhibition":
            # Removing inhibition: large value means no inhibition
            # Approximate SSE increase proportional to how much the term is currently used
            if param_value > 100:  # Already weak inhibition
                sse_increase = base_sse * 0.01  # 1% increase
            else:
                sse_increase = base_sse * (100.0 / (param_value + 1)) * 0.05
        elif term_type == "maintenance":
            # Removing maintenance activation: very small value means no activation
            # Approximate SSE increase based on activation strength
            if param_value < 0.1:  # Already weak activation
                sse_increase = base_sse * 0.02
            else:
                sse_increase = base_sse * (param_value / 10.0) * 0.1
        else:
            # Other terms
            sse_increase = base_sse * 0.05
        
        return base_sse + min(sse_increase, base_sse * 0.5)  # Cap at 50% increase

    def _fix_parameter(
        self, 
        params: ModelParameters, 
        param_name: str, 
        fixed_value: float
    ) -> ModelParameters:
        """Create a new ModelParameters with one parameter fixed."""
        param_dict = params.__dict__.copy()
        param_dict[param_name] = fixed_value
        return ModelParameters(**param_dict)


def fit_model_to_data(
    data_path: str | Path,
    output_dir: Optional[str | Path] = None,
    optimizer: OptimizerType = OptimizerType.CMA_ES,
    config: Optional[FitConfig] = None,
) -> FitResult:
    """
    Convenience function to fit the OptFed model to data.

    Args:
        data_path: Path to traintest_compiled.xlsx
        output_dir: Directory to save results
        optimizer: Optimizer type to use (DE, CMA-ES, or HYBRID)
        config: Optional FitConfig with custom optimization settings

    Returns:
        FitResult with fitted parameters
    """
    # Load data
    loader = DataLoader(data_path, TRAIN_RUN_IDS)
    data = loader.load_data()

    # Create model with default initial parameters
    params = ModelParameters(Gf=400.0)
    init_cond = InitialConditions(
        X0=1.0,  # Will be updated per run
        P0=0.0,
        G0=0.0,
        V0=1.1,
    )
    model = OptFedModel(params, init_cond)

    # Create config with specified optimizer
    fit_config = config or FitConfig()
    fit_config.optimizer = optimizer

    # Fit parameters
    estimator = ParameterEstimator(model, data, fit_config)
    result = estimator.fit()

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save parameters
        with open(output_dir / "fitted_params.pkl", "wb") as f:
            pickle.dump(result, f)

        # Save as text
        with open(output_dir / "fitted_params.txt", "w") as f:
            f.write("Fitted Model Parameters\n")
            f.write("=" * 40 + "\n")
            f.write(f"Optimizer: {optimizer.value}\n")
            f.write(f"Message: {result.message}\n")
            f.write("-" * 40 + "\n")
            for key, value in result.params.__dict__.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.6g}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("-" * 40 + "\n")
            f.write(f"\nObjective value: {result.objective_value:.6g}\n")
            f.write(f"Success: {result.success}\n")
            f.write(f"Iterations/Evaluations: {result.n_iterations}\n")

    return result


if __name__ == "__main__":
    # Example: Fit model to training data with different optimizers

    # Choose optimizer: "differential_evolution", "cma_es", or "hybrid"
    optimizer_choice = OptimizerType.CMA_ES

    data_path = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\data\traintest_compiled.xlsx"
    output_dir = Path(r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\optfed\results")

    result = fit_model_to_data(data_path, output_dir, optimizer=optimizer_choice)

    print("\nFitted Parameters:")
    for key, value in result.params.__dict__.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6g}")
