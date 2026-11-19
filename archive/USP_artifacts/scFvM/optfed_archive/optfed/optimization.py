"""
Optimization for OptFed mechanistic model (Stage III).

Implements optimization of control variables (feed rate, temperature) to
maximize objective functions such as final product titer.

Based on: Schlögel et al. (2024)
"""

import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from model import (
    ControlProfile,
    FeedStrategy,
    InitialConditions,
    ModelParameters,
    OptFedModel,
)
from scipy.optimize import Bounds, NonlinearConstraint, minimize


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MAXIMIZE_PRODUCT_TITER = "maximize_product_titer"  # P(t_end)
    MAXIMIZE_YIELD = "maximize_yield"  # P/X at t_end
    MAXIMIZE_PRODUCTIVITY = "maximize_productivity"  # P/(V*t_end)
    CUSTOM = "custom"


@dataclass
class OptimizationConstraints:
    """Constraints for optimization."""
    V_max: float = 2.5  # Maximum reactor volume (L)
    t_max: float = 12.0  # Maximum process time (h)
    f_min: float = 0.0  # Minimum feed rate (L/h)
    f_max: float = 0.5  # Maximum feed rate (L/h)
    T_min: float = 25.0  # Minimum temperature (°C)
    T_max: float = 42.0  # Maximum temperature (°C)
    Gf: float = 400.0  # Feed substrate concentration (g/L) - fixed


@dataclass
class OptimizationResult:
    """Results from optimization."""
    success: bool
    message: str
    objective_value: float
    control_profile: ControlProfile
    final_state: dict
    optimization_trajectory: Optional[list] = None


class FeedParameterization:
    """
    Parameterizations for feed rate profiles.

    Supports:
    - Constant feed
    - Linear feed: f(t) = phi1 + phi2 * t
    - Piecewise constant feed
    """

    @staticmethod
    def constant(n_intervals: int = 1) -> Callable[[np.ndarray], Callable[[float], float]]:
        """
        Create constant feed rate function.

        Args:
            n_intervals: Number of intervals with different constant rates

        Returns:
            Function that takes parameter vector and returns feed rate function
        """
        def feed_func(params: np.ndarray) -> Callable[[float], float]:
            rates = params[:n_intervals]
            if n_intervals == 1:
                rate_val = float(rates[0])
                return lambda t: float(np.maximum(0.0, rate_val))
            else:
                # Piecewise constant
                interval_length = 12.0 / n_intervals  # Default to 12h
                def f(t):
                    t_val = float(t) if np.isscalar(t) else float(t[0])
                    idx = min(int(t_val / interval_length), n_intervals - 1)
                    return float(np.maximum(0.0, float(rates[idx])))
                return f
        return feed_func

    @staticmethod
    def linear() -> Callable[[np.ndarray], Callable[[float], float]]:
        """
        Create linear feed rate function: f(t) = phi1 + phi2 * t

        Returns:
            Function that takes [phi1, phi2] and returns feed rate function
        """
        def feed_func(params: np.ndarray) -> Callable[[float], float]:
            phi1, phi2 = float(params[0]), float(params[1])
            def f(t):
                # Handle both scalar and array t (for scipy gradient computation)
                t_val = float(t) if np.isscalar(t) else float(t[0])
                return float(np.maximum(0.0, phi1 + phi2 * t_val))
            return f
        return feed_func

    @staticmethod
    def exponential() -> Callable[[np.ndarray], Callable[[float], float]]:
        """
        Create exponential feed rate function: f(t) = f0 * exp(mu_f * t)

        Returns:
            Function that takes [f0, mu_f] and returns feed rate function
        """
        def feed_func(params: np.ndarray) -> Callable[[float], float]:
            f0, mu_f = float(params[0]), float(params[1])
            def f(t):
                # Handle both scalar and array t (for scipy gradient computation)
                t_val = float(t) if np.isscalar(t) else float(t[0])
                return float(np.maximum(0.0, f0 * np.exp(mu_f * t_val)))
            return f
        return feed_func


class ProcessOptimizer:
    """
    Optimize fed-batch process control variables.

    Maximize objective function subject to:
    - Model dynamics (differential equations)
    - Volume constraint: V <= V_max
    - Time constraint: t_end <= t_max
    """

    def __init__(
        self,
        model: OptFedModel,
        constraints: Optional[OptimizationConstraints] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            model: Fitted OptFed model
            constraints: Optimization constraints
        """
        self.model = model
        self.constraints = constraints or OptimizationConstraints()

    def _simulate_with_control(
        self,
        control_params: np.ndarray,
        feed_parameterization: Callable[[np.ndarray], Callable[[float], float]],
        temperature: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> dict:
        """
        Simulate process with given control parameters.

        Args:
            control_params: Feed parameters
            feed_parameterization: Function to create feed rate from params
            temperature: Constant temperature (°C)
            t_end: Process end time (h)

        Returns:
            Simulation result dictionary
        """
        # Extract control variables
        t_end = t_end or self.constraints.t_max
        feed_func = feed_parameterization(control_params)

        # Create control profile (temperature must be specified for optimization)
        if temperature is None:
            temperature = (self.constraints.T_min + self.constraints.T_max) / 2
        control = ControlProfile(feed_rate=feed_func, temperature=temperature)

        # Simulate
        try:
            result = self.model.simulate(
                t_span=(0, t_end),
                t_eval=np.linspace(0, t_end, 100),
                control=control,
            )

            return {
                "success": True,
                "result": result,
                "t_end": t_end,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "t_end": t_end,
            }

    def _objective_product_titer(self, sim_result: dict) -> float:
        """
        Objective: maximize final product titer P(t_end).

        Returns negative for minimization.
        """
        if not sim_result["success"]:
            return 1e6  # Penalty for failed simulation

        result = sim_result["result"]
        P_final = result["P_conc"][-1]  # g/L

        return -P_final  # Negative for minimization

    def _objective_yield(self, sim_result: dict) -> float:
        """
        Objective: maximize yield P/X at t_end.
        """
        if not sim_result["success"]:
            return 1e6

        result = sim_result["result"]
        P_final = result["P_conc"][-1]
        X_final = result["X_conc"][-1]

        if X_final < 0.01:
            return 1e6

        yield_px = P_final / X_final
        return -yield_px

    def _objective_productivity(self, sim_result: dict) -> float:
        """
        Objective: maximize volumetric productivity P/(V*t_end).
        """
        if not sim_result["success"]:
            return 1e6

        result = sim_result["result"]
        P_final = result["P_conc"][-1]
        V_final = result["V"][-1]
        t_end = sim_result["t_end"]

        if V_final < 0.1 or t_end < 0.1:
            return 1e6

        productivity = P_final * V_final / t_end
        return -productivity

    def _volume_constraint(self, args_tuple: tuple) -> float:
        """
        Volume constraint: V(t) <= V_max for all t.

        Returns V_max - max(V(t)) (should be >= 0)

        Args:
            args_tuple: (x, feed_func, t_end, temperature) where x is optimization vector
        """
        control_params, feed_func, t_end, temperature = args_tuple
        control = ControlProfile(feed_rate=feed_func(control_params), temperature=temperature)

        try:
            result = self.model.simulate(
                t_span=(0, t_end),
                t_eval=np.linspace(0, t_end, 50),
                control=control,
                method="LSODA",
            )
            max_volume = result["V"].max()
            return self.constraints.V_max - max_volume
        except:
            return -self.constraints.V_max  # Violation

    def _feed_nonneg_constraint(self, args_tuple: tuple) -> float:
        """
        Feed non-negativity constraint: f(t) >= 0 for all t in [0, t_end].

        Returns min(f(t)) over time (should be >= 0).

        Args:
            args_tuple: (x, feed_func, t_end) where x is optimization vector
        """
        control_params, feed_func_or_value, t_end = args_tuple

        # Check if feed_func_or_value is a callable or a constant
        if callable(feed_func_or_value):
            # It's a function - evaluate at multiple time points
            t_check = np.linspace(0, t_end, 20)
            feed_values = np.array([float(feed_func_or_value(t)) for t in t_check])
            return float(feed_values.min())
        else:
            # It's a constant value
            return float(feed_func_or_value)

    def optimize(
        self,
        objective_type: ObjectiveType = ObjectiveType.MAXIMIZE_PRODUCT_TITER,
        feed_type: str = "linear",
        temperature: Optional[float] = None,
        t_end: Optional[float] = None,
        initial_guess: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Optimize process control variables.

        Args:
            objective_type: Type of objective to maximize
            feed_type: Type of feed parameterization ('constant', 'linear', 'exponential')
            temperature: Fixed temperature (°C), or None to optimize
            t_end: Process time (h), or None to use t_max
            initial_guess: Initial parameter values

        Returns:
            OptimizationResult with optimal control profile
        """
        # Select feed parameterization
        if feed_type == "constant":
            feed_param_func = FeedParameterization.constant()
            n_params = 1
            bounds = [(self.constraints.f_min, self.constraints.f_max)]
            initial_guess = initial_guess or np.array([0.05])
        elif feed_type == "linear":
            feed_param_func = FeedParameterization.linear()
            n_params = 2
            bounds = [
                (self.constraints.f_min, self.constraints.f_max),
                (-self.constraints.f_max, self.constraints.f_max),
            ]
            initial_guess = initial_guess or np.array([0.02, 0.001])
        elif feed_type == "exponential":
            feed_param_func = FeedParameterization.exponential()
            n_params = 2
            bounds = [
                (self.constraints.f_min, self.constraints.f_max),
                (0.0, 0.5),  # mu_f
            ]
            initial_guess = initial_guess or np.array([0.01, 0.1])
        else:
            raise ValueError(f"Unknown feed type: {feed_type}")

        t_end = t_end or self.constraints.t_max
        if temperature is None:
            temperature = (self.constraints.T_min + self.constraints.T_max) / 2

        # Select objective function
        if objective_type == ObjectiveType.MAXIMIZE_PRODUCT_TITER:
            objective_func = self._objective_product_titer
        elif objective_type == ObjectiveType.MAXIMIZE_YIELD:
            objective_func = self._objective_yield
        elif objective_type == ObjectiveType.MAXIMIZE_PRODUCTIVITY:
            objective_func = self._objective_productivity
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")

        # Define objective for scipy
        def objective(x):
            sim_result = self._simulate_with_control(
                x, feed_param_func, temperature, t_end
            )
            return objective_func(sim_result)

        # Define constraint for volume
        feed_func = feed_param_func(initial_guess)
        volume_constraint = {
            "type": "ineq",
            "fun": lambda x: self._volume_constraint((x, feed_param_func(x), t_end, temperature)),
        }

        # Define constraint for feed non-negativity
        feed_nonneg_constraint = {
            "type": "ineq",
            "fun": lambda x: self._feed_nonneg_constraint((x, feed_param_func(x), t_end)),
        }

        # Run optimization
        result = minimize(
            fun=objective,
            x0=initial_guess,
            bounds=bounds,
            constraints=[volume_constraint, feed_nonneg_constraint],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-6},
        )

        # Get final simulation with optimal parameters
        final_sim = self._simulate_with_control(
            result.x, feed_param_func, temperature, t_end
        )

        # Create control profile
        optimal_feed = feed_param_func(result.x)
        optimal_control = ControlProfile(
            feed_rate=optimal_feed,
            temperature=temperature,
        )

        return OptimizationResult(
            success=result.success and final_sim["success"],
            message=result.message,
            objective_value=-result.fun,  # Convert back to maximization
            control_profile=optimal_control,
            final_state=final_sim["result"] if final_sim["success"] else {},
        )


def optimize_process(
    params: ModelParameters,
    output_dir: Optional[str | Path] = None,
) -> OptimizationResult:
    """
    Convenience function to optimize the process.

    Args:
        params: Fitted model parameters
        output_dir: Directory to save results

    Returns:
        OptimizationResult with optimal control profile
    """
    # Create model with appropriate initial conditions
    # Based on typical data: X ~10 g/L, V ~1.1 L at t=0
    init_cond = InitialConditions(
        X0=11.0,  # Total biomass (g) = 10 g/L * 1.1 L
        P0=0.0,
        G0=0.5,   # Small initial substrate to kickstart metabolism
        V0=1.1,
    )
    model = OptFedModel(params, init_cond)

    # Create optimizer
    constraints = OptimizationConstraints(
        V_max=2.5,
        t_max=12.0,
        f_min=0.0,
        f_max=0.3,
    )
    optimizer = ProcessOptimizer(model, constraints)

    # Optimize for maximum product titer with linear feed
    # Temperature will be optimized within [T_min, T_max] bounds
    result = optimizer.optimize(
        objective_type=ObjectiveType.MAXIMIZE_PRODUCT_TITER,
        feed_type="linear",
        temperature=None,  # Will use midpoint of constraint range as initial guess
    )

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "optimization_result.pkl", "wb") as f:
            pickle.dump(result, f)

        with open(output_dir / "optimization_result.txt", "w") as f:
            f.write("Optimization Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Success: {result.success}\n")
            f.write(f"Message: {result.message}\n")
            f.write(f"Objective value: {result.objective_value:.4f} g/L\n\n")

            if result.final_state:
                f.write("Final State:\n")
                f.write(f"  X: {result.final_state['X_conc'][-1]:.2f} g/L\n")
                f.write(f"  P: {result.final_state['P_conc'][-1]:.2f} g/L\n")
                f.write(f"  V: {result.final_state['V'][-1]:.2f} L\n")

    return result


if __name__ == "__main__":
    # Use fitted parameters from Stage II
    # Update these values based on your fitting results
    params = ModelParameters(
        Y_XG=0.627,
        Y_PG=0.5,
        gamma_max=0.49,
        K_m=0.001,
        K_G_inhib=89,
        K_P_inhib=50,
        K_X_inhib=500,
        gamma_pi_max=1,
        K_pi_m=0.5,
        gamma_alpha_min=0.001,
        K_alpha_gamma=0.1,
        K_alpha_G=5,
        K_alpha_n=1.5,
        K_alpha_PX=0.11,
        K_alpha_X=200,
        Gf=400.0,
    )

    result = optimize_process(params)

    print(f"Optimization successful: {result.success}")
    print(f"Objective value: {result.objective_value:.2f} g/L")
    if result.final_state:
        print(f"Final product: {result.final_state['P_conc'][-1]:.2f} g/L")
        print(f"Final biomass: {result.final_state['X_conc'][-1]:.2f} g/L")
        print(f"Final volume: {result.final_state['V'][-1]:.2f} L")
