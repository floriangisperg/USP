"""
Incremental Model Fitting Troubleshooting Script - FAST VERSION

This script systematically tests model fitting with increasing complexity levels
to identify where the fitting starts to fail. It uses only 2 fermentation runs
for quick testing and generates comprehensive validation plots and metrics.

This is a FAST version with reduced iterations for quick testing.

Usage:
    python troubleshoot_incremental_fast.py

Author: Troubleshooting script for OptFed model fitting
Date: 2024-12-26
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

# Progress reporting
def print_progress(msg: str):
    """Print progress message with flush."""
    print(msg, flush=True)

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class SimpleRunData:
    """Simplified data container for a single fermentation run."""
    run_id: str
    time: np.ndarray  # hours
    X: np.ndarray     # g/L - biomass
    P: np.ndarray     # g/L - product
    V: np.ndarray     # L - volume
    feed: np.ndarray  # L - cumulative feed
    temp: np.ndarray  # C - temperature
    induction_time: Optional[float] = None  # hours - when induction occurs

    def __len__(self):
        return len(self.time)


def load_two_runs(
    data_path: str,
    run_ids: List[str] = ["ferm_01", "ferm_02"]
) -> List[SimpleRunData]:
    """Load exactly 2 fermentation runs for troubleshooting."""
    print_progress(f"Loading data from: {data_path}")

    df = pd.read_excel(data_path)
    print_progress(f"  Total rows in file: {len(df)}")

    runs = []
    for run_id in run_ids:
        run_df = df[df["RunID"] == run_id].copy()
        run_df = run_df.sort_values("time").reset_index(drop=True)

        # Find induction time
        ind_points = run_df[run_df["inductor_switch"] > 0]
        induction_time = ind_points["time"].min() if len(ind_points) > 0 else None

        # Filter to rows with biomass measurements
        run_df = run_df[run_df["X"].notna()].copy()

        if len(run_df) < 3:
            print_progress(f"Warning: {run_id} has only {len(run_df)} data points")
            continue

        runs.append(SimpleRunData(
            run_id=run_id,
            time=run_df["time"].values,
            X=run_df["X"].values,
            P=run_df.get("P", pd.Series([0]*len(run_df))).values,
            V=run_df["reactor_volume"].values / 1000.0,  # mL to L
            feed=run_df.get("feed", pd.Series([0]*len(run_df))).values / 1000.0,  # mL to L
            temp=run_df.get("temp", pd.Series([37.0]*len(run_df))).values,
            induction_time=induction_time,
        ))

        if induction_time is not None:
            print_progress(f"  {run_id}: induction at {induction_time:.2f} h")

    print_progress(f"  Loaded {len(runs)} runs successfully")
    return runs


def calculate_feed_rate_from_cumulative(time: np.ndarray, feed_cumulative: np.ndarray) -> np.ndarray:
    """Calculate instantaneous feed rate from cumulative feed."""
    feed_rate = np.gradient(feed_cumulative, time)
    return np.maximum(feed_rate, 0)  # Clip negative values


# =============================================================================
# MODEL DEFINITIONS - INCREASING COMPLEXITY
# =============================================================================

@dataclass
class ModelResult:
    """Result from model fitting."""
    success: bool
    params: dict
    objective_value: float
    message: str
    n_iterations: int


class BaseModel:
    """Base class for all model variants."""

    def __init__(self, params: dict):
        self.params = params
        self.param_names = list(params.keys())

    def simulate(
        self,
        t_eval: np.ndarray,
        feed_func: Callable[[float], float],
        temp_func: Callable[[float], float],
        X0: float,
        P0: float,
        V0: float,
        G0: float = 0.1,
        induction_time: Optional[float] = None,
    ) -> dict:
        """Simulate the model. Must be implemented by subclasses."""
        raise NotImplementedError


# Level 0: Simple Monod Growth (no maintenance, no product, no inhibition)
class MonodModel(BaseModel):
    """
    Level 0: Simplest possible model.

    dX/dt = mu * X - (f/V) * X
    dV/dt = f

    mu = mu_max * S / (Ks + S)

    Parameters: mu_max, Ks, Y_XS
    """

    def __init__(self, params: dict):
        super().__init__(params)
        # Default parameters
        self.params = {
            "mu_max": 0.5,    # 1/h - maximum growth rate
            "Ks": 0.1,        # g/L - half-saturation constant
            "Y_XS": 0.5,      # g/g - biomass yield on substrate
            **params
        }

    def simulate(
        self,
        t_eval: np.ndarray,
        feed_func: Callable[[float], float],
        temp_func: Callable[[float], float],
        X0: float,
        P0: float,
        V0: float,
        G0: float = 0.1,
        Sf: float = 400.0,
        induction_time: Optional[float] = None,  # Not used but for consistency
    ) -> dict:
        # Unpack parameters
        mu_max = self.params["mu_max"]
        Ks = self.params["Ks"]
        Y_XS = self.params["Y_XS"]

        def derivatives(t, y):
            X, S, V = y  # Biomass (g), Substrate (g), Volume (L)

            f = feed_func(t)

            # Concentrations
            S_conc = S / V if V > 0 else 0

            # Monod growth
            mu = mu_max * S_conc / (Ks + S_conc)

            # ODEs
            dXdt = mu * X - (f / V) * X if V > 0 else 0
            dSdt = -mu * X / Y_XS + (f / V) * (Sf - S_conc) if V > 0 else 0
            dVdt = f

            return [dXdt, dSdt, dVdt]

        y0 = [X0 * V0, G0 * V0, V0]  # Convert X0 to total mass

        sol = solve_ivp(
            fun=derivatives,
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        V = sol.y[2, :]
        X_conc = sol.y[0, :] / V
        S_conc = sol.y[1, :] / V

        return {
            "t": sol.t,
            "X_conc": X_conc,
            "P_conc": np.zeros_like(X_conc),  # No product in this model
            "V": V,
            "S_conc": S_conc,
            "success": True,
        }


# Level 1: Monod + Product Formation
class MonodProductModel(BaseModel):
    """
    Level 1: Add product formation (Luedeking-Piret).

    dX/dt = mu * X - (f/V) * X
    dP/dt = pi * X - (f/V) * P
    dV/dt = f

    mu = mu_max * S / (Ks + S)
    pi = alpha * mu + beta  (growth-associated + non-growth associated)

    Parameters: mu_max, Ks, Y_XS, alpha, beta, Y_PS
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.params = {
            "mu_max": 0.5,
            "Ks": 0.1,
            "Y_XS": 0.5,
            "alpha": 0.1,    # gP/gX - growth-associated product
            "beta": 0.01,    # gP/gX/h - non-growth-associated
            "Y_PS": 0.2,     # g/g - product yield
            **params
        }

    def simulate(
        self,
        t_eval: np.ndarray,
        feed_func: Callable[[float], float],
        temp_func: Callable[[float], float],
        X0: float,
        P0: float,
        V0: float,
        G0: float = 0.1,
        Sf: float = 400.0,
        induction_time: Optional[float] = None,
    ) -> dict:
        mu_max = self.params["mu_max"]
        Ks = self.params["Ks"]
        Y_XS = self.params["Y_XS"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        Y_PS = self.params["Y_PS"]

        def derivatives(t, y):
            X, P, S, V = y

            f = feed_func(t)
            S_conc = S / V if V > 0 else 0

            mu = mu_max * S_conc / (Ks + S_conc)

            # Product formation only after induction
            if induction_time is not None and t >= induction_time:
                pi = alpha * mu + beta
            else:
                pi = 0.0

            dXdt = mu * X - (f / V) * X if V > 0 else 0
            dPdt = pi * X - (f / V) * P if V > 0 else 0
            dSdt = -mu * X / Y_XS - pi * X / Y_PS + (f / V) * (Sf - S_conc) if V > 0 else 0
            dVdt = f

            return [dXdt, dPdt, dSdt, dVdt]

        y0 = [X0 * V0, P0 * V0, G0 * V0, V0]

        sol = solve_ivp(
            fun=derivatives,
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        V = sol.y[3, :]
        X_conc = sol.y[0, :] / V
        P_conc = sol.y[1, :] / V
        S_conc = sol.y[2, :] / V

        return {
            "t": sol.t,
            "X_conc": X_conc,
            "P_conc": P_conc,
            "V": V,
            "S_conc": S_conc,
            "success": True,
        }


# Level 2: Add Substrate Inhibition
class InhibitionModel(BaseModel):
    """
    Level 2: Add substrate inhibition (Andrews kinetics).

    mu = mu_max * S / (Ks + S + S^2/Ki)

    Parameters: mu_max, Ks, Ki, Y_XS, alpha, beta, Y_PS
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.params = {
            "mu_max": 0.5,
            "Ks": 0.1,
            "Ki": 100.0,    # g/L - inhibition constant
            "Y_XS": 0.5,
            "alpha": 0.1,
            "beta": 0.01,
            "Y_PS": 0.2,
            **params
        }

    def simulate(
        self,
        t_eval: np.ndarray,
        feed_func: Callable[[float], float],
        temp_func: Callable[[float], float],
        X0: float,
        P0: float,
        V0: float,
        G0: float = 0.1,
        Sf: float = 400.0,
        induction_time: Optional[float] = None,
    ) -> dict:
        mu_max = self.params["mu_max"]
        Ks = self.params["Ks"]
        Ki = self.params["Ki"]
        Y_XS = self.params["Y_XS"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        Y_PS = self.params["Y_PS"]

        def derivatives(t, y):
            X, P, S, V = y

            f = feed_func(t)
            S_conc = S / V if V > 0 else 0

            # Andrews kinetics (substrate inhibition)
            mu = mu_max * S_conc / (Ks + S_conc + S_conc**2 / Ki)

            # Product formation only after induction
            if induction_time is not None and t >= induction_time:
                pi = alpha * mu + beta
            else:
                pi = 0.0

            dXdt = mu * X - (f / V) * X if V > 0 else 0
            dPdt = pi * X - (f / V) * P if V > 0 else 0
            dSdt = -mu * X / Y_XS - pi * X / Y_PS + (f / V) * (Sf - S_conc) if V > 0 else 0
            dVdt = f

            return [dXdt, dPdt, dSdt, dVdt]

        y0 = [X0 * V0, P0 * V0, G0 * V0, V0]

        sol = solve_ivp(
            fun=derivatives,
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        V = sol.y[3, :]
        X_conc = sol.y[0, :] / V
        P_conc = sol.y[1, :] / V
        S_conc = sol.y[2, :] / V

        return {
            "t": sol.t,
            "X_conc": X_conc,
            "P_conc": P_conc,
            "V": V,
            "S_conc": S_conc,
            "success": True,
        }


# Level 3: Add Maintenance Term
class MaintenanceModel(BaseModel):
    """
    Level 3: Add maintenance energy requirement.

    Substrate uptake = growth + production + maintenance

    Parameters: mu_max, Ks, Ki, Y_XS, alpha, beta, Y_PS, m_s
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.params = {
            "mu_max": 0.5,
            "Ks": 0.1,
            "Ki": 100.0,
            "Y_XS": 0.5,
            "alpha": 0.1,
            "beta": 0.01,
            "Y_PS": 0.2,
            "m_s": 0.02,    # gS/gX/h - maintenance coefficient
            **params
        }

    def simulate(
        self,
        t_eval: np.ndarray,
        feed_func: Callable[[float], float],
        temp_func: Callable[[float], float],
        X0: float,
        P0: float,
        V0: float,
        G0: float = 0.1,
        Sf: float = 400.0,
        induction_time: Optional[float] = None,
    ) -> dict:
        mu_max = self.params["mu_max"]
        Ks = self.params["Ks"]
        Ki = self.params["Ki"]
        Y_XS = self.params["Y_XS"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        Y_PS = self.params["Y_PS"]
        m_s = self.params["m_s"]

        def derivatives(t, y):
            X, P, S, V = y

            f = feed_func(t)
            X_conc = X / V if V > 0 else 0
            S_conc = S / V if V > 0 else 0

            # Growth with substrate inhibition
            mu = mu_max * S_conc / (Ks + S_conc + S_conc**2 / Ki)

            # Product formation only after induction
            if induction_time is not None and t >= induction_time:
                pi = alpha * mu + beta
            else:
                pi = 0.0

            # Substrate consumption: growth + production + maintenance
            q_S = mu / Y_XS + pi / Y_PS + m_s

            dXdt = mu * X - (f / V) * X if V > 0 else 0
            dPdt = pi * X - (f / V) * P if V > 0 else 0
            dSdt = -q_S * X + (f / V) * (Sf - S_conc) if V > 0 else 0
            dVdt = f

            return [dXdt, dPdt, dSdt, dVdt]

        y0 = [X0 * V0, P0 * V0, G0 * V0, V0]

        sol = solve_ivp(
            fun=derivatives,
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        V = sol.y[3, :]
        X_conc = sol.y[0, :] / V
        P_conc = sol.y[1, :] / V
        S_conc = sol.y[2, :] / V

        return {
            "t": sol.t,
            "X_conc": X_conc,
            "P_conc": P_conc,
            "V": V,
            "S_conc": S_conc,
            "success": True,
        }


# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================

class DifferentialEvolutionCallback:
    """Callback for tracking DE progress."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.nfeval = 0
        self.last_report = 0

    def __call__(self, xk, convergence):
        self.nfeval += 1
        # Report every 50 evaluations
        if self.nfeval - self.last_report >= 50:
            print_progress(f"    {self.model_name}: {self.nfeval} evaluations, convergence: {convergence:.6f}")
            self.last_report = self.nfeval
        return False


def fit_model(
    model: BaseModel,
    runs: List[SimpleRunData],
    bounds: List[Tuple[float, float]],
    maxiter: int = 100,
    popsize: int = 10,
    model_name: str = "Model",
) -> ModelResult:
    """Fit model parameters to experimental data using differential evolution."""

    # Pre-compute feed functions for each run
    run_feed_funcs = []
    run_temps = []

    for run in runs:
        feed_rate = calculate_feed_rate_from_cumulative(run.time, run.feed)

        def feed_func(t, feed_rate=feed_rate, time=run.time):
            if t <= time[0]:
                return feed_rate[0]
            elif t >= time[-1]:
                return feed_rate[-1]
            else:
                return np.interp(t, time, feed_rate)

        run_feed_funcs.append(feed_func)
        run_temps.append(np.mean(run.temp))

    def objective(x):
        # Update model parameters
        for i, name in enumerate(model.param_names):
            model.params[name] = x[i]

        total_error = 0.0
        n_points = 0

        for i, run in enumerate(runs):
            try:
                # Simulate
                result = model.simulate(
                    t_eval=run.time,
                    feed_func=run_feed_funcs[i],
                    temp_func=lambda t: run_temps[i],
                    X0=run.X[0],
                    P0=run.P[0],
                    V0=run.V[0],
                    induction_time=run.induction_time,
                )

                # Calculate error
                X_pred = result["X_conc"]
                X_true = run.X

                valid_mask = np.isfinite(X_pred) & np.isfinite(X_true)
                if valid_mask.sum() > 0:
                    error = np.sum((X_pred[valid_mask] - X_true[valid_mask]) ** 2)
                    total_error += error
                    n_points += valid_mask.sum()

                # Product error if available
                if np.any(run.P > 0):
                    P_pred = result["P_conc"]
                    P_valid_mask = np.isfinite(P_pred) & np.isfinite(run.P)
                    if P_valid_mask.sum() > 0:
                        error = np.sum((P_pred[P_valid_mask] - run.P[P_valid_mask]) ** 2)
                        total_error += error
                        n_points += P_valid_mask.sum()

            except Exception as e:
                # Penalize failed simulations
                total_error += 1e6
                n_points += 1

        return total_error / max(1, n_points)

    print_progress(f"  Fitting {model_name} (maxiter={maxiter}, popsize={popsize})...")

    # Create callback
    callback = DifferentialEvolutionCallback(model_name)

    # Run optimization
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        workers=1,
        disp=False,
        callback=callback,
    )

    # Update model with best parameters
    for i, name in enumerate(model.param_names):
        model.params[name] = result.x[i]

    print_progress(f"    Final objective: {result.fun:.6g}, iterations: {result.nit}")

    return ModelResult(
        success=result.success,
        params=model.params.copy(),
        objective_value=result.fun,
        message=result.message,
        n_iterations=result.nit,
    )


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate validation metrics."""
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]

    if len(y_true_v) == 0:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "MAPE": np.nan,
            "n_points": 0,
        }

    # RMSE
    rmse = np.sqrt(np.mean((y_true_v - y_pred_v) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true_v - y_pred_v))

    # R²
    ss_res = np.sum((y_true_v - y_pred_v) ** 2)
    ss_tot = np.sum((y_true_v - np.mean(y_true_v)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # MAPE (avoid division by zero)
    mask_nonzero = y_true_v != 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true_v[mask_nonzero] - y_pred_v[mask_nonzero]) / y_true_v[mask_nonzero])) * 100
    else:
        mape = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "n_points": len(y_true_v),
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_fitting_results(
    runs: List[SimpleRunData],
    model: BaseModel,
    output_dir: Path,
    model_name: str,
):
    """Generate comprehensive fitting plots."""

    n_runs = len(runs)
    fig = plt.figure(figsize=(14, 4 * n_runs))
    gs = gridspec.GridSpec(n_runs, 3, figure=fig, hspace=0.3, wspace=0.3)

    all_metrics = []

    for i, run in enumerate(runs):
        # Simulate
        feed_rate = calculate_feed_rate_from_cumulative(run.time, run.feed)

        def feed_func(t):
            if t <= run.time[0]:
                return feed_rate[0]
            elif t >= run.time[-1]:
                return feed_rate[-1]
            else:
                return np.interp(t, run.time, feed_rate)

        temp_mean = np.mean(run.temp)

        try:
            result = model.simulate(
                t_eval=run.time,
                feed_func=feed_func,
                temp_func=lambda t: temp_mean,
                X0=run.X[0],
                P0=run.P[0],
                V0=run.V[0],
                induction_time=run.induction_time,
            )
        except Exception as e:
            print_progress(f"  Simulation failed for {run.run_id}: {e}")
            continue

        # Plot Biomass
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(run.time, run.X, 'o-', label='Data', color='steelblue', markersize=4)
        ax1.plot(result["t"], result["X_conc"], '--', label='Model', color='darkorange', linewidth=2)
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Biomass (g/L)')
        ax1.set_title(f'{run.run_id} - Biomass')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Product
        ax2 = fig.add_subplot(gs[i, 1])
        if np.any(run.P > 0):
            ax2.plot(run.time, run.P, 'o-', label='Data', color='forestgreen', markersize=4)
        ax2.plot(result["t"], result["P_conc"], '--', label='Model', color='crimson', linewidth=2)
        ax2.set_xlabel('Time (h)')
        ax2.set_ylabel('Product (g/L)')
        ax2.set_title(f'{run.run_id} - Product')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot Residuals
        ax3 = fig.add_subplot(gs[i, 2])
        residuals = run.X - result["X_conc"]
        ax3.plot(run.time, residuals, 'o-', color='gray', markersize=4)
        ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time (h)')
        ax3.set_ylabel('Residual (g/L)')
        ax3.set_title(f'{run.run_id} - Residuals')
        ax3.grid(True, alpha=0.3)

        # Calculate metrics
        metrics_X = calculate_metrics(run.X, result["X_conc"])
        metrics_P = calculate_metrics(run.P, result["P_conc"])

        all_metrics.append({
            "run_id": run.run_id,
            "metrics_X": metrics_X,
            "metrics_P": metrics_P,
        })

        # Add metrics text to biomass plot
        metrics_text = f"RMSE: {metrics_X['RMSE']:.3f} g/L\nR²: {metrics_X['R2']:.3f}"
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{model_name} - Fitting Results', fontsize=14, fontweight='bold')
    output_path = output_dir / f"{model_name.replace(' ', '_')}_fits.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print_progress(f"    Saved: {output_path}")

    return all_metrics


def plot_parameter_comparison(
    results: List[ModelResult],
    model_names: List[str],
    output_dir: Path,
):
    """Compare parameter values across models."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Objective values
    ax1 = axes[0]
    objectives = [r.objective_value for r in results]
    bars = ax1.bar(model_names, objectives, color=['steelblue', 'forestgreen', 'darkorange', 'crimson'])
    ax1.set_ylabel('Objective Value (MSE)')
    ax1.set_title('Fitting Quality Comparison')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, obj in zip(bars, objectives):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{obj:.2e}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Parameter values
    ax2 = axes[1]
    # Get common parameters across all models
    all_params = set()
    for r in results:
        all_params.update(r.params.keys())

    param_names = sorted(all_params)
    x = np.arange(len(param_names))
    width = 0.2

    for i, (result, name, color) in enumerate(zip(results, model_names,
                                                   ['steelblue', 'forestgreen', 'darkorange', 'crimson'])):
        values = [result.params.get(k, np.nan) for k in param_names]
        # Only plot if parameter exists in this model
        mask = [not np.isnan(v) for v in values]
        x_masked = [x[j] for j in range(len(x)) if mask[j]]
        values_masked = [values[j] for j in range(len(x)) if mask[j]]

        if x_masked:
            ax2.bar([xi + i * width for xi in x_masked], values_masked, width, label=name, color=color)

    ax2.set_xlabel('Parameters')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Fitted Parameters Comparison')
    ax2.set_xticks([xi + width * 1.5 for xi in x])
    ax2.set_xticklabels(param_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "parameter_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print_progress(f"    Saved: {output_path}")


# =============================================================================
# MAIN TROUBLESHOOTING WORKFLOW
# =============================================================================

def run_troubleshooting(
    data_path: str,
    output_dir: str,
    run_ids: List[str] = ["ferm_01", "ferm_02"],
    maxiter: int = 100,  # Reduced for faster execution
):
    """Run the complete troubleshooting workflow with increasing model complexity."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print_progress("=" * 80)
    print_progress("INCREMENTAL MODEL FITTING TROUBLESHOOTING (FAST VERSION)")
    print_progress("=" * 80)
    print_progress(f"\nData path: {data_path}")
    print_progress(f"Output directory: {output_path}")
    print_progress(f"Test runs: {run_ids}")
    print_progress(f"Max iterations: {maxiter}")

    # Load data
    print_progress("\n" + "-" * 80)
    print_progress("STEP 1: LOADING DATA")
    print_progress("-" * 80)

    runs = load_two_runs(data_path, run_ids)

    if len(runs) == 0:
        print_progress("ERROR: No valid runs found!")
        return []

    print_progress(f"Loaded {len(runs)} runs:")
    for run in runs:
        print_progress(f"  {run.run_id}: {len(run.time)} time points, X range: [{run.X.min():.1f}, {run.X.max():.1f}] g/L")

    # Define models and their bounds
    models_config = [
        {
            "name": "Level_0_Monod",
            "class": MonodModel,
            "initial": {},
            "bounds": [(0.1, 1.0), (0.001, 1.0), (0.1, 0.8)],
        },
        {
            "name": "Level_1_Monod_Product",
            "class": MonodProductModel,
            "initial": {},
            "bounds": [(0.1, 1.0), (0.001, 1.0), (0.1, 0.8), (0.001, 0.5), (0.001, 0.1), (0.05, 0.5)],
        },
        {
            "name": "Level_2_Inhibition",
            "class": InhibitionModel,
            "initial": {},
            "bounds": [(0.1, 1.0), (0.001, 1.0), (10.0, 500.0), (0.1, 0.8), (0.001, 0.5), (0.001, 0.1), (0.05, 0.5)],
        },
        {
            "name": "Level_3_Maintenance",
            "class": MaintenanceModel,
            "initial": {},
            "bounds": [(0.1, 1.0), (0.001, 1.0), (10.0, 500.0), (0.1, 0.8), (0.001, 0.5), (0.001, 0.1), (0.05, 0.5), (0.001, 0.1)],
        },
    ]

    results = []
    all_metrics = []

    # Fit each model
    for i, config in enumerate(models_config):
        print_progress("\n" + "-" * 80)
        print_progress(f"STEP {2 + i}: FITTING {config['name'].upper()}")
        print_progress("-" * 80)

        model = config["class"](config["initial"])

        print_progress(f"  Parameters: {', '.join(model.param_names)}")

        result = fit_model(
            model=model,
            runs=runs,
            bounds=config["bounds"],
            maxiter=maxiter,
            popsize=10,
            model_name=config["name"],
        )

        results.append(result)

        print_progress(f"\n  Results:")
        print_progress(f"    Success: {result.success}")
        print_progress(f"    Objective value: {result.objective_value:.6g}")
        print_progress(f"    Iterations: {result.n_iterations}")
        print_progress(f"    Fitted parameters:")
        for name, value in result.params.items():
            print_progress(f"      {name}: {value:.6g}")

        # Generate plots
        print_progress("\n  Generating plots...")
        metrics = plot_fitting_results(
            runs=runs,
            model=model,
            output_dir=output_path,
            model_name=config["name"],
        )
        all_metrics.append({
            "model": config["name"],
            "result": result,
            "metrics": metrics,
        })

    # Generate comparison plot
    print_progress("\n" + "-" * 80)
    print_progress("GENERATING COMPARISON PLOTS")
    print_progress("-" * 80)

    model_names = [c["name"] for c in models_config]
    plot_parameter_comparison(results, model_names, output_path)

    # Generate summary report
    print_progress("\n" + "-" * 80)
    print_progress("GENERATING SUMMARY REPORT")
    print_progress("-" * 80)

    report_path = output_path / "troubleshooting_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("INCREMENTAL MODEL FITTING TROUBLESHOOTING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Data path: {data_path}\n")
        f.write(f"Test runs: {run_ids}\n")
        f.write(f"Max iterations: {maxiter}\n\n")

        for item in all_metrics:
            f.write("-" * 80 + "\n")
            f.write(f"MODEL: {item['model']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Success: {item['result'].success}\n")
            f.write(f"Objective value: {item['result'].objective_value:.6g}\n")
            f.write(f"Iterations: {item['result'].n_iterations}\n")
            f.write(f"\nFitted parameters:\n")
            for name, value in item['result'].params.items():
                f.write(f"  {name}: {value:.6g}\n")

            f.write(f"\nValidation metrics:\n")
            for m in item['metrics']:
                f.write(f"  Run: {m['run_id']}\n")
                f.write(f"    Biomass RMSE: {m['metrics_X']['RMSE']:.4f} g/L\n")
                f.write(f"    Biomass R²: {m['metrics_X']['R2']:.4f}\n")
                f.write(f"    Biomass MAPE: {m['metrics_X']['MAPE']:.2f} %%\n")
                if m['metrics_P']['n_points'] > 0:
                    f.write(f"    Product RMSE: {m['metrics_P']['RMSE']:.4f} g/L\n")
                    f.write(f"    Product R²: {m['metrics_P']['R2']:.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write("\nObjective values (lower is better):\n")
        for item in all_metrics:
            f.write(f"  {item['model']:30s}: {item['result'].objective_value:.6g}\n")

        best_model = min(all_metrics, key=lambda x: x['result'].objective_value)
        f.write(f"\nBest fitting model: {best_model['model']}\n")
        f.write(f"Objective value: {best_model['result'].objective_value:.6g}\n")

    print_progress(f"    Saved: {report_path}")

    print_progress("\n" + "=" * 80)
    print_progress("TROUBLESHOOTING COMPLETE")
    print_progress("=" * 80)
    print_progress(f"\nAll results saved to: {output_path}")
    print_progress("\nSummary of results:")
    for item in all_metrics:
        obj_val = item['result'].objective_value
        status = "✓" if item['result'].success else "✗"
        print_progress(f"  {status} {item['model']:30s}: MSE = {obj_val:.6g}")

    print_progress("\nRecommendations:")
    best_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['result'].objective_value)

    if best_idx == 0:
        print_progress("  → Simple Monod model fits best. Do not add complexity.")
    elif best_idx < len(all_metrics) - 1:
        print_progress(f"  → {all_metrics[best_idx]['model']} is the sweet spot.")
        print_progress("  → More complex models do not improve the fit.")
    else:
        print_progress(f"  → Most complex model ({all_metrics[best_idx]['model']}) fits best.")
        print_progress("  → Consider adding even more terms if justified.")

    return all_metrics


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Configuration
    DATA_PATH = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\data\traintest_compiled.xlsx"
    OUTPUT_DIR = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\optfed\troubleshooting_results"

    # You can change which runs to test
    # Options: "ferm_01" through "ferm_07" (training), "ferm_10" through "ferm_12" (test), "ferm_opt"
    TEST_RUNS = ["ferm_01", "ferm_02"]

    # Run troubleshooting
    all_metrics = run_troubleshooting(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        run_ids=TEST_RUNS,
        maxiter=100,  # Reduced iterations for faster execution
    )

    print_progress("\nDone!")
