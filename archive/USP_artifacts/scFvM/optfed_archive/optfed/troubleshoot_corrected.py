"""
Corrected Incremental Model Fitting Troubleshooting Script

Key fixes:
1. Uses HIGH-RESOLUTION feed data (not just biomass time points)
2. Assumes glucose-limited (S ≈ 0 at all times)
3. Properly handles two-phase feed profile
4. Product formation only after induction

Usage:
    python troubleshoot_corrected.py
"""

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
    print(msg, flush=True)

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class FermentationRun:
    """Complete fermentation run data."""
    run_id: str
    time: np.ndarray           # All time points (high resolution)
    X: np.ndarray              # Biomass concentration (g/L) - NaN for non-measurement points
    P: np.ndarray              # Product concentration (g/L)
    V: np.ndarray              # Volume (L)
    feed_rate: np.ndarray      # Feed rate (L/h) - calculated from cumulative feed
    temp: np.ndarray           # Temperature (C)
    induction_time: Optional[float] = None

    # Indices where biomass is measured
    measurement_indices: np.ndarray = None

    def __post_init__(self):
        # Find indices where biomass was measured
        self.measurement_indices = np.where(~np.isnan(self.X))[0]


def load_fermentation_runs(
    data_path: str,
    run_ids: List[str] = ["ferm_01", "ferm_02"]
) -> List[FermentationRun]:
    """Load fermentation runs with HIGH-RESOLUTION feed data."""
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

        # Use ALL time points (high resolution) - don't filter to biomass measurements only
        # This ensures we capture the detailed feed profile

        # Calculate feed rate from cumulative feed
        feed_cumulative = run_df["feed"].values / 1000.0  # mL to L
        time = run_df["time"].values

        # Calculate feed rate (gradient of cumulative feed)
        feed_rate = np.gradient(feed_cumulative, time)
        feed_rate = np.maximum(feed_rate, 0)  # Clip negative values

        runs.append(FermentationRun(
            run_id=run_id,
            time=time,
            X=run_df["X"].values,
            P=run_df.get("P", pd.Series([0]*len(run_df))).values,
            V=run_df["reactor_volume"].values / 1000.0,  # mL to L
            feed_rate=feed_rate,
            temp=run_df.get("temp", pd.Series([37.0]*len(run_df))).values,
            induction_time=induction_time,
        ))

        n_measured = np.sum(~np.isnan(run_df["X"].values))
        if induction_time is not None:
            print_progress(f"  {run_id}: {len(time)} time points, {n_measured} biomass measurements, induction at {induction_time:.2f} h")
        else:
            print_progress(f"  {run_id}: {len(time)} time points, {n_measured} biomass measurements, no induction")

    print_progress(f"  Loaded {len(runs)} runs successfully")
    return runs


# =============================================================================
# MODEL: GLUCOSE-LIMITED FED-BATCH
# =============================================================================

class GlucoseLimitedModel:
    """
    Glucose-limited fed-batch model.

    Key assumption: S ≈ 0 at all times (substrate is immediately consumed)

    State variables:
    - X: Total biomass (g)
    - P: Total product (g)
    - V: Volume (L)

    Parameters:
    - mu_max: Maximum specific growth rate (1/h)
    - Y_XS: Biomass yield on substrate (gX/gS)
    - alpha: Growth-associated product formation (gP/gX)
    - beta: Non-growth-associated product formation (gP/gX/h)
    - Y_PS: Product yield on substrate (gP/gS)
    """

    def __init__(self, params: dict):
        self.params = {
            "mu_max": 0.3,     # 1/h - maximum growth rate
            "Y_XS": 0.5,       # g/g - biomass yield
            "alpha": 0.05,     # gP/gX - growth-associated product
            "beta": 0.005,     # gP/gX/h - non-growth-associated product
            "Y_PS": 0.2,       # g/g - product yield
            **params
        }
        self.param_names = list(self.params.keys())

    def simulate(
        self,
        t_eval: np.ndarray,
        feed_rate_func: Callable[[float], float],
        X0: float,
        P0: float,
        V0: float,
        Sf: float = 400.0,
        induction_time: Optional[float] = None,
    ) -> dict:
        """
        Simulate glucose-limited fed-batch fermentation.

        Key: In glucose-limited conditions, mu is controlled by feed rate:
        mu = (f/V) * Sf / (1/Y_XS + alpha/Y_PS)

        This ensures that all incoming substrate is immediately consumed.
        """
        mu_max = self.params["mu_max"]
        Y_XS = self.params["Y_XS"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        Y_PS = self.params["Y_PS"]

        # Substrate required per unit biomass growth
        # 1 g biomass requires 1/Y_XS g substrate
        sigma_X = 1.0 / Y_XS  # gS/gX

        # Substrate required per unit product
        sigma_P = 1.0 / Y_PS  # gS/gP

        def derivatives(t, y):
            X, P, V = y  # Total biomass (g), total product (g), volume (L)

            f = feed_rate_func(t)  # Feed rate (L/h)

            # In glucose-limited fed-batch:
            # Growth rate is controlled by substrate supply
            # mu = (f/V) * Sf / (sigma_X + alpha*sigma_P)
            # But we also cap it at mu_max

            # Maximum possible growth rate given feed
            if V > 0:
                mu_feed = (f / V) * Sf / (sigma_X + alpha * sigma_P)
                mu = min(mu_feed, mu_max)
            else:
                mu = 0

            # Product formation (only after induction)
            if induction_time is not None and t >= induction_time:
                pi = alpha * mu + beta  # 1/h specific production rate
            else:
                pi = 0.0

            # Differential equations
            dXdt = mu * X - (f / V) * X if V > 0 else 0  # Total biomass change
            dPdt = pi * X - (f / V) * P if V > 0 else 0  # Total product change
            dVdt = f  # Volume change

            return [dXdt, dPdt, dVdt]

        y0 = [X0, P0, V0]

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
        P_conc = sol.y[1, :] / V

        return {
            "t": sol.t,
            "X": sol.y[0, :],      # Total biomass (g)
            "P": sol.y[1, :],      # Total product (g)
            "V": V,                # Volume (L)
            "X_conc": X_conc,      # Biomass concentration (g/L)
            "P_conc": P_conc,      # Product concentration (g/L)
            "success": True,
        }


# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================

def fit_model(
    model: GlucoseLimitedModel,
    runs: List[FermentationRun],
    bounds: List[Tuple[float, float]],
    maxiter: int = 100,
) -> dict:
    """Fit model parameters using differential evolution."""

    # Create feed rate functions for each run
    run_feed_funcs = []

    for run in runs:
        def feed_func(t, time=run.time, rate=run.feed_rate):
            if t <= time[0]:
                return rate[0]
            elif t >= time[-1]:
                return rate[-1]
            else:
                return np.interp(t, time, rate)

        run_feed_funcs.append(feed_func)

    def objective(x):
        # Update model parameters
        for i, name in enumerate(model.param_names):
            model.params[name] = x[i]

        total_error = 0.0
        n_points = 0

        for i, run in enumerate(runs):
            try:
                # Get initial conditions from first biomass measurement
                first_idx = run.measurement_indices[0]

                X0 = run.X[first_idx] * run.V[first_idx]  # Total biomass (g)
                P0 = run.P[first_idx] * run.V[first_idx]  # Total product (g)
                V0 = run.V[first_idx]  # Volume (L)

                # Simulate at ALL time points (high resolution)
                result = model.simulate(
                    t_eval=run.time,
                    feed_rate_func=run_feed_funcs[i],
                    X0=X0,
                    P0=P0,
                    V0=V0,
                    Sf=400.0,
                    induction_time=run.induction_time,
                )

                # Calculate error ONLY at biomass measurement points
                for idx in run.measurement_indices:
                    X_pred = result["X_conc"][idx]
                    X_true = run.X[idx]

                    if np.isfinite(X_pred) and np.isfinite(X_true):
                        error = (X_pred - X_true) ** 2
                        total_error += error
                        n_points += 1

                # Product error (if available and after induction)
                if run.induction_time is not None:
                    for idx in run.measurement_indices:
                        if run.time[idx] >= run.induction_time:
                            P_pred = result["P_conc"][idx]
                            P_true = run.P[idx]

                            if np.isfinite(P_pred) and np.isfinite(P_true) and P_true > 0:
                                error = (P_pred - P_true) ** 2
                                total_error += error
                                n_points += 1

            except Exception as e:
                total_error += 1e6
                n_points += 1

        return total_error / max(1, n_points)

    print_progress(f"  Fitting with {len(model.param_names)} parameters (maxiter={maxiter})...")

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=10,
        tol=1e-6,
        workers=1,
        disp=False,
    )

    # Update model with best parameters
    for i, name in enumerate(model.param_names):
        model.params[name] = result.x[i]

    print_progress(f"    Final objective: {result.fun:.6g}, iterations: {result.nit}")

    return {
        "success": result.success,
        "params": model.params.copy(),
        "objective_value": result.fun,
        "n_iterations": result.nit,
    }


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate validation metrics."""
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]

    if len(y_true_v) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "MAPE": np.nan}

    rmse = np.sqrt(np.mean((y_true_v - y_pred_v) ** 2))
    mae = np.mean(np.abs(y_true_v - y_pred_v))

    ss_res = np.sum((y_true_v - y_pred_v) ** 2)
    ss_tot = np.sum((y_true_v - np.mean(y_true_v)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    mask_nonzero = y_true_v != 0
    mape = np.mean(np.abs((y_true_v[mask_nonzero] - y_pred_v[mask_nonzero]) / y_true_v[mask_nonzero])) * 100 if mask_nonzero.sum() > 0 else np.nan

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# =============================================================================
# PLOTTING
# =============================================================================

def plot_fitting_results(
    runs: List[FermentationRun],
    model: GlucoseLimitedModel,
    output_dir: Path,
    model_name: str,
):
    """Generate fitting plots."""

    n_runs = len(runs)
    fig = plt.figure(figsize=(16, 5 * n_runs))
    gs = gridspec.GridSpec(n_runs, 4, figure=fig, hspace=0.3, wspace=0.3)

    for i, run in enumerate(runs):
        # Create feed function
        def feed_func(t):
            if t <= run.time[0]:
                return run.feed_rate[0]
            elif t >= run.time[-1]:
                return run.feed_rate[-1]
            else:
                return np.interp(t, run.time, run.feed_rate)

        # Get initial conditions
        first_idx = run.measurement_indices[0]
        X0 = run.X[first_idx] * run.V[first_idx]
        P0 = run.P[first_idx] * run.V[first_idx]
        V0 = run.V[first_idx]

        # Simulate
        result = model.simulate(
            t_eval=run.time,
            feed_rate_func=feed_func,
            X0=X0,
            P0=P0,
            V0=V0,
            Sf=400.0,
            induction_time=run.induction_time,
        )

        # Plot 1: Biomass
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(run.time, run.X, 'o', label='Data', color='steelblue', markersize=6)
        ax1.plot(result["t"], result["X_conc"], '-', label='Model', color='darkorange', linewidth=2)
        if run.induction_time is not None:
            ax1.axvline(run.induction_time, color='red', linestyle='--', alpha=0.5, label='Induction')
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Biomass (g/L)')
        ax1.set_title(f'{run.run_id} - Biomass')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Product
        ax2 = fig.add_subplot(gs[i, 1])
        if np.any(run.P > 0):
            ax2.plot(run.time, run.P, 'o', label='Data', color='forestgreen', markersize=6)
        ax2.plot(result["t"], result["P_conc"], '-', label='Model', color='crimson', linewidth=2)
        if run.induction_time is not None:
            ax2.axvline(run.induction_time, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (h)')
        ax2.set_ylabel('Product (g/L)')
        ax2.set_title(f'{run.run_id} - Product')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Volume
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(run.time, run.V, 'o-', label='Data', color='purple', markersize=4)
        ax3.plot(result["t"], result["V"], '--', label='Model', color='darkorchid', linewidth=2)
        ax3.set_xlabel('Time (h)')
        ax3.set_ylabel('Volume (L)')
        ax3.set_title(f'{run.run_id} - Volume')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Feed Rate
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.plot(run.time, run.feed_rate, '-', color='brown', linewidth=1.5)
        if run.induction_time is not None:
            ax4.axvline(run.induction_time, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time (h)')
        ax4.set_ylabel('Feed Rate (L/h)')
        ax4.set_title(f'{run.run_id} - Feed Rate')
        ax4.grid(True, alpha=0.3)

        # Calculate metrics
        X_pred_meas = result["X_conc"][run.measurement_indices]
        X_true_meas = run.X[run.measurement_indices]
        metrics = calculate_metrics(X_true_meas, X_pred_meas)

        ax1.text(0.05, 0.95, f"RMSE: {metrics['RMSE']:.3f} g/L\nR²: {metrics['R2']:.3f}",
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{model_name} - Corrected Model', fontsize=14, fontweight='bold')
    output_path = output_dir / f"{model_name.replace(' ', '_')}_corrected.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print_progress(f"    Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_troubleshooting(
    data_path: str,
    output_dir: str,
    run_ids: List[str] = ["ferm_01", "ferm_02"],
):
    """Run the troubleshooting workflow."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print_progress("=" * 80)
    print_progress("CORRECTED MODEL FITTING - GLUCOSE-LIMITED FED-BATCH")
    print_progress("=" * 80)
    print_progress(f"\nData path: {data_path}")
    print_progress(f"Output directory: {output_path}")
    print_progress(f"Test runs: {run_ids}")

    # Load data
    print_progress("\n" + "-" * 80)
    print_progress("STEP 1: LOADING DATA")
    print_progress("-" * 80)

    runs = load_fermentation_runs(data_path, run_ids)

    if len(runs) == 0:
        print_progress("ERROR: No valid runs found!")
        return

    # Create model
    print_progress("\n" + "-" * 80)
    print_progress("STEP 2: FITTING MODEL")
    print_progress("-" * 80)

    model = GlucoseLimitedModel({})

    # Parameter bounds
    bounds = [
        (0.1, 0.8),    # mu_max (1/h)
        (0.2, 0.8),    # Y_XS (g/g)
        (0.001, 0.3),  # alpha (gP/gX)
        (0.001, 0.05), # beta (gP/gX/h)
        (0.1, 0.5),    # Y_PS (g/g)
    ]

    result = fit_model(model, runs, bounds, maxiter=200)

    print_progress(f"\n  Results:")
    print_progress(f"    Success: {result['success']}")
    print_progress(f"    Objective value: {result['objective_value']:.6g}")
    print_progress(f"    Fitted parameters:")
    for name, value in result['params'].items():
        print_progress(f"      {name}: {value:.6g}")

    # Generate plots
    print_progress("\n  Generating plots...")
    plot_fitting_results(runs, model, output_path, "Glucose_Limited_Model")

    # Generate report
    print_progress("\n" + "-" * 80)
    print_progress("STEP 3: GENERATING REPORT")
    print_progress("-" * 80)

    report_path = output_path / "corrected_model_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CORRECTED MODEL FITTING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Test runs: {run_ids}\n\n")

        f.write("Model: Glucose-Limited Fed-Batch\n")
        f.write("Key assumption: S ≈ 0 (glucose is immediately consumed)\n\n")

        f.write("Fitted parameters:\n")
        for name, value in result['params'].items():
            f.write(f"  {name}: {value:.6g}\n")

        f.write(f"\nObjective value: {result['objective_value']:.6g}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n")
        f.write("1. Feed rate increases 4x after induction (~0.011 -> ~0.049 L/h)\n")
        f.write("2. This causes biomass growth rate to increase after induction\n")
        f.write("3. Glucose is assumed to be ~0 (limiting substrate)\n")
        f.write("4. Model uses high-resolution feed data for accurate simulation\n")

    print_progress(f"    Saved: {report_path}")

    print_progress("\n" + "=" * 80)
    print_progress("TROUBLESHOOTING COMPLETE")
    print_progress("=" * 80)
    print_progress(f"\nResults saved to: {output_path}")

    return result


if __name__ == "__main__":
    DATA_PATH = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\data\traintest_compiled.xlsx"
    OUTPUT_DIR = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\optfed\corrected_results"
    TEST_RUNS = ["ferm_01", "ferm_02"]

    run_troubleshooting(DATA_PATH, OUTPUT_DIR, TEST_RUNS)
    print_progress("\nDone!")
