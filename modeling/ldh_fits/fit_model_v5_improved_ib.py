"""
Hybrid Mechanistic Model v5: Improved IB Model

Key changes from v4:
- IB model now includes baseline fraction: phi_IB = phi_min + (1-phi_min) * sigmoid
- This captures the ~13% IB observed at 30°C (not 0% as v4 predicted)

Data to match:
  T=25°C: 0% IB
  T=30°C: ~13% IB  <-- Key improvement
  T=35°C: ~100% IB

Author: Modeling Playground
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

# ============================================================
# CONSTANTS AND DATA LOADING
# ============================================================

DATA_PATH = Path(__file__).parent / "data" / "LDH_consolidated_results.xlsx"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

DOE_CONDITIONS = {
    "ferm_01": {"qs_setpoint": 0.2, "T_induced": 35.0},
    "ferm_02": {"qs_setpoint": 0.2, "T_induced": 25.0},
    "ferm_03": {"qs_setpoint": 0.6, "T_induced": 25.0},
    "ferm_04": {"qs_setpoint": 0.6, "T_induced": 35.0},
    "ferm_05": {"qs_setpoint": 0.4, "T_induced": 30.0},
    "ferm_06": {"qs_setpoint": 0.4, "T_induced": 30.0},
    "ferm_07": {"qs_setpoint": 0.4, "T_induced": 30.0},
}

C_FEED = 400.0
T_FEDBATCH = 37.0
SAMPLE_VOLUME = 10.0


def load_run_data(run_name: str) -> pd.DataFrame:
    """Load data for a single run."""
    df_all = pd.read_excel(DATA_PATH, sheet_name='All_Time_Points')
    df = df_all[df_all['run_name'] == run_name].copy()
    if df.empty:
        raise ValueError(f"No data found for run: {run_name}")
    return df


def prepare_full_run_data(run_name: str) -> Dict:
    """Prepare FULL run data (fedbatch + induced phases)."""
    df = load_run_data(run_name)

    has_biomass = df['Biomass [g/L]'].notna()
    samples = df[has_biomass].copy()

    times_all = df['Process time [h]'].values

    volume_mL = pd.Series(df['Volume [mL]'].values).ffill().bfill().values
    volume_corrected_mL = pd.Series(df['Volume corrected [mL]'].values).ffill().bfill().values if 'Volume corrected [mL]' in df.columns else volume_mL
    volume_L = volume_corrected_mL / 1000.0

    feed_cumulative_mL = df['Feed [mL]'].fillna(0).values
    dt = np.diff(times_all)
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 1.0)
    dt = np.maximum(dt, 0.01)

    feed_rate_mL_h = np.diff(feed_cumulative_mL, prepend=feed_cumulative_mL[0]) / dt
    feed_rate_mL_h = np.maximum(feed_rate_mL_h, 0)
    feed_rate_L_h = feed_rate_mL_h / 1000.0

    T_induced = DOE_CONDITIONS[run_name]["T_induced"]
    is_induced = df['Is induced [-]'].fillna(0).values.astype(float)
    T_values = np.where(is_induced > 0.5, T_induced, T_FEDBATCH)

    data = {
        "run_name": run_name,
        "conditions": DOE_CONDITIONS[run_name],
        "sample_times": samples['Process time [h]'].values,
        "X_obs": samples['Biomass [g/L]'].values,
        "P_sol_obs": samples['Protein soluble [g/L]'].values if 'Protein soluble [g/L]' in samples.columns else np.full(len(samples), np.nan),
        "P_IB_obs": samples['Protein IB [g/L]'].values if 'Protein IB [g/L]' in samples.columns else np.full(len(samples), np.nan),
        "times_all": times_all,
        "Volume_L": volume_L,
        "Feed_rate_L_h": feed_rate_L_h,
        "T_values": T_values,
        "is_induced": is_induced,
        "t0": float(times_all[0]),
        "t1": float(times_all[-1]),
    }

    return data


# ============================================================
# HYBRID ODE MODEL (v5: Improved IB Model)
# ============================================================

PARAM_NAMES = [
    "Y_XS",     # Yield on glucose [gX/gS]
    "m_s",      # Maintenance coefficient [gS/(gX*h)]
    "alpha",    # Growth-associated product formation [gP/gX]
    "beta",     # Non-growth-associated product formation [gP/(gX*h)]
    "phi_min",  # Baseline IB fraction [0-1]  <-- NEW PARAMETER
    "T_50",     # Temperature for 50% of remaining IB partition [°C]
    "k_T",      # Steepness of IB partition sigmoid [1/°C]
    "k_base",   # Base addition proportional to X [L/(gX*h)]
]

PARAM_BOUNDS = [
    (0.3, 0.7),      # Y_XS
    (0.0, 0.05),     # m_s
    (0.01, 0.3),     # alpha
    (0.0, 0.1),      # beta
    (0.0, 0.3),      # phi_min: baseline IB fraction 0-30% <-- NEW
    (25.0, 35.0),    # T_50
    (0.1, 2.0),      # k_T
    (0.0, 0.001),    # k_base
]

DEFAULT_PARAMS = np.array([0.5, 0.01, 0.05, 0.02, 0.10, 30.0, 0.5, 0.0001])


def create_interpolators(data: Dict) -> Tuple:
    """Create interpolation functions for time-varying inputs."""
    times = data["times_all"]

    feed_interp = interp1d(times, data["Feed_rate_L_h"], kind='linear',
                           bounds_error=False, fill_value=(data["Feed_rate_L_h"][0], data["Feed_rate_L_h"][-1]))
    T_interp = interp1d(times, data["T_values"], kind='linear',
                        bounds_error=False, fill_value=(data["T_values"][0], data["T_values"][-1]))
    is_induced_interp = interp1d(times, data["is_induced"], kind='linear',
                                  bounds_error=False, fill_value=(data["is_induced"][0], data["is_induced"][-1]))
    V_data_interp = interp1d(times, data["Volume_L"], kind='linear',
                              bounds_error=False, fill_value=(data["Volume_L"][0], data["Volume_L"][-1]))

    return feed_interp, T_interp, is_induced_interp, V_data_interp


def ode_system(t, y, params, feed_interp, T_interp, is_induced_interp, V_data_interp):
    """
    Hybrid ODE system for fed-batch fermentation (v5: improved IB model).

    States: y = [X, P_sol, P_IB, V]

    KEY CHANGE: IB partitioning now has baseline fraction
    phi_IB(T) = phi_min + (1 - phi_min) * sigmoid(k_T * (T - T_50))

    This ensures some IB forms even at low temperatures.
    """
    X, P_sol, P_IB, V = y

    # Unpack parameters
    Y_XS, m_s, alpha, beta, phi_min, T_50, k_T, k_base = params

    # Get interpolated inputs
    F = float(feed_interp(t))
    T = float(T_interp(t))
    I = float(is_induced_interp(t))

    # Safety
    X_safe = max(X, 0.1)
    V_safe = max(V, 0.1)

    # Derived qs
    qs = (F * C_FEED) / (X_safe * V_safe)
    qs = min(qs, 2.0)

    # Growth rate
    mu = max(Y_XS * qs - m_s, 0.0)

    # Volume dynamics
    F_base = k_base * X_safe * V_safe
    dV_dt = F + F_base
    D = (F + F_base) / V_safe

    # Biomass
    dX_dt = mu * X - D * X

    # Product formation (Luedeking-Piret)
    T_opt = 30.0
    T_width = 10.0
    f_T = np.exp(-0.5 * ((T - T_opt) / T_width)**2)
    q_p = (alpha * mu + beta) * I * f_T

    # === IMPROVED IB PARTITIONING ===
    # phi_IB = phi_min + (1 - phi_min) * sigmoid(k_T * (T - T_50))
    sigmoid = 1.0 / (1.0 + np.exp(-k_T * (T - T_50)))
    phi_IB = phi_min + (1.0 - phi_min) * sigmoid

    # Protein ODEs
    dP_sol_dt = q_p * (1 - phi_IB) * X - D * P_sol
    dP_IB_dt = q_p * phi_IB * X - D * P_IB

    return [dX_dt, dP_sol_dt, dP_IB_dt, dV_dt]


def solve_ode(params: np.ndarray, data: Dict,
              dense_output: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the ODE system."""
    feed_interp, T_interp, is_induced_interp, V_data_interp = create_interpolators(data)

    X0 = data["X_obs"][0] if not np.isnan(data["X_obs"][0]) else 5.0
    P_sol_0 = data["P_sol_obs"][0] if not np.isnan(data["P_sol_obs"][0]) else 0.0
    P_IB_0 = data["P_IB_obs"][0] if not np.isnan(data["P_IB_obs"][0]) else 0.0

    t0_sample = data["sample_times"][0]
    idx0 = np.argmin(np.abs(data["times_all"] - t0_sample))
    V0 = data["Volume_L"][idx0]

    y0 = [X0, P_sol_0, P_IB_0, V0]

    t_span = (data["t0"], data["t1"])

    if dense_output:
        t_eval = np.linspace(data["t0"], data["t1"], 200)
    else:
        t_eval = data["sample_times"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(
            lambda t, y: ode_system(t, y, params, feed_interp, T_interp,
                                    is_induced_interp, V_data_interp),
            t_span, y0, t_eval=t_eval, method='RK45',
            max_step=0.1, rtol=1e-5, atol=1e-7
        )

    if not sol.success:
        return t_eval, np.full((len(t_eval), 4), np.nan)

    return sol.t, sol.y.T


# ============================================================
# LOSS FUNCTION
# ============================================================

def compute_loss_single(params: np.ndarray, data: Dict) -> float:
    """Compute loss for a single run."""
    try:
        times, solution = solve_ode(params, data, dense_output=False)
    except Exception as e:
        return 1e10

    if np.any(np.isnan(solution)):
        return 1e10

    X_pred = solution[:, 0]
    P_sol_pred = solution[:, 1]
    P_IB_pred = solution[:, 2]

    X_obs = data["X_obs"]
    P_sol_obs = data["P_sol_obs"]
    P_IB_obs = data["P_IB_obs"]

    def masked_mse(pred, obs, scale=1.0):
        mask = ~np.isnan(obs)
        if not np.any(mask):
            return 0.0
        return np.mean((pred[mask] - obs[mask])**2) / scale

    loss_X = masked_mse(X_pred, X_obs, scale=100.0)
    loss_P_sol = masked_mse(P_sol_pred, P_sol_obs, scale=1.0)
    loss_P_IB = masked_mse(P_IB_pred, P_IB_obs, scale=1.0)

    return loss_X + loss_P_sol + loss_P_IB


def compute_loss_multi(params: np.ndarray, data_list: List[Dict]) -> float:
    """Compute total loss across multiple runs."""
    total_loss = 0.0
    for data in data_list:
        total_loss += compute_loss_single(params, data)
    return total_loss / len(data_list)


# ============================================================
# OPTIMIZATION
# ============================================================

def fit_multi_run(data_list: List[Dict]) -> Tuple[np.ndarray, float]:
    """Fit shared parameters across multiple runs."""
    run_names = [d["run_name"] for d in data_list]
    print(f"Multi-run fitting (hybrid v5 - improved IB) on: {run_names}")
    print(f"Initial loss: {compute_loss_multi(DEFAULT_PARAMS, data_list):.4f}")

    result = differential_evolution(
        compute_loss_multi,
        bounds=PARAM_BOUNDS,
        args=(data_list,),
        maxiter=100,
        seed=42,
        disp=True,
        polish=True,
        workers=1
    )

    print(f"\nFinal loss: {result.fun:.4f}")
    return result.x, result.fun


# ============================================================
# VISUALIZATION
# ============================================================

def plot_multi_run(params: np.ndarray, data_list: List[Dict], save_path: Optional[Path] = None):
    """Plot model fits for all runs."""
    n_runs = len(data_list)
    fig, axes = plt.subplots(n_runs, 3, figsize=(15, 4*n_runs))

    if n_runs == 1:
        axes = axes.reshape(1, -1)

    for i, data in enumerate(data_list):
        times_dense, solution = solve_ode(params, data, dense_output=True)

        cond = data["conditions"]
        run_label = f"{data['run_name']}: qs_set={cond['qs_setpoint']}, T={cond['T_induced']}C"

        # Biomass
        ax = axes[i, 0]
        ax.plot(times_dense, solution[:, 0], 'b-', linewidth=2, label='Model')
        ax.scatter(data["sample_times"], data["X_obs"], c='blue', s=60,
                   zorder=5, edgecolors='black', label='Observed')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Biomass [g/L]')
        ax.set_title(f'{run_label}\nBiomass (X)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Soluble Protein
        ax = axes[i, 1]
        ax.plot(times_dense, solution[:, 1], 'g-', linewidth=2, label='Model')
        valid = ~np.isnan(data["P_sol_obs"])
        ax.scatter(data["sample_times"][valid], data["P_sol_obs"][valid],
                   c='green', s=60, zorder=5, edgecolors='black', label='Observed')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Protein [g/L]')
        ax.set_title('Soluble Protein (P_sol)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # IB
        ax = axes[i, 2]
        ax.plot(times_dense, solution[:, 2], 'r-', linewidth=2, label='Model')
        valid = ~np.isnan(data["P_IB_obs"])
        ax.scatter(data["sample_times"][valid], data["P_IB_obs"][valid],
                   c='red', s=60, zorder=5, edgecolors='black', label='Observed')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Protein [g/L]')
        ax.set_title('Inclusion Bodies (P_IB)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


def plot_ib_temperature_response(params: np.ndarray, save_path: Optional[Path] = None):
    """Plot the IB temperature response curve."""
    Y_XS, m_s, alpha, beta, phi_min, T_50, k_T, k_base = params

    T_range = np.linspace(20, 40, 200)
    sigmoid = 1.0 / (1.0 + np.exp(-k_T * (T_range - T_50)))
    phi_IB = phi_min + (1.0 - phi_min) * sigmoid

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T_range, phi_IB * 100, 'b-', linewidth=2, label='Model (v5)')
    ax.axhline(phi_min * 100, color='gray', linestyle='--', alpha=0.5,
               label=f'Baseline: {phi_min*100:.1f}%')
    ax.scatter([25, 30, 35], [0, 13, 100], c='red', s=100, zorder=5,
               edgecolors='black', label='Observed data')
    ax.set_xlabel('Temperature [C]')
    ax.set_ylabel('IB Fraction [%]')
    ax.set_title(f'IB Temperature Response (v5: phi_min={phi_min:.3f}, T_50={T_50:.1f}, k_T={k_T:.2f})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"IB response plot saved to {save_path}")

    plt.show()
    return fig


def print_params(params: np.ndarray):
    """Print fitted parameters."""
    print("\n" + "=" * 60)
    print("FITTED PARAMETERS (hybrid v5 - improved IB model):")
    print("=" * 60)
    for name, value in zip(PARAM_NAMES, params):
        print(f"  {name:12s} = {value:.6f}")
    print("=" * 60)

    # Interpretation
    Y_XS, m_s, alpha, beta, phi_min, T_50, k_T, k_base = params
    print("\nInterpretation:")

    # IB interpretation
    print(f"\n  IB Model: phi_IB(T) = {phi_min:.3f} + (1-{phi_min:.3f}) * sigmoid({k_T:.2f}*(T-{T_50:.1f}))")
    print(f"    At T=25C: phi_IB = {phi_min + (1-phi_min)/(1+np.exp(-k_T*(25-T_50))):.1%}")
    print(f"    At T=30C: phi_IB = {phi_min + (1-phi_min)/(1+np.exp(-k_T*(30-T_50))):.1%}  <-- should match ~13%")
    print(f"    At T=35C: phi_IB = {phi_min + (1-phi_min)/(1+np.exp(-k_T*(35-T_50))):.1%}")

    print(f"\n  Growth: At qs=0.5: mu = {Y_XS * 0.5 - m_s:.3f} 1/h")


# ============================================================
# MAIN
# ============================================================

def main():
    """Multi-run fitting with improved IB model."""
    print("=" * 60)
    print("Hybrid Mechanistic Model (v5)")
    print("Key Improvement: IB model with baseline fraction")
    print("=" * 60)

    # Load corner point data
    corner_runs = ["ferm_01", "ferm_02", "ferm_03", "ferm_04"]
    data_list = []

    for run_name in corner_runs:
        print(f"\nLoading {run_name} (full run)...")
        try:
            data = prepare_full_run_data(run_name)
            print(f"  Time: {data['t0']:.1f} - {data['t1']:.1f} h")
            print(f"  Samples: {len(data['sample_times'])}")
            print(f"  DOE setpoint: qs={data['conditions']['qs_setpoint']}, T={data['conditions']['T_induced']}C")
            data_list.append(data)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n[1] Fitting improved IB model across {len(data_list)} runs...")
    fitted_params, final_loss = fit_multi_run(data_list)

    print_params(fitted_params)

    print(f"\n[2] Per-run losses:")
    for data in data_list:
        loss = compute_loss_single(fitted_params, data)
        print(f"  {data['run_name']}: {loss:.4f}")

    print(f"\n[3] Plotting IB temperature response...")
    plot_ib_temperature_response(fitted_params,
                                save_path=RESULTS_PATH / "hybrid_v5_ib_response.png")

    print(f"\n[4] Plotting all fits...")
    plot_multi_run(fitted_params, data_list,
                   save_path=RESULTS_PATH / "hybrid_v5_fit.png")

    # Save parameters
    with open(RESULTS_PATH / "hybrid_v5_fit_params.txt", "w") as f:
        f.write("Hybrid v5 Fitted Parameters\n")
        f.write("=" * 40 + "\n")
        for name, value in zip(PARAM_NAMES, fitted_params):
            f.write(f"{name:12s} = {value:.6f}\n")
        f.write("=" * 40 + "\n")
    print(f"\nParameters saved to {RESULTS_PATH / 'hybrid_v5_fit_params.txt'}")

    print("\nDone! Improved IB model fitting complete.")
    print("\nNext: Validate on ferm_05-07 (triplicates at 30C)")


if __name__ == "__main__":
    main()
