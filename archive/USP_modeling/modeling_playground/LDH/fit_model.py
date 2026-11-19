"""
Hybrid Mechanistic Model for E. coli LDH Fed-Batch Data (v4)

Key features (based on arXiv:2504.04665 approach):
- qs is DERIVED from state: qs(t) = F(t) * C_f / (X(t) * V(t))
- Growth: μ = Y_XS * qs - m_s (mechanistic link to feed)
- Product: Luedeking-Piret q_p = (α*μ + β) * I * f_T(T)
- IB partition: φ_IB(T) = sigmoid(k * (T - T_50))
- Base addition: dV/dt = F + k_base * X
- Full fedbatch data (not just induced phase)

This allows objective testing: "does qs matter?"

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

# DOE conditions (for reference only - qs is now DERIVED, not prescribed)
DOE_CONDITIONS = {
    "ferm_01": {"qs_setpoint": 0.2, "T_induced": 35.0},
    "ferm_02": {"qs_setpoint": 0.2, "T_induced": 25.0},
    "ferm_03": {"qs_setpoint": 0.6, "T_induced": 25.0},
    "ferm_04": {"qs_setpoint": 0.6, "T_induced": 35.0},
    "ferm_05": {"qs_setpoint": 0.4, "T_induced": 30.0},
    "ferm_06": {"qs_setpoint": 0.4, "T_induced": 30.0},
    "ferm_07": {"qs_setpoint": 0.4, "T_induced": 30.0},
}

# Known constants
C_FEED = 400.0  # Feed glucose concentration [g/L]
T_FEDBATCH = 37.0  # Temperature before induction [°C]
SAMPLE_VOLUME = 10.0  # mL per sample


def load_run_data(run_name: str) -> pd.DataFrame:
    """Load data for a single run."""
    df_all = pd.read_excel(DATA_PATH, sheet_name='All_Time_Points')
    df = df_all[df_all['run_name'] == run_name].copy()
    if df.empty:
        raise ValueError(f"No data found for run: {run_name}")
    return df


def prepare_full_run_data(run_name: str) -> Dict:
    """
    Prepare FULL run data (fedbatch + induced phases).
    
    Key: extract feed RATE from cumulative feed volume.
    """
    df = load_run_data(run_name)
    
    # Sample points: where Biomass is measured
    has_biomass = df['Biomass [g/L]'].notna()
    samples = df[has_biomass].copy()
    
    # All time points for online data
    times_all = df['Process time [h]'].values
    
    # Volume (mL -> L for consistency)
    volume_mL = pd.Series(df['Volume [mL]'].values).ffill().bfill().values
    volume_corrected_mL = pd.Series(df['Volume corrected [mL]'].values).ffill().bfill().values if 'Volume corrected [mL]' in df.columns else volume_mL
    
    # Convert to L
    volume_L = volume_corrected_mL / 1000.0
    
    # Feed: convert cumulative [mL] to rate [L/h]
    feed_cumulative_mL = df['Feed [mL]'].fillna(0).values
    dt = np.diff(times_all)
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 1.0)
    dt = np.maximum(dt, 0.01)  # Avoid division by zero
    
    feed_rate_mL_h = np.diff(feed_cumulative_mL, prepend=feed_cumulative_mL[0]) / dt
    feed_rate_mL_h = np.maximum(feed_rate_mL_h, 0)  # No negative feed
    feed_rate_L_h = feed_rate_mL_h / 1000.0  # mL/h -> L/h
    
    # Temperature profile
    T_induced = DOE_CONDITIONS[run_name]["T_induced"]
    is_induced = df['Is induced [-]'].fillna(0).values.astype(float)
    T_values = np.where(is_induced > 0.5, T_induced, T_FEDBATCH)
    
    data = {
        "run_name": run_name,
        "conditions": DOE_CONDITIONS[run_name],
        # Sample data
        "sample_times": samples['Process time [h]'].values,
        "X_obs": samples['Biomass [g/L]'].values,
        "P_sol_obs": samples['Protein soluble [g/L]'].values if 'Protein soluble [g/L]' in samples.columns else np.full(len(samples), np.nan),
        "P_IB_obs": samples['Protein IB [g/L]'].values if 'Protein IB [g/L]' in samples.columns else np.full(len(samples), np.nan),
        # Online data for interpolation
        "times_all": times_all,
        "Volume_L": volume_L,
        "Feed_rate_L_h": feed_rate_L_h,
        "T_values": T_values,
        "is_induced": is_induced,
        # Time range
        "t0": float(times_all[0]),
        "t1": float(times_all[-1]),
    }
    
    return data


# ============================================================
# HYBRID ODE MODEL
# ============================================================

# Parameters to fit (~5-7)
PARAM_NAMES = [
    "Y_XS",     # Yield on glucose [gX/gS]
    "m_s",      # Maintenance coefficient [gS/(gX*h)]
    "alpha",    # Growth-associated product formation [gP/gX]
    "beta",     # Non-growth-associated product formation [gP/(gX*h)]
    "T_50",     # Temperature for 50% IB partition [°C]
    "k_T",      # Steepness of IB partition sigmoid [1/°C]
    "k_base",   # Base addition proportional to X [L/(gX*h)]
]

PARAM_BOUNDS = [
    (0.3, 0.7),      # Y_XS: typical E. coli yield
    (0.0, 0.05),     # m_s: maintenance (can start at 0)
    (0.01, 0.3),     # alpha: growth-associated
    (0.0, 0.1),      # beta: non-growth-associated
    (25.0, 35.0),    # T_50: midpoint for IB formation
    (0.1, 2.0),      # k_T: steepness
    (0.0, 0.001),    # k_base: base addition [L/(gX*h)]
]

DEFAULT_PARAMS = np.array([0.5, 0.01, 0.05, 0.02, 30.0, 0.5, 0.0001])


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
    Hybrid ODE system for fed-batch fermentation.
    
    States: y = [X, P_sol, P_IB, V]
    - X: biomass concentration [g/L]
    - P_sol: soluble protein concentration [g/L]
    - P_IB: inclusion body concentration [g/L]
    - V: broth volume [L]
    
    KEY: qs is DERIVED from state, not prescribed!
    qs(t) = F(t) * C_f / (X(t) * V(t))
    """
    X, P_sol, P_IB, V = y
    
    # Unpack parameters
    Y_XS, m_s, alpha, beta, T_50, k_T, k_base = params
    
    # Get interpolated inputs
    F = float(feed_interp(t))        # Feed rate [L/h]
    T = float(T_interp(t))           # Temperature [°C]
    I = float(is_induced_interp(t))  # Induction flag [0 or 1]
    
    # Safety: prevent division by zero
    X_safe = max(X, 0.1)  # At least 0.1 g/L
    V_safe = max(V, 0.1)  # At least 0.1 L
    
    # === DERIVED qs (the key mechanistic link!) ===
    # qs = substrate uptake rate [gS/(gX*h)]
    # In C-limited fedbatch: qs = F * C_f / (X * V)
    qs = (F * C_FEED) / (X_safe * V_safe)
    
    # Apply a reasonable upper bound (max biological uptake)
    qs = min(qs, 2.0)  # Max ~2 g glucose / (g biomass * h)
    
    # === Growth rate (mechanistic via qs) ===
    # μ = Y_XS * qs - m_s
    mu = max(Y_XS * qs - m_s, 0.0)  # No negative growth
    
    # === Volume dynamics ===
    # dV/dt = feed inflow + base addition (proportional to X)
    # Base addition compensates for ammonia used for pH control
    F_base = k_base * X_safe * V_safe  # [L/h] = [L/(gX*h)] * [gX/L] * [L]
    dV_dt = F + F_base
    
    # === Dilution rate ===
    D = (F + F_base) / V_safe  # [1/h]
    
    # === Biomass: growth - dilution ===
    dX_dt = mu * X - D * X
    
    # === Product formation (Luedeking-Piret) ===
    # q_p = (α*μ + β) * I * f_T(T)
    # Only during induction, with optional temperature effect
    
    # Temperature effect on expression (Gaussian around optimum)
    T_opt = 30.0  # Optimal expression temperature
    T_width = 10.0  # Width of Gaussian
    f_T = np.exp(-0.5 * ((T - T_opt) / T_width)**2)
    
    q_p = (alpha * mu + beta) * I * f_T  # [gP/(gX*h)]
    
    # === IB partitioning (sigmoid in T) ===
    # φ_IB(T) = sigmoid(k_T * (T - T_50))
    # Higher T → more IB
    phi_IB = 1.0 / (1.0 + np.exp(-k_T * (T - T_50)))
    
    # === Protein ODEs ===
    dP_sol_dt = q_p * (1 - phi_IB) * X - D * P_sol
    dP_IB_dt = q_p * phi_IB * X - D * P_IB
    
    return [dX_dt, dP_sol_dt, dP_IB_dt, dV_dt]


def solve_ode(params: np.ndarray, data: Dict, 
              dense_output: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the ODE system."""
    feed_interp, T_interp, is_induced_interp, V_data_interp = create_interpolators(data)
    
    # Initial conditions from first sample
    X0 = data["X_obs"][0] if not np.isnan(data["X_obs"][0]) else 5.0
    P_sol_0 = data["P_sol_obs"][0] if not np.isnan(data["P_sol_obs"][0]) else 0.0
    P_IB_0 = data["P_IB_obs"][0] if not np.isnan(data["P_IB_obs"][0]) else 0.0
    
    # Volume at first sample time
    t0_sample = data["sample_times"][0]
    idx0 = np.argmin(np.abs(data["times_all"] - t0_sample))
    V0 = data["Volume_L"][idx0]
    
    y0 = [X0, P_sol_0, P_IB_0, V0]
    
    # Time span
    t_span = (data["t0"], data["t1"])
    
    # Evaluation points
    if dense_output:
        t_eval = np.linspace(data["t0"], data["t1"], 200)
    else:
        t_eval = data["sample_times"]
    
    # Solve
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
    
    # Weighted loss
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
    print(f"Multi-run fitting (hybrid v4) on: {run_names}")
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
        run_label = f"{data['run_name']}: qs_set={cond['qs_setpoint']}, T={cond['T_induced']}°C"
        
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


def print_params(params: np.ndarray):
    """Print fitted parameters."""
    print("\n" + "=" * 60)
    print("FITTED PARAMETERS (hybrid v4 - qs derived):")
    print("=" * 60)
    for name, value in zip(PARAM_NAMES, params):
        print(f"  {name:12s} = {value:.6f}")
    print("=" * 60)
    
    # Interpretation
    Y_XS, m_s, alpha, beta, T_50, k_T, k_base = params
    print("\nInterpretation:")
    print(f"  At qs=0.5: μ ≈ {Y_XS * 0.5 - m_s:.3f} 1/h")
    print(f"  At T=25°C: φ_IB = {1/(1+np.exp(-k_T*(25-T_50))):.2%}")
    print(f"  At T=35°C: φ_IB = {1/(1+np.exp(-k_T*(35-T_50))):.2%}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Multi-run fitting with hybrid model."""
    print("=" * 60)
    print("Hybrid Mechanistic Model (v4)")
    print("Key: qs is DERIVED from state, not prescribed!")
    print("=" * 60)
    
    # Load all corner point data (full runs, not just induced phase)
    corner_runs = ["ferm_01", "ferm_02", "ferm_03", "ferm_04"]
    data_list = []
    
    for run_name in corner_runs:
        print(f"\nLoading {run_name} (full run)...")
        try:
            data = prepare_full_run_data(run_name)
            print(f"  Time: {data['t0']:.1f} - {data['t1']:.1f} h")
            print(f"  Samples: {len(data['sample_times'])}")
            print(f"  DOE setpoint: qs={data['conditions']['qs_setpoint']}, T={data['conditions']['T_induced']}°C")
            data_list.append(data)
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n[1] Fitting hybrid model across {len(data_list)} runs...")
    fitted_params, final_loss = fit_multi_run(data_list)
    
    print_params(fitted_params)
    
    print(f"\n[2] Per-run losses:")
    for data in data_list:
        loss = compute_loss_single(fitted_params, data)
        print(f"  {data['run_name']}: {loss:.4f}")
    
    print(f"\n[3] Plotting all fits...")
    plot_multi_run(fitted_params, data_list, 
                   save_path=RESULTS_PATH / "hybrid_v4_fit.png")
    
    print("\nDone! Hybrid model fitting complete.")
    print("\nNext steps for qs analysis:")
    print("  - Test A: Ablation (compare qs-enabled vs qs-disabled)")
    print("  - Test B: Counterfactual simulation (vary feed profile)")
    print("  - Test C: Parameter identifiability (profile likelihood)")


if __name__ == "__main__":
    main()
