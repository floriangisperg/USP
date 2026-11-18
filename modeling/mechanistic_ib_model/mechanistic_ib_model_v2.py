"""
Mechanistic IB Formation Model v2: Multiplicative Structure

Changes from v1:
- Switched from additive to multiplicative structure
- OLD: phi_IB = phi_synthesis + phi_aggregation (over-predicts at low T)
- NEW: phi_IB = 1 - (1 - phi_synthesis) * (1 - phi_aggregation)

Mechanistic interpretation:
- (1 - phi_synthesis) = fraction that folds correctly despite synthesis rate
- (1 - phi_aggregation) = fraction that doesn't aggregate thermally
- Product = fraction that remains soluble
- Therefore IB = 1 - soluble fraction

This ensures both mechanisms must contribute for IB to form.

Author: Modeling Playground - USP Project
Date: 2025-01-02
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

# ============================================================
# CONSTANTS AND CONFIGURATION
# ============================================================

DATA_PATH = Path(__file__).parent.parent / "LDH" / "data" / "LDH_consolidated_results.xlsx"
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


# ============================================================
# DATA LOADING (Same as v1)
# ============================================================

def load_run_data(run_name: str) -> pd.DataFrame:
    """Load data for a single run from consolidated Excel file."""
    df_all = pd.read_excel(DATA_PATH, sheet_name='All_Time_Points')
    df = df_all[df_all['run_name'] == run_name].copy()
    if df.empty:
        raise ValueError(f"No data found for run: {run_name}")
    return df


def derive_qs_time_series(df: pd.DataFrame) -> np.ndarray:
    """Derive specific substrate uptake rate (qs) from feed rate data."""
    dt = np.diff(df['Process time [h]'].values, prepend=df['Process time [h]'].values[0])
    dt = np.maximum(dt, 0.01)

    feed_cumulative = df['Feed [mL]'].fillna(0).values
    feed_rate_mL_h = np.diff(feed_cumulative, prepend=feed_cumulative[0]) / dt
    feed_rate_mL_h = np.maximum(feed_rate_mL_h, 0)
    feed_rate_L_h = feed_rate_mL_h / 1000.0

    volume_mL = df['Volume corrected [mL]'].values if 'Volume corrected [mL]' in df.columns else df['Volume [mL]'].values
    volume_L = volume_mL / 1000.0

    biomass_g_L = df['Biomass [g/L]'].values

    qs = np.zeros_like(feed_rate_L_h)
    valid_mask = (biomass_g_L > 0.1) & (volume_L > 0.1)
    qs[valid_mask] = (feed_rate_L_h[valid_mask] * C_FEED) / (biomass_g_L[valid_mask] * volume_L[valid_mask])
    qs = np.clip(qs, 0, 2.0)
    qs = pd.Series(qs).ffill().bfill().values

    return qs


def prepare_full_run_data(run_name: str) -> Dict:
    """Prepare FULL run data (fedbatch + induced phases) with qs derivation."""
    df = load_run_data(run_name)

    has_biomass = df['Biomass [g/L]'].notna()
    samples = df[has_biomass].copy()

    times_all = df['Process time [h]'].values

    volume_mL = df['Volume corrected [mL]'].values if 'Volume corrected [mL]' in df.columns else df['Volume [mL]'].values
    volume_L = volume_mL / 1000.0

    dt = np.diff(times_all, prepend=times_all[0])
    dt = np.maximum(dt, 0.01)
    feed_cumulative_mL = df['Feed [mL]'].fillna(0).values
    feed_rate_mL_h = np.diff(feed_cumulative_mL, prepend=feed_cumulative_mL[0]) / dt
    feed_rate_mL_h = np.maximum(feed_rate_mL_h, 0)
    feed_rate_L_h = feed_rate_mL_h / 1000.0

    T_induced = DOE_CONDITIONS[run_name]["T_induced"]
    is_induced = df['Is induced [-]'].fillna(0).values.astype(float)
    T_values = np.where(is_induced > 0.5, T_induced, T_FEDBATCH)

    qs_values = derive_qs_time_series(df)

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
        "qs_values": qs_values,
        "is_induced": is_induced,
        "t0": float(times_all[0]),
        "t1": float(times_all[-1]),
    }

    return data


# ============================================================
# MECHANISTIC IB FORMATION MODEL v2: MULTIPLICATIVE
# ============================================================

def phi_synthesis_overload(qs: float, k_syn: float, K_syn: float) -> float:
    """
    Synthesis-driven misfolding component.

    High qs → rapid protein synthesis → folding capacity overload
    """
    return k_syn * qs / (K_syn + qs)


def phi_thermal_aggregation(T: float, qs: float, k_agg: float, T_ref: float,
                            Ea_over_R: float, alpha_qs: float) -> float:
    """
    Thermal aggregation component (temperature + qs interaction).
    """
    T_K = T + 273.15
    T_ref_K = T_ref + 273.15

    f_T = np.exp(Ea_over_R * (1/T_ref_K - 1/T_K))
    g_qs = 1 + alpha_qs * qs

    return k_agg * f_T * g_qs


def phi_ib_multiplicative(T: float, qs: float,
                         k_syn: float, K_syn: float,
                         k_agg: float, T_ref: float,
                         Ea_over_R: float, alpha_qs: float) -> float:
    """
    MULTIPLICATIVE IB formation model (v2).

    Mechanistic interpretation:
    - (1 - phi_synthesis) = probability protein folds correctly despite synthesis rate
    - (1 - phi_aggregation) = probability protein doesn't aggregate thermally
    - Product = probability protein remains soluble
    - Therefore: phi_IB = 1 - (1 - phi_syn) * (1 - phi_agg)

    This ensures BOTH mechanisms must contribute for IB to form.

    Key difference from v1:
    - v1 (additive): phi_IB = phi_syn + phi_agg (over-predicts at low T)
    - v2 (multiplicative): phi_IB = 1 - (1 - phi_syn) * (1 - phi_agg)
    """
    phi_syn = phi_synthesis_overload(qs, k_syn, K_syn)
    phi_agg = phi_thermal_aggregation(T, qs, k_agg, T_ref, Ea_over_R, alpha_qs)

    # Soluble fraction = product of probabilities of surviving each mechanism
    phi_soluble = (1 - phi_syn) * (1 - phi_agg)

    # IB fraction = 1 - soluble fraction
    phi_IB = 1 - phi_soluble

    return max(0, min(phi_IB, 1.0))  # Clamp to [0, 1]


# ============================================================
# PARAMETERS (Same as v1)
# ============================================================

PARAM_NAMES = [
    "Y_XS", "m_s", "alpha", "beta",
    "k_syn", "K_syn",
    "k_agg", "T_ref", "Ea_over_R", "alpha_qs",
]

PARAM_BOUNDS = [
    (0.3, 0.7),      # Y_XS
    (0.0, 0.05),     # m_s
    (0.01, 0.3),     # alpha
    (0.0, 0.1),      # beta
    (0.01, 0.5),     # k_syn
    (0.05, 0.5),     # K_syn
    (0.001, 0.3),    # k_agg
    (20.0, 28.0),    # T_ref
    (3000, 8000),    # Ea_over_R
    (0.0, 3.0),      # alpha_qs
]

DEFAULT_PARAMS = np.array([
    0.5,     # Y_XS
    0.01,    # m_s
    0.05,    # alpha
    0.02,    # beta
    0.15,    # k_syn
    0.2,     # K_syn
    0.05,    # k_agg
    24.0,    # T_ref
    5000,    # Ea_over_R
    0.5,     # alpha_qs
])


# ============================================================
# ODE SYSTEM
# ============================================================

def create_interpolators(data: Dict) -> Tuple:
    """Create interpolation functions for time-varying inputs."""
    times = data["times_all"]

    feed_interp = interp1d(times, data["Feed_rate_L_h"], kind='linear',
                           bounds_error=False, fill_value=(data["Feed_rate_L_h"][0], data["Feed_rate_L_h"][-1]))
    T_interp = interp1d(times, data["T_values"], kind='linear',
                        bounds_error=False, fill_value=(data["T_values"][0], data["T_values"][-1]))
    qs_interp = interp1d(times, data["qs_values"], kind='linear',
                         bounds_error=False, fill_value=(data["qs_values"][0], data["qs_values"][-1]))
    is_induced_interp = interp1d(times, data["is_induced"], kind='linear',
                                  bounds_error=False, fill_value=(data["is_induced"][0], data["is_induced"][-1]))

    return feed_interp, T_interp, qs_interp, is_induced_interp


def ode_system(t, y, params, feed_interp, T_interp, qs_interp, is_induced_interp):
    """
    Mechanistic ODE system with MULTIPLICATIVE IB model (v2).
    """
    X, P_sol, P_IB, V = y

    Y_XS, m_s, alpha, beta, k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs = params

    F = float(feed_interp(t))
    T = float(T_interp(t))
    qs = float(qs_interp(t))
    I = float(is_induced_interp(t))

    X_safe = max(X, 0.1)
    V_safe = max(V, 0.1)

    # Growth rate
    mu = Y_XS * qs - m_s
    mu = max(mu, 0)

    D = F / V_safe

    # Biomass ODE
    dX_dt = mu * X_safe - D * X

    # Product formation rate
    T_opt = 30.0
    T_width = 10.0
    f_T_expr = np.exp(-0.5 * ((T - T_opt) / T_width)**2)
    q_p = (alpha * mu + beta) * I * f_T_expr

    # IB partitioning (MULTIPLICATIVE model)
    phi_IB = phi_ib_multiplicative(T, qs, k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs)

    # Protein ODEs
    dP_sol_dt = q_p * (1 - phi_IB) * X_safe - D * P_sol
    dP_IB_dt = q_p * phi_IB * X_safe - D * P_IB

    dV_dt = F

    return [dX_dt, dP_sol_dt, dP_IB_dt, dV_dt]


def solve_ode(params: np.ndarray, data: Dict,
              dense_output: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the ODE system."""
    feed_interp, T_interp, qs_interp, is_induced_interp = create_interpolators(data)

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
                                    qs_interp, is_induced_interp),
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

def fit_multi_run(data_list: List[Dict], runs: List[str]) -> Tuple[np.ndarray, float]:
    """Fit shared parameters across multiple runs."""
    print(f"Multi-run fitting (multiplicative IB model v2) on: {runs}")
    print(f"Initial loss: {compute_loss_multi(DEFAULT_PARAMS, data_list):.4f}")

    result = differential_evolution(
        compute_loss_multi,
        bounds=PARAM_BOUNDS,
        args=(data_list,),
        maxiter=100,
        seed=42,
        disp=True,
        polish=True,
        workers=1,
        popsize=10
    )

    print(f"\nFinal loss: {result.fun:.4f}")
    return result.x, result.fun


# ============================================================
# VISUALIZATION
# ============================================================

def print_params(params: np.ndarray):
    """Print fitted parameters."""
    print("\n" + "=" * 60)
    print("FITTED PARAMETERS (Multiplicative IB Model v2)")
    print("=" * 60)
    for name, value in zip(PARAM_NAMES, params):
        print(f"  {name:12s} = {value:.6f}")
    print("=" * 60)

    Y_XS, m_s, alpha, beta, k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs = params
    print("\nInterpretation (MULTIPLICATIVE):")

    print(f"\n  Synthesis Overload Component:")
    print(f"    phi_syn = {k_syn:.3f} * qs / ({K_syn:.3f} + qs)")
    print(f"    At qs=0.2: {phi_synthesis_overload(0.2, k_syn, K_syn):.1%}")
    print(f"    At qs=0.4: {phi_synthesis_overload(0.4, k_syn, K_syn):.1%}")
    print(f"    At qs=0.6: {phi_synthesis_overload(0.6, k_syn, K_syn):.1%}")

    print(f"\n  Thermal Aggregation Component:")
    print(f"    phi_agg = {k_agg:.3f} * f_T(T) * (1 + {alpha_qs:.2f}*qs)")
    print(f"    At T=25°C, qs=0.2: {phi_thermal_aggregation(25, 0.2, k_agg, T_ref, Ea_over_R, alpha_qs):.1%}")
    print(f"    At T=30°C, qs=0.4: {phi_thermal_aggregation(30, 0.4, k_agg, T_ref, Ea_over_R, alpha_qs):.1%}")
    print(f"    At T=35°C, qs=0.6: {phi_thermal_aggregation(35, 0.6, k_agg, T_ref, Ea_over_R, alpha_qs):.1%}")

    print(f"\n  Total IB Fraction (MULTIPLICATIVE):")
    print(f"    phi_IB = 1 - (1 - phi_syn) * (1 - phi_agg)")
    print(f"    At T=25°C, qs=0.2: {phi_ib_multiplicative(25, 0.2, k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs):.1%} (target: 0%)")
    print(f"    At T=30°C, qs=0.4: {phi_ib_multiplicative(30, 0.4, k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs):.1%} (target: ~13%)")
    print(f"    At T=35°C, qs=0.6: {phi_ib_multiplicative(35, 0.6, k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs):.1%} (target: ~100%)")


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
        ax.plot(times_dense, solution[:, 0], 'b-', linewidth=2, label='Model v2')
        ax.scatter(data["sample_times"], data["X_obs"], c='blue', s=60,
                   zorder=5, edgecolors='black', label='Observed')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Biomass [g/L]')
        ax.set_title(f'{run_label}\nBiomass (X)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Soluble Protein
        ax = axes[i, 1]
        ax.plot(times_dense, solution[:, 1], 'g-', linewidth=2, label='Model v2')
        valid = ~np.isnan(data["P_sol_obs"])
        ax.scatter(data["sample_times"][valid], data["P_sol_obs"][valid],
                   c='green', s=60, zorder=5, edgecolors='black', label='Observed')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Protein [g/L]')
        ax.set_title('Soluble Protein (P_sol)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # IB
        ax = axes[i, 2]
        ax.plot(times_dense, solution[:, 2], 'r-', linewidth=2, label='Model v2')
        valid = ~np.isnan(data["P_IB_obs"])
        ax.scatter(data["sample_times"][valid], data["P_IB_obs"][valid],
                   c='red', s=60, zorder=5, edgecolors='black', label='Observed')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Protein [g/L]')
        ax.set_title('Inclusion Bodies (P_IB)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    """Main fitting routine with multiplicative model."""
    print("=" * 60)
    print("Mechanistic IB Formation Model v2: MULTIPLICATIVE")
    print("phi_IB = 1 - (1 - phi_syn) * (1 - phi_agg)")
    print("=" * 60)

    all_runs = list(DOE_CONDITIONS.keys())
    data_list = []

    for run_name in all_runs:
        print(f"\nLoading {run_name}...")
        try:
            data = prepare_full_run_data(run_name)
            print(f"  Time: {data['t0']:.1f} - {data['t1']:.1f} h")
            print(f"  Samples: {len(data['sample_times'])}")
            print(f"  Conditions: qs={data['conditions']['qs_setpoint']}, T={data['conditions']['T_induced']}°C")
            data_list.append(data)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n[1] Fitting multiplicative IB model across {len(data_list)} runs...")
    fitted_params, final_loss = fit_multi_run(data_list, all_runs)

    print_params(fitted_params)

    print(f"\n[2] Per-run losses:")
    for data in data_list:
        loss = compute_loss_single(fitted_params, data)
        print(f"  {data['run_name']}: {loss:.4f}")

    print(f"\n[3] Plotting all fits...")
    plot_multi_run(fitted_params, data_list,
                   save_path=RESULTS_PATH / "mechanistic_ib_v2_fit.png")

    # Save parameters
    with open(RESULTS_PATH / "mechanistic_ib_v2_params.txt", "w") as f:
        f.write("Mechanistic IB Model v2 Fitted Parameters (MULTIPLICATIVE)\n")
        f.write("phi_IB = 1 - (1 - phi_syn) * (1 - phi_agg)\n")
        f.write("=" * 55 + "\n")
        for name, value in zip(PARAM_NAMES, fitted_params):
            f.write(f"{name:12s} = {value:.6f}\n")
        f.write("=" * 55 + "\n")
    print(f"\nParameters saved to {RESULTS_PATH / 'mechanistic_ib_v2_params.txt'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
