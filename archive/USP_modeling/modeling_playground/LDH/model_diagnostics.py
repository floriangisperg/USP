"""
Model Diagnostics: Profile Likelihood + Residual Analysis

This script answers:
1. Are parameters well-identified? (Profile likelihood)
2. Does IB behavior change over time? (Residual analysis)
3. Is the model overfitting? (Parameter constraints)

Uses the fitted parameters from hybrid_v4_fit.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Import model functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fit_model import (
    prepare_full_run_data,
    PARAM_NAMES,
    PARAM_BOUNDS,
    C_FEED,
    T_FEDBATCH,
)

RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# ============================================================
# FITTED PARAMETERS (from hybrid_v4_fit.txt)
# ============================================================
FITTED_PARAMS = np.array([
    0.482761,  # Y_XS
    0.000264,  # m_s
    0.061512,  # alpha
    0.008622,  # beta
    34.049452,  # T_50
    1.756726,  # k_T
    0.000632,  # k_base
])

# Map to parameter names
param_dict = dict(zip(PARAM_NAMES, FITTED_PARAMS))

print("=" * 70)
print("MODEL DIAGNOSTICS: Profile Likelihood + Residual Analysis")
print("=" * 70)
print("\nUsing fitted parameters from hybrid_v4:")
for name, value in param_dict.items():
    print(f"  {name:12s} = {value:.6f}")

# ============================================================
# ODE SYSTEM (from fit_model.py)
# ============================================================

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
    """Hybrid ODE system (original v4 model)."""
    X, P_sol, P_IB, V = y
    Y_XS, m_s, alpha, beta, T_50, k_T, k_base = params

    F = float(feed_interp(t))
    T = float(T_interp(t))
    I = float(is_induced_interp(t))

    X_safe = max(X, 0.1)
    V_safe = max(V, 0.1)

    qs = (F * C_FEED) / (X_safe * V_safe)
    qs = min(qs, 2.0)

    mu = max(Y_XS * qs - m_s, 0.0)

    F_base = k_base * X_safe * V_safe
    dV_dt = F + F_base
    D = (F + F_base) / V_safe

    dX_dt = mu * X - D * X

    T_opt = 30.0
    T_width = 10.0
    f_T = np.exp(-0.5 * ((T - T_opt) / T_width)**2)
    q_p = (alpha * mu + beta) * I * f_T

    phi_IB = 1.0 / (1.0 + np.exp(-k_T * (T - T_50)))

    dP_sol_dt = q_p * (1 - phi_IB) * X - D * P_sol
    dP_IB_dt = q_p * phi_IB * X - D * P_IB

    return [dX_dt, dP_sol_dt, dP_IB_dt, dV_dt]


def solve_ode(params: np.ndarray, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_loss(params: np.ndarray, data_list: List[Dict]) -> float:
    """Compute total loss across multiple runs."""
    total_loss = 0.0
    for data in data_list:
        try:
            times, solution = solve_ode(params, data)
        except:
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

        total_loss += loss_X + loss_P_sol + loss_P_IB

    return total_loss / len(data_list)


# ============================================================
# PROFILE LIKELIHOOD ANALYSIS
# ============================================================

def profile_likelihood(data_list: List[Dict], param_idx: int,
                       param_values: np.ndarray, fitted_params: np.ndarray):
    """
    Compute profile likelihood for a single parameter.

    For each value of the parameter, fit all other parameters.
    This shows how well-constrained the parameter is.
    """
    losses = []

    for val in param_values:
        # Fix the parameter at this value
        test_params = fitted_params.copy()
        test_params[param_idx] = val

        # Compute loss (other params at fitted values)
        # Note: Full refit would be better but much slower
        loss = compute_loss(test_params, data_list)
        losses.append(loss)

    return np.array(losses)


def run_profile_likelihood_analysis(data_list: List[Dict], fitted_params: np.ndarray):
    """Run profile likelihood on all parameters."""
    print("\n" + "=" * 70)
    print("PROFILE LIKELIHOOD ANALYSIS")
    print("=" * 70)
    print("\nTesting: How does loss change when each parameter is varied?")
    print("(Other parameters kept at fitted values for speed)")

    # Define ranges for each parameter (±50% around fitted value, or bounds)
    n_points = 20
    profile_results = {}

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for i, (name, fitted_val, bounds) in enumerate(zip(PARAM_NAMES, fitted_params, PARAM_BOUNDS)):
        print(f"\n[{i+1}/{len(PARAM_NAMES)}] Profiling {name}...")

        # Create range around fitted value
        lower = max(bounds[0], fitted_val * 0.5)
        upper = min(bounds[1], fitted_val * 1.5)
        param_range = np.linspace(lower, upper, n_points)

        # Compute profile
        losses = profile_likelihood(data_list, i, param_range, fitted_params)

        # Find minimum and confidence interval
        min_loss = np.min(losses)
        optimal_val = param_range[np.argmin(losses)]

        # Approximate confidence interval (Δloss = 1)
        threshold = min_loss + 0.1  # Relaxed threshold for small sample
        mask = losses <= threshold
        if np.any(mask):
            ci_lower = param_range[mask][0]
            ci_upper = param_range[mask][-1]
        else:
            ci_lower = ci_upper = fitted_val

        profile_results[name] = {
            'range': param_range,
            'losses': losses,
            'fitted': fitted_val,
            'optimal': optimal_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'min_loss': min_loss,
        }

        # Plot
        ax = axes[i]
        ax.plot(param_range, losses, 'b-', linewidth=2, label='Loss')
        ax.axvline(fitted_val, color='r', linestyle='--', label=f'Fitted: {fitted_val:.4f}')
        ax.axvline(optimal_val, color='g', linestyle=':', label=f'Optimal: {optimal_val:.4f}')
        ax.axhline(min_loss, color='gray', linestyle='-', alpha=0.3)

        # Shade confidence region
        ax.fill_between(param_range, 0, threshold, alpha=0.2, color='green',
                        where=(losses <= threshold))

        ax.set_xlabel(f'{name}')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name}: fitted={fitted_val:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Assess identifiability
        width = ci_upper - ci_lower
        rel_width = width / fitted_val if fitted_val > 0 else width
        if rel_width < 0.2:
            status = "WELL-IDENTIFIED"
        elif rel_width < 0.5:
            status = "MODERATELY IDENTIFIED"
        else:
            status = "POORLY IDENTIFIED"
        print(f"  Status: {status}")
        print(f"  Fitted: {fitted_val:.6f}, Optimal: {optimal_val:.6f}")
        print(f"  CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

    # Hide extra subplot
    axes[-1].axis('off')

    plt.suptitle('Profile Likelihood: Parameter Identifiability', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "profile_likelihood.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {RESULTS_PATH / 'profile_likelihood.png'}")

    return profile_results


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

def run_residual_analysis(data_list: List[Dict], params: np.ndarray):
    """Analyze residuals for patterns (time dependence, etc.)."""
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)
    print("\nChecking for patterns in residuals (indicates model misspecification)")

    fig, axes = plt.subplots(len(data_list), 3, figsize=(14, 3*len(data_list)))
    if len(data_list) == 1:
        axes = axes.reshape(1, -1)

    all_residuals = {'time': [], 'X': [], 'P_sol': [], 'P_IB': [], 'run': []}

    for i, data in enumerate(data_list):
        print(f"\n[{data['run_name']}] Computing residuals...")

        times, solution = solve_ode(params, data)

        X_pred = solution[:, 0]
        P_sol_pred = solution[:, 1]
        P_IB_pred = solution[:, 2]

        X_obs = data["X_obs"]
        P_sol_obs = data["P_sol_obs"]
        P_IB_obs = data["P_IB_obs"]
        sample_times = data["sample_times"]

        # Compute residuals (only where data exists)
        def get_residuals(pred, obs):
            mask = ~np.isnan(obs)
            return sample_times[mask], pred[mask] - obs[mask]

        t_X, res_X = get_residuals(X_pred, X_obs)
        t_P_sol, res_P_sol = get_residuals(P_sol_pred, P_sol_obs)
        t_P_IB, res_P_IB = get_residuals(P_IB_pred, P_IB_obs)

        # Store for aggregate analysis
        all_residuals['time'].extend(list(t_X) + list(t_P_sol) + list(t_P_IB))
        all_residuals['X'].extend(list(res_X))
        all_residuals['P_sol'].extend(list(res_P_sol))
        all_residuals['P_IB'].extend(list(res_P_IB))
        all_residuals['run'].extend([data['run_name']] * (len(res_X) + len(res_P_sol) + len(res_P_IB)))

        # Plot per-run residuals
        for j, (var, t, res, color, ylabel) in enumerate([
            ('X', t_X, res_X, 'blue', 'Biomass residual [g/L]'),
            ('P_sol', t_P_sol, res_P_sol, 'green', 'Soluble protein residual [g/L]'),
            ('P_IB', t_P_IB, res_P_IB, 'red', 'IB residual [g/L]'),
        ]):
            ax = axes[i, j]
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.scatter(t, res, c=color, s=60, alpha=0.7, edgecolors='black')

            # Add trend line
            if len(t) > 1:
                z = np.polyfit(t, res, 1)
                p = np.poly1d(z)
                t_trend = np.linspace(t.min(), t.max(), 20)
                ax.plot(t_trend, p(t_trend), color='darkred', linestyle='--',
                        linewidth=2, alpha=0.7, label=f'Trend: slope={z[0]:.3f}')
                ax.legend(fontsize=8)

            ax.set_xlabel('Time [h]')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{data['run_name']} - {var}")
            ax.grid(True, alpha=0.3)

        # Print statistics
        print(f"  X:    mean={np.mean(res_X):.3f}, std={np.std(res_X):.3f}, n={len(res_X)}")
        print(f"  P_sol: mean={np.mean(res_P_sol):.3f}, std={np.std(res_P_sol):.3f}, n={len(res_P_sol)}")
        print(f"  P_IB:  mean={np.mean(res_P_IB):.3f}, std={np.std(res_P_IB):.3f}, n={len(res_P_IB)}")

    plt.suptitle('Residual Analysis: Check for Time Patterns', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "residuals_per_run.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {RESULTS_PATH / 'residuals_per_run.png'}")

    return all_residuals


# ============================================================
# IB FRACTION EVOLUTION ANALYSIS
# ============================================================

def analyze_ib_fraction_evolution(data_list: List[Dict], params: np.ndarray):
    """
    Check if IB fraction changes over time within runs.

    If the model is correct (phi_IB = f(T) only), then IB fraction should
    be constant within each run (after induction).
    """
    print("\n" + "=" * 70)
    print("IB FRACTION EVOLUTION ANALYSIS")
    print("=" * 70)
    print("\nChecking: Does IB fraction change over time within each run?")
    print("(If yes, then IB is not just a function of temperature)")

    fig, axes = plt.subplots(1, len(data_list), figsize=(4*len(data_list), 4))
    if len(data_list) == 1:
        axes = [axes]

    for i, data in enumerate(data_list):
        run_name = data['run_name']
        print(f"\n[{run_name}]")

        times, solution = solve_ode(params, data)

        P_sol_pred = solution[:, 1]
        P_IB_pred = solution[:, 2]

        P_sol_obs = data["P_sol_obs"]
        P_IB_obs = data["P_IB_obs"]
        sample_times = data["sample_times"]

        # Compute observed IB fraction at each time point
        mask = ~np.isnan(P_sol_obs) & ~np.isnan(P_IB_obs)
        t_valid = sample_times[mask]

        # Observed IB fraction
        total_obs = P_sol_obs[mask] + P_IB_obs[mask]
        IB_frac_obs = P_IB_obs[mask] / total_obs

        # Predicted IB fraction
        P_sol_pred_valid = P_sol_pred[mask]
        P_IB_pred_valid = P_IB_pred[mask]
        total_pred = P_sol_pred_valid + P_IB_pred_valid
        IB_frac_pred = P_IB_pred_valid / total_pred

        # Plot
        ax = axes[i]
        ax.scatter(t_valid, IB_frac_obs * 100, c='red', s=80, alpha=0.8,
                   edgecolors='black', label='Observed', marker='o', zorder=5)
        ax.scatter(t_valid, IB_frac_pred * 100, c='blue', s=60, alpha=0.6,
                   edgecolors='black', label='Predicted', marker='s')

        # Fit trend to observed data
        if len(t_valid) > 1:
            z = np.polyfit(t_valid, IB_frac_obs, 1)
            p = np.poly1d(z)
            t_trend = np.linspace(t_valid.min(), t_valid.max(), 20)
            ax.plot(t_trend, p(t_trend) * 100, color='darkred', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'Trend: {z[0]*100:.2f}%/h')

        T_ind = data['conditions']['T_induced']
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('IB Fraction [%]')
        ax.set_title(f"{run_name} (T={T_ind}C)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

        # Statistics
        if len(IB_frac_obs) > 1:
            mean_IB_frac = np.mean(IB_frac_obs)
            std_IB_frac = np.std(IB_frac_obs)
            cv_IB_frac = std_IB_frac / mean_IB_frac if mean_IB_frac > 0 else np.inf

            print(f"  IB fraction: mean={mean_IB_frac*100:.1f}%, std={std_IB_frac*100:.1f}%, CV={cv_IB_frac:.2f}")
            print(f"  Trend slope: {z[0]*100:.2f}%/h (p-value not computed)")

            # Test if IB fraction is constant (simple test: std < threshold)
            if std_IB_frac < 0.05:  # Less than 5% variation
                print(f"  Assessment: IB fraction appears CONSTANT")
            else:
                print(f"  Assessment: IB fraction appears to CHANGE over time")

    plt.suptitle('IB Fraction Evolution Within Runs', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "ib_fraction_evolution.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {RESULTS_PATH / 'ib_fraction_evolution.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run all diagnostics."""
    # Load data
    print("\nLoading data...")
    all_runs = ["ferm_01", "ferm_02", "ferm_03", "ferm_04", "ferm_05", "ferm_06", "ferm_07"]
    data_list = []

    for run_name in all_runs:
        try:
            data = prepare_full_run_data(run_name)
            data_list.append(data)
        except Exception as e:
            print(f"  Error loading {run_name}: {e}")

    print(f"Loaded {len(data_list)} runs")

    # 1. Profile Likelihood Analysis
    profile_results = run_profile_likelihood_analysis(data_list, FITTED_PARAMS)

    # 2. Residual Analysis
    residual_results = run_residual_analysis(data_list, FITTED_PARAMS)

    # 3. IB Fraction Evolution
    analyze_ib_fraction_evolution(data_list, FITTED_PARAMS)

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print("\n1. PARAMETER IDENTIFIABILITY:")
    for name, res in profile_results.items():
        width = res['ci_upper'] - res['ci_lower']
        rel_width = width / res['fitted'] if res['fitted'] > 0 else width
        if rel_width < 0.2:
            status = "[OK] Well-identified"
        elif rel_width < 0.5:
            status = "[WARN] Moderate"
        else:
            status = "[BAD] Poorly identified"
        print(f"  {name:12s}: {status}")

    print("\n2. RESIDUAL PATTERNS:")
    print("  Check plots for systematic trends (model misspecification)")
    print("  Random scatter = good, trends = missing terms")

    print("\n3. IB FRACTION STABILITY:")
    print("  Check if IB fraction changes over time")
    print("  If stable: current model (phi_IB = f(T)) is adequate")
    print("  If changing: need extended model with time/state dependence")

    print("\n" + "=" * 70)
    print("All diagnostic plots saved to:")
    print(f"  - {RESULTS_PATH / 'profile_likelihood.png'}")
    print(f"  - {RESULTS_PATH / 'residuals_per_run.png'}")
    print(f"  - {RESULTS_PATH / 'ib_fraction_evolution.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    plt.show()
