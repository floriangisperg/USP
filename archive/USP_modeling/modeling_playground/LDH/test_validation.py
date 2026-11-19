"""
Validation Test: Test fitted parameters on triplicate setpoint

The hybrid model was fitted on ferm_01-04 (corner points of DOE).
Now we test on ferm_05-07 (triplicates at qs=0.4, T=30°C).

This tests whether the model GENERALIZES to unseen conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

# Import model functions from fit_model.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fit_model import (
    prepare_full_run_data,
    solve_ode,
    PARAM_NAMES,
    create_interpolators,
    ode_system
)

# ============================================================
# PRE-FITTED PARAMETERS (from hybrid_v4_fit.txt)
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

RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)


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


def compute_r_squared(pred: np.ndarray, obs: np.ndarray) -> float:
    """Compute R² for predictions."""
    mask = ~np.isnan(obs)
    if not np.any(mask):
        return np.nan
    obs_valid = obs[mask]
    pred_valid = pred[mask]
    ss_res = np.sum((obs_valid - pred_valid)**2)
    ss_tot = np.sum((obs_valid - np.mean(obs_valid))**2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def plot_validation_run(params: np.ndarray, data: Dict, ax_row,
                        show_metrics: bool = True):
    """Plot model fit for a validation run."""
    times_dense, solution = solve_ode(params, data, dense_output=True)

    cond = data["conditions"]
    run_label = f"{data['run_name']}: qs_set={cond['qs_setpoint']}, T={cond['T_induced']}°C"

    # Get predictions at sample times for metrics
    times_sample, solution_sample = solve_ode(params, data, dense_output=False)

    metrics = {}
    if show_metrics:
        X_pred = solution_sample[:, 0]
        P_sol_pred = solution_sample[:, 1]
        P_IB_pred = solution_sample[:, 2]

        metrics['R²_X'] = compute_r_squared(X_pred, data["X_obs"])
        metrics['R²_P_sol'] = compute_r_squared(P_sol_pred, data["P_sol_obs"])
        metrics['R²_P_IB'] = compute_r_squared(P_IB_pred, data["P_IB_obs"])
        metrics['Loss'] = compute_loss_single(params, data)

    # Biomass
    ax = ax_row[0]
    ax.plot(times_dense, solution[:, 0], 'b-', linewidth=2, label='Model (prediction)')
    ax.scatter(data["sample_times"], data["X_obs"], c='blue', s=60,
               zorder=5, edgecolors='black', label='Observed', marker='o', alpha=0.8)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Biomass [g/L]')
    title = f'{run_label}\nBiomass (X)'
    if show_metrics and 'R²_X' in metrics:
        title += f'\nR² = {metrics["R²_X"]:.3f}'
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Soluble Protein
    ax = ax_row[1]
    ax.plot(times_dense, solution[:, 1], 'g-', linewidth=2, label='Model (prediction)')
    valid = ~np.isnan(data["P_sol_obs"])
    ax.scatter(data["sample_times"][valid], data["P_sol_obs"][valid],
               c='green', s=60, zorder=5, edgecolors='black', label='Observed', marker='o', alpha=0.8)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Protein [g/L]')
    title = 'Soluble Protein (P_sol)'
    if show_metrics and 'R²_P_sol' in metrics:
        title += f'\nR² = {metrics["R²_P_sol"]:.3f}'
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # IB
    ax = ax_row[2]
    ax.plot(times_dense, solution[:, 2], 'r-', linewidth=2, label='Model (prediction)')
    valid = ~np.isnan(data["P_IB_obs"])
    ax.scatter(data["sample_times"][valid], data["P_IB_obs"][valid],
               c='red', s=60, zorder=5, edgecolors='black', label='Observed', marker='o', alpha=0.8)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Protein [g/L]')
    title = 'Inclusion Bodies (P_IB)'
    if show_metrics and 'R²_P_IB' in metrics:
        title += f'\nR² = {metrics["R²_P_IB"]:.3f}'
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    return metrics


def test_single_validation_run(run_name: str = "ferm_05"):
    """Test the model on a single validation run."""
    print("=" * 60)
    print(f"VALIDATION TEST: {run_name}")
    print("=" * 60)
    print("\nUsing PRE-FITTED parameters from hybrid_v4_fit.txt:")
    print("-" * 60)
    for name, value in zip(PARAM_NAMES, FITTED_PARAMS):
        print(f"  {name:12s} = {value:.6f}")
    print("-" * 60)

    # Load the validation data
    print(f"\nLoading {run_name} data...")
    data = prepare_full_run_data(run_name)
    print(f"  Time: {data['t0']:.1f} - {data['t1']:.1f} h")
    print(f"  Samples: {len(data['sample_times'])}")
    print(f"  DOE setpoint: qs={data['conditions']['qs_setpoint']}, T={data['conditions']['T_induced']}°C")

    # Compute loss
    loss = compute_loss_single(FITTED_PARAMS, data)
    print(f"\nValidation Loss: {loss:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = plot_validation_run(FITTED_PARAMS, data, axes, show_metrics=True)
    plt.suptitle(f"Validation Test: {run_name} (qs={data['conditions']['qs_setpoint']}, T={data['conditions']['T_induced']}°C)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = RESULTS_PATH / f"validation_{run_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

    return fig, metrics, loss


def test_all_triplicates():
    """Test the model on ALL triplicate runs."""
    print("=" * 60)
    print("VALIDATION TEST: ALL REPLICATES (ferm_05, ferm_06, ferm_07)")
    print("=" * 60)
    print("\nUsing PRE-FITTED parameters from hybrid_v4_fit.txt:")
    print("-" * 60)
    for name, value in zip(PARAM_NAMES, FITTED_PARAMS):
        print(f"  {name:12s} = {value:.6f}")
    print("-" * 60)

    triplicate_runs = ["ferm_05", "ferm_06", "ferm_07"]
    data_list = []
    metrics_list = []

    for run_name in triplicate_runs:
        print(f"\nLoading {run_name} data...")
        try:
            data = prepare_full_run_data(run_name)
            print(f"  Time: {data['t0']:.1f} - {data['t1']:.1f} h")
            print(f"  Samples: {len(data['sample_times'])}")
            data_list.append(data)
        except Exception as e:
            print(f"  Error: {e}")

    # Plot all three
    n_runs = len(data_list)
    fig, axes = plt.subplots(n_runs, 3, figsize=(15, 4*n_runs))

    for i, data in enumerate(data_list):
        metrics = plot_validation_run(FITTED_PARAMS, data, axes[i], show_metrics=True)
        metrics_list.append(metrics)
        loss = metrics.get('Loss', np.nan)
        print(f"\n{data['run_name']}:")
        print(f"  R²_X     = {metrics['R²_X']:.3f}")
        print(f"  R²_P_sol = {metrics['R²_P_sol']:.3f}")
        print(f"  R²_P_IB  = {metrics['R²_P_IB']:.3f}")
        print(f"  Loss     = {loss:.4f}")

    plt.suptitle(f"Validation Test: Triplicate Setpoint (qs=0.4, T=30°C)\nModel fitted on ferm_01-04 only",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = RESULTS_PATH / "validation_all_triplicates.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (across triplicates)")
    print("=" * 60)
    for key in ['R²_X', 'R²_P_sol', 'R²_P_IB', 'Loss']:
        values = [m.get(key, np.nan) for m in metrics_list]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            print(f"  {key:12s}: mean = {np.mean(valid_values):.4f}, std = {np.std(valid_values):.4f}")

    return fig, metrics_list


def compare_with_training_data():
    """Compare validation performance with training performance."""
    print("=" * 60)
    print("COMPARISON: Training vs Validation")
    print("=" * 60)

    # Training runs (fitted on)
    training_runs = ["ferm_01", "ferm_02", "ferm_03", "ferm_04"]
    # Validation runs (not fitted on)
    validation_runs = ["ferm_05", "ferm_06", "ferm_07"]

    results = {'training': [], 'validation': []}

    print("\nTraining data (fitted on):")
    for run_name in training_runs:
        data = prepare_full_run_data(run_name)
        loss = compute_loss_single(FITTED_PARAMS, data)
        results['training'].append(loss)
        print(f"  {run_name}: {loss:.4f}")

    print("\nValidation data (not fitted on):")
    for run_name in validation_runs:
        try:
            data = prepare_full_run_data(run_name)
            loss = compute_loss_single(FITTED_PARAMS, data)
            results['validation'].append(loss)
            print(f"  {run_name}: {loss:.4f}")
        except Exception as e:
            print(f"  {run_name}: Error - {e}")

    print("\n" + "-" * 60)
    print(f"Training loss:    mean = {np.mean(results['training']):.4f}, std = {np.std(results['training']):.4f}")
    print(f"Validation loss:  mean = {np.mean(results['validation']):.4f}, std = {np.std(results['validation']):.4f}")

    # Check for overfitting
    train_mean = np.mean(results['training'])
    val_mean = np.mean(results['validation'])
    if val_mean < train_mean * 1.5:
        print("\n[OK] Model generalizes well (validation loss comparable to training)")
    else:
        print("\n[WARNING] Possible overfitting (validation loss significantly higher)")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    """Run validation tests."""
    import argparse
    parser = argparse.ArgumentParser(description='Validate fitted model on triplicates')
    parser.add_argument('--run', type=str, default='all',
                        choices=['all', 'ferm_05', 'ferm_06', 'ferm_07'],
                        help='Which run(s) to test')
    parser.add_argument('--compare', action='store_true',
                        help='Compare training vs validation performance')
    args = parser.parse_args()

    if args.compare:
        compare_with_training_data()
    elif args.run == 'all':
        test_all_triplicates()
    else:
        test_single_validation_run(args.run)

    plt.show()


if __name__ == "__main__":
    main()
