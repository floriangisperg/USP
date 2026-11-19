"""
Visualization utilities for OptFed mechanistic model.

Provides plotting functions for:
- Model vs data comparison
- Optimization trajectories
- Parity plots for validation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import FermentationRun, ProcessData
from model import ControlProfile, OptFedModel
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_model_vs_data(
    model: OptFedModel,
    run: FermentationRun,
    control: ControlProfile,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot model simulation vs experimental data for a single run.
    
    Shows biomass (X), product (P), and volume (V) over time,
    comparing model predictions with measurements.
    
    Args:
        model: OptFed model with fitted parameters
        run: Fermentation run data
        control: Control profile (feed rate, temperature)
        save_path: Optional path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    df = run.data.copy()
    
    # Get induction time if available
    induction_mask = df.get("induction", pd.Series([0] * len(df))) > 0
    has_induction = induction_mask.any()
    
    # Simulate model
    t_eval = np.linspace(df["time"].min(), df["time"].max(), 500)
    result = model.simulate(
        t_span=(df["time"].min(), df["time"].max()),
        t_eval=t_eval,
        control=control,
        method="LSODA",
    )
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
    
    # Biomass subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_eval, result["X_conc"], 'b-', linewidth=2, label='Model', alpha=0.8)
    if "X" in df.columns:
        X_data = df[df["X"].notna()]
        ax1.plot(X_data["time"], X_data["X"], 'ro', markersize=6, 
                 label='Data', markerfacecolor='white', markeredgewidth=2)
    if has_induction:
        t_ind = df[induction_mask]["time"].min()
        ax1.axvline(t_ind, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Induction')
    ax1.set_ylabel('Biomass X (g/L)')
    ax1.set_title(f'Biomass - {run.run_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Product subplot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_eval, result["P_conc"], 'g-', linewidth=2, label='Model', alpha=0.8)
    if "P" in df.columns:
        P_data = df[df["P"].notna()]
        ax2.plot(P_data["time"], P_data["P"], 'ro', markersize=6,
                 label='Data', markerfacecolor='white', markeredgewidth=2)
    if has_induction:
        ax2.axvline(t_ind, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Induction')
    ax2.set_xlabel('Time (h)')
    ax2.set_ylabel('Product P (g/L)')
    ax2.set_title(f'Product - {run.run_id}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Volume subplot
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(t_eval, result["V"], 'm-', linewidth=2, label='Model', alpha=0.8)
    if "V" in df.columns:
        V_data = df[df["V"].notna()]
        ax3.plot(V_data["time"], V_data["V"] / 1000.0, 'ro', markersize=6,
                 label='Data', markerfacecolor='white', markeredgewidth=2)
    ax3.set_ylabel('Volume V (L)')
    ax3.set_title('Reactor Volume')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Feed rate subplot
    ax4 = fig.add_subplot(gs[1, 1])
    feed_rates = [control.get_feed_rate(t) for t in t_eval]
    ax4.plot(t_eval, feed_rates, 'c-', linewidth=2, label='Feed', alpha=0.8)
    if "feed" in df.columns:
        # Calculate derivative of cumulative feed
        feed_cum = df["feed"].values / 1000.0
        t_feed = df["time"].values
        feed_derivative = np.gradient(feed_cum, t_feed)
        ax4.plot(t_feed, feed_derivative, 'ko', markersize=4,
                 label='Data derivative', markerfacecolor='white', markeredgewidth=1.5, alpha=0.6)
    ax4.set_xlabel('Time (h)')
    ax4.set_ylabel('Feed Rate (L/h)')
    ax4.set_title('Feed Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_parity(
    model: OptFedModel,
    data: ProcessData,
    control_dict: dict[str, ControlProfile],
    variable: str = "X",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create parity plot comparing model predictions vs experimental data.
    
    Args:
        model: OptFed model with fitted parameters
        data: Process data with multiple runs
        control_dict: Dictionary mapping run_id to control profiles
        variable: Variable to plot ("X", "P", or "V")
        save_path: Optional path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    var_labels = {"X": "Biomass (g/L)", "P": "Product (g/L)", "V": "Volume (L)"}
    var_colors = {"X": "blue", "P": "green", "V": "magenta"}
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    all_predicted = []
    all_measured = []
    all_labels = []
    
    for run in data.train_runs:
        if run.run_id not in control_dict:
            continue
        
        df = run.data
        if variable not in df.columns:
            continue
        
        # Get measurements
        data_subset = df[df[variable].notna()]
        if len(data_subset) == 0:
            continue
        
        t_data = data_subset["time"].values
        
        # Simulate model
        result = model.simulate(
            t_span=(df["time"].min(), df["time"].max()),
            t_eval=t_data,
            control=control_dict[run.run_id],
            method="LSODA",
        )
        
        # Get model predictions
        if variable == "X":
            predicted = result["X_conc"]
        elif variable == "P":
            predicted = result["P_conc"]
        elif variable == "V":
            predicted = result["V"] * 1000.0  # Convert to mL
        else:
            continue
        
        all_measured.extend(data_subset[variable].values)
        all_predicted.extend(predicted)
        all_labels.extend([run.run_id] * len(data_subset))
    
    # Convert to arrays
    all_measured = np.array(all_measured)
    all_predicted = np.array(all_predicted)
    
    # Plot parity
    ax.scatter(all_measured, all_predicted, c=var_colors.get(variable, "blue"), 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(all_measured.min(), all_predicted.min())
    max_val = max(all_measured.max(), all_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
            label='Perfect prediction', alpha=0.7)
    
    # Calculate metrics
    residuals = all_predicted - all_measured
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    # Add text box with metrics
    textstr = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {np.corrcoef(all_measured, all_predicted)[0,1]**2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel(f'Measured {var_labels[variable]}')
    ax.set_ylabel(f'Predicted {var_labels[variable]}')
    ax.set_title(f'Parity Plot - {var_labels[variable]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_optimization_trajectory(
    history: List[dict],
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot optimization trajectory showing objective value over iterations.
    
    Args:
        history: List of optimization state dictionaries
        save_path: Optional path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    if not history:
        raise ValueError("Optimization history is empty")
    
    iterations = [h.get("iteration", i) for i, h in enumerate(history)]
    objectives = [h["objective"] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Objective value trajectory
    ax1.plot(iterations, objectives, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization Trajectory')
    ax1.grid(True, alpha=0.3)
    
    # Log scale for better visualization of convergence
    ax2.semilogy(iterations, np.array(objectives) - min(objectives) + 1e-10, 
                 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Objective - Best (log scale)')
    ax2.set_title('Convergence (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_parameter_sensitivity(
    model: OptFedModel,
    param_names: List[str],
    sensitivities: np.ndarray,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot parameter sensitivity analysis results.
    
    Args:
        model: OptFed model
        param_names: List of parameter names
        sensitivities: Array of sensitivity coefficients
        save_path: Optional path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by absolute sensitivity
    idx = np.argsort(np.abs(sensitivities))[::-1]
    sorted_params = [param_names[i] for i in idx]
    sorted_sens = sensitivities[idx]
    
    colors = ['green' if s > 0 else 'red' for s in sorted_sens]
    
    bars = ax.barh(range(len(sorted_params)), sorted_sens, color=colors, 
                    edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(sorted_params)))
    ax.set_yticklabels(sorted_params)
    ax.set_xlabel('Sensitivity Coefficient')
    ax.set_title('Parameter Sensitivity Analysis')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Positive effect'),
        Patch(facecolor='red', edgecolor='black', label='Negative effect'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_multi_run_comparison(
    model: OptFedModel,
    data: ProcessData,
    control_dict: dict[str, ControlProfile],
    variable: str = "X",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot model predictions vs data for multiple runs on a single plot.
    
    Args:
        model: OptFed model with fitted parameters
        data: Process data with multiple runs
        control_dict: Dictionary mapping run_id to control profiles
        variable: Variable to plot ("X" or "P")
        save_path: Optional path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    var_labels = {"X": "Biomass (g/L)", "P": "Product (g/L)"}
    var_colors = plt.cm.tab10(np.linspace(0, 1, len(data.train_runs)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, run in enumerate(data.train_runs):
        if run.run_id not in control_dict:
            continue
        
        df = run.data
        if variable not in df.columns:
            continue
        
        # Get measurements
        data_subset = df[df[variable].notna()]
        if len(data_subset) == 0:
            continue
        
        t_data = data_subset["time"].values
        
        # Simulate model
        t_eval = np.linspace(df["time"].min(), df["time"].max(), 200)
        result = model.simulate(
            t_span=(df["time"].min(), df["time"].max()),
            t_eval=t_eval,
            control=control_dict[run.run_id],
            method="LSODA",
        )
        
        # Get model predictions
        if variable == "X":
            predicted = result["X_conc"]
        elif variable == "P":
            predicted = result["P_conc"]
        else:
            continue
        
        # Plot model prediction
        ax.plot(t_eval, predicted, color=var_colors[i], linewidth=2, 
                label=f'{run.run_id} (model)', alpha=0.8)
        
        # Plot data
        ax.plot(data_subset["time"], data_subset[variable].values, 
                'o', color=var_colors[i], markersize=6,
                markerfacecolor='white', markeredgewidth=2,
                label=f'{run.run_id} (data)')
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(var_labels[variable])
    ax.set_title(f'Multi-Run Comparison - {var_labels[variable]}')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_validation_summary(
    model: OptFedModel,
    train_data: ProcessData,
    test_data: ProcessData,
    control_dict: dict[str, ControlProfile],
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a summary plot showing training and validation performance.
    
    Args:
        model: OptFed model with fitted parameters
        train_data: Training data
        test_data: Test/validation data
        control_dict: Dictionary mapping run_id to control profiles
        save_path: Optional path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Training parity plot
    ax1 = fig.add_subplot(gs[0, 0])
    plot_parity_on_ax(ax1, model, train_data, control_dict, "X", "Training - Biomass")
    
    # Test parity plot
    ax2 = fig.add_subplot(gs[0, 1])
    plot_parity_on_ax(ax2, model, test_data, control_dict, "X", "Validation - Biomass")
    
    # Training residuals
    ax3 = fig.add_subplot(gs[1, 0])
    plot_residuals_on_ax(ax3, model, train_data, control_dict, "X")
    
    # Test residuals
    ax4 = fig.add_subplot(gs[1, 1])
    plot_residuals_on_ax(ax4, model, test_data, control_dict, "X")
    
    plt.suptitle('Model Validation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_parity_on_ax(ax, model, data, control_dict, variable, title):
    """Helper function to create parity plot on given axes."""
    all_measured = []
    all_predicted = []
    
    for run in data.train_runs + data.test_runs:
        if run.run_id not in control_dict:
            continue
        
        df = run.data
        if variable not in df.columns:
            continue
        
        data_subset = df[df[variable].notna()]
        if len(data_subset) == 0:
            continue
        
        t_data = data_subset["time"].values
        result = model.simulate(
            t_span=(df["time"].min(), df["time"].max()),
            t_eval=t_data,
            control=control_dict[run.run_id],
            method="LSODA",
        )
        
        predicted = result["X_conc"] if variable == "X" else result["P_conc"]
        all_measured.extend(data_subset[variable].values)
        all_predicted.extend(predicted)
    
    all_measured = np.array(all_measured)
    all_predicted = np.array(all_predicted)
    
    ax.scatter(all_measured, all_predicted, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    min_val = min(all_measured.min(), all_predicted.min())
    max_val = max(all_measured.max(), all_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7)
    
    residuals = all_predicted - all_measured
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = np.corrcoef(all_measured, all_predicted)[0, 1]**2
    
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{title}\nRMSE: {rmse:.3f}, R²: {r2:.3f}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def plot_residuals_on_ax(ax, model, data, control_dict, variable):
    """Helper function to create residuals plot on given axes."""
    all_measured = []
    all_residuals = []
    
    for run in data.train_runs + data.test_runs:
        if run.run_id not in control_dict:
            continue
        
        df = run.data
        if variable not in df.columns:
            continue
        
        data_subset = df[df[variable].notna()]
        if len(data_subset) == 0:
            continue
        
        t_data = data_subset["time"].values
        result = model.simulate(
            t_span=(df["time"].min(), df["time"].max()),
            t_eval=t_data,
            control=control_dict[run.run_id],
            method="LSODA",
        )
        
        predicted = result["X_conc"] if variable == "X" else result["P_conc"]
        residuals = predicted - data_subset[variable].values
        
        all_measured.extend(data_subset[variable].values)
        all_residuals.extend(residuals)
    
    ax.scatter(all_measured, all_residuals, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Measured Value')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Measured')
    ax.grid(True, alpha=0.3)
