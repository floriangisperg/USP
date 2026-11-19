"""
Visualization Tools for Mechanistic IB Model

Creates visualizations showing:
1. Temperature effect on IB fraction at different qs levels
2. qs effect on IB fraction at different temperatures
3. 2D surface response of IB formation
4. Component breakdown (synthesis vs aggregation)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from mechanistic_ib_model import (
    phi_ib_mechanistic,
    phi_synthesis_overload,
    phi_thermal_aggregation,
    DEFAULT_PARAMS,
    RESULTS_PATH
)

RESULTS_PATH.mkdir(exist_ok=True)


def plot_ib_temperature_response(params: np.ndarray, qs_values: list,
                                  save_path: Optional[Path] = None):
    """
    Plot IB fraction vs temperature at different qs levels.

    Shows how temperature affects IB partitioning, and whether qs
    modulates this effect.
    """
    k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs = params[4:10]

    T_range = np.linspace(20, 40, 200)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'green', 'red']
    for i, qs in enumerate(qs_values):
        IB_frac = [phi_ib_mechanistic(T, qs, *params[4:10]) for T in T_range]
        IB_frac = np.array(IB_frac)
        ax.plot(T_range, IB_frac * 100, color=colors[i], linewidth=2.5,
                label=f'qs = {qs} g/g/h')

    # Add observed data points
    ax.scatter([25, 25], [0, 0], c='blue', s=100, zorder=5,
               edgecolors='black', marker='o', label='Observed (25°C)')
    ax.scatter([30], [13], c='green', s=150, zorder=5,
               edgecolors='black', marker='s', label='Observed (30°C)')
    ax.scatter([35, 35], [100, 100], c='red', s=100, zorder=5,
               edgecolors='black', marker='^', label='Observed (35°C)')

    ax.set_xlabel('Temperature [°C]', fontsize=12)
    ax.set_ylabel('IB Fraction [%]', fontsize=12)
    ax.set_title('IB Temperature Response at Different qs Levels\n(Mechanistic Model)',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(22, 38)

    # Add reference lines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig


def plot_ib_qs_response(params: np.ndarray, T_values: list,
                        save_path: Optional[Path] = None):
    """
    Plot IB fraction vs qs at different temperatures.

    Shows how qs affects IB partitioning, and whether this depends
    on temperature.
    """
    qs_range = np.linspace(0.1, 0.8, 200)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'green', 'red']
    for i, T in enumerate(T_values):
        IB_frac = [phi_ib_mechanistic(T, qs, *params[4:10]) for qs in qs_range]
        IB_frac = np.array(IB_frac)
        ax.plot(qs_range, IB_frac * 100, color=colors[i], linewidth=2.5,
                label=f'T = {T}°C')

    # Add observed data points (qs setpoints)
    ax.scatter([0.2], [0], c='blue', s=100, zorder=5,
               edgecolors='black', marker='o', label='Observed (T=25°C)')
    ax.scatter([0.4], [13], c='green', s=150, zorder=5,
               edgecolors='black', marker='s', label='Observed (T=30°C)')
    ax.scatter([0.2, 0.6], [100, 100], c='red', s=100, zorder=5,
               edgecolors='black', marker='^', label='Observed (T=35°C)')

    ax.set_xlabel('Specific Substrate Uptake (qs) [g/g/h]', fontsize=12)
    ax.set_ylabel('IB Fraction [%]', fontsize=12)
    ax.set_title('IB qs Response at Different Temperatures\n(Mechanistic Model)',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0.1, 0.7)

    # Add reference lines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig


def plot_ib_surface(params: np.ndarray, save_path: Optional[Path] = None):
    """
    Create 2D surface plot of IB fraction as function of T and qs.
    """
    qs_range = np.linspace(0.1, 0.7, 100)
    T_range = np.linspace(22, 38, 100)

    qs_grid, T_grid = np.meshgrid(qs_range, T_range)
    IB_grid = np.zeros_like(qs_grid)

    for i in range(len(T_range)):
        for j in range(len(qs_range)):
            IB_grid[i, j] = phi_ib_mechanistic(T_grid[i, j], qs_grid[i, j],
                                               *params[4:10])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour plot
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    contour = ax.contourf(qs_grid, T_grid, IB_grid * 100, levels=levels,
                          cmap='RdYlGn_r', extend='both')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('IB Fraction [%]', fontsize=11)

    # Add contour lines
    ax.contour(qs_grid, T_grid, IB_grid * 100, levels=levels,
               colors='black', alpha=0.3, linewidths=0.5)

    # Add observed data points
    ax.scatter([0.2], [25], c='white', s=150, zorder=5,
               edgecolors='black', linewidth=2, marker='o')
    ax.text(0.22, 25.5, '0%', fontsize=10, fontweight='bold')

    ax.scatter([0.6], [25], c='white', s=150, zorder=5,
               edgecolors='black', linewidth=2, marker='o')
    ax.text(0.62, 25.5, '0%', fontsize=10, fontweight='bold')

    ax.scatter([0.4], [30], c='white', s=150, zorder=5,
               edgecolors='black', linewidth=2, marker='s')
    ax.text(0.42, 30.5, '~13%', fontsize=10, fontweight='bold')

    ax.scatter([0.2], [35], c='white', s=150, zorder=5,
               edgecolors='black', linewidth=2, marker='^')
    ax.text(0.22, 35.5, '100%', fontsize=10, fontweight='bold')

    ax.scatter([0.6], [35], c='white', s=150, zorder=5,
               edgecolors='black', linewidth=2, marker='^')
    ax.text(0.62, 35.5, '100%', fontsize=10, fontweight='bold')

    ax.set_xlabel('Specific Substrate Uptake (qs) [g/g/h]', fontsize=12)
    ax.set_ylabel('Temperature [°C]', fontsize=12)
    ax.set_title('IB Formation Surface Response\n(Mechanistic Model: T × qs Interaction)',
                 fontweight='bold', fontsize=13)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig


def plot_component_breakdown(params: np.ndarray, T_values: list, qs_values: list,
                             save_path: Optional[Path] = None):
    """
    Plot breakdown of IB fraction into synthesis and aggregation components.
    """
    k_syn, K_syn, k_agg, T_ref, Ea_over_R, alpha_qs = params[4:10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Temperature effect at fixed qs
    ax = axes[0]
    qs_fixed = 0.4  # Center point
    T_range = np.linspace(22, 38, 200)

    phi_syn = [phi_synthesis_overload(qs_fixed, k_syn, K_syn) for T in T_range]
    phi_agg = [phi_thermal_aggregation(T, qs_fixed, k_agg, T_ref, Ea_over_R, alpha_qs)
               for T in T_range]
    phi_total = [s + a for s, a in zip(phi_syn, phi_agg)]

    ax.plot(T_range, np.array(phi_syn) * 100, 'b--', linewidth=2, label='Synthesis overload')
    ax.plot(T_range, np.array(phi_agg) * 100, 'r-', linewidth=2, label='Thermal aggregation')
    ax.plot(T_range, np.array(phi_total) * 100, 'k-', linewidth=3, label='Total')

    ax.set_xlabel('Temperature [°C]', fontsize=12)
    ax.set_ylabel('IB Fraction [%]', fontsize=12)
    ax.set_title(f'IB Component Breakdown (qs = {qs_fixed} g/g/h)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Plot 2: qs effect at fixed T
    ax = axes[1]
    T_fixed = 30.0  # Center point
    qs_range = np.linspace(0.1, 0.7, 200)

    phi_syn = [phi_synthesis_overload(qs, k_syn, K_syn) for qs in qs_range]
    phi_agg = [phi_thermal_aggregation(T_fixed, qs, k_agg, T_ref, Ea_over_R, alpha_qs)
               for qs in qs_range]
    phi_total = [s + a for s, a in zip(phi_syn, phi_agg)]

    ax.plot(qs_range, np.array(phi_syn) * 100, 'b--', linewidth=2, label='Synthesis overload')
    ax.plot(qs_range, np.array(phi_agg) * 100, 'r-', linewidth=2, label='Thermal aggregation')
    ax.plot(qs_range, np.array(phi_total) * 100, 'k-', linewidth=3, label='Total')

    ax.set_xlabel('Specific Substrate Uptake (qs) [g/g/h]', fontsize=12)
    ax.set_ylabel('IB Fraction [%]', fontsize=12)
    ax.set_title(f'IB Component Breakdown (T = {T_fixed}°C)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.suptitle('Mechanistic IB Model: Component Breakdown',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig


def main():
    """Generate all visualizations."""
    print("Generating visualizations for mechanistic IB model...")
    print("=" * 60)

    # Use default parameters for now (or load fitted parameters)
    params = DEFAULT_PARAMS
    print("Using default parameters (fit will update these)")

    print("\n[1] Temperature response at different qs levels...")
    plot_ib_temperature_response(params, qs_values=[0.2, 0.4, 0.6],
                                 save_path=RESULTS_PATH / "ib_temp_response.png")

    print("\n[2] qs response at different temperatures...")
    plot_ib_qs_response(params, T_values=[25, 30, 35],
                        save_path=RESULTS_PATH / "ib_qs_response.png")

    print("\n[3] 2D surface response...")
    plot_ib_surface(params, save_path=RESULTS_PATH / "ib_surface.png")

    print("\n[4] Component breakdown...")
    plot_component_breakdown(params, T_values=[25, 30, 35], qs_values=[0.2, 0.4, 0.6],
                             save_path=RESULTS_PATH / "ib_components.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
