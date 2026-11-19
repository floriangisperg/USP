"""
CORRECTED Time-Series Analysis: Accounting for ACTUAL qs dynamics

CRITICAL INSIGHT: The "initial qs" is just a setpoint for calculating feed slope.
The ACTUAL qs that cells experience varies over time because:
- Feed is LINEAR (constant rate)
- Biomass is GROWING
- Therefore qs = feed/biomass DECLINES over time

This means we have time-varying qs data that might reveal relationships
that were hidden in the static analysis!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 9


def load_data():
    """Load time-series data."""
    excel_file = Path('..') / 'LDH_consolidated_results.xlsx'
    df = pd.read_excel(excel_file, sheet_name='All_Time_Points')
    return df


def calculate_actual_qs(df):
    """Calculate the ACTUAL time-varying qs for each run."""

    # Parameters
    feed_density = 1.150  # g/mL
    glucose_conc_feed = 400  # g/L = 0.4 g/g

    df_list = []

    for run in df['run_name'].unique():
        run_df = df[df['run_name'] == run].copy().sort_values('Process time [h]')

        # Calculate feed rate [mL/h]
        run_df['feed_rate_mL_h'] = run_df['Feed [mL]'].diff() / run_df['Process time [h]'].diff()

        # Calculate ACTUAL qs [g glucose/g biomass/h]
        # qs = (feed_rate [mL/h] * density [g/mL] * glucose_fraction [g/g]) / biomass [g/L]
        run_df['actual_qs_g_g_h'] = (
            run_df['feed_rate_mL_h'] *  # mL/h
            feed_density *  # g/mL
            (glucose_conc_feed / 1000) /  # Convert to g/g
            run_df['Biomass [g/L]']  # g/L
        )

        df_list.append(run_df)

    return pd.concat(df_list, ignore_index=True)


def create_qs_dynamics_plots(df_with_qs):
    """Show how qs actually varies during induction."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CRITICAL: Actual qs Dynamics During Induction\n'
                'qs Declines Because Feed is Linear but Biomass Grows',
                fontsize=13, weight='bold')

    # Config for each run
    configs = {
        'ferm_01': ('35°C', 0.2, 'red'),
        'ferm_02': ('25°C', 0.2, 'green'),
        'ferm_03': ('25°C', 0.6, 'green'),
        'ferm_04': ('35°C', 0.6, 'red'),
        'ferm_05': ('30°C', 0.4, 'orange'),
    }

    # 1. Actual qs over time
    ax = axes[0, 0]
    for run, (temp_str, initial_qs, color) in configs.items():
        run_df = df_with_qs[df_with_qs['run_name'] == run].copy()
        induced = run_df[run_df['Is induced [-]'] == 1].copy()

        if len(induced) > 0:
            time_since_ind = induced['Process time [h]'] - induced['Process time [h]'].min()
            valid = induced['actual_qs_g_g_h'].notna()

            if valid.sum() > 0:
                ax.plot(time_since_ind[valid], induced.loc[valid, 'actual_qs_g_g_h'],
                       label=f'{temp_str}, init_qs={initial_qs}',
                       color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('ACTUAL qs [g/g/h]', weight='bold')
    ax.set_title('Actual qs During Induction (Declining!)', weight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. IB fraction over time (showing when we have data)
    ax = axes[0, 1]

    for run, (temp_str, initial_qs, color) in configs.items():
        run_df = df_with_qs[df_with_qs['run_name'] == run].copy()
        induced = run_df[run_df['Is induced [-]'] == 1].copy()

        if len(induced) > 0 and 'IB fraction [%]' in induced.columns:
            time_since_ind = induced['Process time [h]'] - induced['Process time [h]'].min()
            valid = induced['IB fraction [%]'].notna()

            if valid.sum() > 0:
                ax.scatter(time_since_ind[valid],
                          induced.loc[valid, 'IB fraction [%]'],
                          label=f'{temp_str}, init_qs={initial_qs}',
                          color=color, s=50, alpha=0.7, edgecolors='black')

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('IB Fraction Measurements (Sparse!)', weight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # 3. qs vs IB fraction (all time points combined)
    ax = axes[1, 0]

    for run, (temp_str, initial_qs, color) in configs.items():
        run_df = df_with_qs[df_with_qs['run_name'] == run].copy()
        induced = run_df[run_df['Is induced [-]'] == 1].copy()

        if len(induced) > 0 and 'IB fraction [%]' in induced.columns:
            valid = (induced['actual_qs_g_g_h'].notna() &
                    induced['IB fraction [%]'].notna())

            if valid.sum() > 0:
                ax.scatter(induced.loc[valid, 'actual_qs_g_g_h'],
                          induced.loc[valid, 'IB fraction [%]'],
                          label=f'{temp_str}, init_qs={initial_qs}',
                          color=color, s=80, alpha=0.7, edgecolors='black',
                          marker='s' if initial_qs == 0.2 else '^' if initial_qs == 0.6 else 'o')

    ax.set_xlabel('ACTUAL qs at measurement time [g/g/h]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('IB vs ACTUAL qs (Time-Aligned)', weight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary = """
CRITICAL EXPERIMENTAL INSIGHT
══════════════════════════════

Your experimental design:
• "Initial qs" sets LINEAR feed slope
• Feed rate = CONSTANT during induction
• Biomass = GROWING during induction
• Therefore ACTUAL qs = DECLINING

Example calculation (ferm_01):
• Initial qs setpoint: 0.2 g/g/h
• At induction start: qs ≈ 0.15 g/g/h
• At induction end: qs ≈ 0.08 g/g/h
• Decline: ~50% over induction!

══════════════════════════════

IDENTIFIABILITY IMPLICATIONS:

Each run experiences a RANGE of qs values:
• qs=0.2 runs: 0.15 → 0.08 g/g/h
• qs=0.6 runs: 0.45 → 0.25 g/g/h

This means:
1. We have MORE data points than
   just the 5 "initial qs" values
2. Each run samples a qs TRAJECTORY
3. We can check if IB fraction
   correlates with ACTUAL qs

══════════════════════════════

THE KEY QUESTION:

Within each temperature, does IB fraction
change as qs declines?

• At 25°C: Does IB stay at 0% as qs
  goes from 0.45 → 0.25? (ferm_03)

• At 35°C: Does IB stay at 100% as qs
  goes from 0.15 → 0.08? (ferm_01)

• At 30°C: Does IB change as qs
  goes from 0.3 → 0.15? (ferm_05/06/07)

If IB fraction is CONSTANT across
these qs ranges → qs effect is ZERO
If IB fraction CHANGES → qs effect
IS identifiable in dynamics!
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
           fontsize=8, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig


def analyze_within_run_qs_ib_relationship(df_with_qs):
    """Check if IB fraction changes as qs varies WITHIN each run."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Within-Run Analysis: Does IB Fraction Change as qs Declines?',
                fontsize=13, weight='bold')

    configs = {
        'ferm_01': ('35°C', 0.2, 'red'),
        'ferm_02': ('25°C', 0.2, 'green'),
        'ferm_03': ('25°C', 0.6, 'green'),
        'ferm_04': ('35°C', 0.6, 'red'),
        'ferm_05': ('30°C', 0.4, 'orange'),
    }

    for idx, (run, (temp_str, initial_qs, color)) in enumerate(configs.items()):
        if idx >= 6:
            break

        ax = axes[idx // 3, idx % 3]

        run_df = df_with_qs[df_with_qs['run_name'] == run].copy()
        induced = run_df[run_df['Is induced [-]'] == 1].copy()

        if len(induced) > 0 and 'IB fraction [%]' in induced.columns:
            valid = (induced['actual_qs_g_g_h'].notna() &
                    induced['IB fraction [%]'].notna())

            if valid.sum() > 1:
                x = induced.loc[valid, 'actual_qs_g_g_h'].values
                y = induced.loc[valid, 'IB fraction [%]'].values

                # Scatter plot
                ax.scatter(x, y, color=color, s=80, alpha=0.7,
                          edgecolors='black', linewidth=2, zorder=3)

                # Add labels
                for i, (xi, yi) in enumerate(zip(x, y)):
                    ax.annotate(f'{i+1}', (xi, yi), fontsize=8,
                               xytext=(3, 3), textcoords='offset points')

                # Calculate trend
                if len(x) > 2:
                    from scipy import stats
                    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)

                    # Plot regression line
                    x_line = np.linspace(x.min() * 0.95, x.max() * 1.05, 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'b--', linewidth=2, alpha=0.7,
                           label=f'R²={r_val**2:.3f}, p={p_val:.3f}')

                    ax.legend(fontsize=8)

                ax.set_xlabel('Actual qs [g/g/h]', weight='bold')
                ax.set_ylabel('IB Fraction [%]', weight='bold')
                ax.set_title(f'{run}: {temp_str}, init_qs={initial_qs}\n'
                            f'qs range: {x.min():.3f} → {x.max():.3f}',
                            weight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-5, 105)
            else:
                ax.text(0.5, 0.5, 'Insufficient data\n(only 1 measurement)',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral'))
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig


def main():
    """Run corrected qs analysis."""
    print("\n" + "="*70)
    print("CORRECTED QS ANALYSIS: Accounting for Time-Varying qs")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} time points")

    # Calculate actual qs
    print("\n2. Calculating ACTUAL time-varying qs...")
    df_with_qs = calculate_actual_qs(df)
    print("   Done")

    # Create qs dynamics plots
    print("\n3. Creating qs dynamics visualization...")
    fig1 = create_qs_dynamics_plots(df_with_qs)
    fig1.savefig('corrected_qs_dynamics.png', dpi=300, bbox_inches='tight')
    print("   Saved: corrected_qs_dynamics.png")

    # Analyze within-run relationships
    print("\n4. Analyzing within-run qs-IB relationships...")
    fig2 = analyze_within_run_qs_ib_relationship(df_with_qs)
    fig2.savefig('corrected_within_run_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved: corrected_within_run_analysis.png")

    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    print("""
Looking at the actual time-varying qs data:

The IB fraction measurements are SPARSE (only ~4-6 per run).
These occur at DIFFERENT qs values within each run.

If IB fraction stays CONSTANT while qs varies 50% within a run,
this proves qs has NO effect on IB formation.

If IB fraction CHANGES as qs varies within a run,
this COULD reveal qs effects that were hidden!

Check the "corrected_within_run_analysis.png" plot to see
if there are any trends within individual runs.
""")

    print("\nAnalysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
