"""
Analysis of LDH Inclusion Body vs Soluble Expression

This script analyzes the factors driving the partitioning of LDH protein
between soluble fraction and inclusion bodies (IBs) in E. coli fermentations.

Experimental design:
- Temperature: 25°C, 30°C (center), 35°C
- Initial qs: 0.2, 0.4 (center), 0.6 g/g/h
- 7 runs total (2² factorial design + 3 center points)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import re
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150


def parse_induction_summary(summary_text):
    """Parse induction_summary string to extract experimental conditions."""
    # Extract temperature
    temp_match = re.search(r'T=(\d+)', summary_text)
    temp_c = float(temp_match.group(1)) if temp_match else np.nan

    # Extract initial qs
    qs_match = re.search(r'qs=([\d.]+)', summary_text)
    qs_initial = float(qs_match.group(1)) if qs_match else np.nan

    # Extract induction time
    time_match = re.search(r'(\d+)h', summary_text)
    induction_time_h = float(time_match.group(1)) if time_match else np.nan

    return temp_c, qs_initial, induction_time_h


def load_run_configs():
    """Load experimental conditions from run_config.yaml files."""
    configs = {}
    for run_dir in Path('..').glob('ferm_*/'):
        config_file = run_dir / 'run_config.yaml'
        if config_file.exists():
            run_name = run_dir.name
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Parse induction summary
            induction_summary = config.get('metadata', {}).get('induction_summary', '')
            temp_c, qs_initial, induction_time = parse_induction_summary(induction_summary)

            configs[run_name] = {
                'temperature_C': temp_c,
                'qs_initial_g_g_h': qs_initial,
                'induction_time_h': induction_time,
                'induction_summary': induction_summary
            }

    df = pd.DataFrame.from_dict(configs, orient='index')
    df.index.name = 'run_name'  # Set the index name explicitly
    return df


def load_consolidated_data():
    """Load the consolidated fermentation data."""
    excel_file = Path('..') / 'LDH_consolidated_results.xlsx'
    df = pd.read_excel(excel_file, sheet_name='All_Time_Points')

    # Fix column encoding issues if present
    df.columns = [col.replace('[°C]', '[°C]').replace('[°C]', '[°C]')
                  if isinstance(col, str) else col for col in df.columns]

    return df


def calculate_run_summaries(df):
    """Calculate summary statistics for each run."""
    summaries = []

    for run_name in df['run_name'].unique():
        run_df = df[df['run_name'] == run_name].copy()

        # Filter to induced period only (after induction starts)
        induced_df = run_df[run_df['Is induced [-]'] == 1]

        summary = {'run_name': run_name}

        if len(induced_df) > 0:
            # Final values (last sample point with protein data)
            protein_data = induced_df.dropna(subset=['Protein total calculated [g/L]'])
            if len(protein_data) > 0:
                final = protein_data.iloc[-1]

                summary['final_protein_soluble_g_L'] = final['Protein soluble [g/L]']
                summary['final_protein_IB_g_L'] = final['Protein IB [g/L]']
                summary['final_protein_total_g_L'] = final['Protein total calculated [g/L]']
                summary['final_soluble_fraction_pct'] = final['Soluble fraction [%]']
                summary['final_IB_fraction_pct'] = final['IB fraction [%]']
                summary['final_biomass_g_L'] = final['Biomass [g/L]']

                # Yields
                if 'Protein total yield [g/g]' in induced_df.columns:
                    final_yield = induced_df['Protein total yield [g/g]'].dropna()
                    if len(final_yield) > 0:
                        summary['final_yield_g_g'] = final_yield.iloc[-1]

                # Maximum concentrations during induction
                summary['max_protein_soluble_g_L'] = induced_df['Protein soluble [g/L]'].max()
                summary['max_protein_IB_g_L'] = induced_df['Protein IB [g/L]'].max()
                summary['max_biomass_g_L'] = induced_df['Biomass [g/L]'].max()

                # Calculate volumetric productivity (total protein / induction time)
                induction_duration = induced_df['Process time [h]'].max() - induced_df['Process time [h]'].min()
                summary['induction_duration_h'] = induction_duration
                summary['volumetric_productivity_g_L_h'] = (
                    summary['final_protein_total_g_L'] / induction_duration
                    if induction_duration > 0 else np.nan
                )

                # Calculate specific productivity (yield / induction time)
                if 'final_yield_g_g' in summary:
                    summary['specific_productivity_g_g_h'] = (
                        summary['final_yield_g_g'] / induction_duration
                        if induction_duration > 0 else np.nan
                    )

                # Average temperature during induction
                summary['avg_temperature_C'] = induced_df['Temperature [°C]'].mean()

                # Final feed time (cumulative feeding time)
                summary['final_feed_time_h'] = induced_df['Feed time [h]'].max()

                # Average pO2 during induction
                if 'pO2 [%]' in induced_df.columns:
                    summary['avg_pO2_pct'] = induced_df['pO2 [%]'].mean()

        summaries.append(summary)

    return pd.DataFrame(summaries).set_index('run_name')


def calculate_derived_parameters(df, configs):
    """Calculate additional derived physiological parameters."""

    # For each run, calculate time-dependent parameters
    results = []

    for run_name in df['run_name'].unique():
        run_df = df[df['run_name'] == run_name].copy().sort_values('Process time [h]')

        if run_name in configs.index:
            config = configs.loc[run_name]

            # Calculate instantaneous feed rate (mL/h)
            run_df['feed_rate_mL_h'] = run_df['Feed [mL]'].diff() / run_df['Process time [h]'].diff()
            run_df['feed_rate_g_L_h'] = (
                run_df['feed_rate_mL_h'] * 1.150 /  # density of feed
                (run_df['Volume corrected [mL]'] / 1000)  # convert to L
            )

            # Calculate specific substrate uptake rate (qs) assuming glucose-only feed
            # qs = (feed_rate * glucose_concentration) / biomass
            # Feed is 400 g/L glucose (standard DeLisa feed)
            glucose_conc_feed = 400  # g/L
            run_df['qs_g_g_h'] = (
                run_df['feed_rate_g_L_h'] * glucose_conc_feed /
                run_df['Biomass [g/L]']
            )

            # Store summary for induced period
            induced = run_df[run_df['Is induced [-]'] == 1]
            if len(induced) > 0:
                results.append({
                    'run_name': run_name,
                    'avg_feed_rate_induced_mL_h': induced['feed_rate_mL_h'].mean(),
                    'avg_qs_induced_g_g_h': induced['qs_g_g_h'].mean(),
                    'initial_qs_from_config': config['qs_initial_g_g_h'],
                    'temp_from_config': config['temperature_C'],
                })

    return pd.DataFrame(results).set_index('run_name')


def create_experimental_design_plot(configs, summaries):
    """Create experimental design matrix plot with outcomes."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare data
    plot_data = configs.join(summaries)

    # Create scatter plot
    scatter = ax.scatter(
        plot_data['temperature_C'],
        plot_data['qs_initial_g_g_h'],
        s=plot_data['final_protein_total_g_L'] * 100,  # Size by total protein
        c=plot_data['final_IB_fraction_pct'],
        cmap='RdYlGn_r',  # Red=IB, Green=Soluble
        edgecolors='black',
        linewidths=2,
        alpha=0.8,
        vmin=0, vmax=100
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('IB Fraction [%]', rotation=270, labelpad=20)

    # Add run labels
    for run_name, row in plot_data.iterrows():
        ax.annotate(
            run_name,
            (row['temperature_C'], row['qs_initial_g_g_h']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            weight='bold'
        )

    # Design space visualization
    ax.set_xlabel('Temperature upon Induction [°C]', fontsize=12, weight='bold')
    ax.set_ylabel('Initial qs [g/g/h]', fontsize=12, weight='bold')
    ax.set_title('Experimental Design Matrix: IB Formation Patterns\n'
                 'Bubble size = Total protein concentration, Color = IB fraction',
                 fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(20, 40)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    return fig


def create_ib_fraction_vs_params_plot(configs, summaries):
    """Create multi-panel plot of IB fraction vs key parameters."""
    plot_data = configs.join(summaries).reset_index()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. IB fraction vs Temperature
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(plot_data['temperature_C'], plot_data['final_IB_fraction_pct'],
                s=100, alpha=0.7, edgecolors='black')
    for idx, row in plot_data.iterrows():
        ax1.annotate(row['run_name'], (row['temperature_C'], row['final_IB_fraction_pct']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)

    # Add regression line
    temp_valid = plot_data[['temperature_C', 'final_IB_fraction_pct']].dropna()
    if len(temp_valid) > 2:
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            temp_valid['temperature_C'], temp_valid['final_IB_fraction_pct']
        )
        x_line = np.linspace(temp_valid['temperature_C'].min(), temp_valid['temperature_C'].max(), 100)
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, 'r--', alpha=0.7,
                label=f'R² = {r_val**2:.3f}, p = {p_val:.3f}')
        ax1.legend()

    ax1.set_xlabel('Temperature [°C]', weight='bold')
    ax1.set_ylabel('Final IB Fraction [%]', weight='bold')
    ax1.set_title('IB Fraction vs Temperature', weight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. IB fraction vs Initial qs
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(plot_data['qs_initial_g_g_h'], plot_data['final_IB_fraction_pct'],
                s=100, alpha=0.7, edgecolors='black')
    for idx, row in plot_data.iterrows():
        ax2.annotate(row['run_name'], (row['qs_initial_g_g_h'], row['final_IB_fraction_pct']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)

    qs_valid = plot_data[['qs_initial_g_g_h', 'final_IB_fraction_pct']].dropna()
    if len(qs_valid) > 2:
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            qs_valid['qs_initial_g_g_h'], qs_valid['final_IB_fraction_pct']
        )
        x_line = np.linspace(qs_valid['qs_initial_g_g_h'].min(), qs_valid['qs_initial_g_g_h'].max(), 100)
        y_line = slope * x_line + intercept
        ax2.plot(x_line, y_line, 'r--', alpha=0.7,
                label=f'R² = {r_val**2:.3f}, p = {p_val:.3f}')
        ax2.legend()

    ax2.set_xlabel('Initial qs [g/g/h]', weight='bold')
    ax2.set_ylabel('Final IB Fraction [%]', weight='bold')
    ax2.set_title('IB Fraction vs Initial qs', weight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. IB fraction vs Volumetric Productivity
    if 'volumetric_productivity_g_L_h' in plot_data.columns:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(plot_data['volumetric_productivity_g_L_h'], plot_data['final_IB_fraction_pct'],
                    s=100, alpha=0.7, edgecolors='black')
        for idx, row in plot_data.iterrows():
            ax3.annotate(row['run_name'], (row['volumetric_productivity_g_L_h'], row['final_IB_fraction_pct']),
                        xytext=(3, 3), textcoords='offset points', fontsize=8)

        prod_valid = plot_data[['volumetric_productivity_g_L_h', 'final_IB_fraction_pct']].dropna()
        if len(prod_valid) > 2:
            slope, intercept, r_val, p_val, std_err = stats.linregress(
                prod_valid['volumetric_productivity_g_L_h'], prod_valid['final_IB_fraction_pct']
            )
            x_line = np.linspace(prod_valid['volumetric_productivity_g_L_h'].min(),
                                prod_valid['volumetric_productivity_g_L_h'].max(), 100)
            y_line = slope * x_line + intercept
            ax3.plot(x_line, y_line, 'r--', alpha=0.7,
                    label=f'R² = {r_val**2:.3f}, p = {p_val:.3f}')
            ax3.legend()

        ax3.set_xlabel('Volumetric Productivity [g/L/h]', weight='bold')
        ax3.set_ylabel('Final IB Fraction [%]', weight='bold')
        ax3.set_title('IB Fraction vs Productivity', weight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Total protein vs Temperature
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(plot_data['run_name'], plot_data['final_protein_total_g_L'],
                   color=['red' if ib > 50 else 'green' for ib in plot_data['final_IB_fraction_pct']],
                   alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Run', weight='bold')
    ax4.set_ylabel('Total Protein [g/L]', weight='bold')
    ax4.set_title('Total Protein by Run (Red=IB>50%, Green=Soluble>50%)', weight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Soluble vs IB stacked bar
    ax5 = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(plot_data))
    ax5.bar(x_pos, plot_data['final_protein_IB_g_L'], label='IB',
            color='red', alpha=0.7, edgecolor='black')
    ax5.bar(x_pos, plot_data['final_protein_soluble_g_L'], bottom=plot_data['final_protein_IB_g_L'],
            label='Soluble', color='green', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Run', weight='bold')
    ax5.set_ylabel('Protein Concentration [g/L]', weight='bold')
    ax5.set_title('Protein Partitioning: Soluble vs IB', weight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(plot_data['run_name'], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Yield comparison
    if 'final_yield_g_g' in plot_data.columns:
        ax6 = fig.add_subplot(gs[1, 2])
        bars = ax6.bar(plot_data['run_name'], plot_data['final_yield_g_g'],
                       color=['red' if ib > 50 else 'green' for ib in plot_data['final_IB_fraction_pct']],
                       alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Run', weight='bold')
        ax6.set_ylabel('Final Yield [g/g]', weight='bold')
        ax6.set_title('Protein Yield by Run', weight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Inclusion Body Formation Analysis', fontsize=16, weight='bold', y=0.995)

    return fig


def create_correlation_heatmap(plot_data):
    """Create correlation heatmap for key parameters."""
    # Select relevant columns
    cols_of_interest = [
        'temperature_C',
        'qs_initial_g_g_h',
        'final_IB_fraction_pct',
        'final_protein_total_g_L',
        'final_protein_soluble_g_L',
        'final_protein_IB_g_L',
        'volumetric_productivity_g_L_h',
        'final_yield_g_g',
        'avg_pO2_pct'
    ]

    # Filter to available columns
    available_cols = [col for col in cols_of_interest if col in plot_data.columns]
    corr_data = plot_data[available_cols].copy()

    # Calculate correlation
    corr_matrix = corr_data.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)

    ax.set_title('Correlation Matrix: IB Formation Drivers', weight='bold', fontsize=14)

    # Create cleaner labels
    label_map = {
        'temperature_C': 'Temperature (°C)',
        'qs_initial_g_g_h': 'Initial qs',
        'final_IB_fraction_pct': 'IB Fraction (%)',
        'final_protein_total_g_L': 'Total Protein (g/L)',
        'final_protein_soluble_g_L': 'Soluble Protein (g/L)',
        'final_protein_IB_g_L': 'IB Protein (g/L)',
        'volumetric_productivity_g_L_h': 'Vol. Productivity',
        'final_yield_g_g': 'Yield (g/g)',
        'avg_pO2_pct': 'Avg pO2 (%)'
    }

    new_labels = [label_map.get(col, col) for col in corr_matrix.columns]
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    ax.set_yticklabels(new_labels, rotation=0)

    plt.tight_layout()
    return fig


def create_time_profiles(df, configs):
    """Create time profile plots for key parameters."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Time Profiles: Induction Phase', fontsize=16, weight='bold')

    # Get unique runs with their conditions
    runs = df['run_name'].unique()

    # Color mapping by IB fraction
    run_colors = {}
    for run in runs:
        run_df = df[df['run_name'] == run]
        induced = run_df[run_df['Is induced [-]'] == 1]
        if len(induced) > 0 and 'IB fraction [%]' in induced.columns:
            final_ib = induced['IB fraction [%]'].dropna()
            if len(final_ib) > 0:
                ib_frac = final_ib.iloc[-1]
                # Red for high IB, green for low IB
                run_colors[run] = (ib_frac / 100, 1 - ib_frac / 100, 0)

    # 1. Protein concentrations over time
    ax = axes[0, 0]
    for run in runs:
        run_df = df[df['run_name'] == run]
        induced = run_df[run_df['Is induced [-]'] == 1].copy()
        if len(induced) > 0:
            time_since_induction = induced['Process time [h]'] - induced['Process time [h]'].min()
            ax.plot(time_since_induction, induced['Protein soluble [g/L]'],
                   label=f'{run} (sol)', color=run_colors.get(run, 'blue'), linestyle='-')
            ax.plot(time_since_induction, induced['Protein IB [g/L]'],
                   label=f'{run} (IB)', color=run_colors.get(run, 'blue'), linestyle='--')

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('Protein [g/L]', weight='bold')
    ax.set_title('Protein Concentration Profiles', weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. IB fraction over time
    ax = axes[0, 1]
    for run in runs:
        run_df = df[df['run_name'] == run]
        induced = run_df[run_df['Is induced [-]'] == 1].copy()
        if len(induced) > 0 and 'IB fraction [%]' in induced.columns:
            time_since_induction = induced['Process time [h]'] - induced['Process time [h]'].min()
            ax.plot(time_since_induction, induced['IB fraction [%]'],
                   label=run, color=run_colors.get(run, 'blue'))

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('IB Fraction Evolution', weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # 3. Biomass over time
    ax = axes[1, 0]
    for run in runs:
        run_df = df[df['run_name'] == run]
        induced = run_df[run_df['Is induced [-]'] == 1].copy()
        if len(induced) > 0:
            time_since_induction = induced['Process time [h]'] - induced['Process time [h]'].min()
            ax.plot(time_since_induction, induced['Biomass [g/L]'],
                   label=run, color=run_colors.get(run, 'blue'))

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('Biomass [g/L]', weight='bold')
    ax.set_title('Biomass Profiles', weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Temperature over time
    ax = axes[1, 1]
    for run in runs:
        run_df = df[df['run_name'] == run]
        induced = run_df[run_df['Is induced [-]'] == 1].copy()
        if len(induced) > 0:
            time_since_induction = induced['Process time [h]'] - induced['Process time [h]'].min()
            ax.plot(time_since_induction, induced['Temperature [°C]'],
                   label=run, color=run_colors.get(run, 'blue'))

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('Temperature [°C]', weight='bold')
    ax.set_title('Temperature Profiles', weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Feed rate over time
    ax = axes[2, 0]
    for run in runs:
        run_df = df[df['run_name'] == run].sort_values('Process time [h]')
        induced = run_df[run_df['Is induced [-]'] == 1].copy()
        if len(induced) > 0:
            # Calculate feed rate
            induced = induced.copy()
            induced['feed_rate'] = induced['Feed [mL]'].diff() / induced['Process time [h]'].diff()
            time_since_induction = induced['Process time [h]'] - induced['Process time [h]'].min()
            ax.plot(time_since_induction, induced['feed_rate'],
                   label=run, color=run_colors.get(run, 'blue'))

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('Feed Rate [mL/h]', weight='bold')
    ax.set_title('Feed Rate Profiles', weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. pO2 over time
    ax = axes[2, 1]
    for run in runs:
        run_df = df[df['run_name'] == run]
        induced = run_df[run_df['Is induced [-]'] == 1].copy()
        if len(induced) > 0 and 'pO2 [%]' in induced.columns:
            time_since_induction = induced['Process time [h]'] - induced['Process time [h]'].min()
            ax.plot(time_since_induction, induced['pO2 [%]'],
                   label=run, color=run_colors.get(run, 'blue'))

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('pO2 [%]', weight='bold')
    ax.set_title('pO2 Profiles', weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_statistical_analysis(configs, summaries):
    """Perform and visualize statistical analysis."""
    plot_data = configs.join(summaries).reset_index()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Temperature effect analysis (box plot by temperature level)
    ax1 = fig.add_subplot(gs[0, 0])
    temp_groups = []
    temp_labels = []
    for temp in sorted(plot_data['temperature_C'].unique()):
        temp_data = plot_data[plot_data['temperature_C'] == temp]['final_IB_fraction_pct']
        temp_groups.append(temp_data.dropna().values)
        temp_labels.append(f'{temp:.0f}°C\n(n={len(temp_data.dropna())})')

    bp1 = ax1.boxplot(temp_groups, labels=temp_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))

    # Add individual points
    for i, group in enumerate(temp_groups):
        x = np.random.normal(i+1, 0.04, size=len(group))
        ax1.scatter(x, group, alpha=0.7, s=50, edgecolors='black')

    ax1.set_ylabel('IB Fraction [%]', weight='bold')
    ax1.set_title('Effect of Temperature on IB Formation', weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. qs effect analysis (box plot by qs level)
    ax2 = fig.add_subplot(gs[0, 1])
    qs_groups = []
    qs_labels = []
    for qs in sorted(plot_data['qs_initial_g_g_h'].unique()):
        qs_data = plot_data[plot_data['qs_initial_g_g_h'] == qs]['final_IB_fraction_pct']
        qs_groups.append(qs_data.dropna().values)
        qs_labels.append(f'qs={qs:.1f}\n(n={len(qs_data.dropna())})')

    bp2 = ax2.boxplot(qs_groups, labels=qs_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(facecolor='lightcoral', alpha=0.7))

    for i, group in enumerate(qs_groups):
        x = np.random.normal(i+1, 0.04, size=len(group))
        ax2.scatter(x, group, alpha=0.7, s=50, edgecolors='black')

    ax2.set_ylabel('IB Fraction [%]', weight='bold')
    ax2.set_title('Effect of Initial qs on IB Formation', weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Interaction plot (temperature x qs)
    ax3 = fig.add_subplot(gs[1, :])

    # Create pivot table for interaction
    pivot_data = plot_data.pivot_table(
        values='final_IB_fraction_pct',
        index='temperature_C',
        columns='qs_initial_g_g_h',
        aggfunc='mean'
    )

    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                vmin=0, vmax=100, cbar_kws={'label': 'IB Fraction [%]'},
                linewidths=2, ax=ax3)

    ax3.set_xlabel('Initial qs [g/g/h]', weight='bold')
    ax3.set_ylabel('Temperature [°C]', weight='bold')
    ax3.set_title('Interaction Effect: Temperature × qs on IB Formation', weight='bold')

    fig.suptitle('Statistical Analysis of IB Formation', fontsize=16, weight='bold', y=0.995)

    return fig


def main():
    """Main analysis function."""
    print("="*70)
    print("LDH Inclusion Body vs Soluble Expression Analysis")
    print("="*70)

    # Create output directory
    output_dir = Path('.')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    configs = load_run_configs()
    df = load_consolidated_data()
    print(f"   Loaded {len(df)} time points from {df['run_name'].nunique()} runs")

    # Calculate summaries
    print("\n2. Calculating run summaries...")
    summaries = calculate_run_summaries(df)

    # Calculate derived parameters
    print("\n3. Calculating derived physiological parameters...")
    derived = calculate_derived_parameters(df, configs)

    # Combine all data
    plot_data = configs.join(summaries).join(derived)

    # Print summary table
    print("\n" + "="*70)
    print("EXPERIMENTAL SUMMARY")
    print("="*70)
    summary_cols = ['temperature_C', 'qs_initial_g_g_h', 'final_protein_total_g_L',
                   'final_soluble_fraction_pct', 'final_IB_fraction_pct',
                   'final_yield_g_g', 'volumetric_productivity_g_L_h']
    print(plot_data[summary_cols].to_string(float_format=lambda x: f'{x:.3f}'))

    # Generate plots
    print("\n4. Generating plots...")

    # Experimental design plot
    print("   - Experimental design matrix...")
    fig1 = create_experimental_design_plot(configs, summaries)
    fig1.savefig(output_dir / 'fig1_experimental_design.png', dpi=300, bbox_inches='tight')
    print(f"     Saved: fig1_experimental_design.png")

    # IB fraction vs parameters
    print("   - IB fraction analysis...")
    fig2 = create_ib_fraction_vs_params_plot(configs, summaries)
    fig2.savefig(output_dir / 'fig2_ib_fraction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"     Saved: fig2_ib_fraction_analysis.png")

    # Correlation heatmap
    print("   - Correlation analysis...")
    fig3 = create_correlation_heatmap(plot_data)
    fig3.savefig(output_dir / 'fig3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"     Saved: fig3_correlation_heatmap.png")

    # Time profiles
    print("   - Time profiles...")
    fig4 = create_time_profiles(df, configs)
    fig4.savefig(output_dir / 'fig4_time_profiles.png', dpi=300, bbox_inches='tight')
    print(f"     Saved: fig4_time_profiles.png")

    # Statistical analysis
    print("   - Statistical analysis...")
    fig5 = create_statistical_analysis(configs, summaries)
    fig5.savefig(output_dir / 'fig5_statistical_analysis.png', dpi=300, bbox_inches='tight')
    print(f"     Saved: fig5_statistical_analysis.png")

    # Save processed data
    print("\n5. Saving processed data...")
    plot_data.to_excel(output_dir / 'analysis_summary.xlsx')
    print(f"   Saved: analysis_summary.xlsx")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - fig1_experimental_design.png: Experimental design matrix")
    print("  - fig2_ib_fraction_analysis.png: Multi-parameter analysis")
    print("  - fig3_correlation_heatmap.png: Correlation matrix")
    print("  - fig4_time_profiles.png: Time-series profiles")
    print("  - fig5_statistical_analysis.png: Statistical analysis")
    print("  - analysis_summary.xlsx: All processed data")


if __name__ == "__main__":
    main()
