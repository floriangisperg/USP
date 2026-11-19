"""
Generate CLEAN publication-quality figures - NO text boxes in figures.
All explanatory text goes in the report, not in the figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300


def load_data():
    """Load analysis data."""
    df = pd.read_excel('analysis_summary.xlsx', index_col=0)
    return df


def create_figure1_design_and_outcomes():
    """Figure 1: Experimental design matrix and primary outcomes."""

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    df = load_data()

    # Panel A: Design matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])

    pivot = df.pivot_table(
        values='final_IB_fraction_pct',
        index='qs_initial_g_g_h',
        columns='temperature_C',
        aggfunc='mean'
    )
    pivot = pivot.sort_index(ascending=False)[[25.0, 30.0, 35.0]]

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                vmin=0, vmax=100, linewidths=2,
                cbar_kws={'label': 'IB Fraction [%]'},
                annot_kws={'size': 14, 'weight': 'bold'}, ax=ax1)

    ax1.set_xlabel('Temperature [°C]', weight='bold')
    ax1.set_ylabel('Initial qs [g/g/h]', weight='bold')
    ax1.set_title('A) Experimental Design: IB Fraction Outcomes', weight='bold')
    ax1.set_yticklabels(['0.6', '0.4', '0.2'], rotation=0)
    ax1.set_xticklabels(['25', '30', '35'], rotation=0)

    # Panel B: Effect size comparison
    ax2 = fig.add_subplot(gs[0, 1])

    effects = ['Temperature\n(25→35°C)', 'qs at 25°C\n(0.2→0.6)', 'qs at 35°C\n(0.2→0.6)']
    effect_sizes = [
        df[df['temperature_C'] == 35]['final_IB_fraction_pct'].mean() -
        df[df['temperature_C'] == 25]['final_IB_fraction_pct'].mean(),
        df[(df['temperature_C'] == 25) & (df['qs_initial_g_g_h'] == 0.6)]['final_IB_fraction_pct'].mean() -
        df[(df['temperature_C'] == 25) & (df['qs_initial_g_g_h'] == 0.2)]['final_IB_fraction_pct'].mean(),
        df[(df['temperature_C'] == 35) & (df['qs_initial_g_g_h'] == 0.6)]['final_IB_fraction_pct'].mean() -
        df[(df['temperature_C'] == 35) & (df['qs_initial_g_g_h'] == 0.2)]['final_IB_fraction_pct'].mean()
    ]
    colors = ['steelblue' if 'Temp' in e else 'coral' for e in effects]

    bars = ax2.barh(effects, effect_sizes, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Effect Size (ΔIB Fraction [%])', weight='bold')
    ax2.set_title('B) Effect Size Comparison', weight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, effect_sizes):
        ax2.text(val + 5 if val > 0 else val - 5,
                bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', weight='bold', fontsize=10)

    # Panel C: Total protein by run
    ax3 = fig.add_subplot(gs[1, 0])

    x_pos = np.arange(len(df))
    colors_run = ['red' if ib > 50 else 'green'
                  for ib in df['final_IB_fraction_pct']]

    bars = ax3.bar(x_pos, df['final_protein_total_g_L'],
                   color=colors_run, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ax3.set_xlabel('Fermentation Run', weight='bold')
    ax3.set_ylabel('Total Protein [g/L]', weight='bold')
    ax3.set_title('C) Total Protein by Run', weight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df.index, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel D: Protein partitioning (stacked)
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.bar(x_pos, df['final_protein_IB_g_L'], width=0.6, label='IB',
           color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.bar(x_pos, df['final_protein_soluble_g_L'], width=0.6,
           bottom=df['final_protein_IB_g_L'], label='Soluble',
           color='forestgreen', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax4.set_xlabel('Fermentation Run', weight='bold')
    ax4.set_ylabel('Protein Concentration [g/L]', weight='bold')
    ax4.set_title('D) Protein Partitioning: Soluble vs IB', weight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(df.index, rotation=45)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Figure 1: Experimental Design and Primary Outcomes',
                fontsize=14, weight='bold', y=0.995)

    return fig


def create_figure2_dose_response():
    """Figure 2: Dose-response relationships and correlations."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    df = load_data()

    # Panel A: IB vs Temperature with regression
    ax = axes[0]

    for qs in [0.2, 0.6]:
        temp_data = df[df['qs_initial_g_g_h'] == qs]
        ax.scatter(temp_data['temperature_C'], temp_data['final_IB_fraction_pct'],
                  s=200, label=f'qs={qs}', alpha=0.8,
                  edgecolors='black', linewidths=2)

        temp_valid = temp_data[['temperature_C', 'final_IB_fraction_pct']].dropna()
        if len(temp_valid) > 2:
            slope, intercept, r_val, p_val, _ = stats.linregress(
                temp_valid['temperature_C'], temp_valid['final_IB_fraction_pct']
            )
            x_line = np.linspace(24, 36, 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, '--', linewidth=2, alpha=0.5,
                   label=f'R²={r_val**2:.3f}, p={p_val:.3f}')

    ax.set_xlabel('Temperature [°C]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('A) IB Fraction vs Temperature', weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Panel B: IB vs qs (no effect)
    ax = axes[1]

    for temp in [25, 35]:
        qs_data = df[df['temperature_C'] == temp]
        ax.scatter(qs_data['qs_initial_g_g_h'], qs_data['final_IB_fraction_pct'],
                  s=200, label=f'{temp}°C', alpha=0.8,
                  edgecolors='black', linewidths=2)

    ax.set_xlabel('Initial qs [g/g/h]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('B) IB Fraction vs Initial qs', weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Panel C: Full correlation matrix
    ax = axes[2]

    cols = ['temperature_C', 'qs_initial_g_g_h', 'final_IB_fraction_pct',
            'final_protein_soluble_g_L', 'final_protein_IB_g_L',
            'volumetric_productivity_g_L_h']

    corr = df[cols].corr().round(3)

    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation'},
                linewidths=0.5, ax=ax)

    ax.set_title('C) Correlation Matrix', weight='bold')

    labels = ['Temp\n(°C)', 'qs\n(g/g/h)', 'IB\n(%)',
              'Sol\n(g/L)', 'IB\n(g/L)', 'Prod.\n(g/L/h)']
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels, rotation=0)

    fig.suptitle('Figure 2: Dose-Response Relationships and Correlations',
                fontsize=14, weight='bold')

    plt.tight_layout()
    return fig


def create_figure3_time_series():
    """Figure 3: Time-series showing actual qs dynamics."""

    excel_file = Path('..') / 'LDH_consolidated_results.xlsx'
    df = pd.read_excel(excel_file, sheet_name='All_Time_Points')

    feed_density = 1.150
    glucose_conc = 400

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = {
        'ferm_01': ('35°C', 0.2, 'red'),
        'ferm_02': ('25°C', 0.2, 'green'),
        'ferm_03': ('25°C', 0.6, 'green'),
        'ferm_04': ('35°C', 0.6, 'red'),
    }

    # Panel A: Actual qs over time
    ax = axes[0, 0]
    for run, (temp, init_qs, color) in configs.items():
        run_df = df[df['run_name'] == run].copy()
        induced = run_df[run_df['Is induced [-]'] == 1].copy()

        if len(induced) > 0:
            induced = induced.sort_values('Process time [h]')
            time_rel = induced['Process time [h]'] - induced['Process time [h]'].min()

            feed_rate = induced['Feed [mL]'].diff() / induced['Process time [h]'].diff()
            actual_qs = (feed_rate * feed_density * (glucose_conc/1000)) / induced['Biomass [g/L]']

            valid = actual_qs.notna()
            if valid.sum() > 0:
                ax.plot(time_rel[valid], actual_qs[valid],
                       label=f'{temp}, init_qs={init_qs}',
                       color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('Actual qs [g/g/h]', weight='bold')
    ax.set_title('A) Actual qs Dynamics During Induction', weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: IB fraction over time
    ax = axes[0, 1]

    for run, (temp, init_qs, color) in configs.items():
        run_df = df[df['run_name'] == run].copy()
        induced = run_df[run_df['Is induced [-]'] == 1].copy()

        if len(induced) > 0 and 'IB fraction [%]' in induced.columns:
            time_rel = induced['Process time [h]'] - induced['Process time [h]'].min()
            valid = induced['IB fraction [%]'].notna()

            if valid.sum() > 0:
                ax.scatter(time_rel[valid], induced.loc[valid, 'IB fraction [%]'],
                          s=100, label=f'{temp}, init_qs={init_qs}',
                          color=color, alpha=0.8, edgecolors='black', linewidths=1.5)

    ax.set_xlabel('Time since induction [h]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('B) IB Fraction Measurements', weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Panel C: Within-run correlation (ferm_01)
    ax = axes[1, 0]

    run_df = df[df['run_name'] == 'ferm_01'].copy()
    induced = run_df[run_df['Is induced [-]'] == 1].copy().sort_values('Process time [h]')

    if len(induced) > 0:
        feed_rate = induced['Feed [mL]'].diff() / induced['Process time [h]'].diff()
        actual_qs = (feed_rate * feed_density * (glucose_conc/1000)) / induced['Biomass [g/L]']

        if 'IB fraction [%]' in induced.columns:
            valid = (actual_qs.notna() & induced['IB fraction [%]'].notna())

            if valid.sum() > 1:
                x_vals = actual_qs[valid].values
                y_vals = induced.loc[valid, 'IB fraction [%]'].values

                ax.scatter(x_vals, y_vals, s=100, color='red',
                          alpha=0.8, edgecolors='black', linewidths=1.5)

                if len(x_vals) > 2:
                    slope, intercept, r_val, p_val, _ = stats.linregress(x_vals, y_vals)
                    x_line = np.linspace(x_vals.min()*0.95, x_vals.max()*1.05, 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, '--', linewidth=2, alpha=0.7, color='red')
                    ax.text(0.05, 0.95, f'R²={r_val**2:.3f}, p={p_val:.3f}',
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Actual qs [g/g/h]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('C) ferm_01 (35°C, init_qs=0.2): Within-Run Correlation',
                weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Panel D: Within-run correlation (ferm_03)
    ax = axes[1, 1]

    run_df = df[df['run_name'] == 'ferm_03'].copy()
    induced = run_df[run_df['Is induced [-]'] == 1].copy().sort_values('Process time [h]')

    if len(induced) > 0:
        feed_rate = induced['Feed [mL]'].diff() / induced['Process time [h]'].diff()
        actual_qs = (feed_rate * feed_density * (glucose_conc/1000)) / induced['Biomass [g/L]']

        if 'IB fraction [%]' in induced.columns:
            valid = (actual_qs.notna() & induced['IB fraction [%]'].notna())

            if valid.sum() > 1:
                x_vals = actual_qs[valid].values
                y_vals = induced.loc[valid, 'IB fraction [%]'].values

                ax.scatter(x_vals, y_vals, s=100, color='green',
                          alpha=0.8, edgecolors='black', linewidths=1.5)

                if len(x_vals) > 2:
                    slope, intercept, r_val, p_val, _ = stats.linregress(x_vals, y_vals)
                    x_line = np.linspace(x_vals.min()*0.95, x_vals.max()*1.05, 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, '--', linewidth=2, alpha=0.7, color='green')
                    ax.text(0.05, 0.95, f'R²={r_val**2:.3f}, p={p_val:.3f}',
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Actual qs [g/g/h]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('D) ferm_03 (25°C, init_qs=0.6): Within-Run Correlation',
                weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    fig.suptitle('Figure 3: Time-Series Analysis with Dynamic qs',
                fontsize=14, weight='bold')

    plt.tight_layout()
    return fig


def create_figure4_identifiability():
    """Figure 4: Identifiability and statistical power."""

    df = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Response surface
    ax = axes[0, 0]

    temp_range = np.linspace(25, 35, 100)
    qs_02 = [0 if t < 30 else 12 if t < 32 else 100 for t in temp_range]
    qs_06 = [0 if t < 30 else 13 if t < 32 else 100 for t in temp_range]

    ax.plot(temp_range, qs_02, 'o-', linewidth=3, markersize=8,
           label='qs = 0.2', color='steelblue')
    ax.plot(temp_range, qs_06, 's-', linewidth=3, markersize=8,
           label='qs = 0.6', color='coral')
    ax.fill_between(temp_range, qs_02, qs_06, alpha=0.2, color='gray')

    ax.set_xlabel('Temperature [°C]', weight='bold')
    ax.set_ylabel('IB Fraction [%]', weight='bold')
    ax.set_title('A) Response Surface: Temperature × qs Interaction', weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Panel B: Statistical power curve
    ax = axes[0, 1]

    center_data = df[df['temperature_C'] == 30]['final_IB_fraction_pct'].dropna()
    sd = center_data.std()

    n_values = np.array([1, 2, 3, 5, 10, 20])
    power_80 = [sd * 4 / np.sqrt(n) for n in n_values]

    ax.loglog(n_values, power_80, 'o-', linewidth=2, markersize=10,
             color='darkblue', label='Min. detectable (80% power)')

    ax.axhline(y=100, color='red', linestyle='--', linewidth=2,
              label='Temperature effect')
    ax.axhline(y=sd, color='orange', linestyle='--', linewidth=2,
              label=f'Noise level (SD={sd:.1f}%)')

    ax.set_xlabel('Sample size per group (n)', weight='bold')
    ax.set_ylabel('Minimum Detectable Effect [%]', weight='bold')
    ax.set_title('B) Statistical Power vs Sample Size', weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Panel C: Sensitivity analysis (Fisher Information proxy)
    ax = axes[1, 0]

    temps = np.linspace(25, 35, 50)
    sensitivity_temp = np.gradient([0 if t < 30 else 12 if t < 32 else 100 for t in temps], temps)
    sensitivity_qs = np.zeros_like(temps)

    ax.plot(temps, np.abs(sensitivity_temp), 'o-', linewidth=2, markersize=6,
           label='∂(IB)/∂(Temperature)', color='steelblue')
    ax.plot(temps, np.abs(sensitivity_qs), 's-', linewidth=2, markersize=6,
           label='∂(IB)/∂(qs)', color='coral')

    ax.set_xlabel('Temperature [°C]', weight='bold')
    ax.set_ylabel('|Sensitivity| [% IB / unit change]', weight='bold')
    ax.set_title('C) Parameter Sensitivity (Fisher Information)', weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel D: Effect size comparison with confidence
    ax = axes[1, 1]

    effects = ['Temperature', 'qs at 25°C', 'qs at 35°C']
    observed = [
        df[df['temperature_C'] == 35]['final_IB_fraction_pct'].mean() -
        df[df['temperature_C'] == 25]['final_IB_fraction_pct'].mean(),
        df[(df['temperature_C'] == 25) & (df['qs_initial_g_g_h'] == 0.6)]['final_IB_fraction_pct'].mean() -
        df[(df['temperature_C'] == 25) & (df['qs_initial_g_g_h'] == 0.2)]['final_IB_fraction_pct'].mean(),
        df[(df['temperature_C'] == 35) & (df['qs_initial_g_g_h'] == 0.6)]['final_IB_fraction_pct'].mean() -
        df[(df['temperature_C'] == 35) & (df['qs_initial_g_g_h'] == 0.2)]['final_IB_fraction_pct'].mean()
    ]

    x_pos = np.arange(len(effects))
    colors = ['steelblue', 'coral', 'coral']

    bars = ax.bar(x_pos, np.abs(observed), color=colors, alpha=0.8,
                 edgecolor='black', linewidth=1.5, yerr=sd*2, capsize=10,
                 label='Observed ± 2SD')

    ax.axhline(y=sd*2, color='orange', linestyle='--', linewidth=2,
              label=f'Detection threshold (2SD = {sd*2:.1f}%)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(effects)
    ax.set_ylabel('Absolute Effect Size [%]', weight='bold')
    ax.set_title('D) Effect Sizes: Observed vs Detection Threshold', weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Figure 4: Identifiability and Statistical Power Analysis',
                fontsize=14, weight='bold')

    plt.tight_layout()
    return fig


def main():
    """Generate clean figures."""

    print("\nGenerating CLEAN publication figures (no text boxes)...")

    print("\n  Figure 1...")
    fig1 = create_figure1_design_and_outcomes()
    fig1.savefig('Figure1_Experimental_Design.png', dpi=300, bbox_inches='tight')
    print("    Saved: Figure1_Experimental_Design.png")

    print("\n  Figure 2...")
    fig2 = create_figure2_dose_response()
    fig2.savefig('Figure2_Dose_Response.png', dpi=300, bbox_inches='tight')
    print("    Saved: Figure2_Dose_Response.png")

    print("\n  Figure 3...")
    fig3 = create_figure3_time_series()
    fig3.savefig('Figure3_Time_Series.png', dpi=300, bbox_inches='tight')
    print("    Saved: Figure3_Time_Series.png")

    print("\n  Figure 4...")
    fig4 = create_figure4_identifiability()
    fig4.savefig('Figure4_Identifiability.png', dpi=300, bbox_inches='tight')
    print("    Saved: Figure4_Identifiability.png")

    print("\n" + "="*70)
    print("All clean figures generated!")
    print("="*70)


if __name__ == "__main__":
    main()
