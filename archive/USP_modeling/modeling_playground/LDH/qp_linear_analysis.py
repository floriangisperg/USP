"""
Analysis of Specific Product Formation Rate (q_p) Linear Decline in LDH Processes

This script analyzes the linear decline of specific product formation rate over time
due to metabolic burden, using only a linear decay model.

Linear model: q_p(t) = q_0 - k * (t - t_induction)

Where:
    q_0 = initial q_p at induction [g/g/h]
    k = decline rate constant [g/g/h²]
    t_induction = time of induction [h]

Author: Analysis based on publication findings
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class LinearQPAnalyzer:
    """Analyzer for linear q_p decline patterns."""

    def __init__(self, data_path):
        """Initialize with data path."""
        self.data_path = Path(data_path)
        self.df = None
        self.runs = []
        self.results = {}

    def load_data(self):
        """Load consolidated data from Excel file."""
        print("Loading data...")
        self.df = pd.read_excel(
            self.data_path,
            sheet_name='All_Time_Points'
        )
        self.runs = self.df['run_name'].unique()
        print(f"  Loaded {len(self.df)} total data points")
        print(f"  Found {len(self.runs)} runs: {self.runs}")
        return self.df

    def calculate_qp(self, run_df, smoothness=3.0):
        """
        Calculate specific product formation rate for a single run.

        Args:
            run_df: DataFrame for a single run
            smoothness: Spline smoothing parameter

        Returns:
            DataFrame with added q_p columns
        """
        # Filter to sample points only (where protein measurements exist AND biomass is available)
        mask = (run_df['Protein total calculated [g/L]'].notna() &
                run_df['Biomass [g/L]'].notna())
        sample_data = run_df[mask].copy()

        if len(sample_data) < 3:
            return None

        # Sort by time and reset index
        sample_data = sample_data.sort_values('Process time [h]').reset_index(drop=True)

        # Extract time and concentrations
        t = sample_data['Process time [h]'].values
        X = sample_data['Biomass [g/L]'].values
        P_sol = sample_data['Protein soluble [g/L]'].values
        P_ib = sample_data['Protein IB [g/L]'].values
        P_tot = sample_data['Protein total calculated [g/L]'].values

        # Remove any NaN values
        mask_clean = ~(np.isnan(X) | np.isnan(P_sol) | np.isnan(P_ib) | np.isnan(P_tot))

        if np.sum(mask_clean) < 3:
            return None

        # Apply the mask
        sample_data = sample_data[mask_clean].copy().reset_index(drop=True)
        t = sample_data['Process time [h]'].values
        X = sample_data['Biomass [g/L]'].values
        P_sol = sample_data['Protein soluble [g/L]'].values
        P_ib = sample_data['Protein IB [g/L]'].values
        P_tot = sample_data['Protein total calculated [g/L]'].values

        # Create splines for smooth differentiation
        # Use smoothing to reduce noise from measurement errors
        try:
            spline_sol = UnivariateSpline(t, P_sol, s=smoothness, k=3)
            spline_ib = UnivariateSpline(t, P_ib, s=smoothness, k=3)
            spline_tot = UnivariateSpline(t, P_tot, s=smoothness, k=3)

            # Calculate derivatives
            dP_sol_dt = spline_sol.derivative()(t)
            dP_ib_dt = spline_ib.derivative()(t)
            dP_tot_dt = spline_tot.derivative()(t)

            # Calculate q_p = (1/X) * dP/dt
            # Avoid division by zero
            X_safe = np.where(X > 0.1, X, 0.1)

            qp_sol = dP_sol_dt / X_safe
            qp_ib = dP_ib_dt / X_safe
            qp_tot = dP_tot_dt / X_safe

            # Store in dataframe
            sample_data['qp_soluble'] = qp_sol
            sample_data['qp_ib'] = qp_ib
            sample_data['qp_total'] = qp_tot
            sample_data['dP_sol_dt'] = dP_sol_dt
            sample_data['dP_ib_dt'] = dP_ib_dt
            sample_data['dP_tot_dt'] = dP_tot_dt

            return sample_data

        except Exception as e:
            print(f"    Warning: Could not calculate q_p: {e}")
            return None

    def fit_linear_decline(self, t, qp, t_induction=0):
        """
        Fit linear decline model to q_p data.

        Model: q_p(t) = q_0 - k * (t - t_induction)

        Args:
            t: Time points [h]
            qp: q_p values [g/g/h]
            t_induction: Induction time [h]

        Returns:
            Dictionary with fit parameters
        """
        # Filter to post-induction period
        t_post = t[t >= t_induction]
        qp_post = qp[t >= t_induction]

        if len(t_post) < 3:
            return None

        # Remove only extreme outliers (using IQR method instead of percentile)
        # Keep all non-negative values unless they're extreme outliers
        q1 = np.percentile(qp_post, 25)
        q3 = np.percentile(qp_post, 75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr  # Very conservative outlier removal
        
        mask = (qp_post >= 0) & (qp_post < upper_bound)
        t_post = t_post[mask]
        qp_post = qp_post[mask]

        if len(t_post) < 3:
            print(f"    Warning: Only {len(t_post)} points after filtering, need at least 3")
            return None

        # Normalize time to post-induction
        t_norm = t_post - t_induction

        # Linear decay model (allow negative values in fit, but clip predictions)
        def linear_decay(t, q0, k):
            return q0 - k * t

        try:
            popt, pcov = curve_fit(linear_decay, t_norm, qp_post,
                                  p0=[qp_post[0], 0.001],
                                  bounds=([0, 0], [np.inf, np.inf]),
                                  maxfev=5000)
            q0, k = popt

            # Calculate statistics (clipped predictions for comparison)
            qp_pred_clipped = np.maximum(linear_decay(t_norm, *popt), 0)
            ss_res = np.sum((qp_post - qp_pred_clipped) ** 2)
            ss_tot = np.sum((qp_post - np.mean(qp_post)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(ss_res / len(qp_post))
            stderr_q0 = np.sqrt(np.diag(pcov))[0] if pcov[0, 0] > 0 else 0
            stderr_k = np.sqrt(np.diag(pcov))[1] if pcov[1, 1] > 0 else 0

            # Calculate time when q_p reaches 50% of initial
            t_half = q0 / (2 * k) if k > 0 else np.inf

            # Calculate time when q_p reaches 10% of initial
            t_90 = 0.9 * q0 / k if k > 0 else np.inf

            return {
                'q0': q0,
                'k': k,
                'stderr_q0': stderr_q0,
                'stderr_k': stderr_k,
                'r_squared': r_squared,
                'rmse': rmse,
                't_half': t_half,
                't_90_decline': t_90,
                'function': linear_decay,
                't_fit': t_norm,
                'qp_fit': qp_pred_clipped,
                'qp_post': qp_post,
                'n_points': len(qp_post)
            }

        except Exception as e:
            print(f"    Warning: Could not fit linear model: {e}")
            return None

    def find_induction_time(self, run_df):
        """Find the induction time for a run."""
        induced_rows = run_df[run_df['Is induced [-]'] == 1]
        if len(induced_rows) > 0:
            return induced_rows['Process time [h]'].min()
        return run_df['Process time [h]'].min()

    def analyze_run(self, run, protein_type='total'):
        """Analyze a single run for linear q_p decline."""
        qp_col = f'qp_{protein_type}'

        # Get run data
        run_df = self.df[self.df['run_name'] == run].copy()

        # Calculate q_p
        qp_data = self.calculate_qp(run_df)

        if qp_data is None or len(qp_data) < 3:
            return None

        # Find induction time
        t_ind = self.find_induction_time(run_df)

        # Extract time and qp
        t = qp_data['Process time [h]'].values
        qp = qp_data[qp_col].values

        # Fit linear decline model
        fit_result = self.fit_linear_decline(t, qp, t_induction=t_ind)

        if fit_result is None:
            return None

        # Calculate basic stats
        qp_post = qp[t >= t_ind]
        qp_initial = qp_post[0] if len(qp_post) > 0 else np.nan
        qp_final = qp_post[-1] if len(qp_post) > 0 else np.nan

        # Calculate total production
        if protein_type == 'total':
            P_col = 'Protein total calculated [g/L]'
        elif protein_type == 'ib':
            P_col = 'Protein IB [g/L]'
        else:  # soluble
            P_col = 'Protein soluble [g/L]'
        P = qp_data[P_col].values
        P_final = P[-1] if len(P) > 0 else np.nan

        return {
            'run': run,
            'protein_type': protein_type,
            't_induction': t_ind,
            'n_samples': len(qp_data),
            'qp_initial': qp_initial,
            'qp_final': qp_final,
            'P_final': P_final,
            'qp_data': qp_data,
            **fit_result
        }

    def analyze_all_runs(self):
        """Analyze all runs for linear q_p decline."""
        print(f"\n{'='*70}")
        print("LINEAR Q_P DECLINE ANALYSIS")
        print(f"{'='*70}")

        all_results = []

        for protein_type in ['soluble', 'ib', 'total']:
            print(f"\n--- {protein_type.upper()} PROTEIN ---")

            for run in self.runs:
                print(f"\nRun: {run}")

                result = self.analyze_run(run, protein_type)

                if result:
                    print(f"  Induction: {result['t_induction']:.2f} h")
                    print(f"  q0 (initial): {result['q0']:.4f} ± {result['stderr_q0']:.4f} g/g/h")
                    print(f"  k (decline rate): {result['k']:.4f} ± {result['stderr_k']:.4f} g/g/h²")
                    print(f"  R²: {result['r_squared']:.4f}")
                    print(f"  RMSE: {result['rmse']:.4f}")
                    print(f"  t_50%: {result['t_half']:.2f} h (time to 50% initial q_p)")
                    print(f"  t_90%: {result['t_90_decline']:.2f} h (time to 90% decline)")

                    all_results.append(result)
                else:
                    print(f"  Skipped: insufficient data")

        self.results = all_results
        return all_results

    def plot_linear_fits(self, protein_type='total', save_path=None):
        """Create plot showing linear fits for all runs."""
        results = [r for r in self.results if r['protein_type'] == protein_type]

        if not results:
            print(f"  No results for {protein_type}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Linear q_p Decline - {protein_type.capitalize()} Protein',
                    fontsize=14, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # Plot 1: q_p vs time with linear fits
        ax = axes[0, 0]
        for i, res in enumerate(results):
            try:
                qp_data = res['qp_data']
                if isinstance(qp_data, pd.DataFrame):
                    t = qp_data['Process time [h]'].values
                    qp_col = f'qp_{protein_type}'
                    if qp_col in qp_data.columns:
                        qp = qp_data[qp_col].values
                        t_ind = res['t_induction']

                        # Plot data
                        mask = t >= t_ind
                        ax.scatter(t[mask] - t_ind, qp[mask], color=colors[i],
                                  alpha=0.6, s=50, label=f"{res['run']}", zorder=3)

                        # Plot linear fit
                        if 't_fit' in res and 'qp_fit' in res:
                            ax.plot(res['t_fit'], res['qp_fit'], color=colors[i],
                                   linewidth=2, linestyle='-', alpha=0.8)
            except Exception as e:
                print(f"  Warning: Could not plot {res.get('run', 'unknown')}: {e}")

        ax.set_xlabel('Time Post Induction [h]')
        ax.set_ylabel(f'q_p ({protein_type}) [g/g/h]')
        ax.set_title('Linear Decline Fits')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Decline rate comparison
        ax = axes[0, 1]
        runs = [r['run'] for r in results]
        k_values = [r['k'] * 1000 for r in results]  # Convert to more readable units
        k_errors = [r['stderr_k'] * 1000 if not np.isinf(r['stderr_k']) else 0 for r in results]

        bars = ax.bar(range(len(runs)), k_values, yerr=k_errors,
                     color=colors[:len(runs)], alpha=0.7, capsize=5)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_ylabel('Decline Rate k [×10⁻³ g/g/h²]')
        ax.set_title('Linear Decline Rate Constants')
        ax.grid(True, axis='y', alpha=0.3)

        # Plot 3: Initial q_p comparison
        ax = axes[1, 0]
        q0_values = [r['q0'] for r in results]
        q0_errors = [r['stderr_q0'] if not np.isinf(r['stderr_q0']) else 0 for r in results]

        bars = ax.bar(range(len(runs)), q0_values, yerr=q0_errors,
                     color=colors[:len(runs)], alpha=0.7, capsize=5)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_ylabel('Initial q_p₀ [g/g/h]')
        ax.set_title('Initial q_p at Induction')
        ax.grid(True, axis='y', alpha=0.3)

        # Plot 4: Time to 50% decline
        ax = axes[1, 1]
        t_half_values = [r['t_half'] if r['t_half'] < 100 else 100 for r in results]

        bars = ax.bar(range(len(runs)), t_half_values,
                     color=colors[:len(runs)], alpha=0.7)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_ylabel('Time to 50% q_p [h]')
        ax.set_title('Productivity Half-Life')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()

    def plot_detailed_fits(self, protein_type='total', save_path=None):
        """Create detailed plots for individual runs."""
        results = [r for r in self.results if r['protein_type'] == protein_type]

        if not results:
            print(f"  No results for {protein_type}")
            return

        n_runs = len(results)

        # Calculate grid dimensions
        if n_runs <= 2:
            n_cols = n_runs
            n_rows = 1
        elif n_runs <= 4:
            n_cols = 2
            n_rows = 2
        elif n_runs <= 6:
            n_cols = 3
            n_rows = 2
        else:
            n_cols = 3
            n_rows = (n_runs + 2) // 3

        # Create figure with standard subplot layout
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        fig.suptitle(f'Detailed Linear Fits - {protein_type.capitalize()} Protein',
                    fontsize=14, fontweight='bold')

        # Flatten axes array for easier indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Hide unused subplots
        for idx in range(n_runs, len(axes)):
            axes[idx].set_visible(False)

        for idx, res in enumerate(results):
            ax = axes[idx]
            
            try:
                qp_data = res.get('qp_data')
                
                if not isinstance(qp_data, pd.DataFrame):
                    raise ValueError(f"qp_data is not a DataFrame, got {type(qp_data)}")
                
                if 'Process time [h]' not in qp_data.columns:
                    raise ValueError(f"'Process time [h]' column not found in qp_data. Columns: {list(qp_data.columns)}")
                
                t = qp_data['Process time [h]'].values
                qp_col = f'qp_{protein_type}'
                
                if qp_col not in qp_data.columns:
                    raise ValueError(f"Column '{qp_col}' not found in qp_data. Available columns: {list(qp_data.columns)}")
                
                qp = qp_data[qp_col].values
                t_ind = res['t_induction']

                # Plot data
                mask = t >= t_ind
                n_points = np.sum(mask)
                
                if n_points == 0:
                    raise ValueError(f"No data points with t >= t_induction ({t_ind})")
                
                print(f"    Plotting {res['run']}: {n_points} data points, t range: [{t[mask].min():.2f}, {t[mask].max():.2f}]")
                
                ax.scatter(t[mask] - t_ind, qp[mask], color='steelblue',
                          alpha=0.7, s=60, label='Data', zorder=3)

                # Plot linear fit
                if 't_fit' in res and 'qp_fit' in res:
                    ax.plot(res['t_fit'], res['qp_fit'], color='red',
                           linewidth=2.5, linestyle='-', label='Linear fit', alpha=0.9)

                    # Add confidence interval approximation
                    stderr = res['rmse']
                    ax.fill_between(res['t_fit'],
                                   res['qp_fit'] - 1.96*stderr,
                                   res['qp_fit'] + 1.96*stderr,
                                   color='red', alpha=0.2, label='95% CI')

                # Add annotation
                info_text = (f"q₀ = {res['q0']:.4f} ± {res['stderr_q0']:.4f}\n"
                            f"k = {res['k']:.4f} ± {res['stderr_k']:.4f}\n"
                            f"R² = {res['r_squared']:.4f}\n"
                            f"t₁/₂ = {res['t_half']:.2f} h")
                ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                       
            except Exception as e:
                print(f"  ERROR: Could not plot detailed {res.get('run', 'unknown')}: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                ax.text(0.5, 0.5, f"PLOTTING ERROR:\n{str(e)[:80]}...", 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, 
                       color='red', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Time Post Induction [h]')
            ax.set_ylabel(f'q_p ({protein_type}) [g/g/h]')
            ax.set_title(f"{res['run']}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()

    def calculate_optimal_harvest(self, threshold_fraction=0.2):
        """
        Calculate optimal harvest time based on q_p decline.

        Args:
            threshold_fraction: Harvest when q_p drops to this fraction of initial
                               (default: 0.2 = 20% of initial q_p)
        """
        print(f"\n{'='*70}")
        print(f"OPTIMAL HARVEST TIME CALCULATION")
        print(f"(Threshold: {threshold_fraction*100:.0f}% of initial q_p)")
        print(f"{'='*70}\n")

        harvest_summary = []

        for protein_type in ['soluble', 'ib', 'total']:
            print(f"--- {protein_type.upper()} PROTEIN ---")

            results = [r for r in self.results if r['protein_type'] == protein_type]

            if not results:
                print("  No data available\n")
                continue

            for res in results:
                # Time when q_p reaches threshold fraction
                t_harvest = (1 - threshold_fraction) * res['q0'] / res['k'] if res['k'] > 0 else np.inf
                t_harvest_abs = res['t_induction'] + t_harvest

                # Check if harvest time is within data range
                qp_data = res.get('qp_data')
                if isinstance(qp_data, pd.DataFrame):
                    t_max = qp_data['Process time [h]'].max()
                else:
                    t_max = t_harvest_abs + 10  # Default if no data available
                within_range = t_harvest_abs <= t_max

                print(f"\n{res['run']}:")
                print(f"  Optimal harvest: {t_harvest:.2f} h post-induction (t = {t_harvest_abs:.2f} h)")
                print(f"  Actual process end: {t_max:.2f} h")
                print(f"  Harvest within range: {'Yes' if within_range else 'No (process stopped earlier)'}")

                harvest_summary.append({
                    'run': res['run'],
                    'protein': protein_type,
                    't_induction': res['t_induction'],
                    'q0': res['q0'],
                    'k': res['k'],
                    'optimal_harvest_post': t_harvest,
                    'optimal_harvest_abs': t_harvest_abs,
                    'actual_end_time': t_max,
                    'harvested_optimally': within_range,
                    'final_product_conc': res['P_final']
                })

        return pd.DataFrame(harvest_summary)

    def compare_runs(self):
        """Compare decline patterns across runs."""
        print(f"\n{'='*70}")
        print("CROSS-RUN COMPARISON")
        print(f"{'='*70}\n")

        for protein_type in ['soluble', 'ib', 'total']:
            results = [r for r in self.results if r['protein_type'] == protein_type]

            if not results:
                continue

            print(f"\n--- {protein_type.upper()} PROTEIN ---\n")

            # Sort by decline rate
            results_sorted = sorted(results, key=lambda x: x['k'])

            print("Ranking by decline rate (k):")
            print("(Lower k = slower decline = longer productive period)\n")

            for i, res in enumerate(results_sorted):
                print(f"{i+1}. {res['run']}:")
                print(f"     k = {res['k']:.4f} g/g/h²")
                print(f"     q0 = {res['q0']:.4f} g/g/h")
                print(f"     t_50% = {res['t_half']:.2f} h")
                print(f"     Final product = {res['P_final']:.2f} g/L\n")

            # Calculate statistics
            k_values = [r['k'] for r in results]
            q0_values = [r['q0'] for r in results]
            t_half_values = [r['t_half'] for r in results]

            print(f"Statistics across {len(results)} runs:")
            print(f"  Mean k: {np.mean(k_values):.4f} ± {np.std(k_values):.4f} g/g/h²")
            print(f"  Mean q0: {np.mean(q0_values):.4f} ± {np.std(q0_values):.4f} g/g/h")
            print(f"  Mean t_50%: {np.mean(t_half_values):.2f} ± {np.std(t_half_values):.2f} h")

    def create_summary_table(self, save_path=None):
        """Create comprehensive summary table."""
        summary_data = []

        for res in self.results:
            summary_data.append({
                'Run': res['run'],
                'Protein': res['protein_type'],
                'Induction [h]': res['t_induction'],
                'q0 [g/g/h]': res['q0'],
                'stderr_q0': res['stderr_q0'],
                'k [g/g/h²]': res['k'],
                'stderr_k': res['stderr_k'],
                'R²': res['r_squared'],
                'RMSE': res['rmse'],
                't_50% [h]': res['t_half'],
                't_90% [h]': res['t_90_decline'],
                'P_final [g/L]': res['P_final']
            })

        summary_df = pd.DataFrame(summary_data)

        if save_path:
            summary_df.to_excel(save_path, index=False)
            print(f"\nSaved summary table: {save_path}")

        return summary_df

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print(f"\n{'='*70}")
        print("LINEAR Q_P DECLINE ANALYSIS - FULL PIPELINE")
        print(f"{'='*70}")

        # Load data
        self.load_data()

        # Analyze all runs
        self.analyze_all_runs()

        # Create outputs
        output_dir = self.data_path.parent / 'qp_linear_analysis_results'
        output_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("GENERATING OUTPUTS")
        print(f"{'='*70}")

        # Generate plots
        for protein_type in ['soluble', 'ib', 'total']:
            self.plot_linear_fits(
                protein_type,
                save_path=output_dir / f'linear_overview_{protein_type}.png'
            )
            self.plot_detailed_fits(
                protein_type,
                save_path=output_dir / f'linear_detailed_{protein_type}.png'
            )

        # Calculate optimal harvest times
        harvest_df = self.calculate_optimal_harvest(threshold_fraction=0.2)
        harvest_save_path = output_dir / 'optimal_harvest_times.xlsx'
        harvest_df.to_excel(harvest_save_path, index=False)
        print(f"\nSaved harvest analysis: {harvest_save_path}")

        # Compare runs
        self.compare_runs()

        # Create summary table
        summary = self.create_summary_table(
            save_path=output_dir / 'linear_summary.xlsx'
        )

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")

        return summary, harvest_df


def main():
    """Main execution function."""
    # Path to data
    data_path = r'C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\modeling_playground\LDH\data\LDH_consolidated_results.xlsx'

    # Run analysis
    analyzer = LinearQPAnalyzer(data_path)
    summary, harvest = analyzer.run_full_analysis()

    return analyzer, summary, harvest


if __name__ == '__main__':
    analyzer, summary, harvest = main()
