"""
Analysis of Specific Product Formation Rate (q_p) Decline in LDH Processes

This script analyzes whether the specific product formation rate declines over time
due to metabolic burden, as described in literature for inclusion body processes.

The specific product formation rate is calculated as:
    q_p = (1/X) * dP/dt

Where:
    X = biomass concentration [g/L]
    P = product concentration [g/L] (soluble, IB, or total)
    dP/dt = rate of change of product concentration

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


class QPDeclineAnalyzer:
    """Analyzer for specific product formation rate decline patterns."""

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

    def fit_decline_models(self, t, qp, t_induction=0):
        """
        Fit various decline models to q_p data.

        Models:
        1. Constant: q_p = q_0
        2. Exponential decay: q_p = q_0 * exp(-k * t_post_induction)
        3. Linear decay: q_p = q_0 - k * t_post_induction
        4. Power law: q_p = q_0 * (1 + k * t_post_induction)^(-n)
        """
        # Filter to post-induction period and non-negative qp
        t_post = t[t >= t_induction]
        qp_post = qp[t >= t_induction]

        # Remove outliers
        mask = (qp_post >= 0) & (qp_post < np.percentile(qp_post, 95))
        t_post = t_post[mask]
        qp_post = qp_post[mask]

        if len(t_post) < 2:
            return None

        # Normalize time to post-induction
        t_norm = t_post - t_induction

        results = {}

        # Model 1: Constant
        q0_const = np.mean(qp_post)
        ss_const = np.sum((qp_post - q0_const)**2)
        results['constant'] = {
            'model': 'Constant',
            'q0': q0_const,
            'k': 0,
            'n': 0,
            'params': [q0_const],
            'ssr': ss_const,
            'rmse': np.sqrt(ss_const / len(qp_post))
        }

        # Model 2: Exponential decay
        def exp_decay(t, q0, k):
            return q0 * np.exp(-k * t)

        try:
            popt, _ = curve_fit(exp_decay, t_norm, qp_post,
                              p0=[qp_post[0], 0.01],
                              bounds=([0, 0], [np.inf, 1]),
                              maxfev=5000)
            q0, k = popt
            ss = np.sum((qp_post - exp_decay(t_norm, *popt))**2)
            results['exponential'] = {
                'model': 'Exponential Decay',
                'q0': q0,
                'k': k,
                'n': 0,
                'params': popt,
                'ssr': ss,
                'rmse': np.sqrt(ss / len(qp_post)),
                'function': exp_decay
            }
        except:
            pass

        # Model 3: Linear decay
        def linear_decay(t, q0, k):
            return np.maximum(q0 - k * t, 0)

        try:
            popt, _ = curve_fit(linear_decay, t_norm, qp_post,
                              p0=[qp_post[0], 0.001],
                              bounds=([0, 0], [np.inf, np.inf]),
                              maxfev=5000)
            q0, k = popt
            ss = np.sum((qp_post - linear_decay(t_norm, *popt))**2)
            results['linear'] = {
                'model': 'Linear Decay',
                'q0': q0,
                'k': k,
                'n': 0,
                'params': popt,
                'ssr': ss,
                'rmse': np.sqrt(ss / len(qp_post)),
                'function': linear_decay
            }
        except:
            pass

        # Model 4: Power law (Luedeking-Piret style with burden)
        def power_law(t, q0, k, n):
            return q0 * (1 + k * t) ** (-n)

        try:
            popt, _ = curve_fit(power_law, t_norm, qp_post,
                              p0=[qp_post[0], 0.01, 1],
                              bounds=([0, 0, 0], [np.inf, 1, 5]),
                              maxfev=5000)
            q0, k, n = popt
            ss = np.sum((qp_post - power_law(t_norm, *popt))**2)
            results['power'] = {
                'model': 'Power Law Decay',
                'q0': q0,
                'k': k,
                'n': n,
                'params': popt,
                'ssr': ss,
                'rmse': np.sqrt(ss / len(qp_post)),
                'function': power_law
            }
        except:
            pass

        # Select best model (lowest RMSE)
        if results:
            best_model_name = min(results.keys(),
                                key=lambda x: results[x]['rmse'])
            for model_name in results:
                results[model_name]['best'] = (model_name == best_model_name)

        return results

    def find_induction_time(self, run_df):
        """Find the induction time for a run."""
        induced_rows = run_df[run_df['Is induced [-]'] == 1]
        if len(induced_rows) > 0:
            return induced_rows['Process time [h]'].min()
        return run_df['Process time [h]'].min()

    def analyze_all_runs(self, protein_type='total'):
        """
        Analyze all runs for q_p decline patterns.

        Args:
            protein_type: 'soluble', 'ib', or 'total'
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING Q_P DECLINE FOR {protein_type.upper()} PROTEIN")
        print(f"{'='*70}")

        qp_col = f'qp_{protein_type}'
        all_results = []

        for run in self.runs:
            print(f"\n--- Run: {run} ---")

            # Get run data
            run_df = self.df[self.df['run_name'] == run].copy()

            # Calculate q_p
            qp_data = self.calculate_qp(run_df)

            if qp_data is None or len(qp_data) < 3:
                print(f"  Skipping: insufficient data")
                continue

            # Find induction time
            t_ind = self.find_induction_time(run_df)
            print(f"  Induction time: {t_ind:.2f} h")
            print(f"  Sample points: {len(qp_data)}")

            # Extract time and qp
            t = qp_data['Process time [h]'].values
            qp = qp_data[qp_col].values

            # Calculate basic stats
            qp_pre_ind = qp[t < t_ind]
            qp_post_ind = qp[t >= t_ind]

            if len(qp_pre_ind) > 0:
                print(f"  Pre-induction q_p: {np.mean(qp_pre_ind):.4f} ± {np.std(qp_pre_ind):.4f}")
            if len(qp_post_ind) > 0:
                print(f"  Post-induction q_p: {np.mean(qp_post_ind):.4f} ± {np.std(qp_post_ind):.4f}")
                print(f"  Decline: {100 * (1 - np.mean(qp_post_ind)/np.mean(qp_pre_ind)):.1f}%" if len(qp_pre_ind) > 0 else "")

            # Fit decline models
            models = self.fit_decline_models(t, qp, t_induction=t_ind)

            if models:
                best = next(m for m in models.values() if m.get('best', False))
                print(f"  Best model: {best['model']} (RMSE: {best['rmse']:.4f})")
                print(f"    q0: {best['q0']:.4f}")
                if best['k'] > 0:
                    print(f"    k: {best['k']:.4f}")
                if best['n'] > 0:
                    print(f"    n: {best['n']:.4f}")

                all_results.append({
                    'run': run,
                    't_induction': t_ind,
                    'n_points': len(qp_data),
                    'qp_pre_mean': np.mean(qp_pre_ind) if len(qp_pre_ind) > 0 else np.nan,
                    'qp_post_mean': np.mean(qp_post_ind) if len(qp_post_ind) > 0 else np.nan,
                    'qp_decline_pct': 100 * (1 - np.mean(qp_post_ind)/np.mean(qp_pre_ind)) if len(qp_pre_ind) > 0 and len(qp_post_ind) > 0 else np.nan,
                    'best_model': best['model'],
                    'q0': best['q0'],
                    'k': best['k'],
                    'n': best['n'],
                    'rmse': best['rmse'],
                    'qp_data': qp_data,
                    'models': models
                })

        self.results[protein_type] = all_results
        return all_results

    def plot_qp_overview(self, protein_type='total', save_path=None):
        """Create overview plots of q_p decline across all runs."""
        results = self.results.get(protein_type, [])

        if not results:
            print(f"No results for {protein_type}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Specific Product Formation Rate (q_p) - {protein_type.capitalize()} Protein',
                    fontsize=14, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # Plot 1: q_p vs time for all runs
        ax = axes[0, 0]
        for i, res in enumerate(results):
            qp_data = res['qp_data']
            t = qp_data['Process time [h]'].values
            qp = qp_data[f'qp_{protein_type}'].values
            t_ind = res['t_induction']

            # Plot pre and post induction
            mask_pre = t < t_ind
            mask_post = t >= t_ind

            ax.scatter(t[mask_pre], qp[mask_pre], color=colors[i], alpha=0.6, s=30, label=f"{res['run']}")
            ax.scatter(t[mask_post], qp[mask_post], color=colors[i], alpha=0.8,
                      marker='x', s=40)

            # Plot fitted curve
            models = res['models']
            best = next((m for m in models.values() if m.get('best', False)), None)
            if best and 'function' in best:
                t_fit = np.linspace(t_ind, t.max(), 100)
                t_norm_fit = t_fit - t_ind
                qp_fit = best['function'](t_norm_fit, *best['params'])
                ax.plot(t_fit, qp_fit, color=colors[i], linewidth=2,
                       linestyle='--', alpha=0.8)

        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Process Time [h]')
        ax.set_ylabel(f'q_p ({protein_type}) [g/g/h]')
        ax.set_title('q_p Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Normalized q_p decline
        ax = axes[0, 1]
        for i, res in enumerate(results):
            qp_data = res['qp_data']
            t = qp_data['Process time [h]'].values
            qp = qp_data[f'qp_{protein_type}'].values
            t_ind = res['t_induction']

            # Normalize to post-induction maximum
            mask = t >= t_ind
            if np.sum(mask) > 0:
                qp_norm = qp / qp[mask].max() if qp[mask].max() > 0 else qp
                t_norm = t - t_ind

                ax.scatter(t_norm[mask], qp_norm[mask], color=colors[i],
                          alpha=0.6, s=30, label=f"{res['run']}")

        ax.set_xlabel('Time Post Induction [h]')
        ax.set_ylabel('Normalized q_p')
        ax.set_title('Normalized q_p Decline')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 3: Decline percentage comparison
        ax = axes[1, 0]
        runs = [r['run'] for r in results]
        declines = [r['qp_decline_pct'] for r in results]

        bars = ax.bar(range(len(runs)), declines, color=colors[:len(runs)], alpha=0.7)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_ylabel('q_p Decline [%]')
        ax.set_title('Post-Induction q_p Decline')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, declines)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

        # Plot 4: Model comparison
        ax = axes[1, 1]
        model_counts = {}
        for res in results:
            model = res['best_model']
            model_counts[model] = model_counts.get(model, 0) + 1

        if model_counts:
            models = list(model_counts.keys())
            counts = list(model_counts.values())
            ax.bar(range(len(models)), counts, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Number of Runs')
            ax.set_title('Best-Fit Decline Models')
            ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()

    def plot_individual_runs(self, protein_type='total', save_path=None):
        """Create detailed plots for individual runs."""
        results = self.results.get(protein_type, [])

        if not results:
            return

        n_runs = len(results)
        n_cols = 3
        n_rows = (n_runs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_runs == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        fig.suptitle(f'q_p Analysis by Run - {protein_type.capitalize()} Protein',
                    fontsize=14, fontweight='bold')

        for idx, res in enumerate(results):
            ax = axes[idx]

            qp_data = res['qp_data']
            t = qp_data['Process time [h]'].values
            qp = qp_data[f'qp_{protein_type}'].values
            t_ind = res['t_induction']

            # Plot data
            mask_pre = t < t_ind
            mask_post = t >= t_ind

            ax.scatter(t[mask_pre], qp[mask_pre], color='steelblue',
                      alpha=0.6, s=40, label='Pre-induction', zorder=3)
            ax.scatter(t[mask_post], qp[mask_post], color='coral',
                      alpha=0.8, s=50, marker='x', label='Post-induction', zorder=3)

            # Plot fitted models
            models = res['models']
            colors_model = ['green', 'red', 'purple', 'orange']

            for i, (model_name, model) in enumerate(models.items()):
                if 'function' in model:
                    t_fit = np.linspace(t_ind, t.max(), 100)
                    t_norm_fit = t_fit - t_ind
                    qp_fit = model['function'](t_norm_fit, *model['params'])

                    linestyle = '-' if model.get('best', False) else ':'
                    linewidth = 2 if model.get('best', False) else 1
                    alpha = 0.9 if model.get('best', False) else 0.5

                    ax.plot(t_fit, qp_fit, color=colors_model[i % len(colors_model)],
                           linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                           label=f"{model['model']}")

            ax.axvline(x=t_ind, color='gray', linestyle='--',
                      linewidth=1.5, alpha=0.7, label='Induction')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            ax.set_xlabel('Process Time [h]')
            ax.set_ylabel(f'q_p ({protein_type}) [g/g/h]')
            ax.set_title(f"{res['run']}: {res['best_model']}\nRMSE: {res['rmse']:.4f}")
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()

    def create_summary_table(self, save_path=None):
        """Create a summary table of all results."""
        all_data = []

        for protein_type in ['soluble', 'ib', 'total']:
            results = self.results.get(protein_type, [])
            for res in results:
                all_data.append({
                    'Run': res['run'],
                    'Protein': protein_type,
                    'Induction [h]': res['t_induction'],
                    'Pre-q_p [g/g/h]': res['qp_pre_mean'],
                    'Post-q_p [g/g/h]': res['qp_post_mean'],
                    'Decline [%]': res['qp_decline_pct'],
                    'Best Model': res['best_model'],
                    'q0': res['q0'],
                    'k': res['k'],
                    'n': res['n'],
                    'RMSE': res['rmse']
                })

        summary_df = pd.DataFrame(all_data)

        if save_path:
            summary_df.to_excel(save_path, index=False)
            print(f"Saved: {save_path}")

        return summary_df

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print(f"\n{'='*70}")
        print("Q_P DECLINE ANALYSIS - FULL PIPELINE")
        print(f"{'='*70}")

        # Load data
        self.load_data()

        # Analyze each protein type
        for protein_type in ['soluble', 'ib', 'total']:
            self.analyze_all_runs(protein_type)

        # Create outputs
        output_dir = self.data_path.parent / 'qp_analysis_results'
        output_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("GENERATING OUTPUTS")
        print(f"{'='*70}")

        # Overview plots
        for protein_type in ['soluble', 'ib', 'total']:
            self.plot_qp_overview(
                protein_type,
                save_path=output_dir / f'qp_overview_{protein_type}.png'
            )
            self.plot_individual_runs(
                protein_type,
                save_path=output_dir / f'qp_individual_{protein_type}.png'
            )

        # Summary table
        summary = self.create_summary_table(
            save_path=output_dir / 'qp_summary.xlsx'
        )

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")
        print("\nKey Findings:")
        print(summary.to_string(index=False))

        return summary


def main():
    """Main execution function."""
    # Path to data
    data_path = r'C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\modeling_playground\LDH\data\LDH_consolidated_results.xlsx'

    # Run analysis
    analyzer = QPDeclineAnalyzer(data_path)
    summary = analyzer.run_full_analysis()

    return analyzer, summary


if __name__ == '__main__':
    analyzer, summary = main()
