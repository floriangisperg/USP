import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 10))

# Read the Excel file
file_path = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\LDH_consolidated_results.xlsx"
df = pd.read_excel(file_path, sheet_name='Run_Summary')

# Calculate partition IB from the data
# Partition IB = Protein IB / (Protein IB + Protein soluble)
df['partition_IB'] = df['Protein IB [g/L]_final'] / (df['Protein IB [g/L]_final'] + df['Protein soluble [g/L]_final'])

# Get temperature column (handling the special character)
temp_col = [col for col in df.columns if 'T_induction' in col][0]
df = df.rename(columns={temp_col: 'Temperature'})

print("="*70)
print("ANALYSIS OF TEMPERATURE VS PARTITION IB (BOUNDED 0-1)")
print("="*70)
print("\nData preview:")
print(df[['Temperature', 'partition_IB', 'Protein IB [g/L]_final', 'Protein soluble [g/L]_final']].head(10))
print(f"\nData shape: {df.shape}")
print(f"\nTemperature range: {df['Temperature'].min():.1f} - {df['Temperature'].max():.1f}")
print(f"Partition IB range: {df['partition_IB'].min():.3f} - {df['partition_IB'].max():.3f}")

# Extract data
x = df['Temperature'].values
y = df['partition_IB'].values

# Remove any NaN values
mask = ~(np.isnan(x) | np.isnan(y))
x = x[mask]
y = y[mask]

print(f"\nValid data points: {len(x)}")

# ============================================================================
# BOUNDED MODELS (0-1)
# ============================================================================

def logistic_4param(x, a, b, c, d):
    """
    4-parameter logistic function bounded between a and b
    y = a + (b-a) / (1 + exp(-c*(x-d)))
    where: a = lower asymptote, b = upper asymptote
           c = slope factor, d = inflection point
    """
    return a + (b - a) / (1 + np.exp(-c * (x - d)))

def hill_equation_bounded(x, bottom, top, ec50, hill_slope):
    """
    Hill equation with bounds [bottom, top]
    Typical for cooperative binding, enzyme kinetics
    """
    x = np.asarray(x)
    return bottom + (top - bottom) * (x**hill_slope) / (ec50**hill_slope + x**hill_slope)

def hill_equation_decreasing(x, bottom, top, ec50, hill_slope):
    """
    Hill equation for decreasing response (inhibition)
    """
    x = np.asarray(x)
    return bottom + (top - bottom) / (1 + (x / ec50)**hill_slope)

def arctan_model(x, a, b, c):
    """
    Arctangent function bounded between 0 and 1
    y = a * arctan(b*(x-c)) / (pi/2) + 0.5
    """
    return a * (2/np.pi) * np.arctan(b * (x - c)) + 0.5

def gompertz(x, a, b, c, d):
    """
    Gompertz function (sigmoidal)
    y = a + (b-a) * exp(-exp(c*(d-x)))
    """
    return a + (b - a) * np.exp(-np.exp(c * (d - x)))

def weibull_cdf(x, a, b, c):
    """
    Weibull CDF (bounded 0-1)
    y = 1 - exp(-(x/a)^b)
    Modified to: y = c * (1 - exp(-(x/a)^b))
    """
    return c * (1 - np.exp(-(x / a)**b))

def boltzmann(x, a1, a2, x0, dx):
    """
    Boltzmann sigmoid function
    y = (a1-a2) / (1 + exp((x-x0)/dx)) + a2
    """
    return (a1 - a2) / (1 + np.exp((x - x0) / dx)) + a2

def logit_linear(x, a, b):
    """
    Logit-linear model (working on log-odds scale)
    log(p/(1-p)) = a*x + b
    p = 1 / (1 + exp(-(a*x + b)))
    """
    logit = a * x + b
    return 1 / (1 + np.exp(-logit))

# Fit models
models = {
    'Logistic (4-param)': (logistic_4param, 4, [0, 1, 0.5, 30],
                          ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])),
    'Hill (Increasing)': (hill_equation_bounded, 4, [0, 1, 30, 2],
                         ([0, 0, x.min()*0.1, 0.1], [1, 1, x.max()*10, 10])),
    'Boltzmann Sigmoid': (boltzmann, 4, [1, 0, 30, 1],
                         ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])),
    'Logit-Linear': (logit_linear, 2, [0.5, -15],
                    ([-np.inf, -np.inf], [np.inf, np.inf])),
    'Arctan Sigmoid': (arctan_model, 3, [1, 0.5, 30],
                      ([0, 0.01, -np.inf], [1, np.inf, np.inf])),
    'Gompertz': (gompertz, 4, [0, 1, 0.1, 30],
               ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])),
}

results = {}
x_fit = np.linspace(x.min() - 2, x.max() + 2, 200)

# Main plot
ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
ax_main.scatter(x, y, alpha=0.7, s=100, edgecolors='black', linewidths=1.5,
                color='steelblue', label='Experimental data', zorder=5)

# Add horizontal lines at 0 and 1 bounds
ax_main.axhline(y=0, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax_main.axhline(y=1, color='red', linestyle=':', linewidth=1, alpha=0.5)

# Fit each model and calculate metrics
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

for idx, (name, (func, n_params, p0, bounds)) in enumerate(models.items()):
    try:
        # Fit the model
        popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000, bounds=bounds)

        # Calculate fit metrics
        y_pred = func(x, *popt)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        aic = len(x) * np.log((ss_res + 1e-10) / len(x)) + 2 * n_params
        bic = len(x) * np.log((ss_res + 1e-10) / len(x)) + n_params * np.log(len(x))

        # Check bounds violations
        y_fit_check = func(x_fit, *popt)
        bound_violations = np.sum((y_fit_check < -0.01) | (y_fit_check > 1.01))

        results[name] = {
            'params': popt,
            'n_params': n_params,
            'R²': r2,
            'RMSE': rmse,
            'AIC': aic,
            'BIC': bic,
            'function': func,
            'bound_violations': bound_violations
        }

        # Plot fitted curve
        y_fit = func(x_fit, *popt)
        ax_main.plot(x_fit, y_fit, '-', alpha=0.8, linewidth=2.5,
                    color=colors[idx], label=f'{name} (R²={r2:.4f})')

    except Exception as e:
        print(f"Failed to fit {name}: {str(e)}")
        results[name] = {'error': str(e)}

ax_main.set_xlabel('Temperature (°C)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Partition IB', fontsize=13, fontweight='bold')
ax_main.set_title('Temperature vs Partition IB - Bounded Models (0-1)', fontsize=14, fontweight='bold')
ax_main.set_ylim(-0.1, 1.1)
ax_main.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax_main.grid(True, alpha=0.3)

# Residuals plot
ax_res = plt.subplot2grid((3, 2), (2, 0))
best_model = max([(n, r) for n, r in results.items() if 'error' not in r],
                 key=lambda x: x[1]['R²'])[0]
residuals = y - results[best_model]['function'](x, *results[best_model]['params'])
ax_res.scatter(x, residuals, alpha=0.7, s=80, edgecolors='black', linewidths=1,
               color='coral', zorder=5)
ax_res.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax_res.set_xlabel('Temperature (°C)', fontsize=11)
ax_res.set_ylabel('Residuals', fontsize=11)
ax_res.set_title(f'Residuals ({best_model})', fontsize=12, fontweight='bold')
ax_res.grid(True, alpha=0.3)

# Metrics comparison bar chart
ax_metrics = plt.subplot2grid((3, 2), (2, 1))

model_names = [name for name in results.keys() if 'error' not in results[name]]
r2_values = [results[name]['R²'] for name in model_names]

bars = ax_metrics.barh(model_names, r2_values, color=colors, edgecolor='black', linewidth=0.8)
ax_metrics.set_xlabel('R² Score', fontsize=11)
ax_metrics.set_title('Model Comparison (R²)', fontsize=12, fontweight='bold')
ax_metrics.set_xlim(0, 1.05)
ax_metrics.grid(True, axis='x', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, r2_values):
    ax_metrics.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(r'C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\temperature_partition_bounded.png',
            dpi=300, bbox_inches='tight')
plt.show()

# Print detailed results
print("\n" + "="*70)
print("BOUNDED MODEL FITTING RESULTS (0-1 CONSTRAINT)")
print("="*70)

# Sort by R²
sorted_results = sorted([(name, metrics) for name, metrics in results.items()
                        if 'error' not in metrics], key=lambda x: x[1]['R²'], reverse=True)

print(f"\n{'Model':<20} {'Params':>8} {'R²':>8} {'RMSE':>10} {'AIC':>10} {'BIC':>10} {'Bounds OK':>10}")
print("-" * 85)

for name, metrics in sorted_results:
    bounds_ok = "Y" if metrics['bound_violations'] == 0 else f"N ({metrics['bound_violations']})"
    r2_val = metrics['R²'] if 'R²' in metrics else metrics.get('R2', 0)
    print(f"{name:<20} {metrics['n_params']:>8} {r2_val:>8.4f} {metrics['RMSE']:>10.5f} {metrics['AIC']:>10.2f} {metrics['BIC']:>10.2f} {bounds_ok:>10}")

# Print best model parameters
print("\n" + "="*70)
print("BEST MODEL PARAMETERS")
print("="*70)
best_name, best_metrics = sorted_results[0]
print(f"\nBest Model: {best_name}")
print(f"Parameters: {best_metrics['params']}")

if 'Logistic' in best_name or 'Boltzmann' in best_name:
    print(f"\nInterpretation:")
    p = best_metrics['params']
    if 'Logistic' in best_name:
        print(f"  - Lower asymptote: {p[0]:.4f}")
        print(f"  - Upper asymptote: {p[1]:.4f}")
        print(f"  - Slope factor: {p[2]:.4f}")
        print(f"  - Inflection point: {p[3]:.2f} °C")
        print(f"  - Temperature at 50% partition: {p[3]:.2f} °C")
    elif 'Boltzmann' in best_name:
        print(f"  - A1 (initial value): {p[0]:.4f}")
        print(f"  - A2 (final value): {p[1]:.4f}")
        print(f"  - X0 (center): {p[2]:.2f} °C")
        print(f"  - dx (width): {p[3]:.2f}")

elif 'Hill' in best_name:
    print(f"\nInterpretation:")
    p = best_metrics['params']
    print(f"  - Bottom: {p[0]:.4f}")
    print(f"  - Top: {p[1]:.4f}")
    print(f"  - EC50 (temperature for half-max): {p[2]:.2f} °C")
    print(f"  - Hill slope (cooperativity): {p[2]:.2f}")
    print(f"\n  Hill slope > 1: Positive cooperativity (sharp transition)")
    print(f"  Hill slope = 1: Hyperbolic (Michaelis-Menten like)")
    print(f"  Hill slope < 1: Negative cooperativity (gradual transition)")

print("\n" + "="*70)
print("BIOLOGICAL INTERPRETATION FOR FERMENTATION PROCESS")
print("="*70)

print("""
PROTEIN SOLUBILITY AND TEMPERATURE:

In recombinant protein fermentation, the partition between soluble protein
and inclusion bodies (IBs) is fundamentally a protein folding problem that
is strongly temperature-dependent.

KEY BIOLOGICAL MECHANISMS:

1. PROTEIN FOLDING THERMODYNAMICS:
   - At lower temperatures: Kinetic folding is slow but proper folding is favored
   - At higher temperatures: Hydrophobic interactions increase, promoting aggregation
   - Each protein has a characteristic "folding transition temperature"

2. CHAPERONE SYSTEM:
   - Molecular chaperones (DnaK, GroEL) assist proper folding
   - Chaperone capacity is overwhelmed at higher expression or temperature
   - Above threshold, misfolded proteins aggregate into IBs

3. HYDROPHOBIC EFFECT:
   - Increases with temperature (unusual but critical for proteins)
   - Drives exposed hydrophobic patches to associate
   - Leads to aggregation and IB formation

4. PROTEIN CONCENTRATION:
   - Higher concentration = more collisions = more aggregation
   - Temperature affects solubility limits

WHY SIGMOIDAL MODELS MAKE SENSE:

The HILL EQUATION and LOGISTIC functions are most appropriate because:

1. COOPERATIVITY: Protein aggregation is cooperative - once a nucleus
   forms, additional proteins add rapidly

2. THRESHOLD BEHAVIOR: Below certain T, proper folding dominates;
   above threshold, aggregation dominates

3. SATURATION: Partition is bounded 0-1 (cannot exceed these limits)

4. EC50 / T50: The temperature at 50% partition is practically useful
   for process optimization

PRACTICAL IMPLICATIONS:

- Find the T50 (temperature for 50% IB formation)
- Operate below T50 for soluble protein
- Operate above T50 for IB production (can be advantageous for easy purification)
- Hill slope indicates how "sharp" the transition is
""")
