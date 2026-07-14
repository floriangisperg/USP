import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

# Read the Excel file
file_path = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\LDH_consolidated_results.xlsx"
df = pd.read_excel(file_path, sheet_name='Run_Summary')

# Calculate partition IB
df['partition_IB'] = df['Protein IB [g/L]_final'] / (df['Protein IB [g/L]_final'] + df['Protein soluble [g/L]_final'])

# Get temperature column
temp_col = [col for col in df.columns if 'T_induction' in col][0]
df = df.rename(columns={temp_col: 'Temperature'})

# Extract data
x = df['Temperature'].values
y = df['partition_IB'].values

# Remove NaN values
mask = ~(np.isnan(x) | np.isnan(y))
x = x[mask]
y = y[mask]

# Define logit-linear model
def logit_linear(x, a, b):
    """
    Logit-linear model: log(p/(1-p)) = a*x + b
    p = 1 / (1 + exp(-(a*x + b)))
    """
    logit = a * x + b
    return 1 / (1 + np.exp(-logit))

# Fit the model
popt, pcov = curve_fit(logit_linear, x, y, p0=[0.5, -15], maxfev=10000)

# Calculate fit metrics
y_pred = logit_linear(x, *popt)
residuals = y - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean(residuals**2))

# Create fitted curve
x_fit = np.linspace(x.min() - 2, x.max() + 2, 200)
y_fit = logit_linear(x_fit, *popt)

# Plot
ax.scatter(x, y, alpha=0.8, s=150, edgecolors='black', linewidths=1.5,
           color='steelblue', label='Experimental data', zorder=5)
ax.plot(x_fit, y_fit, '-', linewidth=3, color='darkred',
        label='Logit-linear fit', zorder=4)

# Add bounds lines
ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# Add T50 line
t50 = -popt[1] / popt[0]
ax.axvline(x=t50, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label=f'T50 = {t50:.2f}°C', zorder=2)
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, zorder=2)

# Formatting
ax.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
ax.set_ylabel('Partition IB', fontsize=14, fontweight='bold')
ax.set_title('Temperature vs Partition IB - Logit-Linear Model\n(2 parameters, R² = {:.4f})'.format(r2),
             fontsize=15, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(x_fit.min(), x_fit.max())
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add equation text box
equation = f'partition = 1 / (1 + exp(-({popt[0]:.3f}·T + {popt[1]:.2f})))'
ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white',
                                          edgecolor='gray', alpha=0.9, linewidth=1.5))

# Set spines
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(r'C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\logit_linear_fit.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("LOGIT-LINEAR MODEL FIT RESULTS")
print("="*70)
print(f"\nEquation: partition = 1 / (1 + exp(-({popt[0]:.4f}*T + {popt[1]:.2f})))")
print(f"\nParameters:")
print(f"  a (slope)        = {popt[0]:.4f}")
print(f"  b (intercept)    = {popt[1]:.2f}")
print(f"\nFit metrics:")
print(f"  R²               = {r2:.4f}")
print(f"  RMSE             = {rmse:.5f}")
print(f"  Number of params = 2")
print(f"\nKey characteristics:")
print(f"  T50 (50% point)  = {t50:.2f} °C")
print(f"  At 25°C:         {logit_linear(25, *popt):.4f} partition")
print(f"  At 30°C:         {logit_linear(30, *popt):.4f} partition")
print(f"  At 35°C:         {logit_linear(35, *popt):.4f} partition")
print("\n" + "="*70)
