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

# Define sigmoid model with s_T and T_50 parameters
def sigmoid_sT_T50(T, s_T, T_50):
    """
    Sigmoid function with interpretable parameters:
    f_IB(T) = sigmoid( s_T * (T - T_50) )
             = 1 / (1 + exp(-s_T * (T - T_50)))

    Parameters:
    - s_T: steepness of the transition
    - T_50: temperature at 50% partition (inflection point)
    """
    return 1 / (1 + np.exp(-s_T * (T - T_50)))

# Fit the model
# Initial guesses: s_T ~ 1-5, T_50 ~ 30
popt, pcov = curve_fit(sigmoid_sT_T50, x, y, p0=[2, 30], maxfev=10000)

s_T, T_50 = popt

# Calculate fit metrics
y_pred = sigmoid_sT_T50(x, s_T, T_50)
residuals = y - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean(residuals**2))

# Create fitted curve
x_fit = np.linspace(x.min() - 2, x.max() + 2, 200)
y_fit = sigmoid_sT_T50(x_fit, s_T, T_50)

# Plot
ax.scatter(x, y, alpha=0.8, s=150, edgecolors='black', linewidths=1.5,
           color='steelblue', label='Experimental data', zorder=5)
ax.plot(x_fit, y_fit, '-', linewidth=3, color='darkred',
        label='Sigmoid fit', zorder=4)

# Add bounds lines
ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# Add T_50 lines
ax.axvline(x=T_50, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label=f'T_50 = {T_50:.2f}°C', zorder=2)
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, zorder=2)

# Formatting
ax.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
ax.set_ylabel('Partition IB', fontsize=14, fontweight='bold')
ax.set_title(f'Temperature vs Partition IB - Sigmoid Model\n(s_T = {s_T:.2f}, T_50 = {T_50:.2f}°C, R² = {r2:.4f})',
             fontsize=15, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(x_fit.min(), x_fit.max())
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add equation text box
equation = r'$f_{IB}(T) = \sigma(s_T \cdot (T - T_{50}))$'
equation_detail = f'= 1 / (1 + exp(-{s_T:.2f} · (T - {T_50:.2f})))'
ax.text(0.05, 0.95, equation + '\n' + equation_detail,
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray',
                  alpha=0.9, linewidth=1.5))

# Set spines
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(r'C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\sigmoid_fit_sT_T50.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("SIGMOID MODEL FIT (s_T, T_50 PARAMETERIZATION)")
print("="*70)
print(f"\nModel: f_IB(T) = sigmoid(s_T * (T - T_50))")
print(f"                = 1 / (1 + exp(-s_T * (T - T_50)))")
print(f"\nFitted parameters:")
print(f"  s_T  = {s_T:.4f}  (steepness)")
print(f"  T_50 = {T_50:.2f}°C (inflection point)")
print(f"\nFit metrics:")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.5f}")
print(f"  Number of parameters = 2")
print(f"\nPredicted values:")
print(f"  At 25°C: {sigmoid_sT_T50(25, s_T, T_50):.4f}")
print(f"  At 30°C: {sigmoid_sT_T50(30, s_T, T_50):.4f}")
print(f"  At 35°C: {sigmoid_sT_T50(35, s_T, T_50):.4f}")
print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print(f"- s_T = {s_T:.2f}: The transition is relatively sharp")
print(f"- T_50 = {T_50:.2f}°C: At this temperature, 50% of protein is in IBs")
print(f"- Below T_50: Mostly soluble protein")
print(f"- Above T_50: Mostly inclusion bodies")
print("="*70)
