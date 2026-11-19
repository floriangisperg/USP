import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(14, 10))

# Read the Excel file
file_path = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\LDH_consolidated_results.xlsx"
df = pd.read_excel(file_path, sheet_name="Run_Summary")

# Calculate partition IB from the data
# Partition IB = Protein IB / (Protein IB + Protein soluble)
df["partition_IB"] = df["Protein IB [g/L]_final"] / (
    df["Protein IB [g/L]_final"] + df["Protein soluble [g/L]_final"]
)

# Get temperature column (handling the special character)
temp_col = [col for col in df.columns if "T_induction" in col][0]
df = df.rename(columns={temp_col: "Temperature"})

print("Data preview:")
print(
    df[
        [
            "Temperature",
            "partition_IB",
            "Protein IB [g/L]_final",
            "Protein soluble [g/L]_final",
        ]
    ].head(10)
)
print(f"\nData shape: {df.shape}")
print(
    f"\nTemperature range: {df['Temperature'].min():.1f} - {df['Temperature'].max():.1f}"
)
print(
    f"Partition IB range: {df['partition_IB'].min():.3f} - {df['partition_IB'].max():.3f}"
)

# Extract data
x = df["Temperature"].values
y = df["partition_IB"].values

# Remove any NaN values
mask = ~(np.isnan(x) | np.isnan(y))
x = x[mask]
y = y[mask]

print(f"\nValid data points: {len(x)}")


# Define models to fit
def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def exponential(x, a, b, c):
    """Exponential: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def hill_equation(x, bottom, top, ec50, hill_slope):
    """Hill equation: sigmoidal dose-response"""
    x = np.asarray(x)
    return bottom + (top - bottom) / (1 + (x / ec50) ** hill_slope)


def hill_increasing(x, bottom, top, ec50, hill_slope):
    """Hill equation (increasing)"""
    x = np.asarray(x)
    return bottom + (top - bottom) * (x / ec50) ** hill_slope / (
        1 + (x / ec50) ** hill_slope
    )


def sigmoid_4param(x, a, b, c, d):
    """4-parameter sigmoid (logistic)"""
    return a + (b - a) / (1 + np.exp(-c * (x - d)))


def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * x**b


def logarithmic(x, a, b):
    """Logarithmic: y = a * ln(x) + b"""
    return a * np.log(x) + b


# Fit all models
models = {
    "Linear": (linear, 2),
    "Quadratic": (quadratic, 3),
    "Exponential Growth": (exponential, 3),
    "Exponential Decay": (exponential_decay, 3),
    "Power Law": (power_law, 2),
    "Logarithmic": (logarithmic, 2),
    "Hill (Increasing)": (hill_increasing, 4),
    "4-Param Sigmoid": (sigmoid_4param, 4),
}

results = {}
x_fit = np.linspace(x.min() * 0.95, x.max() * 1.05, 200)

# Main plot
ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
ax_main.scatter(
    x,
    y,
    alpha=0.6,
    s=80,
    edgecolors="black",
    linewidths=0.5,
    color="steelblue",
    label="Experimental data",
    zorder=5,
)

# Fit each model and calculate metrics
colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

for idx, (name, (func, n_params)) in enumerate(models.items()):
    try:
        # Initial parameter guesses
        if name == "Linear":
            p0 = [0, 0]
            bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
        elif name == "Quadratic":
            p0 = [0, 0, 0]
            bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        elif name == "Exponential Growth":
            p0 = [0.01, 0.1, 0]
            bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        elif name == "Exponential Decay":
            p0 = [1, 0.1, 0]
            bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        elif name == "Power Law":
            p0 = [0.001, 2]
            bounds = ([0, -10], [np.inf, 10])
        elif name == "Logarithmic":
            p0 = [1, 0]
            bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
        elif name == "Hill (Increasing)":
            p0 = [y.min(), y.max(), np.median(x), 2]
            bounds = ([0, 0, x.min() * 0.1, 0.1], [np.inf, np.inf, x.max() * 10, 10])
        elif name == "4-Param Sigmoid":
            p0 = [y.min(), y.max(), 0.1, np.mean(x)]
            bounds = (
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, np.inf],
            )

        # Fit the model
        popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000, bounds=bounds)

        # Calculate fit metrics
        y_pred = func(x, *popt)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        aic = len(x) * np.log(ss_res / len(x)) + 2 * n_params
        bic = len(x) * np.log(ss_res / len(x)) + n_params * np.log(len(x))

        results[name] = {
            "params": popt,
            "R²": r2,
            "RMSE": rmse,
            "AIC": aic,
            "BIC": bic,
            "function": func,
        }

        # Plot fitted curve
        y_fit = func(x_fit, *popt)
        ax_main.plot(
            x_fit,
            y_fit,
            "--",
            alpha=0.7,
            linewidth=2,
            color=colors[idx],
            label=f"{name} (R²={r2:.3f})",
        )

    except Exception as e:
        print(f"Failed to fit {name}: {str(e)}")
        results[name] = {"error": str(e)}

ax_main.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
ax_main.set_ylabel("Partition IB", fontsize=12, fontweight="bold")
ax_main.set_title(
    "Temperature vs Partition IB - Model Comparison", fontsize=14, fontweight="bold"
)
ax_main.legend(loc="best", fontsize=9, framealpha=0.9)
ax_main.grid(True, alpha=0.3)

# Residuals plot
ax_res = plt.subplot2grid((3, 2), (2, 0))
ax_res.scatter(
    x,
    y - results["Linear"]["function"](x, *results["Linear"]["params"]),
    alpha=0.6,
    s=60,
    edgecolors="black",
    linewidths=0.5,
    color="coral",
)
ax_res.axhline(y=0, color="red", linestyle="--", linewidth=2)
ax_res.set_xlabel("Temperature (°C)", fontsize=10)
ax_res.set_ylabel("Residuals", fontsize=10)
ax_res.set_title("Residuals (Linear Model)", fontsize=11, fontweight="bold")
ax_res.grid(True, alpha=0.3)

# Metrics comparison bar chart
ax_metrics = plt.subplot2grid((3, 2), (2, 1))

model_names = [
    name for name in models.keys() if name in results and "error" not in results[name]
]
r2_values = [results[name]["R²"] for name in model_names]

bars = ax_metrics.barh(
    model_names, r2_values, color=colors, edgecolor="black", linewidth=0.5
)
ax_metrics.set_xlabel("R² Score", fontsize=10)
ax_metrics.set_title("Model Comparison (R²)", fontsize=11, fontweight="bold")
ax_metrics.set_xlim(0, 1)
ax_metrics.grid(True, axis="x", alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, r2_values):
    ax_metrics.text(
        val + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(
    r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\temperature_partition_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Print detailed results
print("\n" + "=" * 70)
print("MODEL FITTING RESULTS SUMMARY")
print("=" * 70)

# Sort by R²
sorted_results = sorted(
    [(name, metrics) for name, metrics in results.items() if "error" not in metrics],
    key=lambda x: x[1]["R²"],
    reverse=True,
)

print(f"\n{'Model':<25} {'R²':>8} {'RMSE':>10} {'AIC':>10} {'BIC':>10}")
print("-" * 70)

for name, metrics in sorted_results:
    print(
        f"{name:<25} {metrics['R²']:>8.4f} {metrics['RMSE']:>10.5f} {metrics['AIC']:>10.2f} {metrics['BIC']:>10.2f}"
    )

print("\n" + "=" * 70)
print("BIOLOGICAL INTERPRETATION")
print("=" * 70)

print("""
BEST FITTING MODELS AND BIOLOGICAL RELEVANCE:

1. HILL EQUATION (INCREASING SIGMOID):
   - Typical for: Enzyme-catalyzed reactions, protein-ligand binding
   - Biological meaning: Cooperative effects in protein folding/aggregation
   - Interpretation: Temperature threshold effect - below certain T, most protein
     remains soluble; above threshold, inclusion body formation increases sharply
   - The Hill coefficient indicates cooperativity of aggregation process
   - EC50 represents temperature at which 50% of maximal IB formation occurs

2. 4-PARAMETER SIGMOID (LOGISTIC):
   - Typical for: Growth processes, phase transitions
   - Biological meaning: Biphasic response with clear transition point
   - Interpretation: Similar to Hill but more symmetric, suggests a relatively
     sharp transition temperature for protein solubility

3. EXPONENTIAL MODELS:
   - Typical for: First-order kinetics, degradation processes
   - Biological meaning: Temperature-dependent rate processes following Arrhenius equation
   - Interpretation: Could represent that higher temperatures exponentially increase
     the rate of protein misfolding/aggregation
   - However, may not account for potential saturation at high temperatures

4. QUADRATIC:
   - Typical for: Simple empirical relationships
   - Biological meaning: Limited direct interpretation, mostly empirical fit
   - May suggest curvature but doesn't provide mechanistic insight

5. LINEAR:
   - Simplest model but likely oversimplifies the biology
   - Could be used as null hypothesis for comparison

RECOMMENDATION:
For a fermentation process with temperature-dependent protein aggregation:
- HILL EQUATION or SIGMOID models are most biologically meaningful
- They capture the cooperative nature of protein folding/aggregation
- They suggest a temperature threshold (EC50) which is practically useful
- The Hill slope provides information about the "sharpness" of the transition
""")
