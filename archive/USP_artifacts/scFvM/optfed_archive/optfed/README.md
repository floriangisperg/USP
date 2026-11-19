# OptFed - Mechanistic Model Implementation

This folder contains the implementation of the OptFed mechanistic model for E. coli fed-batch fermentation, based on the paper by Schlögel et al. (2024).

## Paper Reference

**Title:** Optimizing bioprocessing efficiency with OptFed: Dynamic nonlinear modeling improves product-to-biomass yield

**Authors:** Guido Schlögel, Rüdiger Lück, Stefan Kittler, et al.

**Journal:** Computational and Structural Biotechnology Journal, 23 (2024), 3651-3661

**DOI:** https://doi.org/10.1016/j.csbj.2024.07.034

**bioRxiv:** https://www.biorxiv.org/content/10.1101/2024.07.31.605953v2

**GitHub:** https://github.com/gschloegel/OptFed/

### Key Contributions from the Paper

- Three-stage framework: **Define** → **Fit** → **Optimize**
- Model selection based on F-test significance testing
- Temperature dependence using Eyring equation (Eq. 4e)
- 19% improvement in product-to-biomass yield achieved experimentally
- Optimal temperature identified at 35.8°C (missed by RSM)

## Dataset

- **Data file**: `../data/traintest_compiled.xlsx`
- **Feed concentration**: 400 g/L glucose
- **Strain:** E. coli BL21DE3 (pET) producing scFv-His

### Data Columns

| Column | Description | Units |
|--------|-------------|-------|
| time | Process time | h |
| X | Total biomass (dry weight) | g/L |
| P | Product concentration (scFv-His) | g/L |
| temp | Temperature | °C |
| feed | Cumulative feed volume | mL |
| reactor_volume | Current reactor volume | mL |
| inductor_switch | Induction state (0/1) | - |
| pH | pH value | - |
| pO2 | Dissolved oxygen | % |
| StirrerSpeed | Stirrer speed | rpm |
| OffgasO2 | Outlet O2 | % |
| OffgasCO2 | Outlet CO2 | % |
| RunID | Fermentation run identifier | - |

### Train/Test Split

**Training runs** (corners + center points):
- ferm_01, ferm_02, ferm_03, ferm_04 (corners)
- ferm_05, ferm_06, ferm_07 (center points)

**Test runs** (star points):
- ferm_10, ferm_11, ferm_12 (star points)
- ferm_opt (validation run)

## Three-Stage Framework

### Stage I: Define
- General process model with 27-29 parameters
- Full maintenance model (Eq. 4c) with activation by {γ°, G, n, P/X, X}
- Temperature dependence (Eq. 4e) using Eyring equation with enzyme inactivation

### Stage II: Fit
- Parameter estimation using differential evolution
- Model selection using F-test based significance testing
- Removes insignificant terms to prevent overfitting

### Stage III: Optimize
- Optimize feed rate and temperature profiles
- Maximize product-to-biomass yield (P/X)
- Uses orthogonal collocation and nonlinear programming

## Process Description

### Fed-Batch Phases

1. **Batch phase**: Initial growth on batch medium
2. **Fed-batch phase (pre-induction)**: Exponential feed at constant temperature (~37°C)
3. **Fed-batch phase (post-induction)**: Linear feed with temperature shift (25-37°C depending on run)

### Feed Strategies

- **Pre-induction**: Exponential feed `f(t) = f0 * exp(μf * t)`
- **Post-induction**: Linear feed `f(t) = φ1 + φ2 * t`

### Temperature Profiles

- **Pre-induction**: Constant at ~37°C for all runs
- **Post-induction**: Varies by run (25-37°C) - this is one of the DOE factors
- **Temperature shift** occurs just before induction (~22-23h)

## Model Equations (from Paper)

### State Equations (Eq. 1)

```
dX/dt = μ * X - (f/V) * X
dP/dt = π * X - (f/V) * P
dG/dt = -γ° * X + (f/V) * (Gf - G)
dV/dt = f
```

### Kinetic Rate Expressions

Total substrate uptake (Eq. 4a):
```
γ° = γ°_max(T) * G / (K°_m + G) * Π (1 / (1 + i/K°_i))
var1 = {G, n, P/X, X}
```

Production uptake (Eq. 4b):
```
γ_π = γ_π_max(T) * (γ° - γ_α) / (K_π_m + γ° - γ_α) * Π (1 / (1 + i/K_π_i))
```

Maintenance (Eq. 4c - Full Model):
```
γ_α = γ_α_min(T) * Π (1 + i/K_α_i)
var2 = {γ°, G, n, P/X, X}
where n = log2[XV / (X0*V0)]
```

### Temperature Dependence (Eq. 4e)

The Eyring equation with enzyme inactivation:

```
γ_i^j(T) = (E_i^0 * k_B * T / h) * exp(-ΔG_cat^i / (R*T)) / (1 + exp(ΔH_eq^i / R * (1/T_eq^i - 1/T)))
```

Where:
- `E_i^0`: Hypothetical enzyme concentration
- `k_B`: Boltzmann constant (1.38 × 10^-23 J/K)
- `h`: Planck constant (2.39 × 10^-30 J·h)
- `ΔG_cat`: Catalytic activation energy (J/mol)
- `R`: Gas constant (8.314 J/(mol·K))
- `ΔH_eq`: Enthalpy for enzyme inactivation (J/mol)
- `T_eq`: Temperature where half the enzyme is active (K)

## Implementation Features

### 1. Temperature Dependence Models

Three models are supported:

1. **None** (`model="none"`): No temperature correction
2. **Q10** (`model="Q10"`): Simplified Arrhenius model
   - `k(T) = k_ref * Q10^((T - T_ref)/10)`
3. **Arrhenius** (`model="arrhenius"`): Standard Arrhenius model
   - `k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))`
4. **Eyring** (`model="erying"`): From paper Eq. 4e with enzyme inactivation
   - Uses parameters from Table 1 of the paper

### 2. Two-Phase Feed Strategy

The `FeedStrategy.create_two_phase_control()` function creates a complete two-phase control profile:

```python
control = FeedStrategy.create_two_phase_control(
    f0_exp=0.01,           # Initial feed rate for exponential phase (L/h)
    mu_f_exp=0.1,          # Specific growth rate for exponential feed (1/h)
    phi1_lin=0.02,         # Intercept for linear phase (L/h)
    phi2_lin=0.0005,       # Slope for linear phase (L/h²)
    t_induction=23.0,      # Induction time (h)
    T_pre=37.0,            # Pre-induction temperature (°C)
    T_post=30.0,           # Post-induction temperature (°C)
)
```

### 3. Initial Conditions from Data

The `InitialConditions.from_first_measurement()` class method creates initial conditions from the first measurement:

```python
init = InitialConditions.from_first_measurement(
    X_first=10.0,   # First biomass measurement (g/L)
    P_first=0.0,    # First product measurement (g/L)
    V_first=1.1,    # First volume measurement (L)
    G_first=0.5     # First substrate measurement (g/L)
)
```

## Model Parameters from Paper (Table 1)

### Fitted Parameters (Reduced Model)

| Parameter | Description | Value | Unit |
|-----------|-------------|-------|------|
| c_γ°_max | Maximal uptake rate | 0.49 | g g^-1 h^-1 |
| c_γ_α_min | Maintenance requirement | 2.4×10^-5 | g g^-1 h^-1 |
| E_π^0 | Enzyme concentration (production) | 8.8×10^-9 | - |
| K°_m | Substrate dependence | 1.0×10^-3 | g |
| K_α_g | Growth-dependent maintenance | 1.0×10^-4 | g g^-1 h^-1 |
| K_π_m | Production dependence on substrate | 0.62 | g g^-1 h^-1 |
| K°_G | Substrate inhibition | 89 | g L^-1 |
| K_α_P | Product effect on maintenance | 0.11 | g L^-1 |
| K_π_n | Generation inhibition | 1.5 | - |
| ΔG_π_cat | Activation energy (production) | 5.2×10^4 | J mol^-1 |
| ΔH_π_eq | Enthalpy of inactivation | 4.8×10^6 | J mol^-1 |
| T_π_eq | Half-inactivation temperature | 310 | K (36.85°C) |

### Constants (from literature)

| Parameter | Description | Value | Unit |
|-----------|-------------|-------|------|
| Y_Xr/G | Biomass yield | 0.627 | g g^-1 |
| Y_P/G | Product yield | 0.652 | g g^-1 |
| k_B | Boltzmann constant | 1.38×10^-23 | J K^-1 |
| R | Gas constant | 8.314 | J mol^-1 K^-1 |
| h | Planck constant | 2.39×10^-30 | J h |

## Usage Examples

### Basic Simulation

```python
from model import create_default_model, FeedStrategy, ControlProfile
import numpy as np

model = create_default_model()

# Exponential feed
control = ControlProfile(feed_rate=FeedStrategy.exponential(f0=0.01, mu_f=0.1))

t_span = (0, 30)
t_eval = np.linspace(0, 30, 301)
result = model.simulate(t_span, t_eval, control)

print(f"Final product: {result['P_conc'][-1]:.2f} g/L")
```

### Two-Phase Feed with Temperature Shift

```python
from model import create_model_erying, FeedStrategy, InitialConditions
import numpy as np

model = create_model_erying()  # Uses Eyring temperature model from paper
init = InitialConditions.from_first_measurement(10.0, 0.0, 1.1, 0.5)
model.init_cond = init

# Two-phase control
control = FeedStrategy.create_two_phase_control(
    f0_exp=0.01, mu_f_exp=0.1,
    phi1_lin=0.02, phi2_lin=0.0005,
    t_induction=23.0,
    T_pre=37.0, T_post=30.0
)

result = model.simulate((0, 30), np.linspace(0, 30, 301), control)
```

### Running the Full Validation Pipeline

Once parameter fitting is complete, use the validation script to:

1. Apply model selection (F-test based)
2. Validate on training and test data
3. Generate all validation plots

```bash
# Basic validation (includes model selection)
python validate_model.py --data ../data/traintest_compiled.xlsx

# Skip model selection, use fitted parameters as-is
python validate_model.py --skip-selection

# Use custom significance level for F-test
python validate_model.py --alpha 0.01
```

**Outputs:**
- `results/selected_params.pkl` - Model-selected parameters
- `results/validation/metrics.json` - RMSE, MAE, R² for all runs
- `results/validation/plots/` - Parity plots, validation summary, run comparisons

## Project Structure

```
optfed/
├── README.md                 # This file
├── STATUS.md                 # Implementation status
├── model.py                  # Core mechanistic model equations
├── fitting.py                # Parameter estimation (Stage II)
├── optimization.py           # Optimization framework (Stage III)
├── data_loader.py            # Data loading and preprocessing
├── test_optfed.py            # Test script
└── results/                  # Output folder
    ├── fitted_params.pkl
    ├── fitted_params.txt
    ├── simulations/
    └── optimizations/
```

## Implementation Status

See `STATUS.md` for detailed progress tracking.

### Completed
- ✅ Full maintenance model (Eq. 4c)
- ✅ Temperature dependence (Eq. 4e) - Eyring, Q10, Arrhenius models
- ✅ Two-phase feed strategy
- ✅ Initial conditions calibration from data
- ✅ Data loader with train/test split
- ✅ Parameter estimation framework (differential evolution)
- ✅ Model selection algorithm (F-test based)
- ✅ Visualization functions (6 plotting functions)

### In Progress
- 🔄 Parameter fitting on training data (differential evolution running)

### To Do
- ⏳ Model selection on fitted parameters
- ⏳ Validation on test data (ferm_10, 11, 12, ferm_opt)
- ⏳ Optimization implementation
- ⏳ Experimental validation

## Project Structure

```
optfed/
├── README.md                 # This file
├── STATUS.md                 # Implementation status
├── model.py                  # Core mechanistic model equations
├── fitting.py                # Parameter estimation (Stage II) + Model Selection
├── optimization.py           # Optimization framework (Stage III)
├── visualization.py          # Plotting functions for validation
├── data_loader.py            # Data loading and preprocessing
├── validate_model.py         # End-to-end validation script
├── test_optfed.py            # Test script
└── results/                  # Output folder
    ├── fitted_params.pkl
    ├── fitted_params.txt
    ├── selected_params.pkl
    ├── validation/
    │   ├── metrics.json
    │   └── plots/
    │       ├── parity_X.png
    │       ├── parity_P.png
    │       ├── parity_V.png
    │       ├── validation_summary.png
    │       └── *_comparison.png
    ├── simulations/
    └── optimizations/
```
