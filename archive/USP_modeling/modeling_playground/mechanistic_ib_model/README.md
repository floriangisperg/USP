# Mechanistic IB Formation Model

**A biophysically-motivated model of inclusion body formation in E. coli fed-batch fermentation, incorporating temperature × qs interaction effects.**

## Overview

This model implements a mechanistic description of protein partitioning between soluble and inclusion body (IB) fractions during recombinant protein expression in E. coli. The key innovation is the explicit modeling of both:

1. **Temperature effects** - on protein folding/unfolding thermodynamics
2. **qs effects** - on synthesis rate and folding capacity overload
3. **T × qs interaction** - synergistic effects at extreme conditions

## Model Structure

### IB Partitioning Equation

```
phi_IB = phi_synthesis(qs) + phi_aggregation(T, qs)
```

where:

**Synthesis Overload Component:**
```
phi_synthesis = k_syn * qs / (K_syn + qs)
```
- High qs → rapid protein synthesis → folding capacity overload
- Michaelis-Menten saturation behavior

**Thermal Aggregation Component:**
```
phi_aggregation = k_agg * f_T(T) * (1 + alpha_qs * qs)

where f_T(T) = exp(Ea/R * (1/T_ref - 1/T))
```
- High T → protein unfolding
- High qs → overwhelms chaperone system (amplifies aggregation)
- Arrhenius temperature dependence

### Full ODE System

States: `[X, P_sol, P_IB, V]`

- `dX/dt = (Y_XS * qs - m_s) * X - D * X`
- `dP_sol/dt = q_p * (1 - phi_IB) * X - D * P_sol`
- `dP_IB/dt = q_p * phi_IB * X - D * P_IB`
- `dV/dt = F`

## Parameters

| Category | Parameter | Meaning | Expected Range |
|----------|-----------|---------|----------------|
| **Growth** | `Y_XS` | Yield on glucose | 0.3-0.7 g/g |
| | `m_s` | Maintenance coefficient | 0-0.05 1/h |
| **Production** | `alpha` | Growth-associated formation | 0.01-0.3 g/g |
| | `beta` | Non-growth-associated | 0-0.1 g/(g·h) |
| **IB: Synthesis** | `k_syn` | Max synthesis misfolding | 0.01-0.5 |
| | `K_syn` | qs half-saturation | 0.05-0.5 g/g/h |
| **IB: Aggregation** | `k_agg` | Aggregation coefficient | 0.001-0.3 |
| | `T_ref` | Reference T | 20-28 °C |
| | `Ea/R` | Activation energy | 3000-8000 K |
| | `alpha_qs` | qs amplification | 0-3.0 |

**Total: 10 parameters**

## Key Findings from Data Analysis

### Temperature vs qs Effects

| Variable | Controls | Effect |
|----------|----------|--------|
| **Temperature** | IB FRACTION | T=25°C: 0% IB, T=30°C: ~13% IB, T=35°C: 100% IB |
| **qs** | PRODUCTION AMOUNT | Higher qs → more total protein (both IB and soluble) |

### Data Evidence

**At T=35°C (100% IB for both):**
- qs=0.2: Final IB = 1.74 g/L
- qs=0.6: Final IB = 2.85 g/L (**63% more!**)

**At T=25°C (0% IB for both):**
- qs=0.2: Final soluble = 3.09 g/L
- qs=0.6: Final soluble = 4.14 g/L (**34% more!**)

**At T=30°C (mixed):**
- IB fraction varies from 12-18% over time
- This is where T × qs interaction is most visible

## Usage

### Quick Start

```python
# Load data and fit model
from mechanistic_ib_model import prepare_full_run_data, fit_multi_run, plot_multi_run

# Load runs
data_list = [prepare_full_run_data(r) for r in ['ferm_01', 'ferm_02', 'ferm_03', 'ferm_04']]

# Fit model
params, loss = fit_multi_run(data_list, ['ferm_01', 'ferm_02', 'ferm_03', 'ferm_04'])

# Visualize
plot_multi_run(params, data_list, save_path='results/fit.png')
```

### Generate Visualizations

```bash
python visualize_effects.py
```

This generates:
- `ib_temp_response.png` - IB fraction vs T at different qs levels
- `ib_qs_response.png` - IB fraction vs qs at different T
- `ib_surface.png` - 2D surface response (T × qs)
- `ib_components.png` - Component breakdown (synthesis vs aggregation)

### Run Full Fitting

```bash
python mechanistic_ib_model.py
```

This will:
1. Load all 7 fermentation runs
2. Fit the 10-parameter model using differential evolution
3. Generate fit plots
4. Save fitted parameters to `results/mechanistic_ib_params.txt`

## File Structure

```
mechanistic_ib_model/
├── mechanistic_ib_model.py    # Main model and fitting code
├── visualize_effects.py        # Visualization tools
├── README.md                   # This file
└── results/                    # Generated outputs
    ├── mechanistic_ib_params.txt
    ├── mechanistic_ib_fit.png
    ├── ib_temp_response.png
    ├── ib_qs_response.png
    ├── ib_surface.png
    └── ib_components.png
```

## Comparison with Previous Models

| Feature | v6 (Quadratic) | Mechanistic (This) |
|---------|----------------|-------------------|
| IB Model | `a * (T - T_onset)^2` | `phi_synthesis(qs) + phi_aggregation(T, qs)` |
| Parameters | 5 | 10 |
| qs Effect | Implicit (via growth) | Explicit (synthesis + aggregation) |
| T × qs Interaction | No | Yes |
| Mechanistic Basis | Empirical | Biophysical |

## Next Steps

1. **Complete fitting** - Run full optimization on all 7 runs
2. **Parameter sensitivity** - Identify which parameters are well-constrained
3. **Model validation** - Test predictions on new conditions
4. **Uncertainty quantification** - Use MCMC or bootstrap for confidence intervals

## Authors

Modeling Playground - USP Project

## References

- Data from E. coli fed-batch DOE study (IPTG-induced recombinant protein expression)
- Based on biophysics of protein folding and aggregation
