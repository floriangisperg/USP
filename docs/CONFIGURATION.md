# USP Configuration Guide

## Overview

USP uses a hierarchical configuration system with default values that can be overridden at the run level. This design allows you to maintain consistent analysis parameters across experiments while accommodating run-specific variations.

The configuration system follows a two-level hierarchy:

1. **Default Configuration** (`src/upstream_analysis/config/default_config.yaml`) - Contains all default values
2. **Run Configuration** (`examples/*/run_config.yaml`) - Contains only run-specific overrides

At runtime, these configurations are merged with run-specific values taking precedence.

## Configuration Structure

### Top-Level Sections

```yaml
parsing:           # File format settings
column_mapping:    # Raw to standardized column name mapping
cleaning:          # Data quality improvements
alignment:         # Temporal matching settings
processing:        # Calculation parameters
analysis:          # Placeholder for future features
visualization:     # Placeholder for future features
```

## Detailed Configuration Sections

### 1. Parsing (`parsing`)

Controls how raw data files are read and interpreted.

#### Bioreactor Data (Online)
```yaml
parsing:
  bioreactor:
    decimal_separator: ','          # Decimal point character in CSV
    column_separator: ';'           # Column delimiter in CSV
    date_format: '%d.%m.%Y %H:%M'   # Datetime parsing format
    skiprows: 1                     # Rows to skip before header
```

**Common Issues:**
- **Encoding errors with temperature**: If your file has encoding issues with the degree symbol (°), the column mapping includes fallback mappings for common encoding artifacts
- **Date format mismatches**: Ensure `date_format` matches your CSV's datetime format exactly

#### Offline Measurements
```yaml
parsing:
  offline:
    decimal_separator: '.'          # Decimal point character in Excel
    column_separator: ','           # Column delimiter in Excel
    date_format: 'guess'            # Auto-detect datetime format
    sheet_name: 0                   # Excel sheet to read (0 = first sheet)
```

### 2. Column Mapping (`column_mapping`)

Maps raw column headers from your data files to standardized internal names used throughout the package.

**Why This Matters:** Different bioreactor systems and lab protocols use different naming conventions. The column mapping provides a translation layer so the analysis code remains independent of data source conventions.

**Structure:**
```yaml
column_mapping:
  'Raw Header from CSV': standard_internal_name
  'Temperature, °C': Temperature [°C]
  'pH, -': pH [-]
```

**Partial Override:** In run-specific configs, you only need to specify mappings that differ from defaults:

```yaml
# In run_config.yaml
column_mapping:
  'My Custom pH Header': pH [-]
  # All other mappings inherited from default_config.yaml
```

### 3. Cleaning (`cleaning`)

Data quality improvements applied before analysis.

```yaml
cleaning:
  filter_preparation: true               # Remove 'Preparation' phase data
  rename_preparation_to_batch: false     # Don't rename phases
  balance_tare_correction: true          # Apply tare correction to balance data
  calculate_process_time: true           # Calculate process time from timestamps
  median_filter: true                    # Apply median filtering for spike removal
  median_filter_columns: ['balance']     # Columns to filter
  median_filter_window: 10               # Window size for median filter
  median_filter_threshold: 10            # MAD-based threshold for outliers
  balance_tare_threshold: -100           # Tare correction threshold (g)
```

**Balance Tare Correction:**
- Removes abrupt weight changes (taring events) from balance data
- Uses a threshold to identify when the balance was zeroed
- Critical for accurate feed volume calculations

### 4. Alignment (`alignment`)

Settings for merging online and offline datasets by time.

```yaml
alignment:
  online_time_col: 'Process time [h]'       # Time column in online data
  offline_time_col: 'Process time [h]'      # Time column in offline data
  tolerance_hours: 0.1                      # Max time difference for matching (6 min)
  save_separate_samples_file: true          # Create samples-only output
  force_align_last_sample: true             # Ensure last sample is included
```

**How Alignment Works:**
1. Both datasets must have a common time unit (typically "Process time [h]")
2. Offline measurements are matched to the nearest online timestamp within tolerance
3. Matched samples are flagged with `is_sample = True`

### 5. Processing (`processing`)

The core calculation parameters for derived values.

#### Processing Parameters

```yaml
processing:
  processing_parameters:
    balance_mode: 'feed_source'           # How to interpret balance data
```

**Balance Mode:**
- `'feed_source'` (default): Balance measures feed bottle weight (decreasing)
- `'reactor_weight'`: Balance measures reactor weight (increasing)

This parameter affects:
- Tare correction direction
- Feed calculation sign convention
- Median filtering behavior

#### Reactor Setup

```yaml
reactor_setup:
  initial_volume_ml: 1100.0               # Starting batch volume (REQUIRED)
```

**Why This Matters:** All feed and sample volume calculations start from this initial value.

#### Gas Parameters

```yaml
gas_params:
  # Gas composition (percentages)
  o2_in_air_percent: 21.0                 # O2 fraction in dry air
  co2_in_air_percent: 0.04                # CO2 fraction in dry air
  
  # Water vapor (optional, for advanced gas calculations)
  inlet_h2o_mol_pct: 0.0                  # Inlet H2O vapor (% mol/mol)
  outlet_h2o_mol_pct: 0.0                 # Outlet H2O vapor (% mol/mol)
  offgas_is_wet: false                    # True if O2/CO2 are on wet basis
  
  # N2 balance method (alternative gas rate calculation)
  use_n2_balance: false                   # Use N2 balance for CER/OUR
  
  # Physical properties
  molar_volume_air_l_mol: 22.414          # Molar volume at STP (L/mol)
  
  # Calculation switches
  calculate_inlet_composition: true       # Calculate inlet gas composition
  calculate_cer: true                     # CO2 Evolution Rate
  calculate_our: true                     # O2 Uptake Rate
  calculate_rq: true                      # Respiratory Quotient (CER/OUR)
  calculate_specific_rates: true         # Per-biomass rates (qO2, qCO2)
```

**Gas Rate Calculations:**
- **CER (CO2 Evolution Rate)**: CO2 production rate in mmol/L/h
- **OUR (O2 Uptake Rate)**: O2 consumption rate in mmol/L/h
- **RQ (Respiratory Quotient)**: Ratio of CER to OUR

**Run-Specific Override Example:**
```yaml
# In run_config.yaml - for actual reactor conditions
gas_params:
  inlet_o2_mol_pct: 21.0
  inlet_co2_mol_pct: 0.04
  inlet_h2o_mol_pct: 0.00476             # Actual humidity at reactor conditions
  molar_volume_air_l_mol: 25.291         # At reactor temperature
  molar_volume_temp_c: 35.0              # Temperature for molar volume
  air_flow_l_min: 0.12                   # Actual airflow rate
```

#### Feed Definitions

```yaml
feed_definitions:
  feed_1:                                 # Feed linked to feed pump (e.g., Pump4)
    name: 'Glucose Feed'                  # Descriptive name
    density_g_ml: 1.150                   # Feed density (g/mL)
```

**Why Density Matters:** Converts pump volume readings to mass for carbon balance calculations.

#### Medium Definitions

```yaml
medium_definitions:
  density_g_ml: 1.090                     # Batch medium density (optional)
```

If not specified, uses density from the `media_config_file` (if defined).

#### Base Titrant

```yaml
base_titrant:
  compound: 'NH4OH'                       # Base chemical formula
  molarity_mol_l: 6.95                    # Concentration (mol/L)
  density_g_ml: 0.947                    # Density (g/mL)
  pump_channel: 'Pump2'                  # Links to column mapping
  total_added_g: 0                        # Total base added (g)
```

**Why This Matters:** Base addition affects reactor volume and nitrogen balance calculations.

#### Biomass Parameters

```yaml
biomass_params:
  elemental_composition_c_mol:           # Elemental formula per C-mol
    C: 1.0
    H: 1.77
    N: 0.24
    O: 0.49
  molar_mass_g_c_mol: 26.565             # Molar mass per C-mol
  ash_fraction_g_g: 0.08                  # Non-organic fraction
```

**Default Values:** Based on typical E. coli composition (CH1.77N0.24O0.49).

#### Product Parameters

```yaml
product_params:
  name: 'scFv-His'                        # Product identifier
  nitrogen_content_fraction: 0.1647       # gN per g product (REQUIRED for yields)
```

**Nitrogen Content:** Required for nitrogen balance calculations and yield determinations.

#### Induced Phases

```yaml
induced_phases:
  - 'induced Fedbatch'
  - 'Induced fed batch'
  - 'Production'
```

**Purpose:** Marks time periods where induction is active for calculating induction-specific metrics.

#### Induction Time Reference (optional)

The process-control phase transition into the induced phase can **lead** the
physical inducer addition: the recipe switches the phase (and feed profile)
automatically, while IPTG is dosed manually afterwards. Observed offsets range
from a few minutes to more than an hour (e.g. when the phase flips before the
end-of-fedbatch sample is drawn). Two optional keys shift the induction start
to the operator-verified moment:

```yaml
induction_time_h: 22.683                  # Reference induction time (process-time hours)
# -- alternative to induction_time_h: read it from an evaluation workbook --
induction_reference_file: 'Evaluation_F16.xlsx'
induction_reference_sheet: 'Time-Rownr'   # default
induction_reference_label: 'Induction'    # row label in the sheet's first column
induction_time_tolerance_h: 0.25          # warn threshold vs the phase flip [h]
```

**Behavior:** Without a reference, the detected phase-flip time is only logged.
With a reference, `Is induced [-]` (and `Induction time [h]`) are re-derived
from it; an offset beyond `induction_time_tolerance_h` produces a warning
(the correction is applied regardless). A reference outside the data's time
range raises `USPConfigError`.

#### Filtering Settings

```yaml
filtering:
  apply_feed_filter: true                 # Apply median filter to feed calculations
  median_filter_window: 5                 # Window size for feed filter
  median_filter_threshold: 10            # Outlier threshold
```

#### Rate Parameters

```yaml
rate_params:
  apply_smoothing: true                   # Smooth calculated rates
  smoothing_window_points: 5             # Smoothing window
  smoothing_min_periods: 1                # Minimum periods for rolling calc
  interpolation_method: 'spline'          # 'spline' or 'linear'
  spline_order: 3                         # Spline polynomial order
  spline_smoothing: 0.0                   # Spline smoothing factor (0 = exact fit)
```

**Rate Calculation:** Calculates time derivatives (d/dt) for parameters like biomass growth rate.

## Run-Specific Configuration

Each fermentation run should have its own `run_config.yaml` in the run directory:

```yaml
# examples/doe_example/ferm_01/run_config.yaml

metadata:
  fermentation_id: 'Run_01'
  date_start: '2024-01-31'
  strain: 'E. coli BL21DE3 (pET)'
  reactor: 'MF2 R1'

processing:
  processing_parameters:
    reactor_setup:
      initial_volume_ml: 1100.0           # REQUIRED for this run
    
    gas_params:                           # REQUIRED for gas calculations
      inlet_o2_mol_pct: 21.0
      inlet_co2_mol_pct: 0.04
      inlet_h2o_mol_pct: 0.00476
      molar_volume_air_l_mol: 25.291
    
    feed_definitions:
      feed_1:
        name: 'Glucose Feed'
        density_g_ml: 1.150
    
    base_titrant:
      compound: 'NH4OH'
      molarity_mol_l: 6.95
      density_g_ml: 0.947
      pump_channel: 'Pump2'
    
    product_properties:
      name: 'scFv-His'
      nitrogen_content_gN_gP: 0.1647
```

## Backward Compatibility

The configuration system supports both old and new structures:

### Old Structure (Deprecated)
```yaml
# Single-level processing parameters
processing:
  reactor_setup: {...}
  gas_params: {...}
  feed_definitions: {...}
```

### New Structure (Recommended)
```yaml
# Nested under processing_parameters
processing:
  processing_parameters:
    reactor_setup: {...}
    gas_params: {...}
    feed_definitions: {...}
```

**The loader automatically handles both formats** - no manual migration needed.

## Common Issues and Solutions

### Issue: "Unknown config section 'xxx'"

**Cause:** Typo in config section name or using an unsupported section.

**Solution:** 
- Check spelling against `default_config.yaml`
- Ensure section name is one of: `parsing`, `column_mapping`, `cleaning`, `alignment`, `processing`, `analysis`, `visualization`

### Issue: "Missing required parameter: 'initial_volume_ml'"

**Cause:** Run config doesn't specify the initial reactor volume.

**Solution:** Add to your run config:
```yaml
processing:
  processing_parameters:
    reactor_setup:
      initial_volume_ml: 1100.0
```

### Issue: Gas calculations returning NaN or zeros

**Cause:** Missing or incorrect gas parameters.

**Solution:** Verify these are set in your run config:
```yaml
processing:
  processing_parameters:
    gas_params:
      molar_volume_air_l_mol: 25.291
      inlet_o2_mol_pct: 21.0
      air_flow_l_min: 0.12
```

### Issue: Column not found after parsing

**Cause:** Column mapping doesn't match your file headers.

**Solution:** 
1. Check your raw file headers
2. Add specific mapping to run config:
```yaml
column_mapping:
  'Your Actual Header': 'Standard Name'
```

### Issue: Inconsistent defaults between runs

**Cause:** Default values changed between package versions.

**Solution:** 
- Explicitly specify critical parameters in run configs
- Pin package version in requirements.txt
- Review changelog for version updates

## Configuration Validation

The package validates configurations at load time and raises `USPConfigError` for:
- Missing required parameters
- Invalid parameter types
- Out-of-range values
- Unknown configuration sections

**Error Messages Include:**
- Parameter name
- Expected value/type
- Actual value received
- Suggested fix

## Best Practices

1. **Specify Run-Specific Values:** Always override parameters that vary between runs (initial volume, gas conditions, feed properties)

2. **Document Changes:** Add comments in run_config.yaml explaining why values differ from defaults

3. **Version Control:** Commit run configs to git alongside raw data for reproducibility

4. **Validate First:** Test new configs with sample data before processing full runs

5. **Keep Defaults Stable:** Only change `default_config.yaml` for universal improvements

## Scientific Context

### Balance Modes and Feed Calculation

The choice of `balance_mode` affects how feed is calculated from balance data:

- **Feed Source Mode** (`'feed_source'`): Balance weight decreases as feed is pumped out
  - Feed mass = initial weight - current weight - tare corrections
  
- **Reactor Weight Mode** (`'reactor_weight'`): Balance weight increases as feed is added
  - Feed mass = current weight - initial weight - tare corrections

This distinction is critical for accurate carbon balance calculations.

### Gas Rate Calculation Methods

**Standard Method (default):**
```
CER = (F_in * y_CO2_in - F_out * y_CO2_out) / V_liq
OUR = (F_in * y_O2_in - F_out * y_O2_out) / V_liq
```

**N2 Balance Method** (when `use_n2_balance: true`):
Uses nitrogen as an inert tracer to calculate outlet flow rate when direct measurement is unreliable.

### Carbon Balance Closure

For complete carbon balance analysis, ensure all of these are configured:
- Feed density and composition
- Biomass elemental composition
- Product nitrogen content
- Gas parameters (for C in CO2)
- Base addition (for N balance)

The system calculates:
- Carbon in: feed + batch medium
- Carbon out: biomass + product + CO2 + residual substrates
- Closure: (C_out / C_in) × 100%

## References

- Molar volume at STP: 22.414 L/mol (ideal gas, 0°C, 1 atm)
- E. coli composition: CH1.77N0.24O0.49 (per C-mol)
- Typical glucose feed: 440 g/L (high-concentration feeding solution)
