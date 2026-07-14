# USP Examples

This directory contains comprehensive examples demonstrating how to use the USP (Upstream Processing) package for E. coli fermentation data analysis.

## Overview

Each example demonstrates different aspects of the package:
- **DOE Example**: Design of Experiments with standardized analysis
- **LDH Example**: Real production data with advanced features  
- **Real Data Example**: Template for analyzing your own data

---

## DOE Example

**Location:** `examples/doe_example/`

Demonstrates analysis of Design of Experiments fermentation runs with systematic parameter variations and standardized data formats.

### Usage

```bash
cd examples/doe_example
python analyze_doe_run.py
```

### Configuration

Before running, modify the script to specify which run to analyze:

```python
# In analyze_doe_run.py, line 24
target_run_folder_name: str = "ferm_12"  # Change this to your run
```

### Directory Structure

```
doe_example/
├── analyze_doe_run.py              # Main analysis script
├── ferm_01/                        # Individual run folders
│   ├── run_config.yaml            # Run-specific configuration
│   ├── data/                      # Raw data files
│   │   ├── bioreactor.csv         # Online bioreactor data
│   │   └── samples.xlsx           # Offline measurements
│   └── results/                   # Generated outputs (auto-created)
│       ├── processed_data.csv
│       ├── samples_points_data.csv
│       └── *.png                  # Time-series plots
├── ferm_02/
├── ferm_03/
│   ...
└── ferm_opt/                      # Optimization runs
```

### What it does

- Analyzes a specific fermentation run
- Processes online bioreactor data and offline measurements
- Calculates derived parameters (feed rates, gas exchange, carbon balance)
- Generates time-series plots and results CSVs

### Expected Output

Each analysis generates three types of outputs:

1. **`results/processed_data.csv`** - Complete time-series dataset
   - All calculated parameters
   - Online and offline data merged by time
   - Sample points flagged with `is_sample = True`

2. **`results/samples_points_data.csv`** - Sample points only
   - Offline measurements
   - Calculated values at sample times
   - No interpolation (raw data only)

3. **`results/*.png`** - Time-series plots for each parameter
   - Biomass and substrates
   - Products and yields
   - Offgas rates (CER, OUR, RQ)
   - Volume tracking
   - Process indicators

### Features Demonstrated

- Standardized data format handling
- Configuration inheritance from defaults
- Automatic data file detection
- Comprehensive parameter plotting
- Sample alignment with tolerance

---

## LDH Example

**Location:** `examples/LDH/`

Real production data analysis for lactate dehydrogenase (LDH) expression with advanced carbon balance features and Excel format handling.

### Usage

```bash
cd examples/LDH
python analyze_ldh_run.py
```

### Configuration

The script can analyze a single run or all runs:

```python
# In analyze_ldh_run.py, line 29
target_run_folder_name: str = "all"  # Options: "all", "ferm_01", "F07", etc.
```

### Directory Structure

```
LDH/
├── analyze_ldh_run.py              # Main analysis script
├── analyze_partition_temp.py       # Temperature partition analysis
├── analyze_partition_temp_bounded.py # Bounded temperature analysis
├── F07/                            # Production run folders
│   ├── run_config.yaml
│   ├── data/
│   │   ├── Rawdata.xlsx           # Online bioreactor data (Excel format)
│   │   └── samples.xlsx           # Offline measurements
│   └── results/
│       └── processed_data.xlsx    # Excel output with multiple sheets
├── F09/
│   ...
└── F15/
```

### Features

- **Real-world data patterns**
- **Handling of missing measurements**
- **Process phase transitions**
- **Sample timing variations**
- **Excel data input/output**
- **Advanced carbon balance calculations**

### Special Features

1. **Excel Data Input**
   - Bioreactor data in Excel format (not CSV)
   - Multiple sheet handling
   - Encoding robustness

2. **Carbon Balance Calculations**
   - Complete carbon tracking
   - C in: feed, batch, CO2
   - C out: biomass, product, acetate, residual glucose
   - Balance closure percentage

3. **Advanced Output**
   - Excel format with multiple sheets
   - Separate "Sample_Points" sheet
   - Comprehensive parameter set

4. **Batch Processing**
   - Analyze all runs with one command
   - Success/failure reporting
   - Consistent output across runs

### Carbon Balance Parameters

The LDH example calculates advanced carbon balance parameters:

```
C in [C-mol]:
- C from feed
- C from batch medium
- C from CO2 (offgas)

C out [C-mol]:
- C in biomass
- C in product (LDH)
- C in acetate
- C residual glucose

Balance:
- C recovered [%]
- C balance closure [%]
- C to CO2 [% of fed]
- C to biomass [% of fed]
- C to product [% of fed]
```

### Expected Output

**Excel file with multiple sheets:**
- `Processed_Data`: Complete time-series
- `Sample_Points`: Sample data only

**Additional calculated parameters:**
- Protein fractions (soluble, IB, total)
- Yields (g product / g substrate)
- Carbon balance components
- Specific rates (qO2, qCO2)

---

## Real Data Example

**Location:** `examples/real_data_example/`

Template for analyzing your own fermentation data with actual measurements and process variations.

### Usage

```bash
cd examples/real_data_example
python real_data_analysis.py
```

### Setup Requirements

1. **Create run directory structure:**
```bash
real_data_example/
├── real_data_analysis.py           # Your analysis script
├── run_config.yaml                 # Your configuration
├── data/
│   ├── your_bioreactor_data.csv   # Your online data
│   └── your_samples.xlsx          # Your offline data
└── results/                       # Auto-created
```

2. **Configure your data:**
   - Update column mappings in `run_config.yaml`
   - Set run-specific parameters (initial volume, feed density, etc.)
   - Adjust parsing settings if needed

---

## Data Requirements

### Online Data (Bioreactor)

**File format:** CSV (DOE) or Excel (LDH)

**Required columns** (after column mapping):
- `Process time [h]`: Time vector
- `Process phase [-]`: Phase identifiers (Batch, Fedbatch, etc.)
- `Balance [g]`: Feed balance for feed calculation
- `Air flow [L/min]`: Air flow rate
- `Total flow [L/min]`: Total exhaust gas flow

**Optional but recommended:**
- `Offgas O2 [%]`: Exhaust O2 percentage
- `Offgas CO2 [%]`: Exhaust CO2 percentage
- `O2 flow [L/min]`: Pure O2 flow (if used)
- `Base pump duration [-]`: Cumulative base pump
- `Feed pump duration [-]`: Cumulative feed pump
- `pH [-]`: pH value
- `pO2 [%]`: Dissolved oxygen
- `Temperature [°C]`: Temperature

### Offline Data (Excel)

**File format:** Excel with sample sheet

**Required columns:**
- `Process time [h]`: Time of sample (hours)

**Sample measurements** (varies by experiment):
- `OD600`: Optical density for biomass
- `Biomass [g/L]`: Direct biomass measurement
- `Protein [g/L]`: Total protein concentration
- `Protein soluble [g/L]`: Soluble protein fraction
- `Protein IB [g/L]`: Inclusion body fraction
- `Glucose [g/L]`: Residual glucose concentration
- `Acetate [g/L]`: Acetate concentration
- `Sample volume [mL]`: Volume removed for sample

---

## Output Data

### Processed Data Files

**CSV format** (DOE example):
```
results/processed_data.csv
results/samples_points_data.csv
```

**Excel format** (LDH example):
```
results/processed_data.xlsx (with multiple sheets)
```

### Data Content

**Time vectors:**
- `Process time [h]`: Time from inoculation
- `Induction time [h]`: Time from induction start
- `Phase time [s]`: Time within current phase

**Process indicators:**
- `Is induced [-]`: Boolean for induction phase
- `Process phase [-]`: Current phase name

**Volume tracking:**
- `Volume [mL]`: Calculated reactor volume
- `Volume corrected [mL]`: With sample removal
- `Feed [mL]`: Cumulative feed volume
- `Base volume added [mL]`: Cumulative base volume
- `Cumulative sample volume [mL]`: Total samples removed

**Concentrations:**
- `Biomass [g/L]`: Calculated biomass
- `Glucose [g/L]`: Residual glucose
- `Acetate [g/L]`: Acetate concentration
- `Protein [g/L]`: Total protein
- `Protein soluble [g/L]`: Soluble fraction
- `Protein IB [g/L]`: Inclusion bodies

**Rates:**
- `CER [mmol/L/h]`: CO2 evolution rate
- `OUR [mmol/L/h]`: O2 uptake rate
- `RQ [-]`: Respiratory quotient
- `qO2 [mmol/g/h]`: Specific O2 uptake
- `qCO2 [mmol/g/h]`: Specific CO2 production

**Yields:**
- `Protein total yield [g/g]`: Total product yield
- `Protein soluble yield [g/g]`: Soluble fraction yield
- `Protein IB yield [g/g]`: IB fraction yield

---

## Configuration

### Run Configuration (`run_config.yaml`)

Run-specific parameters override defaults:

```yaml
metadata:
  fermentation_id: "Ferm_001"
  date_start: "2025-01-15"
  strain: "E. coli BL21(DE3)"
  reactor: "2L Bioreactor"

processing:
  processing_parameters:
    reactor_setup:
      initial_volume_ml: 1200.0

    gas_params:
      molar_volume_air_l_mol: 22.4
      calculate_cer: true
      calculate_our: true
      calculate_rq: true
```

### Default Configuration

Default values are in `src/upstream_analysis/config/default_config.yaml`

---

## Common Tasks

### Analyze a Different Run
1. Open the example script (`analyze_doe_run.py` or `real_data_analysis.py`)
2. Change `target_run_folder_name` to your desired run
3. Run the script

### Modify Analysis Parameters
Edit the `run_config.yaml` in the specific run folder to adjust:
- Initial volume
- Gas parameters
- Feed properties
- Induction phases

### Generate Custom Plots
Modify the `parameters` list in the analysis script to plot specific parameters.

---

## Troubleshooting

### Common Issues

#### "No data in base addition phases"

**Cause:** Process phase column doesn't contain expected phase names

**Solution:**
- Verify your Process phase column contains 'Batch', 'Fedbatch', or 'induced Fedbatch'
- Check column mapping in config
- Add custom phase names to induced_phases if needed

#### "Feed calculated as 0"

**Cause:** Multiple possible issues

**Solution:**
1. Verify Balance column exists and contains numeric data
2. Check Process phase includes 'Fedbatch' or 'induced Fedbatch'
3. Ensure feed density is positive (> 0)
4. Check balance_mode is correct ('feed_source' vs 'reactor_weight')

#### "Cannot calculate offgas rates"

**Cause:** Missing gas flow or composition data

**Solution:**
- Verify Total flow column exists
- Verify Offgas O2 and CO2 columns exist
- Check column names match expected format
- Ensure Volume corrected has been calculated

#### "Column not found after parsing"

**Cause:** Column mapping doesn't match your file headers

**Solution:**
1. Check your raw file headers
2. Add specific mapping to run config:
```yaml
column_mapping:
  'Your Actual Header': 'Standard Name'
```

#### "Missing plots"

**Cause:** Multiple possible issues

**Solution:**
- Check matplotlib is installed
- Verify data is available for plotting
- Check logs for specific plotting errors
- Ensure output directory is writable

### Getting Help

1. **Check the logs** - Detailed error messages explain what went wrong
2. **Verify data format** - Ensure it matches requirements
3. **Compare with examples** - Use working examples as reference
4. **Review configuration** - Check `docs/CONFIGURATION.md`
5. **Validate data** - Ensure required columns exist

---

## Advanced Usage

### Custom Analysis Scripts

Create your own analysis workflow:

```python
from upstream_analysis.config.loader import load_config
from upstream_analysis.data.parser import BioreactorDataParser, OfflineMeasurementsParser
from upstream_analysis.data.cleaning import FermentationDataCleaner
from upstream_analysis.data.alignment import DataAligner
from upstream_analysis.data.processor import FermentationDataProcessor
from upstream_analysis.visualization.plotting import DataVisualizer

# Load configuration
config = load_config('path/to/run_config.yaml')

# Parse data
online_parser = BioreactorDataParser(config)
offline_parser = OfflineMeasurementsParser(config)
online_data = online_parser.parse_file('path/to/bioreactor.csv')
offline_data = offline_parser.parse_file('path/to/offline.xlsx')

# Clean data
cleaner = FermentationDataCleaner(config)
clean_online = cleaner.clean_data(online_data)
clean_offline = cleaner.clean_data(offline_data)

# Align data
aligner = DataAligner(config.get('alignment', {}))
merged_data, sample_data = aligner.merge_datasets(clean_online, clean_offline)

# Process data
processor = FermentationDataProcessor(config)
processed_data = processor.process_data(merged_data)

# Visualize
visualizer = DataVisualizer(output_dir='results/')
visualizer.plot_selected_parameters(
    df=processed_data,
    parameters=['Biomass [g/L]', 'Glucose [g/L]', 'CER [mmol/L/h]'],
    time_col='Process time [h]'
)

# Save results
processed_data.to_csv('results/my_analysis.csv', index=False)
```

### Batch Processing

Process multiple runs systematically:

```python
from pathlib import Path
import pandas as pd

# Find all run directories
run_dirs = Path('experiments').glob('run_*/')

results = []

for run_dir in run_dirs:
    config_file = run_dir / 'run_config.yaml'
    if config_file.exists():
        print(f"Processing {run_dir.name}...")
        
        # Load and analyze
        config = load_config(config_file)
        # ... run analysis pipeline ...
        
        # Collect results
        results.append({
            'run': run_dir.name,
            'max_biomass': df['Biomass [g/L]'].max(),
            'final_product': df['Protein [g/L]'].iloc[-1],
            'total_feed': df['Feed [mL]'].iloc[-1]
        })

# Summary table
summary = pd.DataFrame(results)
print(summary)
```

### Custom Parameters

Define custom calculation parameters:

```yaml
# In run_config.yaml
processing:
  processing_parameters:
    # Custom feed definitions
    feed_definitions:
      glucose_feed:
        name: 'Concentrated Glucose'
        density_g_ml: 1.150
      glycerol_feed:
        name: 'Glycerol Solution'
        density_g_ml: 1.260
    
    # Custom gas conditions
    gas_params:
      o2_in_air_percent: 20.9
      co2_in_air_percent: 0.03
      molar_volume_air_l_mol: 24.5
    
    # Custom product properties
    product_properties:
      name: 'My Enzyme'
      nitrogen_content_gN_gP: 0.182
```

---

## Best Practices

1. **Start with DOE example** - Cleaner data, easier to understand
2. **Check logs first** - Detailed logging helps diagnose issues
3. **Validate outputs** - Compare calculated values with expectations
4. **Use config wisely** - Run-specific configs override defaults
5. **Document custom settings** - Add comments explaining non-standard values
6. **Version control configs** - Commit run configs for reproducibility

---

## Scientific Context

These examples demonstrate standard bioprocess engineering analysis for E. coli fed-batch fermentations:

- **Feed control** - Exponential feeding to control growth rate
- **Gas analysis** - CER/OUR/RQ for metabolic state monitoring
- **Carbon balance** - Yield and metabolic flux analysis
- **Induction optimization** - Timing and protein production relationships

---

## Tips

1. **Start with DOE example** - Cleaner data, easier to understand
2. **Check logs first** - Detailed logging helps diagnose issues
3. **Validate outputs** - Compare calculated values with expectations
4. **Use config wisely** - Run-specific configs override defaults
5. **Version control configs** - Commit run configs for reproducibility

---

## Next Steps

After running examples:
1. Examine generated CSVs for calculated parameters
2. Review time-series plots for patterns
3. Compare different runs if available
4. Modify configurations and re-run

For detailed configuration options, see [docs/CONFIGURATION.md](../docs/CONFIGURATION.md).
