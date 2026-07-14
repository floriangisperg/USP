# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

USP (Upstream Processing) is a Python package for analyzing E. coli fermentation data. It processes bioreactor sensor data (online) and laboratory measurements (offline) to calculate derived parameters like feed rates, volume changes, and induction states.

## Common Development Commands

### Environment Setup

```bash
# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Running Examples

```bash
# Analyze a specific fermentation run (modify target_run_folder_name in script first)
cd examples/doe_example
python analyze_doe_run.py

# Real data analysis example
cd examples/real_data_example
python real_data_analysis.py
```

### Code Quality

```bash
# Format code with black
black src/ examples/

# Sort imports with isort
isort src/ examples/

# Lint with flake8
flake8 src/ examples/

# Run tests (when available)
pytest
```

## Architecture Overview

The package follows a modular pipeline architecture with distinct processing stages:

### Core Processing Pipeline

1. **Parsing** (`data/parser.py`) - Raw data ingestion from CSV (bioreactor) and Excel (offline samples)
2. **Cleaning** (`data/cleaning.py`) - Data quality improvements, spike removal, time calculation
3. **Alignment** (`data/alignment.py`) - Merging online and offline datasets by time
4. **Processing** (`data/processor.py`) - Calculating derived parameters (feed, volume, induction)
5. **Visualization** (`visualization/plotting.py`) - Generating overview plots

### Configuration System

- **Default config**: [`src/upstream_analysis/config/default_config.yaml`](src/upstream_analysis/config/default_config.yaml)
- **Experiment configs**: Run-specific YAML files in example directories (e.g., `run_config.yaml`)
- **Merging**: Experiment configs override defaults via hierarchical merging in [`config/loader.py`](src/upstream_analysis/config/loader.py:13)

### Data Flow

- Raw data files → Parsers → Cleaner → Aligner → Processor → Visualizer → CSV results
- Each component takes the merged config and returns processed DataFrames
- Final output includes `processed_data.csv` and `samples_points_data.csv` in run-specific `results/` directories

### Key Components

- **`FermentationDataProcessor`**: Core calculations for feed rates, base addition, reactor volume, and induction states
- **`BioreactorDataParser`**: Handles CSV parsing with configurable separators and date formats
- **`OfflineMeasurementsParser`**: Processes Excel sample sheets
- **`DataAligner`**: Temporal alignment of online/offline data with configurable tolerance
- **`DataVisualizer`**: Multi-parameter time series plotting

### Experiment Structure

Example runs follow this directory pattern:

```
examples/doe_example/
├── ferm_XX/
│   ├── run_config.yaml     # Run-specific parameters
│   ├── data/               # Raw CSV/Excel files
│   └── results/            # Generated outputs
```

### Configuration Parameters

Key sections in `default_config.yaml`:

- `parsing`: File format settings (separators, date formats)
- `column_mapping`: Maps raw headers to standardized names
- `cleaning`: Data preprocessing options
- `alignment`: Temporal matching settings
- `processing`: Biological and reactor parameters for calculations

When modifying analysis logic, focus on the processor module for derived calculations and the config system for parameter changes.