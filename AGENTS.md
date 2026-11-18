# AGENTS.md

Guidance for Claude Code (and human contributors) working in this repository.
Start here for an overview; the user-facing README is [`README.md`](README.md).

## Project Overview

USP (Upstream Processing) is a Python package for analyzing **E. coli
fermentation data**. It processes bioreactor sensor data (online, CSV) and
laboratory measurements (offline, Excel) to calculate derived parameters like
feed rates, reactor volume, gas-exchange rates, carbon balance, and induction
states. Python ≥ 3.12.

## Common Development Commands

### Environment Setup

The project uses [uv](https://docs.astral.sh/uv/) (`uv.lock` is committed for
reproducibility):

```bash
uv sync                       # create env + install from uv.lock (recommended)
uv pip install -e ".[dev]"    # editable install with dev tools
```

Plain pip fallback:

```bash
pip install -e ".[dev]"       # runtime + dev deps
pip install -e ".[modeling]"  # adds cma + catalax (parameter fitting only)
pip install -e ".[dashboard]" # adds bokeh (interactive HTML dashboards)
```

### Running Examples

```bash
# Real-data example — self-contained, ships its own data/ and is runnable as-is
cd examples/real_data_example
python real_data_analysis.py

# DOE example — analyze a specific fermentation run
cd examples/doe_example
python analyze_doe_run.py ferm_01     # folder name is a CLI argument

# hGH example — two runs + consolidated comparison
cd examples/hgh
python analyze_hgh_run.py R5_lowlow   # or R6_highhigh (default)
python consolidate_results.py
```

Each example writes `processed_data.csv`, `samples_points_data.csv`, and
overview plots to a run-specific `results/` directory.

### Testing

```bash
pytest                        # full suite (136 tests)
pytest tests/unit             # per-module unit tests only
pytest --cov=upstream_analysis
```

Tests use **in-memory fixtures** (no external data files required), defined in
`tests/conftest.py`. Layout:

```
tests/
├── conftest.py              # shared fixtures (configs, sample DataFrames)
├── fixtures/                # static config fixtures
├── unit/                    # test_parser, test_cleaning, test_alignment,
│                            #   test_processor, test_visualization, test_dashboard
└── integration/             # test_full_pipeline (end-to-end)
```

### Code Quality

```bash
black src/ tests/          # package code: strict, 88 chars, CI-gated
isort src/ tests/
flake8 src/ tests/
```

Analysis scripts under `examples/` and `modeling/` are format-checked
(black/isort) but exempted from pedantic flake8 codes via `.flake8`
per-file-ignores — they are research scripts, not package code.

Configuration: `[tool.black]`, `[tool.isort]`, `[tool.pytest.ini_options]` in
`pyproject.toml`; flake8 in `.flake8` (max line length 100 to tolerate legacy
docstring prose, black-compatible code, per-file exemptions).

## Architecture Overview

Modular pipeline; each stage takes the merged config and returns a DataFrame.

### Core Processing Pipeline

1. **Parsing** (`data/parser.py`) — raw data ingestion from CSV (bioreactor)
   and Excel (offline samples), via `BioreactorDataParser` /
   `OfflineMeasurementsParser`.
2. **Cleaning** (`data/cleaning.py`) — phase filtering, balance tare
   correction, median filtering, process-time calculation
   (`FermentationDataCleaner`).
3. **Alignment** (`data/alignment.py`) — temporal merge of online + offline
   data (`DataAligner.merge_datasets` → `(df_merged, df_samples)`).
4. **Processing** (`data/processor.py` + `data/calculations/`) — derived
   parameters: feed, base, volume, induction, protein, off-gas rates
   (CER/OUR/RQ), specific rates, carbon balance, growth/product yields
   (`FermentationDataProcessor`).
5. **Visualization** — per-parameter time-series plots
   (`visualization/plotting.py`, `DataVisualizer.plot_selected_parameters`)
   and self-contained interactive HTML dashboards
   (`visualization/dashboard.py`, `InteractiveDashboard`, optional `bokeh`
   via the `dashboard` extra).

Supporting layers under `core/`:

- `core/constants.py` — centralized physical constants (molar volumes, molar
  masses, process defaults). No magic numbers in calculations.
- `core/validation.py` — DataFrame/column/numeric/config validators.
- `core/exceptions.py` — `USPError` hierarchy: `USPConfigError`,
  `USPDataError`, `USPCalculationError`.

> Note: `data/processor.py` is a thin facade (~170 lines) over the focused
> calculation modules in `data/calculations/` (`feed`, `volume`, `induction`,
> `protein`, `gas`, `carbon_balance`, `rates_yields`; shared config access in
> `_config`). `FermentationDataProcessor` keeps the historical API
> (`process_data(df)` plus per-step `_<name>(df)` methods the unit tests
> exercise) and delegates each step. The **order of steps in `process_data`
> is load-bearing** (later calculations consume columns built earlier) —
> preserve it when adding a calculation, and put the math in
> `data/calculations/`, not in the facade.

### Configuration System

- **Default config**: [`src/upstream_analysis/config/default_config.yaml`](src/upstream_analysis/config/default_config.yaml)
- **Run configs**: run-specific YAML in example directories (`run_config.yaml`)
- **Merging**: experiment configs override defaults via hierarchical merge in
  [`config/loader.py`](src/upstream_analysis/config/loader.py) (`load_config`)

Top-level sections: `parsing`, `column_mapping`, `cleaning`, `alignment`,
`processing`, `analysis`, `visualization` (last two are placeholders). Full
parameter reference in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

### Data Flow

```
raw files → Parser → Cleaner → Aligner → Processor → Visualizer → results/
```

- Each component takes the merged config and returns processed DataFrames.
- Final output: `processed_data.csv` and `samples_points_data.csv` (CSV with
  `sep=';'`, `decimal=','`) plus `.png` plots in run-specific `results/`.

### Key Components

- **`FermentationDataProcessor`** — facade orchestrating the
  `data/calculations/` modules (feed, base, volume, induction, protein,
  off-gas, carbon balance, rates/yields) in a fixed sequence.
- **`FermentationDataCleaner`** — cleaning and process-time derivation.
- **`BioreactorDataParser`** / **`OfflineMeasurementsParser`** — CSV/Excel
  parsing with configurable separators and date formats.
- **`DataAligner`** — temporal alignment with configurable tolerance.
- **`DataVisualizer`** — multi-parameter time-series plotting.

All of the above are re-exported from `upstream_analysis/__init__.py`.

### Experiment Structure

```
examples/<example>/<run>/
├── run_config.yaml     # run-specific parameter overrides
├── data/               # raw CSV / Excel files (gitignored — large/proprietary)
└── results/            # generated outputs (CSV + plots, gitignored)
```

> `examples/**/data/` and `**/results/` outputs are gitignored. To ship data
> intentionally (as `real_data_example` does), use `git add -f`.

## Notes for Contributors

- Derived calculations live in `data/calculations/` (feed, volume, induction,
  protein, gas, carbon_balance, rates_yields). Add new math there — the
  `FermentationDataProcessor` facade in `data/processor.py` only orchestrates
  the steps, in a load-bearing order.
- Add physical constants to `core/constants.py` rather than inlining magic
  numbers.
- Raise typed exceptions from `core/exceptions.py` and validate inputs with
  `core/validation.py` — avoid silent NaN returns.
- Keep the test suite green: run `pytest` before committing.
