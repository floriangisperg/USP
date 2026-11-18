# USP — Upstream Processing analysis

Python package for analyzing **E. coli fermentation data** from bioreactors. It
ingests continuous bioreactor sensor data (online, CSV) and laboratory
measurements from samples (offline, Excel), aligns them in time, computes
derived bioprocess parameters (feed rates, reactor volume, gas-exchange rates,
carbon balance, growth/product yields, induction timing), and produces
overview plots and CSV result files.

## Features

- **Parsing** of bioreactor CSV and offline Excel sample sheets with
  configurable separators, date formats, and column mapping.
- **Cleaning** — phase filtering, balance tare correction, median filtering,
  process-time calculation.
- **Temporal alignment** of online and offline datasets with configurable
  tolerance.
- **Processing** — feed/base/volume tracking, off-gas rates (CER, OUR, RQ),
  specific rates (qO2, qCO2), carbon balance, growth and production rates,
  induction detection and protein parameters.
- **Visualization** — multi-parameter time-series overview plots.
- **Hierarchical YAML configuration** — a default config overridden per run.
- **Validation & typed exceptions** — clear errors instead of silent failures.

## Installation

Requires Python ≥ 3.12. Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                    # create env and install from uv.lock
# or, with extras:
uv pip install -e ".[dev]"
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `modeling` extra (`pip install -e ".[modeling]"`) adds `cma` and `catalax`,
used only by the parameter-fitting / modeling playground.

## Quickstart

The fastest way to see the full pipeline is the real-data example:

```bash
cd examples/real_data_example
python real_data_analysis.py
```

**Bringing your own fermentation data?** Follow
[`docs/YOUR_OWN_DATA.md`](docs/YOUR_OWN_DATA.md) and scaffold a run folder
with `python scripts/new_run.py <name>`.

This parses `data/rawdata.csv` + `data/samples.xlsx`, runs the full pipeline
using `run_config.yaml`, and writes results to `results/`:

- `processed_data.csv` — aligned online + offline data with all derived columns
- `samples_points_data.csv` — the offline sample points
- overview plots (`.png`)

### Using the library in your own script

```python
from pathlib import Path
from upstream_analysis import (
    load_config,
    BioreactorDataParser,
    OfflineMeasurementsParser,
    FermentationDataCleaner,
    DataAligner,
    FermentationDataProcessor,
    DataVisualizer,
)

config = load_config("run_config.yaml")            # merges with defaults

online  = BioreactorDataParser(config).parse_file("data/rawdata.csv")
offline = OfflineMeasurementsParser(config).parse_file("data/samples.xlsx")

online = FermentationDataCleaner(config).clean_data(online)
df_merged, df_samples = DataAligner(config["alignment"]).merge_datasets(online, offline)

processed = FermentationDataProcessor(config).process_data(df_merged)

output_dir = Path("results")
DataVisualizer(output_dir).plot_selected_parameters(
    processed,
    parameters=["Balance [g]", "Feed [g]", "Volume [mL]", "Biomass [g/L]",
                "CER [mmol/L/h]", "OUR [mmol/L/h]", "RQ [-]"],
    time_col="Process time [h]",
)
processed.to_csv(output_dir / "processed_data.csv", index=False, sep=";", decimal=",")
```

## Examples

| Directory | What it shows |
|---|---|
| `examples/real_data_example/` | Minimal end-to-end run — ships its own `data/`, runnable as-is |
| `examples/doe_example/` | 12 scFv DoE fermentation runs; `python analyze_doe_run.py ferm_01` analyzes one run. Consolidated raw exports live in `all_data/` |
| `examples/hgh/` | Two hGH fed-batch runs (R5 low/low, R6 high/high): analyze each run, then `consolidate_results.py` builds combined workbooks + a bokeh comparison dashboard |
| `examples/LDH/` / `examples/Xiso/` | Per-product analyses: run scripts plus `consolidate_results.py` each |
| `modeling/` | **Tracked** home of mature modeling analyses (mechanistic IB model, LDH fit lineage v5→v6, IB solubility study with report) — promoted out of the playground; figures local-only. See [`modeling/README.md`](modeling/README.md) |
| `modeling_playground/` | Active parameter-fitting sandbox (cma + catalax; `modeling` extra) — scratch work, **not tracked in git** |

Experiment data (`examples/**/data/`, `results/`, `all_data/`) is gitignored
and lives outside the repo; `real_data_example` intentionally ships its data
via `git add -f`.

## Architecture

A modular pipeline, each stage taking the merged config and returning a
DataFrame:

```
raw files → Parser → Cleaner → Aligner → Processor → Visualizer → results/
```

| Stage        | Module                        | Responsibility                                  |
|--------------|-------------------------------|-------------------------------------------------|
| Parsing      | `data/parser.py`              | Ingest CSV (bioreactor) + Excel (offline)       |
| Cleaning     | `data/cleaning.py`            | Spike removal, tare correction, time calc       |
| Alignment    | `data/alignment.py`           | Merge online + offline by time                  |
| Processing   | `data/processor.py` (facade over `data/calculations/`) | Derived parameters (feed, volume, gas, yields) |
| Visualization| `visualization/plotting.py`, `visualization/dashboard.py` | Time-series plots + interactive HTML dashboards |

Supporting layers:

- `core/constants.py` — centralized physical constants (molar volumes, molar masses, defaults).
- `core/validation.py` — DataFrame/column/numeric validators.
- `core/exceptions.py` — `USPError` hierarchy (`USPConfigError`, `USPDataError`, `USPCalculationError`).
- `config/loader.py` — hierarchical YAML merge.

See [`AGENTS.md`](AGENTS.md) for the full developer guide and
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for every configuration option.

## Configuration

Two-level hierarchy, merged at runtime with run values taking precedence:

1. **Defaults** — [`src/upstream_analysis/config/default_config.yaml`](src/upstream_analysis/config/default_config.yaml)
2. **Run overrides** — `run_config.yaml` per experiment

Top-level sections: `parsing`, `column_mapping`, `cleaning`, `alignment`,
`processing`, `analysis`, `visualization`. Full parameter reference in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

### Experiment layout

```
examples/<example>/<run>/
├── run_config.yaml     # run-specific overrides
├── data/               # raw CSV / Excel files
└── results/            # generated outputs (CSV + plots)
```

## Testing

```bash
pytest                  # run the full suite (136 tests)
pytest --cov=upstream_analysis
```

Tests live in `tests/unit/` (per-module) and `tests/integration/` (end-to-end
pipeline), with shared in-memory fixtures in `tests/conftest.py`.

## Code quality

```bash
black src/ tests/          # package code: strict, 88 chars
isort src/ tests/
flake8 src/ tests/         # CI gate on the package
```

Analysis scripts under `examples/` and `modeling/` are format-checked with
black/isort but exempted from pedantic flake8 rules (see `.flake8`
per-file-ignores); `flake8 src/ tests/ examples/ modeling/` runs everything
locally.

Tool configuration lives in `pyproject.toml` (`[tool.black]`, `[tool.isort]`,
`[tool.pytest.ini_options]`) and `.flake8`.

## License

MIT — see `pyproject.toml`.
