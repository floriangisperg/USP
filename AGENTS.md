# AGENTS.md

Guidance for Claude Code (and human contributors) working in this repository.
Start here for an overview; the user-facing README is [`README.md`](README.md).

## Scope

USP (Upstream Processing) is a Python package for analyzing **E. coli
fermentation data**: it ingests bioreactor sensor exports from Eve (online
CSV) and lab measurements from samples (offline Excel), aligns them, and
computes derived process parameters (feed, volume, gas-exchange rates,
carbon balance, yields, induction timing). That is the whole scope —
**data analysis, not modeling**. The mechanistic-modeling research that
grew out of this data was deliberately moved out (see
`../Archive/USP_modeling/`); do not add models, simulators, or fitting
code here. Python ≥ 3.12.

## Documentation map

| Document                                                     | What it covers                                                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| [`README.md`](README.md)                                     | Install, quickstart, feature overview                                                                  |
| [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md)                 | **Canonical** campaign/run folder convention — where files go, what git tracks                         |
| [`docs/YOUR_OWN_DATA.md`](docs/YOUR_OWN_DATA.md)             | Step-by-step: raw bioreactor export → results (column mapping, parsing, minimal config, sanity checks) |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)             | Every configuration option                                                                             |
| [`docs/VERIFICATION_STATUS.md`](docs/VERIFICATION_STATUS.md) | How to verify the package (tests, lint, CI)                                                            |
| [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md)              | Short factual record of structural work on the package                                                 |

## Common Development Commands

### Environment Setup

The project uses [uv](https://docs.astral.sh/uv/) (`uv.lock` is committed
for reproducibility):

```bash
uv sync --extra dev --extra dashboard   # full env: dev tools + bokeh
```

Two extras exist: `dev` (pytest, black, isort, flake8, pandas-stubs) and
`dashboard` (bokeh). The full 150-test suite and the consolidation
dashboards need **both** — without `dashboard`, the 11 dashboard tests
skip. Plain pip fallback: `pip install -e ".[dev]" ".[dashboard]"`.

### Verifying Your Setup

```bash
pytest   # 150 tests incl. end-to-end pipeline/consolidation checks, ~1 min
```

### Running Examples

The pipeline and consolidation live in the package — examples are data +
configs, no scripts:

```bash
# Synthetic run fixture — end-to-end coverage lives in the test suite
pytest tests/unit/test_pipeline.py tests/unit/test_consolidation.py

# One run of a campaign (folder names per example)
python -m upstream_analysis.pipeline examples/scFvM/ferm_01

# Consolidate a campaign into workbooks + comparison dashboard
python -m upstream_analysis.consolidation examples/hgh --prefix hGH
```

Every analysis writes `processed_data.csv`, `samples_points_data.csv` and
overview plots to a run-specific `results/` directory. The other examples
(LDH, Xiso) follow the same pattern; their raw data is gitignored and lives
only where the analyses were run.

### Testing

```bash
pytest                        # full suite (150 tests; 139 without bokeh)
pytest tests/unit             # per-module unit tests only
pytest --cov=upstream_analysis
```

Tests use **in-memory fixtures** (no external data files required), defined
in `tests/conftest.py`: `tests/unit/` per module (parser, cleaning,
alignment, processor, visualization, dashboard), `tests/integration/`
end-to-end, `tests/fixtures/` static configs.

### Code Quality (CI-gated)

```bash
black src/ tests/       # package code: strict, 88 chars
isort src/ tests/
flake8 src/ tests/      # exits non-zero = CI fails
```

Analysis scripts under `examples/` are format-checked (black/isort) but
exempted from pedantic flake8 codes via `.flake8` per-file-ignores — they
are research scripts, not package code. Configuration: `[tool.black]`,
`[tool.isort]`, `[tool.pytest.ini_options]` in `pyproject.toml`; flake8 in
`.flake8` (max line length 100 to tolerate legacy docstring prose).

### Helper scripts (`scripts/`)

- `scripts/new_run.py <name>` — scaffold a run folder (`runs/<name>/`) with
  an annotated `run_config.yaml`, empty `data/` + `results/` and a
  checklist; this is the onboarding path in
  [`docs/YOUR_OWN_DATA.md`](docs/YOUR_OWN_DATA.md).
- `scripts/run_all_tests.py` / `scripts/check_code_quality.py` — thin
  wrappers around `pytest` and black/isort/flake8; prefer the direct
  commands above.

## Architecture Overview

Modular pipeline; each stage takes the merged config and returns a DataFrame.

```
raw files → Parser → Cleaner → Aligner → Processor → Visualizer → results/
```

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
   (`visualization/plotting.py`) and self-contained interactive HTML
   dashboards (`visualization/dashboard.py`, optional `bokeh`).

Supporting layers under `core/`: `constants.py` (centralized physical
constants — no magic numbers in calculations), `validation.py` (DataFrame/
column/numeric validators, incl. `coerce_numeric_columns`), `exceptions.py`
(`USPConfigError`, `USPDataError`, `USPCalculationError`).

Two convenience layers wrap the chain so scripts never re-implement the
orchestration: **`pipeline.py`** (`run_pipeline` / `run_directory` — one
call per run folder, auto-detects config + data files) and
**`consolidation.py`** (`consolidate_runs` — combined workbooks + bokeh
comparison dashboard for a campaign). Prefer them over orchestrating the
stages manually; both are re-exported from the package root and usable as
`python -m upstream_analysis.pipeline <run_dir>` /
`python -m upstream_analysis.consolidation <campaign> --prefix <P>`.

> `data/processor.py` is a thin facade over the focused calculation modules
> in `data/calculations/` (`feed`, `volume`, `induction`, `protein`, `gas`,
> `carbon_balance`, `rates_yields`; shared config access in `_config`).
> `FermentationDataProcessor` keeps the historical API (`process_data(df)`
> plus per-step `_<name>(df)` methods the unit tests exercise). The **order
> of steps in `process_data` is load-bearing** (later calculations consume
> columns built earlier) — preserve it when adding a calculation.

### Configuration System

Two-level hierarchy merged at runtime by `load_config`
(`config/loader.py`): package defaults in
[`src/upstream_analysis/config/default_config.yaml`](src/upstream_analysis/config/default_config.yaml)
overridden per run by `run_config.yaml`. Only `processing.
processing_parameters.reactor_setup.initial_volume_ml` is mandatory;
everything else has defaults. The defaults' `column_mapping` documents the
standard internal column names (it ships preconfigured for DASGIP-style
exports).

### Experiment Structure

```
<campaign>/<run_id>/
├── run_config.yaml     # tracked — sits NEXT TO data/, never inside it
├── data/               # raw inputs (gitignored)
└── results/            # generated (gitignored, auto-created)
```

The full convention (campaign level, file naming, what git tracks, how to
grow from one run to a campaign) is specified in
[`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) — keep that document
authoritative rather than duplicating layout rules here. Raw data stays out
of the repo; no example ships data.

## Notes for Contributors

- **Add derived-calculation math to `data/calculations/`**, not the facade.
  Keep the step order in `process_data` intact.
- Add physical constants to `core/constants.py`; raise typed exceptions
  from `core/exceptions.py`; validate with `core/validation.py` — avoid
  silent NaN returns.
- **Keep the suite green**: `pytest`, then `black/isort/flake8` on
  `src/ tests/`, before committing. CI runs all of it plus the synthetic
  smoke test on every push/PR.
- Docs are part of the contract: when behavior or layout changes, update
  `README.md`, the relevant `docs/*.md`, and this file in the same commit.
