# Development notes

This is a short, factual record of structural work done on the USP package. It
intentionally avoids claims about coverage or "enterprise" quality — see
[`VERIFICATION_STATUS.md`](VERIFICATION_STATUS.md) for how to check the actual
state, and the git log for the full history.

## Core layer (`src/upstream_analysis/core/`)

- **`constants.py`** — physical constants and process defaults (molar volumes,
  molar masses on a C-mol basis, default volumes/feed concentrations, signal
 -processing windows) live in one place instead of as magic numbers.
- **`validation.py`** — helpers used across the pipeline:
  `require_columns`, `validate_dataframe_not_empty`, `validate_numeric_series`,
  `validate_positive_numeric`, `get_nested_config`.
- **`exceptions.py`** — a small hierarchy, all deriving from `ValueError`:
  - `USPError` (base)
    - `USPConfigError` — invalid/missing configuration
    - `USPDataError` — data validation failures
    - `USPCalculationError` — calculation failures

  Because they subclass `ValueError`, callers can catch them generically with
  `except ValueError` while still discriminating by USP type. Invalid config
  values (e.g. a non-positive feed density) now raise instead of failing
  silently; *missing* optional config is still handled gracefully.

## Data pipeline

- **Cleaning** (`data/cleaning.py`)
  - `_calculate_process_time` prefers `Batch time [s]` and falls back to the
    UTC/local datetime columns when it is absent.
  - `_rolling_median_filter` uses a global robust scale (median absolute
    deviation of the residuals) instead of a rolling MAD, which failed to flag
    isolated spikes because the spike inflated its own local window. An
    edge-contamination guard avoids replacing a clean value with a neighbouring
    spike's median.
- **Processing** (`data/processor.py`)
  - Carbon-balance and rate/yield calculations skip gracefully when the offline
    `Biomass [g/L]` / `Sample volume [mL]` columns are absent instead of
    raising `KeyError`.
  - The processor validates `initial_volume_ml` (must be positive when present)
    at construction time.

## Tests

- `tests/unit/` (parser, cleaning, alignment, processor, visualization) and
  `tests/integration/` (full pipeline).
- Shared in-memory fixtures in `tests/conftest.py` — no external data files
  required, so the suite runs from a clean checkout.

## Configuration

- Hierarchical merge: `default_config.yaml` overridden by per-run
  `run_config.yaml` via `config/loader.py`. See
  [`CONFIGURATION.md`](CONFIGURATION.md) for the parameter reference.

## Packaging

- Runtime dependencies are isolated from dev/test/modeling tools (see
  `pyproject.toml`); `uv.lock` is committed for reproducible installs.
- README and `AGENTS.md` document install, quickstart, architecture, and the
  development workflow.
