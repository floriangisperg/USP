# Verification

How to check that the package works, and what the current state is.

## Run the test suite

```bash
pytest                    # full suite (unit + integration)
pytest tests/unit         # unit tests only
pytest --cov=upstream_analysis   # with coverage (requires pytest-cov)
```

Current status: **136 tests, all passing** (Python 3.12). Tests use in-memory
fixtures defined in `tests/conftest.py`, so no data files are required.

## Lint / format

```bash
black   src/ tests/ examples/
isort   src/ tests/ examples/
flake8  src/ tests/ examples/
```

Configuration lives in `pyproject.toml` (`[tool.black]`, `[tool.isort]`,
`[tool.pytest.ini_options]`) and `.flake8`.

## Run an example end-to-end

```bash
python -m upstream_analysis.pipeline examples/real_data_example
python examples/synthetic_example/verify_results.py
```

`real_data_example` ships its own `data/` and writes `processed_data.csv`,
`samples_points_data.csv`, and per-parameter plots to `results/`.

## Known limitations

- CI (`.github/workflows/ci.yml`) runs lint + the full test suite on every
  push to `main` and on pull requests.
- Test coverage is not measured/recorded continuously (`pytest-cov` is
  optional).
- `scripts/verify_improvements.py` / `run_all_tests.py` / `check_code_quality.py`
  exist but are thin wrappers; prefer the direct commands above.
