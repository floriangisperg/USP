# Synthetic example

A tiny **simulated** fermentation (28 h fed-batch, seeded and deterministic)
that exercises the full pipeline end-to-end — the only other example besides
`real_data_example` whose data is committed, so it runs immediately after
cloning. Use it to verify an install or to see what a well-formed dataset
looks like; use `real_data_example` for real-world messiness.

## Run it

```bash
python verify_results.py        # runs the pipeline + asserts sanity (CI does this)
python analyze_synthetic_run.py # pipeline only, writes results/
python generate_data.py         # regenerate the data (reproduces byte-for-byte, seed 42)
```

All 11 sanity checks double as documentation of what "working" means:
phase filtering, feed from the balance, volume continuity, positive
CER/OUR, physiological RQ, sample alignment, induction timing at 16 h,
IB fraction 0.4 after induction.

## What the data shows

`data/rawdata.csv` uses DASGIP-style headers and German CSV formatting
(`;` separator, decimal comma, `%d.%m.%Y %H:%M` timestamps, one preamble
line) — so the package defaults parse it unchanged and `run_config.yaml`
stays minimal: no `column_mapping`, no `parsing` overrides. Compare with
`examples/hgh/R5_lowlow/run_config.yaml` to see what a *non-default*
export needs.

The process model behind the generator (constant yields, exponential
growth, RQ = 1) is deliberately simple; the numbers are self-consistent
(feed rate ↔ growth ↔ offgas ↔ volume) but not calibrated to reality.
