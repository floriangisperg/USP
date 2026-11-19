# USP Examples

Worked examples for the USP (Upstream Processing) package. The pipeline and
consolidation live in the package itself — **examples are data + run
configs, not scripts**:

```bash
# analyze one run (writes <run>/results/)
python -m upstream_analysis.pipeline <run_dir>

# consolidate a campaign into workbooks + comparison dashboard
python -m upstream_analysis.consolidation <campaign_dir> --prefix <PREFIX>
```

## New here?

1. Verify your install: `pytest` (includes end-to-end pipeline checks on
   simulated data)
2. Analyze a campaign run:
   `python -m upstream_analysis.pipeline examples/scFvM/ferm_01`
3. Bringing **your own fermentation data**? Follow
   [`docs/YOUR_OWN_DATA.md`](../docs/YOUR_OWN_DATA.md) and scaffold with
   `python scripts/new_run.py <name>`.
4. Multiple runs? The folder convention is specified in
   [`docs/DATA_LAYOUT.md`](../docs/DATA_LAYOUT.md) — campaign folder,
   one folder per run, then consolidate.

**Runnable status:** these examples document *actual analyses of real
campaigns*; their raw data is gitignored and lives outside the repo, so
they run only where that data exists. End-to-end coverage on simulated
data lives in the test suite, not in the examples.

Each example contains only what an example needs (run configs + notes).
Generated outputs, logs and one-off research analyses that had accumulated
here were moved to `../../Archive/USP_artifacts/` — except in `hgh/`, whose
campaign is still current and keeps its outputs.

## The examples

| Directory | What it shows |
|---|---|
| `scFvM/` | 13 scFv DoE fermentation runs (config per run incl. operator-verified induction times, `all_data/` raw campaign exports) |
| `hgh/` | Two hGH fed-batch runs (R5 low/low, R6 high/high) + consolidated comparison dashboard |
| `LDH/` | Real LDH production data: Excel online exports, advanced carbon balance, the F19 IB correction note |
| `Xiso/` | Xylose-isomerase campaign (3 runs) |

Campaign usage, e.g. the DoE set:

```bash
# bash: analyze every run, then consolidate
for d in examples/scFvM/ferm_*/; do
  python -m upstream_analysis.pipeline "$d"
done
python -m upstream_analysis.consolidation examples/scFvM --prefix doe
```

Notes per example:

- **LDH**: online data is Excel (`Rawdata*.xlsx`) — auto-detected by the
  pipeline runner. `pump2_raw_cache.pkl` is a raw-signal cache (mtime-keyed
  against the run files) also used by the archived M0 campaign.
- **hGH**: `prepare_hgh_data.py` shows how the raw SNO exports were merged
  into the per-run `data/` files. The modeling sandbox that used to live
  under `examples/LDH/modeling_playground/` (the M0 campaign) moved to
  `../../Archive/USP_modeling/LDH_M0_campaign/` — see that README.
- **DOE**: the `induction_time_h` values in the run configs come from an
  operator-vs-SCADA audit (2026-10-23); the audit script and the one-off
  `data_merger.ipynb` live in `../../Archive/USP_artifacts/doe/`.

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
3. Ensure feed density is positive (> 0) — `feed_definitions` must be set
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

1. **Check the logs** — typed errors (`USPConfigError`, `USPDataError`,
   `USPCalculationError`) say what is missing
2. **Verify data format** — see
   [`docs/YOUR_OWN_DATA.md`](../docs/YOUR_OWN_DATA.md) for the input contract
3. **Review configuration** — every option in
   [`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md)
4. **Compare with examples** — the run configs show real campaign settings
