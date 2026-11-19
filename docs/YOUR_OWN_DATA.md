# Analyzing your own fermentation data

This guide takes you from a raw bioreactor export to analyzed results with
USP. It assumes a standard fed-batch E. coli cultivation with online sensor
data (SCADA/bioreactor software export) and offline lab measurements from
sampled time points.

**New to the repo?** First verify your setup by running the self-contained
example:

```bash
uv sync --extra dev --extra dashboard
pytest   # 150 tests incl. end-to-end pipeline checks on simulated data
```

If that produces `results/processed_data.csv` and plots, your environment
works — now bring your own data.

---

## What USP expects

One run = one folder with two input files and one config. For multi-run
campaigns, run folders sit inside a campaign folder next to shared analysis
scripts — the full picture is in [DATA_LAYOUT.md](DATA_LAYOUT.md):

```
my_experiment/
├── run_config.yaml     # run-specific settings (see Step 3) — sits NEXT TO
│                       #   data/, not inside it (data/ is gitignored; the
│                       #   config is what git tracks)
├── data/
│   ├── <online>.csv    # continuous bioreactor sensor export
│   └── <offline>.xlsx  # lab measurements from sampled time points
└── results/            # generated: processed_data.csv, plots (auto-created)
```

The offline file is optional — the pipeline degrades gracefully (no biomass,
no yields) but the core volume/feed/gas analysis only needs the online CSV.

## Step 0 — Scaffold a run folder

```bash
python scripts/new_run.py my_experiment
```

creates `runs/my_experiment/` with a commented `run_config.yaml`, empty
`data/` and `results/`, and a checklist. Everything below refers to the
files it generates.

## Step 1 — Check your online CSV against the standard column names

USP works with **standard internal column names** (e.g. `Balance [g]`,
`Process phase [-]`). Your raw export uses your bioreactor software's
header names. The `column_mapping` section of
[`config/default_config.yaml`](../src/upstream_analysis/config/default_config.yaml)
maps one to the other — it ships preconfigured for a DASGIP-style export
(`'AirFlow, l/min'` → `Air flow [L/min]`, ...).

**Required** standard names (without these the pipeline fails early with a
`USPConfigError`/`USPDataError`):

| Standard name | Needed for |
|---|---|
| `Process time [h]` (or `Batch time [s]` / UTC + local datetime columns) | every time calculation |
| `Process phase [-]` | cleaning, feed phases, induction |
| `Balance [g]` | feed mass, volume balance |
| `Air flow [L/min]` and/or `Total flow [L/min]` | gas rates (CER/OUR/RQ) |

**Recommended** (each unlocks more analysis, all optional):

| Standard name | Unlocks |
|---|---|
| `Offgas O2 [%]`, `Offgas CO2 [%]` | CER, OUR, RQ |
| `O2 flow [L/min]`, `Gas mix [-]` | enriched-air corrections |
| `Base pump duration [-]`, `Feed pump duration [-]` | pump-based volume terms |
| `pH [-]`, `pO2 [%]`, `Temperature [°C]`, `Stirrer speed [1/min]` | context plots |
| `OD600` (online) | online biomass proxy |

To adapt your export, override only the headers that differ, in your run
config:

```yaml
column_mapping:
  'Your Actual Header': 'Standard Name [unit]'
```

Names not listed are passed through unchanged — so if your export already
uses standard names, you need no mapping at all.

**Unit conventions:** flows in `L/min`, masses in `g`, volumes in `mL`,
temperatures in `°C`, percentages as 0–100. The unit tag in brackets is part
of the name. If your export uses different units, convert beforehand or
adjust the corresponding constants in `run_config.yaml` (e.g. feed density).

## Step 2 — Parsing settings

`parsing` in the config describes the file format, not the content:

```yaml
parsing:
  bioreactor:
    decimal_separator: ','    # German Excel exports use ',' — English use '.'
    column_separator: ';'     # ';' (Excel-CSV German) vs ','
    date_format: '%d.%m.%Y %H:%M'
    skiprows: 1               # some SCADA exports have preamble lines
  offline:
    decimal_separator: '.'
    date_format: 'guess'
    sheet_name: 0
```

If parsing fails or all-NaN columns appear, open the raw file and match
these to reality. Known encoding trap: a `°` in headers (`Temperature, °C`)
arrives mangled depending on the export's encoding — the default mapping
already contains the two common corruptions (`Â°C`, `¬∞C`); add yours the
same way.

## Step 3 — The minimal run config

Only `initial_volume_ml` is **required** (there is no sane default); every
other key has defaults. A minimal but complete `run_config.yaml`:

```yaml
metadata:
  fermentation_id: "my_experiment"
  strain: "E. coli BL21(DE3)"

processing:
  processing_parameters:
    reactor_setup:
      initial_volume_ml: 1500.0   # batch medium volume at inoculation
    balance_mode: 'feed_source'   # balance tracks the feed bottle (decreasing)
  induced_phases:                 # phase names AFTER inducer addition
    - 'induced Fedbatch'
    - 'Production'
  induction_time_h: null          # set if the SCADA phase flip leads the
                                  # actual inducer bolus (see below)
```

Phase names must match your export's `Process phase [-]` values: cleaning
removes pre-inoculation phases, `induced_phases` marks the production phase
for induction-dependent calculations. If the SCADA phase flips into
"induced" *before* the manual inducer bolus, set `induction_time_h` to the
operator-verified time (process-time hours) — USP shifts the induction
flag accordingly and warns if the offset is large.

## Step 4 — Offline samples Excel

One sheet (`parsing.offline.sheet_name`), one row per sample. Required
column: `Process time [h]` (same time zero as the online data). Optional
measurement columns, each enabling downstream calculations:

| Column | Unlocks |
|---|---|
| `OD600 [-]` or `Biomass [g/L]` | biomass, yields, specific rates (qO2, qCO2) |
| `Glucose [g/L]`, `Acetate [g/L]` | residual substrate plots, carbon balance |
| `Protein [g/L]`, `Protein soluble [g/L]`, `Protein IB [g/L]` | product, soluble/IB fractions, yields |
| `Sample volume [mL]` | volume-corrected analysis |

## Step 5 — Run it

The pipeline is a package entry point — no example script needed:

```bash
python -m upstream_analysis.pipeline runs/my_experiment
```

Auto-detects `run_config.yaml` and the single online CSV (plus optional
`data/samples.xlsx`) and writes `results/processed_data.csv`,
`samples_points_data.csv` and overview plots. To consolidate a campaign of
runs into combined workbooks and a comparison dashboard:

```bash
python -m upstream_analysis.consolidation runs/ --prefix my_campaign
```

For programmatic use — explicit paths, custom plot selection, unusual
layouts — use the API directly:

```python
from upstream_analysis import load_config, run_pipeline

config = load_config("runs/my_experiment/run_config.yaml")
result = run_pipeline(config, "data/rawdata.csv", offline_path="data/samples.xlsx",
                      output_dir="results")
result.processed  # aligned + derived DataFrame
```

## Step 6 — Sanity-check the results

Before trusting numbers, verify in `results/processed_data.csv`:

1. `Process time [h]` starts near 0 at inoculation and is monotonic — if it
   rezeros at feed start, a phase was filtered that shouldn't have been
   (see the `rename_preparation_to_batch` note in the default config).
2. `Volume [mL]` evolves continuously (feed in, base in, samples out); a
   sawtooth means the balance tare or `balance_mode` is wrong.
3. `Feed [g]` is 0 during batch and grows during fed-batch phases.
4. `CER`/`OUR` are positive and plausible during growth; `RQ` near 1 on
   glucose.
5. Rows with `is_sample = True` match your sampling times.

## Troubleshooting

The typed exceptions (`USPConfigError`, `USPDataError`,
`USPCalculationError`) say what is missing — read the message first. The
extended troubleshooting table (feed = 0, missing plots, column-not-found,
...) lives in [examples/README.md →
Troubleshooting](../examples/README.md#troubleshooting). For every config
knob, see [docs/CONFIGURATION.md](CONFIGURATION.md).
