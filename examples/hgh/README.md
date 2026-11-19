# hGH example — two fed-batch runs + campaign comparison

Two hGH fed-batch fermentations (R5 "lowlow", R6 "highhigh") from the
October/November 2026 campaign at TU Wien, analyzed with USP.

## Layout

```
hgh/
├── prepare_hgh_data.py               # builds per-run data/ from the raw inputs
├── all_data/                         # raw campaign inputs (gitignored)
│   ├── SNO_hgh_*.csv                 #   online bioreactor exports (SNO dialect)
│   ├── Samplingsheet_*.pdf           #   operator sampling notes
│   ├── Laborjournal_KW43.pdf
│   ├── 20261109_Analytik_*.xlsx      #   RP-HPLC / offline analytics
│   ├── 20261019_Media.xlsx, 20261020_DCW_KW43.xlsx
│   └── FeedpumpCalibration.xlsx
├── R5_lowlow/  R6_highhigh/          # run folders (DATA_LAYOUT convention)
└── hGH_consolidated_results*.xlsx    # campaign consolidation outputs
    hGH_consolidated_dashboard.html
```

## Reproduce

```bash
# 1. build per-run data/ from the raw exports (merges online CSV + offline
#    samples from the analytics workbook, caps R5 at the successful
#    re-inoculation, builds the synthetic balance from the pump calibration);
#    paths are script-relative, so run from anywhere
python examples/hgh/prepare_hgh_data.py

# 2. analyze each run
python -m upstream_analysis.pipeline examples/hgh/R5_lowlow
python -m upstream_analysis.pipeline examples/hgh/R6_highhigh

# 3. consolidate
python -m upstream_analysis.consolidation examples/hgh --prefix hGH
```

Notes:
- This export uses US-style 12 h timestamps and dot decimals — the run
  configs override `parsing` accordingly. `inducedFedbatch` (no space) is
  camel-split by the parser to `induced Fedbatch`.
- The R5 export carries NO balance signal; `prepare_hgh_data.py` builds a
  synthetic balance from the pump percentage × a fitted calibration slope
  (see the comments in the run config for the numbers).
- The operator-noted IPTG times lead/lag the SCADA phase flips — hence the
  explicit `induction_time_h` values in both run configs.
