# Modeling work

Tracked home for the mature modeling analyses that grew out of the (local,
gitignored) `modeling_playground/` scratch area. Code and written reports are
version-controlled here; generated figures and fitted outputs remain
local-only and are **not** in the repository.

| Subdirectory | Contents | Status |
|---|---|---|
| `mechanistic_ib_model/` | Mechanistic model of inclusion-body vs soluble protein formation (temperature/qS response surface), v1 + v2 with model comparison. See its `README.md`. | Finished model comparison; figures local |
| `ldh_fits/` | Latest LDH dataset hybrid fits: `fit_model_v5_improved_ib.py` (improved IB submodel), `fit_model_v6_simplified.py` (simplified successor). | Exploratory lineage v1→v6; v6 is the last iteration |
| `ib_solubility_analysis/` | Statistical analysis of IB solubility vs specific substrate uptake (qS): design, dose-response, time series, identifiability, hypothesis tests. See `README.md` and `REPORT.md`. | Reported; figures local |

## Dependencies

All scripts use the base dependencies (numpy, pandas, scipy, matplotlib,
seaborn, PyYAML) — no `modeling` extra needed to *read* them. The
`modeling` extra (`cma`, `catalax`) is only required by the remaining
sandbox in `modeling_playground/`.

## Data

Scripts read consolidated LDH results from the local playground data area
(`modeling_playground/LDH/data/LDH_consolidated_results.xlsx`), which is
gitignored. Paths are hardcoded relative to each script's original location
inside the playground; expect to adjust paths when running from here.

## Related

- `examples/LDH/modeling_playground/` — the *active* M0 model-variant
  campaign (holdout experiments, `HANDOFF.md`); local-only scratch, not
  tracked.
- `../HybridModel` — the JAX hybrid-modelling framework used for the
  mechanistic+learned fits beyond this dataset.
