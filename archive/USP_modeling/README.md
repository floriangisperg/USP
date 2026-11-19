# USP modeling archive

Moved out of `05_Projects/USP` on 2026-11-18/19, then **moved back in-repo
on 2026-11-19** (author's call): it lives at `archive/USP_modeling/` —
separate from the package's data-analysis scope, but inside the project and
its OneDrive sync. Git tracks code/docs/configs under `archive/`; `data/`,
`results/` and `outputs/` dirs and binary formats stay local-only (see
`.gitignore`). The modeling below is standalone (numpy/pandas/scipy — no
USP package imports); it belongs to the inclusion-body / LDH modeling line,
whose framework home is `05_Projects/HybridModel` (see its `examples/`).

| Folder | What | Status |
|---|---|---|
| `LDH_M0_campaign/` | The **active** M0 model-variant campaign (models.py, run_playground.py, holdout results through Jul 2026, HANDOFF.md). | Active — moved 2026-11-18; back in-repo under `archive/` 2026-11-19 |
| `modeling_playground/` | Older scratch: LDH fit lineage v1–v6 (scipy fits), symbolic regression (Jan 2026), IB solubility analysis + REPORT, mechanistic_ib_model v1/v2, base/qP analyses. | Dormant scratch |
| *(the tracked `USP/modeling/` copy)* | Was a curated copy of the mature pieces above; removed from USP git (still in USP git history) — this folder holds the originals. | — |

## Running the M0 campaign

```powershell
cd LDH_M0_campaign
& "C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\.venv\Scripts\python.exe" run_playground.py --help
```

- Consolidated data is bundled at `LDH_M0_campaign/data/LDH_consolidated_results_processed.xlsx`.
- `base_reconstruct.py` still reads **raw** run exports and the pump2 cache
  from `...\05_Projects\USP\examples\LDH\` (mtime-keyed cache — the raw
  files must stay where they are).
- Dependencies beyond numpy/pandas/scipy/matplotlib: none (the old USP
  `modeling` extra with cma/catalax belonged to the dormant playground
  fits, not this campaign).

## Future home

If this line becomes active again, consider promoting `LDH_M0_campaign/`
into `HybridModel/examples/` (its LDH/hGH/refolding modeling examples live
there).
