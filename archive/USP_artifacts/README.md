# USP example artifacts (archived 2026-11-19; back in-repo 2026-11-19)

Outputs and one-off research scripts that had accumulated inside the USP
examples folders. The examples themselves (`05_Projects/USP/examples/`)
now contain only what an example needs: run configs, run data notes, and
the pump2 cache the M0 campaign depends on — analysis and consolidation
are package entry points. This folder preserves everything that was
removed; like `../USP_modeling/`, it lives **inside the USP repo** under
`archive/` (code/docs tracked; data/outputs local-only).

| Folder | From | Contents |
|---|---|---|
| `LDH/partition_temperature_research/` | `examples/LDH/` | The LDH partitioning-vs-temperature study: `analyze_partition_temp*.py`, `plot_*.py` scripts, result PNGs, and the consolidated processed data (`LDH_consolidated_results_processed.xlsx` / `.pkl`) they analyze. Paper-work, not example material. |
| `LDH/consolidated_outputs/` | `examples/LDH/` | Regenerable consolidation outputs: workbooks + bokeh dashboard. Re-run `consolidate_results.py` where the raw data lives to rebuild. |
| `LDH/logs/` | `examples/LDH/` | Consolidation/regeneration logs. |
| `Xiso/consolidated_outputs/` | `examples/Xiso/` | Same as LDH, for the Xiso campaign. |
| `Xiso/logs/` | `examples/Xiso/` | Regeneration log. |
| `doe/data_merger.ipynb` | `examples/doe_example/` | One-off notebook used to assemble `all_data/`. |
| `doe/audit_induction_operator_vs_scada.py` | `examples/doe_example/` | Read-only provenance tool for the `induction_time_h` values in the 12 doe run configs: compares the operator's noted IPTG time (from `*_Auswertung.xlsx` workbooks) against the SCADA phase flip. Audit of 2026-10-23. |
| `doe/data_legacy/` | `examples/doe_example/data/` | Early compiled workbooks (`scFv_compiled_TL.xlsx`, `traintest_compiled.xlsx`) from the ML-era experiments — a campaign-level `data/` folder is not part of the layout convention. |

Note: consolidation is now a package entry point
(`python -m upstream_analysis.consolidation <campaign> --prefix <P>`), so
"regenerate via consolidate_results.py" above means: run that command where
the run data lives.

Note: `LDH_consolidated_results_processed.pkl` was removed from USP git in
the same commit that moved these files — the blob remains in USP git
history. `pump2_raw_cache.pkl` deliberately stays in `examples/LDH/` (the
M0 campaign in `../USP_modeling/LDH_M0_campaign/` reads it from there,
mtime-keyed against the raw run exports).
