# archive/

In-repo archive: work that is kept but sits outside the package's
data-analysis scope.

- [`USP_modeling/`](USP_modeling/README.md) — mechanistic IB / LDH modeling
  line: the M0 model-variant campaign (active line, see its `HANDOFF.md`)
  and the dormant modeling playground (fit lineage, symbolic regression,
  IB solubility analysis).
- [`USP_artifacts/`](USP_artifacts/README.md) — per-example outputs, logs
  and one-off research removed from `examples/` when it was slimmed down:
  consolidation outputs, the LDH partition-vs-temperature study, the optfed
  scratch study, and the DoE provenance (induction audit script +
  `data_merger.ipynb`).

Git policy: code, docs and configs under `archive/` are **tracked**;
`data/`, `results/` and `outputs/` dirs and binary formats (figures,
workbooks, pickles) stay local-only — enforced by `.gitignore`, consistent
with how `examples/` treats raw data. The folder is inside the project's
OneDrive sync, which carries the bulk material.
