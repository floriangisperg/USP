"""Create a new fermentation run folder for USP analysis.

Scaffolds ``<parent>/<name>/`` with an annotated ``run_config.yaml``, empty
``data/`` and ``results/`` directories, and a short checklist. See
``docs/YOUR_OWN_DATA.md`` for the full walkthrough.

Usage::

    python scripts/new_run.py my_experiment
    python scripts/new_run.py my_experiment --parent experiments
"""

import argparse
import sys
from pathlib import Path
from textwrap import dedent

RUN_CONFIG_TEMPLATE = dedent("""\
    # Run configuration for {name}
    # Overrides the package defaults in
    # src/upstream_analysis/config/default_config.yaml — anything not
    # listed here falls back to those defaults. Full reference:
    # docs/CONFIGURATION.md

    metadata:
      fermentation_id: "{name}"
      # date_start: "2026-01-15"
      # strain: "E. coli BL21(DE3)"
      # reactor: "2L Bioreactor"

    # parsing:                      # only if your export differs from the
    #   bioreactor:                 # DASGIP-style defaults (separators,
    #     decimal_separator: ','    # date format, encoding — see
    #     column_separator: ';'     # docs/YOUR_OWN_DATA.md step 2)
    #     date_format: '%d.%m.%Y %H:%M'
    #     skiprows: 1

    # column_mapping:               # map YOUR raw headers to the standard
    #   'Your Header': 'Standard Name [unit]'   # names (docs/YOUR_OWN_DATA.md step 1)

    cleaning:
      # Set true only if your export calls the post-inoculation batch phase
      # 'Preparation' instead of 'Batch' (Run1 quirk — see default config):
      rename_preparation_to_batch: false

    processing:
      processing_parameters:
        reactor_setup:
          initial_volume_ml: null  # REQUIRED: batch volume at inoculation [mL]
        balance_mode: 'feed_source'  # balance tracks the feed bottle (decreasing)
      induced_phases:              # phase names AFTER inducer addition
        - 'induced Fedbatch'
        - 'Production'
      induction_time_h: null       # operator-verified induction time [h],
                                   # if the SCADA phase flip leads the bolus
    """)

RUN_README_TEMPLATE = dedent("""\
    # {name}

    Fermentation run scaffold created by `scripts/new_run.py`.

    ## Checklist

    1. Put the bioreactor export into `data/` (one CSV) and the offline
       sample measurements into `data/` (one xlsx, optional).
    2. Open the raw CSV and compare its headers with
       `src/upstream_analysis/config/default_config.yaml` → `column_mapping`.
       Add overrides for differing headers in `run_config.yaml`.
    3. Match `parsing` settings (separators, date format) to the export.
    4. Fill in `initial_volume_ml` in `run_config.yaml` — it is required.
    5. Check the phase names in your export against `induced_phases` and
       `cleaning` settings.
    6. Run the analysis (see docs/YOUR_OWN_DATA.md step 5), then
       sanity-check `results/processed_data.csv`
       (docs/YOUR_OWN_DATA.md step 6).

    Full walkthrough: `docs/YOUR_OWN_DATA.md`
    """)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a new USP fermentation run folder."
    )
    parser.add_argument("name", help="run name (becomes the folder name)")
    parser.add_argument(
        "--parent",
        default="runs",
        help="parent directory for the run folder (default: runs/)",
    )
    args = parser.parse_args()

    run_dir = Path(args.parent) / args.name
    if run_dir.exists():
        print(f"Refusing to overwrite existing folder: {run_dir}")
        return 1

    (run_dir / "data").mkdir(parents=True)
    (run_dir / "results").mkdir()

    (run_dir / "run_config.yaml").write_text(
        RUN_CONFIG_TEMPLATE.format(name=args.name), encoding="utf-8"
    )
    (run_dir / "README.md").write_text(
        RUN_README_TEMPLATE.format(name=args.name), encoding="utf-8"
    )

    print(f"Created run scaffold: {run_dir}/")
    print("Next steps:")
    print(f"  1. add data files to {run_dir / 'data'}/")
    print(f"  2. fill in {run_dir / 'run_config.yaml'} (initial_volume_ml!)")
    print("  3. follow docs/YOUR_OWN_DATA.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
