"""Run the full USP pipeline on the synthetic demo data.

This is what a *minimal custom analysis script* looks like now: the
pipeline orchestration lives in the package, so a script only picks the
run folder and (optionally) the plot selection. For most cases you don't
need a script at all::

    python -m upstream_analysis.pipeline examples/synthetic_example

``verify_results.py`` uses :func:`run_analysis` below and asserts the
outputs are sane.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe plotting

EXAMPLE_DIR = Path(__file__).resolve().parent

PLOT_PARAMETERS = [
    "Biomass [g/L]",
    "Feed [g]",
    "Volume corrected [mL]",
    "CER [mmol/L/h]",
    "OUR [mmol/L/h]",
    "RQ [-]",
    "pH [-]",
    "pO2 [%]",
]


def run_analysis(example_dir: Path = EXAMPLE_DIR):
    """Run the pipeline on the synthetic data and return the PipelineResult."""
    from upstream_analysis import run_directory

    return run_directory(example_dir, plot_parameters=PLOT_PARAMETERS)


def main() -> int:
    result = run_analysis()
    print(result.processed[["Process time [h]", "Biomass [g/L]", "Volume [mL]"]].tail())
    print(f"results written to {result.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
