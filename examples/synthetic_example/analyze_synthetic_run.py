"""Run the full USP pipeline on the synthetic demo data.

Usage (from anywhere — paths resolve relative to this file)::

    python analyze_synthetic_run.py

Reads ``data/rawdata.csv`` + ``data/samples.xlsx``, runs the pipeline with
``run_config.yaml``, and writes ``processed_data.csv``,
``samples_points_data.csv`` and a few overview plots to ``results/``.
``verify_results.py`` wraps this and asserts the outputs are sane.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe plotting

import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # noqa: E402

from upstream_analysis import (  # noqa: E402
    BioreactorDataParser,
    DataAligner,
    DataVisualizer,
    FermentationDataCleaner,
    FermentationDataProcessor,
    OfflineMeasurementsParser,
    load_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger(__name__)

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


def run_analysis(example_dir: Path) -> pd.DataFrame:
    """Run the pipeline on the synthetic data and return the processed frame."""
    config = load_config(example_dir / "run_config.yaml")

    csv_files = sorted((example_dir / "data").glob("*.csv"))
    xlsx_files = sorted((example_dir / "data").glob("*.xlsx"))
    if len(csv_files) != 1 or len(xlsx_files) != 1:
        raise FileNotFoundError(
            f"expected exactly one csv + one xlsx in {example_dir / 'data'}, "
            f"found {csv_files} + {xlsx_files} — run generate_data.py first"
        )

    online = BioreactorDataParser(config).parse_file(csv_files[0])
    online = FermentationDataCleaner(config).clean_data(online)
    offline = OfflineMeasurementsParser(config).parse_file(xlsx_files[0])

    df_merged, df_samples = DataAligner(config["alignment"]).merge_datasets(
        online, offline
    )
    processed = FermentationDataProcessor(config).process_data(df_merged)

    output_dir = example_dir / "results"
    output_dir.mkdir(exist_ok=True)
    processed.to_csv(
        output_dir / "processed_data.csv", index=False, sep=";", decimal=","
    )
    if not df_samples.empty:
        df_samples.to_csv(
            output_dir / "samples_points_data.csv", index=False, sep=";", decimal=","
        )

    plotable = [p for p in PLOT_PARAMETERS if p in processed.columns]
    DataVisualizer(output_dir=output_dir).plot_selected_parameters(
        processed, parameters=plotable, time_col="Process time [h]"
    )
    logger.info("results written to %s", output_dir)
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    processed = run_analysis(EXAMPLE_DIR)
    print(processed[["Process time [h]", "Biomass [g/L]", "Volume [mL]"]].tail())
    return 0


if __name__ == "__main__":
    sys.exit(main())
