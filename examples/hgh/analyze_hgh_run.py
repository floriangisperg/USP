"""Analyze one hGH example run folder (R5_lowlow or R6_highhigh).

Usage (from examples/hgh or anywhere - paths resolve relative to this file):

    python analyze_hgh_run.py                # default: R6_highhigh
    python analyze_hgh_run.py R5_lowlow

Run ``prepare_hgh_data.py`` once beforehand so data/ holds the online CSV and
samples.xlsx. Writes processed_data.csv, samples_points_data.csv and overview
plots to <run>/results/.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# --- Path setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from upstream_analysis import (  # noqa: E402
    BioreactorDataParser,
    DataAligner,
    DataVisualizer,
    FermentationDataCleaner,
    FermentationDataProcessor,
    OfflineMeasurementsParser,
    load_config,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).resolve().parent

_cli = argparse.ArgumentParser(description="Analyze one hGH example run folder.")
_cli.add_argument(
    "run_folder",
    nargs="?",
    default="R6_highhigh",
    help="run folder inside 'hgh' (R5_lowlow or R6_highhigh; default: R6_highhigh)",
)
target_run_folder_name: str = _cli.parse_args().run_folder

run_dir = (EXAMPLE_DIR / target_run_folder_name).resolve()
if not run_dir.is_dir():
    logger.error(f"Target run directory does not exist: {run_dir}")
    sys.exit(1)

config_path = run_dir / "run_config.yaml"
data_dir = run_dir / "data"
output_dir = run_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)

if not config_path.exists():
    logger.error(f"Configuration file not found: {config_path}")
    sys.exit(1)

logger.info(f"Analyzing run directory: {run_dir}")
config = load_config(config_path)

# --- Auto-detect data files (exactly one CSV + one xlsx expected) ---
csv_files = list(data_dir.glob("*.csv"))
if len(csv_files) != 1:
    names = [f.name for f in csv_files]
    logger.error(
        f"Expected exactly one .csv in {data_dir}, found {names}. "
        "Run prepare_hgh_data.py first."
    )
    sys.exit(1)
bioreactor_file = csv_files[0]

xlsx_files = list(data_dir.glob("*.xlsx"))
offline_file = xlsx_files[0] if len(xlsx_files) == 1 else None
if offline_file is None:
    logger.info("No (unique) offline .xlsx found; proceeding without offline data.")

# --- Instantiate pipeline components ---
online_parser = BioreactorDataParser(config=config)
offline_parser = OfflineMeasurementsParser(config=config)
cleaner = FermentationDataCleaner(config=config)
aligner = DataAligner(config=config.get("alignment", {}))
processor = FermentationDataProcessor(config=config)
visualizer = DataVisualizer(output_dir=output_dir)

try:
    # 1. Parse + clean online data
    logger.info(f"Parsing online data: {bioreactor_file.name}")
    df_online_raw = online_parser.parse_file(bioreactor_file)
    logger.info(f"Parsed online data shape: {df_online_raw.shape}")

    logger.info("Cleaning online data...")
    df_online_cleaned = cleaner.clean_data(df_online_raw)
    logger.info(f"Cleaned online data shape: {df_online_cleaned.shape}")

    # 2. Parse offline data
    df_offline_parsed = None
    offline_time_col = config.get("alignment", {}).get(
        "offline_time_col", "Process time [h]"
    )
    if offline_file is not None:
        try:
            logger.info(f"Parsing offline data: {offline_file.name}")
            df_offline_parsed = offline_parser.parse_file(offline_file)
            logger.info(f"Parsed offline data shape: {df_offline_parsed.shape}")
            if offline_time_col not in df_offline_parsed.columns:
                logger.error(
                    f"Offline data missing '{offline_time_col}' after parsing; "
                    "cannot align."
                )
                df_offline_parsed = None
            elif df_offline_parsed[offline_time_col].isnull().all():
                logger.error(f"Offline '{offline_time_col}' is all NaN; cannot align.")
                df_offline_parsed = None
        except Exception as e:
            logger.warning(f"Could not parse offline file: {e}. Proceeding without.")
            df_offline_parsed = None

    # 3. Align
    logger.info("Aligning data...")
    if df_offline_parsed is not None and not df_offline_parsed.empty:
        df_merged, df_samples = aligner.merge_datasets(
            df_online_cleaned, df_offline_parsed
        )
    else:
        df_merged = df_online_cleaned.copy()
        df_merged["is_sample"] = False
        df_samples = pd.DataFrame(columns=df_merged.columns)

    # 4. Process
    logger.info("Processing aligned data...")
    df_processed = processor.process_data(df_merged)
    logger.info(f"Processed data shape: {df_processed.shape}")

    print_cols = [
        "Process time [h]",
        "Balance [g]",
        "Feed [g]",
        "Feed [mL]",
        "Volume [mL]",
        "Is induced [-]",
    ]
    available = [c for c in print_cols if c in df_processed.columns]
    if available:
        print("\nProcessed Data Head/Tail (selected columns):")
        print(df_processed[available].head())
        print(df_processed[available].tail())

    # 5. Overview plots
    logger.info("Generating overview plots...")
    parameters_to_plot = [
        "Air flow [L/min]",
        "Total flow [L/min]",
        "O2 flow [L/min]",
        "Gas mix [-]",
        "Acid pump [%]",
        "Base pump [%]",
        "Inducer pump [%]",
        "Feed pump [%]",
        "Stirrer speed [1/min]",
        "Temperature [°C]",
        "Balance [g]",
        "pH [-]",
        "pO2 [%]",
        "Offgas CO2 [%]",
        "Offgas O2 [%]",
        "Inlet O2 [%]",
        "Inlet CO2 [%]",
        "CER [mmol/L/h]",
        "OUR [mmol/L/h]",
        "RQ [-]",
        "qO2 [mmol/g/h]",
        "qCO2 [mmol/g/h]",
        "OD600 online [-]",
        "Feed rate exp [%]",
        "Feed rate ind [%]",
        "OD600 [-]",
        "Glucose [g/L]",
        "Acetate [g/L]",
        "Protein [g/L]",
        "Protein soluble [g/L]",
        "Protein IB [g/L]",
        "Protein total calculated [g/L]",
        "Soluble fraction [%]",
        "IB fraction [%]",
        "Protein soluble yield [g/g]",
        "Protein IB yield [g/g]",
        "Protein total yield [g/g]",
        "Biomass [g/L]",
        "Viability [%]",
        "Sample volume [mL]",
        "Feed [g]",
        "Feed [mL]",
        "Base volume added [mL]",
        "Volume [mL]",
        "Cumulative sample volume [mL]",
        "Volume corrected [mL]",
        "Is induced [-]",
        "Induction time [h]",
    ]
    plotable = [p for p in parameters_to_plot if p in df_processed.columns]
    visualizer.plot_selected_parameters(
        df=df_processed, parameters=plotable, time_col="Process time [h]"
    )

    # 6. Save results
    logger.info(f"Saving results to: {output_dir}")
    processed_path = output_dir / "processed_data.csv"
    samples_path = output_dir / "samples_points_data.csv"
    df_processed.to_csv(processed_path, index=False, sep=";", decimal=".")
    logger.info(f"Saved processed data to {processed_path}")

    final_samples = (
        df_processed[df_processed["is_sample"]].copy()
        if "is_sample" in df_processed
        else pd.DataFrame()
    )
    save_samples = config.get("alignment", {}).get("save_separate_samples_file", True)
    if not final_samples.empty and save_samples:
        final_samples.to_csv(samples_path, index=False, sep=";", decimal=".")
        logger.info(f"Saved sample points data to {samples_path}")
    else:
        logger.info("No sample points data to save.")

    # 7. Final volume sanity print
    logger.info("Final check of volume calculation components (tail):")
    check_cols = [
        "Process time [h]",
        "Feed [mL]",
        "Base volume added [mL]",
        "Volume [mL]",
        "Cumulative sample volume [mL]",
        "Volume corrected [mL]",
    ]
    available = [c for c in check_cols if c in df_processed.columns]
    if available:
        print(df_processed[available].tail())

except Exception as e:
    logger.error(
        f"An error occurred during the analysis of {run_dir}: {e}", exc_info=True
    )

logger.info(f"Analysis finished for {target_run_folder_name}.")
