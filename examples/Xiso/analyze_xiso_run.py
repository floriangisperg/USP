# examples/Xiso/analyze_xiso_run.py
"""
Analyze a single Xiso DoE fermentation run through the USP pipeline.

Each run lives in its own folder under examples/Xiso/::

    examples/Xiso/<run>/
        run_config.yaml                       <- run-specific overrides (fill the TODOs)
        data/
            20251110_..._XISO.0X.xlsx          <- online bioreactor export (Excel; same columns as the CSVs)
            Evaluation_FXX.xlsx                <- your external evaluation workbook (reference only; NOT parsed)
            samples.xlsx                       <- atline/lab samples (you fill this; flat table)
        results/                              <- generated CSVs + plots

Usage::

    python analyze_xiso_run.py                # analyzes the run named below
    python analyze_xiso_run.py 01_Ferm02      # analyzes a specific run folder
"""
import logging
from pathlib import Path
import sys
import pandas as pd

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from upstream_analysis import (
    load_config, BioreactorDataParser, OfflineMeasurementsParser,
    FermentationDataCleaner, DataAligner, FermentationDataProcessor,
    DataVisualizer, InteractiveDashboard,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================================================
# <<< --- CONFIGURE THE RUN TO ANALYZE HERE --- >>>
# ==================================================
target_run_folder_name: str = "00_Ferm01"   # e.g. "00_Ferm01", "01_Ferm02", "02_Ferm03"
# Optional: pass the run folder as a command-line argument.
if len(sys.argv) > 1:
    target_run_folder_name = sys.argv[1]
# ==================================================

# --- Derive Paths ---
script_location = Path(__file__).resolve().parent   # examples/Xiso/
run_dir = (script_location / target_run_folder_name).resolve()

if not run_dir.is_dir():
    logger.error(f"Target run directory does not exist: {run_dir}")
    sys.exit(1)

config_path = run_dir / "run_config.yaml"
data_dir = run_dir / "data"
output_dir = run_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load Configuration ---
logger.info(f"Analyzing run directory: {run_dir}")
if not config_path.exists():
    logger.error(f"Configuration file not found: {config_path}")
    sys.exit(1)
config = load_config(config_path)

# --- Locate Data Files ---
# Online = the XISO export xlsx (exclude the Evaluation workbook and the samples template).
online_candidates = [
    f for f in data_dir.glob("*.xlsx")
    if f.name.lower() != "samples.xlsx" and "evaluation" not in f.name.lower()
]
if len(online_candidates) == 1:
    bioreactor_file = online_candidates[0]
    logger.info(f"Found online data file: {bioreactor_file.name}")
elif len(online_candidates) == 0:
    logger.error(f"No online .xlsx found in {data_dir} (expected a '..._XISO.0X.xlsx').")
    sys.exit(1)
else:
    logger.error(f"Multiple online candidates in {data_dir}: {[f.name for f in online_candidates]}. "
                 f"Keep exactly one '*XISO*.xlsx'.")
    sys.exit(1)

# Offline = the flat atline samples template (optional; empty = header only).
offline_file = data_dir / "samples.xlsx"
if not offline_file.exists():
    logger.info(f"No samples.xlsx in {data_dir}; proceeding without offline data.")
    offline_file = None

# --- Instantiate Components ---
online_parser = BioreactorDataParser(config=config)
offline_parser = OfflineMeasurementsParser(config=config)
cleaner = FermentationDataCleaner(config=config)
aligner = DataAligner(config=config.get("alignment", {}))
processor = FermentationDataProcessor(config=config)
visualizer = DataVisualizer(output_dir=output_dir)

# --- Run Pipeline ---
try:
    # 1. Parse + clean online data
    logger.info(f"Parsing online data: {bioreactor_file}")
    df_online_raw = online_parser.parse_file(bioreactor_file)
    logger.info(f"Parsed online data shape: {df_online_raw.shape}")

    logger.info("Cleaning online data...")
    df_online_cleaned = cleaner.clean_data(df_online_raw)
    logger.info(f"Cleaned online data shape: {df_online_cleaned.shape}")

    # 2. Parse offline samples (if any rows are present)
    df_offline_parsed = None
    offline_time_col = config.get("alignment", {}).get("offline_time_col", "Process time [h]")
    if offline_file is not None:
        try:
            logger.info(f"Parsing offline data: {offline_file}")
            df_offline_parsed = offline_parser.parse_file(offline_file)
            logger.info(f"Parsed offline data shape: {df_offline_parsed.shape}")
            if df_offline_parsed.empty:
                logger.info("samples.xlsx is empty (header only) - proceeding without offline data.")
                df_offline_parsed = None
            elif offline_time_col not in df_offline_parsed.columns:
                logger.error(f"Offline data missing time column '{offline_time_col}' after parsing. Cannot align.")
                df_offline_parsed = None
            elif df_offline_parsed[offline_time_col].isnull().all():
                logger.error(f"Offline time column '{offline_time_col}' is all NaN. Cannot align.")
                df_offline_parsed = None
        except Exception as e:
            logger.warning(f"Could not parse offline file {offline_file}: {e}. Proceeding without.")
            df_offline_parsed = None

    # 3. Align online + offline
    logger.info("Aligning data...")
    if df_offline_parsed is not None and not df_offline_parsed.empty:
        df_merged, df_samples = aligner.merge_datasets(df_online_cleaned, df_offline_parsed)
    else:
        logger.info("No valid offline data to align.")
        df_merged = df_online_cleaned.copy()
        df_merged["is_sample"] = False
        df_samples = pd.DataFrame(columns=df_merged.columns)

    # 4. Process (feed, volume, induction, gas exchange, carbon balance, rates/yields)
    logger.info("Processing aligned data...")
    df_processed = processor.process_data(df_merged)
    logger.info(f"Processed data shape: {df_processed.shape}")

    # 5. Plot
    logger.info("Generating overview plots...")
    parameters_to_plot = [
        'Air flow [L/min]', 'Total flow [L/min]', 'O2 flow [L/min]', 'Gas mix [-]',
        'Base pump [%]', 'Base pump duration [-]', 'Inducer pump [%]', 'Inducer pump duration [-]',
        'Feed pump [%]', 'Feed pump duration [-]', 'Stirrer speed [1/min]', 'Temperature [°C]',
        'Balance [g]', 'pH [-]', 'pO2 [%]', 'Offgas CO2 [%]', 'Offgas O2 [%]',
        'Inlet O2 [%]', 'Inlet CO2 [%]', 'CER [mmol/L/h]', 'OUR [mmol/L/h]', 'RQ [-]',
        'qO2 [mmol/g/h]', 'qCO2 [mmol/g/h]', 'Feed rate exp [%]', 'Feed rate ind [%]',
        'OD600 online [-]', 'OD600 [-]', 'Glucose [g/L]', 'Acetate [g/L]',
        'Protein [g/L]', 'Protein soluble [g/L]', 'Protein IB [g/L]', 'Biomass [g/L]',
        'Viability [%]', 'Sample volume [mL]', 'Feed [g]', 'Feed [mL]', 'Base volume added [mL]',
        'Volume [mL]', 'Cumulative sample volume [mL]', 'Volume corrected [mL]',
        'Is induced [-]', 'Induction time [h]',
    ]
    plotable = [p for p in parameters_to_plot if p in df_processed.columns]
    visualizer.plot_selected_parameters(df=df_processed, parameters=plotable, time_col="Process time [h]")

    # 5b. Interactive HTML dashboard (one tabbed file; needs bokeh - skipped if absent)
    try:
        logger.info("Generating interactive HTML dashboard...")
        InteractiveDashboard(output_dir, time_col="Process time [h]").plot_selected_parameters(
            df_processed, parameters=plotable, run_name=target_run_folder_name
        )
    except ImportError as e:
        logger.warning(f"Skipped interactive dashboard (bokeh not installed): {e}")

    # 6. Save
    logger.info(f"Saving results to: {output_dir}")
    processed_output_path = output_dir / "processed_data.csv"
    samples_output_path = output_dir / "samples_points_data.csv"
    df_processed.to_csv(processed_output_path, index=False, sep=";", decimal=".")
    logger.info(f"Saved processed data to {processed_output_path}")

    final_samples_df = df_processed[df_processed["is_sample"]].copy() if "is_sample" in df_processed else pd.DataFrame()
    save_samples = config.get("alignment", {}).get("save_separate_samples_file", True)
    if not final_samples_df.empty and save_samples:
        final_samples_df.to_csv(samples_output_path, index=False, sep=";", decimal=".")
        logger.info(f"Saved sample points data to {samples_output_path}")
    else:
        logger.info("No sample points data to save.")

    # 7. Final check
    logger.info("Final check of volume components (tail):")
    check_cols = ['Process time [h]', 'Feed [mL]', 'Base volume added [mL]', 'Volume [mL]',
                  'Cumulative sample volume [mL]', 'Volume corrected [mL]']
    avail = [c for c in check_cols if c in df_processed.columns]
    print(df_processed[avail].tail() if avail else "Volume columns not available.")

except Exception as e:
    logger.error(f"An error occurred during analysis of {run_dir}: {e}", exc_info=True)

logger.info(f"Analysis finished for {run_dir}.")
