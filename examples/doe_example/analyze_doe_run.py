import logging
from pathlib import Path
import pandas as pd
import sys
import argparse # Use argparse for command-line arguments

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from upstream_analysis import (
    load_config, BioreactorDataParser, OfflineMeasurementsParser,
    FermentationDataCleaner, DataAligner, FermentationDataProcessor,
    DataVisualizer # Make sure DataVisualizer is imported
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================================================
# <<< --- CONFIGURE THE RUN TO ANALYZE HERE --- >>>
# ==================================================
# 1. Specify the name of the run folder inside 'doe_example'
target_run_folder_name: str = "ferm_03" # e.g., "run_01", "ferm_opt", "center_point"

# 2. Specify the exact filenames for the data files within that run's 'data' subfolder
# online_data_filename: str = "20240131_CDL_scFv_DoE_ferm_opt.csv" # <--- SET YOUR FILENAME
# offline_data_filename: str = "samples.xlsx"                 # <--- SET YOUR FILENAME or None
# ==================================================


# --- Derive Paths based on Script Location and Configuration ---
script_location = Path(__file__).resolve().parent # Should be examples/doe_example/
run_dir = (script_location / target_run_folder_name).resolve()

if not run_dir.is_dir():
    logger.error(f"Target run directory does not exist: {run_dir}")
    sys.exit(1) # Exit if the specified directory doesn't exist

config_path = run_dir / "run_config.yaml" # Assume config is named run_config.yaml
data_dir = run_dir / "data"              # Assume data is in data/ subfolder
output_dir = run_dir / "results"         # Assume results go in results/ subfolder
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load Configuration ---
logger.info(f"Analyzing run directory: {run_dir}")
if not config_path.exists():
    logger.error(f"Configuration file not found within run directory: {config_path}")
    sys.exit(1) # Exit if config is missing

logger.info(f"Using experiment configuration: {config_path}")
config = load_config(config_path) # Load the specific run's config

# --- Auto-detect Data Files ---
bioreactor_file = None
offline_file = None

logger.info(f"Searching for data files in: {data_dir}")

# Find the online data file (.csv)
csv_files = list(data_dir.glob('*.csv'))
if len(csv_files) == 1:
    bioreactor_file = csv_files[0]
    logger.info(f"Found online data file: {bioreactor_file.name}")
elif len(csv_files) == 0:
    logger.error(f"No .csv file found in {data_dir}. Cannot proceed.")
    sys.exit(1)
else:
    logger.error(f"Multiple .csv files found in {data_dir}: {[f.name for f in csv_files]}. Please ensure only one exists.")
    sys.exit(1)

# Find the offline data file (.xlsx) - Optional
xlsx_files = list(data_dir.glob('*.xlsx'))
if len(xlsx_files) == 1:
    offline_file = xlsx_files[0]
    offline_data_filename = offline_file.name # Store name for logging if needed later
    logger.info(f"Found offline data file: {offline_file.name}")
elif len(xlsx_files) == 0:
    logger.info("No .xlsx file found in data directory. Proceeding without offline data.")
    offline_file = None
    offline_data_filename = None # No offline file specified
else:
    logger.warning(f"Multiple .xlsx files found in {data_dir}: {[f.name for f in xlsx_files]}. Proceeding without offline data.")
    offline_file = None
    offline_data_filename = None # Treat as if none found


# --- Instantiate Components (pass the loaded config) ---
online_parser = BioreactorDataParser(config=config)
offline_parser = OfflineMeasurementsParser(config=config)
cleaner = FermentationDataCleaner(config=config)
aligner_config = config.get('alignment', {})
aligner = DataAligner(config=aligner_config)
processor = FermentationDataProcessor(config=config)
visualizer = DataVisualizer(output_dir=output_dir) # Instantiate visualizer

# --- Run Steps within a single try block ---
try:
    # 1. Parse Online Data
    logger.info(f"Parsing online data: {bioreactor_file}")
    # No need to check exists() again, already done above
    df_online_raw = online_parser.parse_file(bioreactor_file)
    logger.info(f"Parsed online data shape: {df_online_raw.shape}")

    # 2. Clean Online Data
    logger.info("Cleaning online data...")
    df_online_cleaned = cleaner.clean_data(df_online_raw)
    logger.info(f"Cleaned online data shape: {df_online_cleaned.shape}")

    # 3. Parse Offline Data
    df_offline_parsed = None
    offline_time_col = config.get('alignment',{}).get('offline_time_col','process_time')
    # Check if offline_file path was successfully determined above
    if offline_file is not None:
        try:
            logger.info(f"Parsing offline data: {offline_file}")
            df_offline_parsed = offline_parser.parse_file(offline_file)
            logger.info(f"Parsed offline data shape: {df_offline_parsed.shape}")
            # Validate parsed offline data
            if offline_time_col not in df_offline_parsed.columns:
                 logger.error(f"Offline data missing required time column '{offline_time_col}' AFTER parsing/renaming. Cannot align.")
                 df_offline_parsed = None
            elif df_offline_parsed[offline_time_col].isnull().all():
                 logger.error(f"Offline data time column '{offline_time_col}' contains only NaNs. Cannot align.")
                 df_offline_parsed = None
        except Exception as e:
            logger.warning(f"Could not parse offline file {offline_file}: {e}. Proceeding without.")
            df_offline_parsed = None
    # Logging for no offline file handled during file detection

    # 4. Align Data
    logger.info("Aligning data...")
    if df_offline_parsed is not None and not df_offline_parsed.empty:
         df_merged, df_samples = aligner.merge_datasets(df_online_cleaned, df_offline_parsed)
    else:
        logger.info("No valid offline data to align.")
        df_merged = df_online_cleaned.copy()
        df_merged['is_sample'] = False
        df_samples = pd.DataFrame(columns=df_merged.columns)

    # 5. Process Data (This now includes adding 'is_induced')
    logger.info("Processing aligned data...")
    df_processed = processor.process_data(df_merged)
    logger.info(f"Processed data shape: {df_processed.shape}")
    print_cols = ['process_time', 'balance', 'feed_g', 'feed_ml', 'volume_ml', 'is_induced'] # Add is_induced
    available_print_cols = [col for col in print_cols if col in df_processed.columns]
    if available_print_cols:
         print("\nProcessed Data Head/Tail (Selected Columns):\n", df_processed[available_print_cols].head())
         print(df_processed[available_print_cols].tail())
    else:
         print("\nRequired columns for printing head/tail not found.")


    # 6. Plotting Step
    logger.info("Generating overview plots...")
    parameters_to_plot = [
        'air_flow', 'total_flow', 'o2_flow', 'gas_mix', 'base_pump_pct',
        'base_pump_duration', 'inducer_pump_pct', 'inducer_pump_duration',
        'feed_pump_pct', 'feed_pump_duration', 'stirrer_speed',
        # Check for standard 'temperature' first, then original name
        'temperature', 'Temperature, Â°C', # Add both possibilities
        'balance', 'analog_io1', 'analog_io2', 'ph', 'po2',
        'offgas_co2', 'offgas_o2', 'od600_online', 'feedrate_exp',
        'feedrate_ind', 'od600', 'glucose_g_l', 'acetate_g_l',
        'protein_g_l', 'biomass_g_l', 'viability_pct', 'sample_volume_ml',
        'feed_g', 'feed_ml', 'base_volume_added_ml', 'volume_ml',
        'cumulative_sample_volume_ml', 'volume_corrected_ml',
        'is_induced' # Plot the new indicator
    ]
    # Filter list to only plot columns that actually exist in the final DataFrame
    plotable_parameters = [p for p in parameters_to_plot if p in df_processed.columns]

    visualizer.plot_selected_parameters(
        df=df_processed,
        parameters=plotable_parameters, # Use the filtered list
        time_col='process_time'
    )


    # 7. Save Results
    logger.info(f"Saving results to: {output_dir}")
    processed_output_path = output_dir / "processed_data.csv"
    samples_output_path = output_dir / "samples_points_data.csv"

    df_processed.to_csv(processed_output_path, index=False, sep=';', decimal='.')
    logger.info(f"Saved processed data to {processed_output_path} (decimal='.')")

    final_samples_df = df_processed[df_processed['is_sample']].copy() if 'is_sample' in df_processed else pd.DataFrame()
    save_samples = config.get('alignment', {}).get('save_separate_samples_file', True)
    if not final_samples_df.empty and save_samples:
        final_samples_df.to_csv(samples_output_path, index=False, sep=';', decimal='.')
        logger.info(f"Saved sample points data to {samples_output_path} (decimal='.')")
    else:
        logger.info("No sample points data to save.")

    # 8. Final Check Print
    logger.info("Final check of volume calculation components (Tail):")
    check_cols = ['process_time', 'feed_ml', 'base_volume_added_ml', 'volume_ml', 'cumulative_sample_volume_ml', 'volume_corrected_ml']
    available_check_cols = [col for col in check_cols if col in df_processed.columns]
    if available_check_cols:
        print(df_processed[available_check_cols].tail())
    else:
        print("Required columns for final check not available.")

except Exception as e:
    logger.error(f"An error occurred during the analysis of {run_dir}: {e}", exc_info=True)

logger.info(f"Analysis finished for {run_dir}.")