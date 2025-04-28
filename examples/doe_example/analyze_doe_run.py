# examples/doe_example/analyze_doe_run.py
import logging
from pathlib import Path
import pandas as pd
import sys

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
target_run_folder_name: str = "ferm_opt" # e.g., "run_01", "ferm_opt"
online_data_filename: str = "20240131_CDL_scFv_DoE_ferm_opt.csv" # <--- SET YOUR FILENAME
offline_data_filename: str = "samples.xlsx"                 # <--- SET YOUR FILENAME or None
# ==================================================


# --- Derive Paths ---
script_location = Path(__file__).resolve().parent
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

logger.info(f"Using experiment configuration: {config_path}")
config = load_config(config_path)

# --- Construct Full Data File Paths ---
bioreactor_file = data_dir / online_data_filename
offline_file = data_dir / offline_data_filename if offline_data_filename else None


# --- Instantiate Components ---
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
    if not bioreactor_file.exists():
        raise FileNotFoundError(f"Online data file not found: {bioreactor_file}")
    df_online_raw = online_parser.parse_file(bioreactor_file)
    logger.info(f"Parsed online data shape: {df_online_raw.shape}")

    # 2. Clean Online Data
    logger.info("Cleaning online data...")
    df_online_cleaned = cleaner.clean_data(df_online_raw)
    logger.info(f"Cleaned online data shape: {df_online_cleaned.shape}")

    # 3. Parse Offline Data
    df_offline_parsed = None
    offline_time_col = config.get('alignment',{}).get('offline_time_col','process_time')
    if offline_file and offline_file.exists():
        try:
            logger.info(f"Parsing offline data: {offline_file}")
            df_offline_parsed = offline_parser.parse_file(offline_file)
            logger.info(f"Parsed offline data shape: {df_offline_parsed.shape}")
            if offline_time_col not in df_offline_parsed.columns:
                 logger.error(f"Offline data missing required time column '{offline_time_col}'. Cannot align.")
                 df_offline_parsed = None
            elif df_offline_parsed[offline_time_col].isnull().all():
                 logger.error(f"Offline data time column '{offline_time_col}' contains only NaNs. Cannot align.")
                 df_offline_parsed = None
        except Exception as e:
            logger.warning(f"Could not parse offline file {offline_file}: {e}. Proceeding without.")
            df_offline_parsed = None
    elif offline_data_filename:
        logger.info(f"Offline file not found: {offline_file}. Proceeding without.")
    else:
        logger.info("No offline data file specified. Proceeding without.")


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
    logger.info("Processing aligned data (calculating feed, volume, induction)...")
    df_processed = processor.process_data(df_merged)
    logger.info(f"Processed data shape: {df_processed.shape}")
    print_cols = ['process_time', 'balance', 'feed_g', 'feed_ml', 'volume_ml', 'is_induced'] # Add is_induced
    available_print_cols = [col for col in print_cols if col in df_processed.columns]
    if available_print_cols:
         print("\nProcessed Data Head (with feed/volume/induction):\n", df_processed[available_print_cols].head())
         print(df_processed[available_print_cols].tail())
    else:
         print("\nRequired columns for printing head/tail not found.")


    # 6. Plotting Step (Moved here)
    logger.info("Generating overview plots...")
    parameters_to_plot = [
        'air_flow', 'total_flow', 'o2_flow', 'gas_mix', 'base_pump_pct',
        'base_pump_duration', 'inducer_pump_pct', 'inducer_pump_duration',
        'feed_pump_pct', 'feed_pump_duration', 'stirrer_speed',
        'temperature', # Add back if renaming works, otherwise use 'Temperature, Â°C'
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
    if 'Temperature, Â°C' in df_processed.columns and 'temperature' not in plotable_parameters:
        plotable_parameters.append('Temperature, Â°C') # Add original name if standard name missing

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