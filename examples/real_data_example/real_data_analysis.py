# examples/real_data_example/real_data_analysis.py
import logging
from pathlib import Path
import pandas as pd
import sys

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from upstream_analysis import (
    load_config,
    BioreactorDataParser,
    OfflineMeasurementsParser,
    FermentationDataCleaner,
    DataAligner
)
# --- Import the Processor ---
from upstream_analysis.data.processor import FermentationDataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Paths ---
script_dir = Path(__file__).resolve().parent
config_path = script_dir / "run_config.yaml"
example_data_dir = script_dir / "data"
output_dir = script_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Using experiment configuration: {config_path}")
config = load_config(config_path)

bioreactor_file = example_data_dir / "rawdata.csv"
offline_file = example_data_dir / "samples.xlsx"

# --- Instantiate Components ---
online_parser = BioreactorDataParser(config=config)
offline_parser = OfflineMeasurementsParser(config=config)
cleaner = FermentationDataCleaner(config=config)
aligner = DataAligner(config=config['alignment'])
# --- Instantiate Processor ---
processor = FermentationDataProcessor(config=config) # Pass the full config

# --- Run Steps ---
try:
    # 1. Parse Online Data
    logger.info(f"Parsing online data: {bioreactor_file}")
    if not bioreactor_file.exists(): raise FileNotFoundError(f"Online data file not found: {bioreactor_file}")
    df_online_raw = online_parser.parse_file(bioreactor_file)
    logger.info(f"Parsed online data shape: {df_online_raw.shape}")

    # 2. Clean Online Data
    logger.info("Cleaning online data...")
    df_online_cleaned = cleaner.clean_data(df_online_raw)
    logger.info(f"Cleaned online data shape: {df_online_cleaned.shape}")

    # 3. Parse Offline Data
    df_offline_parsed = None
    if offline_file.exists():
        try:
            logger.info(f"Parsing offline data: {offline_file}")
            df_offline_parsed = offline_parser.parse_file(offline_file)
            logger.info(f"Parsed offline data shape: {df_offline_parsed.shape}")
            # Check for time column
            offline_time_col = config['alignment']['offline_time_col']
            if offline_time_col not in df_offline_parsed.columns:
                 logger.error(f"Offline data missing required time column '{offline_time_col}'. Cannot align.")
                 df_offline_parsed = None # Treat as if file wasn't parsed successfully for alignment
            elif df_offline_parsed[offline_time_col].isnull().all():
                 logger.error(f"Offline data time column '{offline_time_col}' contains only NaNs. Cannot align.")
                 df_offline_parsed = None

        except Exception as e:
            logger.warning(f"Could not parse offline file {offline_file}: {e}. Proceeding without.")
            df_offline_parsed = None
    else:
        logger.info(f"Offline file not found: {offline_file}. Proceeding without.")

    # 4. Align Data
    logger.info("Aligning data...")
    if df_offline_parsed is not None and not df_offline_parsed.empty:
        df_merged, df_samples = aligner.merge_datasets(df_online_cleaned, df_offline_parsed)
    else:
        df_merged = df_online_cleaned.copy()
        df_merged['is_sample'] = False
        df_samples = pd.DataFrame(columns=df_merged.columns) # Empty DF

    # --- 5. Process Data ---
    logger.info("Processing aligned data (calculating feed, volume)...")
    df_processed = processor.process_data(df_merged)
    logger.info(f"Processed data shape: {df_processed.shape}")

    # --- ADD DEBUG LINE ---
    logger.debug(f"Columns available just before final print: {list(df_processed.columns)}")
    # --- END DEBUG LINE ---

    print("\nProcessed Data Head (with feed/volume):\n", df_processed[['process_time', 'balance', 'feed_g', 'feed_ml', 'volume_ml']].head())
    print(df_processed[['process_time', 'balance', 'feed_g', 'feed_ml', 'volume_ml']].tail())

    # --- Save Results ---
    logger.info(f"Saving results to: {output_dir}")
    processed_output_path = output_dir / "processed_data.csv" # Changed filename
    samples_output_path = output_dir / "samples_points_data.csv"

    # Save the *processed* data now
    df_processed.to_csv(processed_output_path, index=False, sep=';', decimal=',')
    logger.info(f"Saved processed data to {processed_output_path}")

    # End of real_data_analysis.py, before saving
    logger.info("Final check of volume calculation components (Tail):")
    print(df_processed[['process_time', 'feed_ml', 'base_volume_added_ml', 'volume_ml', 'cumulative_sample_volume_ml',
                        'volume_corrected_ml']].tail())

    logger.info(f"Saving results to: {output_dir}")

    # Conditionally save samples file (which now includes processed columns)
    # Extract samples *after* processing if you want processed columns in samples file
    final_samples_df = df_processed[df_processed['is_sample']].copy() if 'is_sample' in df_processed else pd.DataFrame()

    save_samples = config.get('alignment', {}).get('save_separate_samples_file', True)
    if not final_samples_df.empty and save_samples:
        final_samples_df.to_csv(samples_output_path, index=False, sep=';', decimal=',')
        logger.info(f"Saved sample points data to {samples_output_path}")
    else:
        logger.info("No sample points data to save.")

except Exception as e:
    logger.error(f"An error occurred during the script execution: {e}", exc_info=True)

logger.info("Script finished.")