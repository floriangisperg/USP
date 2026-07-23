import logging
from pathlib import Path
import pandas as pd
import sys
import argparse
import numpy as np

# --- Path Setup ---
# Get the correct path: from examples/LDH/analyze_ldh_run.py to src/
project_root = Path(__file__).resolve().parent.parent.parent  # Go up from examples/LDH to project root
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
print(f"Added to path: {src_path}")
print(f"Path exists: {src_path.exists()}")

from upstream_analysis import (
    load_config, BioreactorDataParser, OfflineMeasurementsParser,
    FermentationDataCleaner, DataAligner, FermentationDataProcessor,
    DataVisualizer, InteractiveDashboard
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================================================
# CONFIGURE THE RUNS TO ANALYZE HERE
# ==================================================
# Set to specific run name or "all" to process all runs
target_run_folder_name: str = "all"  # Options: "all", "ferm_01", "ferm_02", etc.

def analyze_single_run(run_name, script_location):
    """Analyze a single run."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ANALYZING RUN: {run_name}")
    logger.info(f"{'='*60}")

    # --- Derive Paths ---
    run_dir = (script_location / run_name).resolve()
    config_path = run_dir / "run_config.yaml"
    data_dir = run_dir / "data"
    output_dir = run_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if run exists
    if not run_dir.is_dir():
        logger.error(f"Run directory does not exist: {run_dir}")
        return False

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False

    try:
        # --- Load Configuration ---
        logger.info(f"Analyzing run directory: {run_dir}")
        config = load_config(config_path)

        # --- Auto-detect Data Files ---
        bioreactor_file = None
        offline_file = None

        logger.info(f"Searching for data files in: {data_dir}")

        # Find the online data file (first try CSV, then Excel with Rawdata in name)
        csv_files = list(data_dir.glob('*.csv'))
        if len(csv_files) >= 1:
            bioreactor_file = csv_files[0]  # Take first CSV file
            logger.info(f"Found online data file: {bioreactor_file.name}")
        else:
            # Look for Excel files with Rawdata in the name for online bioreactor data
            rawdata_files = [f for f in data_dir.glob('*.xlsx') if 'Rawdata' in f.name or 'rawdata' in f.name.lower()]
            if len(rawdata_files) >= 1:
                bioreactor_file = rawdata_files[0]
                logger.info(f"Found online bioreactor data file: {bioreactor_file.name}")
            else:
                logger.info(f"No online data files found in {data_dir}. Proceeding with offline data only.")

        # Find the offline data file (.xlsx) - look for samples.xlsx specifically
        samples_file = data_dir / "samples.xlsx"
        if samples_file.exists():
            offline_file = samples_file
            logger.info(f"Found offline data file: {offline_file.name}")
        else:
            # If no samples.xlsx, look for any xlsx file that might contain samples
            xlsx_files = [f for f in data_dir.glob('*.xlsx') if 'Rawdata' not in f.name]
            if len(xlsx_files) >= 1:
                offline_file = xlsx_files[0]
                logger.info(f"Found potential offline data file: {offline_file.name}")
            else:
                logger.info("No offline sample data file found.")
                offline_file = None

        # Check if we have any data
        if bioreactor_file is None and offline_file is None:
            logger.warning(f"No data files found for run {run_name}. Skipping.")
            return False

        # --- Instantiate Components ---
        online_parser = BioreactorDataParser(config=config)
        offline_parser = OfflineMeasurementsParser(config=config)
        cleaner = FermentationDataCleaner(config=config)
        aligner_config = config.get('alignment', {})
        aligner = DataAligner(config=aligner_config)
        processor = FermentationDataProcessor(config=config)
        visualizer = DataVisualizer(output_dir=output_dir)

        # --- Run Analysis ---
        # 1. Parse Online Data (if available)
        df_online_raw = None
        if bioreactor_file:
            logger.info(f"Parsing online data: {bioreactor_file}")
            df_online_raw = online_parser.parse_file(bioreactor_file)
            logger.info(f"Parsed online data shape: {df_online_raw.shape}")

        # 2. Clean Online Data
        df_online_cleaned = None
        if df_online_raw is not None:
            logger.info("Cleaning online data...")
            df_online_cleaned = cleaner.clean_data(df_online_raw)
            logger.info(f"Cleaned online data shape: {df_online_cleaned.shape}")

        # 3. Parse Offline Data
        df_offline_parsed = None
        offline_time_col = config.get('alignment', {}).get('offline_time_col', 'Process time [h]')
        if offline_file is not None:
            try:
                logger.info(f"Parsing offline data: {offline_file}")
                df_offline_parsed = offline_parser.parse_file(offline_file)
                logger.info(f"Parsed offline data shape: {df_offline_parsed.shape}")

                # Show protein-related columns
                protein_cols = [c for c in df_offline_parsed.columns if 'protein' in c.lower() or 'IB' in c]
                if protein_cols:
                    logger.info(f"Protein columns found: {protein_cols}")

                    # Show some protein data if available
                    print(f"\n{run_name} - Sample protein data:")
                    sample_data = df_offline_parsed[['Process time [h]'] + protein_cols].dropna(how='all', subset=['Process time [h]'])
                    if not sample_data.empty:
                        print(sample_data.head(10).round(3))

            except Exception as e:
                logger.warning(f"Could not parse offline file {offline_file}: {e}. Proceeding without.")
                df_offline_parsed = None

        # 4. Align Data
        logger.info("Aligning data...")
        if df_online_cleaned is not None and df_offline_parsed is not None and not df_offline_parsed.empty:
            df_merged, df_samples = aligner.merge_datasets(df_online_cleaned, df_offline_parsed)
        elif df_online_cleaned is not None:
            logger.info("No offline data to align. Using online data only.")
            df_merged = df_online_cleaned.copy()
            df_merged['is_sample'] = False
            df_samples = pd.DataFrame(columns=df_merged.columns)
        elif df_offline_parsed is not None:
            logger.info("No online data to align. Using offline data only.")
            df_merged = df_offline_parsed.copy()
            df_merged['is_sample'] = True

            # Don't create dummy process phases for offline-only data
            # The protein calculations should work without process phases
            # Add minimal required columns for processor to handle offline-only data
            df_merged['Balance [g]'] = np.nan
            df_merged['Base pump [%]'] = 0
            df_merged['Base pump duration [-]'] = 0
            df_samples = df_merged.copy()
        else:
            logger.error("No valid data to process.")
            return False

        # 5. Process Data
        logger.info("Processing aligned data...")
        df_processed = processor.process_data(df_merged)
        logger.info(f"Processed data shape: {df_processed.shape}")

        # Show calculated protein parameters
        calc_protein_cols = [c for c in df_processed.columns if any(x in c for x in ['fraction', 'yield', 'total calculated'])]
        if calc_protein_cols:
            logger.info(f"Calculated protein parameters: {calc_protein_cols}")
            sample_data = df_processed[df_processed['is_sample']] if 'is_sample' in df_processed else df_processed
            print(f"\n{run_name} - Calculated protein parameters:")
            relevant_cols = ['Process time [h]'] + [c for c in ['Protein soluble [g/L]', 'Protein IB [g/L]', 'Protein total calculated [g/L]',
                                                          'Soluble fraction [%]', 'IB fraction [%]', 'Protein total yield [g/g]']
                                                   if c in sample_data.columns]
            if len(relevant_cols) > 1:
                print(sample_data[relevant_cols].dropna(how='all', subset=['Process time [h]']).round(3))

        # 6. Plotting
        logger.info("Generating overview plots...")
        parameters_to_plot = [
            'Air flow [L/min]', 'Total flow [L/min]', 'O2 flow [L/min]', 'Gas mix [-]', 'Base pump [%]',
            'Base pump duration [-]', 'Inducer pump [%]', 'Inducer pump duration [-]',
            'Feed pump [%]', 'Feed pump duration [-]', 'Stirrer speed [1/min]',
            'Temperature [°C]', 'Balance [g]', 'Analog IO1 [%]', 'Analog IO2 [%]', 'pH [-]', 'pO2 [%]',
            'Offgas CO2 [%]', 'Offgas O2 [%]', 'Inlet O2 [%]', 'Inlet CO2 [%]',
            'CER [mmol/L/h]', 'OUR [mmol/L/h]', 'RQ [-]', 'qO2 [mmol/g/h]', 'qCO2 [mmol/g/h]',
            'C in [C-mol]', 'C out [C-mol]', 'C recovered [%]', 'C balance closure [%]',
            'C to CO2 [% of fed]', 'C to biomass [% of fed]', 'C to product [% of fed]', 'C residual glucose [% of fed]',
            'C from feed [C-mol]', 'C from batch [C-mol]', 'C from CO2 [C-mol]',
            'C in biomass [C-mol]', 'C in product [C-mol]', 'C in acetate [C-mol]', 'C residual glucose [C-mol]',
            'OD600 online [-]', 'Feed rate exp [%]',
            'Feed rate ind [%]', 'OD600 [-]', 'Glucose [g/L]', 'Acetate [g/L]',
            'Protein [g/L]', 'Protein soluble [g/L]', 'Protein IB [g/L]', 'Protein total calculated [g/L]',
            'Soluble fraction [%]', 'IB fraction [%]', 'Protein soluble yield [g/g]',
            'Protein IB yield [g/g]', 'Protein total yield [g/g]', 'Biomass [g/L]', 'Viability [%]',
            'Feed [g]', 'Feed [mL]', 'Base volume added [mL]', 'Volume [mL]',
            'Cumulative sample volume [mL]', 'Volume corrected [mL]', 'Is induced [-]', 'Induction time [h]'
        ]

        # Filter to only plot columns that exist
        plotable_parameters = [p for p in parameters_to_plot if p in df_processed.columns]
        logger.info(f"Plotting {len(plotable_parameters)} parameters")

        visualizer.plot_selected_parameters(
            df=df_processed,
            parameters=plotable_parameters,
            time_col='Process time [h]'
        )

        # 6b. Interactive HTML dashboard (one tabbed file; needs bokeh - skipped if absent)
        try:
            logger.info("Generating interactive HTML dashboard...")
            InteractiveDashboard(output_dir, time_col='Process time [h]').plot_selected_parameters(
                df_processed, parameters=plotable_parameters, run_name=run_name
            )
        except ImportError as e:
            logger.warning(f"Skipped interactive dashboard (bokeh not installed): {e}")

        # 7. Save Results (xlsx only)
        logger.info(f"Saving results to: {output_dir}")
        processed_output_path = output_dir / "processed_data.xlsx"

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(processed_output_path, engine='openpyxl') as writer:
            df_processed.to_excel(writer, sheet_name='Processed_Data', index=False)
            
            final_samples_df = df_processed[df_processed['is_sample']].copy() if 'is_sample' in df_processed else pd.DataFrame()
            save_samples = config.get('alignment', {}).get('save_separate_samples_file', True)
            if not final_samples_df.empty and save_samples:
                final_samples_df.to_excel(writer, sheet_name='Sample_Points', index=False)
                logger.info(f"Saved {len(final_samples_df)} sample points to Sample_Points sheet")
        
        logger.info(f"Saved processed data to {processed_output_path}")

        logger.info(f"Successfully completed analysis for {run_name}")
        return True

    except Exception as e:
        logger.error(f"An error occurred during analysis of {run_name}: {e}", exc_info=True)
        return False

# --- Main execution ---
if __name__ == "__main__":
    script_location = Path(__file__).resolve().parent  # Should be examples/LDH/

    if target_run_folder_name == "all":
        # Find all ferm_* directories
        run_dirs = [d.name for d in script_location.iterdir() if d.is_dir() and (d.name.startswith("ferm_") or (d.name.startswith("F") and len(d.name) > 1 and d.name[1].isdigit()))]
        run_dirs.sort()  # Sort to ensure consistent order
        logger.info(f"Found {len(run_dirs)} runs to process: {run_dirs}")

        successful_runs = []
        failed_runs = []

        for run_name in run_dirs:
            success = analyze_single_run(run_name, script_location)
            if success:
                successful_runs.append(run_name)
            else:
                failed_runs.append(run_name)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully processed: {len(successful_runs)} runs")
        if successful_runs:
            logger.info(f"  - {', '.join(successful_runs)}")
        logger.info(f"Failed to process: {len(failed_runs)} runs")
        if failed_runs:
            logger.info(f"  - {', '.join(failed_runs)}")

    else:
        # Process single run
        success = analyze_single_run(target_run_folder_name, script_location)
        if not success:
            logger.error(f"Failed to process run: {target_run_folder_name}")
            sys.exit(1)

    logger.info("\nAll analysis completed!")