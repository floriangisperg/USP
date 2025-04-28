# src/upstream_analysis/data/cleaning.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FermentationDataCleaner:
    """Cleans raw online fermentation data (filter phases, correct sensors, standardize time)."""

    def __init__(self, config: dict):
        self.config = config
        self.cleaning_config = config.get('cleaning', {})
        # Get specific settings needed within this class
        self.balance_cols = self.cleaning_config.get('balance_columns', ['balance'])
        self.tare_threshold = self.cleaning_config.get('balance_tare_threshold', -100)
        self.median_filter_cols = self.cleaning_config.get('median_filter_columns', ['balance'])
        self.median_window = self.cleaning_config.get('median_filter_window', 20)
        self.median_threshold = self.cleaning_config.get('median_filter_threshold', 30)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured cleaning steps to the raw online data."""
        cleaned_df = df.copy()
        logger.info("Starting data cleaning process...")

        phase_col = 'process_phase'
        balance_mode = self.config.get('processing_parameters', {}).get('balance_mode',
                                                                        'reactor_weight')  # Get balance mode

        # --- Handle Preparation Phase ---
        rename_prep = self.cleaning_config.get('rename_preparation_to_batch', False)
        filter_prep = self.cleaning_config.get('filter_preparation', True)
        logger.info(
            f"Cleaning config flags: rename_prep={rename_prep}, filter_prep={filter_prep}, balance_mode='{balance_mode}'")

        if phase_col not in cleaned_df.columns:
            logger.warning(f"No '{phase_col}' column found. Skipping phase renaming/filtering.")
        elif rename_prep:
            logger.info("Executing rename logic...")
            prep_mask = cleaned_df[phase_col].astype(str).str.strip() == 'Preparation'
            rows_renamed = prep_mask.sum()
            if rows_renamed > 0:
                cleaned_df.loc[prep_mask, phase_col] = 'Batch'
                logger.info(f"Renamed {rows_renamed} 'Preparation' rows to 'Batch'.")
            else:
                logger.info("No 'Preparation' phase found to rename.")
        elif filter_prep:
            logger.info("Executing filter logic (rename is false, filter is true)...")
            cleaned_df = self._filter_preparation_phase(cleaned_df)
        else:
            logger.info("Keeping 'Preparation' phase (rename and filter are both disabled).")

        # --- Conditional Tare Correction ---
        # ONLY apply tare correction if balance measures reactor weight
        if balance_mode == 'reactor_weight':
            if self.cleaning_config.get('balance_tare_correction', True):
                logger.info("Applying balance tare correction (mode: reactor_weight).")
                cleaned_df = self._balance_tare_correction(cleaned_df)
            else:
                logger.info("Balance tare correction skipped (disabled in config).")
        else:
            logger.info(f"Balance tare correction skipped (mode: {balance_mode}).")
        # --- End Tare Correction ---

        # Apply median filter (still useful regardless of balance mode)
        if self.cleaning_config.get('median_filter', True):
            cleaned_df = self._rolling_median_filter(cleaned_df)
        else:
            logger.debug("Median filter step skipped (disabled in config).")

        # Calculate standardized process time
        if self.cleaning_config.get('calculate_process_time', True):
            cleaned_df = self._calculate_process_time(cleaned_df)

        logger.info("Data cleaning process finished.")
        return cleaned_df

    # Keep _filter_preparation_phase as is
    def _filter_preparation_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (no changes needed here) ...
        if 'process_phase' in df.columns:
            original_rows = len(df)
            phase_to_filter = 'Preparation'
            mask = df['process_phase'].astype(str).str.strip() != phase_to_filter
            filtered_df = df[mask].copy()
            rows_removed = original_rows - len(filtered_df)
            if rows_removed > 0:
                logger.info(f"Filtered out {rows_removed} rows from '{phase_to_filter}' phase.")
            return filtered_df
        else:
            logger.warning("No 'process_phase' column found, cannot filter preparation phase.")
            return df

    def _balance_tare_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrects raw balance readings. Sets values before the first significant
        reading (post-tare) to zero. Does *not* subtract the tare offset here.
        """
        result_df = df.copy()
        logger.info(f"Applying balance tare correction to columns: {self.balance_cols} (threshold={self.tare_threshold}g).")
        for col_name in self.balance_cols:
            if col_name not in result_df.columns:
                logger.warning(f"Balance column '{col_name}' not found for tare correction.")
                continue
            if not pd.api.types.is_numeric_dtype(result_df[col_name]):
                 logger.warning(f"Balance column '{col_name}' is not numeric, skipping tare correction.")
                 continue

            balance_values = result_df[col_name].copy()
            # Find the first index where balance is likely stable post-tare
            first_valid_index = (balance_values > self.tare_threshold).idxmax()

            if pd.isna(first_valid_index) or first_valid_index == balance_values.index[0] and balance_values.iloc[0] <= self.tare_threshold:
                # Handle case where no value is above threshold or the first value is already above (no clear tare found)
                logger.warning(f"No clear tare point detected for '{col_name}' above threshold {self.tare_threshold}. Correction not applied, keeping original values.")
                continue # Don't modify the column if no tare detected

            # Set all values *before* this index to zero.
            result_df.loc[result_df.index < first_valid_index, col_name] = 0.0
            logger.info(f"Applied tare correction for '{col_name}': Set values before index {first_valid_index} to 0.")

        return result_df

    def _rolling_median_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies rolling median filter to specified raw sensor columns."""
        result_df = df.copy()
        logger.info(f"Applying rolling median filter to columns: {self.median_filter_cols} (window={self.median_window}, threshold={self.median_threshold} MAD units)")

        for col_name in self.median_filter_cols:
            if col_name not in result_df.columns:
                logger.warning(f"Column '{col_name}' specified for median filter not found.")
                continue
            if not pd.api.types.is_numeric_dtype(result_df[col_name]):
                 logger.warning(f"Column '{col_name}' is not numeric, skipping median filter.")
                 continue

            try:
                median_values = result_df[col_name].rolling(window=self.median_window, center=True, min_periods=1).median()
                diff = np.abs(result_df[col_name] - median_values)
                mad = diff.rolling(window=self.median_window, center=True, min_periods=1).median()
                mad = mad * 1.4826
                mad.replace(0, np.nan, inplace=True) # Avoid division by zero MAD

                mask = (diff > self.median_threshold * mad) & mad.notna()
                spikes_found = mask.sum()

                if spikes_found > 0:
                    result_df.loc[mask, col_name] = median_values[mask]
                    logger.info(f"Replaced {spikes_found} spikes in '{col_name}' using rolling median filter.")
                else:
                    logger.info(f"No significant spikes detected in '{col_name}'.")
            except Exception as e:
                logger.error(f"Error during median filtering for column {col_name}: {e}")

        return result_df

    def _calculate_process_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates 'process_time' in hours, relative to the first row of the cleaned data."""
        result_df = df.copy()
        # Use batch_time_sec as the source - ensure this mapping is correct in default_config
        time_col_sec = 'batch_time_sec'

        if time_col_sec in result_df.columns:
            time_numeric = pd.to_numeric(result_df[time_col_sec], errors='coerce')
            if time_numeric.notna().any():
                logger.info(f"Calculating 'process_time' (hours) from '{time_col_sec}'.")
                # Find the first non-NaN time value to use as the reference start time
                first_valid_time_sec = time_numeric.dropna().iloc[0]
                result_df['process_time'] = (time_numeric - first_valid_time_sec) / 3600.0
                logger.info(
                    f"Process time calculated relative to first timestamp's {time_col_sec} = {first_valid_time_sec:.2f} sec.")
            else:
                logger.warning(
                    f"Column '{time_col_sec}' contains only NaNs or non-numeric data. Cannot calculate process time.")
                result_df['process_time'] = np.nan
        else:
            logger.warning(f"Column '{time_col_sec}' not found. Cannot calculate process time.")
            result_df['process_time'] = np.nan

        return result_df
