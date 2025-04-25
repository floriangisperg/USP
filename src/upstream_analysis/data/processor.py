# src/upstream_analysis/data/processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FermentationDataProcessor:
    """Processes cleaned/aligned data. Calculates feed, volume, etc."""

    def __init__(self, config: dict):
        # ... (init remains the same) ...
        self.config = config
        self.processing_config = config.get('processing', {})
        self.feed_params = self.processing_config.get('feed_params', {})

    def _validate_config_parameter(self, param_name: str, section: dict) -> Optional[float]:
        # ... (remains the same) ...
        value = section.get(param_name)
        if value is None:
            logger.error(f"Missing required configuration parameter: {param_name} in section {section}")
            raise ValueError(f"Missing required configuration parameter: {param_name}")
        if not isinstance(value, (int, float)):
            logger.error(f"Configuration parameter '{param_name}' must be numeric, got {type(value)}")
            raise TypeError(f"Configuration parameter '{param_name}' must be numeric")
        return float(value)


    def _calculate_feed(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (This method should be correct from the previous step) ...
        if 'balance' not in df.columns:
            logger.warning("Processor: 'balance' column not found, cannot calculate feed.")
            df['feed_g'] = np.nan
            df['feed_ml'] = np.nan
            return df
        if 'process_phase' not in df.columns:
             logger.warning("Processor: 'process_phase' column not found, cannot reliably identify feed start. Skipping feed calculation.")
             df['feed_g'] = np.nan
             df['feed_ml'] = np.nan
             return df

        try:
            feed_density = self._validate_config_parameter('feed_density', self.feed_params)
        except (ValueError, TypeError) as e:
            logger.error(f"Cannot calculate feed: {e}")
            df['feed_g'] = np.nan
            df['feed_ml'] = np.nan
            return df

        df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
        if df['balance'].isnull().all():
             logger.warning("Processor: Balance column contains only NaNs. Cannot calculate feed.")
             df['feed_g'] = np.nan
             df['feed_ml'] = np.nan
             return df

        feed_start_index = None
        balance_at_feed_start = np.nan
        phases_before_feed = ['Preparation', 'Batch']
        feeding_phases_data = df[~df['process_phase'].isin(phases_before_feed)]

        if not feeding_phases_data.empty:
            feed_start_index = feeding_phases_data.index[0]
            balance_at_feed_start = df.loc[feed_start_index, 'balance']
            if pd.isna(balance_at_feed_start):
                first_valid_balance_after_start = df.loc[feed_start_index:, 'balance'].dropna()
                if not first_valid_balance_after_start.empty:
                    balance_at_feed_start = first_valid_balance_after_start.iloc[0]
                    logger.warning(f"Processor: Balance NaN at detected feed start index {feed_start_index}. Using first valid balance after: {balance_at_feed_start:.2f}g as tare.")
                else:
                    logger.error("Processor: Cannot determine tare value: No valid balance readings found at or after feed start.")
                    feed_start_index = None
                    balance_at_feed_start = np.nan
        else:
            logger.warning("Processor: No feeding phases detected. Cannot calculate feed.")

        if feed_start_index is not None and pd.notna(balance_at_feed_start):
            logger.info(f"Processor: Feed start identified at index {feed_start_index}. Tare balance value = {balance_at_feed_start:.2f}g")
            df['feed_g'] = balance_at_feed_start - df['balance']
            df.loc[df.index < feed_start_index, 'feed_g'] = 0.0
            df.loc[df.index >= feed_start_index, 'feed_g'] = df.loc[df.index >= feed_start_index, 'feed_g'].clip(lower=0).cummax()
            df['feed_ml'] = df['feed_g'] / feed_density
            logger.info(f"Processor: Calculated feed_g and feed_ml using density {feed_density} g/mL.")
        else:
            df['feed_g'] = np.nan
            df['feed_ml'] = np.nan
            logger.warning("Processor: Feed calculation skipped due to issues identifying feed start or tare value.")

        return df


    def _calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates reactor volume, applying sample volume correction correctly
        based on the 'is_sample' flag determined during alignment.
        """
        # Ensure necessary columns exist and get start volume
        if 'feed_ml' not in df.columns or df['feed_ml'].isnull().all():
             logger.warning("Processor: Cannot calculate volume: 'feed_ml' column missing or empty.")
             df['volume_ml'] = np.nan
             df['volume_corrected_ml'] = np.nan
             df['cumulative_sample_volume_ml'] = np.nan # Ensure column exists
             return df
        try:
            start_volume = self._validate_config_parameter('start_volume_ml', self.feed_params)
        except (ValueError, TypeError) as e:
            logger.error(f"Processor: Cannot calculate volume: {e}")
            df['volume_ml'] = np.nan
            df['volume_corrected_ml'] = np.nan
            df['cumulative_sample_volume_ml'] = np.nan
            return df

        # Calculate base volume (without sample correction)
        df['volume_ml'] = start_volume + df['feed_ml']
        logger.info(f"Processor: Calculated base 'volume_ml' starting from {start_volume} mL.")

        # --- Corrected Sample Volume Correction ---
        if 'sample_volume_ml' in df.columns and 'is_sample' in df.columns:
            if df['sample_volume_ml'].notna().any():
                logger.info("Applying sample volume correction.")
                # 1. Ensure the raw sample volume column is numeric
                sample_vol_numeric = pd.to_numeric(df['sample_volume_ml'], errors='coerce')

                # 2. Create a *new* temporary column that is ZERO everywhere *except*
                #    where is_sample is True. At those points, use the actual sample volume.
                sample_vol_at_event = pd.Series(0.0, index=df.index) # Default to 0 removal
                sample_vol_at_event.loc[df['is_sample']] = sample_vol_numeric[df['is_sample']]
                # Handle potential NaNs that might exist even on sample rows
                sample_vol_at_event.fillna(0.0, inplace=True)

                # 3. Calculate the cumulative sum on *this* temporary column
                # This now correctly sums only the volume removed at the *single* event time point.
                df['cumulative_sample_volume_ml'] = sample_vol_at_event.cumsum()

                # 4. Calculate the corrected volume
                df['volume_corrected_ml'] = df['volume_ml'] - df['cumulative_sample_volume_ml']
                logger.info("Calculated 'volume_corrected_ml' accounting for sample removal.")
            else:
                 logger.info("No valid 'sample_volume_ml' data found. Using uncorrected volume.")
                 df['volume_corrected_ml'] = df['volume_ml']
                 df['cumulative_sample_volume_ml'] = 0.0
        else:
            logger.info("No 'sample_volume_ml' or 'is_sample' column found. Using uncorrected volume ('volume_corrected_ml' = 'volume_ml').")
            df['volume_corrected_ml'] = df['volume_ml']
            # Ensure columns exist even if unused
            if 'sample_volume_ml' not in df.columns: df['sample_volume_ml'] = 0.0
            if 'cumulative_sample_volume_ml' not in df.columns: df['cumulative_sample_volume_ml'] = 0.0

        return df

    # --- process_data method remains the same ---
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main processing method."""
        if df is None or df.empty:
             logger.error("Processor: Input DataFrame is None or empty.")
             return pd.DataFrame()

        processed_df = df.copy()
        logger.info("Starting data processing...")

        processed_df = self._calculate_feed(processed_df)
        processed_df = self._calculate_volume(processed_df) # Now uses corrected logic

        logger.info("Data processing finished.")
        return processed_df