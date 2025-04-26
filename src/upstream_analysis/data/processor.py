# src/upstream_analysis/data/processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FermentationDataProcessor:
    """Processes cleaned/aligned data. Calculates feed, volume, etc."""

    def __init__(self, config: dict):
        self.config = config
        # Define expected structure keys for clarity
        self.processing_params_key = 'processing_parameters'
        self.feed_def_key = 'feed_definitions'
        self.medium_def_key = 'medium_definitions'
        self.base_titrant_key = 'base_titrant'
        self.reactor_setup_key = 'reactor_setup'

    # --- Helper to get nested values safely ---
    def _get_nested_param(self, keys: List[str], expected_type=None, is_required=True, default=None):
        """Safely gets and validates nested dictionary values."""
        value = self.config
        path_str = ""
        for key in keys:
            path_str += f"['{key}']"
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    if is_required:
                        msg = f"Missing required configuration parameter at path: config{path_str}"
                        logger.error(msg)
                        raise ValueError(msg)
                    else:
                        logger.debug(f"Optional config parameter not found at path: config{path_str}. Using default: {default}")
                        return default
            else: # Path is broken
                if is_required:
                    msg = f"Invalid config structure. Expected dict at path before {path_str}"
                    logger.error(msg)
                    raise TypeError(msg)
                else:
                    return default

        # Type check if required and type provided
        if is_required and expected_type is not None and not isinstance(value, expected_type):
             # Allow int to satisfy float requirement
             if expected_type is float and isinstance(value, int):
                 value = float(value)
             else:
                msg = f"Config parameter at path config{path_str} has wrong type. Expected {expected_type}, got {type(value)}"
                logger.error(msg)
                raise TypeError(msg)

        # Specific check for numeric > 0 if needed (e.g., density)
        if expected_type in [float, int] and isinstance(value, (int, float)) and value <= 0:
             # Check if this parameter *needs* to be positive
             if any(p in keys for p in ['density_g_ml', 'molarity_mol_l', 'start_volume_ml', 'molar_volume_air_l_mol']):
                 msg = f"Numeric config parameter at path config{path_str} must be positive, got {value}"
                 logger.error(msg)
                 raise ValueError(msg)

        return value

    def _calculate_feed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates cumulative feed added based on balance changes."""
        logger.info("Attempting to calculate feed...")
        df_out = df.copy()
        df_out['feed_g'] = np.nan
        df_out['feed_ml'] = np.nan

        balance_col = 'balance'
        phase_col = 'process_phase'

        if balance_col not in df_out.columns or phase_col not in df_out.columns:
            logger.warning(f"Processor: Missing '{balance_col}' or '{phase_col}'. Cannot calculate feed.")
            return df_out

        try:
            # --- Corrected Config Access ---
            feed_density = self._get_nested_param(
                [self.processing_params_key, self.feed_def_key, 'feed_1', 'density_g_ml'],
                expected_type=float, is_required=True
            )
            # --- End Corrected Access ---
        except (ValueError, TypeError) as e:
            logger.error(f"Cannot calculate feed: {e}")
            return df_out

        df_out[balance_col] = pd.to_numeric(df_out[balance_col], errors='coerce')
        if df_out[balance_col].isnull().all():
            logger.warning("Processor: Balance column empty/NaNs. Cannot calculate feed.")
            return df_out

        feed_start_index = None
        balance_at_feed_start = np.nan
        phases_before_feed = ['Preparation', 'Batch']
        feeding_phases_data = df_out[~df_out[phase_col].isin(phases_before_feed)]

        if not feeding_phases_data.empty:
            feed_start_index = feeding_phases_data.index[0]
            valid_balances = df_out.loc[feed_start_index:, balance_col].dropna()
            if not valid_balances.empty:
                balance_at_feed_start = valid_balances.iloc[0]
                logger.info(f"Processor: Feed start index {feed_start_index}. Tare balance {balance_at_feed_start:.2f}g")
            else:
                logger.error("Processor: Cannot find valid balance reading at/after feed start.")
                feed_start_index = None
        else:
            logger.warning("Processor: No feeding phases detected. Cannot calculate feed.")

        if feed_start_index is not None and pd.notna(balance_at_feed_start):
            df_out['feed_g'] = balance_at_feed_start - df_out[balance_col]
            df_out.loc[df_out.index < feed_start_index, 'feed_g'] = 0.0

            # --- ADD DEBUG ---
            logger.debug(f"Feed_g calculated BEFORE clip/cummax (Tail):\n{df_out['feed_g'].tail()}")
            # --- END DEBUG ---

            feed_g_series = df_out.loc[df_out.index >= feed_start_index, 'feed_g'].copy()
            feed_g_series = feed_g_series.clip(lower=0)

            # --- ADD DEBUG ---
            logger.debug(f"Feed_g series AFTER clip (Tail):\n{feed_g_series.tail()}")
            # --- END DEBUG ---

            # Cummax handling (forward fill NaN, apply cummax, restore NaN) - robust way
            nan_mask = feed_g_series.isna()
            feed_g_series.ffill(inplace=True)
            feed_g_series.fillna(0, inplace=True) # Fill remaining NaNs at start if any
            feed_g_series_cummax = feed_g_series.cummax()
            feed_g_series_cummax[nan_mask] = np.nan # Restore NaNs
            df_out.loc[df_out.index >= feed_start_index, 'feed_g'] = feed_g_series_cummax

            df_out['feed_ml'] = df_out['feed_g'] / feed_density
            logger.info(f"Processor: Calculated feed_g and feed_ml (density {feed_density} g/mL).")
        else:
            logger.warning("Processor: Feed calculation skipped or failed.")

        return df_out

    # In class FermentationDataProcessor (processor.py)
    # In class FermentationDataProcessor (processor.py)
    def _calculate_base_flux_and_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates cumulative base volume based on phases and total added."""
        logger.info("Attempting to calculate base volume added...")
        df_out = df.copy()
        # Initialize column FIRST
        df_out['base_volume_added_ml'] = 0.0

        # --- Get Parameters & Validate ---
        try:
            total_base_g = self._get_nested_param(
                [self.processing_params_key, self.base_titrant_key, 'total_added_g'],
                expected_type=float, is_required=True
            )
            base_density_g_ml = self._get_nested_param(
                [self.processing_params_key, self.base_titrant_key, 'density_g_ml'],
                expected_type=float, is_required=True
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Cannot calculate base volume: {e}")
            return df_out  # Return df with initialized column

        duration_col = 'base_pump_duration'
        phase_col = 'process_phase'

        if duration_col not in df_out.columns or phase_col not in df_out.columns:
            logger.warning(f"Processor: Missing '{duration_col}' or '{phase_col}'. Cannot calculate base volume.")
            return df_out

        duration_numeric = pd.to_numeric(df_out[duration_col], errors='coerce')
        if not pd.api.types.is_numeric_dtype(duration_numeric) or duration_numeric.isnull().all():
            logger.warning(
                f"Processor: Base duration '{duration_col}' not valid numeric or all NaN. Cannot calculate base volume.")
            return df_out

        total_base_ml = total_base_g / base_density_g_ml

        # --- Identify Period Based on Phases ---
        base_addition_phases = ['Batch', 'Fedbatch', 'induced Fedbatch']
        base_period_mask = df_out[phase_col].isin(base_addition_phases)
        if not base_period_mask.any():
            logger.warning(f"No data in base addition phases {base_addition_phases}. Base volume 0.")
            return df_out

        period_durations = duration_numeric[base_period_mask]
        if period_durations.dropna().empty:
            logger.error("No valid base duration values found within the specified phases.")
            return df_out

        first_phase_index = period_durations.index[0]
        last_phase_index = period_durations.index[-1]

        # --- Get Boundary Durations Robustly ---
        duration_start_of_period = period_durations.loc[first_phase_index]
        if pd.isna(duration_start_of_period):
            first_valid_duration = period_durations.dropna()
            if first_valid_duration.empty:
                logger.error("No valid base duration start value found within the period.")
                return df_out
            duration_start_of_period = first_valid_duration.iloc[0]
            logger.warning(
                f"NaN base duration at start index {first_phase_index}. Using first valid: {duration_start_of_period:.2f}s")

        duration_end_of_period = period_durations.dropna().iloc[-1]

        total_duration_s = duration_end_of_period - duration_start_of_period

        if total_duration_s <= 0:
            logger.warning(
                f"Total base pump duration during relevant phases ({total_duration_s}s) not positive. Base volume added set to 0.")
            return df_out

        logger.info(f"Base addition period indices: {first_phase_index} to {last_phase_index}.")
        logger.info(
            f"Duration range within period: {duration_start_of_period:.2f}s to {duration_end_of_period:.2f}s. Total duration: {total_duration_s:.2f}s")
        logger.info(f"Total base added (config): {total_base_g:.2f} g = {total_base_ml:.2f} mL")

        # --- Calculate Cumulative Base Volume Added ---
        # Use the original numeric duration, fill NaNs robustly
        duration_filled = duration_numeric.ffill().fillna(0)

        cumulative_duration_since_period_start = (duration_filled - duration_start_of_period).clip(lower=0)
        volume_added = cumulative_duration_since_period_start * (total_base_ml / total_duration_s)

        # --- Assign and Cap ---
        # Initialize first, then apply calculations for the period
        df_out['base_volume_added_ml'] = 0.0
        # Use .loc with the mask for assignment within the period
        df_out.loc[base_period_mask, 'base_volume_added_ml'] = volume_added[base_period_mask]

        # Set volume AFTER the period ends to the maximum total added volume
        df_out.loc[df_out.index > last_phase_index, 'base_volume_added_ml'] = total_base_ml

        # Ensure values before start are 0 (belt-and-suspenders)
        df_out.loc[df_out.index < first_phase_index, 'base_volume_added_ml'] = 0.0

        # Final clip only (NO fillna(0.0) here)
        df_out['base_volume_added_ml'] = df_out['base_volume_added_ml'].clip(upper=total_base_ml)

        logger.info("Calculated 'base_volume_added_ml' column.")
        logger.debug(f"Tail of base_volume_added_ml after calculation:\n{df_out['base_volume_added_ml'].tail()}")
        return df_out

    def _calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates reactor volume, incorporating feed, base, and samples."""
        logger.info("Calculating reactor volume...")
        df_out = df.copy()

        # --- Get Start Volume ---
        try:
            start_volume = self._get_nested_param(
                [self.processing_params_key, self.reactor_setup_key, 'initial_volume_ml'],
                expected_type=float, is_required=True
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Processor: Cannot calculate volume: {e}")
            df_out['volume_ml'] = np.nan
            df_out['cumulative_sample_volume_ml'] = np.nan
            df_out['volume_corrected_ml'] = np.nan
            return df_out

        # --- Get Input Columns & Force Numeric, Fill NaNs ---
        feed_vol = pd.to_numeric(df_out.get('feed_ml', 0.0), errors='coerce').fillna(0.0)
        base_vol = pd.to_numeric(df_out.get('base_volume_added_ml', 0.0), errors='coerce').fillna(0.0)

        # --- Calculate Base Volume (start + feed + base) ---
        # Explicitly ensure components are floats before summing
        try:
            volume_calc = float(start_volume) + feed_vol.astype(float) + base_vol.astype(float)
            df_out['volume_ml'] = volume_calc
        except Exception as calc_error:
            logger.error(f"Error during volume_ml calculation: {calc_error}")
            df_out['volume_ml'] = np.nan  # Assign NaN if calculation fails

        logger.info(f"Calculated base 'volume_ml' (start={start_volume} + feed + base). Check tail:")
        log_df_tail = pd.DataFrame({
            'feed_ml_used': feed_vol.tail(),
            'base_vol_used': base_vol.tail(),
            'volume_ml_calc': df_out['volume_ml'].tail()  # Log directly from df
        })
        logger.info(f"\n{log_df_tail}")

        # --- Apply Sample Volume Correction ---
        # Check if volume_ml calculation failed
        if df_out['volume_ml'].isnull().all():
            logger.warning("Base volume calculation failed, cannot calculate corrected volume.")
            df_out['volume_corrected_ml'] = np.nan
            df_out['cumulative_sample_volume_ml'] = 0.0
            return df_out  # Return early

        if 'sample_volume_ml' in df_out.columns and 'is_sample' in df_out.columns:
            sample_vol_numeric = pd.to_numeric(df_out['sample_volume_ml'], errors='coerce')
            if sample_vol_numeric.notna().any():
                logger.info("Applying sample volume correction.")
                sample_vol_at_event = pd.Series(0.0, index=df_out.index)
                is_sample_mask = df_out['is_sample'].fillna(False).astype(bool)
                sample_vol_at_event.loc[is_sample_mask] = sample_vol_numeric[is_sample_mask].fillna(0.0)
                df_out['cumulative_sample_volume_ml'] = sample_vol_at_event.cumsum()
                # Calculate corrected volume using the just calculated volume_ml
                df_out['volume_corrected_ml'] = df_out['volume_ml'] - df_out['cumulative_sample_volume_ml']
                logger.info("Calculated 'volume_corrected_ml' accounting for sample removal.")
            else:
                logger.info("No valid 'sample_volume_ml' data. 'volume_corrected_ml' equals base 'volume_ml'.")
                df_out['volume_corrected_ml'] = df_out['volume_ml'].copy()
                df_out['cumulative_sample_volume_ml'] = 0.0
        else:
            logger.info("No sample info. 'volume_corrected_ml' equals base 'volume_ml'.")
            df_out['volume_corrected_ml'] = df_out['volume_ml'].copy()
            df_out['cumulative_sample_volume_ml'] = 0.0

        logger.info(f"Final calculated volume_corrected_ml (Tail):\n{df_out['volume_corrected_ml'].tail()}")

        return df_out


    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main processing method, ensuring correct order of operations."""
        if df is None or df.empty:
             logger.error("Processor: Input DataFrame is None or empty.")
             return pd.DataFrame()

        processed_df = df.copy()
        logger.info("Starting data processing...")

        processed_df = self._calculate_feed(processed_df)
        processed_df = self._calculate_base_flux_and_volume(processed_df)
        processed_df = self._calculate_volume(processed_df)

        logger.info("Data processing finished.")
        return processed_df