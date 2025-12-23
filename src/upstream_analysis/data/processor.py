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
        """
        Calculates cumulative feed added, assuming balance measures feed source.
        1. Finds peak balance value around feed start.
        2. Creates a modified balance series (peak extended backward).
        3. Applies median filter to the modified balance series.
        4. Calculates feed based on the filtered, modified balance.
        5. Overwrites the original 'balance' column with the filtered, modified version.
        """
        logger.info("Attempting to calculate feed (adjusting balance first)...")
        df_out = df.copy()
        df_out['feed_g'] = 0.0
        df_out['feed_ml'] = 0.0

        balance_col = 'balance'
        phase_col = 'process_phase'
        time_col = 'process_time'

        # --- Validate Inputs & Config ---
        if not all(col in df_out.columns for col in [balance_col, phase_col, time_col]):
            logger.warning(f"Processor: Missing required columns. Cannot calculate feed.")
            return df_out

        try:
            feed_density = self._get_nested_param(
                [self.processing_params_key, self.feed_def_key, 'feed_1', 'density_g_ml'],
                expected_type=float, is_required=True
            )
            # Get median filter settings
            cleaning_config = self.config.get('cleaning', {})
            apply_filter = cleaning_config.get('median_filter', True)
            median_window = cleaning_config.get('median_filter_window', 10)
            median_threshold = cleaning_config.get('median_filter_threshold', 10)

        except (ValueError, TypeError) as e:
            logger.error(f"Cannot calculate feed due to config error: {e}")
            return df_out

        balance_numeric = pd.to_numeric(df_out[balance_col], errors='coerce')
        if balance_numeric.isnull().all():
            logger.warning("Processor: Balance column empty/all NaN. Cannot calculate feed.")
            return df_out
        balance_nan_mask = balance_numeric.isna()
        balance_filled = balance_numeric.ffill()
        first_valid_balance_idx = balance_filled.first_valid_index()
        if first_valid_balance_idx is not None:
            balance_filled.fillna(balance_filled[first_valid_balance_idx], inplace=True)
        else:
            balance_filled.fillna(0, inplace=True)

        # --- Identify Start of FIRST Feeding Phase ---
        feeding_phases = ['Fedbatch', 'induced Fedbatch']
        feeding_phases_mask = df_out[phase_col].isin(feeding_phases)

        if not feeding_phases_mask.any():
            logger.warning(f"No feeding phases {feeding_phases} found. Feed calculated as 0.")
            return df_out

        first_feed_index = df_out[feeding_phases_mask].index[0]
        logger.info(f"First feeding phase starts at index: {first_feed_index}")

        # --- Find Initial Feed Weight ---
        time_window_h = 4.0
        end_window_time = df_out.loc[first_feed_index, time_col] + time_window_h
        window_mask = (df_out[time_col] >= df_out.loc[first_feed_index, time_col]) & \
                      (df_out[time_col] <= end_window_time)
        balance_window = balance_filled[window_mask].dropna()

        if balance_window.empty:
            logger.error(f"Cannot find valid balance readings in window after feed start.")
            balance_at_tare_point = balance_filled.loc[first_feed_index]  # Fallback
            if pd.isna(balance_at_tare_point):
                logger.error("Balance at feed start is also NaN. Cannot determine tare point.")
                return df_out
            tare_index_approx = first_feed_index
            logger.warning(f"Using balance at feed start {balance_at_tare_point:.2f}g as fallback tare.")
        else:
            balance_at_tare_point = balance_window.max()
            tare_index_approx = balance_window.idxmax()
            logger.info(
                f"Using peak balance {balance_at_tare_point:.2f}g (near index {tare_index_approx}) as initial feed weight.")

        # --- Create Modified Balance Series ---
        modified_balance = balance_filled.copy()
        modified_balance.loc[modified_balance.index < tare_index_approx] = balance_at_tare_point
        modified_balance = modified_balance.ffill().fillna(balance_at_tare_point)  # Ensure full series

        # --- Apply Median Filter to Modified Balance ---
        if apply_filter:
            logger.info(
                f"Applying median filter (window={median_window}, threshold={median_threshold}) to modified balance.")
            median_values = modified_balance.rolling(window=median_window, center=True, min_periods=1).median()
            diff = np.abs(modified_balance - median_values)

            # --- *** CORRECTED MAD CALCULATION *** ---
            # Use numpy's median function for the rolling apply with raw=True
            mad = (diff.rolling(window=median_window, center=True, min_periods=1)
                   .apply(np.median, raw=True)) * 1.4826  # Use np.median
            # --- *** END CORRECTION *** ---

            mad.replace(0, np.nan, inplace=True)
            mask = (diff > median_threshold * mad) & mad.notna()
            spikes_found = mask.sum()
            if spikes_found > 0:
                filtered_modified_balance = modified_balance.copy()
                filtered_modified_balance[mask] = median_values[mask]
                logger.info(f"Replaced {spikes_found} spikes in modified balance data.")
            else:
                filtered_modified_balance = modified_balance
                logger.info("No significant spikes detected in modified balance data.")
        else:
            logger.info("Median filter disabled in config, skipping filter step.")
            filtered_modified_balance = modified_balance

        # --- Overwrite original balance column ---
        df_out[balance_col] = filtered_modified_balance
        logger.debug(f"Final filtered, modified balance assigned (Tail):\n{df_out[balance_col].tail()}")

        # --- Calculate Feed Added ---
        feed_diff = balance_at_tare_point - df_out[balance_col]
        feed_g_series = feed_diff.loc[df_out.index >= first_feed_index].copy()
        feed_g_series_clipped = feed_g_series.clip(lower=0)
        nan_mask_feed = feed_g_series_clipped.isna()
        feed_g_filled_for_cummax = feed_g_series_clipped.ffill().fillna(0)
        feed_g_cummax = feed_g_filled_for_cummax.cummax()
        feed_g_cummax[nan_mask_feed] = np.nan
        df_out.loc[df_out.index >= first_feed_index, 'feed_g'] = feed_g_cummax
        df_out.loc[balance_nan_mask, 'feed_g'] = np.nan  # Restore NaNs from original balance

        # Calculate feed_ml
        df_out['feed_ml'] = df_out['feed_g'] / feed_density
        logger.info(f"Processor: Calculated feed_g and feed_ml.")
        logger.debug(f"Final feed_g calculated (Tail):\n{df_out['feed_g'].tail()}")

        return df_out

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

    def _add_induction_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a binary 'is_induced' column based on process phases."""
        logger.info("Adding induction indicator column...")
        df_out = df.copy()
        df_out['is_induced'] = 0  # Initialize to 0 (not induced)

        phase_col = 'process_phase'
        if phase_col not in df_out.columns:
            logger.warning(f"Cannot add induction indicator: '{phase_col}' column not found.")
            return df_out

        # Get phase names from the *merged* config (defaults can be overridden)
        induced_phases = self.config.get('processing', {}).get('induced_phases', [])
        if not induced_phases:
            logger.warning("No 'induced_phases' defined in config. 'is_induced' column will remain 0.")
            return df_out

        logger.info(f"Identifying induction start based on phases: {induced_phases}")

        # Create mask for rows belonging to induced phases
        is_induced_phase_mask = df_out[phase_col].isin(induced_phases)

        # Find the first index where an induced phase occurs
        induction_start_index = is_induced_phase_mask.idxmax()

        # Check if any induced phase was actually found
        # idxmax returns the first index if *no* True is found, so check the value at that index
        if is_induced_phase_mask.any():  # Check if at least one True exists
            logger.info(
                f"Induction start detected at index {induction_start_index} (process_time: {df_out.loc[induction_start_index, 'process_time']:.2f}h)")
            # Set indicator to 1 from the start index onwards
            df_out.loc[induction_start_index:, 'is_induced'] = 1
        else:
            logger.info("No specified induction phases found in the data. 'is_induced' column remains 0.")

        return df_out

    def _calculate_protein_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates derived protein parameters including total protein, soluble fraction, etc.
        Always creates both soluble and IB columns, filling with zeros when missing from original data.
        """
        logger.info("Calculating protein-derived parameters...")
        df_out = df.copy()

        # Check what protein columns are available in the ORIGINAL data
        orig_has_soluble = 'protein_soluble_g_L' in df_out.columns
        orig_has_ib = 'IB_g_l' in df_out.columns
        orig_has_total = 'protein_g_l' in df_out.columns

        logger.info(f"Original protein data - Soluble: {orig_has_soluble}, IB: {orig_has_ib}, Total: {orig_has_total}")

        # Always create both protein columns, filling missing ones with zeros
        if not orig_has_soluble:
            df_out['protein_soluble_g_L'] = 0.0
            logger.info("Created soluble protein column (filled with zeros - not present in original data).")
        if not orig_has_ib:
            df_out['IB_g_l'] = 0.0
            logger.info("Created IB protein column (filled with zeros - not present in original data).")

        # Calculate total protein based on ORIGINAL data availability
        if not orig_has_total:
            if orig_has_soluble and orig_has_ib:
                # Both soluble and IB present in original data
                df_out['protein_total_calculated_g_l'] = df_out['protein_soluble_g_L'].fillna(0) + df_out['IB_g_l'].fillna(0)
                logger.info("Calculated total protein from soluble + IB fractions.")
            elif orig_has_soluble and not orig_has_ib:
                # Only soluble present in original data
                df_out['protein_total_calculated_g_l'] = df_out['protein_soluble_g_L']
                logger.info("Using soluble protein as total protein (no IB data available).")
            elif orig_has_ib and not orig_has_soluble:
                # Only IB present in original data
                df_out['protein_total_calculated_g_l'] = df_out['IB_g_l']
                logger.info("Using IB protein as total protein (no soluble data available).")
            else:
                # Neither soluble nor IB present - no protein data to calculate
                logger.info("No protein data available in original data.")

        # Calculate soluble fraction (%) if soluble protein is available in original data
        if orig_has_soluble:
            if orig_has_total:
                df_out['soluble_fraction_pct'] = (df_out['protein_soluble_g_L'] / df_out['protein_g_l'].replace(0, np.nan)) * 100
                logger.info("Calculated soluble fraction from measured total protein.")
            elif 'protein_total_calculated_g_l' in df_out.columns:
                df_out['soluble_fraction_pct'] = (df_out['protein_soluble_g_L'] / df_out['protein_total_calculated_g_l'].replace(0, np.nan)) * 100
                logger.info("Calculated soluble fraction from calculated total protein.")

        # Calculate IB fraction (%) if IB protein is available in original data
        if orig_has_ib:
            if orig_has_total:
                df_out['IB_fraction_pct'] = (df_out['IB_g_l'] / df_out['protein_g_l'].replace(0, np.nan)) * 100
                logger.info("Calculated IB fraction from measured total protein.")
            elif 'protein_total_calculated_g_l' in df_out.columns:
                df_out['IB_fraction_pct'] = (df_out['IB_g_l'] / df_out['protein_total_calculated_g_l'].replace(0, np.nan)) * 100
                logger.info("Calculated IB fraction from calculated total protein.")

        # Calculate protein yields if biomass is available
        if 'biomass_g_l' in df_out.columns:
            if orig_has_soluble:
                df_out['protein_soluble_yield_g_g'] = df_out['protein_soluble_g_L'] / df_out['biomass_g_l'].replace(0, np.nan)
                logger.info("Calculated soluble protein yield (g protein / g biomass).")
            if orig_has_ib:
                df_out['protein_IB_yield_g_g'] = df_out['IB_g_l'] / df_out['biomass_g_l'].replace(0, np.nan)
                logger.info("Calculated IB protein yield (g protein / g biomass).")
            if orig_has_total:
                df_out['protein_total_yield_g_g'] = df_out['protein_g_l'] / df_out['biomass_g_l'].replace(0, np.nan)
                logger.info("Calculated total protein yield (g protein / g biomass).")
            elif 'protein_total_calculated_g_l' in df_out.columns:
                df_out['protein_total_yield_g_g'] = df_out['protein_total_calculated_g_l'] / df_out['biomass_g_l'].replace(0, np.nan)
                logger.info("Calculated total protein yield from calculated total protein.")

        return df_out

    # MODIFY THE process_data METHOD:
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
        processed_df = self._add_induction_indicator(processed_df)  # <-- ADD THIS CALL
        processed_df = self._calculate_protein_parameters(processed_df)  # <-- ADD THIS CALL

        logger.info("Data processing finished.")
        return processed_df
