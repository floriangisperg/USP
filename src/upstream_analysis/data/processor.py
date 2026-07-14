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
        # Support both old and new config structures
        # Old: config['processing_parameters']
        # New: config['processing']['processing_parameters']
        self.processing_params_key = 'processing_parameters'
        self.feed_def_key = 'feed_definitions'
        self.medium_def_key = 'medium_definitions'
        self.base_titrant_key = 'base_titrant'
        self.reactor_setup_key = 'reactor_setup'

        # Determine config structure
        self.processing_config = self._get_processing_config()

    def _get_processing_config(self) -> dict:
        """Get the processing config section, supporting both old and new structures."""
        # Try new structure first: config['processing']['processing_parameters']
        if 'processing' in self.config and isinstance(self.config['processing'], dict):
            if 'processing_parameters' in self.config['processing']:
                return self.config['processing']['processing_parameters']
            # Some configs might have params directly under 'processing'
            return self.config['processing']

        # Fall back to old structure: config['processing_parameters']
        if self.processing_params_key in self.config:
            return self.config[self.processing_params_key]

        # Return empty dict if neither found
        logger.warning("Could not find processing_parameters in config. Using empty dict.")
        return {}

    # --- Helper to get nested values safely ---
    def _get_nested_param(self, keys: List[str], expected_type=None, is_required=True, default=None):
        """Safely gets and validates nested dictionary values from processing config."""
        value = self.processing_config
        path_str = "processing_config"
        for key in keys:
            path_str += f"['{key}']"
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    if is_required:
                        msg = f"Missing required configuration parameter at path: {path_str}"
                        logger.error(msg)
                        raise ValueError(msg)
                    else:
                        logger.debug(f"Optional config parameter not found at path: {path_str}. Using default: {default}")
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
        df_out['Feed [g]'] = 0.0
        df_out['Feed [mL]'] = 0.0

        balance_col = 'Balance [g]'
        phase_col = 'Process phase [-]'
        time_col = 'Process time [h]'

        # --- Validate Inputs & Config ---
        if not all(col in df_out.columns for col in [balance_col, phase_col, time_col]):
            logger.warning(f"Processor: Missing required columns. Cannot calculate feed.")
            return df_out

        try:
            feed_density = self._get_nested_param(
                [self.feed_def_key, 'feed_1', 'density_g_ml'],
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
        df_out.loc[df_out.index >= first_feed_index, 'Feed [g]'] = feed_g_cummax
        df_out.loc[balance_nan_mask, 'Feed [g]'] = np.nan  # Restore NaNs from original balance

        # Calculate Feed [mL]
        df_out['Feed [mL]'] = df_out['Feed [g]'] / feed_density
        logger.info(f"Processor: Calculated Feed [g] and Feed [mL].")
        logger.debug(f"Final Feed [g] calculated (Tail):\n{df_out['Feed [g]'].tail()}")

        return df_out

    # In class FermentationDataProcessor (processor.py)
    def _calculate_base_flux_and_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates cumulative base volume based on phases and total added."""
        logger.info("Attempting to calculate base volume added...")
        df_out = df.copy()
        # Initialize column FIRST
        df_out['Base volume added [mL]'] = 0.0

        # --- Get Parameters & Validate ---
        try:
            total_base_g = self._get_nested_param(
                [self.base_titrant_key, 'total_added_g'],
                expected_type=float, is_required=True
            )
            base_density_g_ml = self._get_nested_param(
                [self.base_titrant_key, 'density_g_ml'],
                expected_type=float, is_required=True
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Cannot calculate base volume: {e}")
            return df_out  # Return df with initialized column

        duration_col = 'Base pump duration [-]'
        phase_col = 'Process phase [-]'

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
        df_out['Base volume added [mL]'] = 0.0
        # Use .loc with the mask for assignment within the period
        df_out.loc[base_period_mask, 'Base volume added [mL]'] = volume_added[base_period_mask]

        # Set volume AFTER the period ends to the maximum total added volume
        df_out.loc[df_out.index > last_phase_index, 'Base volume added [mL]'] = total_base_ml

        # Ensure values before start are 0 (belt-and-suspenders)
        df_out.loc[df_out.index < first_phase_index, 'Base volume added [mL]'] = 0.0

        # Final clip only (NO fillna(0.0) here)
        df_out['Base volume added [mL]'] = df_out['Base volume added [mL]'].clip(upper=total_base_ml)

        logger.info("Calculated 'Base volume added [mL]' column.")
        logger.debug(f"Tail of Base volume added [mL] after calculation:\n{df_out['Base volume added [mL]'].tail()}")
        return df_out

    def _calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates reactor volume, incorporating feed, base, and samples."""
        logger.info("Calculating reactor volume...")
        df_out = df.copy()

        # --- Get Start Volume ---
        try:
            start_volume = self._get_nested_param(
                [self.reactor_setup_key, 'initial_volume_ml'],
                expected_type=float, is_required=True
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Processor: Cannot calculate volume: {e}")
            df_out['Volume [mL]'] = np.nan
            df_out['Cumulative sample volume [mL]'] = np.nan
            df_out['Volume corrected [mL]'] = np.nan
            return df_out

        # --- Get Input Columns & Force Numeric, Fill NaNs ---
        feed_vol = pd.to_numeric(df_out.get('Feed [mL]', 0.0), errors='coerce').fillna(0.0)
        base_vol = pd.to_numeric(df_out.get('Base volume added [mL]', 0.0), errors='coerce').fillna(0.0)

        # --- Calculate Base Volume (start + feed + base) ---
        # Explicitly ensure components are floats before summing
        try:
            volume_calc = float(start_volume) + feed_vol.astype(float) + base_vol.astype(float)
            df_out['Volume [mL]'] = volume_calc
        except Exception as calc_error:
            logger.error(f"Error during Volume [mL] calculation: {calc_error}")
            df_out['Volume [mL]'] = np.nan  # Assign NaN if calculation fails

        logger.info(f"Calculated base 'Volume [mL]' (start={start_volume} + feed + base). Check tail:")
        log_df_tail = pd.DataFrame({
            'Feed [mL]_used': feed_vol.tail(),
            'Base volume used': base_vol.tail(),
            'Volume [mL]_calc': df_out['Volume [mL]'].tail()  # Log directly from df
        })
        logger.info(f"\n{log_df_tail}")

        # --- Apply Sample Volume Correction ---
        # Check if Volume [mL] calculation failed
        if df_out['Volume [mL]'].isnull().all():
            logger.warning("Base volume calculation failed, cannot calculate corrected volume.")
            df_out['Volume corrected [mL]'] = np.nan
            df_out['Cumulative sample volume [mL]'] = 0.0
            return df_out  # Return early

        if 'Sample volume [mL]' in df_out.columns and 'is_sample' in df_out.columns:
            sample_vol_numeric = pd.to_numeric(df_out['Sample volume [mL]'], errors='coerce')
            if sample_vol_numeric.notna().any():
                logger.info("Applying sample volume correction.")
                sample_vol_at_event = pd.Series(0.0, index=df_out.index)
                is_sample_mask = df_out['is_sample'].fillna(False).astype(bool)
                sample_vol_at_event.loc[is_sample_mask] = sample_vol_numeric[is_sample_mask].fillna(0.0)
                df_out['Cumulative sample volume [mL]'] = sample_vol_at_event.cumsum()
                # Calculate corrected volume using the just calculated Volume [mL]
                df_out['Volume corrected [mL]'] = df_out['Volume [mL]'] - df_out['Cumulative sample volume [mL]']
                logger.info("Calculated 'Volume corrected [mL]' accounting for sample removal.")
            else:
                logger.info("No valid 'Sample volume [mL]' data. 'Volume corrected [mL]' equals base 'Volume [mL]'.")
                df_out['Volume corrected [mL]'] = df_out['Volume [mL]'].copy()
                df_out['Cumulative sample volume [mL]'] = 0.0
        else:
            logger.info("No sample info. 'Volume corrected [mL]' equals base 'Volume [mL]'.")
            df_out['Volume corrected [mL]'] = df_out['Volume [mL]'].copy()
            df_out['Cumulative sample volume [mL]'] = 0.0

        logger.info(f"Final calculated Volume corrected [mL] (Tail):\n{df_out['Volume corrected [mL]'].tail()}")

        return df_out

    def _add_induction_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a binary 'Is induced [-]' column based on process phases."""
        logger.info("Adding induction indicator column...")
        df_out = df.copy()
        df_out['Is induced [-]'] = 0  # Initialize to 0 (not induced)

        phase_col = 'Process phase [-]'
        if phase_col not in df_out.columns:
            logger.warning(f"Cannot add induction indicator: '{phase_col}' column not found.")
            return df_out

        # Get phase names from the *merged* config (defaults can be overridden)
        induced_phases = self.config.get('processing', {}).get('induced_phases', [])
        if not induced_phases:
            logger.warning("No 'induced_phases' defined in config. 'Is induced [-]' column will remain 0.")
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
                f"Induction start detected at index {induction_start_index} (Process time [h]: {df_out.loc[induction_start_index, 'Process time [h]']:.2f}h)")
            # Set indicator to 1 from the start index onwards
            df_out.loc[induction_start_index:, 'Is induced [-]'] = 1
        else:
            logger.info("No specified induction phases found in the data. 'Is induced [-]' column remains 0.")

        return df_out

    def _add_induction_timer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'Induction time [h]' column showing time elapsed since induction start."""
        logger.info("Adding induction timer column...")
        df_out = df.copy()

        # Check if induction indicator exists
        if 'Is induced [-]' not in df_out.columns:
            logger.warning("Cannot add induction timer: 'Is induced [-]' column not found. Run _add_induction_indicator first.")
            df_out['Induction time [h]'] = 0.0
            return df_out

        # Check if process time column exists
        time_col = 'Process time [h]'
        if time_col not in df_out.columns:
            logger.warning(f"Cannot add induction timer: '{time_col}' column not found.")
            df_out['Induction time [h]'] = 0.0
            return df_out

        # Find the induction start time (first row where Is induced [-] == 1)
        induced_mask = df_out['Is induced [-]'] == 1
        if not induced_mask.any():
            logger.info("No induction detected. 'Induction time [h]' set to 0.")
            df_out['Induction time [h]'] = 0.0
            return df_out

        induction_start_index = induced_mask.idxmax()
        induction_start_time = df_out.loc[induction_start_index, time_col]

        # Calculate induction timer
        # Before induction: 0
        # After induction start: Process time - induction start time
        df_out['Induction time [h]'] = 0.0
        df_out.loc[df_out.index >= induction_start_index, 'Induction time [h]'] = (
            df_out.loc[df_out.index >= induction_start_index, time_col] - induction_start_time
        )

        logger.info(
            f"Added 'Induction time [h]' column (induction started at {induction_start_time:.2f}h). "
            f"Max timer value: {df_out['Induction time [h]'].max():.2f}h"
        )

        return df_out

    def _calculate_protein_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates derived protein parameters including total protein, soluble fraction, etc.
        Always creates both soluble and IB columns, filling with zeros when missing from original data.
        """
        logger.info("Calculating protein-derived parameters...")
        df_out = df.copy()

        # Check what protein columns are available in the ORIGINAL data
        orig_has_soluble = 'Protein soluble [g/L]' in df_out.columns
        orig_has_ib = 'Protein IB [g/L]' in df_out.columns
        orig_has_total = 'Protein [g/L]' in df_out.columns

        logger.info(f"Original protein data - Soluble: {orig_has_soluble}, IB: {orig_has_ib}, Total: {orig_has_total}")

        # Always create both protein columns, filling missing ones with zeros
        if not orig_has_soluble:
            df_out['Protein soluble [g/L]'] = 0.0
            logger.info("Created soluble protein column (filled with zeros - not present in original data).")
        if not orig_has_ib:
            df_out['Protein IB [g/L]'] = 0.0
            logger.info("Created IB protein column (filled with zeros - not present in original data).")

        # Calculate total protein based on ORIGINAL data availability
        if not orig_has_total:
            if orig_has_soluble and orig_has_ib:
                # Both soluble and IB present in original data
                df_out['Protein total calculated [g/L]'] = df_out['Protein soluble [g/L]'].fillna(0) + df_out['Protein IB [g/L]'].fillna(0)
                logger.info("Calculated total protein from soluble + IB fractions.")
            elif orig_has_soluble and not orig_has_ib:
                # Only soluble present in original data
                df_out['Protein total calculated [g/L]'] = df_out['Protein soluble [g/L]']
                logger.info("Using soluble protein as total protein (no IB data available).")
            elif orig_has_ib and not orig_has_soluble:
                # Only IB present in original data
                df_out['Protein total calculated [g/L]'] = df_out['Protein IB [g/L]']
                logger.info("Using IB protein as total protein (no soluble data available).")
            else:
                # Neither soluble nor IB present - no protein data to calculate
                logger.info("No protein data available in original data.")

        # Calculate soluble fraction (%) if soluble protein is available in original data
        if orig_has_soluble:
            if orig_has_total:
                df_out['Soluble fraction [%]'] = (df_out['Protein soluble [g/L]'] / df_out['Protein [g/L]'].replace(0, np.nan)) * 100
                logger.info("Calculated soluble fraction from measured total protein.")
            elif 'Protein total calculated [g/L]' in df_out.columns:
                df_out['Soluble fraction [%]'] = (df_out['Protein soluble [g/L]'] / df_out['Protein total calculated [g/L]'].replace(0, np.nan)) * 100
                logger.info("Calculated soluble fraction from calculated total protein.")

        # Calculate IB fraction (%) if IB protein is available in original data
        if orig_has_ib:
            if orig_has_total:
                df_out['IB fraction [%]'] = (df_out['Protein IB [g/L]'] / df_out['Protein [g/L]'].replace(0, np.nan)) * 100
                logger.info("Calculated IB fraction from measured total protein.")
            elif 'Protein total calculated [g/L]' in df_out.columns:
                df_out['IB fraction [%]'] = (df_out['Protein IB [g/L]'] / df_out['Protein total calculated [g/L]'].replace(0, np.nan)) * 100
                logger.info("Calculated IB fraction from calculated total protein.")

        # Calculate protein yields if biomass is available
        if 'Biomass [g/L]' in df_out.columns:
            if orig_has_soluble:
                df_out['Protein soluble yield [g/g]'] = df_out['Protein soluble [g/L]'] / df_out['Biomass [g/L]'].replace(0, np.nan)
                logger.info("Calculated soluble protein yield (g protein / g biomass).")
            if orig_has_ib:
                df_out['Protein IB yield [g/g]'] = df_out['Protein IB [g/L]'] / df_out['Biomass [g/L]'].replace(0, np.nan)
                logger.info("Calculated IB protein yield (g protein / g biomass).")
            if orig_has_total:
                df_out['Protein total yield [g/g]'] = df_out['Protein [g/L]'] / df_out['Biomass [g/L]'].replace(0, np.nan)
                logger.info("Calculated total protein yield (g protein / g biomass).")
            elif 'Protein total calculated [g/L]' in df_out.columns:
                df_out['Protein total yield [g/g]'] = df_out['Protein total calculated [g/L]'] / df_out['Biomass [g/L]'].replace(0, np.nan)
                logger.info("Calculated total protein yield from calculated total protein.")

        return df_out

    def _calculate_inlet_gas_composition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates inlet gas composition (O2, CO2) based on air and O2 flow rates.
        Uses weighted average: y_in = (F_air * y_air + F_O2 * y_O2) / (F_air + F_O2)
        """
        logger.info("Calculating inlet gas composition...")
        df_out = df.copy()

        # --- Get gas config (support both old and new config structures) ---
        # New structure: processing.gas_params
        # Old structure: processing_parameters.gas_params
        gas_config = {}
        gas_config.update(self.config.get('processing', {}).get('gas_params', {}) or {})
        gas_config.update(self.config.get('processing_parameters', {}).get('gas_params', {}) or {})

        # --- Check if calculation is enabled ---
        # Default to True if not specified (for backward compatibility)
        calculate_enabled = gas_config.get('calculate_inlet_composition', True)
        if not calculate_enabled:
            logger.info("Inlet gas composition calculation disabled in config.")
            return df_out

        # --- Required columns ---
        air_flow_col = 'Air flow [L/min]'
        o2_flow_col = 'O2 flow [L/min]'

        # Check for standard column names first, then try alternatives
        if air_flow_col not in df_out.columns:
            # Try alternative names
            for alt in ['AirFlow [L/min]', 'Air flow [L/min]', 'AirFlow, l/min']:
                if alt in df_out.columns:
                    air_flow_col = alt
                    break
            else:
                logger.warning(f"Air flow column not found. Cannot calculate inlet gas composition.")
                return df_out

        if o2_flow_col not in df_out.columns:
            # Try alternative names
            for alt in ['Gas2 flow [L/min]', 'Gas2Flow [L/min]', 'Gas2Flow, l/min', 'O2 flow [L/min]']:
                if alt in df_out.columns:
                    o2_flow_col = alt
                    break
            else:
                logger.warning(f"O2 flow column not found. Cannot calculate inlet gas composition.")
                return df_out

        # --- Get gas parameters (support both old and new key names) ---
        try:
            # Try new key names first
            o2_in_air = gas_config.get('o2_in_air_percent')
            co2_in_air = gas_config.get('co2_in_air_percent')

            # Fall back to old key names
            if o2_in_air is None:
                o2_in_air = gas_config.get('inlet_o2_mol_pct', 21.0)
            if co2_in_air is None:
                co2_in_air = gas_config.get('inlet_co2_mol_pct', 0.04)
        except Exception as e:
            logger.error(f"Error reading gas parameters: {e}")
            return df_out

        # --- Get flows as numeric ---
        air_flow = pd.to_numeric(df_out[air_flow_col], errors='coerce')
        o2_flow = pd.to_numeric(df_out[o2_flow_col], errors='coerce')
        total_flow = air_flow + o2_flow

        # --- Calculate inlet O2 concentration (weighted average) ---
        # y_O2_in = (F_air * 0.21 + F_O2 * 1.0) / (F_air + F_O2)
        df_out['Inlet O2 [%]'] = (air_flow * o2_in_air + o2_flow * 100.0) / total_flow.replace(0, np.nan)

        # --- Calculate inlet CO2 concentration ---
        # y_CO2_in = (F_air * 0.0004 + F_O2 * 0.0) / (F_air + F_O2)
        df_out['Inlet CO2 [%]'] = (air_flow * co2_in_air + o2_flow * 0.0) / total_flow.replace(0, np.nan)

        logger.info("Calculated inlet gas composition (O2, CO2).")
        return df_out

    def _calculate_offgas_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates offgas rates: CER, OUR, RQ
        CER (CO2 Evolution Rate) = F_gas * (CO2_out / 100) / Volume
        OUR (O2 Uptake Rate) = F_gas * (O2_in - O2_out) / 100 / Volume
        RQ (Respiratory Quotient) = CER / OUR
        """
        logger.info("Calculating offgas rates (CER, OUR, RQ)...")
        df_out = df.copy()

        # --- Get gas config (support both old and new config structures) ---
        gas_config = {}
        gas_config.update(self.config.get('processing', {}).get('gas_params', {}) or {})
        gas_config.update(self.config.get('processing_parameters', {}).get('gas_params', {}) or {})

        # --- Check if calculations are enabled ---
        # Default to True if not specified (for backward compatibility)
        calculate_cer = gas_config.get('calculate_cer', True)
        calculate_our = gas_config.get('calculate_our', True)
        calculate_rq = gas_config.get('calculate_rq', True)

        if not any([calculate_cer, calculate_our, calculate_rq]):
            logger.info("Offgas rate calculations disabled in config.")
            return df_out

        # --- Required columns ---
        total_flow_col = 'Total flow [L/min]'
        co2_out_col = 'Offgas CO2 [%]'
        o2_out_col = 'Offgas O2 [%]'
        volume_col = 'Volume corrected [mL]'

        # Check for column alternatives
        if total_flow_col not in df_out.columns:
            for alt in ['TotalFlow [L/min]', 'TotalFlow, l/min']:
                if alt in df_out.columns:
                    total_flow_col = alt
                    break
            else:
                logger.warning(f"Total flow column not found. Cannot calculate offgas rates.")
                return df_out

        if volume_col not in df_out.columns:
            logger.warning(f"Volume column '{volume_col}' not found. Cannot calculate offgas rates.")
            return df_out

        if co2_out_col not in df_out.columns:
            logger.warning(f"CO2 outlet column not found. Cannot calculate CER.")
            calculate_cer = False

        if o2_out_col not in df_out.columns:
            logger.warning(f"O2 outlet column not found. Cannot calculate OUR.")
            calculate_our = False

        # --- Get numeric values ---
        f_gas = pd.to_numeric(df_out[total_flow_col], errors='coerce')  # L/min
        volume_l = pd.to_numeric(df_out[volume_col], errors='coerce') / 1000.0  # Convert mL to L

        molar_volume = gas_config.get('molar_volume_air_l_mol', 22.414)
        try:
            molar_volume = float(molar_volume)
        except (TypeError, ValueError):
            molar_volume = 22.414
            logger.warning("Invalid molar_volume_air_l_mol, using 22.414 L/mol.")

        use_n2_balance = gas_config.get('use_n2_balance', False)
        offgas_is_wet = gas_config.get('offgas_is_wet', False)

        # Handle inlet O2/CO2 (calculated or assumed) - support both key names
        if 'Inlet O2 [%]' in df_out.columns:
            o2_in = pd.to_numeric(df_out['Inlet O2 [%]'], errors='coerce')
        else:
            o2_in = gas_config.get('o2_in_air_percent')
            if o2_in is None:
                o2_in = gas_config.get('inlet_o2_mol_pct', 21.0)
            o2_in = pd.Series(o2_in, index=df_out.index)
            logger.info(f"Using assumed inlet O2: {o2_in.iloc[0]}%")

        if 'Inlet CO2 [%]' in df_out.columns:
            co2_in = pd.to_numeric(df_out['Inlet CO2 [%]'], errors='coerce')
        else:
            co2_in = gas_config.get('co2_in_air_percent')
            if co2_in is None:
                co2_in = gas_config.get('inlet_co2_mol_pct', 0.04)
            co2_in = pd.Series(co2_in, index=df_out.index)

        if use_n2_balance:
            # Convert inlet fractions to dry mol fractions
            y_o2_in = o2_in / 100.0
            y_co2_in = co2_in / 100.0
            y_n2_in = (1.0 - y_o2_in - y_co2_in).clip(lower=0.0)

            # Outlet wet/dry handling
            o2_out_wet = pd.to_numeric(df_out[o2_out_col], errors='coerce') / 100.0
            co2_out_wet = pd.to_numeric(df_out[co2_out_col], errors='coerce') / 100.0

            h2o_out_pct = gas_config.get('outlet_h2o_mol_pct')
            if h2o_out_pct is None:
                h2o_out_pct = gas_config.get('inlet_h2o_mol_pct', 0.0)
            try:
                h2o_out_frac = float(h2o_out_pct) / 100.0
            except (TypeError, ValueError):
                h2o_out_frac = 0.0

            if h2o_out_frac < 0 or h2o_out_frac > 0.3:
                logger.warning(f"Outlet H2O mol fraction {h2o_out_frac:.4f} out of expected range. Clipping.")
                h2o_out_frac = min(max(h2o_out_frac, 0.0), 0.3)

            if offgas_is_wet:
                dry_denom = (1.0 - h2o_out_frac)
                if dry_denom <= 0:
                    dry_denom = np.nan
                o2_out_dry = o2_out_wet / dry_denom
                co2_out_dry = co2_out_wet / dry_denom
            else:
                o2_out_dry = o2_out_wet
                co2_out_dry = co2_out_wet

            y_n2_out = (1.0 - o2_out_dry - co2_out_dry).replace(0, np.nan)

            f_in_mol_h = f_gas * 60.0 / molar_volume
            f_out_mol_h = f_in_mol_h * (y_n2_in / y_n2_out)

            if calculate_cer:
                cer_mol_h = (f_out_mol_h * co2_out_dry) - (f_in_mol_h * y_co2_in)
                df_out['CER [mmol/L/h]'] = (cer_mol_h / volume_l) * 1000.0
                logger.info("Calculated CER [mmol/L/h] using N2 balance.")

            if calculate_our:
                our_mol_h = (f_in_mol_h * y_o2_in) - (f_out_mol_h * o2_out_dry)
                df_out['OUR [mmol/L/h]'] = (our_mol_h / volume_l) * 1000.0
                logger.info("Calculated OUR [mmol/L/h] using N2 balance.")
        else:
            # Legacy simple method using measured fractions and total flow
            factor = 60.0 * 1000.0 / molar_volume
            if calculate_cer:
                co2_out = pd.to_numeric(df_out[co2_out_col], errors='coerce')
                df_out['CER [mmol/L/h]'] = (f_gas * (co2_out / 100.0) / volume_l * factor)
                logger.info("Calculated CER [mmol/L/h] using simple method.")

            if calculate_our:
                o2_out = pd.to_numeric(df_out[o2_out_col], errors='coerce')
                df_out['OUR [mmol/L/h]'] = (f_gas * ((o2_in - o2_out) / 100.0) / volume_l * factor)
                logger.info("Calculated OUR [mmol/L/h] using simple method.")

        # --- Calculate RQ (Respiratory Quotient) ---
        if calculate_rq:
            if 'CER [mmol/L/h]' in df_out.columns and 'OUR [mmol/L/h]' in df_out.columns:
                df_out['RQ [-]'] = df_out['CER [mmol/L/h]'] / df_out['OUR [mmol/L/h]'].replace(0, np.nan)
                logger.info("Calculated RQ [-].")
            else:
                logger.warning("Cannot calculate RQ: CER and/or OUR not available.")

        return df_out

    def _calculate_specific_offgas_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates biomass-specific offgas rates: qO2, qCO2
        qO2 = OUR / Biomass [mmol/g/h]
        qCO2 = CER / Biomass [mmol/g/h]
        """
        logger.info("Calculating specific offgas rates (qO2, qCO2)...")
        df_out = df.copy()

        # --- Get gas config (support both old and new config structures) ---
        gas_config = {}
        gas_config.update(self.config.get('processing', {}).get('gas_params', {}) or {})
        gas_config.update(self.config.get('processing_parameters', {}).get('gas_params', {}) or {})

        # --- Check if calculation is enabled ---
        # Default to True if not specified (for backward compatibility)
        calculate_specific_rates = gas_config.get('calculate_specific_rates', True)
        if not calculate_specific_rates:
            logger.info("Specific offgas rate calculations disabled in config.")
            return df_out

        # --- Required columns ---
        biomass_col = 'Biomass [g/L]'

        if biomass_col not in df_out.columns:
            logger.warning(f"Biomass column '{biomass_col}' not found. Cannot calculate specific rates.")
            return df_out

        biomass = pd.to_numeric(df_out[biomass_col], errors='coerce')

        # --- Calculate qO2 (specific OUR) ---
        if 'OUR [mmol/L/h]' in df_out.columns:
            df_out['qO2 [mmol/g/h]'] = df_out['OUR [mmol/L/h]'] / biomass.replace(0, np.nan)
            logger.info("Calculated qO2 [mmol/g/h].")
        else:
            logger.warning("Cannot calculate qO2: OUR column not found.")

        # --- Calculate qCO2 (specific CER) ---
        if 'CER [mmol/L/h]' in df_out.columns:
            df_out['qCO2 [mmol/g/h]'] = df_out['CER [mmol/L/h]'] / biomass.replace(0, np.nan)
            logger.info("Calculated qCO2 [mmol/g/h].")
        else:
            logger.warning("Cannot calculate qCO2: CER column not found.")

        return df_out

    def _calculate_carbon_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates carbon balance for the fermentation.

        Carbon balance: C_in = C_out + C_accumulated

        Inputs:
        - Glucose from feed (calculated from Feed [g] and concentration)
        - Initial glucose (from batch medium)

        Outputs:
        - CO2 evolved (from CER)
        - Biomass carbon (from elemental composition)
        - Product carbon (from elemental composition, if available)
        - Acetate carbon (if available)

        Accumulated:
        - Residual glucose (from offline samples)
        """
        logger.info("Calculating carbon balance...")
        df_out = df.copy()

        # --- Get mass balance config ---
        # Support both new and old config structures
        mb_config = self.config.get('processing', {}).get('mass_balance_params', {})
        if not mb_config:
            mb_config = self.config.get('processing_parameters', {}).get('mass_balance_params', {})

        # Default to True if not specified (for backward compatibility)
        calculate_cb = mb_config.get('calculate_carbon_balance', True)
        if not calculate_cb:
            logger.info("Carbon balance calculation disabled in config.")
            return df_out

        # --- Get elemental compositions ---
        def _get_first(paths, default=None):
            for path in paths:
                value = self.config
                ok = True
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        ok = False
                        break
                if ok and value is not None:
                    return value
            return default

        # Biomass: CH1.77N0.24O0.49 (from default config)
        biomass_cmol_g = _get_first(
            [
                ['processing', 'biomass_params', 'molar_mass_g_c_mol'],
                ['processing_parameters', 'biomass_params', 'molar_mass_g_c_mol'],
                ['processing_parameters', 'biomass', 'molar_mass_g_c_mol'],
            ],
            default=26.565
        )

        # Glucose: C6H12O6 -> CH2O on C-mol basis
        glucose_cmol_g = _get_first(
            [
                ['processing_parameters', 'substrate_properties', 'glucose', 'molar_mass_g_c_mol'],
                ['processing', 'processing_parameters', 'substrate_properties', 'glucose', 'molar_mass_g_c_mol'],
            ],
            default=30.03
        )

        # Product: from run_config (if available)
        product_cmol_g = None
        try:
            product_props = _get_first(
                [
                    ['processing_parameters', 'product_properties'],
                    ['processing', 'product_params'],
                ],
                default={}
            )
            if isinstance(product_props, dict) and 'elemental_composition_c_mol' in product_props:
                product_cmol_g = product_props.get('molar_mass_g_c_mol')
                logger.info(f"Found product elemental composition: {product_cmol_g} g/C-mol")
        except Exception:
            logger.info("No product elemental composition found in config.")

        # Acetate: C2H4O2 -> CH2O on C-mol basis
        acetate_cmol_g = _get_first(
            [
                ['processing_parameters', 'substrate_properties', 'acetate', 'molar_mass_g_c_mol'],
                ['processing', 'processing_parameters', 'substrate_properties', 'acetate', 'molar_mass_g_c_mol'],
            ],
            default=29.5
        )

        # Feed concentration (g glucose per L feed) - from media config
        feed_conc_g_l = _get_first(
            [['analysis', 'delisa_media_config', 'feed_1', 'components', 'glucose', 'concentration_g_l']],
            default=440.0
        )

        # Initial volume and batch glucose concentration
        initial_volume_ml = _get_first(
            [['processing_parameters', 'reactor_setup', 'initial_volume_ml']],
            default=1100.0
        )
        initial_glucose_g_l = _get_first(
            [['analysis', 'delisa_media_config', 'batch_medium', 'components', 'glucose', 'concentration_g_l']],
            default=22.0
        )

        # --- Get required columns ---
        required_cols = ['Feed [mL]', 'Volume corrected [mL]', 'CER [mmol/L/h]']
        missing = [c for c in required_cols if c not in df_out.columns]
        if missing:
            logger.warning(f"Missing required columns for carbon balance: {missing}")
            return df_out

        # --- Calculate time delta (in hours) for integration ---
        time_col = 'Process time [h]'
        if time_col not in df_out.columns:
            logger.warning(f"Time column '{time_col}' not found. Cannot calculate carbon balance.")
            return df_out

        dt_hours = df_out[time_col].diff().fillna(0)

        # --- Carbon INPUTS ---

        # 1. Carbon from feed (cumulative)
        # Feed [mL] × feed_conc (g/L) / glucose_cmol_g (g/C-mol) = C-mol from feed
        # NOTE: Feed [mL] is already a cumulative value, so we do NOT apply cumsum() again
        feed_ml = pd.to_numeric(df_out['Feed [mL]'], errors='coerce')
        feed_g = feed_ml * feed_conc_g_l / 1000.0  # Convert mL to L
        df_out['C from feed [C-mol]'] = feed_g / glucose_cmol_g

        # 2. Initial glucose from batch medium
        initial_volume_l = initial_volume_ml / 1000.0
        initial_c_mol = initial_volume_l * initial_glucose_g_l / glucose_cmol_g
        df_out['C from batch [C-mol]'] = initial_c_mol

        # Total carbon in
        df_out['C in [C-mol]'] = df_out['C from feed [C-mol]'] + df_out['C from batch [C-mol]']

        # --- Carbon OUTPUTS ---

        # 1. CO2 carbon (cumulative)
        # CER [mmol/L/h] × Volume [L] × dt [h] = mmol CO2 produced
        # 1 mmol CO2 = 1 mmol C = 0.001 C-mol
        volume_l = pd.to_numeric(df_out['Volume corrected [mL]'], errors='coerce') / 1000.0
        cer_mmol_l_h = pd.to_numeric(df_out['CER [mmol/L/h]'], errors='coerce')
        co2_production_rate_c_mol_h = cer_mmol_l_h * volume_l * 0.001  # C-mol/h
        df_out['C from CO2 [C-mol]'] = (co2_production_rate_c_mol_h * dt_hours).cumsum()

        # 2. Biomass carbon
        # Biomass [g/L] × Volume [L] / biomass_cmol_g = C-mol biomass (current amount in reactor)
        biomass_g_l = pd.to_numeric(df_out['Biomass [g/L]'], errors='coerce')
        biomass_c_mol_in_reactor = biomass_g_l * volume_l / biomass_cmol_g
        df_out['C in biomass [C-mol]'] = biomass_c_mol_in_reactor

        # 3. Product carbon (if available)
        if 'Protein [g/L]' in df_out.columns and product_cmol_g is not None:
            protein_g_l = pd.to_numeric(df_out['Protein [g/L]'], errors='coerce')
            product_c_mol_in_reactor = protein_g_l * volume_l / product_cmol_g
            df_out['C in product [C-mol]'] = product_c_mol_in_reactor
        else:
            df_out['C in product [C-mol]'] = np.nan
            product_c_mol_in_reactor = pd.Series(np.nan, index=df_out.index)
            logger.info("Product carbon not calculated (no product data or composition)")

        # 4. Acetate carbon (if available)
        if 'Acetate [g/L]' in df_out.columns:
            acetate_g_l = pd.to_numeric(df_out['Acetate [g/L]'], errors='coerce')
            df_out['C in acetate [C-mol]'] = acetate_g_l * volume_l / acetate_cmol_g
        else:
            df_out['C in acetate [C-mol]'] = 0.0
            logger.info("Acetate carbon not calculated (no acetate data)")

        # 5. Residual glucose carbon (if available)
        if 'Glucose [g/L]' in df_out.columns:
            glucose_g_l = pd.to_numeric(df_out['Glucose [g/L]'], errors='coerce')
            df_out['C residual glucose [C-mol]'] = glucose_g_l * volume_l / glucose_cmol_g
        else:
            df_out['C residual glucose [C-mol]'] = 0.0
            logger.info("Residual glucose carbon not calculated (no glucose data)")

        # --- Carbon BALANCE ---
        # Account for cumulative removals from sampling to build a cumulative balance.
        sample_vol_ml = pd.to_numeric(df_out.get('Sample volume [mL]', 0.0), errors='coerce').fillna(0.0)
        is_sample = df_out.get('is_sample', False)
        if isinstance(is_sample, pd.Series):
            is_sample = is_sample.fillna(False).astype(bool)
        else:
            is_sample = pd.Series(False, index=df_out.index)

        sample_vol_l = sample_vol_ml / 1000.0

        removed_biomass_c = pd.Series(0.0, index=df_out.index)
        if biomass_g_l.notna().any():
            removed_biomass_c.loc[is_sample] = (biomass_g_l * sample_vol_l / biomass_cmol_g).loc[is_sample].fillna(0.0)
        df_out['C removed biomass [C-mol]'] = removed_biomass_c.cumsum()

        removed_product_c = pd.Series(0.0, index=df_out.index)
        if 'Protein [g/L]' in df_out.columns and product_cmol_g is not None:
            removed_product_c.loc[is_sample] = (protein_g_l * sample_vol_l / product_cmol_g).loc[is_sample].fillna(0.0)
        df_out['C removed product [C-mol]'] = removed_product_c.cumsum()

        removed_acetate_c = pd.Series(0.0, index=df_out.index)
        if 'Acetate [g/L]' in df_out.columns:
            removed_acetate_c.loc[is_sample] = (acetate_g_l * sample_vol_l / acetate_cmol_g).loc[is_sample].fillna(0.0)
        df_out['C removed acetate [C-mol]'] = removed_acetate_c.cumsum()

        removed_glucose_c = pd.Series(0.0, index=df_out.index)
        if 'Glucose [g/L]' in df_out.columns:
            removed_glucose_c.loc[is_sample] = (glucose_g_l * sample_vol_l / glucose_cmol_g).loc[is_sample].fillna(0.0)
        df_out['C removed glucose [C-mol]'] = removed_glucose_c.cumsum()

        df_out['C biomass total [C-mol]'] = df_out['C in biomass [C-mol]'].fillna(0.0) + df_out['C removed biomass [C-mol]']
        df_out['C product total [C-mol]'] = df_out['C in product [C-mol]'].fillna(0.0) + df_out['C removed product [C-mol]']
        df_out['C acetate total [C-mol]'] = df_out['C in acetate [C-mol]'].fillna(0.0) + df_out['C removed acetate [C-mol]']
        df_out['C glucose total [C-mol]'] = df_out['C residual glucose [C-mol]'].fillna(0.0) + df_out['C removed glucose [C-mol]']

        df_out['C out [C-mol]'] = (
            df_out['C from CO2 [C-mol]'] +
            df_out['C biomass total [C-mol]'] +
            df_out['C product total [C-mol]'] +
            df_out['C acetate total [C-mol]'] +
            df_out['C glucose total [C-mol]']
        )

        # --- Carbon DISTRIBUTION (at sample points) ---
        # Shows what percentage of fed carbon has gone to each compartment
        # Note: This is NOT a mass balance closure (won't sum to 100%)
        # because biomass/product that was sampled out or died is not accounted for
        # However, it shows the carbon flow through the system

        # Calculate percentage of fed C in each compartment
        df_out['C to CO2 [% of fed]'] = (df_out['C from CO2 [C-mol]'] / df_out['C in [C-mol]'].replace(0, np.nan)) * 100.0
        df_out['C to biomass [% of fed]'] = (df_out['C biomass total [C-mol]'].fillna(0) / df_out['C in [C-mol]'].replace(0, np.nan)) * 100.0
        df_out['C to product [% of fed]'] = (df_out['C product total [C-mol]'].fillna(0) / df_out['C in [C-mol]'].replace(0, np.nan)) * 100.0
        df_out['C residual glucose [% of fed]'] = (df_out['C glucose total [C-mol]'].fillna(0) / df_out['C in [C-mol]'].replace(0, np.nan)) * 100.0

        # Calculate a "Carbon Recovery" metric
        # C recovered = CO2 + biomass/product/glucose totals (including sampled removals)
        df_out['C recovered [%]'] = (
            (df_out['C from CO2 [C-mol]'] +
             df_out['C biomass total [C-mol]'] +
             df_out['C product total [C-mol]'] +
             df_out['C glucose total [C-mol]']) /
            df_out['C in [C-mol]'].replace(0, np.nan) * 100.0
        )

        # Keep the old column name for backward compatibility, but rename it
        df_out['C balance closure [%]'] = df_out['C recovered [%]']
        df_out['C balance error [%]'] = 100.0 - df_out['C recovered [%]']

        # Limit to reasonable range for display
        df_out['C recovered [%]'] = df_out['C recovered [%]'].clip(lower=0, upper=200)
        df_out['C balance closure [%]'] = df_out['C balance closure [%]'].clip(lower=0, upper=200)

        logger.info("Calculated carbon balance (C_in, C_out, closure).")
        return df_out

    def _calculate_rates_and_yields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates rates (mu, qs, qP) from online/offline signals."""
        logger.info("Calculating rates (mu, qs, qP)...")
        df_out = df.copy()

        time_col = 'Process time [h]'
        if time_col not in df_out.columns:
            logger.warning(f"Time column '{time_col}' not found. Cannot calculate rates.")
            return df_out

        time_h = pd.to_numeric(df_out[time_col], errors='coerce')
        if time_h.isna().all():
            logger.warning("Time column is all NaN. Cannot calculate rates.")
            return df_out

        rate_config = {}
        rate_config.update(self.config.get('processing', {}).get('rate_params', {}) or {})
        rate_config.update(self.config.get('processing_parameters', {}).get('rate_params', {}) or {})

        apply_smoothing = rate_config.get('apply_smoothing', True)
        window_points = int(rate_config.get('smoothing_window_points', 5))
        min_periods = int(rate_config.get('smoothing_min_periods', 1))

        def _smooth(series: pd.Series) -> pd.Series:
            if not apply_smoothing or window_points <= 1:
                return series
            return series.rolling(window=window_points, center=True, min_periods=min_periods).mean()

        def _interp_series(series: pd.Series) -> pd.Series:
            s = pd.to_numeric(series, errors='coerce')
            mask = time_h.notna() & s.notna()
            if mask.sum() < 2:
                return s
            t_vals = time_h[mask].to_numpy()
            v_vals = s[mask].to_numpy()
            order = t_vals.argsort()
            t_sorted = t_vals[order]
            v_sorted = v_vals[order]

            interp_method = rate_config.get('interpolation_method', 'spline')
            t_all = time_h.to_numpy()

            if interp_method == 'spline':
                try:
                    from scipy.interpolate import UnivariateSpline
                    spline_order = int(rate_config.get('spline_order', 3))
                    spline_s = rate_config.get('spline_smoothing', 0.0)
                    try:
                        spline_s = float(spline_s)
                    except (TypeError, ValueError):
                        spline_s = 0.0

                    # Deduplicate time points for spline stability
                    dedup = pd.DataFrame({'t': t_sorted, 'v': v_sorted}).groupby('t', as_index=False).mean()
                    t_unique = dedup['t'].to_numpy()
                    v_unique = dedup['v'].to_numpy()

                    if len(t_unique) >= spline_order + 1:
                        spline = UnivariateSpline(t_unique, v_unique, k=spline_order, s=spline_s)
                        v_all = spline(t_all)
                        return pd.Series(v_all, index=df_out.index)
                    logger.warning("Not enough points for spline interpolation, falling back to linear.")
                except Exception as e:
                    logger.warning(f"Spline interpolation failed, falling back to linear: {e}")

            v_all = np.interp(t_all, t_sorted, v_sorted)
            return pd.Series(v_all, index=df_out.index)

        dt_h = time_h.diff().fillna(0.0)
        dt_h = dt_h.where(dt_h > 0, np.nan)

        v_corr_l = pd.to_numeric(df_out.get('Volume corrected [mL]'), errors='coerce') / 1000.0
        v_unc_l = pd.to_numeric(df_out.get('Volume [mL]'), errors='coerce') / 1000.0

        biomass_g_l = _interp_series(df_out.get('Biomass [g/L]', pd.Series(index=df_out.index, dtype=float)))
        biomass_g = biomass_g_l * v_corr_l

        protein_col = None
        if 'Protein total calculated [g/L]' in df_out.columns:
            protein_col = 'Protein total calculated [g/L]'
        elif 'Protein [g/L]' in df_out.columns:
            protein_col = 'Protein [g/L]'
        protein_g_l = _interp_series(df_out.get(protein_col, pd.Series(index=df_out.index, dtype=float)))
        protein_g = protein_g_l * v_corr_l

        glucose_g_l = _interp_series(df_out.get('Glucose [g/L]', pd.Series(index=df_out.index, dtype=float)))
        glucose_g = glucose_g_l * v_corr_l

        sample_vol_ml = pd.to_numeric(df_out.get('Sample volume [mL]', 0.0), errors='coerce').fillna(0.0)
        is_sample = df_out.get('is_sample', False)
        if isinstance(is_sample, pd.Series):
            is_sample = is_sample.fillna(False).astype(bool)
        else:
            is_sample = pd.Series(False, index=df_out.index)
        sample_vol_l = sample_vol_ml / 1000.0

        sample_out_glc_g = pd.Series(0.0, index=df_out.index)
        if glucose_g_l.notna().any():
            sample_out_glc_g.loc[is_sample] = (glucose_g_l * sample_vol_l).loc[is_sample].fillna(0.0)
        sample_out_glc_cum = sample_out_glc_g.cumsum()

        sample_out_protein_g = pd.Series(0.0, index=df_out.index)
        if protein_g_l.notna().any():
            sample_out_protein_g.loc[is_sample] = (protein_g_l * sample_vol_l).loc[is_sample].fillna(0.0)
        sample_out_protein_cum = sample_out_protein_g.cumsum()

        sample_out_biomass_g = pd.Series(0.0, index=df_out.index)
        if biomass_g_l.notna().any():
            sample_out_biomass_g.loc[is_sample] = (biomass_g_l * sample_vol_l).loc[is_sample].fillna(0.0)
        sample_out_biomass_cum = sample_out_biomass_g.cumsum()

        feed_ml = pd.to_numeric(df_out.get('Feed [mL]', 0.0), errors='coerce').fillna(0.0)
        d_feed_ml = feed_ml.diff().fillna(0.0).clip(lower=0.0)

        feed_conc_g_l = None
        if isinstance(self.config, dict):
            feed_conc_g_l = (
                self.config.get('analysis', {})
                .get('delisa_media_config', {})
                .get('feed_1', {})
                .get('components', {})
                .get('glucose', {})
                .get('concentration_g_l')
            )
        if feed_conc_g_l is None:
            feed_conc_g_l = 400.0
        try:
            feed_conc_g_l = float(feed_conc_g_l)
        except (TypeError, ValueError):
            feed_conc_g_l = 400.0

        feed_g = d_feed_ml / 1000.0 * feed_conc_g_l
        feed_g_cum = feed_g.cumsum()

        initial_volume_ml = None
        initial_glucose_g_l = None
        if isinstance(self.config, dict):
            initial_volume_ml = (
                self.config.get('processing_parameters', {})
                .get('reactor_setup', {})
                .get('initial_volume_ml')
            )
            initial_glucose_g_l = (
                self.config.get('analysis', {})
                .get('delisa_media_config', {})
                .get('batch_medium', {})
                .get('components', {})
                .get('glucose', {})
                .get('concentration_g_l')
            )
        try:
            initial_volume_ml = float(initial_volume_ml)
        except (TypeError, ValueError):
            initial_volume_ml = 1100.0
        try:
            initial_glucose_g_l = float(initial_glucose_g_l)
        except (TypeError, ValueError):
            initial_glucose_g_l = 22.0

        initial_glucose_g = (initial_volume_ml / 1000.0) * initial_glucose_g_l

        glucose_consumed_g = initial_glucose_g + feed_g_cum - sample_out_glc_cum - glucose_g
        glucose_consumed_rate_g_h = glucose_consumed_g.diff() / dt_h
        glucose_consumed_rate_g_h = _smooth(glucose_consumed_rate_g_h)

        df_out['Glucose consumed [g]'] = glucose_consumed_g
        df_out['Glucose consumed rate [g/h]'] = glucose_consumed_rate_g_h

        total_biomass_g = biomass_g
        df_out['Biomass total [g]'] = total_biomass_g

        qs_glc = glucose_consumed_rate_g_h / total_biomass_g.replace(0, np.nan)
        df_out['qS glucose [g/g/h]'] = qs_glc

        # Growth rate from biomass concentration
        mu = pd.Series(np.nan, index=df_out.index)
        valid_x = biomass_g_l > 0
        if valid_x.any():
            mu = (np.log(biomass_g_l.where(valid_x)).diff() / dt_h)
            mu = _smooth(mu)
        df_out['mu [1/h]'] = mu

        # Product formation rate
        produced_protein_g = protein_g + sample_out_protein_cum
        protein_rate_g_h = produced_protein_g.diff() / dt_h
        protein_rate_g_h = _smooth(protein_rate_g_h)
        df_out['Protein produced [g]'] = produced_protein_g
        df_out['Protein production rate [g/h]'] = protein_rate_g_h
        df_out['qP total [g/g/h]'] = protein_rate_g_h / total_biomass_g.replace(0, np.nan)

        # Offgas-derived qs from carbon balance (if available)
        if 'C out [C-mol]' in df_out.columns:
            c_out = pd.to_numeric(df_out['C out [C-mol]'], errors='coerce')
            c_out_rate = c_out.diff() / dt_h
            c_out_rate = _smooth(c_out_rate)

            glucose_cmol_g = 30.026
            if isinstance(self.config, dict):
                glucose_cmol_g_cfg = (
                    self.config.get('processing_parameters', {})
                    .get('substrate_properties', {})
                    .get('glucose', {})
                    .get('molar_mass_g_c_mol')
                )
                if glucose_cmol_g_cfg is not None:
                    try:
                        glucose_cmol_g = float(glucose_cmol_g_cfg)
                    except (TypeError, ValueError):
                        pass

            offgas_glc_rate_g_h = c_out_rate * glucose_cmol_g
            df_out['Glucose uptake offgas rate [g/h]'] = offgas_glc_rate_g_h
            df_out['qS offgas [g/g/h]'] = offgas_glc_rate_g_h / total_biomass_g.replace(0, np.nan)

            offgas_glc_cum = (offgas_glc_rate_g_h * dt_h).cumsum()
            scale = np.nan
            valid = glucose_consumed_g.notna() & offgas_glc_cum.notna()
            if valid.any():
                idx = valid[valid].index[-1]
                denom = offgas_glc_cum.loc[idx]
                numer = glucose_consumed_g.loc[idx]
                if pd.notna(denom) and denom > 0:
                    scale = numer / denom

            if pd.notna(scale):
                df_out['Offgas glucose scale [-]'] = scale
                df_out['Glucose uptake offgas rate calibrated [g/h]'] = offgas_glc_rate_g_h * scale
                df_out['qS offgas calibrated [g/g/h]'] = (offgas_glc_rate_g_h * scale) / total_biomass_g.replace(0, np.nan)

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
        processed_df = self._add_induction_indicator(processed_df)
        processed_df = self._add_induction_timer(processed_df)
        processed_df = self._calculate_protein_parameters(processed_df)
        processed_df = self._calculate_inlet_gas_composition(processed_df)
        processed_df = self._calculate_offgas_rates(processed_df)
        processed_df = self._calculate_specific_offgas_rates(processed_df)
        processed_df = self._calculate_carbon_balance(processed_df)
        processed_df = self._calculate_rates_and_yields(processed_df)

        logger.info("Data processing finished.")
        return processed_df
