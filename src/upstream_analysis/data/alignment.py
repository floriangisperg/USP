# src/upstream_analysis/data/alignment.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class DataAligner:
    """Class for aligning online and offline fermentation data based on time."""

    def __init__(self, config: dict):
        self.config = config
        self.online_time_col = config.get('online_time_col', 'process_time')
        self.offline_time_col = config.get('offline_time_col', 'process_time')
        self.tolerance_hours = config.get('tolerance_hours', 0.02)
        self.force_align_last = config.get('force_align_last_sample', False)
        logger.info(
            f"DataAligner initialized. Online time: '{self.online_time_col}', "
            f"Offline time: '{self.offline_time_col}', Tolerance: {self.tolerance_hours}h, "
            f"Force align last: {self.force_align_last}"
        )

    def merge_datasets(
        self,
        online_df: pd.DataFrame,
        offline_df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns online and offline datasets using pandas.merge_asof.
        Optionally forces the last offline sample to align with the last online point.
        """
        logger.info("Starting data alignment...")
        online_clean = online_df.copy()

        if offline_df is None or offline_df.empty:
            logger.info("No offline data provided or empty. Skipping alignment.")
            online_clean['is_sample'] = False
            return online_clean, pd.DataFrame(columns=online_clean.columns)

        # --- Input Validation and Preparation ---
        if self.online_time_col not in online_clean.columns or self.offline_time_col not in offline_df.columns:
            raise KeyError(f"Required time columns ('{self.online_time_col}', '{self.offline_time_col}') not found.")

        online_clean[self.online_time_col] = pd.to_numeric(online_clean[self.online_time_col], errors='coerce')
        offline_df[self.offline_time_col] = pd.to_numeric(offline_df[self.offline_time_col], errors='coerce')

        online_sorted = online_clean.dropna(subset=[self.online_time_col]).sort_values(by=self.online_time_col).reset_index(drop=True)
        offline_sorted = offline_df.dropna(subset=[self.offline_time_col]).sort_values(by=self.offline_time_col).reset_index(drop=True)

        if online_sorted.empty or offline_sorted.empty:
            logger.warning("Online or offline data empty after dropping NaN times.")
            online_sorted['is_sample'] = False
            return online_sorted, pd.DataFrame()

        offline_time_col_temp = f"{self.offline_time_col}_offline_temp_merge"
        offline_renamed = offline_sorted.rename(columns={self.offline_time_col: offline_time_col_temp})

        # --- Handle Last Sample ---
        last_offline_sample_row = None
        last_offline_time_val = None
        offline_to_align = offline_renamed

        if self.force_align_last and not offline_renamed.empty:
            logger.info("Separating last offline sample for potential forced alignment.")
            last_offline_sample_row = offline_renamed.iloc[-1:].copy()
            last_offline_time_val = last_offline_sample_row[offline_time_col_temp].iloc[0] # Get its time
            offline_to_align = offline_renamed.iloc[:-1]
            if offline_to_align.empty:
                logger.info("Only one offline sample; will try standard alignment first.")
                offline_to_align = offline_renamed
                last_offline_sample_row = None # Don't force yet

        # --- Perform Standard Alignment ---
        merged_df = online_sorted.copy()
        # Add empty columns for offline data upfront
        offline_cols_original = [col for col in offline_sorted.columns if col != self.offline_time_col]
        for col in offline_cols_original:
             if col not in merged_df.columns: merged_df[col] = pd.NA

        best_match_indices = pd.Index([]) # Indices in merged_df/online_sorted
        temp_merged = pd.DataFrame() # To store merge_asof result

        if not offline_to_align.empty:
            logger.info(f"Performing standard merge_asof on {len(offline_to_align)} sample(s).")
            try:
                temp_merged = pd.merge_asof(
                    left=online_sorted.reset_index(), # Keep original index via reset_index
                    right=offline_to_align,
                    left_on=self.online_time_col,
                    right_on=offline_time_col_temp,
                    direction='nearest',
                    tolerance=self.tolerance_hours,
                    suffixes=('_online', '_offline')
                )

                merged_rows_mask = temp_merged[offline_time_col_temp].notna()
                if merged_rows_mask.any():
                    temp_merged['time_diff'] = abs(
                        temp_merged.loc[merged_rows_mask, self.online_time_col] - temp_merged.loc[merged_rows_mask, offline_time_col_temp]
                    )
                    best_match_temp_indices = temp_merged.loc[merged_rows_mask].groupby(offline_time_col_temp)['time_diff'].idxmin()
                    best_match_indices = temp_merged.loc[best_match_temp_indices, 'index'].astype(int) # Original online indices

                    # --- Update main merged_df ---
                    offline_cols_to_update = [col for col in offline_to_align.columns if col != offline_time_col_temp]
                    for temp_idx, original_idx in zip(best_match_temp_indices, best_match_indices):
                        for col in offline_cols_to_update:
                             source_col = col + '_offline' if col + '_offline' in temp_merged.columns else col
                             if source_col in temp_merged.columns:
                                 merged_df.loc[original_idx, col] = temp_merged.loc[temp_idx, source_col]

                    sample_count_std = len(best_match_indices)
                    logger.info(f"Standard alignment identified {sample_count_std} sample points.")
                    if sample_count_std != len(offline_to_align):
                         logger.warning(f"Std alignment count differs from offline points attempted.")
                else:
                     logger.info("No standard offline samples matched within tolerance.")
            except Exception as e:
                 logger.error(f"Error during standard pd.merge_asof: {e}")

        # --- Force Align Last Sample ---
        force_applied = False
        if self.force_align_last and last_offline_sample_row is not None:
            # --- CORRECTED CHECK: Check using temp_merged IF it was populated ---
            already_matched = False
            if not temp_merged.empty and not best_match_temp_indices.empty:
                 # Check if the last offline time value exists among the successfully merged offline times in temp_merged
                 matched_offline_times = temp_merged.loc[best_match_temp_indices, offline_time_col_temp].dropna()
                 if not matched_offline_times.empty:
                     already_matched = matched_offline_times.isin([last_offline_time_val]).any()
            # --- END CORRECTED CHECK ---

            if not already_matched:
                last_online_index = online_sorted.index[-1]
                logger.info(f"Forcing alignment of last offline sample (original time: {last_offline_time_val:.2f}h) to last online point (index {last_online_index}, time: {online_sorted.loc[last_online_index, self.online_time_col]:.2f}h).")

                if last_online_index in best_match_indices:
                    logger.warning(f"Last online row (index {last_online_index}) was already matched by a different standard sample. Overwriting with forced last sample.")

                for col in offline_sorted.columns:
                    if col == self.offline_time_col: continue
                    source_col = col
                    target_col = col
                    if source_col in last_offline_sample_row.columns:
                        if target_col not in merged_df.columns: merged_df[target_col] = pd.NA
                        merged_df.loc[last_online_index, target_col] = last_offline_sample_row[source_col].iloc[0]
                force_applied = True
            else:
                 logger.info("Last offline sample was already matched during standard alignment. No forced alignment needed.")


        # --- Set Final 'is_sample' Column ---
        merged_df['is_sample'] = False
        if not best_match_indices.empty:
            merged_df.loc[best_match_indices, 'is_sample'] = True # Mark standard matches
        if force_applied:
            merged_df.loc[last_online_index, 'is_sample'] = True # Mark forced match

        # --- Final Cleanup ---
        # Drop only the temporary column if it exists (it won't if only forced align happened)
        cols_to_drop = [offline_time_col_temp, 'time_diff', 'index']
        merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns], inplace=True, errors='ignore')

        samples_df = merged_df[merged_df['is_sample']].copy()
        final_sample_count = len(samples_df)
        logger.info(f"Alignment complete. Final identified sample points: {final_sample_count}.")
        if final_sample_count != len(offline_sorted):
             logger.warning(f"Final sample count ({final_sample_count}) differs from original offline rows ({len(offline_sorted)}).")

        logger.info(f"Final merged data shape: {merged_df.shape}, Samples data shape: {samples_df.shape}")
        return merged_df, samples_df