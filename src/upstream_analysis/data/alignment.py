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
        logger.info(f"DataAligner initialized. Online time: '{self.online_time_col}', Offline time: '{self.offline_time_col}', Tolerance: {self.tolerance_hours}h")

    def merge_datasets(
        self,
        online_df: pd.DataFrame,
        offline_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns online and offline datasets using pandas.merge_asof, ensuring only
        the single best time match for each offline point is marked as a sample.

        Args:
            online_df: DataFrame with cleaned online measurements, MUST contain
                       the numeric `online_time_col` and be sorted by it.
            offline_df: DataFrame with parsed offline measurements, MUST contain
                        the numeric `offline_time_col` and be sorted by it.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - Merged DataFrame: Online data with offline data merged onto the
                  single closest online time point row for each sample.
                - Samples DataFrame: Rows corresponding to the unique offline samples.
        """
        logger.info("Starting data alignment...")

        # --- Input Validation and Preparation ---
        if self.online_time_col not in online_df.columns:
            msg = f"Online time column '{self.online_time_col}' not found in online data."
            logger.error(msg); raise KeyError(msg)
        if self.offline_time_col not in offline_df.columns:
            msg = f"Offline time column '{self.offline_time_col}' not found in offline data."
            logger.error(msg); raise KeyError(msg)

        online_df[self.online_time_col] = pd.to_numeric(online_df[self.online_time_col], errors='coerce')
        offline_df[self.offline_time_col] = pd.to_numeric(offline_df[self.offline_time_col], errors='coerce')

        online_clean = online_df.dropna(subset=[self.online_time_col]).copy()
        offline_clean = offline_df.dropna(subset=[self.offline_time_col]).copy()

        if online_clean.empty or offline_clean.empty:
            logger.warning("One or both dataframes are empty after handling NaN time values. Returning online data only.")
            online_clean['is_sample'] = False
            return online_clean, pd.DataFrame()

        online_sorted = online_clean.sort_values(by=self.online_time_col).reset_index(drop=True)
        offline_sorted = offline_clean.sort_values(by=self.offline_time_col).reset_index(drop=True)

        # --- Explicitly Rename Offline Time Column BEFORE Merge ---
        # This ensures it's preserved even if names are identical.
        offline_time_col_temp = f"{self.offline_time_col}_offline_temp"
        offline_renamed = offline_sorted.rename(columns={self.offline_time_col: offline_time_col_temp})

        # --- Perform Merge ---
        logger.info(f"Performing merge_asof on '{self.online_time_col}' / '{offline_time_col_temp}' with tolerance {self.tolerance_hours} hours.")
        try:
            merged_df = pd.merge_asof(
                left=online_sorted.copy(), # Use copy to avoid modifying original if needed later
                right=offline_renamed,
                left_on=self.online_time_col,
                right_on=offline_time_col_temp,
                direction='nearest',
                tolerance=pd.Timedelta(hours=self.tolerance_hours) if isinstance(online_sorted[self.online_time_col].iloc[0], pd.Timestamp) else self.tolerance_hours,
                suffixes=('', '_offline') # Suffixes for any *other* overlapping columns
            )
        except Exception as e:
             logger.error(f"Error during pd.merge_asof: {e}")
             online_sorted['is_sample'] = False
             return online_sorted, pd.DataFrame()

        # --- Post-Merge Deduplication ---
        # Identify rows where a merge actually happened (the temp offline time column is not NaN)
        merged_rows_mask = merged_df[offline_time_col_temp].notna()

        if not merged_rows_mask.any():
            logger.warning("No offline samples could be matched within the tolerance.")
            merged_df['is_sample'] = False
            merged_df.drop(columns=[offline_time_col_temp], inplace=True, errors='ignore')
            return merged_df, pd.DataFrame()

        # Calculate absolute time difference for merged rows
        merged_df['time_diff'] = np.nan # Initialize column
        merged_df.loc[merged_rows_mask, 'time_diff'] = abs(
            merged_df.loc[merged_rows_mask, self.online_time_col] - merged_df.loc[merged_rows_mask, offline_time_col_temp]
        )

        # Find the index of the minimum time difference for each unique offline time point
        # This gives the index in merged_df that is the *best* match for each offline sample
        best_match_indices = merged_df.loc[merged_rows_mask].groupby(offline_time_col_temp)['time_diff'].idxmin()

        # Create the 'is_sample' column, True only for the best matches
        merged_df['is_sample'] = False
        merged_df.loc[best_match_indices, 'is_sample'] = True

        # Identify columns that came *only* from the offline dataframe (excluding time)
        offline_original_cols = [col for col in offline_sorted.columns if col != self.offline_time_col]
        # Identify columns in merged_df that correspond to these (might have _offline suffix if names clashed)
        offline_cols_in_merged = [col for col in merged_df.columns if
                                  col in offline_original_cols or col.replace('_offline', '') in offline_original_cols]

        # Create a mask for rows that are NOT the best match
        not_best_match_mask = ~merged_df.index.isin(best_match_indices)

        # Set offline columns to NaN for rows that are not the best match
        if offline_cols_in_merged:
            merged_df.loc[not_best_match_mask, offline_cols_in_merged] = np.nan
            logger.info(f"Nulled offline data in {not_best_match_mask.sum()} non-best-match rows.")

        # Create the samples_df using the corrected 'is_sample' flag
        samples_df = merged_df[merged_df['is_sample']].copy()

        # Clean up temporary columns
        merged_df.drop(columns=[offline_time_col_temp, 'time_diff'], inplace=True, errors='ignore')
        # Also drop them from samples_df if they exist there
        samples_df.drop(columns=[offline_time_col_temp, 'time_diff'], inplace=True, errors='ignore')

        sample_count_final = merged_df['is_sample'].sum()
        logger.info(f"Alignment refined. Identified {sample_count_final} unique sample points.")
        if sample_count_final != len(offline_sorted):
             logger.warning(f"Final sample count ({sample_count_final}) still differs from original offline rows ({len(offline_sorted)}). This might happen if multiple offline points map to the same closest online point, or some offline points had no match within tolerance.")

        logger.info(f"Alignment complete. Merged data shape: {merged_df.shape}, Samples data shape: {samples_df.shape}")
        return merged_df, samples_df