# src/upstream_analysis/data/parser.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class BioreactorDataParser:
    """Parser for online bioreactor data files."""

    def __init__(self, config: dict):
        self.config = config
        self.parser_config = config.get('parsing', {}).get('bioreactor', {})
        self.col_map_config = config.get('column_mapping', {})
        self.decimal = self.parser_config.get('decimal_separator', ',')
        self.separator = self.parser_config.get('column_separator', ';')
        self.date_format = self.parser_config.get('date_format', None) # None lets pandas guess
        self.skiprows = self.parser_config.get('skiprows', 1)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies renaming based on the simplified config."""
        # Invert the mapping: {Original: Standard}
        # Assumes values in config are the *exact* original headers now
        mapping = {v: k for k, v in self.col_map_config.items() if isinstance(v, str) and v in df.columns}

        missing_keys = [k for k, v in self.col_map_config.items() if isinstance(v, str) and v not in df.columns]
        if missing_keys:
            logger.warning(f"Expected columns from config not found in online data: {missing_keys}")

        renamed_df = df.rename(columns=mapping)
        logger.info(f"Renamed {len(mapping)} online columns based on config.")
        return renamed_df

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic type conversions for online data."""
        result_df = df.copy()
        date_format_to_use = self.date_format if self.date_format and self.date_format.lower() != 'guess' else None

        # Datetime columns (use standardized names)
        time_cols = ['date_time_utc', 'date_time_local']
        for col in time_cols:
            if col in result_df.columns:
                try:
                    result_df[col] = pd.to_datetime(result_df[col], format=date_format_to_use, errors='coerce')
                except ValueError: # Try without format if specified one failed
                     logger.warning(f"Format '{date_format_to_use}' failed for '{col}'. Attempting default parsing.")
                     result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                except Exception as e:
                     logger.error(f"Error parsing datetime column {col}: {e}")
                     result_df[col] = pd.NaT # Assign NaT on other errors

        # Process phase
        if 'process_phase' in result_df.columns:
            result_df['process_phase'] = result_df['process_phase'].astype(str).str.strip()
            result_df['process_phase'] = result_df['process_phase'].str.replace(r'([a-z])([A-Z])', r'\1 \2', regex=True)

        # Attempt numeric conversion for all other columns
        numeric_cols = [c for c in result_df.columns if c not in time_cols and c != 'process_phase']
        for col in numeric_cols:
             # Check if it's not already numeric/datetime
             if not pd.api.types.is_numeric_dtype(result_df[col]) and not pd.api.types.is_datetime64_any_dtype(result_df[col]):
                 original_dtype = result_df[col].dtype
                 try:
                     # Handle potential non-numeric chars and decimal separator
                     series_to_convert = result_df[col]
                     if isinstance(series_to_convert.iloc[0], str):
                         series_to_convert = series_to_convert.str.replace(self.decimal, '.', regex=False)

                     result_df[col] = pd.to_numeric(series_to_convert, errors='coerce')
                     logger.debug(f"Attempted numeric conversion for column '{col}' (original dtype: {original_dtype})")
                 except Exception as e:
                      logger.debug(f"Could not convert column '{col}' to numeric: {e}. Keeping original dtype.")

        return result_df

    def parse_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Parse online bioreactor data file."""
        file_path = Path(file_path)
        logger.info(f"Parsing online data: {file_path}")
        try:
            df = pd.read_csv(
                file_path,
                sep=self.separator,
                decimal=self.decimal,
                skiprows=self.skiprows,
                encoding='utf-8', # Consider making this configurable
                low_memory=False
            )
            df = self._rename_columns(df)
            df = self._process_dataframe(df)
            logger.info(f"Parsed online data: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except FileNotFoundError:
            logger.error(f"Online data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing online data file {file_path}: {e}")
            raise


class OfflineMeasurementsParser:
    """Parser for offline measurement data files."""

    def __init__(self, config: dict):
        self.config = config
        self.parser_config = config.get('parsing', {}).get('offline', {})
        self.col_map_config = config.get('column_mapping', {})
        self.decimal = self.parser_config.get('decimal_separator', '.')
        self.separator = self.parser_config.get('column_separator', ',')
        self.date_format = self.parser_config.get('date_format', 'guess')
        self.sheet_name = self.parser_config.get('sheet_name', 0)


    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies renaming based on config column_mapping."""
        mapping = {}
        # The mapping in config now has StandardName: OriginalName(s)
        # We need OriginalName: StandardName for rename
        for std_name, orig_names in self.col_map_config.items():
             if isinstance(orig_names, str): # Simplified case
                  if orig_names in df.columns:
                       mapping[orig_names] = std_name
             elif isinstance(orig_names, list): # Handle list case if needed later
                  for orig_name in orig_names:
                       if orig_name in df.columns:
                           mapping[orig_name] = std_name
                           break # Use first match from list
        renamed_df = df.rename(columns=mapping)
        logger.info(f"Renamed {len(mapping)} offline columns based on config.")
        return renamed_df

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic type conversions for offline data."""
        result_df = df.copy()

        # --- Process Time Column FIRST ---
        # We *need* 'process_time' in hours for alignment.
        # Check if it was directly mapped or needs calculation.
        if 'process_time' not in result_df.columns:
            # Try calculating from seconds column if available
            if 'batch_time_sec' in result_df.columns:
                logger.info("Calculating 'process_time' (hours) from 'batch_time_sec' in offline data.")
                result_df['process_time'] = pd.to_numeric(result_df['batch_time_sec'], errors='coerce') / 3600.0
                # Optional: subtract start time if not starting at 0
                if not result_df.empty:
                    first_valid_time = result_df['process_time'].dropna().iloc[0] if result_df['process_time'].notna().any() else 0
                    result_df['process_time'] -= first_valid_time
            else:
                logger.error("Offline data requires a 'process_time' column (in hours) or a mappable 'batch_time_sec' column.")
                # Optionally raise error or return df as is
                # raise ValueError("Missing time information for offline data alignment.")

        # Ensure the final 'process_time' is numeric
        if 'process_time' in result_df.columns:
             result_df['process_time'] = pd.to_numeric(result_df['process_time'], errors='coerce')
             if result_df['process_time'].isna().any():
                 logger.warning("NaNs found in offline 'process_time' column after processing.")

        # Attempt numeric conversion for all other columns
        for col in result_df.columns:
            if col != 'process_time' and not pd.api.types.is_numeric_dtype(result_df[col]):
                 original_dtype = result_df[col].dtype
                 try:
                     series_to_convert = result_df[col]
                     # Handle potential different decimal separators if needed based on config
                     if isinstance(series_to_convert.iloc[0], str) and self.decimal != '.':
                          series_to_convert = series_to_convert.str.replace(self.decimal, '.', regex=False)
                     result_df[col] = pd.to_numeric(series_to_convert, errors='coerce')
                     logger.debug(f"Attempted numeric conversion for offline column '{col}' (original dtype: {original_dtype})")
                 except Exception as e:
                     logger.debug(f"Could not convert offline column {col} to numeric: {e}. Keeping original dtype.")

        return result_df


    def parse_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Parse offline measurement file."""
        file_path = Path(file_path)
        logger.info(f"Parsing offline data: {file_path}")
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(
                    file_path,
                    sep=self.separator,
                    decimal=self.decimal,
                    skipinitialspace=True
                )
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, sheet_name=self.sheet_name)
            else:
                raise ValueError(f"Unsupported offline file format: {file_path.suffix}")

            df = self._rename_columns(df)
            df = self._process_dataframe(df) # Ensure process_time exists/is numeric
            logger.info(f"Parsed offline data: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except FileNotFoundError:
            logger.error(f"Offline data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing offline data file {file_path}: {e}")
            raise