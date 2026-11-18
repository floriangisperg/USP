# src/upstream_analysis/data/parser.py
import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


class BioreactorDataParser:
    """Parser for online bioreactor data files."""

    def __init__(self, config: dict):
        self.config = config
        self.parser_config = config.get("parsing", {}).get("bioreactor", {})
        self.col_map_config = config.get("column_mapping", {})
        self.decimal = self.parser_config.get("decimal_separator", ",")
        self.separator = self.parser_config.get("column_separator", ";")
        self.date_format = self.parser_config.get("date_format", None)
        self.skiprows = self.parser_config.get("skiprows", 1)
        self.encoding = self.parser_config.get(
            "encoding", "cp1252"
        )  # Defaulting to cp1252 based on Â°

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies renaming based on the config."""
        mapping = self.col_map_config  # Should be {'Raw':'Standard'} from merged config

        # Find which keys from mapping are actually in df.columns
        present_keys = {k: v for k, v in mapping.items() if k in df.columns}
        missing_raw_keys = [k for k in mapping.keys() if k not in df.columns]
        if missing_raw_keys:
            logger.warning(
                f"Raw headers from config mapping *NOT* found in data: {missing_raw_keys}"
            )

        # Perform rename ONLY with keys actually present
        if present_keys:
            renamed_df = df.rename(columns=present_keys)
            logger.info(f"Renamed {len(present_keys)} online columns based on config.")
            return renamed_df
        else:
            logger.warning(
                "No matching keys found between config mapping and data "
                "columns. No columns renamed."
            )
            return df  # Return original df if no keys matched

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic type conversions for online data."""
        result_df = df.copy()
        date_format_to_use = (
            self.date_format
            if self.date_format and self.date_format.lower() != "guess"
            else None
        )

        # Datetime columns (use standardized names)
        time_cols = ["date_time_utc", "date_time_local"]
        for col in time_cols:
            if col in result_df.columns:
                try:
                    result_df[col] = pd.to_datetime(
                        result_df[col], format=date_format_to_use, errors="coerce"
                    )
                except ValueError:  # Try without format if specified one failed
                    logger.warning(
                        f"Format '{date_format_to_use}' failed for '{col}'. "
                        "Attempting default parsing."
                    )
                    result_df[col] = pd.to_datetime(result_df[col], errors="coerce")
                except Exception as e:
                    logger.error(f"Error parsing datetime column {col}: {e}")
                    result_df[col] = pd.NaT  # Assign NaT on other errors

        # Process phase - handle both old and new column names
        phase_col_names = ["process_phase", "Process phase [-]"]
        for phase_col in phase_col_names:
            if phase_col in result_df.columns:
                result_df[phase_col] = result_df[phase_col].astype(str).str.strip()
                result_df[phase_col] = result_df[phase_col].str.replace(
                    r"([a-z])([A-Z])", r"\1 \2", regex=True
                )
                # Replace 'nan' strings with actual NaN for proper handling
                result_df[phase_col] = result_df[phase_col].replace("nan", pd.NA)

        # Attempt numeric conversion for all other columns (excluding phase columns)
        exclude_cols = set(time_cols) | set(phase_col_names)
        numeric_cols = [c for c in result_df.columns if c not in exclude_cols]
        for col in numeric_cols:
            # Check if it's not already numeric/datetime
            if not pd.api.types.is_numeric_dtype(
                result_df[col]
            ) and not pd.api.types.is_datetime64_any_dtype(result_df[col]):
                original_dtype = result_df[col].dtype
                try:
                    # Handle potential non-numeric chars and decimal separator
                    series_to_convert = result_df[col]
                    if isinstance(series_to_convert.iloc[0], str):
                        series_to_convert = series_to_convert.str.replace(
                            self.decimal, ".", regex=False
                        )

                    result_df[col] = pd.to_numeric(series_to_convert, errors="coerce")
                    logger.debug(
                        f"Attempted numeric conversion for column '{col}' "
                        f"(original dtype: {original_dtype})"
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not convert column '{col}' to numeric: {e}. Keeping original dtype."
                    )

        return result_df

    def parse_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Parse online bioreactor data file (CSV or Excel)."""
        file_path = Path(file_path)
        logger.info(f"Parsing online data: {file_path}")
        try:
            if file_path.suffix.lower() == ".csv":
                # Parse CSV file
                df = pd.read_csv(
                    file_path,
                    sep=self.separator,
                    decimal=self.decimal,
                    skiprows=self.skiprows,
                    encoding=self.encoding,
                    low_memory=False,
                )
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                # Parse Excel file - use first sheet
                df = pd.read_excel(file_path, sheet_name=0, header=self.skiprows)
                logger.info("Reading Excel online data file, using first sheet")
            else:
                raise ValueError(
                    f"Unsupported online data file format: {file_path.suffix}"
                )

            df = self._rename_columns(df)
            df = self._process_dataframe(df)
            logger.info(
                f"Parsed online data: {df.shape[0]} rows, {df.shape[1]} columns."
            )
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
        self.parser_config = config.get("parsing", {}).get("offline", {})
        self.col_map_config = config.get("column_mapping", {})
        self.decimal = self.parser_config.get("decimal_separator", ".")
        self.separator = self.parser_config.get("column_separator", ",")
        self.date_format = self.parser_config.get("date_format", "guess")
        self.sheet_name = self.parser_config.get("sheet_name", 0)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create mapping from raw headers to standard names
        correct_mapping = {}
        for raw_header, standard_name in self.col_map_config.items():
            # This assumes default_config.yaml has {'Raw Header': standard_name}
            if raw_header in df.columns:
                correct_mapping[raw_header] = standard_name

        renamed_df = df.rename(columns=correct_mapping)
        renamed_count = len(correct_mapping)
        logger.info(f"Renamed {renamed_count} offline columns based on config.")
        return renamed_df

    # Located inside class OfflineMeasurementsParser in src/upstream_analysis/data/parser.py
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic type conversions for offline data."""
        result_df = df.copy()

        # Ensure Process Time is Numeric
        if "process_time" in result_df.columns:
            result_df["process_time"] = pd.to_numeric(
                result_df["process_time"], errors="coerce"
            )
            if result_df["process_time"].isna().any():
                logger.warning(
                    "NaNs found in offline 'process_time' column after processing."
                )
        else:
            # Log a warning if it's still missing after the renaming attempt in _rename_columns
            logger.warning(
                "Offline 'process_time' column not found after renaming. Alignment might fail."
            )

        # Attempt numeric conversion for all other columns
        for col in result_df.columns:
            # Skip already processed time column and ensure it's not already numeric
            if col != "process_time" and not pd.api.types.is_numeric_dtype(
                result_df[col]
            ):
                try:
                    # Convert potential non-string types (like numbers read as
                    # objects) to string first
                    series_to_convert = result_df[col].astype(str)

                    # Handle potential different decimal separators if needed based on config
                    if self.decimal != ".":
                        series_to_convert = series_to_convert.str.replace(
                            self.decimal, ".", regex=False
                        )

                    # Attempt conversion, coercing errors leaves non-numeric as NaN
                    converted_series = pd.to_numeric(series_to_convert, errors="coerce")

                    # Only assign back if conversion was successful for at least some values
                    if converted_series.notna().any():
                        result_df[col] = converted_series

                except Exception as e:
                    logger.debug(
                        f"Could not convert offline column {col} to numeric: "
                        f"{e}. Keeping original dtype."
                    )

        return result_df

    def parse_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Parse offline measurement file."""
        file_path = Path(file_path)
        logger.info(f"Parsing offline data: {file_path}")
        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(
                    file_path,
                    sep=self.separator,
                    decimal=self.decimal,
                    skipinitialspace=True,
                )
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                # Add header=0 to be explicit, adjust if header is on different row
                df = pd.read_excel(file_path, sheet_name=self.sheet_name, header=0)
            else:
                raise ValueError(f"Unsupported offline file format: {file_path.suffix}")

            df = self._rename_columns(df)
            df = self._process_dataframe(df)  # Ensure process_time exists/is numeric
            logger.info(
                f"Parsed offline data: {df.shape[0]} rows, {df.shape[1]} columns."
            )
            return df
        except FileNotFoundError:
            logger.error(f"Offline data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing offline data file {file_path}: {e}")
            raise
