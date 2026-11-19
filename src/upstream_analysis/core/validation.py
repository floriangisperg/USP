"""Input validation utilities for USP data processing."""

import logging
from typing import Any, List

import pandas as pd

from .exceptions import USPConfigError

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class DataFrameValidationError(ValidationError):
    """Raised when DataFrame validation fails."""

    pass


class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""

    pass


def require_columns(
    df: pd.DataFrame, required_cols: List[str], context: str = ""
) -> pd.DataFrame:
    """
    Validate that required columns exist in DataFrame.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        context: Description of what's being validated (for error messages)

    Returns:
        The input DataFrame if validation passes

    Raises:
        DataFrameValidationError: If required columns are missing
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        msg = (
            f"Missing required columns {missing} for {context}"
            if context
            else f"Missing required columns: {missing}"
        )
        logger.error(msg)
        raise DataFrameValidationError(msg)
    return df


def validate_dataframe_not_empty(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """
    Validate DataFrame is not empty.

    Args:
        df: DataFrame to validate
        context: Description of what's being validated (for error messages)

    Returns:
        The input DataFrame if validation passes

    Raises:
        DataFrameValidationError: If DataFrame is empty or None
    """
    if df is None or df.empty:
        msg = (
            "DataFrame is empty or None"
            if not context
            else f"DataFrame for {context} is empty or None"
        )
        logger.error(msg)
        raise DataFrameValidationError(msg)
    return df


def validate_numeric_series(
    series: pd.Series, name: str, allow_empty: bool = False
) -> pd.Series:
    """
    Validate series is numeric and contains valid data.

    Args:
        series: Series to validate
        name: Name of the series for error messages
        allow_empty: Whether to allow all-NaN series

    Returns:
        The validated series

    Raises:
        DataFrameValidationError: If series is not numeric
    """
    if not pd.api.types.is_numeric_dtype(series):
        msg = f"Series '{name}' is not numeric (dtype: {series.dtype})"
        logger.error(msg)
        raise DataFrameValidationError(msg)

    if not allow_empty and series.isna().all():
        logger.warning(f"Series '{name}' contains only NaN values")
    return series


def coerce_float(value: Any, default: float, param_name: str = "") -> float:
    """
    Safely coerce value to float with fallback default.

    Args:
        value: Value to coerce
        default: Default value if coercion fails
        param_name: Parameter name for error logging

    Returns:
        Coerced float value or default
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        if param_name:
            logger.debug(
                f"Could not coerce '{param_name}' to float, using default: {default}"
            )
        return default


def validate_config_structure(config: dict) -> None:
    """
    Validate basic config structure.

    Args:
        config: Configuration dictionary to validate

    Raises:
        USPConfigError: If required sections are missing
    """
    if not isinstance(config, dict):
        raise USPConfigError(
            f"Config must be a dictionary, got {type(config).__name__}"
        )

    required_sections = ["parsing", "column_mapping", "processing"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise USPConfigError(f"Config missing required sections: {missing}")

    # Warn about unknown top-level keys (catch typos)
    known_keys = {
        "parsing",
        "column_mapping",
        "cleaning",
        "alignment",
        "processing",
        "analysis",
        "visualization",
        "metadata",
    }
    unknown = set(config.keys()) - known_keys
    if unknown:
        logger.warning(f"Unknown config sections (possible typos): {unknown}")


def validate_positive_numeric(
    value: Any, param_name: str, allow_zero: bool = False
) -> float:
    """
    Validate that a value is a positive numeric.

    Args:
        value: Value to validate
        param_name: Parameter name for error messages
        allow_zero: Whether to allow zero values

    Returns:
        The validated float value

    Raises:
        USPConfigError: If value is not positive (or non-zero when allow_zero=False)
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        raise USPConfigError(f"Parameter '{param_name}' must be numeric, got: {value}")

    if float_value < 0:
        raise USPConfigError(
            f"Parameter '{param_name}' must be non-negative, got: {float_value}"
        )

    if not allow_zero and float_value == 0:
        raise USPConfigError(
            f"Parameter '{param_name}' must be positive, got: {float_value}"
        )

    return float_value


def get_nested_config(
    config: dict, keys: List[str], default: Any = None, context: str = ""
) -> Any:
    """
    Safely get nested config value with path tracking for error messages.

    Args:
        config: Configuration dictionary
        keys: List of keys for nested access
        default: Default value if path doesn't exist
        context: Context string for error messages

    Returns:
        Value at nested path or default

    Raises:
        USPConfigError: If path is invalid and no default provided
    """
    value = config
    path_str = "config"
    for key in keys:
        path_str += f"['{key}']"
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif default is not None:
            logger.debug(
                f"Optional config parameter not found at path: {path_str}. Using default: {default}"
            )
            return default
        else:
            msg = f"Missing required configuration parameter at path: {path_str}"
            if context:
                msg = f"{context}: {msg}"
            raise USPConfigError(msg)
    return value


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object-dtype columns whose every non-null value parses as a number.

    Alignment can leave offline-derived columns as object dtype holding string
    floats (this is why protein calculations call ``infer_objects`` before
    arithmetic). Downstream consumers — CSV export, consolidation — need real
    numeric dtypes. Columns that contain genuinely non-numeric values (phase
    names, timestamps) are left untouched. Comma decimals are accepted.

    Args:
        df: DataFrame to coerce (not modified in place).

    Returns:
        DataFrame with numeric-convertible object columns converted.
    """
    df_out = df.copy()
    for col in df_out.columns:
        series = df_out[col]
        if series.dtype != object:
            continue
        converted = pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
        # Convert only when every non-null value became numeric (NaN maps to
        # NaN in ``converted``, so the masks align).
        if (converted.notna() == series.notna()).all() and converted.notna().any():
            df_out[col] = converted
    return df_out
