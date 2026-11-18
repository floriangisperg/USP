# src/upstream_analysis/core/__init__.py
"""
Core module for upstream processing analysis.

This module contains fundamental constants and utilities used throughout
the USP package.
"""

from . import exceptions, validation

# Physical constants; Gas compositions; Molar masses; Unit conversions;
# Process defaults; Signal processing defaults
from .constants import (
    ACETATE_MOLAR_MASS_G_PER_C_MOL,
    BIOMASS_MOLAR_MASS_G_PER_C_MOL,
    CO2_IN_AIR_PERCENT,
    DEFAULT_ALIGNMENT_TOLERANCE_HOURS,
    DEFAULT_FEED_GLUCOSE_G_L,
    DEFAULT_INITIAL_GLUCOSE_G_L,
    DEFAULT_INITIAL_VOLUME_ML,
    DEFAULT_MEDIAN_FILTER_THRESHOLD,
    DEFAULT_MEDIAN_FILTER_WINDOW,
    GLUCOSE_MOLAR_MASS_G_PER_C_MOL,
    MAD_TO_SIGMA_SCALE,
    MIN_PER_HOUR,
    ML_PER_L,
    MMOL_PER_MOL,
    MOLAR_VOLUME_AIR_L_PER_MOL,
    O2_IN_AIR_PERCENT,
    PERCENT,
    SEC_PER_HOUR,
)
from .exceptions import USPCalculationError, USPConfigError, USPDataError, USPError
from .validation import (
    ConfigValidationError,
    DataFrameValidationError,
    ValidationError,
    coerce_float,
    get_nested_config,
    require_columns,
    validate_config_structure,
    validate_dataframe_not_empty,
    validate_numeric_series,
    validate_positive_numeric,
)

__all__ = [
    # Physical constants
    "MOLAR_VOLUME_AIR_L_PER_MOL",
    "MAD_TO_SIGMA_SCALE",
    # Gas compositions
    "O2_IN_AIR_PERCENT",
    "CO2_IN_AIR_PERCENT",
    # Molar masses
    "BIOMASS_MOLAR_MASS_G_PER_C_MOL",
    "GLUCOSE_MOLAR_MASS_G_PER_C_MOL",
    "ACETATE_MOLAR_MASS_G_PER_C_MOL",
    # Unit conversions
    "ML_PER_L",
    "SEC_PER_HOUR",
    "MIN_PER_HOUR",
    "PERCENT",
    "MMOL_PER_MOL",
    # Process defaults
    "DEFAULT_INITIAL_VOLUME_ML",
    "DEFAULT_FEED_GLUCOSE_G_L",
    "DEFAULT_INITIAL_GLUCOSE_G_L",
    # Signal processing defaults
    "DEFAULT_MEDIAN_FILTER_WINDOW",
    "DEFAULT_MEDIAN_FILTER_THRESHOLD",
    "DEFAULT_ALIGNMENT_TOLERANCE_HOURS",
    # Validation modules
    "validation",
    "exceptions",
    # Exceptions
    "USPError",
    "USPConfigError",
    "USPDataError",
    "USPCalculationError",
    # Validation functions
    "ValidationError",
    "DataFrameValidationError",
    "ConfigValidationError",
    "require_columns",
    "validate_dataframe_not_empty",
    "validate_numeric_series",
    "coerce_float",
    "validate_config_structure",
    "validate_positive_numeric",
    "get_nested_config",
]
