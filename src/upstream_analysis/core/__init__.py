# src/upstream_analysis/core/__init__.py
"""
Core module for upstream processing analysis.

This module contains fundamental constants and utilities used throughout
the USP package.
"""

from .constants import (
    # Physical constants
    MOLAR_VOLUME_AIR_L_PER_MOL,
    MAD_TO_SIGMA_SCALE,
    # Gas compositions
    O2_IN_AIR_PERCENT,
    CO2_IN_AIR_PERCENT,
    # Molar masses
    BIOMASS_MOLAR_MASS_G_PER_C_MOL,
    GLUCOSE_MOLAR_MASS_G_PER_C_MOL,
    ACETATE_MOLAR_MASS_G_PER_C_MOL,
    # Unit conversions
    ML_PER_L,
    SEC_PER_HOUR,
    MIN_PER_HOUR,
    PERCENT,
    MMOL_PER_MOL,
    # Process defaults
    DEFAULT_INITIAL_VOLUME_ML,
    DEFAULT_FEED_GLUCOSE_G_L,
    DEFAULT_INITIAL_GLUCOSE_G_L,
    # Signal processing defaults
    DEFAULT_MEDIAN_FILTER_WINDOW,
    DEFAULT_MEDIAN_FILTER_THRESHOLD,
    DEFAULT_ALIGNMENT_TOLERANCE_HOURS,
)
from . import validation
from . import exceptions
from .exceptions import USPError, USPConfigError, USPDataError, USPCalculationError
from .validation import (
    ValidationError,
    DataFrameValidationError,
    ConfigValidationError,
    require_columns,
    validate_dataframe_not_empty,
    validate_numeric_series,
    coerce_float,
    validate_config_structure,
    validate_positive_numeric,
    get_nested_config,
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
