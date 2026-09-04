# src/upstream_analysis/core/constants.py
"""
Constants for upstream processing analysis.

This module contains physical constants, conversion factors, and default values
used throughout the USP package for E. coli fermentation data analysis.
"""

# =============================================================================
# Physical Constants
# =============================================================================

# Molar volume of air at standard temperature and pressure (STP)
MOLAR_VOLUME_AIR_L_PER_MOL = 22.414  # L/mol at STP

# Scale factor to convert Median Absolute Deviation (MAD) to standard deviation
# For normally distributed data, sigma ≈ 1.4826 * MAD
MAD_TO_SIGMA_SCALE = 1.4826

# =============================================================================
# Gas Compositions (percentage fractions)
# =============================================================================

# Oxygen fraction in dry air
O2_IN_AIR_PERCENT = 21.0

# Carbon dioxide fraction in dry air
CO2_IN_AIR_PERCENT = 0.04

# =============================================================================
# Molar Masses (on C-mol basis)
# =============================================================================

# Biomass: CH1.77N0.24O0.49 (molar mass per C-mol)
BIOMASS_MOLAR_MASS_G_PER_C_MOL = 26.565

# Glucose: C6H12O6 -> CH2O on C-mol basis (30.03 g/C-mol)
GLUCOSE_MOLAR_MASS_G_PER_C_MOL = 30.03

# Acetate: C2H4O2 -> CH2O on C-mol basis (29.5 g/C-mol)
ACETATE_MOLAR_MASS_G_PER_C_MOL = 29.5

# =============================================================================
# Unit Conversion Factors
# =============================================================================

# Volume conversions
ML_PER_L = 1000.0

# Time conversions
SEC_PER_HOUR = 3600.0
MIN_PER_HOUR = 60.0

# Percentage conversions
PERCENT = 100.0

# Amount conversions
MMOL_PER_MOL = 1000.0

# =============================================================================
# Process Defaults
# =============================================================================

# Reactor setup defaults
DEFAULT_INITIAL_VOLUME_ML = 1100.0

# Feed composition (matches media config)
DEFAULT_FEED_GLUCOSE_G_L = 440.0

# Initial batch glucose concentration
DEFAULT_INITIAL_GLUCOSE_G_L = 22.0

# =============================================================================
# Signal Processing Defaults
# =============================================================================

# Median filter parameters for spike detection
DEFAULT_MEDIAN_FILTER_WINDOW = 10
DEFAULT_MEDIAN_FILTER_THRESHOLD = 10

# Temporal alignment tolerance for online/offline data matching
DEFAULT_ALIGNMENT_TOLERANCE_HOURS = 0.02

# Induction timing: the SCADA phase transition into the induced phase can lead
# the physical inducer addition (manual IPTG bolus) by minutes to over an hour.
# When a reference induction time is configured, warn if it differs from the
# phase-flip time by more than this (hours); the correction is applied regardless.
DEFAULT_INDUCTION_TIME_TOLERANCE_H = 0.25
