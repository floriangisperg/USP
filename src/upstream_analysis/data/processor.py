"""
Fermentation data processing module for USP package.

This module provides :class:`FermentationDataProcessor`, the public entry point
for calculating derived bioprocess parameters from cleaned and aligned
fermentation data. It is a thin orchestrator (facade): each calculation is
implemented as a focused module-level function in the :mod:`.calculations`
subpackage (feed, base/volume, induction, protein, gas exchange, carbon
balance, rates/yields). This class preserves the historical API —
``process_data(df)`` and the per-step ``_<name>(df)`` methods used by the tests
— by delegating to those functions in the fixed sequential order.

The processing order is critical and maintained by ``process_data``:
1. Feed calculation (from balance data)
2. Base titrant volume calculation
3. Reactor volume calculation
4. Induction detection and timing
5. Protein parameter calculation
6. Inlet gas composition
7. Off-gas rate calculation (CER, OUR, RQ)
8. Specific off-gas rates (qO2, qCO2)
9. Carbon balance analysis
10. Growth and production rates

Example:
    >>> from upstream_analysis import FermentationDataProcessor
    >>> processor = FermentationDataProcessor(config)
    >>> processed_data = processor.process_data(cleaned_aligned_df)

Scientific Background:
    Calculations are based on standard bioprocess engineering principles for
    E. coli fermentation in fed-batch mode. Gas exchange rates follow mass
    balance principles, and the carbon balance enables assessment of metabolic
    state and product yield.
"""

import logging

import pandas as pd

from ..core.exceptions import USPConfigError
from ..core.validation import require_columns, validate_dataframe_not_empty
from .calculations import (
    carbon_balance,
    feed,
    gas,
    induction,
    protein,
    rates_yields,
    volume,
)
from .calculations._config import get_processing_config

logger = logging.getLogger(__name__)


class FermentationDataProcessor:
    """Processes cleaned and aligned fermentation data.

    A facade over the focused calculation modules in
    :mod:`.calculations`. Each ``_<name>`` method delegates to the
    corresponding module function, passing ``self.config``. The methods are
    kept (rather than inlined into ``process_data``) because the unit tests
    exercise them individually and several build prerequisite columns in a
    chain.

    Attributes:
        config: The merged configuration dict.
        processing_config: The resolved ``processing_parameters`` section.
    """

    def __init__(self, config: dict):
        self.config = config
        # Resolve the processing_parameters section once (supports old and new
        # config layouts; see calculations._config.get_processing_config).
        self.processing_config = get_processing_config(config)
        self._validate_config()

    def _validate_config(self) -> None:
        """Fail fast on clearly invalid config values.

        Only validates values that *are* present; missing optional config is
        allowed so a processor can still be constructed and each calculation can
        decide how to handle absent parameters.
        """
        reactor_setup = self.processing_config.get("reactor_setup", {}) or {}
        if isinstance(reactor_setup, dict):
            initial_volume = reactor_setup.get("initial_volume_ml", None)
            if initial_volume is not None and initial_volume <= 0:
                raise USPConfigError(
                    f"Invalid initial_volume_ml: must be positive, got {initial_volume}."
                )

    # --- Calculation delegators (each preserves the df-in/df-out contract) ---

    def _calculate_feed(self, df: pd.DataFrame) -> pd.DataFrame:
        return feed.calculate_feed(df, self.config)

    def _calculate_base_flux_and_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        return volume.calculate_base_flux_and_volume(df, self.config)

    def _calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        return volume.calculate_volume(df, self.config)

    def _add_induction_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        return induction.add_induction_indicator(df, self.config)

    def _add_induction_timer(self, df: pd.DataFrame) -> pd.DataFrame:
        return induction.add_induction_timer(df)

    def _calculate_protein_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        return protein.calculate_protein_parameters(df)

    def _calculate_inlet_gas_composition(self, df: pd.DataFrame) -> pd.DataFrame:
        return gas.calculate_inlet_gas_composition(df, self.config)

    def _calculate_offgas_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        return gas.calculate_offgas_rates(df, self.config)

    def _calculate_specific_offgas_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        return gas.calculate_specific_offgas_rates(df, self.config)

    def _calculate_carbon_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        return carbon_balance.calculate_carbon_balance(df, self.config)

    def _calculate_rates_and_yields(self, df: pd.DataFrame) -> pd.DataFrame:
        return rates_yields.calculate_rates_and_yields(df, self.config)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full calculation pipeline in the required order.

        Args:
            df: Input DataFrame with cleaned and aligned data. Must contain at
                least ``Process time [h]`` and ``Process phase [-]``.

        Returns:
            Processed DataFrame with calculated parameters. Returns an empty
            DataFrame if the input is ``None``/empty or fails validation
            (fail-soft, matching the historical contract).
        """
        # Validate input DataFrame
        if df is None or df.empty:
            logger.error("Processor: Input DataFrame is None or empty.")
            return pd.DataFrame()  # Keep existing fail-soft behavior

        # Validate required columns for processing
        try:
            required_cols = ["Process time [h]", "Process phase [-]"]
            validate_dataframe_not_empty(df, "data processing")
            require_columns(df, required_cols, "data processing")
        except Exception as e:
            logger.error(f"Processor: Input validation failed: {e}")
            return pd.DataFrame()  # Keep existing fail-soft behavior

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
