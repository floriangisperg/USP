"""Focused calculation submodules for the fermentation-data processing pipeline.

Each module hosts the df-in/df-out calculation functions; the public
:class:`upstream_analysis.FermentationDataProcessor` (in ``data/processor.py``)
is a thin facade that delegates to these in the fixed sequential order.
"""

from .carbon_balance import calculate_carbon_balance
from .feed import calculate_feed
from .gas import (
    calculate_inlet_gas_composition,
    calculate_offgas_rates,
    calculate_specific_offgas_rates,
)
from .induction import add_induction_indicator, add_induction_timer
from .protein import calculate_protein_parameters
from .rates_yields import calculate_rates_and_yields
from .volume import calculate_base_flux_and_volume, calculate_volume

__all__ = [
    "calculate_feed",
    "calculate_base_flux_and_volume",
    "calculate_volume",
    "add_induction_indicator",
    "add_induction_timer",
    "calculate_protein_parameters",
    "calculate_inlet_gas_composition",
    "calculate_offgas_rates",
    "calculate_specific_offgas_rates",
    "calculate_carbon_balance",
    "calculate_rates_and_yields",
]
