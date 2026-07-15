"""Protein-derived parameters: total, soluble/IB fractions, and yields vs biomass."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_protein_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived protein parameters for recombinant-expression analysis.

    Computes total protein, fractionation (soluble vs. inclusion bodies), and
    yields relative to biomass. Handles incomplete data by creating missing
    columns with appropriate defaults.

    Args:
        df: DataFrame with optional 'Protein [g/L]', 'Protein soluble [g/L]',
            'Protein IB [g/L]', and 'Biomass [g/L]' columns.

    Returns:
        DataFrame with added columns:
            - 'Protein total calculated [g/L]': sum of soluble + IB (if no total
              measured); masked to sample rows.
            - 'Soluble fraction [%]', 'IB fraction [%]'
            - 'Protein soluble yield [g/g]', 'Protein IB yield [g/g]',
              'Protein total yield [g/g]' (if biomass available)
    """
    logger.info("Calculating protein-derived parameters...")
    df_out = df.copy()

    # Check what protein columns are available in the ORIGINAL data
    orig_has_soluble = "Protein soluble [g/L]" in df_out.columns
    orig_has_ib = "Protein IB [g/L]" in df_out.columns
    orig_has_total = "Protein [g/L]" in df_out.columns

    logger.info(
        f"Original protein data - Soluble: {orig_has_soluble}, IB: {orig_has_ib}, Total: {orig_has_total}"
    )

    # Always create both protein columns, filling missing ones with NaNs
    if not orig_has_soluble:
        df_out["Protein soluble [g/L]"] = np.nan
        logger.info(
            "Created soluble protein column (filled with NaNs - not present in original data)."
        )
    if not orig_has_ib:
        df_out["Protein IB [g/L]"] = np.nan
        logger.info(
            "Created IB protein column (filled with NaNs - not present in original data)."
        )

    # Calculate total protein based on ORIGINAL data availability
    # Always create calculated total to show sparse measurement points
    if orig_has_soluble and orig_has_ib:
        # Both soluble and IB present in original data
        df_out["Protein total calculated [g/L]"] = (
            df_out["Protein soluble [g/L]"] + df_out["Protein IB [g/L]"]
        )
        logger.info("Calculated total protein from soluble + IB fractions.")
    elif orig_has_soluble and not orig_has_ib:
        # Only soluble present in original data
        df_out["Protein total calculated [g/L]"] = df_out["Protein soluble [g/L]"]
        logger.info("Using soluble protein as total protein (no IB data available).")
    elif orig_has_ib and not orig_has_soluble:
        # Only IB present in original data
        df_out["Protein total calculated [g/L]"] = df_out["Protein IB [g/L]"]
        logger.info("Using IB protein as total protein (no soluble data available).")
    else:
        # Neither soluble nor IB present - no protein data to calculate
        df_out["Protein total calculated [g/L]"] = np.nan
        logger.info("No protein data available in original data.")

    # CRITICAL FIX: Mask the calculated total protein to only show values at sample points
    # This prevents the plot from showing continuous data between actual measurements
    if (
        "Protein total calculated [g/L]" in df_out.columns
        and "is_sample" in df_out.columns
    ):
        # Create a mask for sample rows
        is_sample_mask = df_out["is_sample"].fillna(False).astype(bool)
        # Set non-sample rows to NaN
        df_out.loc[~is_sample_mask, "Protein total calculated [g/L]"] = np.nan
        logger.info("Masked Protein total calculated [g/L] to show only sample points.")

    # Calculate soluble fraction (%) if soluble protein is available in original data
    if orig_has_soluble:
        if orig_has_total:
            df_out["Soluble fraction [%]"] = (
                df_out["Protein soluble [g/L]"]
                / df_out["Protein [g/L]"].replace(0, np.nan)
            ) * 100
            logger.info("Calculated soluble fraction from measured total protein.")
        elif "Protein total calculated [g/L]" in df_out.columns:
            df_out["Soluble fraction [%]"] = (
                df_out["Protein soluble [g/L]"]
                / df_out["Protein total calculated [g/L]"].replace(0, np.nan)
            ) * 100
            logger.info("Calculated soluble fraction from calculated total protein.")

    # Calculate IB fraction (%) if IB protein is available in original data
    if orig_has_ib:
        if orig_has_total:
            df_out["IB fraction [%]"] = (
                df_out["Protein IB [g/L]"] / df_out["Protein [g/L]"].replace(0, np.nan)
            ) * 100
            logger.info("Calculated IB fraction from measured total protein.")
        elif "Protein total calculated [g/L]" in df_out.columns:
            df_out["IB fraction [%]"] = (
                df_out["Protein IB [g/L]"]
                / df_out["Protein total calculated [g/L]"].replace(0, np.nan)
            ) * 100
            logger.info("Calculated IB fraction from calculated total protein.")

    # Calculate protein yields if biomass is available
    if "Biomass [g/L]" in df_out.columns:
        if orig_has_soluble:
            df_out["Protein soluble yield [g/g]"] = df_out[
                "Protein soluble [g/L]"
            ] / df_out["Biomass [g/L]"].replace(0, np.nan)
            logger.info("Calculated soluble protein yield (g protein / g biomass).")
        if orig_has_ib:
            df_out["Protein IB yield [g/g]"] = df_out["Protein IB [g/L]"] / df_out[
                "Biomass [g/L]"
            ].replace(0, np.nan)
            logger.info("Calculated IB protein yield (g protein / g biomass).")
        if orig_has_total:
            df_out["Protein total yield [g/g]"] = df_out["Protein [g/L]"] / df_out[
                "Biomass [g/L]"
            ].replace(0, np.nan)
            logger.info("Calculated total protein yield (g protein / g biomass).")
        elif "Protein total calculated [g/L]" in df_out.columns:
            df_out["Protein total yield [g/g]"] = df_out[
                "Protein total calculated [g/L]"
            ] / df_out["Biomass [g/L]"].replace(0, np.nan)
            logger.info("Calculated total protein yield from calculated total protein.")

    return df_out
