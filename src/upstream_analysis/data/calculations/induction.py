"""Induction tracking: binary indicator and time-since-induction timer."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_induction_indicator(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add a binary indicator for the induction phase.

    Detects the first occurrence of a configured induced phase (e.g.
    'induced Fedbatch') and creates a column that is 0 before induction and 1
    from induction start onwards.

    Args:
        df: DataFrame with 'Process phase [-]'.
        config: Merged configuration dict. Induced phases are read from
            ``config['processing']['induced_phases']``.

    Returns:
        DataFrame with the added column 'Is induced [-]'.
    """
    logger.info("Adding induction indicator column...")
    df_out = df.copy()
    df_out["Is induced [-]"] = 0  # Initialize to 0 (not induced)

    phase_col = "Process phase [-]"
    if phase_col not in df_out.columns:
        logger.warning(
            f"Cannot add induction indicator: '{phase_col}' column not found."
        )
        return df_out

    # Get phase names from the *merged* config (defaults can be overridden)
    induced_phases = config.get("processing", {}).get("induced_phases", [])
    if not induced_phases:
        logger.warning(
            "No 'induced_phases' defined in config. 'Is induced [-]' column will remain 0."
        )
        return df_out

    logger.info(f"Identifying induction start based on phases: {induced_phases}")

    # Create mask for rows belonging to induced phases
    is_induced_phase_mask = df_out[phase_col].isin(induced_phases)

    # Find the first index where an induced phase occurs
    induction_start_index = is_induced_phase_mask.idxmax()

    # Check if any induced phase was actually found
    # idxmax returns the first index if *no* True is found, so check the value at that index
    if is_induced_phase_mask.any():  # Check if at least one True exists
        logger.info(
            f"Induction start detected at index {induction_start_index} "
            f"(Process time [h]: {df_out.loc[induction_start_index, 'Process time [h]']:.2f}h)"
        )
        # Set indicator to 1 from the start index onwards
        df_out.loc[induction_start_index:, "Is induced [-]"] = 1
    else:
        logger.info(
            "No specified induction phases found in the data. 'Is induced [-]' column remains 0."
        )

    return df_out


def add_induction_timer(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time elapsed since induction start.

    Args:
        df: DataFrame with 'Is induced [-]' (from :func:`add_induction_indicator`)
            and 'Process time [h]'.

    Returns:
        DataFrame with the added column 'Induction time [h]'.
    """
    logger.info("Adding induction timer column...")
    df_out = df.copy()

    # Check if induction indicator exists
    if "Is induced [-]" not in df_out.columns:
        logger.warning(
            "Cannot add induction timer: 'Is induced [-]' column not found. Run add_induction_indicator first."
        )
        df_out["Induction time [h]"] = 0.0
        return df_out

    # Check if process time column exists
    time_col = "Process time [h]"
    if time_col not in df_out.columns:
        logger.warning(f"Cannot add induction timer: '{time_col}' column not found.")
        df_out["Induction time [h]"] = 0.0
        return df_out

    # Find the induction start time (first row where Is induced [-] == 1)
    induced_mask = df_out["Is induced [-]"] == 1
    if not induced_mask.any():
        logger.info("No induction detected. 'Induction time [h]' set to 0.")
        df_out["Induction time [h]"] = 0.0
        return df_out

    induction_start_index = induced_mask.idxmax()
    induction_start_time = df_out.loc[induction_start_index, time_col]

    # Calculate induction timer
    # Before induction: 0
    # After induction start: Process time - induction start time
    df_out["Induction time [h]"] = 0.0
    df_out.loc[df_out.index >= induction_start_index, "Induction time [h]"] = (
        df_out.loc[df_out.index >= induction_start_index, time_col]
        - induction_start_time
    )

    logger.info(
        f"Added 'Induction time [h]' column (induction started at {induction_start_time:.2f}h). "
        f"Max timer value: {df_out['Induction time [h]'].max():.2f}h"
    )

    return df_out
