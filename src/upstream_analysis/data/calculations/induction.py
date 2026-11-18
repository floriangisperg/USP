"""Induction tracking: binary indicator and time-since-induction timer."""

import logging

import pandas as pd

from ...core.constants import DEFAULT_INDUCTION_TIME_TOLERANCE_H, MIN_PER_HOUR
from ...core.exceptions import USPConfigError, USPDataError

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


def _phase_flip_time(df: pd.DataFrame) -> float | None:
    """Return the process time of the phase-based induction start, or None."""
    if "Is induced [-]" not in df.columns or "Process time [h]" not in df.columns:
        return None
    induced_mask = df["Is induced [-]"] == 1
    if not induced_mask.any():
        return None
    return float(df.loc[induced_mask.idxmax(), "Process time [h]"])


def _read_reference_induction_time(path, sheet: str, label: str) -> float:
    """Read the operator-noted induction time from an evaluation workbook.

    The evaluation sheets (e.g. 'Time-Rownr') list event times (inoculation,
    sampling, induction) as rows labelled in the first column, with the decimal
    time in hours on the same time zero as the bioreactor 'Process time [h]'
    (verified: sample times match exactly).

    Raises:
        USPDataError: If the file/sheet cannot be read or no usable entry is found.
    """
    try:
        events = pd.read_excel(path, sheet_name=sheet, header=None)
    except Exception as e:
        raise USPDataError(
            f"Could not read induction reference file '{path}' (sheet '{sheet}'): {e}"
        ) from e

    for _, row in events.iterrows():
        if str(row.iloc[0]).strip().lower() == str(label).strip().lower():
            # Prefer the decimal 'Time' column; fall back to Hour + Minute/60.
            if pd.notna(row.iloc[3]):
                return float(row.iloc[3])
            if pd.notna(row.iloc[1]) and pd.notna(row.iloc[2]):
                return float(row.iloc[1]) + float(row.iloc[2]) / MIN_PER_HOUR
            break

    raise USPDataError(
        f"No '{label}' entry with a usable time found in '{path}' (sheet '{sheet}')."
    )


def correct_induction_time(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Check the phase-based induction start against a reference and correct it.

    The process-control phase transition into the induced phase can *lead* the
    physical inducer addition: the recipe switches the phase (and the feed
    profile) automatically, while IPTG is dosed manually afterwards. Observed
    offsets range from minutes (flip shortly before the operator's bolus) to
    more than an hour (flip before the end-of-fedbatch sample was even drawn).
    This step compares the phase-flip time with a reference induction time —
    the operator-verified moment the inducer entered the reactor — warns when
    the offset exceeds the tolerance, and shifts 'Is induced [-]' (and, if
    present, 'Induction time [h]') to the reference.

    Config keys (all optional, under ``processing``):

    - ``induction_time_h``: reference induction time in process-time hours.
    - ``induction_reference_file``: path to an evaluation workbook whose
      ``induction_reference_sheet`` (default ``'Time-Rownr'``) contains a row
      labelled ``induction_reference_label`` (default ``'Induction'``) with the
      time. Ignored when ``induction_time_h`` is set.
    - ``induction_time_tolerance_h``: warn threshold for the
      reference-vs-flip offset (default 0.25 h). The correction is applied
      regardless of the warning.

    Without any reference configured, the detected phase-flip time is only
    logged — behavior is unchanged.

    Args:
        df: DataFrame with 'Process time [h]' and (from
            :func:`add_induction_indicator`) 'Is induced [-]'.
        config: Merged configuration dict.

    Returns:
        DataFrame with the induction indicator (and timer, if present)
        aligned to the reference time.

    Raises:
        USPConfigError: If the reference time lies outside the data's time range.
        USPDataError: If a configured reference file cannot be parsed.
    """
    logger.info("Checking induction start time against reference...")
    df_out = df.copy()

    if "Process time [h]" not in df_out.columns:
        logger.warning(
            "Cannot correct induction time: 'Process time [h]' column not found."
        )
        return df_out

    processing = config.get("processing", {}) or {}
    reference_h = processing.get("induction_time_h")
    if reference_h is None:
        reference_file = processing.get("induction_reference_file")
        if reference_file:
            reference_h = _read_reference_induction_time(
                reference_file,
                processing.get("induction_reference_sheet", "Time-Rownr"),
                processing.get("induction_reference_label", "Induction"),
            )

    if reference_h is None:
        flip_time = _phase_flip_time(df_out)
        if flip_time is not None:
            logger.info(
                "No induction reference configured ('processing.induction_time_h' or"
                f" 'induction_reference_file'); keeping phase-based induction"
                f" at {flip_time:.3f}h."
            )
        return df_out

    reference_h = float(reference_h)
    t_min = df_out["Process time [h]"].min()
    t_max = df_out["Process time [h]"].max()
    if not (t_min <= reference_h <= t_max):
        raise USPConfigError(
            f"Induction reference time {reference_h:.3f}h is outside the data's "
            f"process-time range [{t_min:.3f}, {t_max:.3f}]h."
        )

    flip_time = _phase_flip_time(df_out)
    if flip_time is not None:
        tolerance_h = float(
            processing.get(
                "induction_time_tolerance_h", DEFAULT_INDUCTION_TIME_TOLERANCE_H
            )
        )
        offset_min = (reference_h - flip_time) * MIN_PER_HOUR
        message = (
            f"Induction reference {reference_h:.3f}h vs phase flip {flip_time:.3f}h "
            f"(offset {offset_min:+.0f} min)."
        )
        if abs(reference_h - flip_time) > tolerance_h:
            logger.warning(
                "Induction time check: %s Offset exceeds the %.2fh tolerance - "
                "correcting 'Is induced [-]' to the reference time.",
                message,
                tolerance_h,
            )
        else:
            logger.info("Induction time check: %s Within tolerance.", message)

    # Re-derive the indicator from the reference time (always has at least one
    # row, since the range check above guarantees reference_h <= t_max).
    from_reference = df_out["Process time [h]"] >= reference_h
    df_out["Is induced [-]"] = 0
    correction_index = from_reference.idxmax()
    df_out.loc[correction_index:, "Is induced [-]"] = 1

    # Keep the timer consistent if it already exists (standalone use; in
    # process_data the timer step runs after this and recomputes anyway).
    if "Induction time [h]" in df_out.columns:
        df_out["Induction time [h]"] = 0.0
        df_out.loc[df_out.index >= correction_index, "Induction time [h]"] = (
            df_out.loc[df_out.index >= correction_index, "Process time [h]"]
            - reference_h
        )

    return df_out
