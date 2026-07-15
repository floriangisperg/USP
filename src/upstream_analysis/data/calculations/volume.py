"""Reactor volume calculations: base-titrant flux and working volume tracking."""

import logging

import numpy as np
import pandas as pd

from ._config import (
    BASE_TITRANT_KEY,
    REACTOR_SETUP_KEY,
    get_nested_param,
    get_processing_config,
)

logger = logging.getLogger(__name__)


def calculate_base_flux_and_volume(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate cumulative base titrant volume added to the bioreactor.

    Determines the amount of base (typically NaOH or NH4OH) added for pH control
    by analyzing pump-duration data. Assumes the pump duration increases
    cumulatively throughout the fermentation and uses the total added base amount
    from configuration to scale duration to actual volume.

    Args:
        df: DataFrame with 'Base pump duration [-]' and 'Process phase [-]'.
        config: Merged configuration dict.

    Returns:
        DataFrame with the added column 'Base volume added [mL]'.
    """
    logger.info("Attempting to calculate base volume added...")
    df_out = df.copy()
    # Initialize column FIRST
    df_out["Base volume added [mL]"] = 0.0

    # --- Get Parameters & Validate ---
    processing_config = get_processing_config(config)
    try:
        total_base_g = get_nested_param(
            processing_config,
            [BASE_TITRANT_KEY, "total_added_g"],
            expected_type=float,
            is_required=True,
        )
        base_density_g_ml = get_nested_param(
            processing_config,
            [BASE_TITRANT_KEY, "density_g_ml"],
            expected_type=float,
            is_required=True,
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Cannot calculate base volume: {e}")
        return df_out  # Return df with initialized column

    duration_col = "Base pump duration [-]"
    phase_col = "Process phase [-]"

    if duration_col not in df_out.columns or phase_col not in df_out.columns:
        logger.warning(
            f"Processor: Missing '{duration_col}' or '{phase_col}'. Cannot calculate base volume."
        )
        return df_out

    duration_numeric = pd.to_numeric(df_out[duration_col], errors="coerce")
    if (
        not pd.api.types.is_numeric_dtype(duration_numeric)
        or duration_numeric.isnull().all()
    ):
        logger.warning(
            f"Processor: Base duration '{duration_col}' not valid numeric or all NaN. Cannot calculate base volume."
        )
        return df_out

    total_base_ml = total_base_g / base_density_g_ml

    # --- Identify Period Based on Phases ---
    base_addition_phases = ["Batch", "Fedbatch", "induced Fedbatch"]
    base_period_mask = df_out[phase_col].isin(base_addition_phases)
    if not base_period_mask.any():
        logger.warning(
            f"No data in base addition phases {base_addition_phases}. Base volume 0."
        )
        return df_out

    period_durations = duration_numeric[base_period_mask]
    if period_durations.dropna().empty:
        logger.error("No valid base duration values found within the specified phases.")
        return df_out

    first_phase_index = period_durations.index[0]
    last_phase_index = period_durations.index[-1]

    # --- Get Boundary Durations Robustly ---
    duration_start_of_period = period_durations.loc[first_phase_index]
    if pd.isna(duration_start_of_period):
        first_valid_duration = period_durations.dropna()
        if first_valid_duration.empty:
            logger.error("No valid base duration start value found within the period.")
            return df_out
        duration_start_of_period = first_valid_duration.iloc[0]
        logger.warning(
            f"NaN base duration at start index {first_phase_index}. Using first valid: {duration_start_of_period:.2f}s"
        )

    duration_end_of_period = period_durations.dropna().iloc[-1]

    total_duration_s = duration_end_of_period - duration_start_of_period

    if total_duration_s <= 0:
        logger.warning(
            f"Total base pump duration during relevant phases ({total_duration_s}s) not positive. Base volume added set to 0."
        )
        return df_out

    logger.info(
        f"Base addition period indices: {first_phase_index} to {last_phase_index}."
    )
    logger.info(
        f"Duration range within period: {duration_start_of_period:.2f}s to {duration_end_of_period:.2f}s. Total duration: {total_duration_s:.2f}s"
    )
    logger.info(
        f"Total base added (config): {total_base_g:.2f} g = {total_base_ml:.2f} mL"
    )

    # --- Calculate Cumulative Base Volume Added ---
    # Use the original numeric duration, fill NaNs robustly
    duration_filled = duration_numeric.ffill().fillna(0)

    cumulative_duration_since_period_start = (
        duration_filled - duration_start_of_period
    ).clip(lower=0)
    volume_added = cumulative_duration_since_period_start * (
        total_base_ml / total_duration_s
    )

    # --- Assign and Cap ---
    # Initialize first, then apply calculations for the period
    df_out["Base volume added [mL]"] = 0.0
    # Use .loc with the mask for assignment within the period
    df_out.loc[base_period_mask, "Base volume added [mL]"] = volume_added[
        base_period_mask
    ]

    # Set volume AFTER the period ends to the maximum total added volume
    df_out.loc[df_out.index > last_phase_index, "Base volume added [mL]"] = (
        total_base_ml
    )

    # Ensure values before start are 0 (belt-and-suspenders)
    df_out.loc[df_out.index < first_phase_index, "Base volume added [mL]"] = 0.0

    # Final clip only (NO fillna(0.0) here)
    df_out["Base volume added [mL]"] = df_out["Base volume added [mL]"].clip(
        upper=total_base_ml
    )

    logger.info("Calculated 'Base volume added [mL]' column.")
    logger.debug(
        f"Tail of Base volume added [mL] after calculation:\n{df_out['Base volume added [mL]'].tail()}"
    )
    return df_out


def calculate_volume(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate reactor volume with sample-removal correction.

    Combines initial volume with cumulative additions (feed, base) and subtracts
    sample removals. The corrected volume is used downstream for
    concentration-based calculations.

    Args:
        df: DataFrame with feed and base volume columns.
        config: Merged configuration dict.

    Returns:
        DataFrame with added columns 'Volume [mL]', 'Volume corrected [mL]',
        and 'Cumulative sample volume [mL]'.
    """
    logger.info("Calculating reactor volume...")
    df_out = df.copy()

    processing_config = get_processing_config(config)

    # --- Get Start Volume ---
    try:
        start_volume = get_nested_param(
            processing_config,
            [REACTOR_SETUP_KEY, "initial_volume_ml"],
            expected_type=float,
            is_required=True,
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Processor: Cannot calculate volume: {e}")
        df_out["Volume [mL]"] = np.nan
        df_out["Cumulative sample volume [mL]"] = np.nan
        df_out["Volume corrected [mL]"] = np.nan
        return df_out

    # --- Get Input Columns & Force Numeric, Fill NaNs ---
    feed_vol = pd.to_numeric(df_out.get("Feed [mL]", 0.0), errors="coerce").fillna(0.0)
    base_vol = pd.to_numeric(
        df_out.get("Base volume added [mL]", 0.0), errors="coerce"
    ).fillna(0.0)

    # --- Calculate Base Volume (start + feed + base) ---
    # Explicitly ensure components are floats before summing
    try:
        volume_calc = (
            float(start_volume) + feed_vol.astype(float) + base_vol.astype(float)
        )
        df_out["Volume [mL]"] = volume_calc
    except Exception as calc_error:
        logger.error(f"Error during Volume [mL] calculation: {calc_error}")
        df_out["Volume [mL]"] = np.nan  # Assign NaN if calculation fails

    logger.info(
        f"Calculated base 'Volume [mL]' (start={start_volume} + feed + base). Check tail:"
    )
    log_df_tail = pd.DataFrame(
        {
            "Feed [mL]_used": feed_vol.tail(),
            "Base volume used": base_vol.tail(),
            "Volume [mL]_calc": df_out["Volume [mL]"].tail(),  # Log directly from df
        }
    )
    logger.info(f"\n{log_df_tail}")

    # --- Apply Sample Volume Correction ---
    # Check if Volume [mL] calculation failed
    if df_out["Volume [mL]"].isnull().all():
        logger.warning(
            "Base volume calculation failed, cannot calculate corrected volume."
        )
        df_out["Volume corrected [mL]"] = np.nan
        df_out["Cumulative sample volume [mL]"] = 0.0
        return df_out  # Return early

    if "Sample volume [mL]" in df_out.columns and "is_sample" in df_out.columns:
        sample_vol_numeric = pd.to_numeric(
            df_out["Sample volume [mL]"], errors="coerce"
        )
        if sample_vol_numeric.notna().any():
            logger.info("Applying sample volume correction.")
            sample_vol_at_event = pd.Series(0.0, index=df_out.index)
            is_sample_mask = df_out["is_sample"].fillna(False).astype(bool)
            sample_vol_at_event.loc[is_sample_mask] = sample_vol_numeric[
                is_sample_mask
            ].fillna(0.0)
            df_out["Cumulative sample volume [mL]"] = sample_vol_at_event.cumsum()
            # Calculate corrected volume using the just calculated Volume [mL]
            df_out["Volume corrected [mL]"] = (
                df_out["Volume [mL]"] - df_out["Cumulative sample volume [mL]"]
            )
            logger.info(
                "Calculated 'Volume corrected [mL]' accounting for sample removal."
            )
        else:
            logger.info(
                "No valid 'Sample volume [mL]' data. 'Volume corrected [mL]' equals base 'Volume [mL]'."
            )
            df_out["Volume corrected [mL]"] = df_out["Volume [mL]"].copy()
            df_out["Cumulative sample volume [mL]"] = 0.0
    else:
        logger.info(
            "No sample info. 'Volume corrected [mL]' equals base 'Volume [mL]'."
        )
        df_out["Volume corrected [mL]"] = df_out["Volume [mL]"].copy()
        df_out["Cumulative sample volume [mL]"] = 0.0

    logger.info(
        f"Final calculated Volume corrected [mL] (Tail):\n{df_out['Volume corrected [mL]'].tail()}"
    )

    return df_out
