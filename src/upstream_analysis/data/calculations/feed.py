"""Feed-rate calculation: cumulative feed added from balance (feed-source mode)."""

import logging

import numpy as np
import pandas as pd

from ...core.constants import MAD_TO_SIGMA_SCALE
from ...core.validation import validate_positive_numeric
from ._config import FEED_DEFINITIONS_KEY, get_nested_param, get_processing_config

logger = logging.getLogger(__name__)


def calculate_feed(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate cumulative feed added, assuming the balance measures the feed source.

    1. Finds the peak balance value around feed start.
    2. Creates a modified balance series (peak extended backward).
    3. Applies a median filter to the modified balance series.
    4. Calculates feed based on the filtered, modified balance.
    5. Overwrites the original 'Balance [g]' column with the filtered, modified version.

    Args:
        df: Input DataFrame with balance and phase data.
        config: Merged configuration dict.

    Returns:
        DataFrame with calculated feed columns.
    """
    logger.info("Attempting to calculate feed (adjusting balance first)...")
    df_out = df.copy()
    df_out["Feed [g]"] = 0.0
    df_out["Feed [mL]"] = 0.0

    balance_col = "Balance [g]"
    phase_col = "Process phase [-]"
    time_col = "Process time [h]"

    # --- Validate Inputs & Config ---
    required_cols = [balance_col, phase_col, time_col]
    if not all(col in df_out.columns for col in required_cols):
        logger.warning(
            f"Processor: Missing required columns {required_cols}. Cannot calculate feed."
        )
        return df_out

    processing_config = get_processing_config(config)

    # Absent feed definitions (e.g. a pure batch run) -> skip feed calculation
    # gracefully. An *invalid* feed density, however, must raise rather than
    # be silently swallowed.
    if not processing_config.get(FEED_DEFINITIONS_KEY):
        logger.info("No feed definitions in config; skipping feed calculation.")
        return df_out

    feed_density = validate_positive_numeric(
        get_nested_param(
            processing_config,
            [FEED_DEFINITIONS_KEY, "feed_1", "density_g_ml"],
            expected_type=float,
            is_required=True,
        ),
        "feed_density_g_ml",
        allow_zero=False,
    )
    # Median filter settings (all optional with sensible defaults)
    cleaning_config = config.get("cleaning", {})
    apply_filter = cleaning_config.get("median_filter", True)
    median_window = cleaning_config.get("median_filter_window", 10)
    median_threshold = cleaning_config.get("median_filter_threshold", 10)

    balance_numeric = pd.to_numeric(df_out[balance_col], errors="coerce")
    if balance_numeric.isnull().all():
        logger.warning(
            "Processor: Balance column empty/all NaN. Cannot calculate feed."
        )
        return df_out
    balance_nan_mask = balance_numeric.isna()
    balance_filled = balance_numeric.ffill()
    first_valid_balance_idx = balance_filled.first_valid_index()
    if first_valid_balance_idx is not None:
        balance_filled.fillna(balance_filled[first_valid_balance_idx], inplace=True)
    else:
        balance_filled.fillna(0, inplace=True)

    # --- Identify Start of FIRST Feeding Phase ---
    feeding_phases = ["Fedbatch", "induced Fedbatch"]
    feeding_phases_mask = df_out[phase_col].isin(feeding_phases)

    if not feeding_phases_mask.any():
        logger.warning(
            f"No feeding phases {feeding_phases} found. Feed calculated as 0."
        )
        return df_out

    first_feed_index = df_out[feeding_phases_mask].index[0]
    logger.info(f"First feeding phase starts at index: {first_feed_index}")

    # --- Find Initial Feed Weight ---
    time_window_h = 4.0
    end_window_time = df_out.loc[first_feed_index, time_col] + time_window_h
    window_mask = (df_out[time_col] >= df_out.loc[first_feed_index, time_col]) & (
        df_out[time_col] <= end_window_time
    )
    balance_window = balance_filled[window_mask].dropna()

    if balance_window.empty:
        logger.error("Cannot find valid balance readings in window after feed start.")
        balance_at_tare_point = balance_filled.loc[first_feed_index]  # Fallback
        if pd.isna(balance_at_tare_point):
            logger.error(
                "Balance at feed start is also NaN. Cannot determine tare point."
            )
            return df_out
        tare_index_approx = first_feed_index
        logger.warning(
            f"Using balance at feed start {balance_at_tare_point:.2f}g as fallback tare."
        )
    else:
        balance_at_tare_point = balance_window.max()
        tare_index_approx = balance_window.idxmax()
        logger.info(
            f"Using peak balance {balance_at_tare_point:.2f}g "
            f"(near index {tare_index_approx}) as initial feed weight."
        )

    # Degenerate-window fallback: the feed bottle may have been placed on the
    # balance only *after* feeding started (balance reads 0 through batch and
    # early fed-batch, then jumps when the bottle is set on). The window peak
    # is then 0 and feed would clip to zero for the whole run.
    #
    # Two-tier recovery:
    #   1. If the feed-pump duration signal is available and the balance is
    #      linear in it on the measured segment (r < -0.9), the pump doubles
    #      as a mass meter: reconstruct the *pre-placement* feed from the
    #      pump duration and synthesize the balance history so feed starts at
    #      the true feeding start.
    #   2. Otherwise, use the peak over all feeding-phase rows (feed then
    #      starts at bottle placement — flagged loudly).
    pump_reconstruction: dict | None = None
    if not balance_window.empty and balance_at_tare_point <= 0:
        phase_balance = balance_filled[feeding_phases_mask].dropna()
        dur_col = "Feed pump duration [-]"
        # 50 g: far above balance noise/pre-tare junk (~8 g), far below the
        # feed-bottle weight (hundreds of g) — excludes the settle-in transient.
        placed_rows = balance_filled > 50.0
        fit_ok = False
        if (
            dur_col in df_out.columns
            and placed_rows.sum() >= 10
            and feeding_phases_mask.any()
        ):
            dur_num = pd.to_numeric(df_out[dur_col], errors="coerce")
            if dur_num.notna().sum() >= 10:
                x = dur_num[placed_rows].to_numpy(dtype=float)
                y = balance_filled[placed_rows].to_numpy(dtype=float)
                if np.std(x) > 0:
                    slope, intercept = np.polyfit(x, y, 1)
                    r = float(np.corrcoef(x, y)[0, 1])
                    # balance = intercept + slope*duration with slope < 0
                    if slope < 0 and r < -0.9:
                        fit_ok = True
                        k_per_s = -slope  # grams delivered per pump-second
                        balance_at_tare_point = phase_balance.max()
                        tare_index_approx = phase_balance.idxmax()
                        dur_start = float(dur_num.loc[first_feed_index])
                        dur_tare = float(dur_num.loc[tare_index_approx])
                        pump_reconstruction = {
                            "k_per_s": k_per_s,
                            "dur_start": dur_start,
                            "dur_tare": dur_tare,
                        }
                        logger.warning(
                            "Balance was measured only from t=%.2f h (bottle "
                            "placed late; degenerate window peak %.2f g). "
                            "Reconstructing pre-placement feed from '%s' "
                            "(fit: balance = %.1f - %.4f g per pump-s, "
                            "r=%.3f). Feed now starts with the feeding phase.",
                            df_out.loc[first_feed_index, time_col],
                            balance_window.max(),
                            dur_col,
                            intercept,
                            k_per_s,
                            r,
                        )
        if not fit_ok:
            if not phase_balance.empty and phase_balance.max() > 0:
                balance_at_tare_point = phase_balance.max()
                tare_index_approx = phase_balance.idxmax()
                logger.warning(
                    "Window peak is degenerate (%.2f g) — feed bottle likely "
                    "placed after feeding start and no usable pump-duration "
                    "signal. Using peak over all feeding phases (%.2f g at "
                    "index %s) as tare; feed starts at bottle placement.",
                    balance_window.max(),
                    balance_at_tare_point,
                    tare_index_approx,
                )

    # --- Create Modified Balance Series ---
    modified_balance = balance_filled.copy()
    if pump_reconstruction is not None:
        # Synthesize the unmeasured history: bottle weight declining with
        # delivered feed (k per pump-second) back from the placement plateau.
        dur_num = pd.to_numeric(df_out["Feed pump duration [-]"], errors="coerce")
        k_per_s = pump_reconstruction["k_per_s"]
        dur_tare = pump_reconstruction["dur_tare"]
        synth = balance_at_tare_point + k_per_s * (dur_tare - dur_num)
        modified_balance.loc[modified_balance.index < tare_index_approx] = synth[
            modified_balance.index < tare_index_approx
        ]
    else:
        modified_balance.loc[modified_balance.index < tare_index_approx] = (
            balance_at_tare_point
        )
    modified_balance = modified_balance.ffill().fillna(
        balance_at_tare_point
    )  # Ensure full series

    # --- Apply Median Filter to Modified Balance ---
    if apply_filter:
        logger.info(
            f"Applying median filter (window={median_window}, "
            f"threshold={median_threshold}) to modified balance."
        )
        median_values = modified_balance.rolling(
            window=median_window, center=True, min_periods=1
        ).median()
        diff = np.abs(modified_balance - median_values)

        # Use numpy's median function for the rolling apply with raw=True
        mad = (
            diff.rolling(window=median_window, center=True, min_periods=1).apply(
                np.median, raw=True
            )
        ) * MAD_TO_SIGMA_SCALE  # Scale MAD to standard deviation

        mad.replace(0, np.nan, inplace=True)
        mask = (diff > median_threshold * mad) & mad.notna()
        spikes_found = mask.sum()
        if spikes_found > 0:
            filtered_modified_balance = modified_balance.copy()
            filtered_modified_balance[mask] = median_values[mask]
            logger.info(f"Replaced {spikes_found} spikes in modified balance data.")
        else:
            filtered_modified_balance = modified_balance
            logger.info("No significant spikes detected in modified balance data.")
    else:
        logger.info("Median filter disabled in config, skipping filter step.")
        filtered_modified_balance = modified_balance

    # --- Overwrite original balance column ---
    df_out[balance_col] = filtered_modified_balance
    logger.debug(
        f"Final filtered, modified balance assigned (Tail):\n{df_out[balance_col].tail()}"
    )

    # --- Calculate Feed Added ---
    feed_diff = balance_at_tare_point - df_out[balance_col]
    feed_g_series = feed_diff.loc[df_out.index >= first_feed_index].copy()
    feed_g_series_clipped = feed_g_series.clip(lower=0)
    nan_mask_feed = feed_g_series_clipped.isna()
    feed_g_filled_for_cummax = feed_g_series_clipped.ffill().fillna(0)
    feed_g_cummax = feed_g_filled_for_cummax.cummax()
    feed_g_cummax[nan_mask_feed] = np.nan
    df_out.loc[df_out.index >= first_feed_index, "Feed [g]"] = feed_g_cummax
    df_out.loc[balance_nan_mask, "Feed [g]"] = (
        np.nan
    )  # Restore NaNs from original balance

    if pump_reconstruction is not None:
        # Pump-based feed for the unmeasured pre-placement rows, plus a
        # continuity offset for the measured segment (feed was already
        # delivered before the bottle was placed on the balance).
        dur_num = pd.to_numeric(df_out["Feed pump duration [-]"], errors="coerce")
        k_per_s = pump_reconstruction["k_per_s"]
        offset = k_per_s * (
            pump_reconstruction["dur_tare"] - pump_reconstruction["dur_start"]
        )
        pre_mask = (df_out.index >= first_feed_index) & (
            df_out.index < tare_index_approx
        )
        df_out.loc[pre_mask, "Feed [g]"] = (
            k_per_s * (dur_num.loc[pre_mask] - pump_reconstruction["dur_start"])
        ).clip(lower=0)
        post_mask = df_out.index >= tare_index_approx
        df_out.loc[post_mask, "Feed [g]"] = offset + df_out.loc[post_mask, "Feed [g]"]

    # Calculate Feed [mL]
    df_out["Feed [mL]"] = df_out["Feed [g]"] / feed_density
    logger.info("Processor: Calculated Feed [g] and Feed [mL].")
    logger.debug(f"Final Feed [g] calculated (Tail):\n{df_out['Feed [g]'].tail()}")

    return df_out
