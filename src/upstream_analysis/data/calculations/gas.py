"""Gas-exchange calculations: inlet composition, off-gas rates (CER/OUR/RQ), and specific rates."""

import logging

import numpy as np
import pandas as pd

from ...core.constants import (
    CO2_IN_AIR_PERCENT,
    MIN_PER_HOUR,
    ML_PER_L,
    MMOL_PER_MOL,
    MOLAR_VOLUME_AIR_L_PER_MOL,
    O2_IN_AIR_PERCENT,
    PERCENT,
)

logger = logging.getLogger(__name__)


def _gas_params(config: dict) -> dict:
    """Resolve the merged gas_params section (old-key wins on conflict).

    Supports both the new structure (``processing.gas_params``) and the legacy
    one (``processing_parameters.gas_params``).
    """
    gas_config = {}
    gas_config.update(config.get("processing", {}).get("gas_params", {}) or {})
    gas_config.update(
        config.get("processing_parameters", {}).get("gas_params", {}) or {}
    )
    return gas_config


def calculate_inlet_gas_composition(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate inlet gas composition from air and pure-O2 flows.

    Uses a flow-weighted average accounting for the mixing of air
    (``O2_IN_AIR_PERCENT``% O2, ``CO2_IN_AIR_PERCENT``% CO2) with pure O2
    (100% O2, 0% CO2)::

        y_O2_in  = (F_air * o2_in_air + F_O2 * 100) / (F_air + F_O2)
        y_CO2_in = (F_air * co2_in_air + F_O2 * 0)   / (F_air + F_O2)

    Args:
        df: DataFrame with air and O2 flow columns.
        config: Merged configuration dict.

    Returns:
        DataFrame with added columns 'Inlet O2 [%]' and 'Inlet CO2 [%]'.
    """
    logger.info("Calculating inlet gas composition...")
    df_out = df.copy()

    gas_config = _gas_params(config)

    # --- Check if calculation is enabled ---
    # Default to True if not specified (for backward compatibility)
    calculate_enabled = gas_config.get("calculate_inlet_composition", True)
    if not calculate_enabled:
        logger.info("Inlet gas composition calculation disabled in config.")
        return df_out

    # --- Required columns ---
    air_flow_col = "Air flow [L/min]"
    o2_flow_col = "O2 flow [L/min]"

    # Check for standard column names first, then try alternatives
    if air_flow_col not in df_out.columns:
        # Try alternative names
        for alt in ["AirFlow [L/min]", "Air flow [L/min]", "AirFlow, l/min"]:
            if alt in df_out.columns:
                air_flow_col = alt
                break
        else:
            logger.warning(
                "Air flow column not found. Cannot calculate inlet gas composition."
            )
            return df_out

    if o2_flow_col not in df_out.columns:
        # Try alternative names
        for alt in [
            "Gas2 flow [L/min]",
            "Gas2Flow [L/min]",
            "Gas2Flow, l/min",
            "O2 flow [L/min]",
        ]:
            if alt in df_out.columns:
                o2_flow_col = alt
                break
        else:
            logger.warning(
                "O2 flow column not found. Cannot calculate inlet gas composition."
            )
            return df_out

    # --- Get gas parameters (support both old and new key names) ---
    try:
        # Try new key names first
        o2_in_air = gas_config.get("o2_in_air_percent")
        co2_in_air = gas_config.get("co2_in_air_percent")

        # Fall back to old key names
        if o2_in_air is None:
            o2_in_air = gas_config.get("inlet_o2_mol_pct", O2_IN_AIR_PERCENT)
        if co2_in_air is None:
            co2_in_air = gas_config.get("inlet_co2_mol_pct", CO2_IN_AIR_PERCENT)
    except Exception as e:
        logger.error(f"Error reading gas parameters: {e}")
        return df_out

    # --- Get flows as numeric ---
    air_flow = pd.to_numeric(df_out[air_flow_col], errors="coerce")
    o2_flow = pd.to_numeric(df_out[o2_flow_col], errors="coerce")
    total_flow = air_flow + o2_flow

    # --- Calculate inlet O2 concentration (weighted average) ---
    # y_O2_in = (F_air * o2_in_air + F_O2 * 100.0) / (F_air + F_O2)
    df_out["Inlet O2 [%]"] = (
        air_flow * o2_in_air + o2_flow * 100.0
    ) / total_flow.replace(0, np.nan)

    # --- Calculate inlet CO2 concentration ---
    # y_CO2_in = (F_air * co2_in_air + F_O2 * 0.0) / (F_air + F_O2)
    df_out["Inlet CO2 [%]"] = (
        air_flow * co2_in_air + o2_flow * 0.0
    ) / total_flow.replace(0, np.nan)

    logger.info("Calculated inlet gas composition (O2, CO2).")
    return df_out


def calculate_offgas_rates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate gas-exchange rates CER, OUR, and RQ.

    Two methods are supported: a simple method using total flow and gas
    fractions directly, and an N2-balance method that accounts for changes in
    total gas flow (more accurate, requires inlet/outlet fractions).

    Args:
        df: DataFrame with off-gas data and 'Volume corrected [mL]'.
        config: Merged configuration dict.

    Returns:
        DataFrame with added columns (as configured) 'CER [mmol/L/h]',
        'OUR [mmol/L/h]', and 'RQ [-]'.
    """
    logger.info("Calculating offgas rates (CER, OUR, RQ)...")
    df_out = df.copy()

    gas_config = _gas_params(config)

    # --- Check if calculations are enabled ---
    # Default to True if not specified (for backward compatibility)
    calculate_cer = gas_config.get("calculate_cer", True)
    calculate_our = gas_config.get("calculate_our", True)
    calculate_rq = gas_config.get("calculate_rq", True)

    if not any([calculate_cer, calculate_our, calculate_rq]):
        logger.info("Offgas rate calculations disabled in config.")
        return df_out

    # --- Required columns ---
    total_flow_col = "Total flow [L/min]"
    co2_out_col = "Offgas CO2 [%]"
    o2_out_col = "Offgas O2 [%]"
    volume_col = "Volume corrected [mL]"

    # Check for column alternatives
    if total_flow_col not in df_out.columns:
        for alt in ["TotalFlow [L/min]", "TotalFlow, l/min"]:
            if alt in df_out.columns:
                total_flow_col = alt
                break
        else:
            logger.warning(
                "Total flow column not found. Cannot calculate offgas rates."
            )
            return df_out

    if volume_col not in df_out.columns:
        logger.warning(
            f"Volume column '{volume_col}' not found. Cannot calculate offgas rates."
        )
        return df_out

    if co2_out_col not in df_out.columns:
        logger.warning("CO2 outlet column not found. Cannot calculate CER.")
        calculate_cer = False

    if o2_out_col not in df_out.columns:
        logger.warning("O2 outlet column not found. Cannot calculate OUR.")
        calculate_our = False

    # --- Get numeric values ---
    f_gas = pd.to_numeric(df_out[total_flow_col], errors="coerce")  # L/min
    volume_l = (
        pd.to_numeric(df_out[volume_col], errors="coerce") / ML_PER_L
    )  # Convert mL to L

    molar_volume = gas_config.get("molar_volume_air_l_mol", MOLAR_VOLUME_AIR_L_PER_MOL)
    try:
        molar_volume = float(molar_volume)
    except (TypeError, ValueError):
        molar_volume = MOLAR_VOLUME_AIR_L_PER_MOL
        logger.warning(
            f"Invalid molar_volume_air_l_mol, using {MOLAR_VOLUME_AIR_L_PER_MOL} L/mol."
        )

    use_n2_balance = gas_config.get("use_n2_balance", False)
    offgas_is_wet = gas_config.get("offgas_is_wet", False)

    # Handle inlet O2/CO2 (calculated or assumed) - support both key names
    if "Inlet O2 [%]" in df_out.columns:
        o2_in = pd.to_numeric(df_out["Inlet O2 [%]"], errors="coerce")
    else:
        o2_in = gas_config.get("o2_in_air_percent")
        if o2_in is None:
            o2_in = gas_config.get("inlet_o2_mol_pct", O2_IN_AIR_PERCENT)
        o2_in = pd.Series(o2_in, index=df_out.index)
        logger.info(f"Using assumed inlet O2: {o2_in.iloc[0]}%")

    if "Inlet CO2 [%]" in df_out.columns:
        co2_in = pd.to_numeric(df_out["Inlet CO2 [%]"], errors="coerce")
    else:
        co2_in = gas_config.get("co2_in_air_percent")
        if co2_in is None:
            co2_in = gas_config.get("inlet_co2_mol_pct", CO2_IN_AIR_PERCENT)
        co2_in = pd.Series(co2_in, index=df_out.index)

    if use_n2_balance:
        # Convert inlet fractions to dry mol fractions
        y_o2_in = o2_in / PERCENT
        y_co2_in = co2_in / PERCENT
        y_n2_in = (1.0 - y_o2_in - y_co2_in).clip(lower=0.0)

        # Outlet wet/dry handling
        o2_out_wet = pd.to_numeric(df_out[o2_out_col], errors="coerce") / PERCENT
        co2_out_wet = pd.to_numeric(df_out[co2_out_col], errors="coerce") / PERCENT

        h2o_out_pct = gas_config.get("outlet_h2o_mol_pct")
        if h2o_out_pct is None:
            h2o_out_pct = gas_config.get("inlet_h2o_mol_pct", 0.0)
        try:
            h2o_out_frac = float(h2o_out_pct) / PERCENT
        except (TypeError, ValueError):
            h2o_out_frac = 0.0

        if h2o_out_frac < 0 or h2o_out_frac > 0.3:
            logger.warning(
                f"Outlet H2O mol fraction {h2o_out_frac:.4f} out of expected range. Clipping."
            )
            h2o_out_frac = min(max(h2o_out_frac, 0.0), 0.3)

        if offgas_is_wet:
            dry_denom = 1.0 - h2o_out_frac
            if dry_denom <= 0:
                dry_denom = np.nan
            o2_out_dry = o2_out_wet / dry_denom
            co2_out_dry = co2_out_wet / dry_denom
        else:
            o2_out_dry = o2_out_wet
            co2_out_dry = co2_out_wet

        y_n2_out = (1.0 - o2_out_dry - co2_out_dry).replace(0, np.nan)

        f_in_mol_h = f_gas * MIN_PER_HOUR / molar_volume
        f_out_mol_h = f_in_mol_h * (y_n2_in / y_n2_out)

        if calculate_cer:
            cer_mol_h = (f_out_mol_h * co2_out_dry) - (f_in_mol_h * y_co2_in)
            df_out["CER [mmol/L/h]"] = (cer_mol_h / volume_l) * 1000.0
            logger.info("Calculated CER [mmol/L/h] using N2 balance.")

        if calculate_our:
            our_mol_h = (f_in_mol_h * y_o2_in) - (f_out_mol_h * o2_out_dry)
            df_out["OUR [mmol/L/h]"] = (our_mol_h / volume_l) * 1000.0
            logger.info("Calculated OUR [mmol/L/h] using N2 balance.")
    else:
        # Legacy simple method using measured fractions and total flow
        factor = MIN_PER_HOUR * MMOL_PER_MOL / molar_volume
        if calculate_cer:
            co2_out = pd.to_numeric(df_out[co2_out_col], errors="coerce")
            df_out["CER [mmol/L/h]"] = f_gas * (co2_out / PERCENT) / volume_l * factor
            logger.info("Calculated CER [mmol/L/h] using simple method.")

        if calculate_our:
            o2_out = pd.to_numeric(df_out[o2_out_col], errors="coerce")
            df_out["OUR [mmol/L/h]"] = (
                f_gas * ((o2_in - o2_out) / PERCENT) / volume_l * factor
            )
            logger.info("Calculated OUR [mmol/L/h] using simple method.")

    # --- Calculate RQ (Respiratory Quotient) ---
    if calculate_rq:
        if "CER [mmol/L/h]" in df_out.columns and "OUR [mmol/L/h]" in df_out.columns:
            df_out["RQ [-]"] = df_out["CER [mmol/L/h]"] / df_out[
                "OUR [mmol/L/h]"
            ].replace(0, np.nan)
            logger.info("Calculated RQ [-].")
        else:
            logger.warning("Cannot calculate RQ: CER and/or OUR not available.")

    return df_out


def calculate_specific_offgas_rates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate biomass-specific gas-exchange rates (qO2, qCO2).

    Args:
        df: DataFrame with 'OUR [mmol/L/h]', 'CER [mmol/L/h]', and 'Biomass [g/L]'.
        config: Merged configuration dict.

    Returns:
        DataFrame with added columns (as available) 'qO2 [mmol/g/h]' and
        'qCO2 [mmol/g/h]'.
    """
    logger.info("Calculating specific offgas rates (qO2, qCO2)...")
    df_out = df.copy()

    gas_config = _gas_params(config)

    # --- Check if calculation is enabled ---
    # Default to True if not specified (for backward compatibility)
    calculate_specific_rates = gas_config.get("calculate_specific_rates", True)
    if not calculate_specific_rates:
        logger.info("Specific offgas rate calculations disabled in config.")
        return df_out

    # --- Required columns ---
    biomass_col = "Biomass [g/L]"

    if biomass_col not in df_out.columns:
        logger.warning(
            f"Biomass column '{biomass_col}' not found. Cannot calculate specific rates."
        )
        return df_out

    biomass = pd.to_numeric(df_out[biomass_col], errors="coerce")

    # --- Calculate qO2 (specific OUR) ---
    if "OUR [mmol/L/h]" in df_out.columns:
        df_out["qO2 [mmol/g/h]"] = df_out["OUR [mmol/L/h]"] / biomass.replace(0, np.nan)
        logger.info("Calculated qO2 [mmol/g/h].")
    else:
        logger.warning("Cannot calculate qO2: OUR column not found.")

    # --- Calculate qCO2 (specific CER) ---
    if "CER [mmol/L/h]" in df_out.columns:
        df_out["qCO2 [mmol/g/h]"] = df_out["CER [mmol/L/h]"] / biomass.replace(
            0, np.nan
        )
        logger.info("Calculated qCO2 [mmol/g/h].")
    else:
        logger.warning("Cannot calculate qCO2: CER column not found.")

    return df_out
