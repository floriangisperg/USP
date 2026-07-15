"""Carbon-balance analysis: track carbon flow and closure across the fermentation."""

import logging

import numpy as np
import pandas as pd

from ...core.constants import (
    ACETATE_MOLAR_MASS_G_PER_C_MOL,
    BIOMASS_MOLAR_MASS_G_PER_C_MOL,
    DEFAULT_FEED_GLUCOSE_G_L,
    DEFAULT_INITIAL_GLUCOSE_G_L,
    DEFAULT_INITIAL_VOLUME_ML,
    GLUCOSE_MOLAR_MASS_G_PER_C_MOL,
    MMOL_PER_MOL,
    PERCENT,
)
from ._config import get_first

logger = logging.getLogger(__name__)


def calculate_carbon_balance(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Perform comprehensive carbon-balance analysis.

    Tracks carbon flow through the system to assess metabolic state, identify
    measurement inconsistencies, and evaluate process efficiency on a C-mol
    basis::

        C_in = C_out + C_accumulated + C_error

    Args:
        df: DataFrame with feed, volume, and CER data (plus optional offline
            biomass/protein/acetate/glucose columns).
        config: Merged configuration dict.

    Returns:
        DataFrame with the 'C *' family of input/output/distribution/closure
        columns. Skips gracefully (returns the df unchanged) when biomass or the
        other required inputs are absent.
    """
    logger.info("Calculating carbon balance...")
    df_out = df.copy()

    # Biomass is required for the carbon balance. Skip gracefully if absent
    # (e.g. online-only data without offline measurements).
    if "Biomass [g/L]" not in df_out.columns:
        logger.warning("Carbon balance skipped: 'Biomass [g/L]' column not found.")
        return df_out

    # --- Get mass balance config ---
    # Support both new and old config structures
    mb_config = config.get("processing", {}).get("mass_balance_params", {})
    if not mb_config:
        mb_config = config.get("processing_parameters", {}).get(
            "mass_balance_params", {}
        )

    # Default to True if not specified (for backward compatibility)
    calculate_cb = mb_config.get("calculate_carbon_balance", True)
    if not calculate_cb:
        logger.info("Carbon balance calculation disabled in config.")
        return df_out

    # --- Get elemental compositions ---
    # Biomass: CH1.77N0.24O0.49 (from default config)
    biomass_cmol_g = get_first(
        config,
        [
            ["processing", "biomass_params", "molar_mass_g_c_mol"],
            ["processing_parameters", "biomass_params", "molar_mass_g_c_mol"],
            ["processing_parameters", "biomass", "molar_mass_g_c_mol"],
        ],
        default=BIOMASS_MOLAR_MASS_G_PER_C_MOL,
    )

    # Glucose: C6H12O6 -> CH2O on C-mol basis
    glucose_cmol_g = get_first(
        config,
        [
            [
                "processing_parameters",
                "substrate_properties",
                "glucose",
                "molar_mass_g_c_mol",
            ],
            [
                "processing",
                "processing_parameters",
                "substrate_properties",
                "glucose",
                "molar_mass_g_c_mol",
            ],
        ],
        default=GLUCOSE_MOLAR_MASS_G_PER_C_MOL,
    )

    # Product: from run_config (if available)
    product_cmol_g = None
    try:
        product_props = get_first(
            config,
            [
                ["processing", "processing_parameters", "product_properties"],
                ["processing_parameters", "product_properties"],
                ["processing", "product_params"],
            ],
            default={},
        )
        if (
            isinstance(product_props, dict)
            and "elemental_composition_c_mol" in product_props
        ):
            product_cmol_g = product_props.get("molar_mass_g_c_mol")
            logger.info(
                f"Found product elemental composition: {product_cmol_g} g/C-mol"
            )
    except Exception:
        logger.info("No product elemental composition found in config.")

    # Acetate: C2H4O2 -> CH2O on C-mol basis
    acetate_cmol_g = get_first(
        config,
        [
            [
                "processing_parameters",
                "substrate_properties",
                "acetate",
                "molar_mass_g_c_mol",
            ],
            [
                "processing",
                "processing_parameters",
                "substrate_properties",
                "acetate",
                "molar_mass_g_c_mol",
            ],
        ],
        default=ACETATE_MOLAR_MASS_G_PER_C_MOL,
    )

    # Feed concentration (g glucose per L feed) - from media config
    feed_conc_g_l = get_first(
        config,
        [
            [
                "analysis",
                "delisa_media_config",
                "feed_1",
                "components",
                "glucose",
                "concentration_g_l",
            ]
        ],
        default=DEFAULT_FEED_GLUCOSE_G_L,
    )

    # Initial volume and batch glucose concentration
    initial_volume_ml = get_first(
        config,
        [["processing_parameters", "reactor_setup", "initial_volume_ml"]],
        default=DEFAULT_INITIAL_VOLUME_ML,
    )
    initial_glucose_g_l = get_first(
        config,
        [
            [
                "analysis",
                "delisa_media_config",
                "batch_medium",
                "components",
                "glucose",
                "concentration_g_l",
            ]
        ],
        default=DEFAULT_INITIAL_GLUCOSE_G_L,
    )

    # --- Get required columns ---
    required_cols = ["Feed [mL]", "Volume corrected [mL]", "CER [mmol/L/h]"]
    missing = [c for c in required_cols if c not in df_out.columns]
    if missing:
        logger.warning(f"Missing required columns for carbon balance: {missing}")
        return df_out

    # --- Calculate time delta (in hours) for integration ---
    time_col = "Process time [h]"
    if time_col not in df_out.columns:
        logger.warning(
            f"Time column '{time_col}' not found. Cannot calculate carbon balance."
        )
        return df_out

    dt_hours = df_out[time_col].diff().fillna(0)

    # Glucose C-mol mass consistent with the media-config concentration basis
    # (e.g. glucose monohydrate C6H12O6*H2O -> mw_g_mol / c_atoms). The
    # substrate_properties value (30.03, anhydrous) must NOT be paired with a
    # monohydrate concentration (440 g/L), or feed carbon is over-counted ~10%.
    feed_glc_mw = get_first(
        config,
        [
            [
                "analysis",
                "delisa_media_config",
                "feed_1",
                "components",
                "glucose",
                "mw_g_mol",
            ]
        ],
        default=None,
    )
    feed_glc_catoms = get_first(
        config,
        [
            [
                "analysis",
                "delisa_media_config",
                "feed_1",
                "components",
                "glucose",
                "c_atoms_per_molecule",
            ]
        ],
        default=None,
    )
    if feed_glc_mw and feed_glc_catoms:
        glucose_balance_cmol_g = float(feed_glc_mw) / float(feed_glc_catoms)
    else:
        glucose_balance_cmol_g = (
            glucose_cmol_g  # fall back to anhydrous substrate value
        )

    # Capture where biomass is actually MEASURED (before interpolation fills the
    # gaps). Used for the rigorous sample-point-only closure metric below.
    measured_mask = pd.to_numeric(df_out["Biomass [g/L]"], errors="coerce").notna()

    # Interpolate sparse offline quantities onto the high-frequency timeline so
    # the running balance is meaningful between sample points (otherwise the
    # biomass/product/glucose terms read 0 at non-sample rows).
    # NOTE: between samples these are ESTIMATES, not measurements. The continuous
    # closure built on them is a smoothed diagnostic; the assumption-free closure
    # is 'C recovered (measured) [%]' reported only at sample timestamps.
    def _interp_offline(colname, log=False):
        if colname not in df_out.columns:
            return None
        s = pd.to_numeric(df_out[colname], errors="coerce")
        if s.notna().sum() < 2:
            return s
        if log and (s[s > 0].size >= 2):
            # log-linear (exponential) interpolation: respects growth kinetics
            # (dX/dt = mu*X) far better than a straight line for biomass.
            log_s = pd.Series(np.nan, index=s.index, dtype=float)
            pos = s > 0
            log_s.loc[pos] = np.log(s.loc[pos])
            log_s = (
                log_s.interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
            return np.exp(log_s)
        return s.interpolate(method="linear", limit_direction="both").ffill().bfill()

    biomass_g_l = _interp_offline("Biomass [g/L]", log=True)
    protein_g_l = _interp_offline("Protein [g/L]")
    acetate_g_l = _interp_offline("Acetate [g/L]")
    glucose_g_l = _interp_offline("Glucose [g/L]")

    # --- Carbon INPUTS ---

    # 1. Carbon from feed (cumulative)
    # Feed [mL] x feed_conc (g/L) / glucose_balance_cmol_g = C-mol from feed
    # NOTE: Feed [mL] is already a cumulative value, so we do NOT apply cumsum() again
    feed_ml = pd.to_numeric(df_out["Feed [mL]"], errors="coerce")
    feed_g = feed_ml * feed_conc_g_l / 1000.0  # Convert mL to L
    df_out["C from feed [C-mol]"] = feed_g / glucose_balance_cmol_g

    # 2. Initial glucose from batch medium
    initial_volume_l = initial_volume_ml / 1000.0
    initial_c_mol = initial_volume_l * initial_glucose_g_l / glucose_balance_cmol_g
    df_out["C from batch [C-mol]"] = initial_c_mol

    # Total carbon in
    df_out["C in [C-mol]"] = (
        df_out["C from feed [C-mol]"] + df_out["C from batch [C-mol]"]
    )

    # --- Carbon OUTPUTS ---

    # 1. CO2 carbon (cumulative)
    # CER [mmol/L/h] x Volume [L] x dt [h] = mmol CO2 produced
    # 1 mmol CO2 = 1 mmol C = 0.001 C-mol
    volume_l = pd.to_numeric(df_out["Volume corrected [mL]"], errors="coerce") / 1000.0
    cer_mmol_l_h = pd.to_numeric(df_out["CER [mmol/L/h]"], errors="coerce")
    co2_production_rate_c_mol_h = cer_mmol_l_h * volume_l / MMOL_PER_MOL  # C-mol/h
    df_out["C from CO2 [C-mol]"] = (co2_production_rate_c_mol_h * dt_hours).cumsum()

    # 2. Biomass carbon (interpolated onto timeline; current amount in reactor)
    if biomass_g_l is None:
        biomass_g_l = pd.Series(np.nan, index=df_out.index)
    biomass_c_mol_in_reactor = biomass_g_l * volume_l / biomass_cmol_g
    df_out["C in biomass [C-mol]"] = biomass_c_mol_in_reactor

    # 3. Product carbon (if available)
    if protein_g_l is not None and product_cmol_g is not None:
        product_c_mol_in_reactor = protein_g_l * volume_l / product_cmol_g
        df_out["C in product [C-mol]"] = product_c_mol_in_reactor
    else:
        df_out["C in product [C-mol]"] = np.nan
        product_c_mol_in_reactor = pd.Series(np.nan, index=df_out.index)
        logger.info("Product carbon not calculated (no product data or composition)")

    # 4. Acetate carbon (if available)
    if acetate_g_l is not None:
        df_out["C in acetate [C-mol]"] = acetate_g_l * volume_l / acetate_cmol_g
    else:
        df_out["C in acetate [C-mol]"] = 0.0
        logger.info("Acetate carbon not calculated (no acetate data)")

    # 5. Residual glucose carbon (offline assay reports anhydrous glucose)
    if glucose_g_l is not None:
        df_out["C residual glucose [C-mol]"] = glucose_g_l * volume_l / glucose_cmol_g
    else:
        df_out["C residual glucose [C-mol]"] = 0.0
        logger.info("Residual glucose carbon not calculated (no glucose data)")

    # --- Carbon BALANCE ---
    # Account for cumulative removals from sampling to build a cumulative balance.
    _sample_vol = df_out.get("Sample volume [mL]")
    if isinstance(_sample_vol, pd.Series):
        sample_vol_ml = pd.to_numeric(_sample_vol, errors="coerce").fillna(0.0)
    else:
        # Column absent -> df.get returns the default scalar; treat as no sampling.
        sample_vol_ml = pd.Series(0.0, index=df_out.index)
    is_sample = df_out.get("is_sample", False)
    if isinstance(is_sample, pd.Series):
        is_sample = is_sample.fillna(False).astype(bool)
    else:
        is_sample = pd.Series(False, index=df_out.index)

    sample_vol_l = sample_vol_ml / 1000.0

    removed_biomass_c = pd.Series(0.0, index=df_out.index)
    if biomass_g_l.notna().any():
        removed_biomass_c.loc[is_sample] = (
            (biomass_g_l * sample_vol_l / biomass_cmol_g).loc[is_sample].fillna(0.0)
        )
    df_out["C removed biomass [C-mol]"] = removed_biomass_c.cumsum()

    removed_product_c = pd.Series(0.0, index=df_out.index)
    if "Protein [g/L]" in df_out.columns and product_cmol_g is not None:
        removed_product_c.loc[is_sample] = (
            (protein_g_l * sample_vol_l / product_cmol_g).loc[is_sample].fillna(0.0)
        )
    df_out["C removed product [C-mol]"] = removed_product_c.cumsum()

    removed_acetate_c = pd.Series(0.0, index=df_out.index)
    if "Acetate [g/L]" in df_out.columns:
        removed_acetate_c.loc[is_sample] = (
            (acetate_g_l * sample_vol_l / acetate_cmol_g).loc[is_sample].fillna(0.0)
        )
    df_out["C removed acetate [C-mol]"] = removed_acetate_c.cumsum()

    removed_glucose_c = pd.Series(0.0, index=df_out.index)
    if "Glucose [g/L]" in df_out.columns:
        removed_glucose_c.loc[is_sample] = (
            (glucose_g_l * sample_vol_l / glucose_cmol_g).loc[is_sample].fillna(0.0)
        )
    df_out["C removed glucose [C-mol]"] = removed_glucose_c.cumsum()

    df_out["C biomass total [C-mol]"] = (
        df_out["C in biomass [C-mol]"].fillna(0.0) + df_out["C removed biomass [C-mol]"]
    )
    df_out["C product total [C-mol]"] = (
        df_out["C in product [C-mol]"].fillna(0.0) + df_out["C removed product [C-mol]"]
    )
    df_out["C acetate total [C-mol]"] = (
        df_out["C in acetate [C-mol]"].fillna(0.0) + df_out["C removed acetate [C-mol]"]
    )
    df_out["C glucose total [C-mol]"] = (
        df_out["C residual glucose [C-mol]"].fillna(0.0)
        + df_out["C removed glucose [C-mol]"]
    )

    df_out["C out [C-mol]"] = (
        df_out["C from CO2 [C-mol]"]
        + df_out["C biomass total [C-mol]"]
        + df_out["C product total [C-mol]"]
        + df_out["C acetate total [C-mol]"]
        + df_out["C glucose total [C-mol]"]
    )

    # --- Carbon DISTRIBUTION (at sample points) ---
    # Shows what percentage of fed carbon has gone to each compartment
    # Note: This is NOT a mass balance closure (won't sum to 100%)
    # because biomass/product that was sampled out or died is not accounted for
    # However, it shows the carbon flow through the system

    # Calculate percentage of fed C in each compartment
    df_out["C to CO2 [% of fed]"] = (
        df_out["C from CO2 [C-mol]"] / df_out["C in [C-mol]"].replace(0, np.nan)
    ) * PERCENT
    df_out["C to biomass [% of fed]"] = (
        df_out["C biomass total [C-mol]"].fillna(0)
        / df_out["C in [C-mol]"].replace(0, np.nan)
    ) * PERCENT
    df_out["C to product [% of fed]"] = (
        df_out["C product total [C-mol]"].fillna(0)
        / df_out["C in [C-mol]"].replace(0, np.nan)
    ) * PERCENT
    df_out["C residual glucose [% of fed]"] = (
        df_out["C glucose total [C-mol]"].fillna(0)
        / df_out["C in [C-mol]"].replace(0, np.nan)
    ) * PERCENT

    # Calculate a "Carbon Recovery" metric
    # C recovered = CO2 + biomass/product/glucose totals (including sampled removals)
    df_out["C recovered [%]"] = (
        (
            df_out["C from CO2 [C-mol]"]
            + df_out["C biomass total [C-mol]"]
            + df_out["C product total [C-mol]"]
            + df_out["C glucose total [C-mol]"]
        )
        / df_out["C in [C-mol]"].replace(0, np.nan)
        * PERCENT
    )

    # Keep the old column name for backward compatibility, but rename it
    df_out["C balance closure [%]"] = df_out["C recovered [%]"]
    df_out["C balance error [%]"] = 100.0 - df_out["C recovered [%]"]

    # Limit to reasonable range for display
    df_out["C recovered [%]"] = df_out["C recovered [%]"].clip(lower=0, upper=200)
    df_out["C balance closure [%]"] = df_out["C balance closure [%]"].clip(
        lower=0, upper=200
    )

    # Rigorous, assumption-free closure: the closure value ONLY at timestamps
    # where biomass is actually measured (sample points). Between samples the
    # continuous closure above relies on interpolation, so it is NOT an
    # independent mass-balance check; this column is.
    df_out["C recovered (measured) [%]"] = df_out["C recovered [%]"].where(
        measured_mask
    )

    logger.info("Calculated carbon balance (C_in, C_out, closure).")
    return df_out
