"""Biological rates and yields: growth (mu), substrate (qS), and product (qP) rates."""

import logging

import numpy as np
import pandas as pd

from ...core.constants import (
    DEFAULT_FEED_GLUCOSE_G_L,
    DEFAULT_INITIAL_GLUCOSE_G_L,
    DEFAULT_INITIAL_VOLUME_ML,
    GLUCOSE_MOLAR_MASS_G_PER_C_MOL,
)

logger = logging.getLogger(__name__)


def calculate_rates_and_yields(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate biological rates and yields from time-series data.

    Computes specific growth rate (mu), substrate consumption (qS glucose), and
    product formation (qP) rates, handling sparse offline data through
    interpolation and applying smoothing to reduce derivative noise. When a
    carbon balance is present, also derives offgas-based glucose-uptake rates.

    Args:
        df: DataFrame with 'Process time [h]', volume, feed, and optional
            offline concentration columns.
        config: Merged configuration dict (``rate_params`` controls smoothing /
            interpolation).

    Returns:
        DataFrame with added rate and yield columns.
    """
    logger.info("Calculating rates (mu, qs, qP)...")
    df_out = df.copy()

    time_col = "Process time [h]"
    if time_col not in df_out.columns:
        logger.warning(f"Time column '{time_col}' not found. Cannot calculate rates.")
        return df_out

    time_h = pd.to_numeric(df_out[time_col], errors="coerce")
    if time_h.isna().all():
        logger.warning("Time column is all NaN. Cannot calculate rates.")
        return df_out

    rate_config = {}
    rate_config.update(config.get("processing", {}).get("rate_params", {}) or {})
    rate_config.update(
        config.get("processing_parameters", {}).get("rate_params", {}) or {}
    )

    apply_smoothing = rate_config.get("apply_smoothing", True)
    window_points = int(rate_config.get("smoothing_window_points", 5))
    min_periods = int(rate_config.get("smoothing_min_periods", 1))

    def _smooth(series: pd.Series) -> pd.Series:
        if not apply_smoothing or window_points <= 1:
            return series
        return series.rolling(
            window=window_points, center=True, min_periods=min_periods
        ).mean()

    def _interp_series(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        mask = time_h.notna() & s.notna()
        if mask.sum() < 2:
            return s
        t_vals = time_h[mask].to_numpy()
        v_vals = s[mask].to_numpy()
        order = t_vals.argsort()
        t_sorted = t_vals[order]
        v_sorted = v_vals[order]

        interp_method = rate_config.get("interpolation_method", "spline")
        t_all = time_h.to_numpy()

        if interp_method == "spline":
            try:
                from scipy.interpolate import UnivariateSpline

                spline_order = int(rate_config.get("spline_order", 3))
                spline_s = rate_config.get("spline_smoothing", 0.0)
                try:
                    spline_s = float(spline_s)
                except (TypeError, ValueError):
                    spline_s = 0.0

                # Deduplicate time points for spline stability
                dedup = (
                    pd.DataFrame({"t": t_sorted, "v": v_sorted})
                    .groupby("t", as_index=False)
                    .mean()
                )
                t_unique = dedup["t"].to_numpy()
                v_unique = dedup["v"].to_numpy()

                if len(t_unique) >= spline_order + 1:
                    spline = UnivariateSpline(
                        t_unique, v_unique, k=spline_order, s=spline_s
                    )
                    v_all = spline(t_all)
                    return pd.Series(v_all, index=df_out.index)
                logger.warning(
                    "Not enough points for spline interpolation, falling back to linear."
                )
            except Exception as e:
                logger.warning(
                    f"Spline interpolation failed, falling back to linear: {e}"
                )

        v_all = np.interp(t_all, t_sorted, v_sorted)
        return pd.Series(v_all, index=df_out.index)

    dt_h = time_h.diff().fillna(0.0)
    dt_h = dt_h.where(dt_h > 0, np.nan)

    v_corr_l = (
        pd.to_numeric(df_out.get("Volume corrected [mL]"), errors="coerce") / 1000.0
    )

    biomass_g_l = _interp_series(
        df_out.get("Biomass [g/L]", pd.Series(index=df_out.index, dtype=float))
    )
    biomass_g = biomass_g_l * v_corr_l

    protein_col = None
    if "Protein total calculated [g/L]" in df_out.columns:
        protein_col = "Protein total calculated [g/L]"
    elif "Protein [g/L]" in df_out.columns:
        protein_col = "Protein [g/L]"
    protein_g_l = _interp_series(
        df_out.get(protein_col, pd.Series(index=df_out.index, dtype=float))
    )
    protein_g = protein_g_l * v_corr_l

    glucose_g_l = _interp_series(
        df_out.get("Glucose [g/L]", pd.Series(index=df_out.index, dtype=float))
    )
    glucose_g = glucose_g_l * v_corr_l

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

    sample_out_glc_g = pd.Series(0.0, index=df_out.index)
    if glucose_g_l.notna().any():
        sample_out_glc_g.loc[is_sample] = (
            (glucose_g_l * sample_vol_l).loc[is_sample].fillna(0.0)
        )
    sample_out_glc_cum = sample_out_glc_g.cumsum()

    sample_out_protein_g = pd.Series(0.0, index=df_out.index)
    if protein_g_l.notna().any():
        sample_out_protein_g.loc[is_sample] = (
            (protein_g_l * sample_vol_l).loc[is_sample].fillna(0.0)
        )
    sample_out_protein_cum = sample_out_protein_g.cumsum()

    feed_ml = pd.to_numeric(df_out.get("Feed [mL]", 0.0), errors="coerce").fillna(0.0)
    d_feed_ml = feed_ml.diff().fillna(0.0).clip(lower=0.0)

    feed_conc_g_l = None
    if isinstance(config, dict):
        feed_conc_g_l = (
            config.get("analysis", {})
            .get("delisa_media_config", {})
            .get("feed_1", {})
            .get("components", {})
            .get("glucose", {})
            .get("concentration_g_l")
        )
    if feed_conc_g_l is None:
        feed_conc_g_l = DEFAULT_FEED_GLUCOSE_G_L
    try:
        feed_conc_g_l = float(feed_conc_g_l)
    except (TypeError, ValueError):
        feed_conc_g_l = DEFAULT_FEED_GLUCOSE_G_L

    feed_g = d_feed_ml / 1000.0 * feed_conc_g_l
    feed_g_cum = feed_g.cumsum()

    initial_volume_ml = None
    initial_glucose_g_l = None
    if isinstance(config, dict):
        initial_volume_ml = (
            config.get("processing_parameters", {})
            .get("reactor_setup", {})
            .get("initial_volume_ml")
        )
        initial_glucose_g_l = (
            config.get("analysis", {})
            .get("delisa_media_config", {})
            .get("batch_medium", {})
            .get("components", {})
            .get("glucose", {})
            .get("concentration_g_l")
        )
    try:
        initial_volume_ml = float(initial_volume_ml)
    except (TypeError, ValueError):
        initial_volume_ml = DEFAULT_INITIAL_VOLUME_ML
    try:
        initial_glucose_g_l = float(initial_glucose_g_l)
    except (TypeError, ValueError):
        initial_glucose_g_l = DEFAULT_INITIAL_GLUCOSE_G_L

    initial_glucose_g = (initial_volume_ml / 1000.0) * initial_glucose_g_l

    glucose_consumed_g = initial_glucose_g + feed_g_cum - sample_out_glc_cum - glucose_g
    glucose_consumed_rate_g_h = glucose_consumed_g.diff() / dt_h
    glucose_consumed_rate_g_h = _smooth(glucose_consumed_rate_g_h)

    df_out["Glucose consumed [g]"] = glucose_consumed_g
    df_out["Glucose consumed rate [g/h]"] = glucose_consumed_rate_g_h

    total_biomass_g = biomass_g
    df_out["Biomass total [g]"] = total_biomass_g

    qs_glc = glucose_consumed_rate_g_h / total_biomass_g.replace(0, np.nan)
    df_out["qS glucose [g/g/h]"] = qs_glc

    # Growth rate from biomass concentration
    mu = pd.Series(np.nan, index=df_out.index)
    valid_x = biomass_g_l > 0
    if valid_x.any():
        mu = np.log(biomass_g_l.where(valid_x)).diff() / dt_h
        mu = _smooth(mu)
    df_out["mu [1/h]"] = mu

    # Product formation rate
    produced_protein_g = protein_g + sample_out_protein_cum
    protein_rate_g_h = produced_protein_g.diff() / dt_h
    protein_rate_g_h = _smooth(protein_rate_g_h)
    df_out["Protein produced [g]"] = produced_protein_g
    df_out["Protein production rate [g/h]"] = protein_rate_g_h
    df_out["qP total [g/g/h]"] = protein_rate_g_h / total_biomass_g.replace(0, np.nan)

    # Offgas-derived qs from carbon balance (if available)
    if "C out [C-mol]" in df_out.columns:
        c_out = pd.to_numeric(df_out["C out [C-mol]"], errors="coerce")
        c_out_rate = c_out.diff() / dt_h
        c_out_rate = _smooth(c_out_rate)

        glucose_cmol_g = GLUCOSE_MOLAR_MASS_G_PER_C_MOL
        if isinstance(config, dict):
            glucose_cmol_g_cfg = (
                config.get("processing_parameters", {})
                .get("substrate_properties", {})
                .get("glucose", {})
                .get("molar_mass_g_c_mol")
            )
            if glucose_cmol_g_cfg is not None:
                try:
                    glucose_cmol_g = float(glucose_cmol_g_cfg)
                except (TypeError, ValueError):
                    pass

        offgas_glc_rate_g_h = c_out_rate * glucose_cmol_g
        df_out["Glucose uptake offgas rate [g/h]"] = offgas_glc_rate_g_h
        df_out["qS offgas [g/g/h]"] = offgas_glc_rate_g_h / total_biomass_g.replace(
            0, np.nan
        )

        offgas_glc_cum = (offgas_glc_rate_g_h * dt_h).cumsum()
        scale = np.nan
        valid = glucose_consumed_g.notna() & offgas_glc_cum.notna()
        if valid.any():
            idx = valid[valid].index[-1]
            denom = offgas_glc_cum.loc[idx]
            numer = glucose_consumed_g.loc[idx]
            if pd.notna(denom) and denom > 0:
                scale = numer / denom

        if pd.notna(scale):
            df_out["Offgas glucose scale [-]"] = scale
            df_out["Glucose uptake offgas rate calibrated [g/h]"] = (
                offgas_glc_rate_g_h * scale
            )
            df_out["qS offgas calibrated [g/g/h]"] = (
                offgas_glc_rate_g_h * scale
            ) / total_biomass_g.replace(0, np.nan)

    return df_out
