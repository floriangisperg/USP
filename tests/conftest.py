"""Pytest fixtures for USP testing."""

import os
import sys
from pathlib import Path

# Headless-safe plotting for all tests (no Tk/window backends).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from upstream_analysis.config.loader import load_config  # noqa: E402
from upstream_analysis.data.processor import FermentationDataProcessor  # noqa: E402


@pytest.fixture
def sample_config():
    """Load test configuration."""
    config_path = Path(__file__).parent / "fixtures" / "test_config.yaml"
    if config_path.exists():
        return load_config(config_path)
    else:
        # Minimal config for testing
        return {
            "parsing": {
                "bioreactor": {"decimal_separator": ",", "column_separator": ";"},
                "offline": {"decimal_separator": ".", "column_separator": ","},
            },
            "column_mapping": {},
            "cleaning": {
                "median_filter": True,
                "median_filter_window": 10,
                "median_filter_threshold": 10,
                "calculate_process_time": True,
            },
            "alignment": {
                "online_time_col": "Process time [h]",
                "offline_time_col": "process_time",
                "tolerance_hours": 0.02,
            },
            "processing": {
                "processing_parameters": {
                    "reactor_setup": {"initial_volume_ml": 1100.0},
                    "gas_params": {
                        "molar_volume_air_l_mol": 22.414,
                        "o2_in_air_percent": 21.0,
                        "co2_in_air_percent": 0.04,
                        "calculate_cer": True,
                        "calculate_our": True,
                        "calculate_rq": True,
                    },
                    "feed_definitions": {"feed_1": {"density_g_ml": 1.1}},
                    "base_titrant": {"total_added_g": 100.0, "density_g_ml": 1.0},
                    "biomass_params": {"molar_mass_g_c_mol": 26.565},
                    "induced_phases": ["induced Fedbatch"],
                }
            },
            "analysis": {
                "delisa_media_config": {
                    "feed_1": {"components": {"glucose": {"concentration_g_l": 440.0}}},
                    "batch_medium": {
                        "components": {"glucose": {"concentration_g_l": 22.0}}
                    },
                }
            },
        }


@pytest.fixture
def sample_bioreactor_df():
    """Create sample bioreactor data for testing."""
    data = {
        "Process time [h]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "Process phase [-]": [
            "Batch",
            "Batch",
            "Fedbatch",
            "Fedbatch",
            "Fedbatch",
            "induced Fedbatch",
            "induced Fedbatch",
            "induced Fedbatch",
            "induced Fedbatch",
        ],
        "Balance [g]": [
            5000.0,
            4950.0,
            4900.0,
            4500.0,
            4000.0,
            3500.0,
            3000.0,
            2500.0,
            2000.0,
        ],
        "Air flow [L/min]": [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.55],
        "O2 flow [L/min]": [0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "Offgas CO2 [%]": [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5],
        "Offgas O2 [%]": [20.5, 20.3, 19.8, 19.5, 19.2, 19.0, 18.8, 18.5, 18.3],
        "Total flow [L/min]": [1.0, 1.05, 1.1, 1.25, 1.4, 1.55, 1.65, 1.75, 1.85],
        "pH [-]": [7.0, 6.9, 6.8, 6.7, 6.6, 6.5, 6.45, 6.4, 6.35],
        "Temperature [°C]": [37.0, 37.1, 37.2, 37.1, 37.0, 36.9, 37.0, 37.1, 37.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_bioreactor_with_base_df():
    """Create sample bioreactor data with base pump duration for testing."""
    data = {
        "Process time [h]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "Process phase [-]": [
            "Batch",
            "Batch",
            "Fedbatch",
            "Fedbatch",
            "Fedbatch",
            "induced Fedbatch",
            "induced Fedbatch",
            "induced Fedbatch",
            "induced Fedbatch",
        ],
        "Balance [g]": [
            5000.0,
            4950.0,
            4900.0,
            4500.0,
            4000.0,
            3500.0,
            3000.0,
            2500.0,
            2000.0,
        ],
        "Air flow [L/min]": [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.55],
        "O2 flow [L/min]": [0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "Offgas CO2 [%]": [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5],
        "Offgas O2 [%]": [20.5, 20.3, 19.8, 19.5, 19.2, 19.0, 18.8, 18.5, 18.3],
        "Total flow [L/min]": [1.0, 1.05, 1.1, 1.25, 1.4, 1.55, 1.65, 1.75, 1.85],
        "pH [-]": [7.0, 6.9, 6.8, 6.7, 6.6, 6.5, 6.45, 6.4, 6.35],
        "Temperature [°C]": [37.0, 37.1, 37.2, 37.1, 37.0, 36.9, 37.0, 37.1, 37.0],
        "Base pump duration [-]": [
            0.0,
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_offline_df():
    """Create sample offline measurement data."""
    data = {
        "process_time": [1.0, 2.5, 4.0, 5.5, 7.0],
        "Biomass [g/L]": [5.0, 15.0, 25.0, 35.0, 40.0],
        "Glucose [g/L]": [20.0, 10.0, 5.0, 2.0, 1.0],
        "Protein soluble [g/L]": [0.5, 2.0, 3.5, 5.0, 6.0],
        "Sample volume [mL]": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_offline_with_protein_df():
    """Create sample offline measurement data with total protein."""
    data = {
        "process_time": [1.0, 2.5, 4.0, 5.5, 7.0],
        "Biomass [g/L]": [5.0, 15.0, 25.0, 35.0, 40.0],
        "Glucose [g/L]": [20.0, 10.0, 5.0, 2.0, 1.0],
        "Protein [g/L]": [0.5, 2.0, 3.5, 5.0, 6.0],
        "Protein soluble [g/L]": [0.3, 1.5, 2.5, 3.5, 4.0],
        "Protein IB [g/L]": [0.2, 0.5, 1.0, 1.5, 2.0],
        "Sample volume [mL]": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_processor(sample_config):
    """Create configured processor for testing."""
    return FermentationDataProcessor(sample_config)


@pytest.fixture
def sample_cleaned_df(sample_bioreactor_with_base_df):
    """Create sample cleaned data frame for testing."""
    df = sample_bioreactor_with_base_df.copy()
    # Add calculated columns that would be present after cleaning
    return df


@pytest.fixture
def sample_aligned_df(sample_cleaned_df, sample_offline_df):
    """Create sample aligned data frame for testing."""
    df = sample_cleaned_df.copy()
    # Add offline data aligned to online data
    df["Biomass [g/L]"] = np.nan
    df["Glucose [g/L]"] = np.nan
    df["Protein soluble [g/L]"] = np.nan
    df["Sample volume [mL]"] = np.nan
    df["is_sample"] = False

    # Add some sample points
    sample_indices = [1, 3, 5, 7]
    for i, idx in enumerate(sample_indices):
        df.loc[idx, "Biomass [g/L]"] = sample_offline_df.loc[i, "Biomass [g/L]"]
        df.loc[idx, "Glucose [g/L]"] = sample_offline_df.loc[i, "Glucose [g/L]"]
        df.loc[idx, "Protein soluble [g/L]"] = sample_offline_df.loc[
            i, "Protein soluble [g/L]"
        ]
        df.loc[idx, "Sample volume [mL]"] = sample_offline_df.loc[
            i, "Sample volume [mL]"
        ]
        df.loc[idx, "is_sample"] = True

    return df


@pytest.fixture
def minimal_config():
    """Create minimal configuration for basic testing."""
    return {
        "parsing": {"bioreactor": {}, "offline": {}},
        "processing": {
            "processing_parameters": {
                "reactor_setup": {"initial_volume_ml": 1000.0},
                "feed_definitions": {"feed_1": {"density_g_ml": 1.0}},
                "base_titrant": {"total_added_g": 50.0, "density_g_ml": 1.0},
                "gas_params": {
                    "molar_volume_air_l_mol": 22.414,
                    "o2_in_air_percent": 21.0,
                    "co2_in_air_percent": 0.04,
                },
            }
        },
        "cleaning": {"median_filter": False, "calculate_process_time": True},
    }


@pytest.fixture
def edge_case_config():
    """Create config for testing edge cases."""
    return {
        "parsing": {"bioreactor": {}, "offline": {}},
        "processing": {
            "processing_parameters": {
                "reactor_setup": {"initial_volume_ml": 1100.0},
                "gas_params": {
                    "molar_volume_air_l_mol": 22.414,
                    "o2_in_air_percent": 21.0,
                    "co2_in_air_percent": 0.04,
                    "calculate_cer": True,
                    "calculate_our": True,
                    "calculate_rq": True,
                    "calculate_specific_rates": True,
                },
                "feed_definitions": {"feed_1": {"density_g_ml": 1.1}},
                "base_titrant": {"total_added_g": 100.0, "density_g_ml": 1.0},
                "biomass_params": {"molar_mass_g_c_mol": 26.565},
                "induced_phases": ["induced Fedbatch"],
            }
        },
        "cleaning": {
            "median_filter": True,
            "median_filter_window": 10,
            "median_filter_threshold": 10,
            "calculate_process_time": True,
        },
        "analysis": {
            "delisa_media_config": {
                "feed_1": {"components": {"glucose": {"concentration_g_l": 440.0}}},
                "batch_medium": {
                    "components": {"glucose": {"concentration_g_l": 22.0}}
                },
            }
        },
    }


@pytest.fixture
def empty_dataframe():
    """Create empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def nan_dataframe():
    """Create DataFrame with NaN values for edge case testing."""
    return pd.DataFrame(
        {
            "Process time [h]": [0.0, 1.0, np.nan, 3.0, 4.0],
            "Process phase [-]": ["Batch", np.nan, "Fedbatch", "Fedbatch", np.nan],
            "Balance [g]": [5000.0, np.nan, 4500.0, np.nan, 3500.0],
            "Air flow [L/min]": [1.0, 1.1, np.nan, 1.3, 1.4],
            "Offgas CO2 [%]": [0.5, np.nan, 1.2, 1.5, np.nan],
            "Offgas O2 [%]": [20.5, 20.3, np.nan, 19.5, 19.2],
        }
    )


# --- Synthetic fermentation run fixture -------------------------------------
# Ported from examples/synthetic_example/generate_data.py (folder removed);
# builds a self-contained run folder so pipeline/consolidation tests have
# realistic data without shipping any example folder.

import matplotlib  # noqa: E402

matplotlib.use("Agg")


def _build_synthetic_run(dest: Path) -> Path:
    """Write a synthetic fermentation run (28 h fed-batch, seeded) to dest."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2026-01-15 09:00")
    dt_h = 2.0 / 60
    v0_l, x0, mu_batch, mu_fed, y_xs, s_feed_g_ml, feed_density = (
        1.5,
        0.5,
        0.28,
        0.08,
        0.45,
        0.4,
        1.15,
    )

    def biomass(t):
        return np.where(
            t < 0,
            x0 * 0.2,
            np.where(
                t < 10,
                x0 * np.exp(mu_batch * np.clip(t, 0, None)),
                x0 * np.exp(mu_batch * 10) * np.exp(mu_fed * (t - 10)),
            ),
        )

    def dxdt(t):
        return np.where(t < 10, mu_batch, mu_fed) * biomass(t)

    t = np.arange(-0.25, 28.0 + dt_h, dt_h)
    stamps = start + pd.to_timedelta(t, unit="h")
    phase = np.where(
        t < 0,
        "Preparation",
        np.where(t < 10, "Batch", np.where(t < 16, "Fedbatch", "induced Fedbatch")),
    )
    x = biomass(t)
    feed_glu = np.maximum(dxdt(t), 0) * v0_l / y_xs
    feed_ml_h = feed_glu / s_feed_g_ml
    feed_rate = np.where(t >= 10, feed_ml_h * feed_density, 0.0)
    feed_cum_g = np.cumsum(feed_rate) * dt_h
    balance = 800.0 - feed_cum_g + rng.normal(0, 0.05, t.size)
    our = np.where(t >= 0, dxdt(t) / 26.565 * 1000, 0.0)
    vol_l = v0_l + feed_cum_g / feed_density / 1000
    f_mol_h = 1.0 * 60 / 22.414
    d_o2 = our * vol_l / (f_mol_h * 1000) * 100 * 1.02
    exit_o2 = 20.9 - d_o2 + rng.normal(0, 0.03, t.size)
    exit_co2 = (
        0.04 + our * 1.0 * vol_l / (f_mol_h * 1000) * 100 + rng.normal(0, 0.02, t.size)
    )
    po2 = np.where(
        (t > 9.2) & (t < 10.4), 12.0, np.where(t < 0, 45.0, 38.0)
    ) + rng.normal(0, 2.0, t.size)
    ph = np.where(t < 16, 7.0, 7.1) + rng.normal(0, 0.02, t.size)
    od = np.clip(x / 0.35 * (1 + rng.normal(0, 0.03, t.size)), 0.05, None)
    dur = np.maximum(t, 0) * 3600

    online = pd.DataFrame(
        {
            "Date Time UTC": stamps.strftime("%d.%m.%Y %H:%M"),
            "Batch Time, sec": (t * 3600).round(0).astype(int),
            "Phase": phase,
            "AirFlow, l/min": np.round(1.0 + rng.normal(0, 0.02, t.size), 3),
            "TotalFlow, l/min": np.round(1.0 + rng.normal(0, 0.02, t.size), 3),
            "GasMix": 21,
            "Pump2.Duration, -": np.round(np.where(t < 0, np.nan, dur), 0),
            "Pump4.Duration, -": np.round(np.where(t < 10, np.nan, (t - 10) * 3600), 0),
            "Stirrer, 1/min": np.round(1200 + rng.normal(0, 8, t.size), 0),
            "Temperature, °C": np.round(37.0 + rng.normal(0, 0.05, t.size), 2),
            "Balance, g": np.round(balance, 3),
            "pH, -": np.round(ph, 3),
            "pO2, %": np.round(po2, 1),
            "GenericGasAnalyser.ExitCO2, %": np.round(exit_co2, 3),
            "GenericGasAnalyser.ExitO2, %": np.round(exit_o2, 3),
            "OD600": np.round(od, 3),
        }
    )

    def de(col):
        s = online[col]
        if s.dtype.kind in "fc":
            return s.map(lambda v: "" if pd.isna(v) else f"{v:.3f}".replace(".", ","))
        return s

    out = pd.DataFrame({c: de(c) for c in online.columns})
    data_dir = dest / "data"
    data_dir.mkdir(parents=True)
    with open(data_dir / "rawdata.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write("DASGIP-Export: synthetic fermenter SIM01 (generated data)\n")
        out.to_csv(fh, sep=";", index=False)

    s_t = np.array([1.5, 4.0, 7.0, 9.5, 12.0, 15.0, 19.0, 23.0, 26.5])
    s_x = biomass(s_t) * (1 + rng.normal(0, 0.04, s_t.size))
    s_glu = np.where(s_t < 10, np.maximum(20.0 - (s_x - x0) / y_xs, 0.15), 0.3)
    s_prot = np.where(s_t > 16, 0.12 * (s_x - biomass(16.0)), 0.0)
    pd.DataFrame(
        {
            "Process Time [h]": s_t,
            "OD": np.round(s_x / 0.35, 2),
            "Biomass [g/L]": np.round(s_x, 3),
            "Glucose [g/L]": np.round(s_glu * (1 + rng.normal(0, 0.08, s_t.size)), 3),
            "Acetate [g/L]": np.round(
                np.clip(
                    0.35 * np.maximum(0, s_t - 6) / 4 - 0.15 * np.maximum(0, s_t - 12),
                    0,
                    1.2,
                ),
                3,
            ),
            "Protein concentration [g/L]": np.round(s_prot, 3),
            "Protein soluble [g/L]": np.round(s_prot * 0.6, 3),
            "Protein IB [g/L]": np.round(s_prot * 0.4, 3),
            "Sample volume [mL]": 10.0,
        }
    ).to_excel(data_dir / "samples.xlsx", index=False)

    (dest / "run_config.yaml").write_text(
        "metadata:\n"
        '  fermentation_id: "synthetic_demo"\n'
        '  strain: "E. coli BL21(DE3) [simulated]"\n'
        "processing:\n"
        "  processing_parameters:\n"
        "    reactor_setup:\n"
        "      initial_volume_ml: 1500.0\n"
        "    balance_mode: 'feed_source'\n"
        "    feed_definitions:\n"
        "      feed_1:\n"
        "        name: 'Glucose feed (400 g/L)'\n"
        "        density_g_ml: 1.15\n"
        "    base_titrant:\n"
        "      compound: 'NH4OH'\n"
        "      molarity_mol_l: 6.95\n"
        "      density_g_ml: 0.947\n"
        "      total_added_g: 15.0\n"
        "    product_properties:\n"
        "      name: 'demo_protein'\n"
        "  induced_phases:\n"
        "    - 'induced Fedbatch'\n",
        encoding="utf-8",
    )
    return dest


@pytest.fixture(scope="session")
def synthetic_run(tmp_path_factory):
    """Session-wide synthetic fermentation run folder for pipeline tests."""
    return _build_synthetic_run(tmp_path_factory.mktemp("synthetic_run"))
