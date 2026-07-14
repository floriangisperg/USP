"""Pytest fixtures for USP testing."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from upstream_analysis.config.loader import load_config
from upstream_analysis.data.parser import BioreactorDataParser, OfflineMeasurementsParser
from upstream_analysis.data.processor import FermentationDataProcessor


@pytest.fixture
def sample_config():
    """Load test configuration."""
    config_path = Path(__file__).parent / 'fixtures' / 'test_config.yaml'
    if config_path.exists():
        return load_config(config_path)
    else:
        # Minimal config for testing
        return {
            'parsing': {
                'bioreactor': {'decimal_separator': ',', 'column_separator': ';'},
                'offline': {'decimal_separator': '.', 'column_separator': ','}
            },
            'column_mapping': {},
            'cleaning': {
                'median_filter': True,
                'median_filter_window': 10,
                'median_filter_threshold': 10,
                'calculate_process_time': True
            },
            'alignment': {
                'online_time_col': 'Process time [h]',
                'offline_time_col': 'process_time',
                'tolerance_hours': 0.02
            },
            'processing': {
                'processing_parameters': {
                    'reactor_setup': {
                        'initial_volume_ml': 1100.0
                    },
                    'gas_params': {
                        'molar_volume_air_l_mol': 22.414,
                        'o2_in_air_percent': 21.0,
                        'co2_in_air_percent': 0.04,
                        'calculate_cer': True,
                        'calculate_our': True,
                        'calculate_rq': True
                    },
                    'feed_definitions': {
                        'feed_1': {
                            'density_g_ml': 1.1
                        }
                    },
                    'base_titrant': {
                        'total_added_g': 100.0,
                        'density_g_ml': 1.0
                    },
                    'biomass_params': {
                        'molar_mass_g_c_mol': 26.565
                    },
                    'induced_phases': ['induced Fedbatch']
                }
            },
            'analysis': {
                'delisa_media_config': {
                    'feed_1': {
                        'components': {
                            'glucose': {'concentration_g_l': 440.0}
                        }
                    },
                    'batch_medium': {
                        'components': {
                            'glucose': {'concentration_g_l': 22.0}
                        }
                    }
                }
            }
        }


@pytest.fixture
def sample_bioreactor_df():
    """Create sample bioreactor data for testing."""
    data = {
        'Process time [h]': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'Process phase [-]': ['Batch', 'Batch', 'Fedbatch', 'Fedbatch', 'Fedbatch',
                            'induced Fedbatch', 'induced Fedbatch', 'induced Fedbatch', 'induced Fedbatch'],
        'Balance [g]': [5000.0, 4950.0, 4900.0, 4500.0, 4000.0, 3500.0, 3000.0, 2500.0, 2000.0],
        'Air flow [L/min]': [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.55],
        'O2 flow [L/min]': [0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'Offgas CO2 [%]': [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5],
        'Offgas O2 [%]': [20.5, 20.3, 19.8, 19.5, 19.2, 19.0, 18.8, 18.5, 18.3],
        'Total flow [L/min]': [1.0, 1.05, 1.1, 1.25, 1.4, 1.55, 1.65, 1.75, 1.85],
        'pH [-]': [7.0, 6.9, 6.8, 6.7, 6.6, 6.5, 6.45, 6.4, 6.35],
        'Temperature [°C]': [37.0, 37.1, 37.2, 37.1, 37.0, 36.9, 37.0, 37.1, 37.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_bioreactor_with_base_df():
    """Create sample bioreactor data with base pump duration for testing."""
    data = {
        'Process time [h]': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'Process phase [-]': ['Batch', 'Batch', 'Fedbatch', 'Fedbatch', 'Fedbatch',
                            'induced Fedbatch', 'induced Fedbatch', 'induced Fedbatch', 'induced Fedbatch'],
        'Balance [g]': [5000.0, 4950.0, 4900.0, 4500.0, 4000.0, 3500.0, 3000.0, 2500.0, 2000.0],
        'Air flow [L/min]': [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.55],
        'O2 flow [L/min]': [0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'Offgas CO2 [%]': [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5],
        'Offgas O2 [%]': [20.5, 20.3, 19.8, 19.5, 19.2, 19.0, 18.8, 18.5, 18.3],
        'Total flow [L/min]': [1.0, 1.05, 1.1, 1.25, 1.4, 1.55, 1.65, 1.75, 1.85],
        'pH [-]': [7.0, 6.9, 6.8, 6.7, 6.6, 6.5, 6.45, 6.4, 6.35],
        'Temperature [°C]': [37.0, 37.1, 37.2, 37.1, 37.0, 36.9, 37.0, 37.1, 37.0],
        'Base pump duration [-]': [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_offline_df():
    """Create sample offline measurement data."""
    data = {
        'process_time': [1.0, 2.5, 4.0, 5.5, 7.0],
        'Biomass [g/L]': [5.0, 15.0, 25.0, 35.0, 40.0],
        'Glucose [g/L]': [20.0, 10.0, 5.0, 2.0, 1.0],
        'Protein soluble [g/L]': [0.5, 2.0, 3.5, 5.0, 6.0],
        'Sample volume [mL]': [10.0, 10.0, 10.0, 10.0, 10.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_offline_with_protein_df():
    """Create sample offline measurement data with total protein."""
    data = {
        'process_time': [1.0, 2.5, 4.0, 5.5, 7.0],
        'Biomass [g/L]': [5.0, 15.0, 25.0, 35.0, 40.0],
        'Glucose [g/L]': [20.0, 10.0, 5.0, 2.0, 1.0],
        'Protein [g/L]': [0.5, 2.0, 3.5, 5.0, 6.0],
        'Protein soluble [g/L]': [0.3, 1.5, 2.5, 3.5, 4.0],
        'Protein IB [g/L]': [0.2, 0.5, 1.0, 1.5, 2.0],
        'Sample volume [mL]': [10.0, 10.0, 10.0, 10.0, 10.0]
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
    df['Biomass [g/L]'] = np.nan
    df['Glucose [g/L]'] = np.nan
    df['Protein soluble [g/L]'] = np.nan
    df['Sample volume [mL]'] = np.nan
    df['is_sample'] = False

    # Add some sample points
    sample_indices = [1, 3, 5, 7]
    for i, idx in enumerate(sample_indices):
        df.loc[idx, 'Biomass [g/L]'] = sample_offline_df.loc[i, 'Biomass [g/L]']
        df.loc[idx, 'Glucose [g/L]'] = sample_offline_df.loc[i, 'Glucose [g/L]']
        df.loc[idx, 'Protein soluble [g/L]'] = sample_offline_df.loc[i, 'Protein soluble [g/L]']
        df.loc[idx, 'Sample volume [mL]'] = sample_offline_df.loc[i, 'Sample volume [mL]']
        df.loc[idx, 'is_sample'] = True

    return df


@pytest.fixture
def minimal_config():
    """Create minimal configuration for basic testing."""
    return {
        'parsing': {
            'bioreactor': {},
            'offline': {}
        },
        'processing': {
            'processing_parameters': {
                'reactor_setup': {
                    'initial_volume_ml': 1000.0
                },
                'feed_definitions': {
                    'feed_1': {
                        'density_g_ml': 1.0
                    }
                },
                'base_titrant': {
                    'total_added_g': 50.0,
                    'density_g_ml': 1.0
                },
                'gas_params': {
                    'molar_volume_air_l_mol': 22.414,
                    'o2_in_air_percent': 21.0,
                    'co2_in_air_percent': 0.04
                }
            }
        },
        'cleaning': {
            'median_filter': False,
            'calculate_process_time': True
        }
    }


@pytest.fixture
def edge_case_config():
    """Create config for testing edge cases."""
    return {
        'parsing': {
            'bioreactor': {},
            'offline': {}
        },
        'processing': {
            'processing_parameters': {
                'reactor_setup': {
                    'initial_volume_ml': 1100.0
                },
                'gas_params': {
                    'molar_volume_air_l_mol': 22.414,
                    'o2_in_air_percent': 21.0,
                    'co2_in_air_percent': 0.04,
                    'calculate_cer': True,
                    'calculate_our': True,
                    'calculate_rq': True,
                    'calculate_specific_rates': True
                },
                'feed_definitions': {
                    'feed_1': {
                        'density_g_ml': 1.1
                    }
                },
                'base_titrant': {
                    'total_added_g': 100.0,
                    'density_g_ml': 1.0
                },
                'biomass_params': {
                    'molar_mass_g_c_mol': 26.565
                },
                'induced_phases': ['induced Fedbatch']
            }
        },
        'cleaning': {
            'median_filter': True,
            'median_filter_window': 10,
            'median_filter_threshold': 10,
            'calculate_process_time': True
        },
        'analysis': {
            'delisa_media_config': {
                'feed_1': {
                    'components': {
                        'glucose': {'concentration_g_l': 440.0}
                    }
                },
                'batch_medium': {
                    'components': {
                        'glucose': {'concentration_g_l': 22.0}
                    }
                }
            }
        }
    }


@pytest.fixture
def empty_dataframe():
    """Create empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def nan_dataframe():
    """Create DataFrame with NaN values for edge case testing."""
    return pd.DataFrame({
        'Process time [h]': [0.0, 1.0, np.nan, 3.0, 4.0],
        'Process phase [-]': ['Batch', np.nan, 'Fedbatch', 'Fedbatch', np.nan],
        'Balance [g]': [5000.0, np.nan, 4500.0, np.nan, 3500.0],
        'Air flow [L/min]': [1.0, 1.1, np.nan, 1.3, 1.4],
        'Offgas CO2 [%]': [0.5, np.nan, 1.2, 1.5, np.nan],
        'Offgas O2 [%]': [20.5, 20.3, np.nan, 19.5, 19.2]
    })
