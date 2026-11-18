"""Tests for data cleaning functionality."""

import numpy as np
import pandas as pd
import pytest

from upstream_analysis.data.cleaning import FermentationDataCleaner


class TestDataCleaner:
    """Test data cleaning functionality."""

    def test_cleaner_initialization(self, sample_config):
        """Test cleaner initialization with config."""
        cleaner = FermentationDataCleaner(sample_config)
        assert cleaner is not None
        assert cleaner.median_window == 10
        assert cleaner.median_threshold == 10

    def test_preparation_phase_filter(self, sample_config):
        """Test filtering of preparation phase."""
        cleaner = FermentationDataCleaner(sample_config)

        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0, 3.0],
                "Process phase [-]": [
                    "Preparation",
                    "Batch",
                    "Fedbatch",
                    "induced Fedbatch",
                ],
                "Balance [g]": [5000.0, 4950.0, 4500.0, 4000.0],
            }
        )

        result = cleaner._filter_preparation_phase(test_df)

        # Preparation phase should be filtered out
        assert "Preparation" not in result["Process phase [-]"].values
        assert len(result) == 3  # One row removed

    def test_preparation_phase_rename(self):
        """Test renaming of preparation phase."""
        config = sample_config_with_rename = {
            "cleaning": {
                "rename_preparation_to_batch": True,
                "filter_preparation": False,
                "median_filter": False,
            },
            "processing_parameters": {"balance_mode": "reactor_weight"},
        }
        cleaner = FermentationDataCleaner(config)

        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Process phase [-]": ["Preparation", "Batch", "Fedbatch"],
                "Balance [g]": [5000.0, 4950.0, 4500.0],
            }
        )

        result = cleaner.clean_data(test_df)

        # Preparation should be renamed to Batch
        assert "Preparation" not in result["Process phase [-]"].values
        assert (result["Process phase [-]"] == "Batch").sum() == 2

    def test_balance_tare_correction(self):
        """Test balance tare correction."""
        config = {
            "cleaning": {
                "median_filter": False,
                "calculate_process_time": False,
                "balance_tare_correction": True,
                "balance_tare_threshold": -100,
            },
            "processing_parameters": {"balance_mode": "reactor_weight"},
        }
        cleaner = FermentationDataCleaner(config)

        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Balance [g]": [5000.0, -200.0, 4500.0],  # Include spike
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
            }
        )

        result = cleaner._balance_tare_correction(test_df)

        # Check that correction is applied
        assert "Balance [g]" in result.columns

    def test_rolling_median_filter(self):
        """Test rolling median filter for spike removal."""
        config = {
            "cleaning": {
                "median_filter": True,
                "median_filter_window": 3,
                "median_filter_threshold": 10,
                "calculate_process_time": False,
                "median_filter_columns": ["Balance [g]", "Temperature [°C]"],
            },
            "processing_parameters": {"balance_mode": "reactor_weight"},
        }
        cleaner = FermentationDataCleaner(config)

        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0, 3.0, 4.0],
                "Balance [g]": [5000.0, 5100.0, 5050.0, 4900.0, 4950.0],
                "Temperature [°C]": [
                    37.0,
                    37.1,
                    37.2,
                    50.0,
                    37.0,
                ],  # Spike in temperature
                "Process phase [-]": [
                    "Batch",
                    "Batch",
                    "Batch",
                    "Fedbatch",
                    "Fedbatch",
                ],
            }
        )

        result = cleaner._rolling_median_filter(
            test_df, columns_to_filter=["Temperature [°C]"]
        )

        # Check that filter is applied
        assert "Temperature [°C]" in result.columns
        # The spike should be reduced
        assert result.loc[3, "Temperature [°C]"] < 50.0

    def test_process_time_calculation(self):
        """Test process time calculation."""
        config = {
            "cleaning": {"median_filter": False, "calculate_process_time": True},
            "processing_parameters": {"balance_mode": "reactor_weight"},
        }
        cleaner = FermentationDataCleaner(config)

        test_df = pd.DataFrame(
            {
                "date_time_utc": pd.to_datetime(
                    ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"]
                ),
                "Balance [g]": [5000.0, 4950.0, 4900.0],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
            }
        )

        result = cleaner._calculate_process_time(test_df)

        # Check that process time is calculated
        assert "Process time [h]" in result.columns
        # First value should be 0
        assert result["Process time [h]"].iloc[0] == 0.0
        # Values should increase
        assert result["Process time [h]"].iloc[1] == 1.0

    def test_full_cleaning_pipeline(self, sample_config):
        """Test the full cleaning pipeline."""
        cleaner = FermentationDataCleaner(sample_config)

        test_df = pd.DataFrame(
            {
                "date_time_utc": pd.to_datetime(
                    ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"]
                ),
                "Process phase [-]": ["Batch", "Fedbatch", "Fedbatch"],
                "Balance [g]": [5000.0, 4950.0, 4900.0],
                "Temperature [°C]": [37.0, 37.1, 37.2],
            }
        )

        result = cleaner.clean_data(test_df)

        # Check that expected columns are present
        assert "Process time [h]" in result.columns
        assert "Balance [g]" in result.columns
        assert "Process phase [-]" in result.columns

    def test_missing_phase_column(self):
        """Test handling when process phase column is missing."""
        config = {
            "cleaning": {"median_filter": False, "calculate_process_time": True},
            "processing_parameters": {"balance_mode": "reactor_weight"},
        }
        cleaner = FermentationDataCleaner(config)

        test_df = pd.DataFrame(
            {
                "date_time_utc": pd.to_datetime(
                    ["2024-01-01 10:00", "2024-01-01 11:00"]
                ),
                "Balance [g]": [5000.0, 4950.0],
                # Missing Process phase column
            }
        )

        result = cleaner.clean_data(test_df)

        # Should not crash
        assert result is not None
        assert not result.empty

    def test_median_filter_disabled(self):
        """Test that median filter can be disabled."""
        config = {
            "cleaning": {"median_filter": False, "calculate_process_time": False},
            "processing_parameters": {"balance_mode": "reactor_weight"},
        }
        cleaner = FermentationDataCleaner(config)

        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Balance [g]": [5000.0, 4950.0, 4900.0],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
            }
        )

        result = cleaner.clean_data(test_df)

        # Values should be unchanged
        pd.testing.assert_series_equal(
            result["Balance [g]"], test_df["Balance [g]"], check_names=False
        )


class TestEdgeCases:
    """Test edge cases in data cleaning."""

    def test_empty_dataframe(self, sample_config):
        """Test handling of empty DataFrame."""
        cleaner = FermentationDataCleaner(sample_config)
        empty_df = pd.DataFrame()

        result = cleaner.clean_data(empty_df)
        assert result.empty

    def test_all_nan_column(self, sample_config):
        """Test handling of column with all NaN values."""
        cleaner = FermentationDataCleaner(sample_config)

        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Balance [g]": [np.nan, np.nan, np.nan],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
            }
        )

        result = cleaner.clean_data(test_df)
        # Should handle NaN gracefully
        assert result["Balance [g]"].isna().all()

    def test_single_row_dataframe(self, sample_config):
        """Test handling of single row DataFrame."""
        cleaner = FermentationDataCleaner(sample_config)

        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0],
                "Balance [g]": [5000.0],
                "Process phase [-]": ["Batch"],
            }
        )

        result = cleaner.clean_data(test_df)
        assert len(result) == 1
        assert not result.empty
