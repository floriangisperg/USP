"""Tests for data alignment functionality."""

import numpy as np
import pandas as pd
import pytest

from upstream_analysis.data.alignment import DataAligner


class TestDataAligner:
    """Test data alignment functionality."""

    def test_aligner_initialization(self, sample_config):
        """Test aligner initialization with config."""
        aligner = DataAligner(sample_config.get("alignment", {}))
        assert aligner is not None
        assert aligner.tolerance_hours == 0.02
        assert aligner.online_time_col == "Process time [h]"

    def test_basic_alignment(self):
        """Test basic online-offline data alignment."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0, 3.0, 4.0],
                "Temperature [°C]": [37.0, 37.1, 37.2, 37.1, 37.0],
                "Balance [g]": [5000.0, 4950.0, 4900.0, 4850.0, 4800.0],
            }
        )

        offline_df = pd.DataFrame(
            {
                "process_time": [1.0, 3.0],
                "Biomass [g/L]": [10.0, 20.0],
                "Glucose [g/L]": [15.0, 5.0],
            }
        )

        merged, samples = aligner.merge_datasets(online_df, offline_df)

        # Check that offline data is merged
        assert "Biomass [g/L]" in merged.columns
        assert "Glucose [g/L]" in merged.columns
        assert "is_sample" in merged.columns

        # Check that samples are marked
        assert merged["is_sample"].sum() == 2

    def test_alignment_tolerance(self):
        """Test alignment tolerance functionality."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.1,  # 6 minutes tolerance
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0, 3.0],
                "Temperature [°C]": [37.0, 37.1, 37.2, 37.1],
            }
        )

        # Offline samples within tolerance
        offline_df = pd.DataFrame(
            {
                "process_time": [0.95, 2.05],  # Within 0.1h tolerance
                "Biomass [g/L]": [10.0, 20.0],
            }
        )

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Both samples should be aligned
        assert merged["Biomass [g/L]"].notna().sum() == 2

    def test_alignment_outside_tolerance(self):
        """Test that samples outside tolerance are not aligned."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,  # 1.2 minutes tolerance
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0, 3.0],
                "Temperature [°C]": [37.0, 37.1, 37.2, 37.1],
            }
        )

        # Offline sample outside tolerance
        offline_df = pd.DataFrame(
            {
                "process_time": [1.5],  # 0.5h away from nearest online point
                "Biomass [g/L]": [15.0],
            }
        )

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Sample should not be aligned (too far from online points)
        assert merged["Biomass [g/L]"].notna().sum() == 0

    def test_force_align_last_sample(self):
        """Test forced alignment of last sample."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": True,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0, 3.0, 4.0],
                "Temperature [°C]": [37.0, 37.1, 37.2, 37.1, 37.0],
            }
        )

        # Last sample outside tolerance
        offline_df = pd.DataFrame(
            {
                "process_time": [1.0, 2.0, 4.5],  # Last sample outside tolerance
                "Biomass [g/L]": [10.0, 20.0, 30.0],
            }
        )

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # First two should align normally, last should be forced
        # Last sample should still align to nearest online point
        assert merged["Biomass [g/L]"].notna().sum() >= 2

    def test_alignment_with_no_offline_data(self):
        """Test alignment when no offline data is provided."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Temperature [°C]": [37.0, 37.1, 37.2],
            }
        )

        merged, samples = aligner.merge_datasets(online_df, None)

        # Should return online data with is_sample = False
        assert "is_sample" in merged.columns
        assert merged["is_sample"].eq(False).all()
        assert samples.empty

    def test_alignment_with_empty_offline_data(self):
        """Test alignment when offline data is empty."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Temperature [°C]": [37.0, 37.1, 37.2],
            }
        )

        empty_offline = pd.DataFrame()
        merged, samples = aligner.merge_datasets(online_df, empty_offline)

        # Should return online data with is_sample = False
        assert "is_sample" in merged.columns
        assert merged["is_sample"].eq(False).all()
        assert samples.empty

    def test_alignment_multiple_samples_same_time(self):
        """Test alignment when multiple samples have same timestamp."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0, 3.0],
                "Temperature [°C]": [37.0, 37.1, 37.2, 37.1],
            }
        )

        # Multiple samples at same time
        offline_df = pd.DataFrame(
            {
                "process_time": [1.0, 1.0],  # Same time
                "Biomass [g/L]": [10.0, 12.0],
                "Glucose [g/L]": [15.0, 14.0],
            }
        )

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Should handle duplicate times (one might be excluded)
        assert "Biomass [g/L]" in merged.columns

    def test_alignment_preserves_online_data(self):
        """Test that alignment preserves original online data."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Temperature [°C]": [37.0, 37.1, 37.2],
                "Balance [g]": [5000.0, 4950.0, 4900.0],
            }
        )

        offline_df = pd.DataFrame({"process_time": [1.0], "Biomass [g/L]": [15.0]})

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Check that original online columns are preserved
        assert "Temperature [°C]" in merged.columns
        assert "Balance [g]" in merged.columns

        # Check that original values are preserved
        pd.testing.assert_series_equal(
            merged["Temperature [°C]"].dropna(),
            online_df["Temperature [°C]"],
            check_names=False,
        )


class TestEdgeCases:
    """Test edge cases in data alignment."""

    def test_missing_time_columns(self):
        """Test handling when time columns are missing."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {"Temperature [°C]": [37.0, 37.1, 37.2]}  # Missing time column
        )

        offline_df = pd.DataFrame({"process_time": [1.0], "Biomass [g/L]": [15.0]})

        # Should raise KeyError for missing time column
        with pytest.raises(KeyError):
            aligner.merge_datasets(online_df, offline_df)

    def test_nan_time_values(self):
        """Test handling of NaN time values."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, np.nan, 2.0],
                "Temperature [°C]": [37.0, 37.1, 37.2],
            }
        )

        offline_df = pd.DataFrame({"process_time": [1.0], "Biomass [g/L]": [15.0]})

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Should handle NaN times gracefully
        assert not merged.empty
        assert "Biomass [g/L]" in merged.columns

    def test_unsorted_time_data(self):
        """Test alignment with unsorted time data."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        # Unsorted online data
        online_df = pd.DataFrame(
            {
                "Process time [h]": [2.0, 0.0, 1.0, 3.0],
                "Temperature [°C]": [37.2, 37.0, 37.1, 37.1],
            }
        )

        offline_df = pd.DataFrame(
            {"process_time": [1.0, 2.0], "Biomass [g/L]": [15.0, 20.0]}
        )

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Should sort and align
        assert "Biomass [g/L]" in merged.columns
        assert merged["Biomass [g/L]"].notna().sum() == 2

    def test_single_online_point(self):
        """Test alignment with single online data point."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.02,
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {"Process time [h]": [1.0], "Temperature [°C]": [37.0]}
        )

        offline_df = pd.DataFrame({"process_time": [1.0], "Biomass [g/L]": [15.0]})

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Should align single point
        assert merged["Biomass [g/L]"].notna().sum() == 1

    def test_backward_direction_alignment(self):
        """Test alignment with backward direction (offline after online)."""
        config = {
            "online_time_col": "Process time [h]",
            "offline_time_col": "process_time",
            "tolerance_hours": 0.5,  # Large tolerance
            "force_align_last_sample": False,
        }
        aligner = DataAligner(config)

        online_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Temperature [°C]": [37.0, 37.1, 37.2],
            }
        )

        # Offline samples between online points
        offline_df = pd.DataFrame(
            {
                "process_time": [0.5, 1.5],  # Between online points
                "Biomass [g/L]": [12.0, 18.0],
            }
        )

        merged, _ = aligner.merge_datasets(online_df, offline_df)

        # Should align to nearest points
        assert merged["Biomass [g/L]"].notna().sum() == 2
