"""Tests for data parsing functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from upstream_analysis.data.parser import (
    BioreactorDataParser,
    OfflineMeasurementsParser,
)


class TestBioreactorParser:
    """Test bioreactor data parsing."""

    def test_parser_initialization(self, sample_config):
        """Test parser initialization with config."""
        parser = BioreactorDataParser(sample_config)
        assert parser is not None
        assert parser.decimal == ","
        assert parser.separator == ";"

    def test_column_rename(self, sample_config):
        """Test column renaming functionality."""
        parser = BioreactorDataParser(sample_config)

        # Create test DataFrame with raw column names
        test_df = pd.DataFrame(
            {
                "Raw Temp": [37.0, 37.5, 38.0],
                "raw_balance": [5000.0, 4950.0, 4900.0],
                "AirFlow": [1.0, 1.1, 1.2],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
            }
        )

        # Test column renaming
        result = parser._rename_columns(test_df)

        # Check that columns were renamed according to config
        assert "Temperature [°C]" in result.columns
        assert "Balance [g]" in result.columns

    def test_column_rename_missing_keys(self, sample_config):
        """Test column renaming with missing keys."""
        parser = BioreactorDataParser(sample_config)

        # Create test DataFrame without raw column names
        test_df = pd.DataFrame(
            {
                "Temperature [°C]": [37.0, 37.5, 38.0],
                "Balance [g]": [5000.0, 4950.0, 4900.0],
            }
        )

        # Should not crash when keys are missing
        result = parser._rename_columns(test_df)
        assert result is not None
        assert len(result) == len(test_df)

    def test_process_dataframe_numeric_conversion(self, sample_config):
        """Test numeric conversion in data processing."""
        parser = BioreactorDataParser(sample_config)

        # Create test DataFrame with string numbers
        test_df = pd.DataFrame(
            {
                "Temperature [°C]": ["37,0", "37,5", "38,0"],
                "Balance [g]": ["5000,0", "4950,0", "4900,0"],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
            }
        )

        result = parser._process_dataframe(test_df)

        # Check numeric conversion
        assert pd.api.types.is_numeric_dtype(result["Temperature [°C]"])
        assert pd.api.types.is_numeric_dtype(result["Balance [g]"])

    def test_process_dataframe_phase_handling(self, sample_config):
        """Test process phase string handling."""
        parser = BioreactorDataParser(sample_config)

        # Create test DataFrame with different phase formats
        test_df = pd.DataFrame(
            {
                "Process phase [-]": ["Fedbatch", "FedBatch", "fedBatch"],
                "Temperature [°C]": [37.0, 37.5, 38.0],
            }
        )

        result = parser._process_dataframe(test_df)

        # Check that phases are standardized
        assert result["Process phase [-]"].iloc[0] == "Fedbatch"

    def test_parse_csv_file(self, sample_config):
        """Test parsing a CSV file."""
        parser = BioreactorDataParser(sample_config)

        # Create temporary CSV file with test data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="cp1252"
        ) as f:
            # Real bioreactor CSVs start with a metadata line that the parser
            # skips via parsing.bioreactor.skiprows (=1). Mirror that here.
            f.write("Run metadata line (skipped by skiprows)\n")
            f.write("Time;Temperature [°C];Balance [g];Process phase [-]\n")
            f.write("0,0;37,0;5000,0;Batch\n")
            f.write("1,0;37,5;4950,0;Batch\n")
            f.write("2,0;38,0;4900,0;Fedbatch\n")
            temp_path = f.name

        try:
            result = parser.parse_file(temp_path)
            assert not result.empty
            assert len(result) == 3
            assert "Temperature [°C]" in result.columns
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_parse_excel_file(self, sample_config):
        """Test parsing an Excel file."""
        parser = BioreactorDataParser(sample_config)

        # Create temporary Excel file with test data
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".xlsx", delete=False) as f:
            test_df = pd.DataFrame(
                {
                    "Time": [0.0, 1.0, 2.0],
                    "Temperature [°C]": [37.0, 37.5, 38.0],
                    "Balance [g]": [5000.0, 4950.0, 4900.0],
                }
            )
            # Mirror real bioreactor files: reserve a metadata row (row 0) that
            # the parser skips via header=skiprows (=1); header lands on row 1.
            with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
                test_df.to_excel(writer, index=False, startrow=1)
            temp_path = f.name

        try:
            result = parser.parse_file(temp_path)
            assert not result.empty
            assert len(result) == 3
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestOfflineParser:
    """Test offline measurements parsing."""

    def test_offline_parser_initialization(self, sample_config):
        """Test offline parser initialization."""
        parser = OfflineMeasurementsParser(sample_config)
        assert parser is not None

    def test_offline_parser_basic(self, sample_config):
        """Test basic offline data parsing."""
        parser = OfflineMeasurementsParser(sample_config)

        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".xlsx", delete=False) as f:
            test_df = pd.DataFrame(
                {
                    "Sample Time": [1.0, 2.5, 4.0],
                    "OD600": [0.5, 1.5, 2.5],
                    "Glucose [g/L]": [20.0, 10.0, 5.0],
                }
            )
            test_df.to_excel(f.name, index=False)
            temp_path = f.name

        try:
            result = parser.parse_file(temp_path)
            assert not result.empty
            assert len(result) == 3
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_offline_parser_with_config(self):
        """Test offline parser with custom config."""
        config = {
            "parsing": {
                "offline": {
                    "decimal_separator": ".",
                    "column_separator": ",",
                    "sheet_name": 0,
                }
            }
        }
        parser = OfflineMeasurementsParser(config)
        assert parser.decimal == "."
        assert parser.separator == ","


class TestEdgeCases:
    """Test edge cases in parsing."""

    def test_empty_dataframe_handling(self, sample_config):
        """Test handling of empty DataFrames."""
        parser = BioreactorDataParser(sample_config)
        empty_df = pd.DataFrame()

        result = parser._process_dataframe(empty_df)
        assert result.empty

    def test_nan_handling(self, sample_config):
        """Test handling of NaN values."""
        parser = BioreactorDataParser(sample_config)

        test_df = pd.DataFrame(
            {
                "Temperature [°C]": [37.0, np.nan, 38.0],
                "Balance [g]": [5000.0, 4950.0, np.nan],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
            }
        )

        result = parser._process_dataframe(test_df)
        # Should handle NaN gracefully
        assert result["Temperature [°C]"].isna().any()
        assert result["Balance [g]"].isna().any()

    def test_mixed_data_types(self, sample_config):
        """Test handling of mixed data types in columns."""
        parser = BioreactorDataParser(sample_config)

        test_df = pd.DataFrame(
            {
                "Temperature [°C]": ["37,0", "invalid", "38,0"],
                "Balance [g]": [5000, 4950, 4900],
            }
        )

        result = parser._process_dataframe(test_df)
        # Invalid strings should become NaN
        assert result["Temperature [°C]"].isna().any()
