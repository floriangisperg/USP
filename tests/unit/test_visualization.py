"""Tests for visualization functionality."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from upstream_analysis.visualization.plotting import DataVisualizer, sanitize_filename


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_sanitize_basic(self):
        """Test basic filename sanitization."""
        filename = "Normal_Filename"
        result = sanitize_filename(filename)
        assert result == "Normal_Filename"

    def test_sanitize_special_chars(self):
        """Test removal of special characters."""
        filename = "Test/With:Special*Chars?"
        result = sanitize_filename(filename)
        assert '/' not in result
        assert ':' not in result
        assert '*' not in result
        assert '?' not in result
        assert result == "Test_With_Special_Chars_"

    def test_sanitize_percent_sign(self):
        """Test percent sign removal."""
        filename = "C recovered [%]"
        result = sanitize_filename(filename)
        assert '%' not in result
        assert ']' not in result
        assert '[' not in result

    def test_sanitize_whitespace(self):
        """Test whitespace replacement."""
        filename = "Test Filename With Spaces"
        result = sanitize_filename(filename)
        assert ' ' not in result
        assert result == "Test_Filename_With_Spaces"

    def test_sanitize_degree_symbol(self):
        """Test degree symbol removal."""
        filename = "Temperature [°C]"
        result = sanitize_filename(filename)
        assert '°' not in result

    def test_sanitize_length_limit(self):
        """Test filename length limiting."""
        long_filename = "a" * 150
        result = sanitize_filename(long_filename)
        assert len(result) <= 100

    def test_sanitize_leading_underscores(self):
        """Test removal of leading underscores."""
        filename = "___test_filename"
        result = sanitize_filename(filename)
        assert not result.startswith('_')


class TestDataVisualizer:
    """Test data visualization functionality."""

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))
            assert visualizer is not None
            assert visualizer.output_dir == Path(tmpdir)

    def test_visualizer_creates_output_dir(self):
        """Test that visualizer creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'plots'
            visualizer = DataVisualizer(output_path)
            assert output_path.exists()
            assert output_path.is_dir()

    def test_plot_single_parameter(self):
        """Test plotting a single parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0, 3.0],
                'Temperature [°C]': [37.0, 37.1, 37.2, 37.1],
                'Balance [g]': [5000.0, 4950.0, 4900.0, 4850.0]
            })

            visualizer.plot_selected_parameters(test_df, ['Temperature [°C]'])

            # Check that plot file was created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_multiple_parameters(self):
        """Test plotting multiple parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0, 3.0],
                'Temperature [°C]': [37.0, 37.1, 37.2, 37.1],
                'Balance [g]': [5000.0, 4950.0, 4900.0, 4850.0],
                'pH [-]': [7.0, 6.9, 6.8, 6.7]
            })

            visualizer.plot_selected_parameters(
                test_df,
                ['Temperature [°C]', 'Balance [g]', 'pH [-]']
            )

            # Check that all plot files were created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 3

    def test_plot_with_offline_data(self):
        """Test plotting sparse offline data (many NaNs)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0, 3.0, 4.0],
                'Biomass [g/L]': [np.nan, 10.0, np.nan, 20.0, np.nan]  # Sparse data
            })

            visualizer.plot_selected_parameters(test_df, ['Biomass [g/L]'])

            # Check that plot file was created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_with_missing_column(self):
        """Test handling when requested column is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0],
                'Temperature [°C]': [37.0, 37.1, 37.2]
            })

            # Should not crash for missing column
            visualizer.plot_selected_parameters(
                test_df,
                ['Temperature [°C]', 'NonExistent [unit]']
            )

            # Only one plot should be created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_with_all_nan_column(self):
        """Test handling when column contains all NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0],
                'Valid Data': [37.0, 37.1, 37.2],
                'All NaN': [np.nan, np.nan, np.nan]
            })

            visualizer.plot_selected_parameters(
                test_df,
                ['Valid Data', 'All NaN']
            )

            # Only one plot should be created (for valid data)
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_with_missing_time_column(self):
        """Test handling when time column is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'Temperature [°C]': [37.0, 37.1, 37.2],
                'Balance [g]': [5000.0, 4950.0, 4900.0]
            })

            # Should not crash when time column is missing
            visualizer.plot_selected_parameters(
                test_df,
                ['Temperature [°C]'],
                time_col='process_time'
            )

            # No plots should be created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 0

    def test_plot_with_all_nan_time_column(self):
        """Test handling when time column contains all NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [np.nan, np.nan, np.nan],
                'Temperature [°C]': [37.0, 37.1, 37.2]
            })

            # Should not crash when time column is all NaN
            visualizer.plot_selected_parameters(
                test_df,
                ['Temperature [°C]'],
                time_col='process_time'
            )

            # No plots should be created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 0

    def test_rq_plot_special_handling(self):
        """Test RQ plot special handling (y-axis limits and reference line)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0, 3.0],
                'RQ [-]': [0.8, 0.9, 1.0, 1.1]
            })

            visualizer.plot_selected_parameters(test_df, ['RQ [-]'])

            # Check that plot was created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_custom_time_column(self):
        """Test plotting with custom time column name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'Time [h]': [0.0, 1.0, 2.0, 3.0],
                'Temperature [°C]': [37.0, 37.1, 37.2, 37.1]
            })

            visualizer.plot_selected_parameters(
                test_df,
                ['Temperature [°C]'],
                time_col='Time [h]'
            )

            # Check that plot was created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_calculated_protein_parameters(self):
        """Test plotting calculated protein parameters (should plot as scatter)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0, 3.0],
                'Protein total calculated [g/L]': [0.5, 1.5, 2.5, 3.5],
                'Protein soluble yield [g/g]': [0.1, 0.2, 0.3, 0.4]
            })

            visualizer.plot_selected_parameters(
                test_df,
                ['Protein total calculated [g/L]', 'Protein soluble yield [g/g]']
            )

            # Check that plots were created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 2

    def test_plot_empty_dataframe(self):
        """Test plotting with empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame()

            # Should not crash
            visualizer.plot_selected_parameters(test_df, ['Temperature [°C]'])

            # No plots should be created
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 0

    def test_plot_filename_sanitization(self):
        """Test that plot filenames are sanitized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0],
                'C recovered [%]': [80.0, 90.0, 95.0]
            })

            visualizer.plot_selected_parameters(test_df, ['C recovered [%]'])

            # Check that filename is sanitized (no special chars)
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1
            assert '%' not in str(plot_files[0])
            assert '[' not in str(plot_files[0])
            assert ']' not in str(plot_files[0])


class TestPlottingEdgeCases:
    """Test edge cases in plotting."""

    def test_plot_single_data_point(self):
        """Test plotting with single data point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0],
                'Temperature [°C]': [37.0]
            })

            visualizer.plot_selected_parameters(test_df, ['Temperature [°C]'])

            # Should still create plot
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_with_negative_values(self):
        """Test plotting with negative values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0],
                'Growth rate [1/h]': [-0.1, 0.0, 0.1]
            })

            visualizer.plot_selected_parameters(test_df, ['Growth rate [1/h]'])

            # Should handle negative values
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1

    def test_plot_with_very_large_values(self):
        """Test plotting with very large values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DataVisualizer(Path(tmpdir))

            test_df = pd.DataFrame({
                'process_time': [0.0, 1.0, 2.0],
                'Large values [-]': [1e6, 1e7, 1e8]
            })

            visualizer.plot_selected_parameters(test_df, ['Large values [-]'])

            # Should handle large values
            plot_files = list(Path(tmpdir).glob('*.png'))
            assert len(plot_files) == 1
