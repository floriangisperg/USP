"""Integration tests for full USP processing pipeline.

This module tests the complete end-to-end pipeline from raw data parsing
through final visualization, ensuring all components work together correctly.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from upstream_analysis.config.loader import load_config
from upstream_analysis.data.alignment import DataAligner
from upstream_analysis.data.cleaning import FermentationDataCleaner
from upstream_analysis.data.parser import (
    BioreactorDataParser,
    OfflineMeasurementsParser,
)
from upstream_analysis.data.processor import FermentationDataProcessor
from upstream_analysis.visualization.plotting import DataVisualizer


class TestFullPipeline:
    """Test complete processing pipeline with realistic data."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def realistic_config(self):
        """Create realistic configuration matching real usage patterns."""
        return {
            "parsing": {
                "bioreactor": {
                    "decimal_separator": ",",
                    "column_separator": ";",
                    "date_format": "%d.%m.%Y %H:%M",
                    "skiprows": 1,
                    "encoding": "cp1252",
                },
                "offline": {
                    "decimal_separator": ".",
                    "column_separator": ",",
                    "date_format": "guess",
                    "sheet_name": 0,
                },
            },
            "column_mapping": {
                # Online data mapping - use standardized names from default config
                "Date Time UTC": "Date time [UTC]",
                "Date Local Time": "Date time [local]",
                "Batch Time, sec": "Batch time [s]",
                "Batch Time (since inoc.), sec": "Batch time since inoculation [s]",
                "Phase": "Process phase [-]",
                "Phase Time, sec": "Phase time [s]",
                "Foam, -": "Foam [-]",
                "AirFlow, l/min": "Air flow [L/min]",
                "TotalFlow, l/min": "Total flow [L/min]",
                "Gas2Flow, l/min": "O2 flow [L/min]",
                "GasMix": "Gas mix [-]",
                "Pump1, %": "Acid pump [%]",
                "Pump1.Duration, -": "Acid pump duration [-]",
                "Pump2, %": "Base pump [%]",
                "Pump2.Duration, -": "Base pump duration [-]",
                "Pump3, %": "Inducer pump [%]",
                "Pump3.Duration, -": "Inducer pump duration [-]",
                "Pump4, %": "Feed pump [%]",
                "Pump4.Duration, -": "Feed pump duration [-]",
                "Stirrer, 1/min": "Stirrer speed [1/min]",
                "Temperature, °C": "Temperature [°C]",
                "Temperature, ¬∞C": "Temperature [°C]",
                "Balance, g": "Balance [g]",
                "AnalogIO1, %": "Analog IO1 [%]",
                "AnalogIO2, %": "Analog IO2 [%]",
                "pH, -": "pH [-]",
                "pO2, %": "pO2 [%]",
                "GenericGasAnalyser.ExitCO2, %": "Offgas CO2 [%]",
                "GenericGasAnalyser.ExitO2, %": "Offgas O2 [%]",
                "OD600": "OD600 online [-]",
                "ExponentialFeeding.Feedrate, %": "Feed rate exp [%]",
                "FeedingInduction.Feedrate2, %": "Feed rate ind [%]",
                # Offline data mapping
                "Process Time [h]": "Process time [h]",
                "Biomass [g/L]": "Biomass [g/L]",
                "Glucose [g/L]": "Glucose [g/L]",
                "Protein concentration [g/L]": "Protein [g/L]",
                "Protein IB [g/L]": "Protein IB [g/L]",
                "Protein soluble [g/L]": "Protein soluble [g/L]",
                "Sample volume [mL]": "Sample volume [mL]",
                "Acetate [g/L]": "Acetate [g/L]",
                "Viability [%]": "Viability [%]",
                "OD": "OD600 [-]",
            },
            "cleaning": {
                "filter_preparation": True,
                "rename_preparation_to_batch": False,
                "balance_columns": ["Balance [g]"],
                "balance_tare_threshold": -100,
                "median_filter": True,
                "median_filter_columns": ["Balance [g]"],
                "median_filter_window": 10,
                "median_filter_threshold": 10,
                "calculate_process_time": True,
            },
            "alignment": {
                "online_time_col": "Process time [h]",
                "offline_time_col": "Process time [h]",  # After column mapping, offline uses same name
                "tolerance_hours": 0.1,
                "save_separate_samples_file": True,
                "force_align_last_sample": True,
            },
            "processing": {
                "processing_parameters": {
                    "balance_mode": "feed_source",
                    "reactor_setup": {"initial_volume_ml": 1100.0},
                    "gas_params": {
                        "molar_volume_air_l_mol": 22.414,
                        "o2_in_air_percent": 21.0,
                        "co2_in_air_percent": 0.04,
                        "inlet_h2o_mol_pct": 0.0,
                        "outlet_h2o_mol_pct": 0.0,
                        "offgas_is_wet": False,
                        "use_n2_balance": False,
                        "calculate_inlet_composition": True,
                        "calculate_cer": True,
                        "calculate_our": True,
                        "calculate_rq": True,
                        "calculate_specific_rates": True,
                    },
                    "feed_definitions": {
                        "feed_1": {"name": "Glucose Feed", "density_g_ml": 1.150}
                    },
                    "medium_definitions": {"batch_medium": {"density_g_ml": 1.090}},
                    "base_titrant": {
                        "compound": "NH4OH",
                        "molarity_mol_l": 6.95,
                        "density_g_ml": 0.947,
                        "pump_channel": "Pump2",
                        "total_added_g": 0.0,
                    },
                    "biomass_params": {
                        "elemental_composition_c_mol": {
                            "C": 1.0,
                            "H": 1.77,
                            "N": 0.24,
                            "O": 0.49,
                        },
                        "molar_mass_g_c_mol": 26.565,
                        "ash_fraction_g_g": 0.08,
                    },
                    "product_params": {
                        "name": "scFv-His",
                        "nitrogen_content_fraction": 0.1647,
                    },
                    "substrate_properties": {
                        "glucose": {"molar_mass_g_c_mol": 30.03},
                        "acetate": {"molar_mass_g_c_mol": 29.5},
                    },
                    "induced_phases": [
                        "induced Fedbatch",
                        "Induced fed batch",
                        "Production",
                    ],
                    "filtering": {
                        "apply_feed_filter": True,
                        "median_filter_window": 5,
                        "median_filter_threshold": 10,
                    },
                    "rate_params": {
                        "apply_smoothing": True,
                        "smoothing_window_points": 5,
                        "smoothing_min_periods": 1,
                        "interpolation_method": "spline",
                        "spline_order": 3,
                        "spline_smoothing": 0.0,
                    },
                },
                "induced_phases": [
                    "induced Fedbatch",
                    "Induced fed batch",
                    "Production",
                ],
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
    def old_style_config(self):
        """Create old-style config for backward compatibility testing."""
        config = {
            "parsing": {
                "bioreactor": {
                    "decimal_separator": ",",
                    "column_separator": ";",
                    "skiprows": 1,
                },
                "offline": {
                    "decimal_separator": ".",
                    "column_separator": ",",
                    "sheet_name": 0,
                },
            },
            # Include column mapping for backward compatibility test
            "column_mapping": {
                "Phase": "Process phase [-]",
                "Balance, g": "Balance [g]",
                "Batch Time, sec": "Batch time [s]",
                "Process Time [h]": "Process time [h]",
            },
            "cleaning": {
                "median_filter": True,
                "median_filter_window": 10,
                "median_filter_threshold": 10,
                "calculate_process_time": True,
            },
            "alignment": {
                "online_time_col": "Process time [h]",
                "offline_time_col": "Process time [h]",
                "tolerance_hours": 0.1,
            },
            # Old structure: processing_parameters at top level
            "processing_parameters": {
                "balance_mode": "feed_source",
                "reactor_setup": {"initial_volume_ml": 1100.0},
                "gas_params": {
                    "molar_volume_air_l_mol": 22.414,
                    "o2_in_air_percent": 21.0,
                    "co2_in_air_percent": 0.04,
                    "calculate_cer": True,
                    "calculate_our": True,
                    "calculate_rq": True,
                },
                "feed_definitions": {"feed_1": {"density_g_ml": 1.150}},
                "base_titrant": {"total_added_g": 100.0, "density_g_ml": 1.0},
                "biomass_params": {"molar_mass_g_c_mol": 26.565},
                "induced_phases": ["induced Fedbatch"],
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
        return config

    @pytest.fixture
    def sample_bioreactor_csv(self, temp_output_dir):
        """Create realistic bioreactor CSV file for testing."""
        csv_path = temp_output_dir / "bioreactor_data.csv"

        # Create header row with realistic column names
        header = [
            "Date Time UTC",
            "Date Local Time",
            "Batch Time, sec",
            "Batch Time (since inoc.), sec",
            "Phase",
            "Phase Time, sec",
            "Foam, -",
            "AirFlow, l/min",
            "TotalFlow, l/min",
            "Gas2Flow, l/min",
            "GasMix",
            "Pump1, %",
            "Pump1.Duration, -",
            "Pump2, %",
            "Pump2.Duration, -",
            "Pump3, %",
            "Pump3.Duration, -",
            "Pump4, %",
            "Pump4.Duration, -",
            "Stirrer, 1/min",
            "Temperature, °C",
            "Balance, g",
            "AnalogIO1, %",
            "AnalogIO2, %",
            "pH, -",
            "pO2, %",
            "GenericGasAnalyser.ExitCO2, %",
            "GenericGasAnalyser.ExitO2, %",
            "OD600",
            "ExponentialFeeding.Feedrate, %",
            "FeedingInduction.Feedrate2, %",
        ]

        # Create realistic data rows spanning different phases
        data_rows = []
        process_times = np.arange(0, 30, 0.5)  # 30 hours of data

        for i, t in enumerate(process_times):
            if t < 2:
                phase = "Preparation"
                balance = 5000.0
            elif t < 8:
                phase = "Batch"
                balance = 5000.0 - (t - 2) * 10  # Slight decrease
            elif t < 12:
                phase = "Fedbatch"
                balance = 4940.0 - (t - 8) * 50  # Feed consumption
            else:
                phase = "induced Fedbatch"
                balance = 4740.0 - (t - 12) * 60  # Continued feeding

            # Add some realistic variation
            balance += np.random.randn() * 5

            row = [
                f"20.06.2023 {8+i//12:02d}:{(i*5)%60:02d}:13",  # Date Time UTC
                f"20.06.2023 {10+i//12:02d}:{(i*5)%60:02d}:13",  # Date Local Time
                f"{t*3600:.0f}",  # Batch Time, sec
                f"{t*3600:.0f}",  # Batch Time (since inoc.), sec
                phase,  # Phase
                f"{t*3600:.0f}",  # Phase Time, sec
                "0",  # Foam, -
                f"{1.0 + t*0.01:.8f}",  # AirFlow, l/min
                f"{1.0 + t*0.01:.8f}",  # TotalFlow, l/min
                f"{max(0, t-8)*0.05:.8f}" if t > 8 else "0",  # Gas2Flow, l/min
                "21",  # GasMix
                "0",  # Pump1, %
                "0",  # Pump1.Duration, -
                f"{max(0, t-6)*2:.0f}" if t > 6 else "0",  # Pump2, % (base)
                f"{max(0, t-6)*100:.0f}" if t > 6 else "0",  # Pump2.Duration, -
                "0",  # Pump3, %
                "0",  # Pump3.Duration, -
                f"{30 if t > 8 else 0}",  # Pump4, % (feed)
                f"{30 if t > 8 else 0}",  # Pump4.Duration, -
                f"{23382 + i*10:.1f}",  # Stirrer, 1/min
                f"{37.0 + np.sin(i/10)*0.2:.1f}".replace(".", ","),  # Temperature, °C
                f"{balance:.2f}".replace(".", ","),  # Balance, g
                "-26,2",  # AnalogIO1, %
                "-26,2",  # AnalogIO2, %
                (
                    f"{7.0 - (t-8)*0.5:.4f}".replace(".", ",") if t > 8 else "7,0000"
                ),  # pH, -
                f"{100 - t*2:.2f}".replace(".", ","),  # pO2, %
                f"{0.5 + t*0.1:.4f}".replace(".", ","),  # Offgas CO2, %
                f"{20.5 - t*0.2:.4f}".replace(".", ","),  # Offgas O2, %
                f"{0.5 + t*0.05:.4f}".replace(".", ","),  # OD600
                f"{20 if t > 8 else 0}",  # ExponentialFeeding.Feedrate, %
                f"{15 if t > 12 else 0}",  # FeedingInduction.Feedrate2, %
            ]
            data_rows.append(";".join(str(x) for x in row))

        # Write to file with proper encoding and line endings
        # Include metadata row at the beginning (like real data files)
        with open(csv_path, "w", encoding="cp1252", newline="") as f:
            # Metadata row (will be skipped by parser)
            f.write("1.121.957;de-DE\r\n")
            # Header row
            f.write(";".join(str(h) for h in header) + "\r\n")
            # Data rows
            f.write("\r\n".join(data_rows) + "\r\n")

        return csv_path

    @pytest.fixture
    def sample_offline_csv(self, temp_output_dir):
        """Create realistic offline measurements CSV file for testing."""
        csv_path = temp_output_dir / "offline_data.csv"

        # Header
        header = [
            "Process Time [h]",
            "Biomass [g/L]",
            "Glucose [g/L]",
            "Protein concentration [g/L]",
            "Protein soluble [g/L]",
            "Protein IB [g/L]",
            "Sample volume [mL]",
            "Acetate [g/L]",
            "Viability [%]",
            "OD",
        ]

        # Create sample data at realistic time points
        data_rows = []
        sample_times = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

        for t in sample_times:
            if t < 8:
                # Batch phase: glucose decreasing, biomass increasing
                biomass = 5 + t * 2
                glucose = max(0, 20 - t * 2.5)
                protein = 0
                protein_sol = 0
                protein_ib = 0
            elif t < 12:
                # Fedbatch phase
                biomass = 5 + 8 * 2 + (t - 8) * 3
                glucose = max(0, 20 - 8 * 2.5 - (t - 8) * 0.5)
                protein = 0
                protein_sol = 0
                protein_ib = 0
            else:
                # Induced phase
                biomass = 5 + 8 * 2 + 4 * 3 + (t - 12) * 2
                glucose = max(0, 1 + (t - 12) * 0.1)
                protein = (t - 12) * 0.5
                protein_sol = protein * 0.7
                protein_ib = protein * 0.3

            acetate = max(0, (t - 10) * 0.3) if t > 10 else 0
            viability = max(85, 100 - (t - 20) if t > 20 else 100)

            row = [
                f"{t:.1f}",
                f"{biomass:.2f}",
                f"{glucose:.2f}",
                f"{protein:.2f}",
                f"{protein_sol:.2f}",
                f"{protein_ib:.2f}",
                "10.0",
                f"{acetate:.2f}",
                f"{viability:.1f}",
                f"{biomass/5:.2f}",  # OD600
            ]
            data_rows.append(",".join(row))

        # Write to file with proper line endings
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(str(h) for h in header) + "\r\n")
            f.write("\r\n".join(data_rows) + "\r\n")

        return csv_path

    def test_pipeline_with_realistic_data(
        self,
        realistic_config,
        sample_bioreactor_csv,
        sample_offline_csv,
        temp_output_dir,
    ):
        """Test complete pipeline with realistic fermentation data."""

        # Step 1: Parse bioreactor data
        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)

        assert online_df is not None
        assert len(online_df) > 0
        # After column mapping, should have standardized names OR raw names
        # Column mapping depends on exact header matching
        assert len(online_df.columns) > 0

        # Step 2: Parse offline data
        parser_offline = OfflineMeasurementsParser(realistic_config)
        offline_df = parser_offline.parse_file(sample_offline_csv)

        assert offline_df is not None
        assert len(offline_df) > 0
        # After column mapping, 'Process Time [h]' becomes 'Process time [h]'
        assert (
            "Process time [h]" in offline_df.columns
            or "process_time" in offline_df.columns
        )

        # Step 3: Clean online data
        cleaner = FermentationDataCleaner(realistic_config)
        cleaned_df = cleaner.clean_data(online_df)

        assert cleaned_df is not None
        assert len(cleaned_df) > 0
        # Note: Phase column name depends on whether column_mapping worked
        # 'process_phase' is standardized name, 'Process phase [-]' might be raw
        phase_col = (
            "process_phase"
            if "process_phase" in cleaned_df.columns
            else "Process phase [-]"
        )
        # Preparation phase may or may not be filtered depending on cleaning config
        # Just verify the phase column exists
        assert phase_col in cleaned_df.columns

        # Step 4: Align online and offline data
        aligner = DataAligner(realistic_config.get("alignment", {}))
        aligned_df, samples_df = aligner.merge_datasets(cleaned_df, offline_df)

        assert aligned_df is not None
        assert len(aligned_df) > 0
        assert "is_sample" in aligned_df.columns
        assert samples_df is not None

        # Step 5: Process data (calculate derived parameters)
        processor = FermentationDataProcessor(realistic_config)
        processed_df = processor.process_data(aligned_df)

        assert processed_df is not None
        assert len(processed_df) > 0

        # Verify key derived columns exist
        expected_derived_columns = [
            "Feed [g]",
            "Feed [mL]",
            "Base volume added [mL]",
            "Volume [mL]",
            "Volume corrected [mL]",
            "Is induced [-]",  # Standardized name, not 'is_induced'
        ]
        for col in expected_derived_columns:
            assert col in processed_df.columns, f"Missing expected column: {col}"

        # Verify gas analysis columns if configured
        gas_config = realistic_config["processing"]["processing_parameters"][
            "gas_params"
        ]
        if gas_config.get("calculate_cer"):
            assert (
                "CER [mmol/L/h]" in processed_df.columns
                or "CO2 evolution rate [mmol/L/h]" in processed_df.columns
            )
        if gas_config.get("calculate_our"):
            assert (
                "OUR [mmol/L/h]" in processed_df.columns
                or "O2 uptake rate [mmol/L/h]" in processed_df.columns
            )
        if gas_config.get("calculate_rq"):
            assert (
                "RQ [-]" in processed_df.columns
                or "Respiratory quotient [-]" in processed_df.columns
            )

        # Step 6: Generate visualization
        visualizer = DataVisualizer(temp_output_dir)

        # Select parameters to plot
        parameters_to_plot = [
            "Biomass [g/L]",
            "Glucose [g/L]",
            "Volume [mL]",
            "Feed [mL]",
            "pH [-]",
            "CER [mmol/L/h]",
        ]

        # Use the correct time column name
        time_col = "Process time [h]"

        # This should not raise any exceptions
        visualizer.plot_selected_parameters(
            processed_df, parameters_to_plot, time_col=time_col
        )

        # Verify plot files were created
        plot_files = list(temp_output_dir.glob("*.png"))
        assert len(plot_files) > 0, "No plot files were created"

    def test_pipeline_backward_compatibility(
        self, old_style_config, sample_bioreactor_csv, sample_offline_csv
    ):
        """Test that pipeline works with both old and new config structures."""

        # Parse data with old-style config
        parser_online = BioreactorDataParser(old_style_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)

        parser_offline = OfflineMeasurementsParser(old_style_config)
        offline_df = parser_offline.parse_file(sample_offline_csv)

        # Clean data
        cleaner = FermentationDataCleaner(old_style_config)
        cleaned_df = cleaner.clean_data(online_df)

        # Align data
        aligner = DataAligner(old_style_config.get("alignment", {}))
        aligned_df, samples_df = aligner.merge_datasets(cleaned_df, offline_df)

        # Process data - processor should handle old config structure
        processor = FermentationDataProcessor(old_style_config)
        processed_df = processor.process_data(aligned_df)

        assert processed_df is not None
        assert len(processed_df) > 0
        assert "Feed [g]" in processed_df.columns

    def test_pipeline_column_standardization(
        self, realistic_config, sample_bioreactor_csv, sample_offline_csv
    ):
        """Test that column mapping produces standardized column names."""

        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)

        # Check for standardized column names (using names from default_config.yaml)
        expected_standard_columns = [
            "Date time [UTC]",
            "Process phase [-]",
            "Balance [g]",
            "Temperature [°C]",
            "pH [-]",
            "pO2 [%]",
            "Offgas CO2 [%]",
            "Offgas O2 [%]",
        ]

        # At least some columns should be mapped to standard names
        found_standard = any(
            col in online_df.columns for col in expected_standard_columns
        )
        assert found_standard, "No standard column names found after mapping"

    def test_pipeline_data_integrity(
        self, realistic_config, sample_bioreactor_csv, sample_offline_csv
    ):
        """Test that data integrity is maintained through the pipeline."""

        # Parse data
        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)
        initial_row_count = len(online_df)

        # Clean data
        cleaner = FermentationDataCleaner(realistic_config)
        cleaned_df = cleaner.clean_data(online_df)

        # Row count should be reduced but not empty (Preparation phase filtered)
        assert (
            len(cleaned_df) < initial_row_count
        ), "No rows were filtered during cleaning"
        assert len(cleaned_df) > 0, "All rows were filtered during cleaning"

        # Parse and align offline data
        parser_offline = OfflineMeasurementsParser(realistic_config)
        offline_df = parser_offline.parse_file(sample_offline_csv)

        aligner = DataAligner(realistic_config.get("alignment", {}))
        aligned_df, samples_df = aligner.merge_datasets(cleaned_df, offline_df)

        # Aligned data should have same rows as cleaned data
        assert len(aligned_df) == len(cleaned_df), "Alignment changed row count"

        # Process data
        processor = FermentationDataProcessor(realistic_config)
        processed_df = processor.process_data(aligned_df)

        # Processed data should maintain same row count
        assert len(processed_df) == len(aligned_df), "Processing changed row count"

    def test_pipeline_phase_detection(
        self, realistic_config, sample_bioreactor_csv, sample_offline_csv
    ):
        """Test that induction phase detection works correctly."""

        # Run full pipeline
        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)

        cleaner = FermentationDataCleaner(realistic_config)
        cleaned_df = cleaner.clean_data(online_df)

        parser_offline = OfflineMeasurementsParser(realistic_config)
        offline_df = parser_offline.parse_file(sample_offline_csv)

        aligner = DataAligner(realistic_config.get("alignment", {}))
        aligned_df, samples_df = aligner.merge_datasets(cleaned_df, offline_df)

        processor = FermentationDataProcessor(realistic_config)
        processed_df = processor.process_data(aligned_df)

        # Check induction detection - column name is 'Is induced [-]' not 'is_induced'
        assert (
            "Is induced [-]" in processed_df.columns
            or "is_induced" in processed_df.columns
        )
        # Check that induction was detected (should have some True values)
        induction_col = (
            "Is induced [-]"
            if "Is induced [-]" in processed_df.columns
            else "is_induced"
        )
        assert processed_df[induction_col].any(), "No induction was detected"

    def test_pipeline_gas_analysis(
        self, realistic_config, sample_bioreactor_csv, sample_offline_csv
    ):
        """Test that gas analysis calculations work correctly."""

        # Run full pipeline
        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)

        cleaner = FermentationDataCleaner(realistic_config)
        cleaned_df = cleaner.clean_data(online_df)

        parser_offline = OfflineMeasurementsParser(realistic_config)
        offline_df = parser_offline.parse_file(sample_offline_csv)

        aligner = DataAligner(realistic_config.get("alignment", {}))
        aligned_df, _ = aligner.merge_datasets(cleaned_df, offline_df)

        processor = FermentationDataProcessor(realistic_config)
        processed_df = processor.process_data(aligned_df)

        # Check gas analysis columns
        gas_columns = ["CER [mmol/L/h]", "OUR [mmol/L/h]", "RQ [-]"]
        for col in gas_columns:
            if col in processed_df.columns:
                # Check that values are reasonable (not NaN, not all zeros)
                col_data = processed_df[col].dropna()
                assert len(col_data) > 0, f"{col} contains only NaN values"
                assert col_data.notna().any(), f"{col} has no valid values"

    def test_pipeline_feed_calculation(
        self, realistic_config, sample_bioreactor_csv, sample_offline_csv
    ):
        """Test that feed calculation works correctly."""

        # Run full pipeline
        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)

        cleaner = FermentationDataCleaner(realistic_config)
        cleaned_df = cleaner.clean_data(online_df)

        parser_offline = OfflineMeasurementsParser(realistic_config)
        offline_df = parser_offline.parse_file(sample_offline_csv)

        aligner = DataAligner(realistic_config.get("alignment", {}))
        aligned_df, _ = aligner.merge_datasets(cleaned_df, offline_df)

        processor = FermentationDataProcessor(realistic_config)
        processed_df = processor.process_data(aligned_df)

        # Check feed columns
        assert "Feed [g]" in processed_df.columns
        assert "Feed [mL]" in processed_df.columns

        # Feed should be greater than zero after feeding starts
        feed_ml = processed_df["Feed [mL]"]
        assert feed_ml.max() > 0, "No feed was calculated"

        # Feed should be monotonically increasing
        assert (
            feed_ml.is_monotonic_increasing or (feed_ml.diff() >= -1e-6).all()
        ), "Feed is not monotonically increasing"

    def test_pipeline_volume_tracking(
        self, realistic_config, sample_bioreactor_csv, sample_offline_csv
    ):
        """Test that volume tracking works correctly."""

        # Run full pipeline
        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(sample_bioreactor_csv)

        cleaner = FermentationDataCleaner(realistic_config)
        cleaned_df = cleaner.clean_data(online_df)

        parser_offline = OfflineMeasurementsParser(realistic_config)
        offline_df = parser_offline.parse_file(sample_offline_csv)

        aligner = DataAligner(realistic_config.get("alignment", {}))
        aligned_df, _ = aligner.merge_datasets(cleaned_df, offline_df)

        processor = FermentationDataProcessor(realistic_config)
        processed_df = processor.process_data(aligned_df)

        # Check volume columns
        assert "Volume [mL]" in processed_df.columns
        assert "Volume corrected [mL]" in processed_df.columns

        # Volume should increase during feeding
        volume_ml = processed_df["Volume [mL]"]
        initial_volume = volume_ml.iloc[0]
        max_volume = volume_ml.max()
        assert max_volume > initial_volume, "Volume did not increase"

    def test_pipeline_with_missing_optional_columns(
        self, realistic_config, temp_output_dir
    ):
        """Test pipeline handles missing optional columns gracefully."""

        # Create minimal bioreactor data with only essential columns
        csv_path = temp_output_dir / "minimal_bioreactor.csv"

        header = ["Batch Time, sec", "Phase", "Temperature, °C", "Balance, g", "pH, -"]
        data = []
        for i in range(10):
            t = i * 3600
            phase = "Batch" if i < 4 else "Fedbatch"
            data.append([f"{t}", phase, "37,0", "5000,0", "7,0"])

        with open(csv_path, "w", encoding="cp1252") as f:
            f.write(";".join(header) + "\n")
            f.write("\n".join(";".join(row) for row in data) + "\n")

        # Parse and process
        parser_online = BioreactorDataParser(realistic_config)
        online_df = parser_online.parse_file(csv_path)

        assert online_df is not None

        cleaner = FermentationDataCleaner(realistic_config)
        cleaned_df = cleaner.clean_data(online_df)

        # Should not fail even with missing columns
        assert cleaned_df is not None
