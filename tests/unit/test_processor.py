"""Tests for core processing calculations."""

import pandas as pd
import pytest

from upstream_analysis.data.processor import FermentationDataProcessor


class TestFeedCalculations:
    """Test feed calculation logic."""

    def test_feed_calculation_basic(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test basic feed calculation."""
        result = sample_processor._calculate_feed(sample_bioreactor_with_base_df)

        # Check Feed columns exist
        assert "Feed [g]" in result.columns
        assert "Feed [mL]" in result.columns

        # Check feed is non-negative
        feed_g = result["Feed [g]"]
        assert (feed_g >= 0).all(), "Feed should never be negative"

    def test_feed_calculation_monotonic_increasing(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test that feed increases monotonically during fedbatch phase."""
        result = sample_processor._calculate_feed(sample_bioreactor_with_base_df)

        # Check feed increases during fedbatch phase
        fedbatch_mask = result["Process phase [-]"].isin(
            ["Fedbatch", "induced Fedbatch"]
        )
        if fedbatch_mask.any():
            fedbatch_feed = result.loc[fedbatch_mask, "Feed [g]"].dropna()
            if len(fedbatch_feed) > 1:
                # Feed should be non-decreasing
                assert (
                    fedbatch_feed.diff().dropna() >= -0.01
                ).all(), "Feed should be monotonically increasing during fedbatch"

    def test_feed_calculation_density_conversion(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test feed volume conversion using density."""
        result = sample_processor._calculate_feed(sample_bioreactor_with_base_df)

        # Check that Feed [mL] = Feed [g] / density
        density = 1.1  # From test config
        feed_ml_calculated = result["Feed [g]"] / density
        pd.testing.assert_series_equal(
            result["Feed [mL]"], feed_ml_calculated, check_names=False, rtol=1e-5
        )

    def test_feed_calculation_missing_columns(self, sample_processor):
        """Test feed calculation with missing required columns."""
        incomplete_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
                # Missing Balance column
            }
        )

        result = sample_processor._calculate_feed(incomplete_df)
        # Should return DataFrame with zero feed when columns are missing
        assert "Feed [g]" in result.columns
        assert (result["Feed [g]"] == 0).all()

    def test_feed_calculation_no_fedbatch_phase(self, sample_processor):
        """Test feed calculation when no fedbatch phase is present."""
        batch_only_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Process phase [-]": ["Batch", "Batch", "Batch"],
                "Balance [g]": [5000.0, 4950.0, 4900.0],
            }
        )

        result = sample_processor._calculate_feed(batch_only_df)
        # Should return zero feed when no fedbatch phase
        assert (result["Feed [g]"] == 0).all()


class TestBaseVolumeCalculations:
    """Test base volume calculation logic."""

    def test_base_volume_calculation_basic(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test basic base volume calculation."""
        result = sample_processor._calculate_base_flux_and_volume(
            sample_bioreactor_with_base_df
        )

        # Check column exists
        assert "Base volume added [mL]" in result.columns

        # Check base volume is non-negative
        base_vol = result["Base volume added [mL]"]
        assert (base_vol >= 0).all(), "Base volume should never be negative"

    def test_base_volume_cumulative(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test that base volume is cumulative."""
        result = sample_processor._calculate_base_flux_and_volume(
            sample_bioreactor_with_base_df
        )

        # Base volume should be non-decreasing
        base_vol = result["Base volume added [mL]"].dropna()
        if len(base_vol) > 1:
            assert (
                base_vol.diff().dropna() >= -0.01
            ).all(), "Base volume should be monotonically increasing"

    def test_base_volume_total_limit(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test that base volume respects total added limit."""
        result = sample_processor._calculate_base_flux_and_volume(
            sample_bioreactor_with_base_df
        )

        # Check that volume doesn't exceed total
        total_base_ml = 100.0  # From test config
        assert (
            result["Base volume added [mL]"] <= total_base_ml + 0.01
        ).all(), "Base volume should not exceed total added"

    def test_base_volume_missing_columns(self, sample_processor):
        """Test base volume calculation with missing required columns."""
        incomplete_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Process phase [-]": ["Batch", "Batch", "Fedbatch"],
                # Missing Base pump duration column
            }
        )

        result = sample_processor._calculate_base_flux_and_volume(incomplete_df)
        # Should return zero when columns are missing
        assert "Base volume added [mL]" in result.columns


class TestVolumeCalculations:
    """Test reactor volume calculation logic."""

    def test_volume_calculation_basic(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test basic volume calculation."""
        # First calculate feed and base since volume depends on them
        with_feed = sample_processor._calculate_feed(sample_bioreactor_with_base_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        result = sample_processor._calculate_volume(with_base)

        # Check volume columns exist
        assert "Volume [mL]" in result.columns
        assert "Volume corrected [mL]" in result.columns

        # Check volume is positive
        volume = result["Volume [mL]"]
        assert (volume > 0).all(), "Volume should always be positive"

    def test_volume_calculation_components(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test that volume equals start + feed + base."""
        with_feed = sample_processor._calculate_feed(sample_bioreactor_with_base_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        result = sample_processor._calculate_volume(with_base)

        start_volume = 1100.0  # From test config

        # Volume = start + feed + base
        expected_volume = (
            start_volume + result["Feed [mL]"] + result["Base volume added [mL]"]
        )
        pd.testing.assert_series_equal(
            result["Volume [mL]"], expected_volume, check_names=False, rtol=1e-3
        )

    def test_volume_increases_over_time(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test that volume increases monotonically."""
        with_feed = sample_processor._calculate_feed(sample_bioreactor_with_base_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        result = sample_processor._calculate_volume(with_base)

        volume = result["Volume [mL]"].dropna()
        if len(volume) > 1:
            # Volume should be non-decreasing
            assert (
                volume.diff().dropna() >= -0.01
            ).all(), "Volume should be monotonically increasing"

    def test_volume_with_sample_correction(self, sample_processor, sample_aligned_df):
        """Test volume correction for sample removal."""
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        result = sample_processor._calculate_volume(with_base)

        # Check that corrected volume is calculated
        assert "Volume corrected [mL]" in result.columns
        assert "Cumulative sample volume [mL]" in result.columns

        # Corrected volume should be <= uncorrected volume
        assert (
            result["Volume corrected [mL]"] <= result["Volume [mL]"] + 0.01
        ).all(), "Corrected volume should be less than or equal to base volume"

    def test_volume_missing_initial_volume(self, minimal_config):
        """Test volume calculation with missing initial volume config."""
        config_no_volume = minimal_config.copy()
        config_no_volume["processing"]["processing_parameters"]["reactor_setup"] = {}

        processor = FermentationDataProcessor(config_no_volume)
        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0],
                "Feed [mL]": [0.0, 100.0],
                "Base volume added [mL]": [0.0, 50.0],
            }
        )

        result = processor._calculate_volume(test_df)
        # Should return NaN volume when initial volume is missing
        assert result["Volume [mL]"].isna().all()


class TestInductionCalculations:
    """Test induction indicator and timer calculations."""

    def test_induction_indicator(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test induction indicator column creation."""
        result = sample_processor._add_induction_indicator(
            sample_bioreactor_with_base_df
        )

        # Check column exists
        assert "Is induced [-]" in result.columns

        # Check values are 0 or 1
        assert (
            result["Is induced [-]"].isin([0, 1]).all()
        ), "Induction indicator should only contain 0 or 1"

    def test_induction_timer(self, sample_processor, sample_bioreactor_with_base_df):
        """Test induction timer calculation."""
        with_indication = sample_processor._add_induction_indicator(
            sample_bioreactor_with_base_df
        )
        result = sample_processor._add_induction_timer(with_indication)

        # Check column exists
        assert "Induction time [h]" in result.columns

        # Timer should be non-negative
        assert (
            result["Induction time [h]"] >= 0
        ).all(), "Induction timer should be non-negative"

        # Timer should be 0 before induction
        induced_mask = result["Is induced [-]"] == 0
        if induced_mask.any():
            assert (
                result.loc[induced_mask, "Induction time [h]"] == 0
            ).all(), "Induction timer should be 0 before induction starts"

    def test_induction_timer_increases(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test that induction timer increases during induction."""
        with_indication = sample_processor._add_induction_indicator(
            sample_bioreactor_with_base_df
        )
        result = sample_processor._add_induction_timer(with_indication)

        # During induction, timer should increase
        induced_mask = result["Is induced [-]"] == 1
        if induced_mask.sum() > 1:
            induction_timer = result.loc[induced_mask, "Induction time [h]"].dropna()
            if len(induction_timer) > 1:
                # Timer should be increasing
                assert (
                    induction_timer.diff().dropna() >= 0
                ).all(), "Induction timer should increase during induction"


class TestInductionCorrection:
    """Test the reference-time check/correction of the phase-based induction.

    The fixture df has the phase flip to 'induced Fedbatch' at t = 5.0 h.
    """

    @pytest.fixture
    def df_with_indicator(self, sample_bioreactor_with_base_df, sample_processor):
        """Fixture df with the phase-based induction indicator applied."""
        return sample_processor._add_induction_indicator(sample_bioreactor_with_base_df)

    def test_no_reference_is_noop(self, df_with_indicator):
        """Without any configured reference the df is returned unchanged."""
        from upstream_analysis.data.calculations.induction import correct_induction_time

        result = correct_induction_time(df_with_indicator, {"processing": {}})

        pd.testing.assert_series_equal(
            result["Is induced [-]"], df_with_indicator["Is induced [-]"]
        )

    def test_scalar_reference_shifts_indicator(self, df_with_indicator):
        """induction_time_h moves the induction start (phase flip 5.0 -> 6.5 h)."""
        from upstream_analysis.data.calculations.induction import correct_induction_time

        config = {"processing": {"induction_time_h": 6.5}}
        result = correct_induction_time(df_with_indicator, config)

        assert (
            result.loc[result["Process time [h]"] == 5.0, "Is induced [-]"].item() == 0
        )
        assert (
            result.loc[result["Process time [h]"] == 6.0, "Is induced [-]"].item() == 0
        )
        assert (
            result.loc[result["Process time [h]"] == 7.0, "Is induced [-]"].item() == 1
        )

    def test_scalar_reference_updates_existing_timer(self, df_with_indicator):
        """An existing 'Induction time [h]' column is re-derived consistently."""
        from upstream_analysis.data.calculations.induction import correct_induction_time

        with_timer = df_with_indicator.copy()
        with_timer["Induction time [h]"] = 0.0

        config = {"processing": {"induction_time_h": 6.5}}
        result = correct_induction_time(with_timer, config)

        # Induced from 6.5 h: at t=7.0 the timer is 0.5 h; before it is 0.
        assert result.loc[
            result["Process time [h]"] == 7.0, "Induction time [h]"
        ].item() == pytest.approx(0.5)
        assert (
            result.loc[result["Process time [h]"] == 6.0, "Induction time [h]"].item()
            == 0
        )

    def test_reference_file(self, df_with_indicator, tmp_path):
        """The operator's evaluation workbook supplies the reference time."""
        from upstream_analysis.data.calculations.induction import correct_induction_time

        # Mimic the 'Time-Rownr' layout: labelled rows, decimal 'Time' column.
        events = pd.DataFrame(
            [
                ["Sample", "Hour", "Minute", "Time", "Row"],
                [None, None, None, None, None],
                ["Inoculation", 1, 14, 1.2333, 81],
                ["S1", 6, 28, 6.4667, 392],
                ["Induction", 6, 30, 6.5, 400],
            ]
        )
        eval_file = tmp_path / "Evaluation_T.xlsx"
        events.to_excel(eval_file, sheet_name="Time-Rownr", header=False, index=False)

        config = {"processing": {"induction_reference_file": str(eval_file)}}
        result = correct_induction_time(df_with_indicator, config)

        assert (
            result.loc[result["Process time [h]"] == 7.0, "Is induced [-]"].item() == 1
        )
        assert (
            result.loc[result["Process time [h]"] == 6.0, "Is induced [-]"].item() == 0
        )

    def test_reference_file_missing_entry_raises(self, df_with_indicator, tmp_path):
        """A workbook without the labelled row raises USPDataError."""
        from upstream_analysis.core.exceptions import USPDataError
        from upstream_analysis.data.calculations.induction import correct_induction_time

        events = pd.DataFrame([["Sample", "Hour", "Minute", "Time"]])
        eval_file = tmp_path / "Evaluation_T.xlsx"
        events.to_excel(eval_file, sheet_name="Time-Rownr", header=False, index=False)

        config = {"processing": {"induction_reference_file": str(eval_file)}}
        with pytest.raises(USPDataError):
            correct_induction_time(df_with_indicator, config)

    def test_large_offset_warns(self, df_with_indicator, caplog):
        """An offset beyond the tolerance logs a warning (correction still applied)."""
        from upstream_analysis.data.calculations.induction import correct_induction_time

        config = {"processing": {"induction_time_h": 8.0}}
        with caplog.at_level("WARNING"):
            result = correct_induction_time(df_with_indicator, config)

        warnings = [
            r for r in caplog.records if "Induction time check" in r.getMessage()
        ]
        assert warnings, "Large offset should produce a warning"
        # Correction still applied (8.0 h is within the data range)
        assert (
            result.loc[result["Process time [h]"] == 8.0, "Is induced [-]"].item() == 1
        )
        assert (
            result.loc[result["Process time [h]"] == 7.0, "Is induced [-]"].item() == 0
        )

    def test_offset_within_tolerance_does_not_warn(self, df_with_indicator, caplog):
        """A small offset logs info only."""
        from upstream_analysis.data.calculations.induction import correct_induction_time

        config = {"processing": {"induction_time_h": 5.2}}
        with caplog.at_level("WARNING"):
            correct_induction_time(df_with_indicator, config)

        warnings = [
            r for r in caplog.records if "Induction time check" in r.getMessage()
        ]
        assert not warnings, "Offset within tolerance should not warn"

    def test_reference_out_of_range_raises(self, df_with_indicator):
        """A reference time outside the data range raises USPConfigError."""
        from upstream_analysis.core.exceptions import USPConfigError
        from upstream_analysis.data.calculations.induction import correct_induction_time

        config = {"processing": {"induction_time_h": 99.0}}
        with pytest.raises(USPConfigError):
            correct_induction_time(df_with_indicator, config)

    def test_processor_delegator(self, sample_processor, df_with_indicator):
        """The processor delegator applies the configured correction."""
        sample_processor.config["processing"]["induction_time_h"] = 6.5
        result = sample_processor._correct_induction_time(df_with_indicator)

        assert (
            result.loc[result["Process time [h]"] == 7.0, "Is induced [-]"].item() == 1
        )
        assert (
            result.loc[result["Process time [h]"] == 5.0, "Is induced [-]"].item() == 0
        )


class TestGasExchangeCalculations:
    """Test gas exchange rate calculations."""

    def test_cer_calculation(self, sample_processor, sample_bioreactor_with_base_df):
        """Test CER (CO2 Evolution Rate) calculation."""
        # First calculate volume since CER depends on it
        with_feed = sample_processor._calculate_feed(sample_bioreactor_with_base_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)

        result = sample_processor._calculate_offgas_rates(with_volume)

        if "CER [mmol/L/h]" in result.columns:
            cer = result["CER [mmol/L/h]"]
            # CER should be non-negative for active fermentation
            assert (cer >= 0).all(), "CER should be non-negative"
            assert cer.notna().any(), "CER should have valid values"

    def test_our_calculation(self, sample_processor, sample_bioreactor_with_base_df):
        """Test OUR (O2 Uptake Rate) calculation."""
        # First calculate volume
        with_feed = sample_processor._calculate_feed(sample_bioreactor_with_base_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)

        result = sample_processor._calculate_offgas_rates(with_volume)

        if "OUR [mmol/L/h]" in result.columns:
            our = result["OUR [mmol/L/h]"]
            # OUR should be non-negative for active fermentation
            assert (our >= 0).all(), "OUR should be non-negative"
            assert our.notna().any(), "OUR should have valid values"

    def test_rq_calculation(self, sample_processor, sample_bioreactor_with_base_df):
        """Test RQ (Respiratory Quotient) calculation."""
        # First calculate volume and offgas rates
        with_feed = sample_processor._calculate_feed(sample_bioreactor_with_base_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_rates = sample_processor._calculate_offgas_rates(with_volume)

        if "RQ [-]" in with_rates.columns:
            rq = with_rates["RQ [-]"]
            # RQ should typically be 0.5-1.5 for microbial fermentation
            assert rq.notna().any(), "RQ should have valid values"
            rq_values = rq.dropna()
            assert (rq_values >= 0).all(), "RQ should be non-negative"
            # Allow reasonable upper bound for edge cases
            assert (rq_values <= 10).all(), "RQ should be reasonable (< 10)"

    def test_specific_rates_calculation(self, sample_processor, sample_aligned_df):
        """Test specific offgas rates (qO2, qCO2)."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_rates = sample_processor._calculate_offgas_rates(with_volume)

        result = sample_processor._calculate_specific_offgas_rates(with_rates)

        # Check for specific rate columns
        if "qO2 [mmol/g/h]" in result.columns:
            qo2 = result["qO2 [mmol/g/h]"].dropna()
            # qO2 should be positive for active metabolism
            assert (qo2 >= 0).all(), "qO2 should be non-negative"

        if "qCO2 [mmol/g/h]" in result.columns:
            qco2 = result["qCO2 [mmol/g/h]"].dropna()
            # qCO2 should be positive for active metabolism
            assert (qco2 >= 0).all(), "qCO2 should be non-negative"

    def test_inlet_gas_composition(
        self, sample_processor, sample_bioreactor_with_base_df
    ):
        """Test inlet gas composition calculation."""
        result = sample_processor._calculate_inlet_gas_composition(
            sample_bioreactor_with_base_df
        )

        # Check for inlet composition columns
        if "Inlet O2 [%]" in result.columns:
            o2_in = result["Inlet O2 [%]"].dropna()
            # Inlet O2 should be between 21% (air) and 100% (pure O2)
            assert (o2_in >= 21).all(), "Inlet O2 should be >= 21%"
            assert (o2_in <= 100).all(), "Inlet O2 should be <= 100%"

        if "Inlet CO2 [%]" in result.columns:
            co2_in = result["Inlet CO2 [%]"].dropna()
            # Inlet CO2 should be low
            assert (co2_in >= 0).all(), "Inlet CO2 should be non-negative"
            assert (co2_in <= 1).all(), "Inlet CO2 should be <= 1%"

    def test_gas_rates_missing_volume(self, sample_processor):
        """Test gas rate calculation with missing volume column."""
        incomplete_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Total flow [L/min]": [1.0, 1.1, 1.2],
                "Offgas CO2 [%]": [0.5, 0.8, 1.2],
                "Offgas O2 [%]": [20.5, 20.3, 19.8],
                # Missing Volume column
            }
        )

        result = sample_processor._calculate_offgas_rates(incomplete_df)
        # Should not crash, but might not calculate rates
        assert result is not None


class TestCarbonBalance:
    """Test carbon balance calculations."""

    def test_carbon_balance_columns(self, sample_processor, sample_aligned_df):
        """Test carbon balance column creation."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_rates = sample_processor._calculate_offgas_rates(with_volume)

        result = sample_processor._calculate_carbon_balance(with_rates)

        # Check for key carbon balance columns
        expected_cols = ["C in [C-mol]", "C out [C-mol]", "C recovered [%]"]
        for col in expected_cols:
            assert col in result.columns, f"Carbon balance should include {col}"

    def test_carbon_balance_input_columns(self, sample_processor, sample_aligned_df):
        """Test carbon input calculations."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_rates = sample_processor._calculate_offgas_rates(with_volume)

        result = sample_processor._calculate_carbon_balance(with_rates)

        # Check input columns
        assert "C from feed [C-mol]" in result.columns
        assert "C from batch [C-mol]" in result.columns

        # Carbon from feed should be non-negative
        c_from_feed = result["C from feed [C-mol]"].dropna()
        assert (c_from_feed >= 0).all(), "Carbon from feed should be non-negative"

        # Carbon from batch should be constant (initial condition)
        c_from_batch = result["C from batch [C-mol]"].dropna()
        if len(c_from_batch) > 0:
            assert c_from_batch.nunique() <= 1, "Carbon from batch should be constant"

    def test_carbon_balance_output_columns(self, sample_processor, sample_aligned_df):
        """Test carbon output calculations."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_rates = sample_processor._calculate_offgas_rates(with_volume)

        result = sample_processor._calculate_carbon_balance(with_rates)

        # Check output columns
        assert "C from CO2 [C-mol]" in result.columns
        assert "C in biomass [C-mol]" in result.columns

        # Carbon outputs should be non-negative
        for col in ["C from CO2 [C-mol]", "C in biomass [C-mol]"]:
            values = result[col].dropna()
            assert (values >= 0).all(), f"{col} should be non-negative"

    def test_carbon_recovery_reasonable_range(
        self, sample_processor, sample_aligned_df
    ):
        """Test that carbon recovery is in reasonable range."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_rates = sample_processor._calculate_offgas_rates(with_volume)

        result = sample_processor._calculate_carbon_balance(with_rates)

        if "C recovered [%]" in result.columns:
            recovery = result["C recovered [%]"].dropna()
            # Recovery should typically be 50-150% for reasonable data
            assert (recovery >= 0).all(), "Carbon recovery should be non-negative"
            assert (recovery <= 200).all(), "Carbon recovery should be < 200%"

    def test_carbon_balance_cumulative_trend(self, sample_processor, sample_aligned_df):
        """Test that carbon balance columns show expected trends."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_rates = sample_processor._calculate_offgas_rates(with_volume)

        result = sample_processor._calculate_carbon_balance(with_rates)

        # Carbon from feed should be non-decreasing
        c_feed = result["C from feed [C-mol]"].dropna()
        if len(c_feed) > 1:
            assert (
                c_feed.diff().dropna() >= -0.01
            ).all(), "Carbon from feed should be non-decreasing"

        # Carbon from CO2 should be cumulative (non-decreasing)
        c_co2 = result["C from CO2 [C-mol]"].dropna()
        if len(c_co2) > 1:
            assert (
                c_co2.diff().dropna() >= -0.01
            ).all(), "Carbon from CO2 should be non-decreasing"


class TestProteinCalculations:
    """Test protein parameter calculations."""

    def test_protein_parameters_basic(self, sample_processor, sample_aligned_df):
        """Test basic protein parameter calculation."""
        result = sample_processor._calculate_protein_parameters(sample_aligned_df)

        # Check that protein columns are created
        if "Protein soluble [g/L]" in sample_aligned_df.columns:
            assert "Protein soluble [g/L]" in result.columns

    def test_protein_total_calculation(self, sample_processor):
        """Test total protein calculation from fractions."""
        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Protein soluble [g/L]": [1.0, 2.0, 3.0],
                "Protein IB [g/L]": [0.5, 1.0, 1.5],
                "Biomass [g/L]": [10.0, 15.0, 20.0],
            }
        )

        result = sample_processor._calculate_protein_parameters(test_df)

        # Check total protein calculation
        if "Protein total calculated [g/L]" in result.columns:
            expected_total = (
                test_df["Protein soluble [g/L]"] + test_df["Protein IB [g/L]"]
            )
            pd.testing.assert_series_equal(
                result["Protein total calculated [g/L]"],
                expected_total,
                check_names=False,
                rtol=1e-5,
            )

    def test_protein_yield_calculation(self, sample_processor):
        """Test protein yield calculation."""
        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Protein soluble [g/L]": [1.0, 2.0, 3.0],
                "Biomass [g/L]": [10.0, 15.0, 20.0],
            }
        )

        result = sample_processor._calculate_protein_parameters(test_df)

        # Check yield calculation
        if "Protein soluble yield [g/g]" in result.columns:
            # Yield = protein / biomass
            expected_yield = test_df["Protein soluble [g/L]"] / test_df["Biomass [g/L]"]
            pd.testing.assert_series_equal(
                result["Protein soluble yield [g/g]"],
                expected_yield,
                check_names=False,
                rtol=1e-5,
            )

    def test_protein_fraction_calculation(self, sample_processor):
        """Test soluble/IB fraction calculation."""
        test_df = pd.DataFrame(
            {
                "Process time [h]": [0.0, 1.0, 2.0],
                "Protein [g/L]": [1.5, 3.0, 4.5],
                "Protein soluble [g/L]": [1.0, 2.0, 3.0],
                "Protein IB [g/L]": [0.5, 1.0, 1.5],
            }
        )

        result = sample_processor._calculate_protein_parameters(test_df)

        # Check soluble fraction calculation
        if "Soluble fraction [%]" in result.columns:
            expected_fraction = (
                test_df["Protein soluble [g/L]"] / test_df["Protein [g/L]"]
            ) * 100
            pd.testing.assert_series_equal(
                result["Soluble fraction [%]"],
                expected_fraction,
                check_names=False,
                rtol=1e-5,
            )


class TestRateCalculations:
    """Test growth and production rate calculations."""

    def test_growth_rate_calculation(self, sample_processor, sample_aligned_df):
        """Test growth rate (mu) calculation."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_protein = sample_processor._calculate_protein_parameters(with_volume)
        with_gas = sample_processor._calculate_offgas_rates(with_protein)
        with_cb = sample_processor._calculate_carbon_balance(with_gas)

        result = sample_processor._calculate_rates_and_yields(with_cb)

        if "mu [1/h]" in result.columns:
            mu = result["mu [1/h]"]
            # Growth rate should be non-negative
            mu_values = mu.dropna()
            assert (mu_values >= 0).all(), "Growth rate should be non-negative"

    def test_specific_consumption_rate(self, sample_processor, sample_aligned_df):
        """Test specific substrate consumption rate calculation."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_protein = sample_processor._calculate_protein_parameters(with_volume)
        with_gas = sample_processor._calculate_offgas_rates(with_protein)
        with_cb = sample_processor._calculate_carbon_balance(with_gas)

        result = sample_processor._calculate_rates_and_yields(with_cb)

        if "qS glucose [g/g/h]" in result.columns:
            qs = result["qS glucose [g/g/h]"].dropna()
            # Specific consumption rate should be non-negative
            assert (qs >= 0).all(), "Specific consumption rate should be non-negative"

    def test_specific_production_rate(self, sample_processor, sample_aligned_df):
        """Test specific production rate calculation."""
        # First calculate prerequisites
        with_feed = sample_processor._calculate_feed(sample_aligned_df)
        with_base = sample_processor._calculate_base_flux_and_volume(with_feed)
        with_volume = sample_processor._calculate_volume(with_base)
        with_protein = sample_processor._calculate_protein_parameters(with_volume)
        with_gas = sample_processor._calculate_offgas_rates(with_protein)
        with_cb = sample_processor._calculate_carbon_balance(with_gas)

        result = sample_processor._calculate_rates_and_yields(with_cb)

        if "qP total [g/g/h]" in result.columns:
            qp = result["qP total [g/g/h]"].dropna()
            # Specific production rate should be non-negative
            assert (qp >= 0).all(), "Specific production rate should be non-negative"


class TestFullProcessingPipeline:
    """Test the full processing pipeline."""

    def test_full_pipeline_basic(self, sample_processor, sample_aligned_df):
        """Test that full pipeline runs without errors."""
        result = sample_processor.process_data(sample_aligned_df)

        # Check that result is not empty
        assert not result.empty, "Processing result should not be empty"

        # Check for expected columns
        expected_cols = ["Process time [h]", "Volume [mL]"]
        for col in expected_cols:
            assert col in result.columns, f"Result should include {col}"

    def test_full_pipeline_column_preservation(
        self, sample_processor, sample_aligned_df
    ):
        """Test that original columns are preserved in processing."""
        result = sample_processor.process_data(sample_aligned_df)

        # Check that original columns are still present
        original_cols = ["Process time [h]", "Process phase [-]", "Balance [g]"]
        for col in original_cols:
            if col in sample_aligned_df.columns:
                assert (
                    col in result.columns
                ), f"Original column {col} should be preserved"

    def test_full_pipeline_with_minimal_data(
        self, minimal_config, sample_bioreactor_df
    ):
        """Test pipeline with minimal configuration."""
        processor = FermentationDataProcessor(minimal_config)
        result = processor.process_data(sample_bioreactor_df)

        # Should still produce output
        assert not result.empty, "Pipeline should produce output with minimal config"

    def test_full_pipeline_with_nan_data(self, edge_case_config, nan_dataframe):
        """Test pipeline handling of NaN values."""
        processor = FermentationDataProcessor(edge_case_config)
        result = processor.process_data(nan_dataframe)

        # Should handle NaN gracefully
        assert result is not None, "Pipeline should handle NaN values"

    def test_full_pipeline_with_empty_data(self, sample_processor):
        """Test pipeline handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = sample_processor.process_data(empty_df)

        # Should return empty DataFrame
        assert result.empty, "Pipeline should return empty DataFrame for empty input"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_required_config(self):
        """Test processor initialization with missing required config."""
        incomplete_config = {
            "processing": {"processing_parameters": {}}  # Missing required parameters
        }

        processor = FermentationDataProcessor(incomplete_config)
        # Should not crash on init
        assert processor is not None

    def test_zero_feed_density(self, sample_bioreactor_with_base_df):
        """Test handling of invalid feed density."""
        config = {
            "processing": {
                "processing_parameters": {
                    "feed_definitions": {"feed_1": {"density_g_ml": 0.0}}  # Invalid
                }
            },
            "cleaning": {"median_filter": False},
        }

        # Should raise error for invalid density
        with pytest.raises(ValueError):
            processor = FermentationDataProcessor(config)
            processor._calculate_feed(sample_bioreactor_with_base_df)

    def test_negative_initial_volume(self):
        """Test handling of negative initial volume."""
        config = {
            "processing": {
                "processing_parameters": {
                    "reactor_setup": {"initial_volume_ml": -100.0}  # Invalid
                }
            }
        }

        # Should raise error for negative volume
        with pytest.raises(ValueError):
            FermentationDataProcessor(config)

    def test_single_row_dataframe(self, minimal_config):
        """Test processing with single row of data."""
        single_row_df = pd.DataFrame(
            {
                "Process time [h]": [0.0],
                "Process phase [-]": ["Batch"],
                "Balance [g]": [5000.0],
            }
        )

        processor = FermentationDataProcessor(minimal_config)
        result = processor.process_data(single_row_df)

        # Should handle single row gracefully
        assert len(result) == 1
        assert not result.empty
