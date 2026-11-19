"""Tests for the consolidation module (uses the committed synthetic example)."""

from pathlib import Path

import pandas as pd
import pytest

from upstream_analysis import ConsolidationConfig, consolidate_runs, find_column
from upstream_analysis.consolidation import improve_protein_concentrations

SYNTHETIC_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "synthetic_example"
)


def test_find_column_exact_and_encoding_fallback():
    df = pd.DataFrame(columns=["Temperature [Â°C]", "pH [-]"])
    assert find_column(df, "pH [-]") == "pH [-]"
    assert find_column(df, "Temperature [°C]") == "Temperature [Â°C]"
    assert find_column(df, "Not There [x]") is None


def test_improve_protein_concentrations_recomputes_derived_columns():
    df = pd.DataFrame(
        {
            "Protein soluble [g/L]": [0.6, 1.0],
            "Protein IB [g/L]": [0.4, 1.0],
            "Biomass [g/L]": [10.0, 20.0],
        }
    )
    out = improve_protein_concentrations(df)
    assert out["Protein total calculated [g/L]"].tolist() == [1.0, 2.0]
    assert out["Soluble fraction [%]"].tolist() == [60.0, 50.0]
    assert out["IB fraction [%]"].tolist() == [40.0, 50.0]
    assert out["Protein IB yield [g/g]"].iloc[0] == pytest.approx(0.04)


@pytest.fixture(scope="module")
def consolidated_campaign(tmp_path_factory):
    """Run the pipeline on the synthetic example as a one-run campaign."""
    from upstream_analysis import run_directory

    campaign = tmp_path_factory.mktemp("campaign")
    run_dir = campaign / "SIM01"
    (run_dir / "results").mkdir(parents=True)
    run_directory(SYNTHETIC_EXAMPLE, output_dir=run_dir / "results")
    data = consolidate_runs(campaign, ConsolidationConfig(output_prefix="test"))
    return campaign, data


def test_consolidate_runs_writes_workbooks(consolidated_campaign):
    campaign, _ = consolidated_campaign
    assert (campaign / "test_consolidated_results.xlsx").exists()
    assert (campaign / "test_consolidated_results_sparse.xlsx").exists()


def test_consolidate_runs_tags_run_name_and_filters_feed_phase(consolidated_campaign):
    _, data = consolidated_campaign
    assert list(data.keys()) == ["SIM01"]
    df = data["SIM01"]
    assert (df["run_name"] == "SIM01").all()
    # feed-phase filtering keeps any pre-feed sample points but drops batch rows
    assert (df["Feed [mL]"].fillna(0) > 0).any()
    assert df["Process time [h]"].min() > 1.0  # batch start (t=0) trimmed


def test_consolidate_runs_reads_excel_sheets(consolidated_campaign):
    campaign, _ = consolidated_campaign
    sheets = pd.read_excel(
        campaign / "test_consolidated_results.xlsx", sheet_name=None, engine="openpyxl"
    )
    assert set(sheets) >= {"All_Time_Points", "Run_Summary", "SIM01"}
    assert len(sheets["Run_Summary"]) == 1


def test_consolidate_runs_empty_campaign_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no runs with results"):
        consolidate_runs(tmp_path)
