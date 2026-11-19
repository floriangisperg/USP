"""Tests for the one-call pipeline runner (uses the synthetic run fixture)."""

import pandas as pd
import pytest

from upstream_analysis import load_config, run_directory, run_pipeline


@pytest.fixture(scope="module")
def synthetic_result(tmp_path_factory, synthetic_run):
    out_dir = tmp_path_factory.mktemp("synthetic_results")
    return run_directory(synthetic_run, output_dir=out_dir)


def test_run_directory_processes_synthetic_run(synthetic_result):
    df = synthetic_result.processed
    assert len(df) >= 800
    assert df["Process time [h]"].min() >= -1e-9
    assert not (df["Process phase [-]"] == "Preparation").any()


def test_run_directory_writes_outputs(synthetic_result):
    out = synthetic_result.output_dir
    assert (out / "processed_data.csv").exists()
    assert (out / "samples_points_data.csv").exists()
    assert synthetic_result.plot_paths, "expected overview plots"


def test_samples_are_flagged(synthetic_result):
    samples = synthetic_result.processed[
        synthetic_result.processed["is_sample"].eq(True)
    ]
    assert len(samples) == 9


def test_feed_and_induction_sanity(synthetic_result):
    df = synthetic_result.processed
    feed_final = df["Feed [g]"].dropna().iloc[-1]
    assert 150 < feed_final < 300, f"unexpected cumulative feed: {feed_final}"
    induced = df[df["Is induced [-]"] > 0]["Process time [h]"]
    assert len(induced) > 0 and abs(induced.min() - 16.0) < 0.34
    late = df[df["is_sample"].eq(True) & (df["Process time [h]"] > 16)]
    frac = (late["Protein IB [g/L]"] / late["Protein [g/L]"]).dropna()
    assert 0.3 < frac.iloc[-1] < 0.5


def test_run_pipeline_with_explicit_paths(tmp_path, synthetic_run):
    config = load_config(synthetic_run / "run_config.yaml")
    csv_files = list((synthetic_run / "data").glob("*.csv"))
    xlsx_files = list((synthetic_run / "data").glob("*.xlsx"))
    result = run_pipeline(
        config,
        csv_files[0],
        offline_path=xlsx_files[0],
        output_dir=tmp_path,
        plot_parameters=["Biomass [g/L]"],
    )
    assert isinstance(result.processed, pd.DataFrame)
    assert len(result.plot_paths) == 1  # only the requested parameter


def test_run_directory_rejects_ambiguous_data(tmp_path):
    run_dir = tmp_path / "broken_run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "run_config.yaml").write_text("metadata:\n  fermentation_id: x\n")
    (run_dir / "data" / "a.csv").write_text("x", encoding="utf-8")
    (run_dir / "data" / "b.csv").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="expected one online CSV"):
        run_directory(run_dir)
