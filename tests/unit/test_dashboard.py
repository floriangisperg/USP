"""Tests for the interactive Bokeh dashboard (src/upstream_analysis/visualization/dashboard.py).

Mirrors the style of test_visualization.py: inline DataFrames, stdlib
TemporaryDirectory, no shared fixtures. Skips entirely when bokeh is absent.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("bokeh")

from upstream_analysis.visualization.dashboard import (  # noqa: E402
    DEFAULT_CATEGORIES,
    InteractiveDashboard,
)


def _sample_df(time_col="Process time [h]", n=6):
    """A small DataFrame spanning a few DEFAULT_CATEGORIES columns."""
    return pd.DataFrame(
        {
            time_col: list(range(n)),
            "CER [mmol/L/h]": [0.1 * i for i in range(n)],  # continuous
            "OUR [mmol/L/h]": [0.2 * i for i in range(n)],
            "RQ [-]": [1.0] * n,
            "Temperature [°C]": [37.0] * n,
            "pH [-]": [7.0] * n,
            "Biomass [g/L]": [None, None, 5.0, None, None, 6.0],  # sparse
        }
    )


class TestInteractiveDashboard:
    def test_single_run_produces_html(self):
        with tempfile.TemporaryDirectory() as td:
            out = InteractiveDashboard(td).plot_selected_parameters(
                _sample_df(), run_name="F1"
            )
            assert out is not None
            assert Path(out).exists()
            assert Path(out).suffix == ".html"

    def test_comparison_produces_html(self):
        with tempfile.TemporaryDirectory() as td:
            out = InteractiveDashboard(td).plot_comparison(
                {"F1": _sample_df(), "F2": _sample_df()}
            )
            assert out is not None and Path(out).exists()

    def test_custom_time_col(self):
        df = _sample_df(time_col="Time [h]")
        with tempfile.TemporaryDirectory() as td:
            out = InteractiveDashboard(
                td, time_col="Time [h]"
            ).plot_selected_parameters(df)
            assert out is not None and Path(out).exists()

    def test_all_nan_column_skipped(self):
        df = _sample_df()
        df["CER [mmol/L/h]"] = pd.NA
        with tempfile.TemporaryDirectory() as td:
            out = InteractiveDashboard(td).plot_selected_parameters(df)
            # Other columns still produce a dashboard.
            assert out is not None and Path(out).exists()

    def test_empty_df_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            out = InteractiveDashboard(td).plot_selected_parameters(pd.DataFrame())
            assert out is None

    def test_parameters_filter(self):
        with tempfile.TemporaryDirectory() as td:
            out = InteractiveDashboard(td).plot_selected_parameters(
                _sample_df(), parameters=["CER [mmol/L/h]"]
            )
            assert out is not None and Path(out).exists()

    def test_categories_override(self):
        cats = {"Custom": ["CER [mmol/L/h]", "OUR [mmol/L/h]"]}
        with tempfile.TemporaryDirectory() as td:
            out = InteractiveDashboard(td).plot_selected_parameters(
                _sample_df(), categories=cats, output_filename="c.html"
            )
            assert out is not None
            assert Path(out).name == "c.html"

    def test_output_dir_auto_created(self):
        with tempfile.TemporaryDirectory() as td:
            nested = Path(td) / "nested" / "out"
            out = InteractiveDashboard(nested).plot_selected_parameters(_sample_df())
            assert nested.is_dir()
            assert out is not None and Path(out).exists()

    def test_comparison_empty_dict_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            assert InteractiveDashboard(td).plot_comparison({}) is None

    def test_missing_bokeh_raises_helpful_error(self, monkeypatch):
        # Simulate bokeh being unavailable so the constructor gives a clear message.
        monkeypatch.setattr(
            "upstream_analysis.visualization.dashboard._BOKEH_OK", False
        )
        with pytest.raises(ImportError, match=r"bokeh"):
            InteractiveDashboard("/tmp/whatever")


def test_default_categories_shape():
    assert isinstance(DEFAULT_CATEGORIES, dict)
    assert DEFAULT_CATEGORIES  # non-empty
    assert all(isinstance(v, list) and v for v in DEFAULT_CATEGORIES.values())
