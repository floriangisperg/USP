"""Interactive HTML dashboards for fermentation data (Bokeh).

A self-contained alternative to the per-parameter PNGs produced by
:class:`~upstream_analysis.visualization.plotting.DataVisualizer`. Emits a
single ``.html`` file (no server, opens offline in any browser) with:

- **category tabs** grouping the derived parameters,
- one small multiple per parameter, plus a curated **Functional** tab of
  relationship plots (pO2 vs O2 enrichment, pO2 vs OUR, growth, mu, carbon
  closure, product vs biomass),
- multiple runs overlaid, with a **per-tab run selector** (checkboxes) that
  filters every plot in the tab at once (and per-plot legend toggling), and
- hover tooltips (run / time / value), plus pan/zoom/reset.

Bokeh is an optional dependency::

    pip install -e ".[dashboard]"

Example::

    from upstream_analysis import InteractiveDashboard
    InteractiveDashboard("results").plot_selected_parameters(df_processed, run_name="F1")
    InteractiveDashboard("results").plot_comparison({"F1": df1, "F2": df2})
"""
import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .plotting import SPARSE_NAN_FRACTION

logger = logging.getLogger(__name__)

# --- Optional Bokeh dependency -------------------------------------------------
try:  # pragma: no cover - exercised when bokeh is installed
    from bokeh.layouts import column, gridplot
    from bokeh.models import (
        CheckboxGroup,
        CustomJS,
        DataRange1d,
        Div,
        HoverTool,
        LinearAxis,
        Tabs,
    )
    from bokeh.plotting import figure, output_file, save

    try:  # Bokeh 3.x names the tabs child 'TabPanel' (3.4 briefly 'TabPane'); 2.x used 'Panel'
        from bokeh.models import TabPanel
    except ImportError:
        try:
            from bokeh.models import TabPane as TabPanel
        except ImportError:  # pragma: no cover - bokeh 2.x
            from bokeh.models import Panel as TabPanel
    _BOKEH_OK = True
    _IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _e:  # pragma: no cover - exercised when bokeh is absent
    _BOKEH_OK = False
    _IMPORT_ERROR = _e


# Default parameter grouping, keyed by tab title. Column names are the standard
# internal names produced by the pipeline. Only columns actually present in the
# supplied DataFrame are plotted, so this can be a superset of any single run.
DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "Gas flows": [
        "Air flow [L/min]", "Total flow [L/min]", "O2 flow [L/min]", "Gas mix [-]",
        "Offgas CO2 [%]", "Offgas O2 [%]", "Inlet O2 [%]", "Inlet CO2 [%]",
    ],
    "Gas-exchange rates": [
        "CER [mmol/L/h]", "OUR [mmol/L/h]", "RQ [-]",
        "qO2 [mmol/g/h]", "qCO2 [mmol/g/h]",
    ],
    "Pumps": [
        "Acid pump [%]", "Acid pump duration [-]",
        "Base pump [%]", "Base pump duration [-]",
        "Inducer pump [%]", "Inducer pump duration [-]",
        "Feed pump [%]", "Feed pump duration [-]",
    ],
    "Environment": [
        "Stirrer speed [1/min]", "Temperature [°C]", "Balance [g]",
        "Analog IO1 [%]", "Analog IO2 [%]", "pH [-]", "pO2 [%]", "Foam [-]",
    ],
    "Feed & volume": [
        "Feed [g]", "Feed [mL]", "Base volume added [mL]",
        "Volume [mL]", "Cumulative sample volume [mL]", "Volume corrected [mL]",
        "Feed rate exp [%]", "Feed rate ind [%]",
    ],
    "Biomass & substrate": [
        "Biomass [g/L]", "Biomass total [g]", "Glucose [g/L]", "Acetate [g/L]",
        "OD600 online [-]", "OD600 [-]",
        "Glucose consumed [g]", "Glucose consumed rate [g/h]",
        "qS glucose [g/g/h]", "mu [1/h]",
        "qS offgas [g/g/h]", "qS offgas calibrated [g/g/h]",
    ],
    "Protein": [
        "Protein [g/L]", "Protein soluble [g/L]", "Protein IB [g/L]",
        "Protein total calculated [g/L]",
        "Soluble fraction [%]", "IB fraction [%]",
        "Protein soluble yield [g/g]", "Protein IB yield [g/g]", "Protein total yield [g/g]",
        "Protein produced [g]", "Protein production rate [g/h]", "qP total [g/g/h]",
    ],
    "Carbon balance": [
        "C from feed [C-mol]", "C from batch [C-mol]", "C in [C-mol]", "C from CO2 [C-mol]",
        "C in biomass [C-mol]", "C in product [C-mol]", "C in acetate [C-mol]",
        "C residual glucose [C-mol]", "C out [C-mol]",
        "C to CO2 [% of fed]", "C to biomass [% of fed]", "C to product [% of fed]",
        "C residual glucose [% of fed]",
        "C recovered [%]", "C balance closure [%]", "C balance error [%]",
        "C recovered (measured) [%]",
    ],
    "Induction": [
        "Is induced [-]", "Induction time [h]",
    ],
}

# Distinct colors for overlaying runs (cycled if more runs than colors).
_RUN_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#17becf", "#e377c2",
]

_TOOLS = "pan,box_zoom,wheel_zoom,reset,save"


class InteractiveDashboard:
    """Builds a self-contained interactive HTML dashboard for one or more runs.

    Args:
        output_dir: Directory to write the HTML file into (created if missing).
        time_col: Column to use for the x-axis (all runs).
    """

    def __init__(self, output_dir: Union[str, Path], time_col: str = "Process time [h]"):
        if not _BOKEH_OK:
            raise ImportError(
                "Interactive dashboards require bokeh. Install it with: "
                "pip install -e '.[dashboard]'"
            ) from _IMPORT_ERROR
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_col = time_col
        logger.info(f"InteractiveDashboard initialized. HTML will be saved to: {self.output_dir}")

    # --- public API ------------------------------------------------------------

    def plot_selected_parameters(
        self,
        df: pd.DataFrame,
        parameters: Optional[list] = None,
        run_name: str = "run",
        categories: Optional[dict] = None,
        output_filename: str = "dashboard.html",
        max_points: Optional[int] = None,
    ) -> Optional[Path]:
        """Single run -> one HTML with category tabs (+ Functional tab)."""
        return self._build({run_name: df}, parameters, categories, output_filename, max_points)

    def plot_comparison(
        self,
        runs: dict,
        parameters: Optional[list] = None,
        categories: Optional[dict] = None,
        output_filename: str = "comparison.html",
        max_points: Optional[int] = None,
    ) -> Optional[Path]:
        """Multiple runs overlaid on every plot (toggle runs via per-tab selector)."""
        if not runs:
            logger.error("No runs supplied to plot_comparison.")
            return None
        return self._build(runs, parameters, categories, output_filename, max_points)

    # --- assembly --------------------------------------------------------------

    def _build(
        self,
        runs: dict,
        parameters: Optional[list],
        categories: Optional[dict],
        output_filename: str,
        max_points: Optional[int],
    ) -> Optional[Path]:
        use_default_categories = categories is None
        categories = categories if categories is not None else DEFAULT_CATEGORIES
        param_filter = set(parameters) if parameters is not None else None
        run_names = list(runs.keys())

        tabs = []
        for cat_name, cols in categories.items():
            figs_with_renders = []
            for col in cols:
                if param_filter is not None and col not in param_filter:
                    continue
                result = self._time_series_figure(col, runs, max_points)
                if result is not None:
                    figs_with_renders.append(result)
            if figs_with_renders:
                tabs.append(self._make_tab(cat_name, figs_with_renders, run_names))

        # Curated relationship plots (only when using the default categories).
        if use_default_categories:
            func_figs = self._functional_figures(runs, max_points)
            if func_figs:
                tabs.append(self._make_tab("Functional", func_figs, run_names))

        if not tabs:
            logger.warning("No plottable parameters found; dashboard not written.")
            return None

        title = "Fermentation dashboard"
        header = Div(
            text=(
                f"<h2>{title}</h2>"
                f"<p>Runs: {', '.join(run_names)} &nbsp;|&nbsp; x-axis: {self.time_col} "
                "&nbsp;|&nbsp; use the per-tab checkboxes to filter runs; click a legend "
                "entry to toggle a single trace; hover for values.</p>"
            )
        )
        layout = column(header, Tabs(tabs=tabs), sizing_mode="scale_width")

        out = self.output_dir / output_filename
        output_file(out, title=title, mode="inline")
        save(layout)
        logger.info(f"Saved interactive dashboard: {out}")
        return out

    def _make_tab(self, title: str, figs_with_renders: list, run_names: list):
        """Wrap a grid of figures in a tab; add a run selector when >1 run."""
        figs = [f for f, _ in figs_with_renders]
        renders = [(r, ri) for _, rs in figs_with_renders for (r, ri) in rs]
        body = gridplot(figs, ncols=2)

        if len(run_names) > 1 and renders:
            flat = [r for r, _ in renders]
            idx = [ri for _, ri in renders]
            cbg = CheckboxGroup(labels=run_names, active=list(range(len(run_names))))
            code = (
                "const a = cbg.active;\n"
                f"const idx = {json.dumps(idx)};\n"
                "for (let k = 0; k < rs.length; k++) { rs[k].visible = (a.indexOf(idx[k]) >= 0); }"
            )
            cbg.js_on_change("active", CustomJS(args={"rs": flat, "cbg": cbg}, code=code))
            hint = Div(text="<b>Show runs:</b> (uncheck to hide a run across this whole tab)")
            body = column(hint, cbg, body)

        return TabPanel(title=title, child=body)

    # --- figure builders -------------------------------------------------------

    def _series(self, df: pd.DataFrame, col: str, max_points: Optional[int]):
        """Numeric (time, col) rows, NaN-dropped and optionally downsampled."""
        if col not in df.columns or self.time_col not in df.columns:
            return None
        sub = df[[self.time_col, col]].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.empty:
            return None
        if max_points and len(sub) > max_points:
            sub = sub.iloc[:: max(1, int(np.ceil(len(sub) / max_points)))]
        return sub

    def _draw(self, fig, x, y, is_sparse, color, legend_label, name,
              y_range_name=None, dash="solid", alpha=1.0):
        """Add one line (continuous) or scatter (sparse) renderer; return it."""
        common = dict(color=color, legend_label=legend_label, name=name)
        if y_range_name is not None:
            common["y_range_name"] = y_range_name
        if is_sparse:
            return fig.scatter(x, y, size=6, alpha=0.85 * alpha, **common)
        return fig.line(x, y, line_width=1.5, line_dash=dash, alpha=alpha, **common)

    def _style(self, p, hover_label: str, legend_loc="top_left"):
        p.legend.click_policy = "hide"
        p.legend.location = legend_loc
        p.legend.label_text_font_size = "9pt"
        p.add_tools(HoverTool(tooltips=[("run", "$name"), ("time", "@x"), (hover_label, "@y")]))
        p.title.text_font_size = "10pt"

    def _time_series_figure(self, col: str, runs: dict, max_points: Optional[int]):
        """One parameter vs time, one renderer per run. Returns (fig, [(rend, run_idx), ...])."""
        total_len = total_valid = 0
        for df in runs.values():
            if col in df.columns:
                total_len += len(df[col])
                total_valid += int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        if total_valid == 0:
            return None
        nan_fraction = 1.0 - (total_valid / total_len) if total_len else 1.0
        is_sparse = nan_fraction > SPARSE_NAN_FRACTION

        p = figure(
            height=280, width=460, title=col, x_axis_label=self.time_col, y_axis_label=col,
            tools=_TOOLS, active_scroll="wheel_zoom", output_backend="webgl",
        )
        renders = []
        for i, (run_name, df) in enumerate(runs.items()):
            sub = self._series(df, col, max_points)
            if sub is None:
                continue
            color = _RUN_PALETTE[i % len(_RUN_PALETTE)]
            label = run_name if len(runs) > 1 else col
            r = self._draw(p, sub[self.time_col], sub[col], is_sparse, color, label, run_name)
            if r is not None:
                renders.append((r, i))
        if not renders:
            return None
        self._style(p, col)
        return p, renders

    def _twin_time_figure(self, left_col, right_col, runs, max_points, title, right_label):
        """Two quantities vs time on twin axes (left solid, right dashed)."""
        if not any(left_col in df.columns or right_col in df.columns for df in runs.values()):
            return None
        p = figure(
            height=300, width=620, title=title, x_axis_label=self.time_col, y_axis_label=left_col,
            tools=_TOOLS, active_scroll="wheel_zoom", output_backend="webgl",
        )
        p.extra_y_ranges = {"right": DataRange1d()}
        p.add_layout(LinearAxis(y_range_name="right", axis_label=right_label), "right")
        renders = []
        for i, (run_name, df) in enumerate(runs.items()):
            color = _RUN_PALETTE[i % len(_RUN_PALETTE)]
            sub_l = self._series(df, left_col, max_points)
            if sub_l is not None:
                r = self._draw(p, sub_l[self.time_col], sub_l[left_col], False, color,
                               f"{run_name} · {left_col}", run_name)
                if r is not None:
                    renders.append((r, i))
            sub_r = self._series(df, right_col, max_points)
            if sub_r is not None:
                r = self._draw(p, sub_r[self.time_col], sub_r[right_col], False, color,
                               f"{run_name} · {right_col}", run_name,
                               y_range_name="right", dash="dashed", alpha=0.85)
                if r is not None:
                    renders.append((r, i))
        if not renders:
            return None
        self._style(p, "value", legend_loc="top_right")
        return p, renders

    def _scatter_figure(self, x_col, y_col, runs, max_points, title=None):
        """y vs x scatter (e.g. pO2 vs OUR), one renderer per run."""
        if not any(x_col in df.columns and y_col in df.columns for df in runs.values()):
            return None
        title = title or f"{y_col} vs {x_col}"
        p = figure(
            height=300, width=520, title=title, x_axis_label=x_col, y_axis_label=y_col,
            tools=_TOOLS, active_scroll="wheel_zoom", output_backend="webgl",
        )
        renders = []
        for i, (run_name, df) in enumerate(runs.items()):
            if x_col not in df.columns or y_col not in df.columns or self.time_col not in df.columns:
                continue
            sub = df[[self.time_col, x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if sub.empty:
                continue
            if max_points and len(sub) > max_points:
                sub = sub.iloc[:: max(1, int(np.ceil(len(sub) / max_points)))]
            color = _RUN_PALETTE[i % len(_RUN_PALETTE)]
            r = p.scatter(
                sub[x_col], sub[y_col], size=5, color=color, alpha=0.7,
                legend_label=(run_name if len(runs) > 1 else title), name=run_name,
            )
            renders.append((r, i))
        if not renders:
            return None
        self._style(p, y_col)
        return p, renders

    def _functional_figures(self, runs: dict, max_points: Optional[int]) -> list:
        """Curated relationship plots that show the bioreactor's actual dynamics."""
        out = []
        # O2 control: dissolved pO2 vs O2 enrichment (gas mix) — the controller chasing demand.
        f = self._twin_time_figure(
            "pO2 [%]", "Gas mix [-]", runs, max_points,
            title="O2 control: pO2 (solid) vs O2 enrichment / Gas mix (dashed)",
            right_label="Gas mix [%]",
        )
        if f:
            out.append(f)
        # Oxygen limitation: pO2 collapses at high OUR.
        f = self._scatter_figure(
            "OUR [mmol/L/h]", "pO2 [%]", runs, max_points,
            title="Oxygen limitation: pO2 vs OUR",
        )
        if f:
            out.append(f)
        # Growth curve (total biomass).
        f = self._time_series_figure("Biomass total [g]", runs, max_points)
        if f:
            out.append(f)
        # Specific growth rate.
        f = self._time_series_figure("mu [1/h]", runs, max_points)
        if f:
            out.append(f)
        # Carbon-balance closure (process consistency).
        f = self._time_series_figure("C recovered [%]", runs, max_points)
        if f:
            out.append(f)
        # Product formation vs biomass (yield).
        f = self._scatter_figure(
            "Biomass [g/L]", "Protein total calculated [g/L]", runs, max_points,
            title="Product formation: Protein total vs Biomass",
        )
        if f:
            out.append(f)
        return out
