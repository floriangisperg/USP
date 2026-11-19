"""Consolidate multiple runs' ``results/processed_data.csv`` into combined outputs.

Embeds what used to live as one ``consolidate_results.py`` copy per example
campaign. Reads every ``<campaign>/<run_id>/results/processed_data.csv``,
tags rows with a ``run_name`` column, recomputes protein totals/fractions/
yields consistently, builds a run summary, and writes:

- ``<prefix>_consolidated_results.xlsx`` — All_Time_Points | Run_Summary |
  one sheet per run
- ``<prefix>_consolidated_results_sparse.xlsx`` — Sample_Points | Summary
- ``<prefix>_consolidated_dashboard.html`` — interactive run comparison
  (bokeh; requires the ``dashboard`` extra, skipped gracefully otherwise)

Usage::

    from upstream_analysis import consolidate_runs, ConsolidationConfig

    consolidate_runs("examples/hgh", ConsolidationConfig(output_prefix="hGH"))

Campaign specifics go into the config: ``conditions_parser`` receives each
run directory and returns extra summary entries (e.g. nominal setpoints
parsed from the run config), ``run_columns`` controls the per-run sheet
column order, ``filter_to_feed_phase`` trims the batch pre-phase from the
all-time-points sheet (pre-feed sample points are kept).
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .core.validation import coerce_numeric_columns
from .visualization.dashboard import InteractiveDashboard

logger = logging.getLogger(__name__)

DEFAULT_RUN_COLUMNS = [
    "Process time [h]",
    "Temperature [°C]",
    "Is induced [-]",
    "Induction time [h]",
    "Biomass [g/L]",
    "Glucose [g/L]",
    "Acetate [g/L]",
    "Protein [g/L]",
    "Protein soluble [g/L]",
    "Protein IB [g/L]",
    "Protein total calculated [g/L]",
    "Soluble fraction [%]",
    "IB fraction [%]",
    "Protein soluble yield [g/g]",
    "Protein IB yield [g/g]",
    "Protein total yield [g/g]",
    "Volume [mL]",
    "Volume corrected [mL]",
    "Feed [mL]",
    "Feed [g]",
    "Air flow [L/min]",
    "Total flow [L/min]",
    "O2 flow [L/min]",
    "Offgas CO2 [%]",
    "Offgas O2 [%]",
    "Gas mix [-]",
    "CER [mmol/L/h]",
    "OUR [mmol/L/h]",
    "RQ [-]",
    "pH [-]",
    "pO2 [%]",
    "is_sample",
]


@dataclass
class ConsolidationConfig:
    """Campaign-specific consolidation settings."""

    output_prefix: str = "consolidated"
    run_columns: list[str] = field(default_factory=lambda: list(DEFAULT_RUN_COLUMNS))
    conditions_parser: Callable[[Path], dict] | None = None
    filter_to_feed_phase: bool = True
    dashboard_filename: str | None = None


def find_column(df: pd.DataFrame, target_col: str) -> str | None:
    """Find a column, tolerating special-character encoding (e.g. the degree sign).

    Exact match first, then a fallback on the base name before ``' ['``.
    """
    if target_col in df.columns:
        return target_col
    base_name = target_col.split(" [")[0].lower()
    for col in df.columns:
        if col.split(" [")[0].lower() == base_name:
            logger.info(
                "Column encoding match: using '%s' instead of '%s'", col, target_col
            )
            return col
    return None


def improve_protein_concentrations(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute protein total (= soluble + IB), fractions, and yields consistently."""
    df_improved = df.copy()
    if (
        "Protein soluble [g/L]" in df_improved.columns
        and "Protein IB [g/L]" in df_improved.columns
    ):
        df_improved["Protein total calculated [g/L]"] = (
            df_improved["Protein soluble [g/L]"] + df_improved["Protein IB [g/L]"]
        )
        mask_total = df_improved["Protein total calculated [g/L]"] > 0
        df_improved.loc[mask_total, "Soluble fraction [%]"] = (
            df_improved.loc[mask_total, "Protein soluble [g/L]"]
            / df_improved.loc[mask_total, "Protein total calculated [g/L]"]
            * 100
        )
        df_improved.loc[mask_total, "IB fraction [%]"] = (
            df_improved.loc[mask_total, "Protein IB [g/L]"]
            / df_improved.loc[mask_total, "Protein total calculated [g/L]"]
            * 100
        )
        if "Biomass [g/L]" in df_improved.columns:
            mask_x = df_improved["Biomass [g/L]"] > 0
            for num, label in [
                ("Protein soluble [g/L]", "Protein soluble yield [g/g]"),
                ("Protein total calculated [g/L]", "Protein total yield [g/g]"),
                ("Protein IB [g/L]", "Protein IB yield [g/g]"),
            ]:
                df_improved.loc[mask_x, label] = (
                    df_improved.loc[mask_x, num]
                    / df_improved.loc[mask_x, "Biomass [g/L]"]
                )
    return df_improved


def default_conditions_parser(run_dir: Path) -> dict:
    """Pull induction_summary hints from the run config (qs / T setpoints)."""
    qs = t_set = None
    cfg_path = run_dir / "run_config.yaml"
    if cfg_path.exists():
        try:
            txt = cfg_path.read_text(encoding="utf-8")
            m_qs = re.search(
                r"induction_summary:.*?qs\s*=\s*([0-9.]+)", txt, re.IGNORECASE
            )
            m_t = re.search(
                r"induction_summary:.*?T\s*=\s*([0-9.]+)", txt, re.IGNORECASE
            )
            if m_qs:
                qs = float(m_qs.group(1))
            if m_t:
                t_set = float(m_t.group(1))
        except OSError as e:
            logger.warning("Could not parse run_config for %s: %s", run_dir.name, e)
    return {"qs_initial [g/g/h]": qs, "T_setpoint [°C]": t_set}


def _feed_phase_onward(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to the feed phase onward, keeping any pre-feed sample points."""
    if "Feed [mL]" not in df.columns or "Process time [h]" not in df.columns:
        return df
    feed_start_mask = df["Feed [mL]"].fillna(0) > 0
    if not feed_start_mask.any():
        logger.warning("no feed detected, keeping all data")
        return df
    first_feed_idx = feed_start_mask.idxmax()
    pre_feed_sample_idx = df.index[
        (df.index < first_feed_idx) & (df["is_sample"].astype(str) == "True")
    ].tolist()
    trimmed = df.loc[first_feed_idx:].copy()
    if pre_feed_sample_idx:
        trimmed = pd.concat([df.loc[pre_feed_sample_idx].copy(), trimmed]).sort_index()
    return trimmed


def _run_summary(df: pd.DataFrame, conditions: dict) -> dict:
    summary: dict = {"run_name": str(df["run_name"].iloc[0])}
    summary.update(conditions)

    ind_col = find_column(df, "Is induced [-]")
    t_col = find_column(df, "Temperature [°C]")
    if ind_col and t_col:
        ind_mask = pd.to_numeric(df[ind_col], errors="coerce").eq(1)
        t_ind = pd.to_numeric(df.loc[ind_mask, t_col], errors="coerce").median()
        if pd.notna(t_ind):
            summary["T_induction [°C]"] = round(float(t_ind), 1)

    if "Process time [h]" in df.columns:
        summary["total_time_h"] = df["Process time [h]"].max()
    dfs = df.sort_values("Process time [h]")
    td = dfs["Process time [h]"].diff().fillna(0)
    if "Feed [mL]" in dfs.columns:
        summary["Feed time [h]"] = td[dfs["Feed [mL]"].fillna(0) > 0].sum()

    for col in [
        "Biomass [g/L]",
        "Protein soluble [g/L]",
        "Protein IB [g/L]",
        "Protein total calculated [g/L]",
    ]:
        if col in df.columns:
            summary[f"{col}_max"] = df[col].max()
            # "final" = last measured value (offline columns only exist at
            # sample rows, so iloc[-1] would be NaN)
            vals = df[col].dropna()
            summary[f"{col}_final"] = vals.iloc[-1] if not vals.empty else float("nan")

    if "Protein total yield [g/g]" in df.columns:
        fy = df["Protein total yield [g/g]"].dropna()
        if not fy.empty:
            summary["final_yield"] = fy.iloc[-1]

    cm = find_column(df, "C recovered (measured)")
    if cm:
        mvals = pd.to_numeric(df[cm], errors="coerce").dropna()
        if not mvals.empty:
            summary["C recovered (measured) [%]_mean"] = mvals.mean()
    return summary


def _read_processed(path: Path) -> pd.DataFrame:
    """Read a processed_data.csv robustly.

    Older exports may contain mixed decimal styles (the pipeline writes
    ``decimal=','``, legacy scripts wrote ``'.'``, and pre-fix pipelines
    leaked string-float columns). ``coerce_numeric_columns`` normalizes both
    after a plain ``sep=';'`` read.
    """
    return coerce_numeric_columns(pd.read_csv(path, sep=";"))


def consolidate_runs(
    campaign_dir: Path | str, config: ConsolidationConfig | None = None
) -> dict[str, pd.DataFrame]:
    """Consolidate all runs with results under ``campaign_dir``.

    Returns the dict of per-run consolidated frames that was written out.
    """
    cfg = config or ConsolidationConfig()
    campaign = Path(campaign_dir)

    run_dirs = sorted(
        d
        for d in campaign.iterdir()
        if d.is_dir() and (d / "results" / "processed_data.csv").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"no runs with results/processed_data.csv under {campaign} — "
            "run the pipeline on each run folder first"
        )
    logger.info(
        "Found %d runs to consolidate: %s", len(run_dirs), [d.name for d in run_dirs]
    )

    all_data: dict[str, pd.DataFrame] = {}
    all_samples: dict[str, pd.DataFrame] = {}
    full_runs: dict[str, pd.DataFrame] = {}
    for run_dir in run_dirs:
        run_name = run_dir.name
        processed_file = run_dir / "results" / "processed_data.csv"
        try:
            df = _read_processed(processed_file)
        except OSError as e:
            logger.error("Error loading %s: %s", processed_file, e)
            continue

        df = improve_protein_concentrations(df)
        sample_df = (
            df[df["is_sample"]].copy() if "is_sample" in df.columns else pd.DataFrame()
        )
        df["run_name"] = run_name
        full_runs[run_name] = df.copy()
        logger.info(
            "Loaded %d rows, %d sample points from %s",
            len(df),
            len(sample_df),
            run_name,
        )
        all_data[run_name] = _feed_phase_onward(df) if cfg.filter_to_feed_phase else df
        all_samples[run_name] = sample_df

    if not all_data:
        raise ValueError("no processed_data.csv could be loaded")

    conditions_parser = cfg.conditions_parser or default_conditions_parser
    summary_df = pd.DataFrame(
        [
            _run_summary(df, conditions_parser(campaign / name))
            for name, df in all_data.items()
        ]
    )

    # --- main workbook -----------------------------------------------------
    output_file = campaign / f"{cfg.output_prefix}_consolidated_results.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        all_timepoints = pd.concat(all_data.values(), ignore_index=True)
        all_timepoints = all_timepoints.sort_values(["run_name", "Process time [h]"])
        all_timepoints["Feed time [h]"] = 0.0
        for run_name in all_timepoints["run_name"].unique():
            run_mask = all_timepoints["run_name"] == run_name
            run_df = all_timepoints[run_mask].sort_values("Process time [h]")
            time_diffs = run_df["Process time [h]"].diff().fillna(0)
            feeding = run_df["Feed [mL]"].fillna(0) > 0
            all_timepoints.loc[run_mask, "Feed time [h]"] = (
                (time_diffs * feeding).cumsum().values
            )

        key_columns = (
            ["run_name"]
            + [c for c in cfg.run_columns if c != "is_sample"]
            + ["Feed time [h]"]
        )
        available = list(
            dict.fromkeys(
                c
                for c in (find_column(all_timepoints, col) for col in key_columns)
                if c
            )
        )
        all_timepoints[available].to_excel(
            writer, sheet_name="All_Time_Points", index=False
        )

        summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)

        for run_name, df in all_data.items():
            sheet_name = run_name.replace("_", "-")[:31]
            cols = [c for c in (find_column(df, c) for c in cfg.run_columns) if c]
            df[cols].to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Created consolidated workbook: %s", output_file)

    # --- sparse workbook -----------------------------------------------------
    sparse_file = campaign / f"{cfg.output_prefix}_consolidated_results_sparse.xlsx"
    sample_frames = [
        sdf.copy().assign(run_name=run_name)
        for run_name, sdf in all_samples.items()
        if sdf is not None and len(sdf) > 0
    ]
    if sample_frames:
        all_samples_df = pd.concat(sample_frames, ignore_index=True).sort_values(
            ["run_name", "Process time [h]"]
        )
        sample_cols = [
            c
            for c in (
                find_column(all_samples_df, c) for c in ["run_name"] + cfg.run_columns
            )
            if c
        ]
        sample_out = all_samples_df[list(dict.fromkeys(sample_cols))]
    else:
        sample_out = pd.DataFrame()

    n_samples = {rn: len(sdf) for rn, sdf in all_samples.items() if sdf is not None}
    sparse_summary = summary_df.copy()
    sparse_summary["num_samples"] = sparse_summary["run_name"].map(n_samples)
    with pd.ExcelWriter(sparse_file, engine="openpyxl") as writer:
        sample_out.to_excel(writer, sheet_name="Sample_Points", index=False)
        sparse_summary.to_excel(writer, sheet_name="Summary", index=False)
    logger.info("Saved sparse file: %s", sparse_file)

    # --- dashboard -----------------------------------------------------------
    dashboard_name = (
        cfg.dashboard_filename or f"{cfg.output_prefix}_consolidated_dashboard.html"
    )
    try:
        InteractiveDashboard(campaign).plot_comparison(
            full_runs, output_filename=dashboard_name
        )
    except ImportError as e:
        logger.warning("Skipped consolidated dashboard (bokeh not installed): %s", e)

    logger.info("Consolidation done: %d runs", len(all_data))
    return all_data


def main() -> int:
    """CLI: ``python -m upstream_analysis.consolidation <campaign_dir> [--prefix P]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "campaign_dir", help="campaign folder containing <run_id>/results/"
    )
    parser.add_argument(
        "--prefix", default="consolidated", help="output filename prefix"
    )
    args = parser.parse_args()
    consolidate_runs(args.campaign_dir, ConsolidationConfig(output_prefix=args.prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
