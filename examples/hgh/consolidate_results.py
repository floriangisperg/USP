# examples/hgh/consolidate_results.py
"""Consolidate both hGH runs' processed_data.csv into combined outputs.

Mirrors examples/Xiso/consolidate_results.py. Reads each run's
``results/processed_data.csv`` (written by analyze_hgh_run.py), tags it with a
``run_name`` column, filters to the feed phase onward (keeping any pre-feed
sample points), and writes:

  - hGH_consolidated_results.xlsx      All_Time_Points | Run_Summary | one sheet per run
  - hGH_consolidated_results_sparse.xlsx  Sample_Points | Summary
  - hGH_consolidated_dashboard.html        interactive run comparison (bokeh)

Run it after analyzing the individual runs::

    python analyze_hgh_run.py R5_lowlow
    python analyze_hgh_run.py R6_highhigh
    python consolidate_results.py
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd

# Make the library importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from upstream_analysis.visualization.dashboard import InteractiveDashboard  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_column(df, target_col):
    """Find a column, tolerating special-character encoding (e.g. the degree sign).

    Exact match first, then a fallback on the base name before ' ['.
    """
    if target_col in df.columns:
        return target_col
    base_name = target_col.split(" [")[0].lower()
    for col in df.columns:
        if col.split(" [")[0].lower() == base_name:
            logger.info(
                f"Column encoding match: using '{col}' instead of '{target_col}'"
            )
            return col
    return None


def improve_protein_concentrations(df):
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


def _parse_conditions(run_dir):
    """Pull nominal qs and T setpoint from the run_config induction_summary."""
    qs, t_set = None, None
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
        except Exception as e:
            logger.warning(f"Could not parse run_config for {run_dir.name}: {e}")
    return {"qs_initial [g/g/h]": qs, "T_setpoint [°C]": t_set}


RUN_COLS = [
    "Process time [h]",
    "Temperature [°C]",
    "Is induced [-]",
    "Induction time [h]",
    "Biomass [g/L]",
    "Glucose [g/L]",
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


def consolidate_hgh_results():
    """Consolidate all hGH run results into combined Excel files + dashboard."""
    script_location = Path(__file__).resolve().parent

    run_dirs = sorted(
        d
        for d in script_location.iterdir()
        if d.is_dir() and (d / "results" / "processed_data.csv").exists()
    )
    if not run_dirs:
        logger.error(
            "No runs with results/processed_data.csv found. "
            "Run analyze_hgh_run.py first."
        )
        return

    logger.info(
        f"Found {len(run_dirs)} runs to consolidate: {[d.name for d in run_dirs]}"
    )

    all_data, all_samples, full_runs = {}, {}, {}
    for run_dir in run_dirs:
        run_name = run_dir.name
        processed_file = run_dir / "results" / "processed_data.csv"
        try:
            logger.info(f"Loading processed data from: {processed_file}")
            df = pd.read_csv(
                processed_file, sep=";"
            )  # runner writes sep=';' decimal='.'
        except Exception as e:
            logger.error(f"Error loading {processed_file}: {e}")
            continue

        df = improve_protein_concentrations(df)
        sample_df = (
            df[df["is_sample"]].copy() if "is_sample" in df.columns else pd.DataFrame()
        )
        df["run_name"] = run_name
        full_runs[run_name] = df.copy()  # full timeline for the dashboard
        logger.info(
            f"Loaded {len(df)} rows, {len(sample_df)} sample points from {run_name}"
        )

        # Filter to feed phase onward, but keep any pre-feed sample points
        all_df = df.copy()
        if "Feed [mL]" in all_df.columns and "Process time [h]" in all_df.columns:
            feed_start_mask = all_df["Feed [mL]"].fillna(0) > 0
            if feed_start_mask.any():
                first_feed_idx = feed_start_mask.idxmax()
                pre_feed_sample_idx = all_df.index[
                    (all_df.index < first_feed_idx)
                    & (all_df["is_sample"].astype(str) == "True")
                ].tolist()
                all_df = all_df.loc[first_feed_idx:].copy()
                if pre_feed_sample_idx:
                    all_df = pd.concat(
                        [df.loc[pre_feed_sample_idx].copy(), all_df]
                    ).sort_index()
                logger.info(f"{run_name}: feed phase onward, {len(all_df)} rows")
            else:
                logger.warning(f"{run_name}: no feed detected, keeping all data")

        all_data[run_name] = all_df
        all_samples[run_name] = sample_df

    if not all_data:
        logger.error("No data files loaded")
        return

    # ------------------------------------------------------------------
    # Run summary (nominal qs/T + measured endpoints)
    # ------------------------------------------------------------------
    summary_data = []
    for run_name, df in all_data.items():
        if df.empty:
            continue
        summary = {"run_name": run_name}
        summary.update(_parse_conditions(script_location / run_name))

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
                summary[f"{col}_final"] = (
                    vals.iloc[-1] if not vals.empty else float("nan")
                )

        if "Protein total yield [g/g]" in df.columns:
            fy = df["Protein total yield [g/g]"].dropna()
            if not fy.empty:
                summary["final_yield"] = fy.iloc[-1]

        cm = find_column(df, "C recovered (measured)")
        if cm:
            mvals = pd.to_numeric(df[cm], errors="coerce").dropna()
            if not mvals.empty:
                summary["C recovered (measured) [%]_mean"] = mvals.mean()

        summary_data.append(summary)
    summary_df = pd.DataFrame(summary_data)

    # ------------------------------------------------------------------
    # Main consolidated workbook
    # ------------------------------------------------------------------
    output_file = script_location / "hGH_consolidated_results.xlsx"
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
            ["run_name"] + [c for c in RUN_COLS if c != "is_sample"] + ["Feed time [h]"]
        )
        available_cols = [
            c
            for c in (find_column(all_timepoints, col) for col in key_columns)
            if c is not None
        ]
        available_cols = list(dict.fromkeys(available_cols))  # dedupe, keep order
        all_timepoints[available_cols].to_excel(
            writer, sheet_name="All_Time_Points", index=False
        )
        logger.info(f"Saved {len(all_timepoints)} time points to All_Time_Points")

        summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)
        logger.info(f"Saved summary for {len(summary_df)} runs")

        for run_name, df in all_data.items():
            sheet_name = run_name.replace("_", "-")[:31]
            cols = [c for c in (find_column(df, c) for c in RUN_COLS) if c is not None]
            df[cols].to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Saved {len(df)} rows for {run_name} -> sheet '{sheet_name}'")

    logger.info(f"Created consolidated workbook: {output_file}")

    # ------------------------------------------------------------------
    # Sparse workbook (sample points + endpoint summary)
    # ------------------------------------------------------------------
    sparse_file = script_location / "hGH_consolidated_results_sparse.xlsx"
    sample_frames = [
        sdf.copy().assign(run_name=run_name)
        for run_name, sdf in all_samples.items()
        if sdf is not None and len(sdf) > 0
    ]
    if sample_frames:
        all_samples_df = pd.concat(sample_frames, ignore_index=True).sort_values(
            ["run_name", "Process time [h]"]
        )
        sample_key = ["run_name"] + RUN_COLS
        sp_cols = [
            c
            for c in (find_column(all_samples_df, c) for c in sample_key)
            if c is not None
        ]
        sample_out = all_samples_df[list(dict.fromkeys(sp_cols))]
    else:
        sample_out = pd.DataFrame()

    n_samples = {rn: len(sdf) for rn, sdf in all_samples.items() if sdf is not None}
    sparse_summary = summary_df.copy()
    sparse_summary["num_samples"] = sparse_summary["run_name"].map(n_samples)
    sparse_cols_pref = [
        "run_name",
        "T_induction [°C]",
        "qs_initial [g/g/h]",
        "T_setpoint [°C]",
        "num_samples",
        "total_time_h",
        "Feed time [h]",
        "Biomass [g/L]_final",
        "Protein soluble [g/L]_final",
        "Protein IB [g/L]_final",
        "Protein total calculated [g/L]_final",
        "final_yield",
    ]
    sparse_cols = [c for c in sparse_cols_pref if c in sparse_summary.columns]

    with pd.ExcelWriter(sparse_file, engine="openpyxl") as writer:
        sample_out.to_excel(writer, sheet_name="Sample_Points", index=False)
        sparse_summary[sparse_cols].to_excel(writer, sheet_name="Summary", index=False)
    logger.info(
        f"Saved sparse file: {len(sample_out)} sample points, "
        f"{len(sparse_summary)} runs -> {sparse_file}"
    )

    # ------------------------------------------------------------------
    # Interactive comparison dashboard (runs overlaid, category tabs)
    # ------------------------------------------------------------------
    try:
        InteractiveDashboard(script_location).plot_comparison(
            full_runs, output_filename="hGH_consolidated_dashboard.html"
        )
    except ImportError as e:
        logger.warning(f"Skipped consolidated dashboard (bokeh not installed): {e}")

    logger.info("\nhGH Consolidated Results Summary:")
    logger.info(f"- Runs processed: {len(all_data)}")
    logger.info(f"- Workbook: {output_file}")
    logger.info(f"- Sparse:   {sparse_file}")


if __name__ == "__main__":
    consolidate_hgh_results()
