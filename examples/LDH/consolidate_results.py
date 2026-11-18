import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the library importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from upstream_analysis.visualization.dashboard import InteractiveDashboard  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Columns kept in the consolidated processed-modeling file (mechanistic-model
# relevant subset). The per-run processed_data.xlsx still holds every column;
# edit this list to add/remove columns from the consolidated export.
MODEL_COLUMNS = [
    # identity / time / phase
    "run_name",
    "Process time [h]",
    "Process phase [-]",
    "Is induced [-]",
    "Induction time [h]",
    "is_sample",
    # environment
    "Temperature [°C]",
    "pH [-]",
    "pO2 [%]",
    "Stirrer speed [1/min]",
    # gas flows + offgas (model inputs)
    "Air flow [L/min]",
    "Total flow [L/min]",
    "O2 flow [L/min]",
    "Gas mix [-]",
    "Inlet O2 [%]",
    "Inlet CO2 [%]",
    "Offgas O2 [%]",
    "Offgas CO2 [%]",
    # offgas rates (derived)
    "CER [mmol/L/h]",
    "OUR [mmol/L/h]",
    "RQ [-]",
    "qO2 [mmol/g/h]",
    "qCO2 [mmol/g/h]",
    # feed / volume
    "Feed [g]",
    "Feed [mL]",
    "Volume [mL]",
    "Volume corrected [mL]",
    "Base volume added [mL]",
    "Feed rate exp [%]",
    "Feed rate ind [%]",
    # biomass / substrate
    "Biomass [g/L]",
    "Biomass total [g]",
    "Glucose [g/L]",
    "OD600 [-]",
    "Glucose consumed [g]",
    "Glucose consumed rate [g/h]",
    # kinetics
    "mu [1/h]",
    "qS glucose [g/g/h]",
    "qP total [g/g/h]",
    # protein
    "Protein soluble [g/L]",
    "Protein IB [g/L]",
    "Protein total calculated [g/L]",
    "Protein total yield [g/g]",
    # carbon balance (summary)
    "C recovered [%]",
    "C balance closure [%]",
    "C recovered (measured) [%]",
    "C to CO2 [% of fed]",
    "C to biomass [% of fed]",
    "C to product [% of fed]",
]


def find_column(df, target_col):
    """Find a column in dataframe, handling encoding issues with special characters.

    Tries exact match first, then falls back to partial matching for columns
    that might have encoding issues with special characters (like degree symbol °).
    """
    # Try exact match first
    if target_col in df.columns:
        return target_col

    # Fall back to partial match for columns with special characters
    # Extract the base name before the bracket for fuzzy matching
    base_name = target_col.split(" [")[0].lower()

    for col in df.columns:
        col_base = col.split(" [")[0].lower()
        if col_base == base_name:
            logger.info(
                f"Column encoding match: using '{col}' instead of '{target_col}'"
            )
            return col

    # Not found
    return None


def improve_protein_concentrations(df):
    """Improve soluble and IB protein concentration calculations."""

    # Create a copy to avoid modifying the original
    df_improved = df.copy()

    # Ensure protein columns exist
    if (
        "Protein soluble [g/L]" in df_improved.columns
        and "Protein IB [g/L]" in df_improved.columns
    ):
        # Recalculate protein total as sum of soluble + IB
        df_improved["Protein total calculated [g/L]"] = (
            df_improved["Protein soluble [g/L]"] + df_improved["Protein IB [g/L]"]
        )

        # Recalculate fractions to avoid division by zero
        mask_total_gt_zero = df_improved["Protein total calculated [g/L]"] > 0
        df_improved.loc[mask_total_gt_zero, "Soluble fraction [%]"] = (
            df_improved.loc[mask_total_gt_zero, "Protein soluble [g/L]"]
            / df_improved.loc[mask_total_gt_zero, "Protein total calculated [g/L]"]
            * 100
        )

        df_improved.loc[mask_total_gt_zero, "IB fraction [%]"] = (
            df_improved.loc[mask_total_gt_zero, "Protein IB [g/L]"]
            / df_improved.loc[mask_total_gt_zero, "Protein total calculated [g/L]"]
            * 100
        )

        # Recalculate yields if biomass is available
        if "Biomass [g/L]" in df_improved.columns:
            mask_biomass_gt_zero = df_improved["Biomass [g/L]"] > 0
            df_improved.loc[mask_biomass_gt_zero, "Protein soluble yield [g/g]"] = (
                df_improved.loc[mask_biomass_gt_zero, "Protein soluble [g/L]"]
                / df_improved.loc[mask_biomass_gt_zero, "Biomass [g/L]"]
            )

            df_improved.loc[mask_biomass_gt_zero, "Protein total yield [g/g]"] = (
                df_improved.loc[mask_biomass_gt_zero, "Protein total calculated [g/L]"]
                / df_improved.loc[mask_biomass_gt_zero, "Biomass [g/L]"]
            )

            df_improved.loc[mask_biomass_gt_zero, "Protein IB yield [g/g]"] = (
                df_improved.loc[mask_biomass_gt_zero, "Protein IB [g/L]"]
                / df_improved.loc[mask_biomass_gt_zero, "Biomass [g/L]"]
            )

    return df_improved


def consolidate_ldh_results():
    """Consolidate all LDH experiment results, generating outputs similar to DOE example."""

    # Get the script location (examples/LDH/)
    script_location = Path(__file__).resolve().parent

    # Find all ferm_* directories
    run_dirs = [
        d
        for d in script_location.iterdir()
        if d.is_dir()
        and (
            d.name.startswith("ferm_")
            or (d.name.startswith("F") and len(d.name) > 1 and d.name[1].isdigit())
        )
    ]
    run_dirs.sort()

    if not run_dirs:
        logger.error("No ferm_* directories found")
        return

    logger.info(f"Found {len(run_dirs)} runs to consolidate: {run_dirs}")

    # Initialize dictionary to store dataframes
    all_data = {}
    all_time_data = {}
    all_samples = {}  # sparse: per-run sample-point rows
    full_runs = {}  # full-timeline df per run, for the interactive dashboard

    # Process each run
    for run_dir in run_dirs:
        run_name = run_dir.name
        results_dir = run_dir / "results"

        if not results_dir.exists():
            logger.warning(f"No results directory found for {run_name}, skipping")
            continue

        # Load processed_data.xlsx (the primary output from analyze_ldh_run.py)
        processed_file = results_dir / "processed_data.xlsx"

        df = None
        sample_df = None

        if processed_file.exists():
            try:
                logger.info(f"Loading processed data from: {processed_file}")
                df = pd.read_excel(processed_file, sheet_name="Processed_Data")

                # Improve protein concentration calculations if needed
                df = improve_protein_concentrations(df)

                sample_df = (
                    df[df["is_sample"]].copy()
                    if "is_sample" in df.columns
                    else pd.DataFrame()
                )
                logger.info(f"Loaded {len(df)} rows, {len(sample_df)} sample points")
            except Exception as e:
                logger.error(f"Error loading {processed_file}: {e}")
                df = None
        else:
            logger.warning(
                f"No processed_data.xlsx found in {results_dir}. Run analyze_ldh_run.py first."
            )

        # Store data for consolidation if successfully loaded
        if df is not None and not df.empty:
            # Add run identifier for consolidated analysis
            df["run_name"] = run_name
            full_runs[run_name] = df.copy()  # full timeline for the dashboard
            all_df = df.copy()

            # Filter to keep only data from feed phase onwards (exclude batch phase)
            # BUT keep all sample points even if they're before feed phase
            if "Feed [mL]" in all_df.columns and "Process time [h]" in all_df.columns:
                # Find first time feed > 0
                feed_start_mask = all_df["Feed [mL]"].fillna(0) > 0
                if feed_start_mask.any():
                    first_feed_idx = feed_start_mask.idxmax()
                    feed_start_time = all_df.loc[first_feed_idx, "Process time [h]"]

                    # Get indices of all sample points before feed phase
                    if "is_sample" in all_df.columns:
                        pre_feed_sample_mask = (all_df.index < first_feed_idx) & (
                            all_df["is_sample"] == True
                        )
                        pre_feed_sample_indices = all_df.index[
                            pre_feed_sample_mask
                        ].tolist()
                    else:
                        pre_feed_sample_indices = []

                    # Keep all data from first feed onwards
                    all_df = all_df.loc[first_feed_idx:].copy()

                    # Also include any sample points that were before feed phase
                    if pre_feed_sample_indices:
                        pre_feed_samples = df.loc[pre_feed_sample_indices].copy()
                        all_df = pd.concat(
                            [pre_feed_samples, all_df], ignore_index=False
                        )
                        all_df = all_df.sort_index()
                        logger.info(
                            f"{run_name}: Filtered to feed phase onwards (t >= {feed_start_time:.2f}h), {len(all_df)} rows (+{len(pre_feed_sample_indices)} pre-feed samples)"
                        )
                    else:
                        logger.info(
                            f"{run_name}: Filtered to feed phase onwards (t >= {feed_start_time:.2f}h), {len(all_df)} rows"
                        )
                else:
                    logger.warning(f"{run_name}: No feed detected, keeping all data")

            all_data[run_name] = all_df
            all_time_data[run_name] = all_df
            all_samples[run_name] = sample_df  # may be None/empty
            sample_count = len(sample_df) if sample_df is not None else 0
            logger.info(
                f"Processed {len(all_df)} rows from {run_name} ({sample_count} sample points)"
            )
        else:
            logger.warning(f"No data loaded for {run_name}")

    if not all_data:
        logger.error("No data files loaded")
        return

    # Also create the consolidated Excel file (keeping the original functionality)
    output_file = script_location / "LDH_consolidated_results.xlsx"

    # Create Excel writer
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        # 1. Consolidated all time points sheet (like DOE)
        logger.info("Creating consolidated all time points sheet...")
        all_timepoints = pd.concat(all_time_data.values(), ignore_index=True)

        # Sort by run_name and Process time [h]
        all_timepoints = all_timepoints.sort_values(["run_name", "Process time [h]"])

        # Calculate feed time per run for consolidated view (summary only)
        all_timepoints["Feed time [h]"] = 0.0
        for run_name in all_timepoints["run_name"].unique():
            run_mask = all_timepoints["run_name"] == run_name
            run_df = all_timepoints[run_mask].copy()
            run_df = run_df.sort_values("Process time [h]")

            # Calculate time deltas
            time_diffs = run_df["Process time [h]"].diff().fillna(0)
            # Feed time accumulates when Feed [mL] > 0
            feeding_mask = run_df["Feed [mL]"].fillna(0) > 0
            run_df["is_feeding"] = feeding_mask
            run_df["time_delta"] = time_diffs
            run_df["feed_delta"] = run_df["time_delta"] * run_df["is_feeding"]
            run_df["cumulative_feed_time"] = run_df["feed_delta"].cumsum()
            all_timepoints.loc[run_mask, "Feed time [h]"] = run_df[
                "cumulative_feed_time"
            ].values

        # Select key columns for the consolidated view (updated based on requirements)
        key_columns = [
            "run_name",
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
            "Air flow [L/min]",
            "Total flow [L/min]",
            "O2 flow [L/min]",
            "Offgas CO2 [%]",
            "Offgas O2 [%]",
            "Gas mix [-]",
            "Base pump [%]",
            "Base pump duration [-]",
            "pH [-]",
            "pO2 [%]",
            "Feed time [h]",
        ]

        # Filter to available columns (with fuzzy matching for encoding issues)
        available_cols = [find_column(all_timepoints, col) for col in key_columns]
        available_cols = [col for col in available_cols if col is not None]
        consolidated_data = all_timepoints[available_cols]

        consolidated_data.to_excel(writer, sheet_name="All_Time_Points", index=False)
        logger.info(
            f"Saved {len(consolidated_data)} time points to All_Time_Points sheet"
        )

        # 2. Summary statistics per run
        logger.info("Creating run summary sheet...")

        # Parse nominal setpoints (qs, T) from each run_config.yaml induction_summary
        conditions = {}
        for run_dir in run_dirs:
            qs = None
            t_set = None
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
                    logger.warning(
                        f"Could not parse run_config for {run_dir.name}: {e}"
                    )
            conditions[run_dir.name] = {
                "qs_initial [g/g/h]": qs,
                "T_setpoint [°C]": t_set,
            }

        summary_data = []
        for run_name, df in all_time_data.items():
            if df.empty:
                continue

            summary = {"run_name": run_name}
            summary.update(conditions.get(run_name, {}))

            # Measured induction temperature (median T while induced)
            ind_col = find_column(df, "Is induced [-]")
            t_col = find_column(df, "Temperature [°C]")
            if ind_col and t_col:
                ind_mask = pd.to_numeric(df[ind_col], errors="coerce").eq(1)
                t_ind = pd.to_numeric(df.loc[ind_mask, t_col], errors="coerce").median()
                if pd.notna(t_ind):
                    summary["T_induction [°C]"] = round(float(t_ind), 1)

            # Get key metrics from all time points
            if "Process time [h]" in df.columns:
                summary["total_time_h"] = df["Process time [h]"].max()

            # Calculate feed time for summary (cumulative hours when Feed [mL] > 0)
            if "Feed [mL]" in df.columns and "Process time [h]" in df.columns:
                df_sorted = df.sort_values("Process time [h]")
                time_diffs = df_sorted["Process time [h]"].diff().fillna(0)
                feeding_mask = df_sorted["Feed [mL]"].fillna(0) > 0
                feed_time_total = time_diffs[feeding_mask].sum()
                summary["Feed time [h]"] = feed_time_total

            for col in [
                "Biomass [g/L]",
                "Protein soluble [g/L]",
                "Protein IB [g/L]",
                "Protein total calculated [g/L]",
            ]:
                if col in df.columns:
                    summary[f"{col}_max"] = df[col].max()
                    # Get final value from last row
                    summary[f"{col}_final"] = df[col].iloc[-1]

            if "Protein total yield [g/g]" in df.columns:
                # Get final yield from last non-NaN value
                final_yield = df["Protein total yield [g/g]"].dropna()
                if not final_yield.empty:
                    summary["final_yield"] = final_yield.iloc[-1]

            # Rigorous (assumption-free) carbon closure: mean over measured timestamps
            cm = find_column(df, "C recovered (measured)")
            if cm:
                mvals = pd.to_numeric(df[cm], errors="coerce").dropna()
                if not mvals.empty:
                    summary["C recovered (measured) [%]_mean"] = mvals.mean()

            summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)
        logger.info(f"Saved summary for {len(summary_df)} runs")

        # 3. Individual run sheets with full data
        logger.info("Creating individual run sheets...")
        for run_name, df in all_data.items():
            # Clean sheet name (Excel sheet names have restrictions)
            sheet_name = run_name.replace("_", "-")[:31]  # Excel limit is 31 chars
            try:
                # Select columns for individual run sheets (no feed time here - it's a summary metric)
                run_cols = [
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
                    "Air flow [L/min]",
                    "Total flow [L/min]",
                    "O2 flow [L/min]",
                    "Offgas CO2 [%]",
                    "Offgas O2 [%]",
                    "Gas mix [-]",
                    "Base pump [%]",
                    "Base pump duration [-]",
                    "pH [-]",
                    "pO2 [%]",
                    "is_sample",
                ]
                available_run_cols = [find_column(df, col) for col in run_cols]
                available_run_cols = [
                    col for col in available_run_cols if col is not None
                ]

                df[available_run_cols].to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
                logger.info(
                    f"Saved {len(df)} rows for {run_name} to sheet {sheet_name}"
                )
            except Exception as e:
                logger.error(f"Error saving sheet for {run_name}: {e}")

    logger.info(f"Successfully created consolidated Excel file: {output_file}")

    # ============================================================
    # Sparse consolidated file (sample points + endpoint summary)
    # ============================================================
    sparse_file = script_location / "LDH_consolidated_results_sparse.xlsx"
    logger.info(f"Creating sparse consolidated file: {sparse_file}")

    # Collect sample-point rows across all runs
    sample_frames = []
    for run_name, sdf in all_samples.items():
        if sdf is None or len(sdf) == 0:
            continue
        sdf = sdf.copy()
        sdf["run_name"] = run_name
        sample_frames.append(sdf)

    sample_key = [
        "run_name",
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
        "Offgas CO2 [%]",
        "Offgas O2 [%]",
        "pH [-]",
        "pO2 [%]",
    ]
    if sample_frames:
        all_samples_df = pd.concat(sample_frames, ignore_index=True)
        all_samples_df = all_samples_df.sort_values(["run_name", "Process time [h]"])
        sp_cols = [find_column(all_samples_df, c) for c in sample_key]
        sp_cols = [c for c in sp_cols if c is not None]
        sample_out = all_samples_df[sp_cols]
    else:
        sample_out = pd.DataFrame()

    # Endpoint summary per run (focused view) derived from Run_Summary + sample counts
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
    sparse_summary_out = sparse_summary[sparse_cols]

    with pd.ExcelWriter(sparse_file, engine="openpyxl") as writer:
        sample_out.to_excel(writer, sheet_name="Sample_Points", index=False)
        sparse_summary_out.to_excel(writer, sheet_name="Summary", index=False)
    logger.info(
        f"Saved sparse file: {len(sample_out)} sample points, {len(sparse_summary_out)} runs -> {sparse_file}"
    )

    # ------------------------------------------------------------------
    # Interactive comparison dashboard (runs overlaid, category + Functional tabs)
    # ------------------------------------------------------------------
    try:
        InteractiveDashboard(script_location).plot_comparison(
            full_runs, output_filename="LDH_consolidated_dashboard.html", max_points=800
        )
    except ImportError as e:
        logger.warning(f"Skipped consolidated dashboard (bokeh not installed): {e}")

    # ------------------------------------------------------------------
    # All-columns, full-timeline consolidated file (for mechanistic modeling)
    # Every processed column from every run, tagged with run_name, unfiltered.
    # ------------------------------------------------------------------
    processed_all = pd.concat(full_runs.values(), ignore_index=True)
    processed_all = processed_all.sort_values(["run_name", "Process time [h]"])
    keep = [c for c in MODEL_COLUMNS if c in processed_all.columns]
    processed_file = script_location / "LDH_consolidated_results_processed.xlsx"
    with pd.ExcelWriter(processed_file, engine="openpyxl") as writer:
        processed_all[keep].to_excel(writer, sheet_name="All_Processed", index=False)
    missing = [c for c in MODEL_COLUMNS if c not in processed_all.columns]
    logger.info(
        f"Saved processed-modeling file: {processed_file} "
        f"({len(processed_all)} rows, {len(keep)} cols)"
    )
    if missing:
        logger.warning(f"  requested MODEL_COLUMNS not present: {missing}")

    # Print summary
    logger.info(f"\nLDH Consolidated Results Summary:")
    logger.info(f"- Total runs processed: {len(all_data)}")
    logger.info(
        f"- Total time points: {len(consolidated_data)} (like DOE - all time points, not just samples)"
    )
    logger.info(f"- Consolidated Excel file: {output_file}")
    logger.info(
        f"- Individual processed_data.xlsx files are in each run's results directory"
    )
    logger.info(
        f"- Sheets created: All_Time_Points, Run_Summary, {len(all_data)} individual run sheets"
    )
    logger.info(f"- All column names use 'Readable name [unit]' format")


if __name__ == "__main__":
    consolidate_ldh_results()
