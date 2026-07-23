# examples/Xiso/consolidate_results.py
"""
Consolidate every Xiso DoE run's processed_data.csv into one workbook.

Mirrors examples/LDH/consolidate_results.py. Reads each run's
``results/processed_data.csv`` (written by analyze_xiso_run.py), tags it with a
``run_name`` column, filters to the fed-batch phase onward (keeping any pre-feed
sample points), and writes:

  - Xiso_consolidated_results.xlsx        All_Time_Points | Run_Summary | one sheet per run
  - Xiso_consolidated_results_sparse.xlsx Sample_Points  | Summary

Run it after analyzing the individual runs::

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Columns kept in the consolidated processed-modeling file (mechanistic-model
# relevant subset). The per-run processed_data.xlsx still holds every column;
# edit this list to add/remove columns from the consolidated export.
MODEL_COLUMNS = [
    # identity / time / phase
    "run_name", "Process time [h]", "Process phase [-]", "Is induced [-]", "Induction time [h]", "is_sample",
    # environment
    "Temperature [°C]", "pH [-]", "pO2 [%]", "Stirrer speed [1/min]",
    # gas flows + offgas (model inputs)
    "Air flow [L/min]", "Total flow [L/min]", "O2 flow [L/min]", "Gas mix [-]",
    "Inlet O2 [%]", "Inlet CO2 [%]", "Offgas O2 [%]", "Offgas CO2 [%]",
    # offgas rates (derived)
    "CER [mmol/L/h]", "OUR [mmol/L/h]", "RQ [-]", "qO2 [mmol/g/h]", "qCO2 [mmol/g/h]",
    # feed / volume
    "Feed [g]", "Feed [mL]", "Volume [mL]", "Volume corrected [mL]", "Base volume added [mL]",
    "Feed rate exp [%]", "Feed rate ind [%]",
    # biomass / substrate
    "Biomass [g/L]", "Biomass total [g]", "Glucose [g/L]", "OD600 [-]",
    "Glucose consumed [g]", "Glucose consumed rate [g/h]",
    # kinetics
    "mu [1/h]", "qS glucose [g/g/h]", "qP total [g/g/h]",
    # protein
    "Protein soluble [g/L]", "Protein IB [g/L]", "Protein total calculated [g/L]", "Protein total yield [g/g]",
    # carbon balance (summary)
    "C recovered [%]", "C balance closure [%]", "C recovered (measured) [%]",
    "C to CO2 [% of fed]", "C to biomass [% of fed]", "C to product [% of fed]",
]


def find_column(df, target_col):
    """Find a column, tolerating special-character encoding (e.g. the degree sign).

    Exact match first, then a fallback on the base name before ' ['.
    """
    if target_col in df.columns:
        return target_col
    base_name = target_col.split(' [')[0].lower()
    for col in df.columns:
        if col.split(' [')[0].lower() == base_name:
            logger.info(f"Column encoding match: using '{col}' instead of '{target_col}'")
            return col
    return None


def improve_protein_concentrations(df):
    """Recompute protein total (= soluble + IB), fractions, and yields consistently."""
    df_improved = df.copy()
    if 'Protein soluble [g/L]' in df_improved.columns and 'Protein IB [g/L]' in df_improved.columns:
        df_improved['Protein total calculated [g/L]'] = (
            df_improved['Protein soluble [g/L]'] + df_improved['Protein IB [g/L]']
        )
        mask_total = df_improved['Protein total calculated [g/L]'] > 0
        df_improved.loc[mask_total, 'Soluble fraction [%]'] = (
            df_improved.loc[mask_total, 'Protein soluble [g/L]']
            / df_improved.loc[mask_total, 'Protein total calculated [g/L]'] * 100
        )
        df_improved.loc[mask_total, 'IB fraction [%]'] = (
            df_improved.loc[mask_total, 'Protein IB [g/L]']
            / df_improved.loc[mask_total, 'Protein total calculated [g/L]'] * 100
        )
        if 'Biomass [g/L]' in df_improved.columns:
            mask_x = df_improved['Biomass [g/L]'] > 0
            for num, label in [
                ('Protein soluble [g/L]', 'Protein soluble yield [g/g]'),
                ('Protein total calculated [g/L]', 'Protein total yield [g/g]'),
                ('Protein IB [g/L]', 'Protein IB yield [g/g]'),
            ]:
                df_improved.loc[mask_x, label] = (
                    df_improved.loc[mask_x, num] / df_improved.loc[mask_x, 'Biomass [g/L]']
                )
    return df_improved


def _parse_conditions(run_dir):
    """Pull nominal qs and T setpoint from the run_config induction_summary."""
    qs, t_set = None, None
    cfg_path = run_dir / 'run_config.yaml'
    if cfg_path.exists():
        try:
            txt = cfg_path.read_text(encoding='utf-8')
            m_qs = re.search(r'induction_summary:.*?qs\s*=\s*([0-9.]+)', txt, re.IGNORECASE)
            m_t = re.search(r'induction_summary:.*?T\s*=\s*([0-9.]+)', txt, re.IGNORECASE)
            if m_qs:
                qs = float(m_qs.group(1))
            if m_t:
                t_set = float(m_t.group(1))
        except Exception as e:
            logger.warning(f"Could not parse run_config for {run_dir.name}: {e}")
    return {'qs_initial [g/g/h]': qs, 'T_setpoint [°C]': t_set}


def consolidate_xiso_results():
    """Consolidate all Xiso run results into combined Excel files."""
    script_location = Path(__file__).resolve().parent

    # Discover runs: any subdir that has results/processed_data.csv
    run_dirs = sorted(
        d for d in script_location.iterdir()
        if d.is_dir() and (d / "results" / "processed_data.csv").exists()
    )
    if not run_dirs:
        logger.error("No runs with results/processed_data.csv found. Run analyze_xiso_run.py first.")
        return

    logger.info(f"Found {len(run_dirs)} runs to consolidate: {[d.name for d in run_dirs]}")

    all_data = {}       # feed-phase-onward (per run)
    all_time_data = {}  # same, for the All_Time_Points sheet
    all_samples = {}    # sparse sample-point rows (per run)
    full_runs = {}      # full-timeline df per run, for the interactive dashboard

    for run_dir in run_dirs:
        run_name = run_dir.name
        processed_file = run_dir / "results" / "processed_data.csv"

        try:
            logger.info(f"Loading processed data from: {processed_file}")
            df = pd.read_csv(processed_file, sep=';')  # Xiso runner writes sep=';' decimal='.'
        except Exception as e:
            logger.error(f"Error loading {processed_file}: {e}")
            continue

        df = improve_protein_concentrations(df)
        sample_df = df[df['is_sample']].copy() if 'is_sample' in df.columns else pd.DataFrame()
        df['run_name'] = run_name
        full_runs[run_name] = df.copy()  # full timeline for the dashboard
        logger.info(f"Loaded {len(df)} rows, {len(sample_df)} sample points from {run_name}")

        # Filter to feed phase onward, but keep any pre-feed sample points
        all_df = df.copy()
        if 'Feed [mL]' in all_df.columns and 'Process time [h]' in all_df.columns:
            feed_start_mask = all_df['Feed [mL]'].fillna(0) > 0
            if feed_start_mask.any():
                first_feed_idx = feed_start_mask.idxmax()
                feed_start_time = all_df.loc[first_feed_idx, 'Process time [h]']
                pre_feed_sample_idx = []
                if 'is_sample' in all_df.columns:
                    pre_feed_sample_idx = all_df.index[
                        (all_df.index < first_feed_idx) & (all_df['is_sample'] == True)
                    ].tolist()
                all_df = all_df.loc[first_feed_idx:].copy()
                if pre_feed_sample_idx:
                    all_df = pd.concat([df.loc[pre_feed_sample_idx].copy(), all_df]).sort_index()
                logger.info(f"{run_name}: feed phase onward (t >= {feed_start_time:.2f}h), {len(all_df)} rows")
            else:
                logger.warning(f"{run_name}: no feed detected, keeping all data")

        all_data[run_name] = all_df
        all_time_data[run_name] = all_df
        all_samples[run_name] = sample_df

    if not all_data:
        logger.error("No data files loaded")
        return

    # ------------------------------------------------------------------
    # Main consolidated workbook
    # ------------------------------------------------------------------
    output_file = script_location / "Xiso_consolidated_results.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # 1. All time points (with cumulative feed time per run)
        logger.info("Creating consolidated all-time-points sheet...")
        all_timepoints = pd.concat(all_time_data.values(), ignore_index=True)
        all_timepoints = all_timepoints.sort_values(['run_name', 'Process time [h]'])

        all_timepoints['Feed time [h]'] = 0.0
        for run_name in all_timepoints['run_name'].unique():
            run_mask = all_timepoints['run_name'] == run_name
            run_df = all_timepoints[run_mask].sort_values('Process time [h]')
            time_diffs = run_df['Process time [h]'].diff().fillna(0)
            feeding = run_df['Feed [mL]'].fillna(0) > 0
            all_timepoints.loc[run_mask, 'Feed time [h]'] = (time_diffs * feeding).cumsum().values

        key_columns = [
            'run_name', 'Process time [h]', 'Temperature [°C]', 'Is induced [-]', 'Induction time [h]',
            'Biomass [g/L]', 'Glucose [g/L]',
            'Protein soluble [g/L]', 'Protein IB [g/L]', 'Protein total calculated [g/L]',
            'Soluble fraction [%]', 'IB fraction [%]', 'Protein soluble yield [g/g]',
            'Protein IB yield [g/g]', 'Protein total yield [g/g]',
            'Volume [mL]', 'Volume corrected [mL]', 'Feed [mL]',
            'Air flow [L/min]', 'Total flow [L/min]', 'O2 flow [L/min]',
            'Offgas CO2 [%]', 'Offgas O2 [%]', 'Gas mix [-]',
            'Base pump [%]', 'Base pump duration [-]', 'pH [-]', 'pO2 [%]', 'Feed time [h]',
        ]
        available_cols = [c for c in (find_column(all_timepoints, col) for col in key_columns) if c is not None]
        consolidated_data = all_timepoints[available_cols]
        consolidated_data.to_excel(writer, sheet_name='All_Time_Points', index=False)
        logger.info(f"Saved {len(consolidated_data)} time points to All_Time_Points")

        # 2. Run summary (nominal qs/T + measured endpoints)
        logger.info("Creating run summary sheet...")
        summary_data = []
        for run_name, df in all_time_data.items():
            if df.empty:
                continue
            summary = {'run_name': run_name}
            summary.update(_parse_conditions(script_location / run_name))

            ind_col = find_column(df, 'Is induced [-]')
            t_col = find_column(df, 'Temperature [°C]')
            if ind_col and t_col:
                ind_mask = pd.to_numeric(df[ind_col], errors='coerce').eq(1)
                t_ind = pd.to_numeric(df.loc[ind_mask, t_col], errors='coerce').median()
                if pd.notna(t_ind):
                    summary['T_induction [°C]'] = round(float(t_ind), 1)

            if 'Process time [h]' in df.columns:
                summary['total_time_h'] = df['Process time [h]'].max()
            if 'Feed [mL]' in df.columns and 'Process time [h]' in df.columns:
                dfs = df.sort_values('Process time [h]')
                td = dfs['Process time [h]'].diff().fillna(0)
                summary['Feed time [h]'] = td[dfs['Feed [mL]'].fillna(0) > 0].sum()

            for col in ['Biomass [g/L]', 'Protein soluble [g/L]', 'Protein IB [g/L]',
                        'Protein total calculated [g/L]']:
                if col in df.columns:
                    summary[f'{col}_max'] = df[col].max()
                    summary[f'{col}_final'] = df[col].iloc[-1]

            if 'Protein total yield [g/g]' in df.columns:
                fy = df['Protein total yield [g/g]'].dropna()
                if not fy.empty:
                    summary['final_yield'] = fy.iloc[-1]

            cm = find_column(df, 'C recovered (measured)')
            if cm:
                mvals = pd.to_numeric(df[cm], errors='coerce').dropna()
                if not mvals.empty:
                    summary['C recovered (measured) [%]_mean'] = mvals.mean()

            summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Run_Summary', index=False)
        logger.info(f"Saved summary for {len(summary_df)} runs")

        # 3. One sheet per run
        logger.info("Creating individual run sheets...")
        for run_name, df in all_data.items():
            sheet_name = run_name.replace('_', '-')[:31]
            run_cols = [
                'Process time [h]', 'Temperature [°C]', 'Is induced [-]', 'Induction time [h]',
                'Biomass [g/L]', 'Glucose [g/L]',
                'Protein soluble [g/L]', 'Protein IB [g/L]', 'Protein total calculated [g/L]',
                'Soluble fraction [%]', 'IB fraction [%]', 'Protein soluble yield [g/g]',
                'Protein IB yield [g/g]', 'Protein total yield [g/g]',
                'Volume [mL]', 'Volume corrected [mL]', 'Feed [mL]',
                'Air flow [L/min]', 'Total flow [L/min]', 'O2 flow [L/min]',
                'Offgas CO2 [%]', 'Offgas O2 [%]', 'Gas mix [-]',
                'Base pump [%]', 'Base pump duration [-]', 'pH [-]', 'pO2 [%]', 'is_sample',
            ]
            available_run_cols = [c for c in (find_column(df, col) for col in run_cols) if c is not None]
            df[available_run_cols].to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Saved {len(df)} rows for {run_name} -> sheet '{sheet_name}'")

    logger.info(f"Created consolidated workbook: {output_file}")

    # ------------------------------------------------------------------
    # Sparse workbook (sample points + endpoint summary)
    # ------------------------------------------------------------------
    sparse_file = script_location / "Xiso_consolidated_results_sparse.xlsx"
    sample_frames = []
    for run_name, sdf in all_samples.items():
        if sdf is None or len(sdf) == 0:
            continue
        sdf = sdf.copy()
        sdf['run_name'] = run_name
        sample_frames.append(sdf)

    sample_key = [
        'run_name', 'Process time [h]', 'Temperature [°C]', 'Is induced [-]', 'Induction time [h]',
        'Biomass [g/L]', 'Glucose [g/L]',
        'Protein soluble [g/L]', 'Protein IB [g/L]', 'Protein total calculated [g/L]',
        'Soluble fraction [%]', 'IB fraction [%]',
        'Protein soluble yield [g/g]', 'Protein IB yield [g/g]', 'Protein total yield [g/g]',
        'Volume [mL]', 'Volume corrected [mL]', 'Feed [mL]',
        'Offgas CO2 [%]', 'Offgas O2 [%]', 'pH [-]', 'pO2 [%]',
    ]
    if sample_frames:
        all_samples_df = pd.concat(sample_frames, ignore_index=True).sort_values(['run_name', 'Process time [h]'])
        sp_cols = [c for c in (find_column(all_samples_df, c) for c in sample_key) if c is not None]
        sample_out = all_samples_df[sp_cols]
    else:
        sample_out = pd.DataFrame()

    n_samples = {rn: len(sdf) for rn, sdf in all_samples.items() if sdf is not None}
    sparse_summary = summary_df.copy()
    sparse_summary['num_samples'] = sparse_summary['run_name'].map(n_samples)
    sparse_cols_pref = [
        'run_name', 'T_induction [°C]', 'qs_initial [g/g/h]', 'T_setpoint [°C]', 'num_samples',
        'total_time_h', 'Feed time [h]',
        'Biomass [g/L]_final', 'Protein soluble [g/L]_final', 'Protein IB [g/L]_final',
        'Protein total calculated [g/L]_final', 'final_yield',
    ]
    sparse_cols = [c for c in sparse_cols_pref if c in sparse_summary.columns]

    with pd.ExcelWriter(sparse_file, engine='openpyxl') as writer:
        sample_out.to_excel(writer, sheet_name='Sample_Points', index=False)
        sparse_summary[sparse_cols].to_excel(writer, sheet_name='Summary', index=False)
    logger.info(f"Saved sparse file: {len(sample_out)} sample points, {len(sparse_summary)} runs -> {sparse_file}")

    # ------------------------------------------------------------------
    # Interactive comparison dashboard (runs overlaid, category tabs)
    # ------------------------------------------------------------------
    try:
        InteractiveDashboard(script_location).plot_comparison(
            full_runs, output_filename="Xiso_consolidated_dashboard.html"
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
    processed_file = script_location / "Xiso_consolidated_results_processed.xlsx"
    with pd.ExcelWriter(processed_file, engine="openpyxl") as writer:
        processed_all[keep].to_excel(writer, sheet_name="All_Processed", index=False)
    missing = [c for c in MODEL_COLUMNS if c not in processed_all.columns]
    logger.info(
        f"Saved processed-modeling file: {processed_file} "
        f"({len(processed_all)} rows, {len(keep)} cols)"
    )
    if missing:
        logger.warning(f"  requested MODEL_COLUMNS not present: {missing}")

    # Console summary
    logger.info("\nXiso Consolidated Results Summary:")
    logger.info(f"- Runs processed: {len(all_data)}")
    logger.info(f"- Total time points: {len(consolidated_data)}")
    logger.info(f"- Workbook: {output_file} (All_Time_Points, Run_Summary, {len(all_data)} run sheets)")
    logger.info(f"- Sparse:    {sparse_file}")


if __name__ == "__main__":
    consolidate_xiso_results()
