"""One-call pipeline runner.

Wraps the Parser → Cleaner → Aligner → Processor → Visualizer chain so that
scripts and users do not re-implement the orchestration. Typical use::

    from upstream_analysis import run_directory

    result = run_directory("examples/hgh/R5_lowlow")
    result.processed.to_csv("processed.csv", sep=";", decimal=".")

``run_directory`` expects the layout documented in docs/DATA_LAYOUT.md:
``<run_dir>/run_config.yaml`` plus exactly one online CSV (and optionally
one offline xlsx) in ``<run_dir>/data/``.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config.loader import load_config
from .core.validation import coerce_numeric_columns
from .data.alignment import DataAligner
from .data.cleaning import FermentationDataCleaner
from .data.parser import BioreactorDataParser, OfflineMeasurementsParser
from .data.processor import FermentationDataProcessor
from .visualization.plotting import DataVisualizer

logger = logging.getLogger(__name__)

DEFAULT_PLOT_PARAMETERS = [
    "Biomass [g/L]",
    "Feed [g]",
    "Feed [mL]",
    "Volume [mL]",
    "Volume corrected [mL]",
    "CER [mmol/L/h]",
    "OUR [mmol/L/h]",
    "RQ [-]",
    "pH [-]",
    "pO2 [%]",
    "Is induced [-]",
]


@dataclass
class PipelineResult:
    """Outputs of a pipeline run."""

    processed: pd.DataFrame
    samples: pd.DataFrame
    output_dir: Path | None = None
    plot_paths: list[Path] = field(default_factory=list)


def run_pipeline(
    config: dict,
    online_path: Path | str,
    offline_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    plot_parameters: list[str] | None = None,
) -> PipelineResult:
    """Run parse → clean → align → process (→ plot, save) for one fermentation.

    Args:
        config: merged configuration (see :func:`load_config`).
        online_path: bioreactor export (CSV/Excel per the parsing config).
        offline_path: offline sample measurements (Excel); optional — the
            pipeline degrades gracefully without it.
        output_dir: when given, writes ``processed_data.csv`` and
            ``samples_points_data.csv`` (``sep=';'``, ``decimal=','``) and
            renders overview plots there.
        plot_parameters: parameters to plot; defaults to a compact standard
            set.

    Returns:
        :class:`PipelineResult` with the processed and sample frames.
    """
    online = BioreactorDataParser(config).parse_file(Path(online_path))
    logger.info("Parsed online data: %s rows", len(online))
    online = FermentationDataCleaner(config).clean_data(online)

    offline = None
    if offline_path is not None:
        offline = OfflineMeasurementsParser(config).parse_file(Path(offline_path))
        logger.info("Parsed offline data: %s rows", len(offline))

    df_merged, df_samples = DataAligner(config.get("alignment", {})).merge_datasets(
        online, offline
    )
    processed = FermentationDataProcessor(config).process_data(df_merged)
    # Alignment can leave offline-derived columns as string-float objects;
    # normalize so CSV export and downstream arithmetic get real numerics.
    processed = coerce_numeric_columns(processed)
    df_samples = coerce_numeric_columns(df_samples)
    logger.info("Processed data: %s rows", len(processed))

    result = PipelineResult(processed=processed, samples=df_samples)
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        result.output_dir.mkdir(parents=True, exist_ok=True)
        processed.to_csv(
            result.output_dir / "processed_data.csv", index=False, sep=";", decimal=","
        )
        if not df_samples.empty:
            df_samples.to_csv(
                result.output_dir / "samples_points_data.csv",
                index=False,
                sep=";",
                decimal=",",
            )
        params = plot_parameters or DEFAULT_PLOT_PARAMETERS
        plotable = [p for p in params if p in processed.columns]
        visualizer = DataVisualizer(output_dir=result.output_dir)
        visualizer.plot_selected_parameters(
            processed, parameters=plotable, time_col="Process time [h]"
        )
        result.plot_paths = sorted(result.output_dir.glob("*timeseries.png"))
    return result


def run_directory(
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    plot_parameters: list[str] | None = None,
) -> PipelineResult:
    """Run the pipeline for a run folder (docs/DATA_LAYOUT.md layout).

    Auto-detects ``<run_dir>/run_config.yaml`` and exactly one online CSV in
    ``<run_dir>/data/``; uses the single xlsx as offline measurements when
    present. ``output_dir`` defaults to ``<run_dir>/results``.
    """
    run_path = Path(run_dir)
    output = Path(output_dir) if output_dir is not None else run_path / "results"

    config_path = run_path / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"no run_config.yaml in {run_path}")
    config = load_config(config_path)

    data_dir = run_path / "data"
    csv_files = sorted(data_dir.glob("*.csv"))
    xlsx_files = sorted(data_dir.glob("*.xlsx"))

    online_file: Path
    offline_file: Path | None = None
    if len(csv_files) == 1:
        # Convention: single CSV is the online export, optional xlsx the offline sheet.
        online_file = csv_files[0]
        offline_file = xlsx_files[0] if len(xlsx_files) == 1 else None
    elif not csv_files and xlsx_files:
        # Excel-online layouts: the offline workbook is the one named
        # *samples*; the online export is whatever remains after excluding
        # it and operator reference workbooks (Evaluation_*.xlsx).
        samples = [f for f in xlsx_files if "sample" in f.name.lower()]
        evaluation = [f for f in xlsx_files if "evaluation" in f.name.lower()]
        rest = [f for f in xlsx_files if f not in samples and f not in evaluation]
        if len(samples) == 1 and len(rest) == 1:
            online_file, offline_file = rest[0], samples[0]
            if evaluation:
                logger.info(
                    "ignoring reference workbook(s) in data/: %s",
                    [f.name for f in evaluation],
                )
        else:
            raise ValueError(
                f"ambiguous data files in {data_dir}: {[f.name for f in xlsx_files]} — "
                "name the offline workbook samples.xlsx, keep exactly one online "
                "export, and move reference workbooks (Evaluation_*.xlsx) out of "
                "data/, or call run_pipeline() with explicit paths"
            )
    else:
        raise ValueError(
            f"expected one online CSV (+ optional offline xlsx) in {data_dir}, "
            f"found CSVs {[f.name for f in csv_files]} and xlsx {[f.name for f in xlsx_files]} — "
            "call run_pipeline() with explicit paths for unusual layouts"
        )

    return run_pipeline(
        config,
        online_file,
        offline_path=offline_file,
        output_dir=output,
        plot_parameters=plot_parameters,
    )


def main() -> int:
    """CLI: ``python -m upstream_analysis.pipeline <run_dir> [--output-dir D]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the USP pipeline for one run folder."
    )
    parser.add_argument("run_dir", help="run folder (run_config.yaml + data/)")
    parser.add_argument("--output-dir", default=None, help="default: <run_dir>/results")
    args = parser.parse_args()
    run_directory(args.run_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
