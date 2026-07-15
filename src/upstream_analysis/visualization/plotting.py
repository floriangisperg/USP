# src/upstream_analysis/visualization/plotting.py
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

# A column whose values are missing more than this fraction of the timeline is
# treated as intermittent (offline-sample-derived) and plotted as markers
# rather than a continuous line.
SPARSE_NAN_FRACTION = 0.75


def sanitize_filename(filename: str) -> str:
    """Removes characters potentially problematic for filenames."""
    # Replace characters unsafe in filenames: path separators, quotes, wildcards,
    # brackets/parentheses, percent, punctuation, whitespace, and the degree symbol.
    sanitized = re.sub(r'[<>:"/\\|?*%\.,\s°\[\]\(\)]', "_", filename)
    # Strip leading underscores only; trailing underscores are kept intentionally.
    sanitized = sanitized.lstrip("_")
    # Limit length if necessary
    return sanitized[:100]  # Limit length to avoid issues


class DataVisualizer:
    """Generates time-series plots for fermentation data."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style
        logger.info(
            f"DataVisualizer initialized. Plots will be saved to: {self.output_dir}"
        )

    def plot_selected_parameters(
        self, df: pd.DataFrame, parameters: list, time_col: str = "process_time"
    ):
        """
        Creates and saves individual plots for selected parameters against time.

        Args:
            df: The processed DataFrame containing the data.
            parameters: A list of column names to plot.
            time_col: The name of the column to use for the x-axis (time).
        """
        if time_col not in df.columns:
            logger.error(
                f"Time column '{time_col}' not found in DataFrame. Cannot generate plots."
            )
            return
        if df[time_col].isnull().all():
            logger.error(
                f"Time column '{time_col}' contains only NaNs. Cannot generate plots."
            )
            return

        logger.info(f"Generating plots for {len(parameters)} parameters...")

        time_data = df[time_col]

        for param in parameters:
            if param not in df.columns:
                logger.warning(
                    f"Parameter '{param}' not found in DataFrame. Skipping plot."
                )
                continue

            param_data = df[param]

            # Skip plotting if the parameter column is entirely empty
            if param_data.isnull().all():
                logger.info(f"Parameter '{param}' contains only NaNs. Skipping plot.")
                continue

            fig, ax = plt.subplots(figsize=(10, 5))  # Create a new figure for each plot

            # Decide the plot style from the data, not the column name: a column
            # whose values are mostly missing is intermittent (an offline-sample
            # measurement, or a quantity derived from one) and is plotted as
            # markers; otherwise it is a continuous online signal plotted as a
            # line. This subsumes the old "calculated protein" name match, since
            # those columns are sparse by construction.
            is_sparse = (
                param_data.isnull().sum() / len(param_data) > SPARSE_NAN_FRACTION
            )

            try:
                if is_sparse:
                    # Plot only the non-NaN points for sparse (offline-derived) data
                    valid_mask = param_data.notna()
                    if valid_mask.any():
                        ax.plot(
                            time_data[valid_mask],
                            param_data[valid_mask],
                            marker="o",
                            linestyle="--",
                            markersize=5,
                            label=param,
                        )
                        logger.debug(
                            f"Plotting '{param}' as sparse (offline-derived) data."
                        )
                    else:
                        logger.info(
                            f"Parameter '{param}' has no valid data points despite not being all NaNs. Skipping plot."
                        )
                        plt.close(fig)
                        continue

                else:
                    # Plot continuous online data
                    ax.plot(time_data, param_data, label=param, linewidth=1.5)
                    logger.debug(f"Plotting '{param}' as continuous (online) data.")

                # Format time label - only append [h] if not already present
                time_label = time_col.replace("_", " ").capitalize()
                if not time_label.endswith("[h]"):
                    time_label += " [h]"
                ax.set_xlabel(time_label)
                ax.set_ylabel(
                    param.replace("_", " ").capitalize()
                )  # Use standard name as label base
                ax.set_title(f"{param.replace('_', ' ').capitalize()} vs Time")
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                ax.tick_params(axis="x", rotation=45)

                fig.tight_layout()

                # Save the plot
                safe_filename = sanitize_filename(param) + "_timeseries.png"
                save_path = self.output_dir / safe_filename
                fig.savefig(save_path, dpi=150)
                logger.info(f"Saved plot: {save_path}")

            except Exception as e:
                logger.error(f"Failed to generate plot for '{param}': {e}")
            finally:
                plt.close(fig)  # Close the figure to free memory

        logger.info("Finished generating plots.")
