# src/upstream_analysis/visualization/plotting.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """Removes characters potentially problematic for filenames."""
    # Remove or replace special characters like %, /, :, whitespace, °
    sanitized = re.sub(r'[<>:"/\\|?*%\.,\s°]', '_', filename)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length if necessary
    return sanitized[:100] # Limit length to avoid issues

class DataVisualizer:
    """Generates time-series plots for fermentation data."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
        logger.info(f"DataVisualizer initialized. Plots will be saved to: {self.output_dir}")

    def plot_selected_parameters(self, df: pd.DataFrame, parameters: list, time_col: str = 'process_time'):
        """
        Creates and saves individual plots for selected parameters against time.

        Args:
            df: The processed DataFrame containing the data.
            parameters: A list of column names to plot.
            time_col: The name of the column to use for the x-axis (time).
        """
        if time_col not in df.columns:
            logger.error(f"Time column '{time_col}' not found in DataFrame. Cannot generate plots.")
            return
        if df[time_col].isnull().all():
            logger.error(f"Time column '{time_col}' contains only NaNs. Cannot generate plots.")
            return

        logger.info(f"Generating plots for {len(parameters)} parameters...")

        time_data = df[time_col]

        for param in parameters:
            if param not in df.columns:
                logger.warning(f"Parameter '{param}' not found in DataFrame. Skipping plot.")
                continue

            param_data = df[param]

            # Skip plotting if the parameter column is entirely empty
            if param_data.isnull().all():
                logger.info(f"Parameter '{param}' contains only NaNs. Skipping plot.")
                continue

            fig, ax = plt.subplots(figsize=(10, 5)) # Create a new figure for each plot

            # Check if it looks like an offline measurement (many NaNs) or is a calculated protein parameter
            # Heuristic: if more than 75% are NaN, plot as scatter/markers
            # Also plot calculated protein parameters as scatter since they're derived from offline measurements
            is_likely_offline = param_data.isnull().sum() / len(param_data) > 0.75
            is_calculated_protein = any(x in param.lower() for x in ['total_calculated', 'yield', 'fraction'])

            try:
                if is_likely_offline or is_calculated_protein:
                    # Plot only non-NaN points for sparse offline data or calculated protein parameters
                    valid_mask = param_data.notna()
                    if valid_mask.any():
                        ax.plot(time_data[valid_mask], param_data[valid_mask],
                                marker='o', linestyle='--', markersize=5, label=param)
                        logger.debug(f"Plotting '{param}' as sparse (offline/calculated protein) data.")
                    else:
                         logger.info(f"Parameter '{param}' has no valid data points despite not being all NaNs. Skipping plot.")
                         plt.close(fig)
                         continue

                else:
                    # Plot continuous online data
                    ax.plot(time_data, param_data, label=param, linewidth=1.5)
                    logger.debug(f"Plotting '{param}' as continuous (online) data.")


                # Format time label - only append [h] if not already present
                time_label = time_col.replace('_', ' ').capitalize()
                if not time_label.endswith('[h]'):
                    time_label += " [h]"
                ax.set_xlabel(time_label)
                ax.set_ylabel(param.replace('_', ' ').capitalize()) # Use standard name as label base
                ax.set_title(f"{param.replace('_', ' ').capitalize()} vs Time")
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.tick_params(axis='x', rotation=45)

                # Special handling for RQ plot - limit y-axis to meaningful range
                if 'rq' in param.lower() and param.replace('_', ' ').lower().startswith('rq'):
                    # Get the valid data range
                    valid_data = param_data.dropna()
                    if not valid_data.empty:
                        # Use percentiles to set reasonable limits, excluding extreme outliers
                        q1, q99 = valid_data.quantile([0.01, 0.99])
                        # Set range to include most data, with minimum range of 0-2 for RQ
                        y_min = min(0, q1)
                        y_max = max(2.5, q99)  # Upper limit of 2.5 to capture meaningful variation
                        ax.set_ylim(y_min, y_max)
                        # Add a horizontal line at RQ = 1 (complete glucose oxidation)
                        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='RQ = 1.0')
                        ax.legend()
                        logger.debug(f"Set RQ y-axis limits to [{y_min:.2f}, {y_max:.2f}] with reference line at y=1.0")

                fig.tight_layout()

                # Save the plot
                safe_filename = sanitize_filename(param) + "_timeseries.png"
                save_path = self.output_dir / safe_filename
                fig.savefig(save_path, dpi=150)
                logger.info(f"Saved plot: {save_path}")

            except Exception as e:
                logger.error(f"Failed to generate plot for '{param}': {e}")
            finally:
                plt.close(fig) # Close the figure to free memory

        logger.info("Finished generating plots.")