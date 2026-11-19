"""
Data loader for OptFed mechanistic model training and testing.

Handles loading the compiled fermentation data, splitting into train/test sets,
and preprocessing for model fitting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

# Avoid circular import
if TYPE_CHECKING:
    from model import ControlProfile
else:
    # Import at runtime to avoid circular dependency
    ControlProfile = None

# Import for actual use (will be resolved at runtime)
try:
    from model import ControlProfile
except ImportError:
    pass


# Train/test split as specified
TRAIN_RUN_IDS = [
    "ferm_01",
    "ferm_02",
    "ferm_03",
    "ferm_04",  # Corners
    "ferm_05",
    "ferm_06",
    "ferm_07",  # Center points
]

TEST_RUN_IDS = [
    "ferm_10",
    "ferm_11",
    "ferm_12",  # Star points
    "ferm_opt",
]

FEED_CONCENTRATION = 400.0  # g/L glucose


@dataclass
class FermentationRun:
    """Data for a single fermentation run."""
    run_id: str
    data: pd.DataFrame
    is_induction_phase: bool = False

    @property
    def time(self) -> np.ndarray:
        """Time points (h)."""
        return self.data["time"].values

    @property
    def X(self) -> np.ndarray:
        """Biomass concentration (g/L)."""
        return self.data.get("X", pd.Series([np.nan] * len(self.data))).values

    @property
    def P(self) -> np.ndarray:
        """Product concentration (g/L)."""
        return self.data.get("P", pd.Series([np.nan] * len(self.data))).values

    @property
    def V(self) -> np.ndarray:
        """Volume (mL)."""
        return self.data.get("V", pd.Series([np.nan] * len(self.data))).values

    @property
    def G(self) -> np.ndarray:
        """Substrate concentration (g/L) - if available."""
        return self.data.get("G", pd.Series([np.nan] * len(self.data))).values

    @property
    def temp(self) -> np.ndarray:
        """Temperature (°C)."""
        # Use midpoint of typical range (25-42°C) if no temperature data
        return self.data.get("temp", pd.Series([33.5] * len(self.data))).values

    @property
    def feed(self) -> np.ndarray:
        """Cumulative feed (mL)."""
        return self.data.get("feed", pd.Series([0.0] * len(self.data))).values


@dataclass
class ProcessData:
    """Container for training and test data."""
    train_runs: list[FermentationRun]
    test_runs: list[FermentationRun]

    @property
    def runs(self) -> dict[str, FermentationRun]:
        """All runs as a dictionary keyed by run_id."""
        return {run.run_id: run for run in self.train_runs + self.test_runs}

    def get_train_dataframe(self) -> pd.DataFrame:
        """Combine all training runs into a single DataFrame."""
        return pd.concat([run.data for run in self.train_runs], ignore_index=True)

    def get_test_dataframe(self) -> pd.DataFrame:
        """Combine all test runs into a single DataFrame."""
        return pd.concat([run.data for run in self.test_runs], ignore_index=True)


class DataLoader:
    """
    Load and preprocess fermentation data for OptFed model.

    The compiled data contains:
    - Online measurements (time, temp, feed, reactor_volume, inductor_switch, etc.)
    - Offline measurements (X, P - biomass and product)
    """

    def __init__(
        self,
        data_path: str | Path,
        train_run_ids: Optional[list[str]] = None,
        test_run_ids: Optional[list[str]] = None,
    ):
        """
        Initialize the data loader.

        Args:
            data_path: Path to traintest_compiled.xlsx
            train_run_ids: List of training run IDs (default: TRAIN_RUN_IDS)
            test_run_ids: List of test run IDs (default: TEST_RUN_IDS)
        """
        self.data_path = Path(data_path)
        self.train_run_ids = train_run_ids or TRAIN_RUN_IDS
        self.test_run_ids = test_run_ids or TEST_RUN_IDS

        # Column mapping for internal use
        self._column_map = {
            "time": "time",
            "X": "X",
            "P": "P",
            "temp": "temp",
            "feed": "feed",
            "reactor_volume": "V",
            "RunID": "run_id",
            "inductor_switch": "induction",
        }

    def load_data(self) -> ProcessData:
        """
        Load the compiled Excel file and split into train/test sets.

        Returns:
            ProcessData with train and test runs
        """
        df = pd.read_excel(self.data_path)

        # Normalize column names
        df = df.rename(columns=self._column_map)

        # Ensure run_id column exists
        if "run_id" not in df.columns:
            raise ValueError(f"RunID column not found. Available: {df.columns.tolist()}")

        train_runs = []
        test_runs = []

        for run_id in df["run_id"].unique():
            run_df = df[df["run_id"] == run_id].copy()
            run_df = run_df.sort_values("time").reset_index(drop=True)

            ferm_run = FermentationRun(
                run_id=run_id,
                data=run_df,
                is_induction_phase=run_df["induction"].max() > 0 if "induction" in run_df.columns else False,
            )

            if run_id in self.train_run_ids:
                train_runs.append(ferm_run)
            elif run_id in self.test_run_ids:
                test_runs.append(ferm_run)

        return ProcessData(train_runs=train_runs, test_runs=test_runs)

    def calculate_rates(
        self,
        run: FermentationRun,
        smooth: bool = True,
        window: int = 5,
    ) -> pd.DataFrame:
        """
        Calculate uptake, growth, and production rates using the differential method.

        Fits splines to the concentration data and differentiates to estimate rates.

        Args:
            run: Fermentation run with time-series data
            smooth: Whether to apply smoothing spline
            window: Window size for moving average (if smooth=False)

        Returns:
            DataFrame with estimated rates:
            - mu_hat: Specific growth rate (1/h)
            - pi_hat: Specific production rate (1/h)
            - gamma_hat: Total substrate uptake rate (1/h)
        """
        df = run.data.copy()

        # Filter to rows with biomass measurements
        df_valid = df[df["X"].notna()].copy()

        if len(df_valid) < 3:
            raise ValueError(f"Insufficient data points for run {run.run_id}")

        # Convert concentrations from g/L to total amounts
        # Note: X and P are stored as concentrations (g/L)
        # We need total amounts for rate calculations

        t = df_valid["time"].values
        X = df_valid["X"].values  # g/L
        V = df_valid["V"].values / 1000.0  # mL to L

        # Handle product if available
        P = df_valid["P"].values if "P" in df_valid.columns else np.zeros_like(X)

        # Calculate total amounts
        X_total = X * V  # g
        P_total = P * V  # g

        # Calculate derivatives using finite differences
        dt = np.diff(t)
        dX = np.diff(X_total)
        dP = np.diff(P_total)
        dV = np.diff(V)

        # Feed rate (L/h) - from cumulative feed
        if "feed" in df_valid.columns:
            feed_cum = df_valid["feed"].values / 1000.0  # mL to L
            f = np.gradient(feed_cum, t)  # L/h
        else:
            f = np.zeros_like(t)

        # Mid-point time array for rates
        t_mid = (t[:-1] + t[1:]) / 2

        # Specific rates from differential equations
        # dX/dt = mu*X - (f/V)*X  =>  mu = (dX/dt + (f/V)*X) / X
        # dP/dt = pi*X - (f/V)*P  =>  pi = (dP/dt + (f/V)*P) / P

        V_mid = (V[:-1] + V[1:]) / 2
        X_mid = (X_total[:-1] + X_total[1:]) / 2
        P_mid = (P_total[:-1] + P_total[1:]) / 2
        f_mid = (f[:-1] + f[1:]) / 2

        dXdt = dX / dt
        dPdt = dP / dt

        # Calculate specific rates
        mu_hat = np.zeros_like(t_mid)
        pi_hat = np.zeros_like(t_mid)

        valid_mask = (X_mid > 0) & (V_mid > 0)
        mu_hat[valid_mask] = (dXdt[valid_mask] + (f_mid[valid_mask] / V_mid[valid_mask]) * X_mid[valid_mask]) / X_mid[valid_mask]

        valid_mask_p = (P_mid > 0) & (V_mid > 0)
        pi_hat[valid_mask_p] = (dPdt[valid_mask_p] + (f_mid[valid_mask_p] / V_mid[valid_mask_p]) * P_mid[valid_mask_p]) / P_mid[valid_mask_p]

        # Substrate uptake rate (from balance equation)
        # Assuming dG/dt ≈ 0 (quasi-steady state for substrate)
        # gamma ≈ (f/V) * (Gf - G) / X
        # For now, estimate from growth and production
        # gamma = mu/Y_XG + pi/Y_PG + gamma_maintenance

        # Create results dataframe
        result = pd.DataFrame({
            "time": t_mid,
            "mu_hat": mu_hat,
            "pi_hat": pi_hat,
            "V": V_mid,
            "X_conc": X_mid / V_mid,
            "P_conc": P_mid / V_mid,
            "feed_rate": f_mid,
        })

        # Apply smoothing if requested
        if smooth and len(result) > window:
            for col in ["mu_hat", "pi_hat"]:
                result[col] = result[col].rolling(
                    window=window,
                    center=True,
                    min_periods=1,
                ).mean()

        return result

    def get_induction_point(self, run: FermentationRun) -> Optional[float]:
        """
        Find the induction time for a run.

        Args:
            run: Fermentation run data

        Returns:
            Induction time in hours, or None if not found
        """
        if "induction" not in run.data.columns:
            return None

        induction_points = run.data[run.data["induction"] > 0]
        if len(induction_points) == 0:
            return None

        return induction_points["time"].min()


def load_process_data(data_path: str | Path) -> ProcessData:
    """Convenience function to load all data."""
    loader = DataLoader(data_path)
    return loader.load_data()


def load_traintest_data(
    data_path: str | Path,
) -> tuple[ProcessData, ProcessData]:
    """
    Load training and test data as separate ProcessData objects.

    This is a convenience function that separates train and test runs
    into individual ProcessData containers for easier handling.

    Args:
        data_path: Path to traintest_compiled.xlsx

    Returns:
        Tuple of (train_data, test_data) where each is a ProcessData
        containing only the respective runs.
    """
    loader = DataLoader(data_path)
    full_data = loader.load_data()

    # Create separate ProcessData objects
    train_data = ProcessData(
        train_runs=full_data.train_runs,
        test_runs=[],  # No test runs in training data
    )

    test_data = ProcessData(
        train_runs=[],  # No training runs in test data
        test_runs=full_data.test_runs,
    )

    return train_data, test_data


def get_control_profile_for_run(run: FermentationRun) -> ControlProfile:
    """
    Create a ControlProfile from experimental run data.

    Extracts feed rate and temperature profiles from the run data
    to use for model simulation.

    Args:
        run: FermentationRun with experimental data

    Returns:
        ControlProfile with feed_rate and temperature profiles
    """
    from model import ControlProfile, FeedStrategy

    # Extract time points
    time = run.time

    # Calculate feed rate from cumulative feed (L/h)
    # Feed is stored in mL, convert to L
    if "feed" in run.data.columns:
        feed_cum = run.feed / 1000.0  # mL to L
        feed_rate = np.zeros_like(time)

        # Calculate derivative for feed rate
        valid_mask = ~np.isnan(feed_cum)
        if np.sum(valid_mask) > 1:
            feed_rate[valid_mask] = np.gradient(feed_cum[valid_mask], time[valid_mask])
            # Clip negative values (no backward feeding)
            feed_rate = np.clip(feed_rate, 0, None)
    else:
        feed_rate = np.zeros_like(time)

    # Get temperature profile
    temp = run.temp

    # Create feed profile function
    def feed_func(t: float) -> float:
        """Feed rate at time t (L/h)."""
        if t < time[0]:
            return 0.0
        elif t > time[-1]:
            return feed_rate[-1]
        else:
            return np.interp(t, time, feed_rate)

    # Create temperature profile function
    def temp_func(t: float) -> float:
        """Temperature at time t (°C)."""
        if t < time[0]:
            return temp[0]
        elif t > time[-1]:
            return temp[-1]
        else:
            return np.interp(t, time, temp)

    # Create FeedStrategy from feed rate function
    feed_strategy = FeedStrategy.from_function(feed_func)

    # Temperature can be passed directly as a function
    return ControlProfile(feed_rate=feed_strategy, temperature=temp_func)


if __name__ == "__main__":
    # Example usage
    data_path = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\doe_example\data\traintest_compiled.xlsx"

    loader = DataLoader(data_path)
    data = loader.load_data()

    print(f"Training runs: {len(data.train_runs)}")
    for run in data.train_runs:
        print(f"  - {run.run_id}: {len(run.data)} time points")

    print(f"\nTest runs: {len(data.test_runs)}")
    for run in data.test_runs:
        print(f"  - {run.run_id}: {len(run.data)} time points")

    # Example: Calculate rates for one run
    if data.train_runs:
        run = data.train_runs[0]
        print(f"\n--- Rates for {run.run_id} ---")
        induction_time = loader.get_induction_point(run)
        print(f"Induction time: {induction_time}")

        try:
            rates = loader.calculate_rates(run)
            print(f"Calculated {len(rates)} rate estimates")
            print(rates.head())
        except Exception as e:
            print(f"Could not calculate rates: {e}")
