"""
End-to-End Validation Script for OptFed Model

This script performs complete validation of the fitted OptFed model:
1. Loads fitted parameters from results/
2. Applies model selection (F-test based)
3. Validates on training and test data
4. Generates comprehensive validation plots

Usage:
    python validate_model.py
    python validate_model.py --data ../data/traintest_compiled.xlsx
    python validate_model.py --skip-selection  # Skip model selection
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from data_loader import (
    FermentationRun,
    ProcessData,
    get_control_profile_for_run,
    load_traintest_data,
)
from fitting import FitConfig, FitResult, ModelSelector, ParameterEstimator
from model import (
    ControlProfile,
    FeedStrategy,
    InitialConditions,
    ModelParameters,
    OptFedModel,
    create_model_erying,
)
from visualization import (
    plot_model_vs_data,
    plot_multi_run_comparison,
    plot_parity,
    plot_validation_summary,
)


def load_fitted_params(results_dir: str | Path = "results/") -> ModelParameters:
    """Load fitted parameters from pickle file."""
    import sys

    import fitting

    results_path = Path(results_dir)
    pkl_path = results_path / "fitted_params.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Fitted parameters not found at {pkl_path}. "
            "Run parameter fitting first: python -c 'from fitting import fit_model_to_data; fit_model_to_data()'"
        )

    # Map the FitResult class to the fitting module when unpickling
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "FitResult":
                return fitting.FitResult
            return super().find_class(module, name)

    with open(pkl_path, "rb") as f:
        fit_result = CustomUnpickler(f).load()

    print(f"[OK] Loaded fitted parameters from {pkl_path}")
    return fit_result.params


def create_controls_for_data(
    data: ProcessData,
) -> dict[str, ControlProfile]:
    """Create control profiles for all runs in the dataset."""
    controls = {}

    for run_id, run in data.runs.items():
        control = get_control_profile_for_run(run)
        controls[run_id] = control
        print(f"  [OK] Created control for {run_id}")

    return controls


def validate_single_run(
    model: OptFedModel,
    run: FermentationRun,
    control: ControlProfile,
    save_dir: Optional[Path] = None,
) -> dict[str, float]:
    """Validate model on a single run, return error metrics."""
    # Set initial conditions from first measurement
    # Note: Excel data has V in mL, need to convert to L
    # Use default G0=0.5 g/L if no valid substrate data
    G_first = run.G[0] if len(run.G) > 0 and np.isfinite(run.G[0]) else 0.5
    V_first_L = run.V[0] / 1000.0  # Convert mL to L

    init = InitialConditions.from_first_measurement(
        X_first=run.X[0],
        P_first=run.P[0],
        V_first=V_first_L,
        G_first=G_first,
    )
    model.init_cond = init

    # Simulate with stiff solver for better convergence
    t_span = (run.time[0], run.time[-1])
    t_eval = run.time
    result = model.simulate(t_span, t_eval, control, method="BDF", rtol=1e-4, atol=1e-6)

    # Calculate error metrics
    metrics = {}
    for var, name in [("X", "Biomass"), ("P", "Product"), ("V", "Volume")]:
        if var in run.data and len(run.data[var]) > 0:
            y_true = run.data[var]
            # Volume is stored as "V" (in mL in data, L in model result)
            # X and P are stored as concentrations
            if var == "V":
                y_pred = result[var] * 1000.0  # Convert L to mL for comparison
            else:
                y_pred = result[f"{var}_conc"]

            # Align predictions with measurements
            y_pred_aligned = np.interp(run.time, result["t"], y_pred)

            # RMSE
            rmse = np.sqrt(np.mean((y_true - y_pred_aligned) ** 2))

            # MAE
            mae = np.mean(np.abs(y_true - y_pred_aligned))

            # R²
            ss_res = np.sum((y_true - y_pred_aligned) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            metrics[f"{var}_rmse"] = rmse
            metrics[f"{var}_mae"] = mae
            metrics[f"{var}_r2"] = r2

    return metrics


def run_validation(
    params: ModelParameters,
    train_data: ProcessData,
    test_data: ProcessData,
    save_dir: Optional[Path] = None,
) -> dict:
    """Run complete validation on train and test data."""
    # Create model with fitted parameters
    model = create_model_erying()
    model.params = params

    # Create control profiles
    print("\n=== Creating Control Profiles ===")
    train_controls = create_controls_for_data(train_data)
    test_controls = create_controls_for_data(test_data)

    # Combine controls
    all_controls = {**train_controls, **test_controls}

    # Validate on training runs
    print("\n=== Validating on Training Data ===")
    train_metrics = {}
    for run_id, run in train_data.runs.items():
        print(f"  {run_id}...")
        metrics = validate_single_run(
            model, run, train_controls[run_id], save_dir
        )
        train_metrics[run_id] = metrics

    # Validate on test runs
    print("\n=== Validating on Test Data ===")
    test_metrics = {}
    for run_id, run in test_data.runs.items():
        print(f"  {run_id}...")
        metrics = validate_single_run(
            model, run, test_controls[run_id], save_dir
        )
        test_metrics[run_id] = metrics

    # Calculate summary statistics
    print("\n=== Validation Summary ===")

    def summarize_metrics(metrics_dict: dict, label: str):
        print(f"\n{label}:")
        for var in ["X", "P", "V"]:
            r2_values = [m.get(f"{var}_r2", 0) for m in metrics_dict.values()]
            rmse_values = [m.get(f"{var}_rmse", 0) for m in metrics_dict.values()]
            print(f"  {var}: R² = {np.mean(r2_values):.3f} ± {np.std(r2_values):.3f}, "
                  f"RMSE = {np.mean(rmse_values):.3f}")

    summarize_metrics(train_metrics, "Training")
    summarize_metrics(test_metrics, "Test")

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_controls": train_controls,
        "test_controls": test_controls,
    }


def generate_validation_plots(
    model: OptFedModel,
    train_data: ProcessData,
    test_data: ProcessData,
    controls: dict,
    save_dir: Path,
):
    """Generate all validation plots."""
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Generating Validation Plots ===")

    # 1. Parity plots for each variable
    for var in ["X", "P", "V"]:
        print(f"  Creating parity plot for {var}...")
        fig = plot_parity(model, train_data, controls, variable=var, show=False)
        fig.savefig(save_dir / f"parity_{var}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. Validation summary (4-panel plot)
    print("  Creating validation summary...")
    fig = plot_validation_summary(model, train_data, test_data, controls, show=False)
    fig.savefig(save_dir / "validation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Multi-run comparison for training data
    print("  Creating training run comparison...")
    fig = plot_multi_run_comparison(
        model,
        {rid: train_data.runs[rid] for rid in list(train_data.runs.keys())[:3]},
        {rid: controls[rid] for rid in list(train_data.runs.keys())[:3]},
        show=False,
    )
    fig.savefig(save_dir / "train_runs_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Multi-run comparison for test data
    print("  Creating test run comparison...")
    fig = plot_multi_run_comparison(model, test_data.runs, controls, show=False)
    fig.savefig(save_dir / "test_runs_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Individual run plots for each test run
    for run_id, run in test_data.runs.items():
        print(f"  Creating plot for {run_id}...")
        fig = plot_model_vs_data(
            model, run, controls[run_id], show=False
        )
        fig.savefig(save_dir / f"{run_id}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[OK] All plots saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Validate OptFed model")
    parser.add_argument(
        "--data",
        type=str,
        default="../data/traintest_compiled.xlsx",
        help="Path to training data Excel file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/",
        help="Path to results directory",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip model selection, use fitted parameters as-is",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for F-test (default: 0.05)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OptFed Model Validation")
    print("=" * 60)

    # 1. Load data
    print("\n=== Loading Data ===")
    train_data, test_data = load_traintest_data(args.data)
    print(f"[OK] Training runs: {list(train_data.runs.keys())}")
    print(f"[OK] Test runs: {list(test_data.runs.keys())}")

    # 2. Load fitted parameters
    print("\n=== Loading Fitted Parameters ===")
    params = load_fitted_params(args.results_dir)

    # 3. Model selection (optional)
    if not args.skip_selection:
        print("\n=== Running Model Selection ===")
        # Create model with fitted parameters for model selection
        model_for_selection = create_model_erying()
        model_for_selection.params = params
        selector = ModelSelector(model_for_selection, train_data, FitConfig())
        selected_params = selector.select_model(alpha=args.alpha)

        # Save selected parameters
        save_path = Path(args.results_dir) / "selected_params.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(selector.estimator.fit_result, f)
        print(f"[OK] Selected parameters saved to {save_path}")

        # Use selected parameters for validation
        params = selected_params
    else:
        print("\n=== Skipping Model Selection ===")
        print("Using fitted parameters as-is")

    # 4. Run validation
    validation_results = run_validation(
        params, train_data, test_data, Path(args.results_dir) / "validation"
    )

    # 5. Create model with final parameters
    model = create_model_erying()
    model.params = params

    # 6. Generate plots
    all_controls = {**validation_results["train_controls"], **validation_results["test_controls"]}
    generate_validation_plots(
        model,
        train_data,
        test_data,
        all_controls,
        Path(args.results_dir) / "validation" / "plots",
    )

    # 7. Save validation metrics
    import json
    metrics_path = Path(args.results_dir) / "validation" / "metrics.json"
    with open(metrics_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        json.dump(convert_types(validation_results), f, indent=2)
    print(f"\n[OK] Validation metrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {Path(args.results_dir) / 'validation'}/")


if __name__ == "__main__":
    main()
