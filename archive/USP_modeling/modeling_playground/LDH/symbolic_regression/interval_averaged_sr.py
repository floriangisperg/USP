"""
Symbolic Regression for LDH Protein Formation using Interval-Averaged Rates

This script implements the "Option A" approach from the detailed feedback:
- Uses interval-averaged q_p estimates (integrated mass balances, no derivatives)
- Correct mu calculation with proper units (1/h)
- Dilution rate from feed rate (D = F/V)
- Models log(qp_tot) and logit(phi_ib) for physical constraints

Key improvements over previous attempts:
1. No numerical differentiation of sparse/noisy measurements
2. Correct fed-batch mass balances
3. Proper PySR API (binary_operators/unary_operators)
4. Physically-meaningful target transforms
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Try to import PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not available. Install with: pip install pysr")

# ============================================================
# CONSTANTS AND CONFIGURATION
# ============================================================

# Path to consolidated data - needs to go from modeling_playground/LDH/symbolic_regression/ to repo root
# Current structure: USP/modeling_playground/LDH/symbolic_regression/interval_averaged_sr.py
# We need: USP/examples/LDH/LDH_consolidated_results.xlsx
# Going up 4 levels: symbolic_regression -> LDH -> modeling_playground -> USP
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = REPO_ROOT / "examples" / "LDH" / "LDH_consolidated_results.xlsx"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# DoE conditions for each run
DOE_CONDITIONS = {
    "ferm_01": {"qs_setpoint": 0.2, "T_induced": 35.0},
    "ferm_02": {"qs_setpoint": 0.2, "T_induced": 25.0},
    "ferm_03": {"qs_setpoint": 0.6, "T_induced": 25.0},
    "ferm_04": {"qs_setpoint": 0.6, "T_induced": 35.0},
    "ferm_05": {"qs_setpoint": 0.4, "T_induced": 30.0},
    "ferm_06": {"qs_setpoint": 0.4, "T_induced": 30.0},
    "ferm_07": {"qs_setpoint": 0.4, "T_induced": 30.0},
}

# Physical constants
C_FEED = 400.0  # g/L - feed concentration
T_FEDBATCH = 37.0  # °C - temperature before induction
EPS = 1e-9  # Small number to avoid division by zero


# ============================================================
# DATA LOADING
# ============================================================

def load_run(run_name: str) -> pd.DataFrame:
    """Load all data for a single run from consolidated Excel file."""
    df_all = pd.read_excel(DATA_PATH, sheet_name="All_Time_Points")
    df = df_all[df_all["run_name"] == run_name].copy()
    if df.empty:
        raise ValueError(f"No data for {run_name}")
    df = df.sort_values("Process time [h]")
    return df


# ============================================================
# INTERVAL-AVERAGED RATE CALCULATION (NO DERIVATIVES)
# ============================================================

def build_interval_rows(run_name: str, n_grid: int = 40) -> List[Dict]:
    """
    Build rows with interval-averaged q_p for soluble + IB using integrated mass balances.

    This uses the integrated mass balance over each sampling interval [t_i, t_{i+1}]:
        P_{i+1} - P_i = ∫(q_p * X - D*P) dt

    Solving for the interval-average q_p:
        q̄_p = (ΔP + ∫D*P*dt) / ∫X*dt

    This is MUCH more stable with sparse data than numerical differentiation.

    Parameters
    ----------
    run_name : str
        Name of the fermentation run
    n_grid : int
        Number of grid points for integration within each interval

    Returns
    -------
    List[Dict]
        List of dictionaries with interval-averaged features and targets
    """
    df = load_run(run_name)

    t_all = df["Process time [h]"].to_numpy(dtype=float)

    # Induction + temperature profile
    I_all = df["Is induced [-]"].fillna(0).to_numpy(dtype=float)
    T_ind = DOE_CONDITIONS[run_name]["T_induced"]
    T_all = np.where(I_all > 0.5, T_ind, T_FEDBATCH)

    # Volume profile
    vol_col = "Volume corrected [mL]" if "Volume corrected [mL]" in df.columns else "Volume [mL]"
    V_all = (pd.Series(df[vol_col].to_numpy(dtype=float))
             .ffill().bfill().to_numpy()) / 1000.0  # Convert mL to L

    # Cumulative feed (mL) -> L
    feed_cum_L = (df["Feed [mL]"].fillna(0).to_numpy(dtype=float)) / 1000.0

    # Feed rate F(t): stable finite-difference on the dense log timeline
    dt = np.diff(t_all)
    dt = np.where(dt <= 0, np.nan, dt)
    dfeed = np.diff(feed_cum_L)
    F_mid = dfeed / dt  # L/h between points
    # Define F on t_all by padding endpoints
    F_all = np.concatenate([[F_mid[0]], F_mid])
    F_all = np.nan_to_num(F_all, nan=0.0)
    F_all = np.clip(F_all, 0.0, None)

    # Interpolators for controls
    F = interp1d(t_all, F_all, kind="linear", bounds_error=False,
                 fill_value=(F_all[0], F_all[-1]))
    V = interp1d(t_all, V_all, kind="linear", bounds_error=False,
                 fill_value=(V_all[0], V_all[-1]))
    T = interp1d(t_all, T_all, kind="linear", bounds_error=False,
                 fill_value=(T_all[0], T_all[-1]))
    I = interp1d(t_all, I_all, kind="linear", bounds_error=False,
                 fill_value=(I_all[0], I_all[-1]))

    # Sample table (sparse measurements)
    samples = df[df["Biomass [g/L]"].notna()].copy()
    ts = samples["Process time [h]"].to_numpy(dtype=float)

    Xs = samples["Biomass [g/L]"].to_numpy(dtype=float)
    Ps = samples["Protein soluble [g/L]"].to_numpy(dtype=float) \
        if "Protein soluble [g/L]" in samples.columns else None
    Pi = samples["Protein IB [g/L]"].to_numpy(dtype=float) \
        if "Protein IB [g/L]" in samples.columns else None

    if Ps is None or Pi is None:
        raise ValueError("Need both Protein soluble and Protein IB columns.")

    # Induction time estimate
    ind_idx = np.where(I_all > 0.5)[0]
    t_ind = t_all[ind_idx[0]] if len(ind_idx) > 0 else np.nan

    rows = []
    for k in range(len(ts) - 1):
        t0, t1 = ts[k], ts[k + 1]
        if not np.isfinite(t0) or not np.isfinite(t1) or (t1 <= t0):
            continue

        # Only use induced intervals
        I_mid = float(I(0.5 * (t0 + t1)))
        if I_mid <= 0.5:
            continue

        # Build a small grid inside the interval for integration
        tg = np.linspace(t0, t1, n_grid)

        # Linear interpolation of X and P between sample endpoints (stable!)
        Xg = np.interp(tg, [t0, t1], [Xs[k], Xs[k + 1]])
        Psol_g = np.interp(tg, [t0, t1], [Ps[k], Ps[k + 1]])
        PIB_g = np.interp(tg, [t0, t1], [Pi[k], Pi[k + 1]])

        Fg = F(tg)
        Vg = np.maximum(V(tg), EPS)
        Dg = Fg / Vg
        Tg = T(tg)

        # Substrate uptake rate: qs(t) = F*C_feed/(X*V)
        qs_g = (Fg * C_FEED) / (np.maximum(Xg, EPS) * Vg)

        # Integrals
        intX = np.trapz(Xg, tg)

        intDP_sol = np.trapz(Dg * Psol_g, tg)
        intDP_ib = np.trapz(Dg * PIB_g, tg)

        dPsol = Ps[k + 1] - Ps[k]
        dPIB = Pi[k + 1] - Pi[k]

        # Interval-averaged formation rates from integrated balance
        qp_sol = (dPsol + intDP_sol) / np.maximum(intX, EPS)
        qp_ib = (dPIB + intDP_ib) / np.maximum(intX, EPS)

        # Basic sanity - rates can't be negative
        qp_sol = max(qp_sol, 0.0)
        qp_ib = max(qp_ib, 0.0)

        qp_tot = qp_sol + qp_ib
        phi_ib = qp_ib / np.maximum(qp_tot, EPS)

        # Interval features (averages)
        T_bar = float(np.trapz(Tg, tg) / (t1 - t0))
        qs_bar = float(np.trapz(qs_g, tg) / (t1 - t0))

        # mu interval estimate: μ ≈ (ln X1 - ln X0)/dt + D̄
        D_bar = float(np.trapz(Dg, tg) / (t1 - t0))
        mu_bar = float((np.log(max(Xs[k + 1], EPS)) - np.log(max(Xs[k], EPS))) / (t1 - t0) + D_bar)

        # Time since induction
        tau = float(0.5 * (t0 + t1) - t_ind) if np.isfinite(t_ind) else np.nan

        rows.append({
            "run": run_name,
            "t_mid": 0.5 * (t0 + t1),
            "tau": tau,
            "T": T_bar,
            "qs": qs_bar,
            "mu": mu_bar,
            "qp_sol": qp_sol,
            "qp_ib": qp_ib,
            "qp_tot": qp_tot,
            "phi_ib": phi_ib,
        })

    return rows


def build_dataset(runs: List[str]) -> pd.DataFrame:
    """
    Build the complete SR dataset from multiple runs.

    Parameters
    ----------
    runs : List[str]
        List of run names to include

    Returns
    -------
    pd.DataFrame
        DataFrame with interval-averaged features and targets
    """
    all_rows = []
    for r in runs:
        print(f"  Processing {r}...")
        try:
            rows = build_interval_rows(r)
            all_rows.extend(rows)
            print(f"    Got {len(rows)} intervals")
        except Exception as e:
            print(f"    Error: {e}")

    df = pd.DataFrame(all_rows)

    # Filter obvious numerical garbage
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["qp_tot"] >= 0) & (df["qp_tot"] < 10)]  # Reasonable range

    return df


# ============================================================
# SYMBOLIC REGRESSION
# ============================================================

def fit_sr_models(df: pd.DataFrame, feat_cols: List[str] = None):
    """
    Fit symbolic regression models for:
    - log(qp_tot): ensures positivity (qp_tot = exp(prediction))
    - logit(phi_ib): ensures [0,1] bounds (phi = sigmoid(prediction))

    Parameters
    ----------
    df : pd.DataFrame
        Training dataset with interval-averaged features and targets
    feat_cols : List[str]
        Feature columns to use (default: ["T", "qs", "tau"])

    Returns
    -------
    Tuple
        (feature_names, model_qptot, model_phi)
    """
    if feat_cols is None:
        feat_cols = ["T", "qs", "tau"]  # Keep it low-dimensional first

    X = df[feat_cols].to_numpy(dtype=float)

    # Target 1: log(qp_tot) for positivity constraint
    y_log_qptot = np.log(df["qp_tot"].to_numpy(dtype=float) + 1e-6)

    # Target 2: logit(phi_ib) for [0,1] constraint
    # logit(p) = log(p / (1-p))
    phi_clipped = np.clip(df["phi_ib"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    y_logit_phi = np.log(phi_clipped / (1 - phi_clipped))

    if not PYSR_AVAILABLE:
        raise ImportError("PySR is not available. Install with: pip install pysr")

    # Model for log(qp_tot)
    model_qptot = PySRRegressor(
        niterations=200,
        maxsize=12,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "tanh"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=42,
    )

    # Model for logit(phi_ib)
    model_phi = PySRRegressor(
        niterations=200,
        maxsize=10,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "tanh"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=42,
    )

    print("\n" + "=" * 60)
    print("FITTING LOG(QP_TOT) MODEL")
    print("=" * 60)
    model_qptot.fit(X, y_log_qptot)

    print("\n" + "=" * 60)
    print("FITTING LOGIT(PHI_IB) MODEL")
    print("=" * 60)
    model_phi.fit(X, y_logit_phi)

    return feat_cols, model_qptot, model_phi


# ============================================================
# MODEL EVALUATION AND PLOTTING
# ============================================================

def evaluate_models(
    model_qptot: 'PySRRegressor',
    model_phi: 'PySRRegressor',
    df: pd.DataFrame,
    feat_cols: List[str],
):
    """Evaluate and plot the fitted models."""
    X = df[feat_cols].to_numpy(dtype=float)

    # Predict on training data
    y_log_qp_pred = model_qptot.predict(X)
    y_logit_phi_pred = model_phi.predict(X)

    # Transform back to original scale
    qp_tot_true = df["qp_tot"].values
    qp_tot_pred = np.exp(y_log_qp_pred) - 1e-6
    qp_tot_pred = np.maximum(qp_tot_pred, 0)

    phi_ib_true = df["phi_ib"].values
    phi_ib_pred = 1 / (1 + np.exp(-y_logit_phi_pred))  # sigmoid

    # Metrics for qp_tot
    mse_qp = np.mean((qp_tot_true - qp_tot_pred)**2)
    rmse_qp = np.sqrt(mse_qp)
    mae_qp = np.mean(np.abs(qp_tot_true - qp_tot_pred))
    r2_qp = 1 - np.sum((qp_tot_true - qp_tot_pred)**2) / np.sum((qp_tot_true - np.mean(qp_tot_true))**2)

    # Metrics for phi_ib
    mse_phi = np.mean((phi_ib_true - phi_ib_pred)**2)
    rmse_phi = np.sqrt(mse_phi)
    mae_phi = np.mean(np.abs(phi_ib_true - phi_ib_pred))
    r2_phi = 1 - np.sum((phi_ib_true - phi_ib_pred)**2) / np.sum((phi_ib_true - np.mean(phi_ib_true))**2)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nqp_tot model (log space -> exp transform):")
    print(f"  R²:   {r2_qp:.4f}")
    print(f"  RMSE: {rmse_qp:.6f}")
    print(f"  MAE:  {mae_qp:.6f}")

    print(f"\nphi_ib model (logit space -> sigmoid):")
    print(f"  R²:   {r2_phi:.4f}")
    print(f"  RMSE: {rmse_phi:.6f}")
    print(f"  MAE:  {mae_phi:.6f}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # qp_tot parity plot
    ax = axes[0, 0]
    ax.scatter(qp_tot_true, qp_tot_pred, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    min_val = min(qp_tot_true.min(), qp_tot_pred.min())
    max_val = max(qp_tot_true.max(), qp_tot_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True qp_tot')
    ax.set_ylabel('Predicted qp_tot')
    ax.set_title(f'Total Protein Formation Rate (R²={r2_qp:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # qp_tot residuals
    ax = axes[0, 1]
    residuals_qp = qp_tot_true - qp_tot_pred
    ax.scatter(qp_tot_pred, residuals_qp, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted qp_tot')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals Plot')
    ax.grid(True, alpha=0.3)

    # phi_ib parity plot
    ax = axes[1, 0]
    ax.scatter(phi_ib_true, phi_ib_pred, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    min_val = min(phi_ib_true.min(), phi_ib_pred.min())
    max_val = max(phi_ib_true.max(), phi_ib_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True phi_IB')
    ax.set_ylabel('Predicted phi_IB')
    ax.set_title(f'IB Fraction (R²={r2_phi:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # phi_ib residuals
    ax = axes[1, 1]
    residuals_phi = phi_ib_true - phi_ib_pred
    ax.scatter(phi_ib_pred, residuals_phi, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted phi_IB')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals Plot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'interval_sr_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plot saved to {RESULTS_PATH / 'interval_sr_evaluation.png'}")

    # Plot phi_ib vs T to see temperature dependence
    fig, ax = plt.subplots(figsize=(8, 6))
    for run in df["run"].unique():
        mask = df["run"] == run
        ax.scatter(df.loc[mask, "T"], df.loc[mask, "phi_ib"],
                  label=run, alpha=0.7, s=50)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("IB Fraction (phi_IB)")
    ax.set_title("IB Fraction vs Temperature (raw data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'phi_vs_temperature.png', dpi=150, bbox_inches='tight')
    print(f"Temperature plot saved to {RESULTS_PATH / 'phi_vs_temperature.png'}")

    plt.show()


def save_equations(model_qptot: 'PySRRegressor', model_phi: 'PySRRegressor',
                   feat_cols: List[str]):
    """Save the discovered equations to a file."""
    with open(RESULTS_PATH / 'interval_sr_equations.txt', 'w') as f:
        f.write("Symbolic Regression Results (Interval-Averaged Method)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Features:\n")
        f.write(f"  {', '.join(feat_cols)}\n\n")

        f.write("log(qp_tot) equation:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{model_qptot.sympy()}\n\n")
        f.write("Interpretation: qp_tot = exp(equation) - 1e-6\n\n")

        f.write("logit(phi_ib) equation:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{model_phi.sympy()}\n\n")
        f.write("Interpretation: phi_ib = sigmoid(equation) = 1/(1+exp(-equation))\n\n")

        f.write("Usage:\n")
        f.write("-" * 60 + "\n")
        f.write("qp_IB = phi_ib * qp_tot\n")
        f.write("qp_sol = (1 - phi_ib) * qp_tot\n")

    print(f"\nEquations saved to {RESULTS_PATH / 'interval_sr_equations.txt'}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("Symbolic Regression: Interval-Averaged Method")
    print("=" * 60)

    # Check PySR availability
    if not PYSR_AVAILABLE:
        print("\nERROR: PySR is not installed!")
        print("\nTo install PySR, run:")
        print("  pip install pysr")
        print("\nNote: PySR also requires Julia to be installed.")
        print("See: https://github.com/MilesCranmer/PySR")
        return

    # Use corner point runs for training
    train_runs = ["ferm_01", "ferm_02", "ferm_03", "ferm_04"]
    print(f"\nUsing {train_runs} as training dataset...")

    print("\nBuilding interval-averaged dataset...")
    df = build_dataset(train_runs)
    print(f"\nDataset shape: {df.shape}")
    print(f"Samples per run:")
    print(df.groupby("run").size())

    if len(df) < 5:
        print("\nERROR: Not enough samples for symbolic regression")
        return

    print("\nDataset summary:")
    print(df.describe())

    # Fit models
    print("\n" + "=" * 60)
    print("FITTING SYMBOLIC REGRESSION MODELS")
    print("=" * 60)

    feat_cols, model_qptot, model_phi = fit_sr_models(df)

    # Evaluate
    evaluate_models(model_qptot, model_phi, df, feat_cols)

    # Save equations
    print("\n" + "=" * 60)
    print("DISCOVERED EQUATIONS")
    print("=" * 60)

    print(f"\nBest log(qp_tot) equation:")
    print(f"  {model_qptot.sympy()}")
    print(f"  (so: qp_tot = exp(equation) - 1e-6)")

    print(f"\nBest logit(phi_ib) equation:")
    print(f"  {model_phi.sympy()}")
    print(f"  (so: phi_ib = sigmoid(equation) = 1/(1+exp(-equation)))")

    save_equations(model_qptot, model_phi, feat_cols)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Review the equations above - do they make physical sense?")
    print("2. Check the phi_vs_temperature.png plot")
    print("3. Try leave-one-run-out validation:")
    print("   - Train on ferm_01, ferm_02, ferm_03")
    print("   - Test on ferm_04")
    print("4. If equations look good, we can implement them in a hybrid model")

    print("\nDone!")


if __name__ == "__main__":
    main()
