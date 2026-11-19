"""
Symbolic Regression for LDH Protein Formation - IMPROVED VERSION

Key improvements based on feedback:
1. phi from measured concentrations (not clipped rates) → continuous values
2. q_p,tot from total protein balance (less noise)
3. Filter out intervals with tiny q_p,tot (meaningless phi)
4. For q_p,tot model: remove division operator to avoid spurious poles
5. Leave-one-run-out validation
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
# Current structure: USP/modeling_playground/LDH/symbolic_regression/interval_averaged_sr_v2.py
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

# Filtering thresholds
QP_TOT_MIN = 1e-4  # Minimum q_p,tot for phi to be meaningful


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
# IMPROVED INTERVAL-AVERAGED RATE CALCULATION
# ============================================================

def build_interval_rows_improved(run_name: str, n_grid: int = 40) -> List[Dict]:
    """
    Build rows with interval-averaged rates - IMPROVED VERSION.

    Key changes:
    1. phi from measured concentrations (continuous, not 0/1)
    2. q_p,tot from total protein (P_sol + P_IB) balance → less noise
    3. Filter intervals with tiny q_p,tot

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

    # Feed rate F(t)
    dt = np.diff(t_all)
    dt = np.where(dt <= 0, np.nan, dt)
    dfeed = np.diff(feed_cum_L)
    F_mid = dfeed / dt  # L/h between points
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

    # Total protein at sample points
    Ptot = Ps + Pi

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

        # Linear interpolation of X between sample endpoints
        Xg = np.interp(tg, [t0, t1], [Xs[k], Xs[k + 1]])

        # Total protein interpolation (for q_p,tot balance)
        Ptot_g = np.interp(tg, [t0, t1], [Ptot[k], Ptot[k + 1]])

        Fg = F(tg)
        Vg = np.maximum(V(tg), EPS)
        Dg = Fg / Vg
        Tg = T(tg)

        # Substrate uptake rate: qs(t) = F*C_feed/(X*V)
        qs_g = (Fg * C_FEED) / (np.maximum(Xg, EPS) * Vg)

        # ============================================================
        # NEW: Compute q_p,tot from TOTAL PROTEIN balance (less noise)
        # ============================================================
        # dP_tot/dt = q_p,tot * X - D * P_tot
        # Integrated: q_p,tot = (ΔP_tot + ∫D*P_tot*dt) / ∫X*dt

        intX = np.trapezoid(Xg, tg)
        intDP_tot = np.trapezoid(Dg * Ptot_g, tg)
        dPtot = Ptot[k + 1] - Ptot[k]

        qp_tot = (dPtot + intDP_tot) / np.maximum(intX, EPS)

        # Basic sanity - rates can't be negative
        qp_tot = max(qp_tot, 0.0)

        # ============================================================
        # NEW: phi from measured CONCENTRATIONS (continuous!)
        # ============================================================
        # phi(t) = P_IB / (P_sol + P_IB) at the END of the interval
        # Using interval-end value (t1) as representative
        phi_ib = Pi[k + 1] / np.maximum(Ps[k + 1] + Pi[k + 1], EPS)

        # Filter: if q_p,tot is tiny, phi is meaningless
        if qp_tot < QP_TOT_MIN:
            continue

        # Interval features (averages)
        T_bar = float(np.trapezoid(Tg, tg) / (t1 - t0))
        qs_bar = float(np.trapezoid(qs_g, tg) / (t1 - t0))

        # mu interval estimate: μ ≈ (ln X1 - ln X0)/dt + D̄
        D_bar = float(np.trapezoid(Dg, tg) / (t1 - t0))
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
            "qp_tot": qp_tot,
            "phi_ib": phi_ib,
            "P_sol_end": Ps[k + 1],
            "P_IB_end": Pi[k + 1],
            "P_tot_end": Ptot[k + 1],
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
            rows = build_interval_rows_improved(r)
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
# SYMBOLIC REGRESSION (IMPROVED)
# ============================================================

def fit_sr_models_improved(df: pd.DataFrame, feat_cols: List[str] = None):
    """
    Fit symbolic regression models with improvements:
    - For q_p,tot: NO division operator (avoid spurious poles)
    - For phi: continuous values from concentrations

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
    # Now phi is continuous, not clipped to 0/1!
    phi_clipped = np.clip(df["phi_ib"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    y_logit_phi = np.log(phi_clipped / (1 - phi_clipped))

    if not PYSR_AVAILABLE:
        raise ImportError("PySR is not available. Install with: pip install pysr")

    # Model for log(qp_tot) - NO DIVISION to avoid spurious poles
    model_qptot = PySRRegressor(
        niterations=200,
        maxsize=12,
        model_selection="best",
        binary_operators=["+", "-", "*"],  # NO DIVISION!
        unary_operators=["exp", "log", "tanh"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=42,
    )

    # Model for logit(phi_ib) - keep division here (it's well-behaved for phi)
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
    print("FITTING LOG(QP_TOT) MODEL (NO DIVISION OPERATOR)")
    print("=" * 60)
    model_qptot.fit(X, y_log_qptot)

    print("\n" + "=" * 60)
    print("FITTING LOGIT(PHI_IB) MODEL (CONTINUOUS phi FROM CONC)")
    print("=" * 60)
    model_phi.fit(X, y_logit_phi)

    return feat_cols, model_qptot, model_phi


# ============================================================
# LEAVE-ONE-RUN-OUT VALIDATION
# ============================================================

def leave_one_run_out_validation(all_runs: List[str], feat_cols: List[str] = None):
    """
    Perform leave-one-run-out validation to test generalization.

    Parameters
    ----------
    all_runs : List[str]
        All runs to use in validation
    feat_cols : List[str]
        Feature columns to use
    """
    if feat_cols is None:
        feat_cols = ["T", "qs", "tau"]

    results = []

    for test_run in all_runs:
        train_runs = [r for r in all_runs if r != test_run]

        print(f"\n{'=' * 60}")
        print(f"LEAVE-ONE-OUT: Test on {test_run}, Train on {train_runs}")
        print(f"{'=' * 60}")

        # Build datasets
        df_train = build_dataset(train_runs)
        df_test = build_dataset([test_run])

        if len(df_train) < 3 or len(df_test) < 1:
            print(f"  Skipping: not enough data (train={len(df_train)}, test={len(df_test)})")
            continue

        print(f"  Train size: {len(df_train)}, Test size: {len(df_test)}")

        try:
            # Fit models
            X_train = df_train[feat_cols].to_numpy(dtype=float)
            X_test = df_test[feat_cols].to_numpy(dtype=float)

            y_train_log = np.log(df_train["qp_tot"].values + 1e-6)
            y_train_phi = np.log(np.clip(df_train["phi_ib"].values, 1e-6, 1 - 1e-6) /
                                (1 - np.clip(df_train["phi_ib"].values, 1e-6, 1 - 1e-6)))

            # Fit qp_tot model
            model_qp = PySRRegressor(
                niterations=100,  # Fewer iterations for speed
                maxsize=10,
                binary_operators=["+", "-", "*"],  # No division
                unary_operators=["exp", "log", "tanh"],
                random_state=42,
            )
            model_qp.fit(X_train, y_train_log)

            # Fit phi model
            model_phi = PySRRegressor(
                niterations=100,
                maxsize=8,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "tanh"],
                random_state=42,
            )
            model_phi.fit(X_train, y_train_phi)

            # Predict on test set
            qp_pred = np.exp(model_qp.predict(X_test)) - 1e-6
            qp_pred = np.maximum(qp_pred, 0)
            qp_true = df_test["qp_tot"].values

            phi_pred = 1 / (1 + np.exp(-model_phi.predict(X_test)))
            phi_true = df_test["phi_ib"].values

            # Metrics
            r2_qp = 1 - np.sum((qp_true - qp_pred)**2) / np.sum((qp_true - np.mean(qp_true))**2)
            rmse_qp = np.sqrt(np.mean((qp_true - qp_pred)**2))

            r2_phi = 1 - np.sum((phi_true - phi_pred)**2) / np.sum((phi_true - np.mean(phi_true))**2)
            rmse_phi = np.sqrt(np.mean((phi_true - phi_pred)**2))

            print(f"\n  Results for {test_run}:")
            print(f"    qp_tot: R²={r2_qp:.3f}, RMSE={rmse_qp:.6f}")
            print(f"    phi_ib: R²={r2_phi:.3f}, RMSE={rmse_phi:.6f}")

            results.append({
                "test_run": test_run,
                "train_runs": ",".join(train_runs),
                "r2_qp": r2_qp,
                "rmse_qp": rmse_qp,
                "r2_phi": r2_phi,
                "rmse_phi": rmse_phi,
                "n_train": len(df_train),
                "n_test": len(df_test),
            })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Summary
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("LEAVE-ONE-OUT VALIDATION SUMMARY")
        print("=" * 60)
        print(df_results.to_string(index=False))

        # Save results
        df_results.to_csv(RESULTS_PATH / "leave_one_out_results.csv", index=False)
        print(f"\nResults saved to {RESULTS_PATH / 'leave_one_out_results.csv'}")

        return df_results
    else:
        print("\nNo valid results from leave-one-out validation")
        return None


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

    # Metrics
    mse_qp = np.mean((qp_tot_true - qp_tot_pred)**2)
    rmse_qp = np.sqrt(mse_qp)
    r2_qp = 1 - np.sum((qp_tot_true - qp_tot_pred)**2) / np.sum((qp_tot_true - np.mean(qp_tot_true))**2)

    mse_phi = np.mean((phi_ib_true - phi_ib_pred)**2)
    rmse_phi = np.sqrt(mse_phi)
    r2_phi = 1 - np.sum((phi_ib_true - phi_ib_pred)**2) / np.sum((phi_ib_true - np.mean(phi_ib_true))**2)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nqp_tot model (NO division operator):")
    print(f"  R²:   {r2_qp:.4f}")
    print(f"  RMSE: {rmse_qp:.6f}")

    print(f"\nphi_ib model (continuous phi from concentrations):")
    print(f"  R²:   {r2_phi:.4f}")
    print(f"  RMSE: {rmse_phi:.6f}")

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
    plt.savefig(RESULTS_PATH / 'interval_sr_evaluation_v2.png', dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plot saved to {RESULTS_PATH / 'interval_sr_evaluation_v2.png'}")

    # Plot phi_ib distribution to show it's now continuous
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(df["phi_ib"].values, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel("phi_IB")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of phi_IB (should be continuous now!)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for run in df["run"].unique():
        mask = df["run"] == run
        ax.scatter(df.loc[mask, "T"], df.loc[mask, "phi_ib"],
                  label=run, alpha=0.7, s=50)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("IB Fraction (phi_IB)")
    ax.set_title("IB Fraction vs Temperature (continuous values)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'phi_continuous_v2.png', dpi=150, bbox_inches='tight')
    print(f"phi plots saved to {RESULTS_PATH / 'phi_continuous_v2.png'}")

    plt.show()


def save_equations(model_qptot: 'PySRRegressor', model_phi: 'PySRRegressor',
                   feat_cols: List[str]):
    """Save the discovered equations to a file."""
    with open(RESULTS_PATH / 'interval_sr_equations_v2.txt', 'w') as f:
        f.write("Symbolic Regression Results v2 (IMPROVED)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Key improvements:\n")
        f.write("  1. phi from measured concentrations (continuous, not 0/1)\n")
        f.write("  2. q_p,tot from total protein balance (less noise)\n")
        f.write("  3. Filter intervals with q_p,tot < 1e-4\n")
        f.write("  4. NO division operator for q_p,tot model (no spurious poles)\n\n")

        f.write("Features:\n")
        f.write(f"  {', '.join(feat_cols)}\n\n")

        f.write("log(qp_tot) equation (NO DIVISION!):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{model_qptot.sympy()}\n\n")
        f.write("Interpretation: qp_tot = exp(equation) - 1e-6\n\n")

        f.write("logit(phi_ib) equation (continuous phi):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{model_phi.sympy()}\n\n")
        f.write("Interpretation: phi_ib = sigmoid(equation) = 1/(1+exp(-equation))\n\n")

        f.write("Usage:\n")
        f.write("-" * 60 + "\n")
        f.write("qp_IB = phi_ib * qp_tot\n")
        f.write("qp_sol = (1 - phi_ib) * qp_tot\n")

    print(f"\nEquations saved to {RESULTS_PATH / 'interval_sr_equations_v2.txt'}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("Symbolic Regression v2: IMPROVED METHOD")
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

    print("\nBuilding IMPROVED interval-averaged dataset...")
    print("Changes:")
    print("  1. phi from measured concentrations (not clipped rates)")
    print("  2. q_p,tot from total protein balance (less noise)")
    print("  3. Filter intervals with q_p,tot < 1e-4")

    df = build_dataset(train_runs)
    print(f"\nDataset shape: {df.shape}")
    print(f"Samples per run:")
    print(df.groupby("run").size())

    if len(df) < 5:
        print("\nERROR: Not enough samples for symbolic regression")
        return

    print("\nDataset summary:")
    print(df.describe())

    # Check that phi is now continuous
    print("\n" + "=" * 60)
    print("CHECKING phi DISTRIBUTION")
    print("=" * 60)
    phi_values = df["phi_ib"].values
    print(f"phi min: {phi_values.min():.4f}, max: {phi_values.max():.4f}")
    print(f"Fraction at exactly 0: {(phi_values == 0).sum() / len(phi_values):.2%}")
    print(f"Fraction at exactly 1: {(phi_values == 1).sum() / len(phi_values):.2%}")
    print("Should be mostly continuous now!")

    # Fit models
    print("\n" + "=" * 60)
    print("FITTING SYMBOLIC REGRESSION MODELS (IMPROVED)")
    print("=" * 60)

    feat_cols, model_qptot, model_phi = fit_sr_models_improved(df)

    # Evaluate
    evaluate_models(model_qptot, model_phi, df, feat_cols)

    # Save equations
    print("\n" + "=" * 60)
    print("DISCOVERED EQUATIONS (IMPROVED)")
    print("=" * 60)

    print(f"\nBest log(qp_tot) equation (NO DIVISION):")
    print(f"  {model_qptot.sympy()}")
    print(f"  (so: qp_tot = exp(equation) - 1e-6)")

    print(f"\nBest logit(phi_ib) equation (continuous phi):")
    print(f"  {model_phi.sympy()}")
    print(f"  (so: phi_ib = sigmoid(equation) = 1/(1+exp(-equation)))")

    save_equations(model_qptot, model_phi, feat_cols)

    # Leave-one-run-out validation
    print("\n" + "=" * 60)
    print("LEAVE-ONE-RUN-OUT VALIDATION")
    print("=" * 60)

    leave_one_run_out_validation(train_runs, feat_cols)

    print("\n" + "=" * 60)
    print("Done! Check the results folder:")
    print(f"  {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
