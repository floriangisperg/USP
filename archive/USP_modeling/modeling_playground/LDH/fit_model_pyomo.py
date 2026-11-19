"""
Simultaneous Approach for Training Neural DAEs (Pyomo + IPOPT)
Reference: arXiv:2504.04665 "A Simultaneous Approach for Training Neural Differential-Algebraic Systems of Equations"

This script implements the Simultaneous Approach using Pyomo (DAE) and IPOPT.
It models the LDH fermentation as a Differential-Algebraic Equation (DAE) system.
Global parameters are shared across multiple experimental runs (Blocks).

Requires:
- Pyomo (`pip install pyomo`)
- IPOPT executable in system PATH
"""

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ============================================================
# DATA & CONSTANTS
# ============================================================

DATA_PATH = Path(__file__).parent / "data" / "LDH_consolidated_results.xlsx"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# Constants
C_FEED = 400.0
T_FEDBATCH = 37.0
T_OPT = 30.0
T_WIDTH = 10.0

DOE_CONDITIONS = {
    "ferm_01": {"qs_setpoint": 0.2, "T_induced": 35.0},
    "ferm_02": {"qs_setpoint": 0.2, "T_induced": 25.0},
    "ferm_03": {"qs_setpoint": 0.6, "T_induced": 25.0},
    "ferm_04": {"qs_setpoint": 0.6, "T_induced": 35.0},
}

def load_data(run_name):
    """Load and process data for a single run."""
    df_all = pd.read_excel(DATA_PATH, sheet_name='All_Time_Points')
    df = df_all[df_all['run_name'] == run_name].copy()
    if df.empty: raise ValueError(f"Run {run_name} not found")

    times = df['Process time [h]'].values
    
    # Calculate Feed Rate [L/h]
    feed_cum = df['Feed [mL]'].fillna(0).values / 1000.0
    dt_raw = np.diff(times, prepend=times[0])
    dt_raw = np.maximum(dt_raw, 0.01)
    feed_rate = np.diff(feed_cum, prepend=0) / dt_raw
    
    # Inputs
    cond = DOE_CONDITIONS[run_name]
    is_induced = df['Is induced [-]'].fillna(0).values.astype(float)
    T_vals = np.where(is_induced > 0.5, cond["T_induced"], T_FEDBATCH)
    
    # Measurements (samples) - Filter specific sample points
    samples = df[df['Biomass [g/L]'].notna()].copy()
    meas = {
        'times': samples['Process time [h]'].values,
        'X': samples['Biomass [g/L]'].values,
        'P_sol': samples['Protein soluble [g/L]'].values,
        'P_IB': samples['Protein IB [g/L]'].values
    }
    
    # Interpolators for inputs (to be called by Pyomo model)
    # We use 'previous' kind for step inputs like Feed, or 'linear'
    # 'linear' is safer for DAE solvers to avoid discontinuities
    feed_interp = interp1d(times, feed_rate, kind='linear', fill_value="extrapolate")
    T_interp = interp1d(times, T_vals, kind='linear', fill_value="extrapolate")
    I_interp = interp1d(times, is_induced, kind='linear', fill_value="extrapolate")
    
    return {
        'run_name': run_name,
        't_start': times[0],
        't_end': times[-1],
        'V0': df['Volume [mL]'].iloc[0]/1000.0,
        'X0': meas['X'][0],
        'feed_fn': feed_interp,
        'T_fn': T_interp,
        'I_fn': I_interp,
        'meas': meas
    }

# ============================================================
# PYOMO MODEL
# ============================================================

def create_pyomo_model(data_list):
    """
    Creates a Pyomo ConcreteModel with multiple blocks (one per run).
    """
    m = pyo.ConcreteModel(name="LDH_Simultaneous")
    
    # --- Global Parameters (Optimization Variables) ---
    m.Y_XS = pyo.Var(bounds=(0.1, 1.0), initialize=0.5)
    m.m_s  = pyo.Var(bounds=(0.0, 0.1), initialize=0.01)
    m.alpha= pyo.Var(bounds=(0.0, 1.0), initialize=0.2)
    m.beta = pyo.Var(bounds=(0.0, 0.2), initialize=0.2)
    # Bounds strictly between experimental conditions (25 and 35) to prevent mode collapse
    m.T_50 = pyo.Var(bounds=(25.1, 34.9), initialize=30.0)
    m.k_T  = pyo.Var(bounds=(0.01, 5.0), initialize=0.5)
    m.k_base= pyo.Var(bounds=(0.0, 0.01), initialize=0.0001)
    
    # --- Scaling Factors (IPOPT) ---
    # Tell IPOPT to treat these variables as if they are on the same scale
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.Y_XS] = 1.0
    m.scaling_factor[m.alpha] = 10.0  # Scale up small parameters
    m.scaling_factor[m.beta] = 10.0
    m.scaling_factor[m.k_base] = 1000.0 # Scale up very small parameters
    
    # --- Function to build each run block ---
    def create_run_block(b, run_name):
        data = next(d for d in data_list if d['run_name'] == run_name)
        meas = data['meas']
        
        # 1. Time Set
        # Initialize with boundaries and measurement points to ensure perfect grid match
        t_init = np.sort(np.unique(np.concatenate([
            [data['t_start'], data['t_end']],
            meas['times']
        ])))
        b.t = ContinuousSet(initialize=t_init)
        
        # 2. State Variables
        b.X = pyo.Var(b.t, bounds=(0, 100), initialize=data['X0'])
        b.P_sol = pyo.Var(b.t, bounds=(0, 50), initialize=0)
        b.P_IB = pyo.Var(b.t, bounds=(0, 50), initialize=0)
        b.V = pyo.Var(b.t, bounds=(0.1, 10), initialize=data['V0'])
        
        # Derivatives
        b.dXdt = DerivativeVar(b.X, wrt=b.t)
        b.dP_soldt = DerivativeVar(b.P_sol, wrt=b.t)
        b.dP_IBdt = DerivativeVar(b.P_IB, wrt=b.t)
        b.dVdt = DerivativeVar(b.V, wrt=b.t)
        
        # Helper helpers (expressions)
        def get_inputs(t):
            # Pyomo can be finicky calling python functions in expressions
            # But inside rule it's simply constructing the expression graph
            # NOTE: We simply evaluate them as constants at each time point
            # This makes the DAE non-autonomous but explicit.
            return float(data['feed_fn'](t)), float(data['T_fn'](t)), float(data['I_fn'](t))

        # 3. ODE Constraints
        def dX_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            
            # qs = F * Cf / (X * V)
            # To avoid division by zero during solve, we can multiply through:
            # But derived variables are safer.
            qs = (F * C_FEED) / (X * V + 1e-6) 
            # Saturate assumption if needed (min/max hard in smooth optimizers)
            # Smooth min: qs_eff using Chen-Harp alg or generalized mean?
            # For IPOPT, let's trust bounds or use smooth approx if it crashes.
            # Simple soft constraint: qs_eff = qs (since F determines it)
            
            mu = m.model().Y_XS * qs - m.model().m_s
            # Monod-like smoothing usually safer, but model says 'max(..., 0)'
            # Smooth approximation of max(0, x): 0.5 * (x + sqrt(x^2 + eps))
            mu_safe = 0.5 * (mu + pyo.sqrt(mu**2 + 1e-8))
            
            F_base = m.model().k_base * X * V
            D = (F + F_base) / (V + 1e-6)
            
            return m.dXdt[t] == mu_safe * X - D * X
        b.ode_X = pyo.Constraint(b.t, rule=dX_rule)

        def dV_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            F_base = m.model().k_base * X * V
            return m.dVdt[t] == F + F_base
        b.ode_V = pyo.Constraint(b.t, rule=dV_rule)
        
        def dP_rule(m, t):
            # Combined for Psol and PIB
            # Need to re-calculate mu/qs here effectively or store as Intermediate vars
            # For brevity, recalculate
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            
            qs = (F * C_FEED) / (X * V + 1e-6)
            mu = m.model().Y_XS * qs - m.model().m_s
            mu_safe = 0.5 * (mu + pyo.sqrt(mu**2 + 1e-8))
            
            # f_T
            err = (T - T_OPT) / T_WIDTH
            f_T = pyo.exp(-0.5 * err**2)
            
            # q_p
            q_p = (m.model().alpha * mu_safe + m.model().beta) * I * f_T
            
            # phi_IB
            # sig = 1 / (1 + exp(-k(T-T50)))
            arg = -m.model().k_T * (T - m.model().T_50)
            phi_IB = 1.0 / (1.0 + pyo.exp(arg))
            
            F_base = m.model().k_base * X * V
            D = (F + F_base) / (V + 1e-6)
            
            return m.dP_soldt[t] == q_p * (1 - phi_IB) * X - D * m.P_sol[t]
        b.ode_Psol = pyo.Constraint(b.t, rule=dP_rule)
        
        def dPIB_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            qs = (F * C_FEED) / (X * V + 1e-6)
            mu = m.model().Y_XS * qs - m.model().m_s
            mu_safe = 0.5 * (mu + pyo.sqrt(mu**2 + 1e-8))
            err = (T - T_OPT) / T_WIDTH
            f_T = pyo.exp(-0.5 * err**2)
            q_p = (m.model().alpha * mu_safe + m.model().beta) * I * f_T
            arg = -m.model().k_T * (T - m.model().T_50)
            phi_IB = 1.0 / (1.0 + pyo.exp(arg))
            F_base = m.model().k_base * X * V
            D = (F + F_base) / (V + 1e-6)
            
            return m.dP_IBdt[t] == q_p * phi_IB * X - D * m.P_IB[t]
        b.ode_PIB = pyo.Constraint(b.t, rule=dPIB_rule)

        # 4. Initial Conditions (at t_start)
        # Note: Pyomo DAE doesn't auto-enforce ICs unless told
        def init_rule(m):
            yield m.X[data['t_start']] == data['X0']
            yield m.V[data['t_start']] == data['V0']
            yield m.P_sol[data['t_start']] == 0.0
            yield m.P_IB[data['t_start']] == 0.0
        b.init_con = pyo.ConstraintList(rule=init_rule)

    m.run_blocks = pyo.Block([d['run_name'] for d in data_list], rule=create_run_block)
    
    # --- Objective: Normalized Least Squares ---
    # Calculate max observed values for normalization
    all_X = np.concatenate([d['meas']['X'] for d in data_list])
    all_P_sol = np.concatenate([d['meas']['P_sol'] for d in data_list])
    all_P_IB = np.concatenate([d['meas']['P_IB'] for d in data_list])

    max_X = np.nanmax(all_X) if np.any(~np.isnan(all_X)) else 1.0
    max_P_sol = np.nanmax(all_P_sol) if np.any(~np.isnan(all_P_sol)) else 1.0
    max_P_IB = np.nanmax(all_P_IB) if np.any(~np.isnan(all_P_IB)) else 1.0

    # Avoid zero division
    max_X = max(max_X, 1e-6)
    max_P_sol = max(max_P_sol, 1e-6)
    max_P_IB = max(max_P_IB, 1e-6)

    def obj_rule(m):
        expr = 0.0
        for d in data_list:
            b = m.run_blocks[d['run_name']]
            meas = d['meas']
            
            for i, t_meas in enumerate(meas['times']):
                x_val = meas['X'][i]
                p_val = meas['P_sol'][i]
                ib_val = meas['P_IB'][i]
                
                # Normalize error by max observed value: ((Fit - Obs) / Max)**2
                if not np.isnan(x_val):
                    expr += ((b.X[t_meas] - x_val) / max_X)**2 * 1.0
                
                # Multiply protein error by 100 to force the solver to care!
                if not np.isnan(p_val):
                    expr += ((b.P_sol[t_meas] - p_val) / max_P_sol)**2 * 100.0
                    
                if not np.isnan(ib_val):
                    expr += ((b.P_IB[t_meas] - ib_val) / max_P_IB)**2 * 100.0
                    
        return expr
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Scale objective here, after it is defined
    m.scaling_factor[m.obj] = 0.1
    
    return m

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("Simultaneous Approach (Pyomo + IPOPT)")
    print("Reference: Neural DAE Training")
    print("="*60)
    
    # 1. Load Data
    run_ids = ["ferm_01", "ferm_02", "ferm_03", "ferm_04"]
    data_list = []
    for rid in run_ids:
        try:
            print(f"Loading {rid}...")
            data_list.append(load_data(rid))
        except Exception as e:
            print(f"Skipping {rid}: {e}")
            
    # 2. Build Model
    print("\nBuilding Pyomo Model...")
    model = create_pyomo_model(data_list)
    
    # 3. Discretize (Collocation)
    # Collocation adds the algebraic equations for the derivatives
    print("Discretizing equations (Collocation - Lagrange Radau)...")
    discretizer = pyo.TransformationFactory('dae.collocation')
    
    # Apply to the whole model (it finds all ContinuousSets)
    discretizer.apply_to(model, nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
        
    # 4. Solve with IPOPT
    print("\nSolving with IPOPT...")
    solver = pyo.SolverFactory('ipopt')
    
    # Options for robustness
    solver.options['max_iter'] = 1000        # Give it more time
    solver.options['tol'] = 1e-4             # Slightly looser tolerance helps convergence
    solver.options['nlp_scaling_method'] = 'user-scaling' # Tell it to use our scaling
    # solver.options['halt_on_ampl_error'] = 'yes'
    
    try:
        results = solver.solve(model, tee=True)
    except Exception as e:
        print(f"Solver Error: {e}")
        print("Ensure 'ipopt' is in your PATH.")
        return

    # 5. Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("Parameters:")
    for p in [model.Y_XS, model.m_s, model.alpha, model.beta, model.T_50, model.k_T, model.k_base]:
        print(f"  {p.name}: {pyo.value(p):.6f}")
        
    # 6. Plot
    print("\nPlotting...")
    fig, axes = plt.subplots(len(run_ids), 3, figsize=(15, 3*len(run_ids)))
    if len(run_ids) == 1: axes = axes.reshape(1, -1)
    
    for i, rid in enumerate(run_ids):
        b = model.run_blocks[rid]
        
        # Get time points from the discretized set
        t_points = list(b.t)
        t_points.sort()
        
        # Extract values
        X_vals = [pyo.value(b.X[t]) for t in t_points]
        P_vals = [pyo.value(b.P_sol[t]) for t in t_points]
        IB_vals = [pyo.value(b.P_IB[t]) for t in t_points]
        
        # Plot X
        axes[i, 0].plot(t_points, X_vals, 'b-', label='Fit')
        meas = next(d for d in data_list if d['run_name']==rid)['meas']
        axes[i, 0].scatter(meas['times'], meas['X'], c='k', marker='x', label='Obs')
        axes[i, 0].set_title(f"{rid} - Biomass")
        axes[i, 0].legend()
        
        # Plot P
        axes[i, 1].plot(t_points, P_vals, 'g-')
        axes[i, 1].scatter(meas['times'], meas['P_sol'], c='k', marker='x')
        axes[i, 1].set_title(f"Soluble Protein")
        
        # Plot IB
        axes[i, 2].plot(t_points, IB_vals, 'r-')
        axes[i, 2].scatter(meas['times'], meas['P_IB'], c='k', marker='x')
        axes[i, 2].set_title(f"Inclusion Bodies")
        
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "pyomo_ipopt_fit.png")
    print(f"Plot saved to {RESULTS_PATH / 'pyomo_ipopt_fit.png'}")

if __name__ == "__main__":
    main()
