"""
Simultaneous Approach for Training Neural DAEs (Approximate Implementation)
Reference: arXiv:2504.04665 "A Simultaneous Approach for Training Neural Differential-Algebraic Systems of Equations"

This script implements the "Simultaneous" (Direct Collocation) strategy for the LDH Hybrid Model.
Instead of shooting (ODE simulation), we solve a large constrained optimization problem.

Method:
- Discretize time into N intervals (Trapezoidal Collocation).
- Treat STATES (X, P, V) at every time point as unknown variables.
- Treat PARAMETERS (Y_XS, etc.) as unknown variables.
- Enforce ODE dynamics as EQUALITY CONSTRAINTS between time points.
- Minimize difference between State Variables and Observations.

Libraries:
- JAX: For automatic differentiation (Jacobian of constraints).
- Scipy: For the NLP solver (SLSQP/Trust-Constr).
"""

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, jacfwd
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import time
from typing import Dict, List, Tuple

# Enable 64-bit precision for stability
jax.config.update("jax_enable_x64", True)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(__file__).parent / "data" / "LDH_consolidated_results.xlsx"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# Grid settings
DT_GRID = 0.5  # Time step for collocation grid [h] (0.5h is reasonable for fed-batch)

# Known constants
C_FEED = 400.0
T_FEDBATCH = 37.0

# Parameter mapping (7 parameters)
# [Y_XS, m_s, alpha, beta, T_50, k_T, k_base]
PARAM_BOUNDS = [
    (1e-6, 1.0),     # Y_XS
    (0.0, 0.1),      # m_s
    (0.0, 1.0),      # alpha
    (0.0, 0.2),      # beta
    (20.0, 40.0),    # T_50
    (0.01, 5.0),     # k_T
    (0.0, 0.01),     # k_base
]
PARAM_NAMES = ["Y_XS", "m_s", "alpha", "beta", "T_50", "k_T", "k_base"]

# ============================================================
# 1. DATA PREPARATION (Reused logic)
# ============================================================

DOE_CONDITIONS = {
    "ferm_01": {"qs_setpoint": 0.2, "T_induced": 35.0},
    "ferm_02": {"qs_setpoint": 0.2, "T_induced": 25.0},
    "ferm_03": {"qs_setpoint": 0.6, "T_induced": 25.0},
    "ferm_04": {"qs_setpoint": 0.6, "T_induced": 35.0},
}

def load_and_grid_data(run_name: str, dt: float) -> Dict:
    """
    Load data and interpolate it onto a FIXED discrete grid for collocation.
    """
    # 1. Load Raw Data
    df_all = pd.read_excel(DATA_PATH, sheet_name='All_Time_Points')
    df = df_all[df_all['run_name'] == run_name].copy()
    if df.empty: raise ValueError(f"Run {run_name} not found.")

    # 2. Extract raw time inputs (feed, volume, etc.)
    # Note: We need continuous inputs for the grid.
    times_raw = df['Process time [h]'].values
    
    # Calculate Feed Rate [L/h] from cumulative
    feed_cum = df['Feed [mL]'].fillna(0).values / 1000.0 # L
    dt_raw = np.diff(times_raw, prepend=times_raw[0])
    dt_raw = np.maximum(dt_raw, 0.01)
    feed_rate = np.diff(feed_cum, prepend=0) / dt_raw
    
    # Temperature & Induction
    cond = DOE_CONDITIONS[run_name]
    is_induced_raw = df['Is induced [-]'].fillna(0).values
    T_raw = np.where(is_induced_raw > 0.5, cond["T_induced"], T_FEDBATCH)
    
    # create interpolators for inputs
    # We use numpy interp for simplicity during grid creation
    t_start, t_end = times_raw[0], times_raw[-1]
    
    # 3. Create Uniform Collocation Grid
    # The simultaneous method prefers a structured grid.
    t_grid = np.arange(t_start, t_end + 1e-6, dt)
    N_steps = len(t_grid)
    
    # Interpolate Inputs onto Grid
    F_grid = np.interp(t_grid, times_raw, feed_rate)
    T_grid = np.interp(t_grid, times_raw, T_raw)
    I_grid = np.interp(t_grid, times_raw, is_induced_raw)
    
    # 4. Map Observations to Grid indices
    # We only penalize error at grid points closest to actual samples
    samples = df[df['Biomass [g/L]'].notna()].copy()
    obs_indices = []
    obs_data = [] # List of [X_obs, P_sol_obs, P_IB_obs]
    
    for _, row in samples.iterrows():
        t_sample = row['Process time [h]']
        # Find closest grid point
        idx = np.abs(t_grid - t_sample).argmin()
        
        # Check if reasonably close (within dt/2)
        if np.abs(t_grid[idx] - t_sample) < dt:
            obs_indices.append(idx)
            obs_data.append([
                row['Biomass [g/L]'],
                row['Protein soluble [g/L]'] if pd.notna(row['Protein soluble [g/L]']) else np.nan,
                row['Protein IB [g/L]'] if pd.notna(row['Protein IB [g/L]']) else np.nan
            ])
            
    return {
        "run_name": run_name,
        "t_grid": t_grid,       # [N]
        "dt": dt,
        "N": N_steps,
        # Inputs on grid
        "F": F_grid,
        "T": T_grid,
        "I": I_grid,
        # Observations
        "obs_idx": np.array(obs_indices, dtype=int),
        "obs_vals": np.array(obs_data), # [N_obs, 3]
        # Initial condition guess (from first data point)
        "x0_guess": np.array([
            obs_data[0][0], # X
            0.0,            # P_sol
            0.0,            # P_IB
            df['Volume [mL]'].iloc[0]/1000.0 # V
        ])
    }

# ============================================================
# 2. JAX MODEL DEFINITION
# ============================================================

@jax.jit
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

@jax.jit
def ode_dynamics(x, u, p):
    """
    Computes dx/dt given state x, inputs u, parameters p.
    x: [X, P_sol, P_IB, V]
    u: [F, T, I]
    p: [Y_XS, m_s, alpha, beta, T_50, k_T, k_base]
    """
    X, P_sol, P_IB, V = x
    F, T, I = u
    Y_XS, m_s, alpha, beta, T_50, k_T, k_base = p
    
    # Safety clips (softplus or similar usually better for AD, but clip ok for now)
    X_safe = jnp.maximum(X, 1e-3)
    V_safe = jnp.maximum(V, 1e-3)
    
    # --- Derived qs (Algebraic Relation) ---
    qs = (F * C_FEED) / (X_safe * V_safe)
    qs = jnp.minimum(qs, 2.0) # Saturation
    
    # --- Kinetics ---
    mu = jnp.maximum(Y_XS * qs - m_s, 0.0)
    
    # Base addition
    F_base = k_base * X_safe * V_safe
    D = (F + F_base) / V_safe
    
    # Product formation
    T_opt = 30.0
    f_T = jnp.exp(-0.5 * ((T - T_opt) / 10.0)**2)
    q_p = (alpha * mu + beta) * I * f_T
    
    # IB Partitioning
    phi_IB = sigmoid(k_T * (T - T_50))
    
    # --- ODEs ---
    dX = mu * X - D * X
    dP_sol = q_p * (1.0 - phi_IB) * X - D * P_sol
    dP_IB = q_p * phi_IB * X - D * P_IB
    dV = F + F_base
    
    return jnp.array([dX, dP_sol, dP_IB, dV])

# ============================================================
# 3. OPTIMIZATION PROBLEM WRAPPER
# ============================================================

def build_optimization_problem(runs_data):
    """
    Constructs the vector packing/unpacking and loss/constraint functions.
    """
    n_params = 7
    n_states = 4
    
    # Calculate offsets for one big vector z
    # z = [Params (7)] + [Run1_States (N1*4)] + [Run2_States (N2*4)] + ...
    
    run_shapes = [r["N"] for r in runs_data]
    total_states = sum(run_shapes) * n_states
    total_vars = n_params + total_states
    
    # Pre-structure inputs for JAX efficiency
    # Create static arrays for inputs to avoid rebuilding graph
    inputs_list = []
    for r in runs_data:
        # stack inputs: [F, T, I] -> (N, 3)
        inp = np.stack([r["F"], r["T"], r["I"]], axis=1)
        inputs_list.append(jnp.array(inp))

    obs_idx_list = [jnp.array(r["obs_idx"]) for r in runs_data]
    obs_vals_list = [jnp.array(r["obs_vals"]) for r in runs_data] # NaN handling needed
    dt_list = [r["dt"] for r in runs_data]

    # --- 3a. UNPACKER ---
    def unpack(z):
        theta = z[:n_params]
        states_flat = z[n_params:]
        
        trajectories = []
        offset = 0
        for i, n_steps in enumerate(run_shapes):
            n_vals = n_steps * n_states
            traj = states_flat[offset : offset + n_vals].reshape((n_steps, n_states))
            trajectories.append(traj)
            offset += n_vals
        return theta, trajectories

    # --- 3b. OBJECTIVE (Maximize Data Fit) ---
    @jax.jit
    def objective_fn(z):
        theta, trajectories = unpack(z)
        loss = 0.0
        
        for i, traj in enumerate(trajectories):
            # Extract states at observation indices
            preds = traj[obs_idx_list[i]] # (N_obs, 4)
            # Compare first 3 states (X, Psol, PIB)
            preds_core = preds[:, :3]
            targets = obs_vals_list[i]
            
            # Masked MSE (handle NaNs in targets)
            # We assume NaNs are pre-cleaned or we use safe mask
            mask = ~jnp.isnan(targets)
            diff = (preds_core - targets)
            # Use where to zero out NaNs
            diff_safe = jnp.where(mask, diff, 0.0)
            
            # Simple MSE
            loss += jnp.sum(diff_safe**2) / (jnp.sum(mask) + 1e-6)
            
        return loss

    # --- 3c. CONSTRAINTS (Collocation Defect) ---
    @jax.jit
    def constraints_fn(z):
        theta, trajectories = unpack(z)
        defects = []
        
        for i, traj in enumerate(trajectories):
            inputs = inputs_list[i] # (N, 3)
            dt = dt_list[i]
            
            # State at k and k+1
            x_k = traj[:-1]
            x_kp1 = traj[1:]
            
            # Inputs at k and k+1
            u_k = inputs[:-1]
            u_kp1 = inputs[1:]
            
            # Dynamics f(x, u, theta)
            # Vectorize over time
            f_k = jax.vmap(lambda x, u: ode_dynamics(x, u, theta))(x_k, u_k)
            f_kp1 = jax.vmap(lambda x, u: ode_dynamics(x, u, theta))(x_kp1, u_kp1)
            
            # Trapezoidal Rule Defect
            # x_{k+1} - x_k - 0.5 * h * (f_k + f_{k+1}) = 0
            defect = (x_kp1 - x_k) - 0.5 * dt * (f_k + f_kp1)
            
            defects.append(defect.flatten())
            
        return jnp.concatenate(defects)

    return total_vars, objective_fn, constraints_fn, unpack

# ============================================================
# 4. INITIALIZATION STRATEGY
# ============================================================

def generate_initial_guess(runs_data, n_params):
    """
    Simultaneous approach needs a good guess for the STATES.
    We linear interpolate the observations to fill the grid.
    """
    x0_guesses = []
    
    # Reasonable parameter guess
    theta0 = np.array([0.5, 0.01, 0.05, 0.02, 30.0, 0.5, 0.0001])
    
    for r in runs_data:
        N = r["N"]
        t_grid = r["t_grid"]
        
        # Interp observations
        # X
        X_obs_idx = r["obs_idx"]
        X_obs_val = r["obs_vals"][:, 0]
        # Fill NaNs in obs first? Assuming X is dense enough
        X_guess = np.interp(t_grid, t_grid[X_obs_idx], X_obs_val)
        
        # P_sol, P_IB (sparse, fill with 0 or linear interp)
        # For simplicity, initialized low
        P_sol_guess = np.zeros(N)
        P_IB_guess = np.zeros(N)
        
        # V (Volume increase is predictable)
        V0 = r["x0_guess"][3]
        # integrated feed roughly
        V_guess = np.interp(t_grid, [t_grid[0], t_grid[-1]], [V0, V0 + 0.5]) # rough linear
        
        state_guess = np.stack([X_guess, P_sol_guess, P_IB_guess, V_guess], axis=1) # (N, 4)
        x0_guesses.append(state_guess.flatten())
        
    z0 = np.concatenate([theta0] + x0_guesses)
    return z0

# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("="*60)
    print("SIMULTANEOUS APPROACH (JAX + Scipy)")
    print("="*60)
    
    # 1. Load Data
    run_ids = ["ferm_01", "ferm_02", "ferm_03", "ferm_04"]
    runs_data = []
    print("Loading data and creating collocation grids...")
    for rid in run_ids:
        try:
            r = load_and_grid_data(rid, dt=DT_GRID)
            runs_data.append(r)
            print(f"  {rid}: {r['N']} grid points ({r['dt']}h step)")
        except Exception as e:
            print(f"  Skipping {rid}: {e}")
            
    # 2. Build Problem
    total_vars, obj_fn, constr_fn, unpacker = build_optimization_problem(runs_data)
    print(f"\nOptimization Problem Size:")
    print(f"  Variables: {total_vars}")
    print(f"    (7 Params + {total_vars-7} State Variables)")
    
    # 3. JIT Compile Derivatives
    print("JIT Compiling Jacobian...")
    t0 = time.time()
    
    # Value and Grad for Objective
    obj_vg = jax.jit(value_and_grad(obj_fn))
    
    # Jacobian for Constraints
    constr_jac = jax.jit(jacfwd(constr_fn))
    
    # Warmup
    z_init = generate_initial_guess(runs_data, 7)
    _ = obj_vg(z_init)
    _ = constr_jac(z_init)
    print(f"Compilation done in {time.time()-t0:.2f}s")
    
    # 4. Solve
    print("\nStarting Optimization (Scipy SLSQP)...")
    
    constraints = [
        {
            'type': 'eq', 
            'fun': lambda z: np.array(constr_fn(z)), 
            'jac': lambda z: np.array(constr_jac(z))
        }
    ]
    
    bounds = []
    # Param bounds
    bounds.extend(PARAM_BOUNDS)
    # State bounds (prevent negative states)
    for r in runs_data:
        n_pts = r["N"]
        # X, Psol, PIB, V >= 0
        state_bounds = [(0, None)] * (n_pts * 4)
        bounds.extend(state_bounds)
        
    res = minimize(
        fun=lambda z: (float(v:=obj_vg(z)[0]), np.array(obj_vg(z)[1]))[0], # weird wrapper for scipy
        jac=lambda z: np.array(obj_vg(z)[1]),
        x0=z_init,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 200, 'disp': True, 'ftol': 1e-4}
    )
    
    # 5. Results
    print("\n" + "="*60)
    print(f"Optimization Finished: {res.message}")
    print(f"Final Objective: {res.fun:.4f}")
    
    theta_opt, trajs_opt = unpacker(res.x)
    print("\nFitted Parameters:")
    for name, val in zip(PARAM_NAMES, theta_opt):
        print(f"  {name:10s}: {val:.6f}")
        
    # --- Visualization ---
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(run_ids), 3, figsize=(15, 3*len(run_ids)))
    if len(run_ids) == 1: axes = axes.reshape(1, -1)
    
    for i, (rid, traj) in enumerate(zip(run_ids, trajs_opt)):
        r = runs_data[i]
        t = r["t_grid"]
        
        # Plot X
        ax = axes[i, 0]
        ax.plot(t, traj[:, 0], 'b-', label='Simultaneous Fit')
        obs_t = t[r["obs_idx"]]
        obs_x = r["obs_vals"][:, 0]
        ax.scatter(obs_t, obs_x, c='k', marker='x', label='Data')
        ax.set_title(f"{rid} - Biomass")
        ax.set_ylim(bottom=0)
        
        # Plot P_sol
        ax = axes[i, 1]
        ax.plot(t, traj[:, 1], 'g-', label='P_sol')
        obs_p = r["obs_vals"][:, 1]
        ax.scatter(obs_t, obs_p, c='k', marker='x')
        ax.set_title("Soluble Protein")
        
        # Plot P_IB
        ax = axes[i, 2]
        ax.plot(t, traj[:, 2], 'r-', label='P_IB')
        obs_ib = r["obs_vals"][:, 2]
        ax.scatter(obs_t, obs_ib, c='k', marker='x')
        ax.set_title("Inclusion Bodies")
        
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "simultaneous_fit.png")
    print(f"\nPlot saved to {RESULTS_PATH / 'simultaneous_fit.png'}")

if __name__ == "__main__":
    main()
