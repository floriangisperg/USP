"""
Neural DAE v2: NN Predicts Product Formation (qp, phi_IB)
Reference: arXiv:2504.04665

Key Insight:
- The GROWTH (biomass) fits well with mechanistic model → Keep it!
- The PRODUCT FORMATION is broken → Let NN learn it!

This version:
1. Mechanistic: μ = Y_XS * qs - m_s (proven to work)
2. Neural: q_p = NN(μ, T, I) - protein production rate
3. Neural: φ_IB = NN(T, μ) - fraction going to inclusion bodies (depends on T and growth rate)

Author: Modeling Playground
"""

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(__file__).parent / "data" / "LDH_consolidated_results.xlsx"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

C_FEED = 400.0
T_FEDBATCH = 37.0

# NN for q_p: 3 inputs (mu_norm, T_norm, I) -> 4 hidden -> 1 output
QP_HIDDEN = 4
# NN for phi_IB: 2 inputs (T_norm, mu_norm) -> 4 hidden -> 1 output  
PHI_HIDDEN = 4
PHI_INPUTS = 2  # T and μ both affect IB formation

# Normalization ranges
T_MIN, T_MAX = 25.0, 37.0
MU_MIN, MU_MAX = 0.0, 0.5
QP_MIN, QP_MAX = 0.0, 0.15  # Typical q_p range [gP/(gX*h)]

L2_PENALTY = 1e-3  # Regularization

DOE_CONDITIONS = {
    "ferm_01": {"T_induced": 35.0},
    "ferm_02": {"T_induced": 25.0},
    "ferm_03": {"T_induced": 25.0},
    "ferm_04": {"T_induced": 35.0},
}

# ============================================================
# DATA LOADING
# ============================================================

def load_data(run_name):
    df_all = pd.read_excel(DATA_PATH, sheet_name='All_Time_Points')
    df = df_all[df_all['run_name'] == run_name].copy()
    if df.empty:
        raise ValueError(f"Run {run_name} not found")

    times = df['Process time [h]'].values
    feed_cum = df['Feed [mL]'].fillna(0).values / 1000.0
    dt_raw = np.maximum(np.diff(times, prepend=times[0]), 0.01)
    feed_rate = np.diff(feed_cum, prepend=0) / dt_raw

    cond = DOE_CONDITIONS[run_name]
    is_induced = df['Is induced [-]'].fillna(0).values.astype(float)
    T_vals = np.where(is_induced > 0.5, cond["T_induced"], T_FEDBATCH)

    samples = df[df['Biomass [g/L]'].notna()].copy()
    meas = {
        'times': samples['Process time [h]'].values,
        'X': samples['Biomass [g/L]'].values,
        'P_sol': samples['Protein soluble [g/L]'].values,
        'P_IB': samples['Protein IB [g/L]'].values
    }

    return {
        'run_name': run_name,
        't_start': times[0],
        't_end': times[-1],
        'V0': df['Volume [mL]'].iloc[0] / 1000.0,
        'X0': meas['X'][0],
        'feed_fn': interp1d(times, feed_rate, fill_value="extrapolate"),
        'T_fn': interp1d(times, T_vals, fill_value="extrapolate"),
        'I_fn': interp1d(times, is_induced, fill_value="extrapolate"),
        'meas': meas
    }


# ============================================================
# PYOMO MODEL
# ============================================================

def create_model(data_list):
    m = pyo.ConcreteModel(name="Neural_DAE_v2")

    # =========================================
    # MECHANISTIC PARAMETERS (Growth)
    # =========================================
    m.Y_XS = pyo.Var(bounds=(0.3, 0.7), initialize=0.5)
    m.m_s = pyo.Var(bounds=(0.0, 0.05), initialize=0.01)
    m.k_base = pyo.Var(bounds=(0.0, 0.001), initialize=0.0001)

    # =========================================
    # NN WEIGHTS: q_p network (3 -> 4 -> 1)
    # =========================================
    xavier_qp = np.sqrt(6.0 / (3 + QP_HIDDEN))
    m.W1_qp = pyo.Var(range(3), range(QP_HIDDEN), bounds=(-2, 2),
                       initialize=lambda m, i, j: np.random.uniform(-xavier_qp, xavier_qp))
    m.b1_qp = pyo.Var(range(QP_HIDDEN), bounds=(-1, 1), initialize=0.01)
    
    xavier_qp2 = np.sqrt(6.0 / (QP_HIDDEN + 1))
    m.W2_qp = pyo.Var(range(QP_HIDDEN), range(1), bounds=(-2, 2),
                       initialize=lambda m, i, j: np.random.uniform(-xavier_qp2, xavier_qp2))
    m.b2_qp = pyo.Var(range(1), bounds=(-1, 1), initialize=0.1)

    # =========================================
    # NN WEIGHTS: phi_IB network (2 -> 4 -> 1) - T and μ both affect folding
    # =========================================
    xavier_phi = np.sqrt(6.0 / (PHI_INPUTS + PHI_HIDDEN))
    m.W1_phi = pyo.Var(range(PHI_INPUTS), range(PHI_HIDDEN), bounds=(-2, 2),
                        initialize=lambda m, i, j: np.random.uniform(-xavier_phi, xavier_phi))
    m.b1_phi = pyo.Var(range(PHI_HIDDEN), bounds=(-1, 1), initialize=0.01)
    
    xavier_phi2 = np.sqrt(6.0 / (PHI_HIDDEN + 1))
    m.W2_phi = pyo.Var(range(PHI_HIDDEN), range(1), bounds=(-2, 2),
                        initialize=lambda m, i, j: np.random.uniform(-xavier_phi2, xavier_phi2))
    m.b2_phi = pyo.Var(range(1), bounds=(-1, 1), initialize=0.0)

    # =========================================
    # SCALING
    # =========================================
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

    # =========================================
    # RUN BLOCKS
    # =========================================
    def create_run_block(b, run_name):
        data = next(d for d in data_list if d['run_name'] == run_name)
        meas = data['meas']

        t_init = np.sort(np.unique(np.concatenate([
            [data['t_start'], data['t_end']], meas['times']
        ])))
        b.t = ContinuousSet(initialize=t_init)

        # States
        b.X = pyo.Var(b.t, bounds=(0.1, 100), initialize=data['X0'])
        b.P_sol = pyo.Var(b.t, bounds=(0, 20), initialize=0.1)
        b.P_IB = pyo.Var(b.t, bounds=(0, 20), initialize=0.1)
        b.V = pyo.Var(b.t, bounds=(0.1, 10), initialize=data['V0'])

        b.dXdt = DerivativeVar(b.X, wrt=b.t)
        b.dP_soldt = DerivativeVar(b.P_sol, wrt=b.t)
        b.dP_IBdt = DerivativeVar(b.P_IB, wrt=b.t)
        b.dVdt = DerivativeVar(b.V, wrt=b.t)

        # NN intermediate variables
        b.h_qp = pyo.Var(b.t, range(QP_HIDDEN), bounds=(-1, 1), initialize=0)
        b.q_p = pyo.Var(b.t, bounds=(0, 0.5), initialize=0.01)
        
        b.h_phi = pyo.Var(b.t, range(PHI_HIDDEN), bounds=(-1, 1), initialize=0)
        b.phi_IB = pyo.Var(b.t, bounds=(0, 1), initialize=0.5)

        b.mu = pyo.Var(b.t, bounds=(0, 1), initialize=0.1)

        def get_inputs(t):
            return (float(data['feed_fn'](t)), 
                    float(data['T_fn'](t)), 
                    float(data['I_fn'](t)))

        # --- MECHANISTIC GROWTH ---
        def mu_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            qs = (F * C_FEED) / (X * V + 1e-6)
            qs = pyo.value(qs) if isinstance(qs, (int, float)) else qs
            mu_raw = m.model().Y_XS * qs - m.model().m_s
            # Smooth max(0, x)
            return m.mu[t] == 0.5 * (mu_raw + pyo.sqrt(mu_raw**2 + 1e-6))
        b.mu_con = pyo.Constraint(b.t, rule=mu_rule)

        # --- NN for q_p: inputs = [mu_norm, T_norm, I] ---
        def qp_hidden_rule(m, t, j):
            F, T, I = get_inputs(t)
            mu_norm = (m.mu[t] - MU_MIN) / (MU_MAX - MU_MIN + 1e-8)
            T_norm = (T - T_MIN) / (T_MAX - T_MIN + 1e-8)
            inputs = [mu_norm, T_norm, I]
            
            z = sum(m.model().W1_qp[i, j] * inputs[i] for i in range(3))
            z += m.model().b1_qp[j]
            return m.h_qp[t, j] == (pyo.exp(z) - pyo.exp(-z)) / (pyo.exp(z) + pyo.exp(-z) + 1e-10)
        b.qp_h_con = pyo.Constraint(b.t, range(QP_HIDDEN), rule=qp_hidden_rule)

        def qp_output_rule(m, t):
            z = sum(m.model().W2_qp[j, 0] * m.h_qp[t, j] for j in range(QP_HIDDEN))
            z += m.model().b2_qp[0]
            # Sigmoid output * max q_p
            sig = 1.0 / (1.0 + pyo.exp(-z))
            return m.q_p[t] == sig * QP_MAX
        b.qp_out_con = pyo.Constraint(b.t, rule=qp_output_rule)

        # --- NN for phi_IB: inputs = [T_norm, mu_norm] ---
        def phi_hidden_rule(m, t, j):
            F, T, I = get_inputs(t)
            T_norm = (T - T_MIN) / (T_MAX - T_MIN + 1e-8)
            mu_norm = (m.mu[t] - MU_MIN) / (MU_MAX - MU_MIN + 1e-8)
            inputs = [T_norm, mu_norm]
            
            z = sum(m.model().W1_phi[i, j] * inputs[i] for i in range(PHI_INPUTS))
            z += m.model().b1_phi[j]
            return m.h_phi[t, j] == (pyo.exp(z) - pyo.exp(-z)) / (pyo.exp(z) + pyo.exp(-z) + 1e-10)
        b.phi_h_con = pyo.Constraint(b.t, range(PHI_HIDDEN), rule=phi_hidden_rule)

        def phi_output_rule(m, t):
            z = sum(m.model().W2_phi[j, 0] * m.h_phi[t, j] for j in range(PHI_HIDDEN))
            z += m.model().b2_phi[0]
            return m.phi_IB[t] == 1.0 / (1.0 + pyo.exp(-z))  # Sigmoid [0,1]
        b.phi_out_con = pyo.Constraint(b.t, rule=phi_output_rule)

        # --- ODEs ---
        def dX_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            F_base = m.model().k_base * X * V
            D = (F + F_base) / (V + 1e-6)
            return m.dXdt[t] == m.mu[t] * X - D * X
        b.ode_X = pyo.Constraint(b.t, rule=dX_rule)

        def dV_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            F_base = m.model().k_base * X * V
            return m.dVdt[t] == F + F_base
        b.ode_V = pyo.Constraint(b.t, rule=dV_rule)

        def dP_sol_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            F_base = m.model().k_base * X * V
            D = (F + F_base) / (V + 1e-6)
            # Only produce during induction
            return m.dP_soldt[t] == m.q_p[t] * (1 - m.phi_IB[t]) * X * I - D * m.P_sol[t]
        b.ode_Psol = pyo.Constraint(b.t, rule=dP_sol_rule)

        def dP_IB_rule(m, t):
            F, T, I = get_inputs(t)
            X, V = m.X[t], m.V[t]
            F_base = m.model().k_base * X * V
            D = (F + F_base) / (V + 1e-6)
            return m.dP_IBdt[t] == m.q_p[t] * m.phi_IB[t] * X * I - D * m.P_IB[t]
        b.ode_PIB = pyo.Constraint(b.t, rule=dP_IB_rule)

        # Initial conditions
        def init_rule(m):
            yield m.X[data['t_start']] == data['X0']
            yield m.V[data['t_start']] == data['V0']
            yield m.P_sol[data['t_start']] == 0.0
            yield m.P_IB[data['t_start']] == 0.0
        b.init_con = pyo.ConstraintList(rule=init_rule)

    m.run_blocks = pyo.Block([d['run_name'] for d in data_list], rule=create_run_block)

    # =========================================
    # OBJECTIVE
    # =========================================
    all_X = np.concatenate([d['meas']['X'] for d in data_list])
    all_P_sol = np.concatenate([d['meas']['P_sol'] for d in data_list])
    all_P_IB = np.concatenate([d['meas']['P_IB'] for d in data_list])
    max_X = max(np.nanmax(all_X), 1.0)
    max_P = max(np.nanmax(all_P_sol[~np.isnan(all_P_sol)]), 0.1)
    max_IB = max(np.nanmax(all_P_IB[~np.isnan(all_P_IB)]), 0.1)

    def obj_rule(m):
        loss = 0.0
        for d in data_list:
            b = m.run_blocks[d['run_name']]
            meas = d['meas']
            for i, t_meas in enumerate(meas['times']):
                x, p, ib = meas['X'][i], meas['P_sol'][i], meas['P_IB'][i]
                if not np.isnan(x):
                    loss += ((b.X[t_meas] - x) / max_X)**2
                if not np.isnan(p):
                    loss += ((b.P_sol[t_meas] - p) / max_P)**2 * 50.0
                if not np.isnan(ib):
                    loss += ((b.P_IB[t_meas] - ib) / max_IB)**2 * 50.0

        # L2 regularization
        reg = 0.0
        for i in range(3):
            for j in range(QP_HIDDEN):
                reg += m.W1_qp[i, j]**2
        for j in range(QP_HIDDEN):
            reg += m.W2_qp[j, 0]**2
        for i in range(PHI_INPUTS):
            for j in range(PHI_HIDDEN):
                reg += m.W1_phi[i, j]**2
        for j in range(PHI_HIDDEN):
            reg += m.W2_phi[j, 0]**2

        return loss + L2_PENALTY * reg

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    return m


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Neural DAE v2: NN Predicts q_p and φ_IB")
    print("=" * 60)

    run_ids = ["ferm_01", "ferm_02", "ferm_03", "ferm_04"]
    data_list = [load_data(rid) for rid in run_ids]

    print("\nBuilding model...")
    model = create_model(data_list)

    print("Discretizing...")
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, nfe=40, ncp=3, scheme='LAGRANGE-RADAU')

    print("\nSolving with IPOPT...")
    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = 3000
    solver.options['tol'] = 1e-4
    solver.options['nlp_scaling_method'] = 'gradient-based'
    solver.options['mu_strategy'] = 'adaptive'
    
    results = solver.solve(model, tee=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Y_XS = {pyo.value(model.Y_XS):.4f}")
    print(f"m_s = {pyo.value(model.m_s):.4f}")

    # Plot
    fig, axes = plt.subplots(len(run_ids), 4, figsize=(18, 3 * len(run_ids)))
    
    for i, rid in enumerate(run_ids):
        b = model.run_blocks[rid]
        t_pts = sorted(b.t)
        
        X_v = [pyo.value(b.X[t]) for t in t_pts]
        P_v = [pyo.value(b.P_sol[t]) for t in t_pts]
        IB_v = [pyo.value(b.P_IB[t]) for t in t_pts]
        qp_v = [pyo.value(b.q_p[t]) for t in t_pts]
        phi_v = [pyo.value(b.phi_IB[t]) for t in t_pts]
        
        meas = next(d for d in data_list if d['run_name'] == rid)['meas']
        
        axes[i, 0].plot(t_pts, X_v, 'b-')
        axes[i, 0].scatter(meas['times'], meas['X'], c='k', marker='x')
        axes[i, 0].set_title(f"{rid} - Biomass")
        
        axes[i, 1].plot(t_pts, P_v, 'g-')
        axes[i, 1].scatter(meas['times'], meas['P_sol'], c='k', marker='x')
        axes[i, 1].set_title("Soluble Protein")
        
        axes[i, 2].plot(t_pts, IB_v, 'r-')
        axes[i, 2].scatter(meas['times'], meas['P_IB'], c='k', marker='x')
        axes[i, 2].set_title("Inclusion Bodies")
        
        axes[i, 3].plot(t_pts, qp_v, 'm-', label='q_p')
        ax2 = axes[i, 3].twinx()
        ax2.plot(t_pts, phi_v, 'c--', label='φ_IB')
        axes[i, 3].set_title("NN: q_p (magenta), φ_IB (cyan)")
        axes[i, 3].set_ylim(0, 0.2)
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "neural_dae_v2_fit.png")
    print(f"\nPlot saved to {RESULTS_PATH / 'neural_dae_v2_fit.png'}")


if __name__ == "__main__":
    main()
