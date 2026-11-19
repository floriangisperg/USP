"""Lean test of M_decoupled_v3 (delayed shutoff): does letting IB formation run at full strength until
t_delay, THEN shut off, capture BOTH the peak and the decline (which v2 couldn't -- peak-vs-decline
trade-off from the immediate-decay clock)? Full-data fit + T30/T35 diagnostic.
"""
import data as D
import fit as F
import numpy as np
from models import simulate_run

MDL = "M_decoupled_v3"
T30 = ["F13", "F14", "F15", "F18"]
T35 = ["F07", "F19", "F12"]
b = D.load_dataset(*D.split_for_fold("full"))

print(f"fitting {MDL} (full, n_starts=4)...")
v3 = F.fit_model(MDL, b, seed=42, n_starts=4, max_nfev=400)
print("v3 params:", {k: round(val, 4) for k, val in v3.params.items()})
print(f"v3 cost={v3.cost:.4f}  (baseline 0.5271 ; v2 decline-basin 0.4476 ; v2 peak-basin 0.4833)")
p = v3.params
for T in [25, 30, 35]:
    gs = 1.0 / (1.0 + np.exp(-(p["theta_sol_0"] - p["theta_sol_T"] * (T - 30))))
    gib = 1.0 / (1.0 + np.exp(-(p["theta_ib_0"] + p["theta_ib_T"] * (T - 30))))
    print(f"  T={T}: g_sol={gs:.2f} g_ib={gib:.2f}")
rdm = {r["run_id"]: r for r in b["train"]}
print("\n  (IBend mdl/obs ; dIB mdl/obs ; solend mdl/obs)")
for rid in T30 + T35:
    r = rdm[rid]
    sim = simulate_run(MDL, v3.params, r, b["c_fold"], eval_gas=False)
    so = sim["obs"]
    print(f"  {rid}: IB {so['P_IB'][-1]:.2f}/{r['obs']['P_IB'][-1]:.2f}  "
          f"dIB {so['P_IB'][-1]-so['P_IB'][-2]:+.3f}/{r['obs']['P_IB'][-1]-r['obs']['P_IB'][-2]:+.3f}  "
          f"| sol {so['P_sol'][-1]:.2f}/{r['obs']['P_sol'][-1]:.2f}")
print("DONE")
