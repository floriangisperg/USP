"""Lean follow-up for M_decoupled_v2: complete the tau_IB identifiability profile + fold C held-out
(the pieces missing when the full run was killed). tau=0.70 (cost 0.448) and tau=1.0 (0.495) are
already known from the full run; this fills tau<0.7 (interior or running to the 0.5 bound?) and the
right side, plus the fold-C generalization test (hold out F18, train on F13/14/15).
"""
import data as D
import fit as F
import numpy as np
from models import simulate_run

MDL = "M_decoupled_v2"
SEED = 42
b = D.load_dataset(*D.split_for_fold("full"))

print("profile tau_IB (full data, n_starts=2):")
print("  (known: tau=0.70 cost=0.4476 qP_IB_max=0.100 ; tau=1.0 cost=0.4949 qP_IB_max=0.056)")
for tv in [0.5, 2.0, 4.0]:
    r = F.fit_model(MDL, b, seed=SEED, n_starts=2, max_nfev=400, fixed={"tau_IB": tv})
    print(f"  tau_IB={tv:>4.1f}: cost={r.cost:.4f}  qP_IB_max={r.params['qP_IB_max']:.3f}  "
          f"qP_sol_max={r.params['qP_sol_max']:.3f}")

print("\nheld-out fold C (holdout F18 -- the IB-transient generalization test):")
tr, te = D.split_for_fold("C")
bf = D.load_dataset(tr, te)
ho = F.fit_model(MDL, bf, seed=SEED, n_starts=3, max_nfev=500)
print(f"  train-fit: tau_IB={ho.params['tau_IB']:.2f}  qP_IB_max={ho.params['qP_IB_max']:.3f}  "
      f"qP_sol_max={ho.params['qP_sol_max']:.3f}  (full: tau 0.70, qP_IB_max 0.10)")
rdm = {r["run_id"]: r for r in bf["test"]}
r = rdm["F18"]
sim = simulate_run(MDL, ho.params, r, bf["c_fold"], eval_gas=False)
so = sim["obs"]
ri = np.sqrt(np.nanmean((so["P_IB"] - r["obs"]["P_IB"]) ** 2))
print(f"  F18: IBend {so['P_IB'][-1]:.2f}/{r['obs']['P_IB'][-1]:.2f}  "
      f"dIB {so['P_IB'][-1]-so['P_IB'][-2]:+.3f}/{r['obs']['P_IB'][-1]-r['obs']['P_IB'][-2]:+.3f}  "
      f"IB RMSE {ri:.3f}  | solend {so['P_sol'][-1]:.2f}/{r['obs']['P_sol'][-1]:.2f}")
print("DONE")
