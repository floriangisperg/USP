"""Test M0_smet_independent_ibdeg (option 1+3, 2026-07-27): the smet-independent rates (NN m13
oracle port) PLUS a net-negative IB channel -- first-order IB proteolysis k_deg*P_IB (-> CO2) on
top of the S_met formation decline. This is the NN's actual recipe (tanh -> negative qP_IB ~ -0.001
late): the gentle exp(-S_met/K_IB) alone only REDUCES formation (>=0) so P_IB plateaus late (model
+0.057 vs obs -0.17 for F13); the degradation channel lets net IB flux drop below dilution -> decline.

Checks (identifiability-first, same bar as the 5 prior IB/trajectory mechanisms):
  (1) does it capture the F13/F18 IB DECLINE (dIB < 0) without breaking soluble?
  (2) is k_deg IDENTIFIABLE -- interior (not at 0/0.5 bound), with a bounded interior profile minimum?
  (3) does qP_IB_max move off its (raised) 0.30 bound -- i.e. is option 3 working?
  (4) does it GENERALIZE held-out (fold B: F13/F14/F15) -- the overfit trap that sank smet_independent,
      induction-decay, M1b, ibtransient?
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D
import fit as F
from models import simulate_run

CIND = ["F13", "F14", "F15", "F18"]
SEED, NS = 42, 2

b = D.load_dataset(*D.split_for_fold("full"))


def state_err(res, bundle, model, runs):
    p, cf = res.params, bundle["c_fold"]
    rdmap = {r["run_id"]: r for r in bundle["train"]}
    rmse = {"P_sol": [], "P_IB": []}
    dib = {}
    for rid in runs:
        r = rdmap[rid]
        sim = simulate_run(model, p, r, cf, eval_gas=False)
        so = sim["obs"]
        for k in ("P_sol", "P_IB"):
            rmse[k].append(np.sqrt(np.nanmean((so[k] - r["obs"][k]) ** 2)))
        dib[rid] = (so["P_IB"][-1] - so["P_IB"][-2], r["obs"]["P_IB"][-1] - r["obs"]["P_IB"][-2])
    return {k: float(np.mean(v)) for k, v in rmse.items()}, dib


print("fitting baseline (full)...")
base = F.fit_model("M0_aload_qPTexp", b, seed=SEED, n_starts=NS, max_nfev=400)
print("fitting M0_smet_independent (full)...")
smet = F.fit_model("M0_smet_independent", b, seed=SEED, n_starts=4, max_nfev=500)
print("fitting M0_smet_independent_ibdeg (full)...")
ibdeg = F.fit_model("M0_smet_independent_ibdeg", b, seed=SEED, n_starts=4, max_nfev=500)

print("\nbaseline:", {k: round(v, 4) for k, v in base.params.items()})
print("smet    :", {k: round(v, 4) for k, v in smet.params.items()})
print("ibdeg   :", {k: round(v, 4) for k, v in ibdeg.params.items()})

for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"),
                      ("smet_indep", smet, "M0_smet_independent"),
                      ("smet_ibdeg", ibdeg, "M0_smet_independent_ibdeg")]:
    rmse, dib = state_err(res, b, mdl, CIND)
    print(f"\n{tag}: cost={res.cost:.4f}  RMSE sol={rmse['P_sol']:.3f} IB={rmse['P_IB']:.3f}")
    for rid in CIND:
        print(f"  {rid}: dIB model {dib[rid][0]:+.3f} vs obs {dib[rid][1]:+.3f}")

# (2) Profile k_deg (the identifiability arbiter). Fix on a grid, refit the rest, report train cost.
print("\nprofile k_deg (full data):")
for kv in [0.0, 0.02, 0.05, 0.10, 0.20]:
    r = F.fit_model("M0_smet_independent_ibdeg", b, seed=SEED, n_starts=NS, max_nfev=400,
                    fixed={"k_deg": kv})
    qpib = r.params.get("qP_IB_max", float("nan"))
    kib = r.params.get("K_IB", float("nan"))
    print(f"  k_deg={kv:>5.2f}: cost={r.cost:.4f}  qP_IB_max={qpib:.3f}  K_IB={kib:.1f}")

# (4) Held-out fold B (F13/F14/F15) -- the overfit trap.
tr, te = D.split_for_fold("B")
bB = D.load_dataset(tr, te)
print("\nheld-out fold B (F13/F14/F15) -- smet_ibdeg fit on train:")
ho = F.fit_model("M0_smet_independent_ibdeg", bB, seed=SEED, n_starts=4, max_nfev=500)
print(f"  k_deg={ho.params['k_deg']:.3f} (full-data {ibdeg.params['k_deg']:.3f})  "
      f"qP_IB_max={ho.params['qP_IB_max']:.3f}  K_IB={ho.params['K_IB']:.1f}")
rdmap = {r["run_id"]: r for r in bB["test"]}
for rid in ["F13", "F14", "F15"]:
    r = rdmap[rid]
    sim = simulate_run("M0_smet_independent_ibdeg", ho.params, r, bB["c_fold"], eval_gas=False)
    so = sim["obs"]
    ib_rmse = np.sqrt(np.nanmean((so["P_IB"] - r["obs"]["P_IB"]) ** 2))
    print(f"  {rid}: IB endp model {so['P_IB'][-1]:.2f} vs obs {r['obs']['P_IB'][-1]:.2f}  "
          f"(dIB {so['P_IB'][-1]-so['P_IB'][-2]:+.3f} vs obs {r['obs']['P_IB'][-1]-r['obs']['P_IB'][-2]:+.3f})  "
          f"IB RMSE {ib_rmse:.3f}  | sol endp model {so['P_sol'][-1]:.2f} vs obs {r['obs']['P_sol'][-1]:.2f}")

# Plot F13/F18: sol + IB, obs vs baseline vs smet vs smet_ibdeg.
rdmap = {r["run_id"]: r for r in b["train"]}
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for col, rid in enumerate(["F13", "F18"]):
    run = rdmap[rid]
    for row, (state, obsk, lab) in enumerate([("sol_Psol", "P_sol", "P_sol"), ("sol_Pib", "P_IB", "P_IB")]):
        ax = axes[row][col]
        ax.scatter(run["t_obs"], run["obs"][obsk], c="k", s=30, zorder=5, label="observed")
        for tag, res, mdl, c in [("baseline", base, "M0_aload_qPTexp", "#888"),
                                 ("smet_indep", smet, "M0_smet_independent", "#009E73"),
                                 ("smet_ibdeg", ibdeg, "M0_smet_independent_ibdeg", "#D55E00")]:
            sim = simulate_run(mdl, res.params, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.4, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("M0_smet_independent_ibdeg (orange) vs smet (green) vs baseline (grey):\n"
             "net-negative IB channel -- does it capture the IB peak/decline (F13, F18)?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("results/smet_ibdeg_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/smet_ibdeg_test.png")
