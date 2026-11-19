"""Test M0_smet_independent (independent S_met-driven qP_sol/qP_IB, NN m13 oracle port):
does it capture the trajectory (sol monotonic rise + IB peak-then-decline) the partition model misses,
without breaking soluble, and generalize held-out?
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
print("\nbaseline:", {k: round(v, 4) for k, v in base.params.items()})
print("smet    :", {k: round(v, 4) for k, v in smet.params.items()})

for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"), ("smet_indep", smet, "M0_smet_independent")]:
    rmse, dib = state_err(res, b, mdl, CIND)
    print(f"\n{tag}: cost={res.cost:.4f}  RMSE sol={rmse['P_sol']:.3f} IB={rmse['P_IB']:.3f}")
    for rid in CIND:
        print(f"  {rid}: dIB model {dib[rid][0]:+.3f} vs obs {dib[rid][1]:+.3f}")

# Held-out fold B.
tr, te = D.split_for_fold("B")
bB = D.load_dataset(tr, te)
print("\nheld-out fold B (F13/F14/F15) -- smet fit on train:")
ho = F.fit_model("M0_smet_independent", bB, seed=SEED, n_starts=NS, max_nfev=500)
print(f"  K_IB={ho.params['K_IB']:.1f} (full-data {smet.params['K_IB']:.1f})")
rdmap = {r["run_id"]: r for r in bB["test"]}
for rid in ["F13", "F14", "F15"]:
    r = rdmap[rid]
    sim = simulate_run("M0_smet_independent", ho.params, r, bB["c_fold"], eval_gas=False)
    so = sim["obs"]
    print(f"  {rid}: IB endp model {so['P_IB'][-1]:.2f} vs obs {r['obs']['P_IB'][-1]:.2f} "
          f"(dIB {so['P_IB'][-1]-so['P_IB'][-2]:+.3f} vs obs {r['obs']['P_IB'][-1]-r['obs']['P_IB'][-2]:+.3f})  "
          f"sol endp model {so['P_sol'][-1]:.2f} vs obs {r['obs']['P_sol'][-1]:.2f}")

# Plot F13/F18: sol + IB.
rdmap = {r["run_id"]: r for r in b["train"]}
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for col, rid in enumerate(["F13", "F18"]):
    run = rdmap[rid]
    for row, (state, obsk, lab) in enumerate([("sol_Psol", "P_sol", "P_sol"), ("sol_Pib", "P_IB", "P_IB")]):
        ax = axes[row][col]
        ax.scatter(run["t_obs"], run["obs"][obsk], c="k", s=30, zorder=5, label="observed")
        for tag, res, mdl, c in [("baseline", base, "M0_aload_qPTexp", "#888"),
                                 ("smet_indep", smet, "M0_smet_independent", "#009E73")]:
            sim = simulate_run(mdl, res.params, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.4, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("M0_smet_independent (green) vs baseline (grey): sol rise + IB peak/decline captured?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/smet_independent_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/smet_independent_test.png")
