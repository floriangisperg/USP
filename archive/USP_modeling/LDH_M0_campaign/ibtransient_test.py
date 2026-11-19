"""Test the IB-specific transient decline (M0_aload_qPTexp_ibtransient): IB formation front-loaded then
decays over induction (tau_IB), soluble = total - decayed-IB (mass-conserving).

Checks: (1) does it capture the IB peak/decline (qP_IB -> 0) WITHOUT breaking soluble? (2) is tau_IB
identifiable (profile likelihood)? (3) does it generalize held-out (fold B: F13/F14/F15) -- the
induction-decay overfit trap.
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import data as D
import fit as F
import matplotlib.pyplot as plt
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
print("fitting ibtransient (full)...")
ibt = F.fit_model("M0_aload_qPTexp_ibtransient", b, seed=SEED, n_starts=3, max_nfev=400)
print("\nbaseline:", {k: round(v, 4) for k, v in base.params.items()})
print("ibtransient:", {k: round(v, 4) for k, v in ibt.params.items()})

for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"), ("ibtransient", ibt, "M0_aload_qPTexp_ibtransient")]:
    rmse, dib = state_err(res, b, mdl, CIND)
    print(f"\n{tag}: cost={res.cost:.4f}  RMSE sol={rmse['P_sol']:.3f} IB={rmse['P_IB']:.3f}")
    for rid in CIND:
        print(f"  {rid}: dIB model {dib[rid][0]:+.3f} vs obs {dib[rid][1]:+.3f}")

# Profile tau_IB (fix, refit).
print("\nprofile tau_IB (full data):")
for tv in [1.0, 1.5, 3.0, 6.0, 12.0]:
    r = F.fit_model("M0_aload_qPTexp_ibtransient", b, seed=SEED, n_starts=NS, max_nfev=400,
                    fixed={"tau_IB": tv})
    print(f"  tau_IB={tv:>5.1f}: cost={r.cost:.4f}")

# Held-out fold B (F13/F14/F15).
tr, te = D.split_for_fold("B")
bB = D.load_dataset(tr, te)
print("\nheld-out fold B (F13/F14/F15) -- ibtransient fit on train:")
ho = F.fit_model("M0_aload_qPTexp_ibtransient", bB, seed=SEED, n_starts=NS, max_nfev=400)
print(f"  tau_IB={ho.params['tau_IB']:.2f} (full-data was {ibt.params['tau_IB']:.2f})")
rdmap = {r["run_id"]: r for r in bB["test"]}
for rid in ["F13", "F14", "F15"]:
    r = rdmap[rid]
    sim = simulate_run("M0_aload_qPTexp_ibtransient", ho.params, r, bB["c_fold"], eval_gas=False)
    so = sim["obs"]
    ib_rmse = np.sqrt(np.nanmean((so["P_IB"] - r["obs"]["P_IB"]) ** 2))
    print(f"  {rid}: IB endpoint model {so['P_IB'][-1]:.2f} vs obs {r['obs']['P_IB'][-1]:.2f}  "
          f"(dIB {so['P_IB'][-1]-so['P_IB'][-2]:+.3f} vs obs {r['obs']['P_IB'][-1]-r['obs']['P_IB'][-2]:+.3f})  "
          f"IB RMSE {ib_rmse:.3f}")

# Plot F13/F18: IB + sol, obs vs baseline vs ibtransient.
rdmap = {r["run_id"]: r for r in b["train"]}
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for col, rid in enumerate(["F13", "F18"]):
    run = rdmap[rid]
    for row, (state, obsk, lab) in enumerate([("sol_Psol", "P_sol", "P_sol"), ("sol_Pib", "P_IB", "P_IB")]):
        ax = axes[row][col]
        ax.scatter(run["t_obs"], run["obs"][obsk], c="k", s=30, zorder=5, label="observed")
        for tag, res, mdl, c in [("baseline", base, "M0_aload_qPTexp", "#888"),
                                 ("ibtransient", ibt, "M0_aload_qPTexp_ibtransient", "#D55E00")]:
            sim = simulate_run(mdl, res.params, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.4, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("IB-transient (orange) vs baseline (grey): IB peak/decline captured, soluble kept?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/ibtransient_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/ibtransient_test.png")
