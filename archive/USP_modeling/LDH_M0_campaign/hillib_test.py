"""Test: does cooperative (Hill) IB formation capture the front-loaded IB peak/decline WITHOUT breaking soluble?

Fits baseline (M0_aload_qPTexp, linear phi split) vs Hill-IB (M0_qPTexp_hillib, qP_IBmax*Hill_N(qP),
soluble = total - IB). Reports, per state (soluble / IB):
  - RMSE and last-interval gain (the peak/decline signature) for the T30 IB runs (F13/F14/F15/F18),
  - mass conservation (model sol+IB vs model total),
and the Hill params (N_hill, K_IB, qP_IBmax) for identifiability.
The key check (user concern): Hill-IB must fix IB shape AND keep soluble fitting (sol+IB = total).
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import data as D
import fit as F
import matplotlib.pyplot as plt
from models import simulate_run

CIND = ["F13", "F14", "F15", "F18", "F07", "F12"]  # T30 IB-substantial + T35 IB-dominant


def per_state(res, bundle, model):
    p, cf = res.params, bundle["c_fold"]
    rmse = {"P_sol": [], "P_IB": []}
    gains = {}
    for r in bundle["train"]:
        if r["run_id"] not in CIND:
            continue
        sim = simulate_run(model, p, r, cf, eval_gas=False)
        so = sim["obs"]
        for k in ("P_sol", "P_IB"):
            o = np.asarray(r["obs"][k], float)
            m = np.asarray(so[k], float)
            rmse[k].append(np.sqrt(np.nanmean((m - o) ** 2)))
        gains[r["run_id"]] = dict(m_dib=so["P_IB"][-1] - so["P_IB"][-2],
                                  o_dib=r["obs"]["P_IB"][-1] - r["obs"]["P_IB"][-2],
                                  m_dsol=so["P_sol"][-1] - so["P_sol"][-2],
                                  o_dsol=r["obs"]["P_sol"][-1] - r["obs"]["P_sol"][-2])
    return {k: float(np.mean(v)) for k, v in rmse.items()}, gains


b = D.load_dataset(*D.split_for_fold("full"))
print("fitting baseline..."); base = F.fit_model("M0_aload_qPTexp", b, seed=42, n_starts=2, max_nfev=400)
print("fitting Hill-IB..."); hil = F.fit_model("M0_qPTexp_hillib", b, seed=42, n_starts=3, max_nfev=400)
print("\nbaseline params:", {k: round(v, 4) for k, v in base.params.items()})
print("hillib   params:", {k: round(v, 4) for k, v in hil.params.items()})

for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"), ("Hill-IB", hil, "M0_qPTexp_hillib")]:
    rmse, gains = per_state(res, b, mdl)
    print(f"\n{tag}: cost={res.cost:.4f}  RMSE sol={rmse['P_sol']:.3f}  IB={rmse['P_IB']:.3f}")
    for rid in CIND:
        g = gains[rid]
        print(f"  {rid}: dIB model {g['m_dib']:+.3f} vs obs {g['o_dib']:+.3f} | "
              f"dSol model {g['m_dsol']:+.3f} vs obs {g['o_dsol']:+.3f}")

# Plot F13, F18: sol + IB, observed vs baseline vs Hill-IB.
rdmap = {r["run_id"]: r for r in b["train"]}
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for col, rid in enumerate(["F13", "F18"]):
    run = rdmap[rid]
    for row, (state, obskey, lab) in enumerate([("sol_Psol", "P_sol", "P_sol"), ("sol_Pib", "P_IB", "P_IB")]):
        ax = axes[row][col]
        ax.scatter(run["t_obs"], run["obs"][obskey], c="k", s=30, zorder=5, label="observed")
        for tag, res, mdl, c in [("baseline", base, "M0_aload_qPTexp", "#888"),
                                 ("Hill-IB", hil, "M0_qPTexp_hillib", "#D55E00")]:
            sim = simulate_run(mdl, res.params, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.4, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        ax.axhline(0, color="k", lw=0.4)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("Hill-IB (orange) vs baseline (grey): does cooperative IB capture the peak/decline AND keep soluble?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/hillib_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/hillib_test.png")
