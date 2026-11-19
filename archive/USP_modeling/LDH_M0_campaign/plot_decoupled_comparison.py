"""Generate the decoupled-model comparison plot using the already-fitted params (no refit).
baseline (M0_aload_qPTexp), v2 (independent gates + clock, decline-basin), v3 (delayed shutoff).
Fitted params are hardcoded from the session's full-data fits."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt
from models import simulate_run

b = D.load_dataset(*D.split_for_fold("full"))
rdm = {r["run_id"]: r for r in b["train"]}

baseline_p = dict(Y_xs=0.3631, qP_max=0.0462, K_S=0.5633, theta_0=-2.0227, theta_T=4.7138,
                  a_qS=1.1809, alpha_T=0.0964)
v2_p = dict(Y_xs=0.3635, qP_sol_max=0.0306, qP_IB_max=0.1, K_S=0.3189, theta_sol_0=6.0,
            theta_sol_T=1.5694, theta_ib_0=-0.8053, theta_ib_T=8.0, alpha_T=0.1885, tau_IB=0.698)
v3_p = dict(Y_xs=0.3633, qP_sol_max=0.0322, qP_IB_max=0.3, K_S=0.3459, theta_sol_0=6.0,
            theta_sol_T=1.367, theta_ib_0=-3.4537, theta_ib_T=2.5074, alpha_T=0.449,
            t_delay=3.1744, tau_IB=0.7212)

models = [("baseline", "M0_aload_qPTexp", baseline_p, "#888888", "-"),
          ("v2 indep+clock", "M_decoupled_v2", v2_p, "#0072B2", "-"),
          ("v3 delayed shutoff", "M_decoupled_v3", v3_p, "#D55E00", "-")]

runs = ["F13", "F14", "F15", "F18", "F07", "F19"]
fig, axes = plt.subplots(2, len(runs), figsize=(2.9 * len(runs), 6.8))
for col, rid in enumerate(runs):
    run = rdm[rid]
    for row, (state, obsk, lab) in enumerate([("sol_Psol", "P_sol", "P_sol [g/L]"),
                                              ("sol_Pib", "P_IB", "P_IB [g/L]")]):
        ax = axes[row][col]
        ax.scatter(run["t_obs"], run["obs"][obsk], c="k", s=28, zorder=5, label="observed")
        for tag, mdl, p, c, ls in models:
            sim = simulate_run(mdl, p, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], ls, c=c, lw=1.6, label=tag)
        ax.set_title(rid, fontsize=10)
        if row == 1:
            ax.set_xlabel("process time [h]", fontsize=8)
        if col == 0:
            ax.set_ylabel(lab, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Decoupled models vs baseline\nT30 (F13/14/15/18): IB peaks then declines  |  T35 (F07/19): IB sustained",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/decoupled_comparison.png", dpi=120)
plt.close(fig)
print("saved results/decoupled_comparison.png")
