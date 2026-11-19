"""Biomass (X) plot: observed vs baseline vs forced-Y_xs(T) (beta_T=0.05) for all 11 runs.
Shows the T35 over-prediction (baseline above obs) and why Y_xs(T) is too blunt
(fixes F07 but over-corrects F19/F12)."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt
from models import simulate_run

b = D.load_dataset(*D.split_for_fold("full"))
rdm = {r["run_id"]: r for r in b["train"]}

base_p = dict(Y_xs=0.3631, qP_max=0.0462, K_S=0.5633, theta_0=-2.0227, theta_T=4.7138,
              a_qS=1.1809, alpha_T=0.0964)
# forced Y_xs(T) (beta_T=0.05) params: refit with beta_T fixed (from the profile, cost 0.621).
# Re-derive quickly is expensive; instead reuse the profile's mechanism: simulate baseline but with
# mu scaled by exp(-0.05*max(T-30,0)) -- i.e., the M0_aload_qPTexp_yxt model with beta_T=0.05 fixed.
# We refit once here to get consistent params.
import fit as F

yxt05 = F.fit_model("M0_aload_qPTexp_yxt", b, seed=42, n_starts=1, max_nfev=300,
                    fixed={"beta_T": 0.05}).params

ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]
TCOL = {25: "#0072B2", 30: "#009E73", 35: "#D55E00"}
TSET = {"F09": 25, "F16": 25, "F10": 25, "F17": 30, "F13": 30, "F14": 30, "F15": 30,
        "F18": 30, "F07": 35, "F19": 35, "F12": 35}

fig, axes = plt.subplots(3, 4, figsize=(14, 9))
axes = axes.ravel()
for i, rid in enumerate(ORDER):
    ax = axes[i]
    r = rdm[rid]
    T = TSET[rid]
    ax.scatter(r["t_obs"], r["obs"]["X"], c="k", s=22, zorder=5, label="observed")
    sb = simulate_run("M0_aload_qPTexp", base_p, r, b["c_fold"], eval_gas=False)
    ax.plot(sb["sol_t"], sb["sol_X"], "-", c="#888888", lw=1.5, label="baseline")
    sy = simulate_run("M0_aload_qPTexp_yxt", yxt05, r, b["c_fold"], eval_gas=False)
    ax.plot(sy["sol_t"], sy["sol_X"], "-", c=TCOL[T], lw=1.5, label="Y_xs(T) b=0.05")
    ax.set_title(f"{rid}  (T={T})", fontsize=9, color=TCOL[T])
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    if i % 4 == 0:
        ax.set_ylabel("X [g/L]", fontsize=8)
    if i >= 7:
        ax.set_xlabel("process time [h]", fontsize=8)
    if i == 0:
        ax.legend(fontsize=7, loc="lower right")
axes[11].axis("off")
fig.suptitle("Biomass X: baseline (grey) over-predicts T35 (F07/F19/F12); Y_xs(T) b=0.05 (coloured) "
             "fixes F07 but over-corrects F19/F12\n(too blunt -> rejected; deviation is run-specific / volume)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/biomass_x.png", dpi=120)
plt.close(fig)
print("saved results/biomass_x.png")
