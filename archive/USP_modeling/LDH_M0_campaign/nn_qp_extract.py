"""Extract the NN's (m13) learned qP_sol / qP_IB trajectories and find the parametric form.

The NN drives independent qP_sol, qP_IB by (T, cumulative_substrate_per_X0, t_since_induction) -- NOT qS.
The diagnostics CSV has the learned qP at sample times. Compute the cumulative load (S_met/X0) and
t_since_induction, then see how qP_sol / qP_IB depend on them -> the parametric decline form to port.
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt

S_IN = 400.0
CSV = ("C:/Users/fgisperg/OneDrive - TU Wien/01_Uni/01_PhD/05_Projects/HybridModel/"
       "examples/LDH_FCCD/results/m13_independent_qp_deep8_staged/full_data_seed42/"
       "independent_qp_diagnostics.csv")
SET = {"F07": (35, .2), "F09": (25, .2), "F10": (25, .6), "F12": (35, .6), "F13": (30, .4),
       "F14": (30, .4), "F15": (30, .4), "F16": (25, .4), "F17": (30, .2), "F18": (30, .6), "F19": (35, .4)}

df = pd.read_csv(CSV)
df = df[df["split"] == "training"].copy()
bundle = D.load_dataset(*D.split_for_fold("full"))
rdmap = {r["run_id"]: r for r in bundle["train"]}

rows = []
for rid, g in df.groupby("run"):
    if rid not in rdmap:
        continue
    run = rdmap[rid]
    t_on = np.asarray(run["t_online"], float)
    feed = np.asarray(run["feed_rate_L_h"], float)
    V = np.asarray(run["V_L"], float)
    Dn = feed / np.where(V > 1e-6, V, np.nan)
    cumD = np.concatenate([[0.0], np.cumsum((Dn[:-1] + Dn[1:]) * 0.5 * np.diff(t_on))])  # int D dt from t0
    cum_sub_per_X0 = S_IN * cumD / run["X0"]                                   # ~ S_met / X0
    ind = np.asarray(run["is_induced"], float) > 0.5
    t_ind = float(t_on[ind][0]) if ind.any() else float(t_on[0])
    g = g.sort_values("time_h")
    for _, row in g.iterrows():
        t = float(row["time_h"])
        if t < t_ind - 0.1:
            continue
        rows.append(dict(run=rid, T=SET[rid][0], qs=SET[rid][1], t=t,
                         t_ind=t - t_ind,
                         cum_sub=float(np.interp(t, t_on, cum_sub_per_X0)),
                         qP_sol=float(row["qP_sol_g_g_h"]),
                         qP_IB=float(row["qP_IB_net_g_g_h"])))
R = pd.DataFrame(rows)
R.to_csv("results/nn_qp_extracted.csv", index=False)

print(f"{'run':>4} {'T':>3} {'qs':>4} | t_ind / cum_sub / qP_sol / qP_IB (post-induction samples)")
for rid in ["F13", "F14", "F15", "F18", "F17", "F10", "F07", "F12", "F19"]:
    g = R[R["run"] == rid].sort_values("t_ind")
    if len(g) == 0:
        continue
    print(f"{rid:>4} {g.iloc[0]['T']:>3.0f} {g.iloc[0]['qs']:>4} |")
    for _, r in g.iterrows():
        print(f"        t_ind={r['t_ind']:>5.1f}h cumS/X0={r['cum_sub']:>5.1f} "
              f"qP_sol={r['qP_sol']:.5f} qP_IB={r['qP_IB']:.5f}")

# Plot qP_sol and qP_IB vs cumulative load and vs t_ind, pooled, colored by T.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
Tcol = {25: "#56B4E9", 30: "#0072B2", 35: "#E69F00"}
for (xy, xlab, title), ax in zip([(("cum_sub", "qP_sol"), "cumulative substrate / X0", "qP_sol vs cumulative load"),
                                  (("cum_sub", "qP_IB"), "cumulative substrate / X0", "qP_IB (net) vs cumulative load"),
                                  (("t_ind", "qP_sol"), "t since induction [h]", "qP_sol vs induction time"),
                                  (("t_ind", "qP_IB"), "t since induction [h]", "qP_IB (net) vs induction time")],
                                 axes.ravel()):
    for T in (25, 30, 35):
        s = R[R["T"] == T]
        ax.scatter(s[xy[0]], s[xy[1]], c=Tcol[T], s=45, edgecolor="k", lw=0.3, label=f"T{T}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(xlab); ax.set_ylabel(xy[1] + " [g/g/h]"); ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("NN (m13) learned production rates: qP_sol sustained, qP_IB declines with cumulative load/time", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/nn_qp_extracted.png", dpi=120)
plt.close(fig)
print("\nsaved results/nn_qp_extracted.png + results/nn_qp_extracted.csv")
