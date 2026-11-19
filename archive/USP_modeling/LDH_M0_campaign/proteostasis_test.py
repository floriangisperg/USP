"""Test M0_proteostasis (chaperone-capacity folding competition), 2026-09-04.

Replaces the adopted model's two phenomenological pieces -- the logit partition
(theta_0/theta_T/a_qS) and the alpha_T hinge -- with ONE mechanism: synthesis -> fold/misfold
race (phi_m(T) = k_m/(k_f+k_m)), chaperone rescue up to a finite capacity C_chap, unrescued
overflow -> recoverable IB (sigma_IB) vs proteolysis -> CO2 (the alpha_T channel, CER-tested).

Hypothesis: the three fitted response surfaces phi(T), phi(qS), eta(T) are one proteostasis
mechanism. Success = match M0_aload_qPTexp at EQUAL parameter count with interpretable units
(regime from phi_m, T30 qS-ordering from capacity overflow, T35 loss from proteolysis), AND
survive the identifiability gates: interior params, bounded profiles, held-out parity.

Eval: full-data fit vs baseline + phi_m regime check + T30 final-IB-fraction ordering (the
a_qS replacement test) + triplicate SNR (F13/14/15) + profiles (C_chap, T50_m, sigma_IB) +
held-out folds A/B/C vs the canonical baseline numbers + trajectories.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D
import fit as F
from models import simulate_run

T30 = ["F13", "F14", "F15", "F18"]
MDL = "M0_proteostasis"
SEED, NS = 42, 4

b = D.load_dataset(*D.split_for_fold("full"))

print("fitting baseline (M0_aload_qPTexp, full)...")
base = F.fit_model("M0_aload_qPTexp", b, seed=SEED, n_starts=2, max_nfev=400)
print("fitting proteostasis (full)...")
prot = F.fit_model(MDL, b, seed=SEED, n_starts=NS, max_nfev=500)
# 6-param sub-variant: theta_m pinned (3 T strata cannot pin a slope).
prot_s1 = F.fit_model(MDL, b, seed=SEED, n_starts=NS, max_nfev=500,
                      fixed={"theta_m": 1.5})

print("\nbaseline cost:", round(base.cost, 4), " params:",
      {k: round(v, 4) for k, v in base.params.items()})
print(f"proteo cost: {prot.cost:.4f}  params:",
      {k: round(v, 4) for k, v in prot.params.items()})
print(f"proteo_s1 (theta_m=1.5) cost: {prot_s1.cost:.4f}  params:",
      {k: round(v, 4) for k, v in prot_s1.params.items()})

# Regime check: phi_m at the three FCCD temperatures (want ~0 / small / ~1).
p = prot.params
for T in [25, 30, 35]:
    phi_m = 1.0 / (1.0 + np.exp(-p["theta_m"] * (T - p["T50_m"])))
    print(f"  phi_m(T={T}) = {phi_m:.3f}")

# Diagnostics per run: final IB fraction (the a_qS replacement test) + Ptot + X.
rdmap = {r["run_id"]: r for r in b["train"]}
print("\nper-run endpoints (obs | baseline | proteo):")
print(f"{'run':>4} {'T':>4} {'IBfrac':>18} {'Ptot':>18} {'X':>18}")
for rid in ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]:
    r = rdmap[rid]
    simb = simulate_run("M0_aload_qPTexp", base.params, r, b["c_fold"], eval_gas=False)
    simp = simulate_run(MDL, p, r, b["c_fold"], eval_gas=False)

    def ibfrac(o):
        tot = o["P_sol"][-1] + o["P_IB"][-1]
        return o["P_IB"][-1] / tot if tot > 1e-9 else np.nan

    def fmt(a, b_, c):
        return f"{a:.3f} | {b_:.3f} | {c:.3f}"
    print(f"{rid:>4} {r['temp'][-1]:>4.0f} "
          f"{fmt(ibfrac(r['obs']), ibfrac(simb['obs']), ibfrac(simp['obs'])):>18} "
          f"{fmt(r['obs']['P_sol'][-1] + r['obs']['P_IB'][-1],
                 simb['obs']['P_sol'][-1] + simb['obs']['P_IB'][-1],
                 simp['obs']['P_sol'][-1] + simp['obs']['P_IB'][-1]):>18} "
          f"{fmt(r['obs']['X'][-1], simb['obs']['X'][-1], simp['obs']['X'][-1]):>18}")

# Triplicate noise floor (F13/14/15): is the proteostasis Ptot within triplicate SNR?
obs_ctr = [rdmap[rid]["obs"]["P_sol"][-1] + rdmap[rid]["obs"]["P_IB"][-1]
           for rid in ["F13", "F14", "F15"]]
sd = np.std(obs_ctr, ddof=1)
mu_obs = np.mean(obs_ctr)
for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"), ("proteo", prot, MDL)]:
    z = [(simulate_run(mdl, res.params, rdmap[rid], b["c_fold"], eval_gas=False)["obs"]["P_sol"][-1]
          + simulate_run(mdl, res.params, rdmap[rid], b["c_fold"], eval_gas=False)["obs"]["P_IB"][-1]
          - mu_obs) / sd for rid in ["F13", "F14", "F15"]]
    print(f"triplicate Ptot z (obs mu={mu_obs:.2f} sd={sd:.2f}) {tag}: "
          + " ".join(f"{zi:+.1f}" for zi in z))

# Profile likelihoods (identifiability gate) on the three new structural params.
print("\nprofiles (full data, theta_m free):")
for nm, grid in [("C_chap", [0.0005, 0.002, 0.004, 0.008, 0.02]),
                 ("T50_m", [30.0, 31.0, 31.5, 32.0, 33.5, 35.0]),
                 ("sigma_IB", [0.2, 0.4, 0.6, 0.8])]:
    print(f"  {nm}:")
    for v in grid:
        r = F.fit_model(MDL, b, seed=SEED, n_starts=1, max_nfev=300, fixed={nm: v})
        print(f"    {nm}={v:>7.4f}: cost={r.cost:.4f}")

# Held-out folds vs the canonical baseline (HANDOFF 2026-09-04 refit numbers).
canon = {"A": ("F17", "IB RMSE 0.181"), "B": ("F13F14F15", "Ptot bias -0.85"),
         "C": ("F18", "Ptot bias +0.69")}
for fold in ["A", "B", "C"]:
    tr, te = D.split_for_fold(fold)
    bf = D.load_dataset(tr, te)
    ho = F.fit_model(MDL, bf, seed=SEED, n_starts=3, max_nfev=500)
    print(f"\nheld-out fold {fold} proteo: cost={ho.cost:.4f} "
          f"params={ {k: round(v, 4) for k, v in ho.params.items()} }")
    rdm = {r["run_id"]: r for r in bf["test"]}
    for rid, r in rdm.items():
        sim = simulate_run(MDL, ho.params, r, bf["c_fold"], eval_gas=False)
        o, s = r["obs"], sim["obs"]
        ri = np.sqrt(np.nanmean((s["P_IB"] - o["P_IB"]) ** 2))
        ptot_bias = (s["P_sol"][-1] + s["P_IB"][-1]) - (o["P_sol"][-1] + o["P_IB"][-1])
        print(f"  {rid}: Xend {s['X'][-1]:.1f}/{o['X'][-1]:.1f}  IB RMSE {ri:.3f}  "
              f"Ptot bias {ptot_bias:+.2f}  "
              f"IBfrac {s['P_IB'][-1]/max(s['P_sol'][-1]+s['P_IB'][-1],1e-9):.3f}"
              f"/{o['P_IB'][-1]/max(o['P_sol'][-1]+o['P_IB'][-1],1e-9):.3f}")

# Trajectories: T30 (F13/F15/F18) + T35 (F07) + T25 (F10), sol + IB, obs vs both models.
fig, axes = plt.subplots(2, 5, figsize=(17, 7))
for col, rid in enumerate(["F13", "F15", "F18", "F07", "F10"]):
    run = rdmap[rid]
    for row, (state, obsk, lab) in enumerate([("sol_Psol", "P_sol", "P_sol"),
                                              ("sol_Pib", "P_IB", "P_IB")]):
        ax = axes[row][col]
        ax.scatter(run["t_obs"], run["obs"][obsk], c="k", s=28, zorder=5, label="obs")
        for tag, res, mdl, c in [("baseline", base, "M0_aload_qPTexp", "#888"),
                                 ("proteo", prot, MDL, "#0072B2")]:
            sim = simulate_run(mdl, res.params, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.4, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
fig.suptitle("M0_proteostasis (blue) vs M0_aload_qPTexp (grey), full-data fits", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/proteostasis_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/proteostasis_test.png")
print("DONE")
