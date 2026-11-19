"""Test M0_proteostasis_v2 (load-coupled folding race), 2026-09-04.

v1 verdict: the rescue-capacity half FAILED (C_chap -> 0, monotone profile; T30 qS-ordering
compressed; cost +9%) but the T-proteostasis half WORKED (T50_m ~ 30.3 C step, sigma_IB = 0.66
interior+bounded -> 34% T35 loss = alpha's eta(35)=0.64 by an independent route). Diagnosis: the
load signal must enter through the folding RATE, not a rescue flux.

v2: k_f_eff = k_f*(1 - q_syn/C_cap) -- the race is lost where synthesis load approaches the
proteome folding capacity. Load still T-gated (the e^(theta_m*(T50_m-T)) ratio keeps T25 soluble
at any load). Same 7 params as baseline.

Eval: full-data fit vs baseline + phi_m regime/load map + knife-edge diagnostic (q_syn0 vs C_cap)
+ per-run IB fractions + triplicate z + profiles (C_cap, theta_m, T50_m) + held-out folds A/B/C
(the gates: interior params, bounded profiles, fold-A/C IB parity with the canonical baseline).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D
import fit as F
from models import simulate_run

MDL = "M0_proteostasis_v2"
SEED = 42

b = D.load_dataset(*D.split_for_fold("full"))

print("fitting baseline (M0_aload_qPTexp, full)...")
base = F.fit_model("M0_aload_qPTexp", b, seed=SEED, n_starts=2, max_nfev=400)
print("fitting v2 (full)...")
v2 = F.fit_model(MDL, b, seed=SEED, n_starts=4, max_nfev=500)

print("\nbaseline cost:", round(base.cost, 4), " params:",
      {k: round(v, 4) for k, v in base.params.items()})
print(f"v2 cost: {v2.cost:.4f}  params:",
      {k: round(v, 4) for k, v in v2.params.items()})

# phi_m regime/load map at the fitted params (per T at the three realized-qS levels).
p = v2.params
print("\nphi_m(T, qs) at fitted params:")
print(f"{'T':>4} {'qs=0.17':>9} {'qs=0.27':>9} {'qs=0.30':>9}")
for T in [25, 30, 35]:
    row = []
    for qs in [0.17, 0.27, 0.30]:
        qsyn0 = p["q_syn_max"] * qs / (qs + p["K_S"])
        kf = max(1.0 - qsyn0 / p["C_cap"], 1e-6)
        row.append(1.0 / (1.0 + np.exp(p["theta_m"] * (T - p["T50_m"])) * kf))
    print(f"{T:>4} " + " ".join(f"{v_:>9.3f}" for v_ in row))

# Knife-edge diagnostic: q_syn0 of each T30 run vs C_cap (capacity must bind only at high load).
rdmap = {r["run_id"]: r for r in b["train"]}
print(f"\nC_cap = {p['C_cap']:.4f} vs fitted-load q_syn0 per run (T30):")
for rid in ["F17", "F13", "F14", "F15", "F18"]:
    r = rdmap[rid]
    X = max(np.mean(r["obs"]["X"][-2:]), 1e-6)
    # realized qS at final biomass from the run's mean feed (rough diagnostic)
    Fm = np.mean(r["feed_rate_L_h"][-20:]) if len(r["feed_rate_L_h"]) > 20 else np.mean(r["feed_rate_L_h"])
    qs_ = min(Fm * 400.0 / (X * np.mean(r["V_L"][-20:])), 2.0)
    qsyn0 = p["q_syn_max"] * qs_ / (qs_ + p["K_S"])
    print(f"  {rid}: q_syn0 ~ {qsyn0:.4f}  ({qsyn0/p['C_cap']*100:.0f}% of C_cap)")

print("\nper-run endpoints (obs | baseline | v2):")
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

obs_ctr = [rdmap[rid]["obs"]["P_sol"][-1] + rdmap[rid]["obs"]["P_IB"][-1]
           for rid in ["F13", "F14", "F15"]]
sd = np.std(obs_ctr, ddof=1)
mu_obs = np.mean(obs_ctr)
z = [(simulate_run(MDL, p, rdmap[rid], b["c_fold"], eval_gas=False)["obs"]["P_sol"][-1]
      + simulate_run(MDL, p, rdmap[rid], b["c_fold"], eval_gas=False)["obs"]["P_IB"][-1]
      - mu_obs) / sd for rid in ["F13", "F14", "F15"]]
print(f"triplicate Ptot z (obs mu={mu_obs:.2f} sd={sd:.2f}) v2: "
      + " ".join(f"{zi:+.1f}" for zi in z))

# Profile likelihoods (identifiability gate) on the structural params.
print("\nprofiles (full data):")
for nm, grid in [("C_cap", [0.008, 0.014, 0.019, 0.026, 0.04]),
                 ("theta_m", [0.5, 1.0, 2.0, 3.0, 4.5]),
                 ("T50_m", [30.0, 31.0, 31.5, 32.5, 34.0])]:
    print(f"  {nm}:")
    for v in grid:
        r = F.fit_model(MDL, b, seed=SEED, n_starts=1, max_nfev=300, fixed={nm: v})
        print(f"    {nm}={v:>7.4f}: cost={r.cost:.4f}")

# Held-out folds A (F17 low-load) and C (F18 high-load) = the discriminating tests; B for stability.
for fold in ["A", "C", "B"]:
    tr, te = D.split_for_fold(fold)
    bf = D.load_dataset(tr, te)
    ho = F.fit_model(MDL, bf, seed=SEED, n_starts=3, max_nfev=500)
    print(f"\nheld-out fold {fold} v2: cost={ho.cost:.4f} "
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
                                 ("v2", v2, MDL, "#0072B2")]:
            sim = simulate_run(mdl, res.params, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.4, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
fig.suptitle("M0_proteostasis_v2 load-coupled race (blue) vs M0_aload_qPTexp (grey), full data",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/proteostasis_v2_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/proteostasis_v2_test.png")
print("DONE")
