"""Test M_decoupled_v2 (INDEPENDENT gates + T30-gated clock), 2026-07-28.

v1 (complementary gates g_sol+g_ib=1) failed: g_ib(T30)->0.05 starved IB -> qP_IB_max pinned at its
bound -> the clock was rejected (tau->20) and the T30 decline was not captured. v2 uses INDEPENDENT
(non-complementary) gates so both sol and IB can be fully on at T30 (data has both); a moderate
qP_IB_max should now reach the IB peak and the clock (z=exp(-t_ind*g_sol/tau), T30-only) should engage
to decline it. This is the test of whether decoupling-done-right captures the dynamics.

Eval: full-data fit + triplicate SNR (F13/14/15) + folds A (F17) and C (F18) held-out (the meaningful
T30 tests) + tau_IB profile. (Fold B -- all T30 centers out -- is T30-center-blind, skipped here.)
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import data as D
import fit as F
import matplotlib.pyplot as plt
from models import simulate_run

T30 = ["F13", "F14", "F15", "F18"]
T35 = ["F07", "F19", "F12"]
MDL = "M_decoupled_v2"
SEED, NS = 42, 3

b = D.load_dataset(*D.split_for_fold("full"))


def diag(res, bundle, model, runs):
    p, cf = res.params, bundle["c_fold"]
    rdmap = {r["run_id"]: r for r in bundle["train"]}
    out = {}
    for rid in runs:
        r = rdmap[rid]
        sim = simulate_run(model, p, r, cf, eval_gas=False)
        so = sim["obs"]
        out[rid] = (np.sqrt(np.nanmean((so["P_sol"] - r["obs"]["P_sol"]) ** 2)),
                    np.sqrt(np.nanmean((so["P_IB"] - r["obs"]["P_IB"]) ** 2)),
                    so["P_IB"][-1] - so["P_IB"][-2], r["obs"]["P_IB"][-1] - r["obs"]["P_IB"][-2],
                    so["P_IB"][-1], r["obs"]["P_IB"][-1])
    return out


print("fitting baseline (full)...")
base = F.fit_model("M0_aload_qPTexp", b, seed=SEED, n_starts=2, max_nfev=400)
print(f"fitting {MDL} (full)...")
v2 = F.fit_model(MDL, b, seed=SEED, n_starts=4, max_nfev=500)
print("\nbaseline cost:", round(base.cost, 4))
print("v2 params:", {k: round(val, 4) for k, val in v2.params.items()})
print(f"v2 cost: {v2.cost:.4f}  (baseline {base.cost:.4f})")

# Gate values at the fitted params (the v2 hypothesis check: g_sol and g_ib both ~1 at T30?).
T_REF = 30.0
p = v2.params
for T in [25, 30, 35]:
    gs = 1.0 / (1.0 + np.exp(-(p["theta_sol_0"] - p["theta_sol_T"] * (T - T_REF))))
    gib = 1.0 / (1.0 + np.exp(-(p["theta_ib_0"] + p["theta_ib_T"] * (T - T_REF))))
    print(f"  T={T}: g_sol={gs:.2f} g_ib={gib:.2f}")

for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"), ("v2", v2, MDL)]:
    d = diag(res, b, mdl, T30 + T35)
    print(f"\n{tag}: (rmse_sol, rmse_ib, dIB_mdl, dIB_obs, IBend_mdl, IBend_obs)")
    for rid in T30 + T35:
        rs, ri, dm, do, em, eo = d[rid]
        print(f"  {rid}: sol={rs:.3f} ib={ri:.3f}  dIB {dm:+.3f}/{do:+.3f}  IBend {em:.2f}/{eo:.2f}")

# tau_IB profile (identifiability).
print("\nprofile tau_IB (full data):")
for tv in [1.0, 2.0, 3.0, 6.0, 12.0]:
    r = F.fit_model(MDL, b, seed=SEED, n_starts=2, max_nfev=400, fixed={"tau_IB": tv})
    print(f"  tau_IB={tv:>5.1f}: cost={r.cost:.4f}  qP_IB_max={r.params['qP_IB_max']:.3f}  "
          f"qP_sol_max={r.params['qP_sol_max']:.3f}")

# Held-out folds A (F17, low-IB T30 corner) and C (F18, the IB-transient test).
rdmap0 = {r["run_id"]: r for r in b["train"]}
for fold, held in [("A", ["F17"]), ("C", ["F18"])]:
    tr, te = D.split_for_fold(fold)
    bf = D.load_dataset(tr, te)
    ho = F.fit_model(MDL, bf, seed=SEED, n_starts=4, max_nfev=500)
    print(f"\nheld-out fold {fold} {held}: tau_IB={ho.params['tau_IB']:.2f} "
          f"qP_IB_max={ho.params['qP_IB_max']:.3f} (full {v2.params['qP_IB_max']:.3f})")
    rdm = {r["run_id"]: r for r in bf["test"]}
    for rid in held:
        r = rdm[rid]
        sim = simulate_run(MDL, ho.params, r, bf["c_fold"], eval_gas=False)
        so = sim["obs"]
        ri = np.sqrt(np.nanmean((so["P_IB"] - r["obs"]["P_IB"]) ** 2))
        print(f"  {rid}: IBend {so['P_IB'][-1]:.2f}/{r['obs']['P_IB'][-1]:.2f}  "
              f"dIB {so['P_IB'][-1]-so['P_IB'][-2]:+.3f}/{r['obs']['P_IB'][-1]-r['obs']['P_IB'][-2]:+.3f}  "
              f"IB RMSE {ri:.3f}  | solend {so['P_sol'][-1]:.2f}/{r['obs']['P_sol'][-1]:.2f}")

# Plot T30 (F13/14/15/18) + T35 (F07): sol + IB, obs vs baseline vs v2.
fig, axes = plt.subplots(2, 5, figsize=(17, 7))
for col, rid in enumerate(["F13", "F14", "F15", "F18", "F07"]):
    run = rdmap0[rid]
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
fig.suptitle("M_decoupled_v2 independent gates + T30 clock (blue) vs baseline (grey)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/decoupled_v2_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/decoupled_v2_test.png")
