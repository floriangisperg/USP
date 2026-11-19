"""Test M_decoupled_ibtransient (decoupled sol/IB + T-gated clock on IB), 2026-07-28.

Reframe (per the data look): sol and IB have DIFFERENT dynamics -- sol a steady accumulator
(present T<=30, absent T35, never declines); IB a T30-transient (peaks then declines) but T35-
sustained (keeps rising). So drop the partition: independent qP_sol/qP_IB, each amplitude*Monod(qS)
*complementary T-gate*eta; sol has no decline channel, IB declines iff formation<dilution. IB carries
a T-gated clock z=exp(-t_ind*gSol/tau) (option A): shutoff at T30 (gSol~1) -> decline; z=1 at T35
(gSol~0) -> sustained. Re-tests ibtransient where qP_IB_max is free to reach the peak.

Evaluation reframe: fold B holds out ALL THREE T30/qs0.4 centers -> train has only F18 as a T30-IB
run, so it structurally CANNOT learn the T30 transient and predict the centers. Fold C (holdout F18,
train on F13/14/15 centers) is the real IB-transient generalization test; fold A tests the low-IB T30
corner. Judge on: full-data fit within triplicate SNR + folds A/C + tau_IB interiority/stability;
fold B reported but flagged T30-center-blind.
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
        rmse_sol = np.sqrt(np.nanmean((so["P_sol"] - r["obs"]["P_sol"]) ** 2))
        rmse_ib = np.sqrt(np.nanmean((so["P_IB"] - r["obs"]["P_IB"]) ** 2))
        dIB = so["P_IB"][-1] - so["P_IB"][-2]
        dIB_obs = r["obs"]["P_IB"][-1] - r["obs"]["P_IB"][-2]
        out[rid] = (rmse_sol, rmse_ib, dIB, dIB_obs, so["P_IB"][-1], r["obs"]["P_IB"][-1])
    return out


print("fitting baseline (M0_aload_qPTexp, full)...")
base = F.fit_model("M0_aload_qPTexp", b, seed=SEED, n_starts=NS, max_nfev=400)
print("fitting M_decoupled_ibtransient (full)...")
dec = F.fit_model("M_decoupled_ibtransient", b, seed=SEED, n_starts=4, max_nfev=500)

print("\nbaseline:", {k: round(v, 4) for k, v in base.params.items()})
print("decoupled:", {k: round(v, 4) for k, v in dec.params.items()})
print(f"\nbaseline  cost={base.cost:.4f}")
print(f"decoupled cost={dec.cost:.4f}")

for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"),
                      ("decoupled", dec, "M_decoupled_ibtransient")]:
    d = diag(res, b, mdl, T30 + T35)
    print(f"\n{tag}: (rmse_sol, rmse_ib, dIB_mdl, dIB_obs, IBend_mdl, IBend_obs)")
    for rid in T30 + T35:
        rs, ri, dm, do, em, eo = d[rid]
        print(f"  {rid}: sol={rs:.3f} ib={ri:.3f}  dIB {dm:+.3f}/{do:+.3f}  "
              f"IBend {em:.2f}/{eo:.2f}")

# Triplicate SNR: F13/F14/F15 -- model residual vs the across-triplicate SD (empirical noise floor).
print("\ntriplicate SNR (F13/F14/F15 P_IB): residual vs across-triplicate SD")
rdmap = {r["run_id"]: r for r in b["train"]}
for model_tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"),
                            ("decoupled", dec, "M_decoupled_ibtransient")]:
    print(f"  {model_tag}:")
    for k in range(3):  # post-induction sample index (F13/14/15 ~aligned)
        obs = [rdmap[rid]["obs"]["P_IB"][k] for rid in ["F13", "F14", "F15"]]
        sd = float(np.std(obs, ddof=1))
        for rid in ["F13", "F14", "F15"]:
            sim = simulate_run(mdl, res.params, rdmap[rid], b["c_fold"], eval_gas=False)
            resid = sim["obs"]["P_IB"][k] - rdmap[rid]["obs"]["P_IB"][k]
            flag = "OK" if abs(resid) <= 2 * sd else ">"
            print(f"    {rid}[{k}] obs={obs[['F13','F14','F15'].index(rid)]:.2f} sd={sd:.3f} "
                  f"resid={resid:+.3f} ({abs(resid)/sd:.1f} SD) {flag}")

# tau_IB profile (identifiability arbiter).
print("\nprofile tau_IB (full data):")
for tv in [1.0, 2.0, 3.0, 6.0, 12.0]:
    r = F.fit_model("M_decoupled_ibtransient", b, seed=SEED, n_starts=2, max_nfev=400,
                    fixed={"tau_IB": tv})
    print(f"  tau_IB={tv:>5.1f}: cost={r.cost:.4f}  qP_IB_max={r.params['qP_IB_max']:.3f}  "
          f"qP_sol_max={r.params['qP_sol_max']:.3f}")

# Held-out folds A (F17), C (F18) -- the meaningful T30 tests -- and B (centers, blind).
for fold, held in [("A", ["F17"]), ("C", ["F18"]), ("B", ["F13", "F14", "F15"])]:
    tr, te = D.split_for_fold(fold)
    bf = D.load_dataset(tr, te)
    ho = F.fit_model("M_decoupled_ibtransient", bf, seed=SEED, n_starts=4, max_nfev=500)
    tag = "  (T30-center-BLIND)" if fold == "B" else ""
    print(f"\nheld-out fold {fold} {held}{tag}: tau_IB={ho.params['tau_IB']:.2f} "
          f"qP_IB_max={ho.params['qP_IB_max']:.3f} (full {dec.params['qP_IB_max']:.3f})")
    rdmap = {r["run_id"]: r for r in bf["test"]}
    for rid in held:
        r = rdmap[rid]
        sim = simulate_run("M_decoupled_ibtransient", ho.params, r, bf["c_fold"], eval_gas=False)
        so = sim["obs"]
        ri = np.sqrt(np.nanmean((so["P_IB"] - r["obs"]["P_IB"]) ** 2))
        print(f"  {rid}: IBend {so['P_IB'][-1]:.2f}/{r['obs']['P_IB'][-1]:.2f}  "
              f"dIB {so['P_IB'][-1]-so['P_IB'][-2]:+.3f}/{r['obs']['P_IB'][-1]-r['obs']['P_IB'][-2]:+.3f}  "
              f"IB RMSE {ri:.3f}  | solend {so['P_sol'][-1]:.2f}/{r['obs']['P_sol'][-1]:.2f}")

# Plot: T30 (F13/14/15/18) and T35 (F07/19), sol + IB.
rdmap = {r["run_id"]: r for r in b["train"]}
fig, axes = plt.subplots(2, 5, figsize=(17, 7))
for col, rid in enumerate(["F13", "F14", "F15", "F18", "F07"]):
    run = rdmap[rid]
    for row, (state, obsk, lab) in enumerate([("sol_Psol", "P_sol", "P_sol"),
                                              ("sol_Pib", "P_IB", "P_IB")]):
        ax = axes[row][col]
        ax.scatter(run["t_obs"], run["obs"][obsk], c="k", s=28, zorder=5, label="obs")
        for tag, res, mdl, c in [("baseline", base, "M0_aload_qPTexp", "#888"),
                                 ("decoupled", dec, "M_decoupled_ibtransient", "#D55E00")]:
            sim = simulate_run(mdl, res.params, run, b["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.4, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
fig.suptitle("M_decoupled_ibtransient (orange) vs baseline (grey): sol steady / IB T30-transient+T35-sustained?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/decoupled_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/decoupled_test.png")
