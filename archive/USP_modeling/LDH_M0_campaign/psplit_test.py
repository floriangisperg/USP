"""Test: is the late P_IB plateau/decline a refolding boost to soluble, IB degradation, or dilution?

Fits, at the adopted (N-floor) volume:
  - baseline M0_aload_qPTexp  (no IB-loss term)
  - +refold  (IB->sol, mass-conserving, k_ref)  -- boosts soluble
  - +ibdeg   (IB degradation -> CO2, k_deg)      -- no soluble boost
and baseline at 2xN volume (dilution test: does more dilution alone produce the IB decline?).

Reports cost, the k values, and whether each captures the observed late P_IB decline (F13/F14/F15/F18)
and whether soluble gets a boost (refold) or not.
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import base_reconstruct as BR
import data as D
import fit as F
import matplotlib.pyplot as plt
from models import simulate_run

CIND = ["F13", "F14", "F15", "F18"]  # T30 IB-substantial runs showing the decline
SEED, NS = 42, 2


def fit_at(model, bundle):
    res = F.fit_model(model, bundle, seed=SEED, n_starts=NS, max_nfev=400)
    return res


def late_dib(res, bundle, model):
    """Predicted last-interval dP_IB and dP_sol for the CIND runs vs observed."""
    p, cfold = res.params, bundle["c_fold"]
    rows = {}
    for r in bundle["train"]:
        if r["run_id"] not in CIND:
            continue
        sim = simulate_run(model, p, r, cfold, eval_gas=False)
        so = sim["obs"]
        rows[r["run_id"]] = dict(
            m_dib=so["P_IB"][-1] - so["P_IB"][-2],
            o_dib=r["obs"]["P_IB"][-1] - r["obs"]["P_IB"][-2],
            m_dsol=so["P_sol"][-1] - so["P_sol"][-2],
            o_dsol=r["obs"]["P_sol"][-1] - r["obs"]["P_sol"][-2],
            m_ib_last=so["P_IB"][-1], o_ib_last=r["obs"]["P_IB"][-1],
            m_sol_last=so["P_sol"][-1], o_sol_last=r["obs"]["P_sol"][-1])
    return rows


b_floor = D.load_dataset(*D.split_for_fold("full"))
b_up = BR.apply_volume(b_floor, "upper")

print("fitting baseline (floor)..."); base = fit_at("M0_aload_qPTexp", b_floor)
print("fitting +refold..."); ref = fit_at("M0_aload_qPTexp_refold", b_floor)
print("fitting +ibdeg..."); ibd = fit_at("M0_aload_qPTexp_ibdeg", b_floor)
print("fitting baseline (2xN volume, dilution)..."); base_up = fit_at("M0_aload_qPTexp", b_up)

for tag, res, mdl, bun in [("baseline", base, "M0_aload_qPTexp", b_floor),
                           ("+refold", ref, "M0_aload_qPTexp_refold", b_floor),
                           ("+ibdeg", ibd, "M0_aload_qPTexp_ibdeg", b_floor),
                           ("baseline@2xN", base_up, "M0_aload_qPTexp", b_up)]:
    rows = late_dib(res, bun, mdl)
    kref = res.params.get("k_ref", 0.0); kdeg = res.params.get("k_deg", 0.0)
    print(f"\n{tag}: cost={res.cost:.4f}  k_ref={kref:.3f} k_deg={kdeg:.3f}")
    for rid in CIND:
        d = rows[rid]
        print(f"  {rid}: dIB model {d['m_dib']:+.3f} vs obs {d['o_dib']:+.3f} | "
              f"dSol model {d['m_dsol']:+.3f} vs obs {d['o_dsol']:+.3f}")

# Plot F13 + F18: observed + baseline + refold + ibdeg trajectories.
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fits = [("baseline", base, "M0_aload_qPTexp", b_floor, "#888"),
        ("+refold", ref, "M0_aload_qPTexp_refold", b_floor, "#0072B2"),
        ("+ibdeg", ibd, "M0_aload_qPTexp_ibdeg", b_floor, "#D55E00"),
        ("baseline@2xN", base_up, "M0_aload_qPTexp", b_up, "#009E73")]
rdmap = {r["run_id"]: r for r in b_floor["train"]}
for col, rid in enumerate(["F13", "F18"]):
    run = rdmap[rid]
    for row, state, lab in [(0, "sol_Psol", "P_sol"), (1, "sol_Pib", "P_IB")]:
        ax = axes[row][col]
        obs = run["obs"]["P_sol"] if "sol" in lab else run["obs"]["P_IB"]
        ax.scatter(run["t_obs"], obs, c="k", s=30, zorder=5, label="observed")
        for tag, res, mdl, bun, c in fits:
            sim = simulate_run(mdl, res.params, rdmap[rid], bun["c_fold"], eval_gas=False)
            ax.plot(sim["sol_t"], sim[state], "-", c=c, lw=1.3, label=tag)
        ax.set_title(f"{rid} {lab}", fontsize=9)
        ax.axhline(0, color="k", lw=0.4)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
fig.suptitle("Late P_IB decline: refold (blue) vs ibdeg (orange) vs baseline (grey) vs 2xN volume (green)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/psplit_test.png", dpi=110)
plt.close(fig)
print("\nsaved results/psplit_test.png")
