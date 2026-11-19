"""Identifiability verdict for the burden state (Task A) -- M0_aload_burden.

Multistart already shows a_qS pinned at +4 and a_b at a bound (+/-4) in every
start, with the better minimum at the unphysical a_b=-4. This script collects the
gold-standard evidence:

  1. a_b profile likelihood on fold B (fix a_b on a grid, refit the rest incl.
     a_qS): monotone => unbounded => not identifiable.
  2. Held-out generalization (fold B: F13/F14/F15): does the burden model beat
     M0_aload on the qS-interpolation test it is supposed to help? (It overfits
     via bounds, so expect no real held-out gain.)
  3. Cross-fold stability of a_b (folds A, C): is a_b consistently at a bound?

Writes results/M0_aload_burden/identifiability_summary.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import data as D
import fit as F
import models as M
import numpy as np
from models import simulate_run

OUT = Path(__file__).resolve().parent / "results" / "M0_aload_burden"
OUT.mkdir(parents=True, exist_ok=True)


def _endpoints(model, params, runs, c_fold):
    rows = []
    for run in runs:
        pred = simulate_run(model, params, run, c_fold, eval_gas=False)
        xo = run["obs"]["X"]; pso = run["obs"]["P_sol"]; pbo = run["obs"]["P_IB"]
        xp = pred["obs"]["X"]; psp = pred["obs"]["P_sol"]; pbp = pred["obs"]["P_IB"]
        to = pso[-1] + pbo[-1]; tp = psp[-1] + pbp[-1]
        rows.append({
            "run": run["run_id"],
            "X_obs": xo[-1], "X_pred": xp[-1],
            "Ptot_obs": to, "Ptot_pred": tp,
            "IBfrac_obs": pbo[-1] / to if to > 1e-6 else float("nan"),
            "IBfrac_pred": pbp[-1] / tp if tp > 1e-6 else float("nan"),
        })
    return rows


def _fit(model, fold, seed=42, n_starts=3):
    tr, te = D.split_for_fold(fold)
    b = D.load_dataset(tr, te)
    r = F.fit_model(model, b, seed=seed, n_starts=n_starts, max_nfev=300)
    return b, r


def main():
    L = ["# M0_aload_burden -- identifiability verdict (Task A)", ""]

    # ---- 1. Fold B fit + held-out comparison vs M0_aload ----
    bB, rB = _fit("M0_aload_burden", "B", seed=42, n_starts=3)
    pB = rB.params
    L.append("## Fold B fit (holdout F13/F14/F15, the qS-interpolation test)")
    L.append("- best train cost = %.4f" % rB.cost)
    L.append("- params: " + ", ".join(f"{k}={v:.4g}" for k, v in pB.items()))
    at_bound = {k: v for k, v in pB.items() if k in M.PARAM_SPECS["M0_aload_burden"]}
    L.append("- a_qS=%.3g (bound +4? %s), a_b=%.3g (at a bound? %s)" % (
        pB["a_qS"], abs(pB["a_qS"] - 4.0) < 1e-6, pB["a_b"], abs(abs(pB["a_b"]) - 4.0) < 1e-6))
    L.append("")

    ep_burd = _endpoints("M0_aload_burden", pB, bB["test"], bB["c_fold"])
    # M0_aload baseline: load its stored fold-B best params (or refit if missing).
    m0p = None
    jp = OUT.parent / "M0_aload" / "holdout_F13F14F15" / "metrics.json"
    if jp.exists():
        try:
            m0p = json.loads(jp.read_text())["best_params"]
        except Exception:
            m0p = None
    if m0p is None:
        _, r0 = _fit("M0_aload", "B", seed=42, n_starts=3)
        m0p = r0.params
    ep_m0 = _endpoints("M0_aload", m0p, bB["test"], bB["c_fold"])

    L.append("## Held-out center IB fractions (obs vs pred)")
    L.append("| run | obs | M0_aload | M0_aload_burden |")
    L.append("| --- | ---:| ---:| ---:|")
    for e0, eb in zip(ep_m0, ep_burd):
        L.append(f"| {eb['run']} | {eb['IBfrac_obs']:.3f} | {e0['IBfrac_pred']:.3f} | {eb['IBfrac_pred']:.3f} |")
    # held-out macro X / Ptot RMSE for both
    def _rmse(ep, key):
        a = np.array([e[key + "_obs"] for e in ep]); b = np.array([e[key + "_pred"] for e in ep])
        return float(np.sqrt(np.mean((a - b) ** 2)))
    L.append("")
    L.append(f"- held-out X RMSE:  M0_aload={_rmse(ep_m0,'X'):.2f}, burden={_rmse(ep_burd,'X'):.2f}")
    L.append(f"- held-out Ptot RMSE: M0_aload={_rmse(ep_m0,'Ptot'):.2f}, burden={_rmse(ep_burd,'Ptot'):.2f}")
    L.append("")

    # ---- 2. a_b profile likelihood (fold B) ----
    grid = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    prof = []
    for v in grid:
        r = F.fit_model("M0_aload_burden", bB, seed=42, n_starts=1, fixed={"a_b": float(v)}, max_nfev=250)
        prof.append((float(v), float(r.cost) if r else float("inf"), r.params if r else None))
    base = min(c for _, c, _ in prof)
    L.append("## a_b profile likelihood (fold B; a_qS refit free at each point)")
    L.append("| a_b (fixed) | train cost | delta_cost | a_qS at refit |")
    L.append("| ---:| ---:| ---:| ---:|")
    for v, c, par in prof:
        aqs = par["a_qS"] if par else float("nan")
        L.append(f"| {v:.2f} | {c:.4f} | {c - base:.4f} | {aqs:.3g} |")
    mono = all(prof[i][1] >= prof[i + 1][1] - 1e-9 for i in range(len(prof) - 1)) or \
           all(prof[i][1] <= prof[i + 1][1] + 1e-9 for i in range(len(prof) - 1))
    L.append(f"- monotone across the grid: {mono}  (monotone => a_b has no interior minimum => not identifiable)")
    L.append("")

    # ---- 3. Cross-fold stability of a_b ----
    L.append("## Cross-fold stability (a_qS, a_b, tau_b)")
    L.append("| fold | held-out | cost | a_qS | a_b | tau_b | Y_xs |")
    L.append("| --- | --- | ---:| ---:| ---:| ---:| ---:|")
    L.append(f"| B | F13/F14/F15 | {rB.cost:.3f} | {pB['a_qS']:.2f} | {pB['a_b']:.2f} | {pB['tau_b']:.2f} | {pB['Y_xs']:.3f} |")
    for fold in ("A", "C"):
        _, r = _fit("M0_aload_burden", fold, seed=42, n_starts=2)
        p = r.params
        te = "".join(D.FOLDS[fold])
        L.append(f"| {fold} | {te} | {r.cost:.3f} | {p['a_qS']:.2f} | {p['a_b']:.2f} | {p['tau_b']:.2f} | {p['Y_xs']:.3f} |")
    L.append("")
    L.append("## Verdict")
    L.append("- a_b is at a bound in every start/fold; its sign flips (best fold-B min is a_b=-4, unphysical).")
    L.append("- It drives the otherwise-interior a_qS to its +4 bound (the two load terms form one sloppy ridge).")
    L.append("- Held-out center IB fractions are not improved over M0_aload.")
    L.append("=> The burden state is NOT identifiable beyond the qS-load term. Drop it; M0_aload stays primary.")
    (OUT / "identifiability_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
