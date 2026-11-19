"""Bracket re-fit: does correcting the missing base volume change M0_aload's
conclusions?

Reconstructs V(t) with base = N-floor x {0, 1, 1.5, 2} (pump-shaped) and re-fits
M0_aload on fold B (holdout F13/F14/F15, the qS-interpolation + center-underprediction
test) at each level. Reports whether held-out center IB fractions, the center
total-product underprediction, Y_xs, and the partition coefficients are robust to the
~factor-of-2 volume uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import data as D
import fit as F
import models as M
import numpy as np
from base_reconstruct import LEVELS, apply_volume
from models import simulate_run

OUT = Path(__file__).resolve().parent / "results" / "base_volume"
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


def main():
    tr, te = D.split_for_fold("B")
    base_bundle = D.load_dataset(tr, te)
    c_fold = base_bundle["c_fold"]
    L = ["# M0_aload bracket re-fit with reconstructed base volume", "",
         "Base level: multiplier on the N-floor (pump-shaped trajectory).",
         "- none  = base 0 (current model)", "- floor = 1x N-floor (assimilation lower bound)",
         "- mid   = 1.5x N-floor", "- upper = 2x N-floor (scfv acid-neutralization analog)", ""]

    summary = []
    detail = {}
    for level in ["none", "floor", "mid", "upper"]:
        bundle = apply_volume(base_bundle, level)
        res = F.fit_model("M0_aload", bundle, seed=42, n_starts=2, max_nfev=300)
        p = res.params
        ep = _endpoints("M0_aload", p, bundle["test"], c_fold)
        detail[level] = {"params": p, "cost": res.cost, "endpoints": ep}
        # held-out center metrics
        ib_rmse = float(np.sqrt(np.mean([(e["IBfrac_obs"] - e["IBfrac_pred"]) ** 2 for e in ep])))
        ptot_bias = float(np.mean([e["Ptot_pred"] - e["Ptot_obs"] for e in ep]))  # neg => underpredict
        summary.append({
            "level": level, "cost": res.cost, "Y_xs": p["Y_xs"], "a_qS": p["a_qS"],
            "theta_T": p["theta_T"], "qP_max": p["qP_max"], "K_S": p["K_S"],
            "IB_rmse": ib_rmse, "Ptot_bias": ptot_bias,
        })
        print(f"{level:6}: cost={res.cost:.4f}  Y_xs={p['Y_xs']:.3f}  a_qS={p['a_qS']:.2f}  "
              f"theta_T={p['theta_T']:.2f}  IB_rmse={ib_rmse:.3f}  Ptot_bias={ptot_bias:+.2f}")

    # Table
    L.append("| level | train cost | Y_xs | a_qS | theta_T | qP_max | K_S | held-out IB_rmse | held-out Ptot_bias |")
    L.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for s in summary:
        L.append(f"| {s['level']} | {s['cost']:.4f} | {s['Y_xs']:.3f} | {s['a_qS']:.2f} | "
                 f"{s['theta_T']:.2f} | {s['qP_max']:.3f} | {s['K_S']:.2f} | {s['IB_rmse']:.3f} | {s['Ptot_bias']:+.2f} |")
    L.append("")
    L.append("Ptot_bias = mean(pred - obs) over held-out F13/F14/F15; negative => underprediction.")
    L.append("")

    # Held-out center endpoints per level
    L.append("## Held-out center endpoints (F13/F14/F15)")
    L.append("| level | F13 Ptot obs/pred | F14 obs/pred | F15 obs/pred | F13 IB obs/pred | F14 IB | F15 IB |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for level in ["none", "floor", "mid", "upper"]:
        ep = detail[level]["endpoints"]
        def f(e, k): return f"{e[k+'_obs']:.2f}/{e[k+'_pred']:.2f}"
        L.append(f"| {level} | {f(ep[0],'Ptot')} | {f(ep[1],'Ptot')} | {f(ep[2],'Ptot')} | "
                 f"{f(ep[0],'IBfrac')} | {f(ep[1],'IBfrac')} | {f(ep[2],'IBfrac')} |")
    L.append("")
    (OUT / "bracket_refit.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'bracket_refit.md'}")


if __name__ == "__main__":
    main()
