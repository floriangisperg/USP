"""IB shape comparison on the T30 runs: observed vs adopted vs kinpart vs kinpart_deg.

The user's critique ("those are only straight lines") judged against THIS figure: does
any model reproduce the observed IB rise-bend-plateau/decline character on the mixed
regime? Full-data fit params (sources: adequacy rerun 2026-09-04 for the adopted model;
smoke tests 2026-09-05 for kinpart / kinpart_deg — costs 0.525 / 0.545 / 0.481).
Output: results/ib_shape_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt
import numpy as np
from fit import gather_predictions

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "ib_shape_comparison.png"

P_ADOPTED = {  # M0_aload_qPTexp, full-data fit 2026-09-04 (cost 0.525)
    "Y_xs": 0.363, "qP_max": 0.0493, "K_S": 0.541, "theta_0": -1.9935,
    "theta_T": 7.3103, "a_qS": 0.7729, "alpha_T": 0.0901,
}
P_KIN = {  # M0_kinpart, full-data fit 2026-09-05 (cost 0.545)
    "Y_xs": 0.3628, "qP_max": 0.056, "K_S": 0.4118, "k_f0": 0.5, "delta_f": 1.0,
    "k_a0": 0.0672, "delta_a": 0.1957, "k_p0": 0.1456, "gamma_p": 0.121,
}
P_KINDEG = {  # M0_kinpart_deg, full-data fit 2026-09-05 (cost 0.481)
    "Y_xs": 0.3629, "qP_max": 0.0569, "K_S": 0.4812, "k_f0": 0.5, "delta_f": 1.0,
    "k_a0": 0.0789, "delta_a": 0.2178, "k_p0": 0.0847, "gamma_p": 0.1227, "k_deg0": 0.0411,
}
P_KINDEG_FREE = {  # M0_kinpart_deg_free (k_deg prior-free), full-data 2026-09-05 (cost 0.4236)
    "Y_xs": 0.3631, "qP_max": 0.0572, "K_S": 0.5645, "k_f0": 0.5, "delta_f": 1.0,
    "k_a0": 0.102, "delta_a": 0.2366, "k_p0": 0.0, "gamma_p": 0.12, "k_deg0": 0.1058,
}

RUNS = ["F13", "F14", "F15", "F18"]
RUNS += ["F17"]

train, _ = D.split_for_fold("full")
bundle = D.load_dataset(train, [])
runs = {r["run_id"]: r for r in bundle["train"]}

fig, axes = plt.subplots(1, len(RUNS), figsize=(4.0 * len(RUNS), 3.8), sharey=True)
for ax, rid in zip(np.atleast_1d(axes), RUNS):
    run = runs[rid]
    t = run["t_obs"]
    ax.plot(t, run["obs"]["P_IB"], "o", color="black", ms=7, label="observed", zorder=5)
    for params, model, label, color, ls in (
        (P_ADOPTED, "M0_aload_qPTexp", "adopted (sigmoid)", "#9e9e9e", "--"),
        (P_KIN, "M0_kinpart", "kinpart", "#4E79A7", "-"),
        (P_KINDEG, "M0_kinpart_deg", "kinpart+deg", "#E58237", "-"),
        (P_KINDEG_FREE, "M0_kinpart_deg_free", "kinpart+deg FREE k_deg", "#59A14F", ":"),
    ):
        pred = gather_predictions(model, params, [run], bundle["c_fold"])[0]
        ax.plot(t, pred["pred"]["obs"]["P_IB"], color=color, ls=ls, lw=2, label=label)
    ax.set_title(f"{rid} (T30)", fontsize=11)
    ax.set_xlabel("Process time [h]")
np.atleast_1d(axes)[0].set_ylabel("P_IB [g/L]")
np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
fig.suptitle("T30 IB trajectories — observed vs adopted (grey, straight) vs kinetic partition (blue) vs +aggregate proteolysis (orange)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
