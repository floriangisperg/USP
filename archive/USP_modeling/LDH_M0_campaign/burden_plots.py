"""Documentation plots for the Task A burden-state verdict (no fitting; final values).

Left  : a_b profile likelihood on fold B (monotone, no interior min -> not identifiable).
Right : held-out center IB fractions (M0_aload_burden is worse than M0_aload -> overfit).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "M0_aload_burden"

# a_b profile on fold B (from identifiability_summary.md).
ab = [-4, -2, 0, 2, 4]
cost = [0.3732, 0.3761, 0.4397, 0.4519, 0.4636]
aqs = [4.0, 3.38, 1.68, 1.58, 1.54]

# Held-out center IB fractions (fold B).
runs = ["F13", "F14", "F15"]
obs = [0.123, 0.147, 0.116]
m0 = [0.123, 0.131, 0.126]
burden = [0.098, 0.112, 0.100]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(ab, cost, "-o", color="#0072B2", ms=7, lw=1.8, label="profile (a_qS refit free)")
ax.axvline(0, color="#888", ls=":", lw=1)
ax.annotate("a_b=0  ->  a_qS=1.68\n(= M0_aload exactly)", (0, 0.4397),
            xytext=(-3.6, 0.455), fontsize=9, color="#444")
ax.annotate("min at a_b=-4\n(unphysical: burden\n*decreasing* IB)", (-4, 0.3732),
            xytext=(-3.4, 0.374), fontsize=9, color="#B22222")
ax.set_xlabel("a_b (burden coefficient, fixed)")
ax.set_ylabel("train cost")
ax.set_title("a_b profile likelihood (fold B)\nmonotone, no interior min -> NOT identifiable")
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
import numpy as np  # noqa: E402

x = np.arange(len(runs)); w = 0.27
ax.bar(x - w, obs, w, color="#444", label="observed")
ax.bar(x, m0, w, color="#0072B2", label="M0_aload")
ax.bar(x + w, burden, w, color="#D55E00", label="M0_aload_burden")
ax.set_xticks(x); ax.set_xticklabels(runs)
ax.set_ylabel("held-out final IB fraction")
ax.set_title("Held-out center (fold B)\nburden model is WORSE -> overfit via bounds")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "burden_verdict.png", dpi=130)
plt.close(fig)
print("saved", OUT / "burden_verdict.png")
