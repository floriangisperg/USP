"""Fresh look at P_sol vs P_IB dynamics (per the user's reframe): the two states have DIFFERENT
dynamics -- sol a steady accumulator, IB a transient that peaks then declines. Characterize each
separately from the data, and ask what drives the difference (qS, mu, X, cumulative load, time).

No model fitting -- pure data characterization to inform the model STRUCTURE.
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt
from models import build_inputs

S_IN = 400.0
b = D.load_dataset(*D.split_for_fold("full"))
runs = {r["run_id"]: r for r in b["train"]}

ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]
TCOL = {"25": "#0072B2", "30": "#009E73", "35": "#D55E00"}
# FCCD induction setpoints (median online temp reads ~37 = batch phase, not the setpoint).
TSET = {"F09": 25, "F16": 25, "F10": 25, "F17": 30, "F13": 30, "F14": 30, "F15": 30,
        "F18": 30, "F07": 35, "F19": 35, "F12": 35}


def tset(r):
    return float(TSET.get(r["run_id"], np.nan))


def apparent(r):
    """Post-induction intervals -> (t_ind, Psol, Pib, X, qS, mu, qPsol, qPib)."""
    t = np.asarray(r["t_obs"], float)
    Psol = np.asarray(r["obs"]["P_sol"], float)
    Pib = np.asarray(r["obs"]["P_IB"], float)
    X = np.asarray(r["obs"]["X"], float)
    t_on = np.asarray(r["t_online"], float)
    feed = np.asarray(r["feed_rate_L_h"], float)
    V = np.asarray(r["V_L"], float)
    ind = np.asarray(r["is_induced"], float) > 0.5
    t_ind0 = float(t_on[ind][0]) if ind.any() else float(t_on[0])
    out = []
    for k in range(len(t) - 1):
        t0, t1 = t[k], t[k + 1]
        if t0 < t_ind0 or not np.all(np.isfinite([Psol[k], Psol[k + 1], Pib[k], Pib[k + 1], X[k], X[k + 1]])):
            continue
        dt = t1 - t0
        if dt <= 0:
            continue
        Xb = 0.5 * (X[k] + X[k + 1])
        if Xb < 1e-3:
            continue
        tm = 0.5 * (t0 + t1)
        Dbar = float(np.interp(tm, t_on, feed / np.where(V > 1e-6, V, np.nan)))
        qPsol = ((Psol[k + 1] - Psol[k]) / dt + Dbar * 0.5 * (Psol[k] + Psol[k + 1])) / Xb
        qPib = ((Pib[k + 1] - Pib[k]) / dt + Dbar * 0.5 * (Pib[k] + Pib[k + 1])) / Xb
        Fm = float(np.interp(tm, t_on, feed))
        Vm = float(np.interp(tm, t_on, V))
        qS = Fm * S_IN / (Xb * Vm) if Vm > 1e-6 else np.nan
        out.append((t0 - t_ind0, Psol[k + 1], Pib[k + 1], X[k + 1], qS, qPsol, qPib))
    return out, t_ind0


print(f"{'run':>4} {'T':>3}   P_sol trajectory (t_ind_h : g/L)        | P_IB trajectory (t_ind_h : g/L)         "
      f"| IB peak")
for rid in ORDER:
    r = runs[rid]
    T = tset(r)
    rows, _ = apparent(r)
    if not rows:
        continue
    sols = " ".join(f"{ti:4.1f}:{ps:4.2f}" for ti, ps, pi, x, qs, qps, qpi in rows)
    ibs = " ".join(f"{ti:4.1f}:{pi:4.2f}" for ti, ps, pi, x, qs, qps, qpi in rows)
    pis = [pi for _, _, pi, _, _, _, _ in rows]
    pk = int(np.argmax(pis))
    pknote = f"t={rows[pk][0]:.1f}h v={pis[pk]:.2f}" if len(rows) > 1 else "n/a"
    print(f"{rid:>4} {T:3.0f}   {sols:38.38s} | {ibs:38.38s} | {pknote}")

# Per-run apparent specific rates (the dynamics signature).
print("\nApparent specific rates [g/g/h] by interval (t_ind_h : qPsol / qPib ; qS):")
for rid in ["F13", "F18", "F07", "F12"]:  # representative: T30 center, T30 high-qS, T35 corners
    r = runs[rid]
    rows, _ = apparent(r)
    print(f"  {rid} (T={tset(r):.0f}):")
    for ti, ps, pi, x, qs, qps, qpi in rows:
        print(f"    t_ind={ti:4.1f}h  X={x:4.1f} qS={qs:.3f}  qPsol={qps:+.4f}  qPib={qpi:+.4f}")

# Plot: P_sol and P_IB vs t_ind, all runs, colored by T.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for rid in ORDER:
    r = runs[rid]
    T = tset(r)
    rows, _ = apparent(r)
    if not rows:
        continue
    ti = [w[0] for w in rows]
    c = TCOL[f"{round(T)}"]
    axes[0].plot(ti, [w[1] for w in rows], "-o", c=c, mfc="w", ms=4, lw=1.2, label=rid)
    axes[1].plot(ti, [w[2] for w in rows], "-o", c=c, mfc="w", ms=4, lw=1.2, label=rid)
axes[0].set_title("P_sol [g/L] vs t since induction")
axes[1].set_title("P_IB [g/L] vs t since induction")
for ax in axes:
    ax.set_xlabel("t since induction [h]")
    ax.axhline(0, c="k", lw=0.5)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)
from matplotlib.patches import Patch

handles = [Patch(color=TCOL["25"], label="25 C"), Patch(color=TCOL["30"], label="30 C"),
           Patch(color=TCOL["35"], label="35 C")]
axes[1].legend(handles=handles, fontsize=8, title="colour = T")
fig.suptitle("P_sol vs P_IB: different dynamics? (sol steady accumulator, IB transient)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/explore_psol_pib.png", dpi=120)
plt.close(fig)
print("\nsaved results/explore_psol_pib.png")
