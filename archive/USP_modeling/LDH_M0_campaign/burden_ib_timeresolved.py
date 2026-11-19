"""Time-resolved burden(qCO2) <-> IB-fraction correlation.

Stronger test than the final-endpoint correlation: pair the post-induction burden
respiration with the IB fraction at EVERY post-induction sample point, across the
30 C stratum (and all runs). If burden tracks IB as it evolves within runs (not just
at the endpoint), the link is dynamic, not a static coincidence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import offgas_diagnostics as od  # noqa: E402
import pandas as pd  # noqa: E402

OUT = od.SCRIPT_DIR / "results" / "offgas_diagnostics"
df = pd.read_pickle(od.DATA_FILE)

RUNS_30 = ["F17", "F13", "F14", "F15", "F18"]
RUNS_ALL = ["F07", "F09", "F10", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]


def collect(runs):
    rows = []
    for run in runs:
        d = od.process_run(df, run)
        t = d["t"]; cer = d["cer"]; X = d["X_meas"]; ind = d["ind"]; pre = d["pre"]
        # maintenance baseline (a*mu + b) from pre-induction qCO2 vs mu
        mu = np.gradient(np.log(np.maximum(X, 1e-3)), t)
        qco2 = cer / np.maximum(X, 1e-3)
        mp = pre & np.isfinite(mu) & np.isfinite(qco2) & (mu > -0.5)
        a = b = np.nan
        if mp.sum() > 20:
            A = np.vstack([mu[mp], np.ones(mp.sum())]).T
            sol, *_ = np.linalg.lstsq(A, qco2[mp], rcond=None)
            a, b = sol
        qco2_pre = np.nanmedian(qco2[pre & np.isfinite(qco2)])
        # sample-point offline P_sol/P_IB
        r = df[df["run_name"] == run].sort_values(od._c(df, "Process time"))
        ts = r[od._c(df, "Process time")].to_numpy(float)
        Ps = np.nan_to_num(r[od._c(df, "Protein soluble")].to_numpy(float), nan=0.0)
        Pb = np.nan_to_num(r[od._c(df, "Protein IB")].to_numpy(float), nan=0.0)
        is_s = r[od._c(df, "is_sample")].fillna(False).to_numpy(bool)
        t0ind = t[ind][0] if ind.any() else np.nan
        for ti, pso, pbo, sm in zip(ts, Ps, Pb, is_s):
            if not sm:
                continue
            ptot = pso + pbo
            if not np.isfinite(ptot) or ptot < 0.2:  # need measurable product
                continue
            ibf = pbo / ptot
            # online qCO2 / burden interpolated to this sample time
            q = float(np.interp(ti, t, qco2))
            burden_lin = q - (a * float(np.interp(ti, t, mu)) + b) if np.isfinite(a) else np.nan
            burden_pre = q - qco2_pre if np.isfinite(qco2_pre) else np.nan
            rows.append((run, ti - t0ind, ibf, q, burden_lin, burden_pre))
    return pd.DataFrame(rows, columns=["run", "t_ind", "ibfrac", "qco2", "burden_lin", "burden_pre"])


def report(tag, sub):
    print(f"\n=== {tag}  (n={len(sub)}) ===")
    for xcol, xlab in [("qco2", "raw qCO2"), ("burden_lin", "burden (a*mu+b baseline)"),
                       ("burden_pre", "burden (pre-median baseline)")]:
        fin = sub[[xcol, "ibfrac"]].dropna()
        if len(fin) >= 3:
            r = float(np.corrcoef(fin[xcol], fin["ibfrac"])[0, 1])
            print(f"  Pearson r( {xlab:28s} , IB frac ) = {r:.2f}")
        else:
            print(f"  {xlab}: too few points")


sub30 = collect(RUNS_30)
report("30 C stratum, time-resolved (all post-induction sample points)", sub30)
suball = collect(RUNS_ALL)
report("ALL runs, time-resolved", suball)

# Plot: burden vs IB fraction, colored by run (30 C).
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, xcol, xlab in zip(axes, ("burden_pre", "qco2"),
                          ("burden qCO2 (pre-median baseline) [mmol/g/h]", "raw qCO2 [mmol/g/h]")):
    for run, g in sub30.groupby("run"):
        ax.scatter(g[xcol], g["ibfrac"], label=run, s=40)
    fin = sub30[[xcol, "ibfrac"]].dropna()
    if len(fin) >= 3:
        r = float(np.corrcoef(fin[xcol], fin["ibfrac"])[0, 1])
        ax.set_title(f"30 C, time-resolved: {xlab}\nvs IB fraction   (r={r:.2f}, n={len(fin)})")
    ax.set_xlabel(xlab); ax.set_ylabel("IB fraction at sample time"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "burden_vs_IB_timeresolved.png", dpi=120); plt.close(fig)
print(f"\nsaved {OUT/'burden_vs_IB_timeresolved.png'}")
sub30.to_csv(OUT / "burden_ib_timeresolved_30C.csv", index=False)
print(f"saved {OUT/'burden_ib_timeresolved_30C.csv'}")
