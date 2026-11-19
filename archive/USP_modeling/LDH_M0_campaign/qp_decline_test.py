"""Diagnostic: is the post-induction production decline qS-driven (model right) or an EXTRA decline
(real burden)? And is it consistent across LDH and Xiso (=> general mechanism)?

For each run, compute over induction intervals:
  qP_obs(t) = (dP_tot/dt + D*P_tot)/X           (apparent specific total production rate)
  qS(t)     = F*S_IN/(X*V)                        (feed-supply proxy, declines under open-loop feed)
Compare decline factors (late/early): f_qP vs f_qS (raw) and f_Monod = Monod(qS_late)/Monod(qS_early)
(model prediction, Monod declines LESS than qS). If f_qP < f_Monod (or < f_qS), observed qP declines
FASTER than qS explains => an EXTRA decline = real production burden, not a qS artifact.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D

S_IN = 400.0
K_S = 0.56  # LDH fit value (for Monod); sensitivity noted


def monod(qs, K=K_S):
    return qs / (qs + K)


def intervals(t_obs, X, Psol, Pib, t_dense, feed, V, t_ind):
    Ptot = np.asarray(Psol, float) + np.asarray(Pib, float)
    X = np.asarray(X, float)
    t_obs = np.asarray(t_obs, float)
    rows = []
    for k in range(len(t_obs) - 1):
        t0, t1 = t_obs[k], t_obs[k + 1]
        if t0 < t_ind or not (np.isfinite(Ptot[k]) and np.isfinite(Ptot[k + 1])
                              and np.isfinite(X[k]) and np.isfinite(X[k + 1])):
            continue
        dt = t1 - t0
        if dt <= 0:
            continue
        Xbar = 0.5 * (X[k] + X[k + 1])
        if Xbar < 1e-3:
            continue
        Pbar = 0.5 * (Ptot[k] + Ptot[k + 1])
        tm = 0.5 * (t0 + t1)
        Dbar = float(np.interp(tm, t_dense, feed / np.where(V > 1e-6, V, np.nan)))
        qP_obs = ((Ptot[k + 1] - Ptot[k]) / dt + Dbar * Pbar) / Xbar
        Fm = float(np.interp(tm, t_dense, feed))
        Vm = float(np.interp(tm, t_dense, V))
        qS = Fm * S_IN / (Xbar * Vm) if Vm > 1e-6 else np.nan
        rows.append((tm, qP_obs, qS))
    return rows


def decline(rows):
    if len(rows) < 2:
        return None
    qP = np.array([r[1] for r in rows]); qS = np.array([r[2] for r in rows])
    qP = qP[np.isfinite(qP) & (qP > 0)]; qS = qS[np.isfinite(qS) & (qS > 0)]
    if len(qP) < 2 or len(qS) < 2:
        return None
    f_qP = qP[-1] / qP[0]
    f_qS = qS[-1] / qS[0]
    f_monod = monod(qS[-1]) / monod(qS[0])
    return dict(f_qP=f_qP, f_qS=f_qS, f_monod=f_monod, rows=rows)


def collect_lhd():
    b = D.load_dataset(*D.split_for_fold("full"))
    out = {}
    for r in b["train"]:
        ind = np.asarray(r["is_induced"], float) > 0.5
        t_ind = float(np.asarray(r["t_online"], float)[ind][0]) if ind.any() else float(r["t_online"][0])
        rows = intervals(r["t_obs"], r["obs"]["X"], r["obs"]["P_sol"], r["obs"]["P_IB"],
                         np.asarray(r["t_online"], float), np.asarray(r["feed_rate_L_h"], float),
                         np.asarray(r["V_L"], float), t_ind)
        d = decline(rows)
        if d:
            out[r["run_id"]] = d
    return out


def collect_xiso(path):
    df = pd.read_excel(path, sheet_name="All_Time_Points")
    out = {}
    for rid, g in df.groupby("run_name"):
        g = g.sort_values("Process time [h]").reset_index(drop=True)
        t = g["Process time [h]"].to_numpy(float)
        sm = g["is_sample"].fillna(False).astype(bool) if "is_sample" in g else np.isfinite(g["Protein soluble [g/L]"])
        feed_mL = g["Feed [mL]"].fillna(0.0).to_numpy(float)
        fr = np.zeros_like(feed_mL)
        dt = np.diff(t); fr[:-1] = np.where(dt > 0, np.diff(feed_mL) / dt, 0.0)
        fr = np.maximum(fr, 0.0) / 1000.0  # L/h
        V = g["Volume corrected [mL]"].ffill().bfill().to_numpy(float) / 1000.0
        ind = g["Is induced [-]"].fillna(0).to_numpy(float) > 0.5
        t_ind = float(t[ind][0]) if ind.any() else float(t[0])
        rows = intervals(t[sm], g["Biomass [g/L]"].to_numpy(float)[sm],
                         g["Protein soluble [g/L]"].to_numpy(float)[sm],
                         g["Protein IB [g/L]"].to_numpy(float)[sm],
                         t, fr, V, t_ind)
        d = decline(rows)
        if d:
            out[str(rid)] = d
    return out


lhd = collect_lhd()
xiso = collect_xiso("../../Xiso/Xiso_consolidated_results.xlsx")

print(f"{'dataset':>8} {'n':>3} {'f_qP':>6} {'f_qS':>6} {'f_Monod':>7}  (late/early; <1 = declined)")
for name, d in [("LDH", lhd), ("Xiso", xiso)]:
    fq = np.array([v["f_qP"] for v in d.values()])
    fs = np.array([v["f_qS"] for v in d.values()])
    fm = np.array([v["f_monod"] for v in d.values()])
    print(f"{name:>8} {len(d):>3} {np.nanmedian(fq):>6.2f} {np.nanmedian(fs):>6.2f} {np.nanmedian(fm):>7.2f}")
    for rid, v in d.items():
        extra = "EXTRA decline" if v["f_qP"] < v["f_monod"] - 0.05 else "qS-consistent"
        print(f"   {rid}: f_qP={v['f_qP']:.2f} f_qS={v['f_qS']:.2f} f_Monod={v['f_monod']:.2f}  {extra}")

# Phase plot qP_obs vs qS, early vs late, LDH vs Xiso.
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, d, mk, c in [("LDH", lhd, "o", "#0072B2"), ("Xiso", xiso, "^", "#D55E00")]:
    for rid, v in d.items():
        rows = v["rows"]
        n = len(rows)
        for i, (tm, qP, qS) in enumerate(rows):
            late = i >= n // 2
            ax.scatter(qS, qP, marker=mk, c=c, s=60, edgecolor="k", lw=0.3,
                       alpha=0.4 if late else 1.0)
ax.set_xlabel("qS(t)  [g/g/h]  (feed-supply proxy, declines over induction)")
ax.set_ylabel("qP_obs(t)  [g/g/h]  (apparent specific production)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", markeredgecolor="k", label="LDH", linestyle=""),
                   Line2D([0], [0], marker="^", color="w", markerfacecolor="#D55E00", markeredgecolor="k", label="Xiso", linestyle=""),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markeredgecolor="k", label="filled=early, faint=late", linestyle="")],
          fontsize=8)
ax.set_title("Production vs substrate over induction\nlate points BELOW early trend => decline beyond qS (real burden)")
fig.tight_layout()
fig.savefig("results/qp_decline_test.png", dpi=120)
plt.close(fig)
print("\nsaved results/qp_decline_test.png")
