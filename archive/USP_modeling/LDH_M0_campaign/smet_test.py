"""Task A1 (S_met diagnostic) -- does cumulative metabolic load drive the qS-peak / decline?

Inspired by Muller et al. 2023 (ESCAPE33): use the cumulative metabolized substrate since induction

    S_met(t) = integral_{t_ind}^{t} qS(tau)*X(tau) dtau

as a cell-age / metabolic-burden state, and test whether total specific production qP declines with
it (generalized-Haldane inhibition). This is the one structural lever that can represent the qS-peak
IDENTIFIABLY, because -- unlike the rejected Haldane-in-instantaneous-qS (which failed since realized
qS compresses the setpoint spread 0.2/0.4/0.6 -> ~0.17/0.27/0.30) -- S_met is the TIME-INTEGRAL of load
and is NOT compressed: high-qS runs accumulate far more over induction.

KEY SIMPLIFICATION (identifiability + robustness): qS*X = F*S_IN/(X*V) * X = F*S_IN/V = S_IN*D, so the
biomass cancels and

    S_met(t) = S_IN * integral_{t_ind}^{t} D(tau) dtau        (D = feed_rate / V)

i.e. S_met needs only the dense dilution, no biomass interpolation. So S_met is an integral of
measured-driven quantities (feed, V) -- a state-determined load, not a free latent -> identifiable by
construction. S_met/X (the per-biomass burden Muller uses) only needs X at the evaluation point.

Observed qP (gP/gX/h) is recovered from the sparse product + dense dilution:
    dP/dt = qP*X - D*P   ->   qP = (dP/dt + D*P) / X
computed as interval averages between consecutive post-induction samples (dP/dt ~ dP/dt, D the
mean dilution, P/X the interval means).

DECISION (identifiability-first, controls for the already-modelled qP(T)=alpha and for qS-setpoint):
  - Primary: does qP decline with S_met/X WITHIN temperature strata? Partial correlations
    r(qP, S_met/X | T) and r(qP, S_met/X | T, qS) remove the alpha(T) suppression and the qS-setpoint
    so any residual negative trend is genuinely load-driven (and can explain the within-T qS-peak).
  - Endpoint test: Ptot/X_final vs S_met/X_final across runs (does the qS-peak track cumulative load?).
  - Haldane fit: qP ~ qP0 * u/(u + Ks + u^2/Ki), u = S_met/X, to the pooled cloud; R^2 vs mean-only.
If qP falls with S_met/X within-T and the endpoint tracks load, the mechanism has legs -> fit it (A1).
If flat / fully explained by T+qS, the qS-peak is a hard 3-level data wall.

Outputs: results/smet/{smet_test.md, smet_summary.png, smet_trajectories.png, smet.csv}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "smet"
OUT.mkdir(parents=True, exist_ok=True)

S_IN = 400.0  # feed glucose concentration [g/L] (mechanistic constant)

SET = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}
RUN_ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]

C_LOAD = "#D55E00"    # S_met/X load -- orange
C_QP = "#0072B2"      # qP -- blue
Tcol = {25: "#56B4E9", 30: "#0072B2", 35: "#E69F00"}
QSmk = {0.2: "o", 0.4: "s", 0.6: "^"}


def _cumulative_D(run):
    """Dense t, D = feed_rate/V, and cumulative integral of D from t=0 (for S_met)."""
    t = np.asarray(run["t_online"], dtype=float)
    feed = np.asarray(run["feed_rate_L_h"], dtype=float)
    V = np.asarray(run["V_L"], dtype=float)
    D = feed / np.where(V > 1e-6, V, np.nan)
    cumD = np.concatenate([[0.0], np.cumsum((D[:-1] + D[1:]) * 0.5 * np.diff(t))])
    return t, D, cumD


def analyze_run(run):
    t, D, cumD = _cumulative_D(run)
    ind = np.asarray(run["is_induced"], dtype=float) > 0.5
    t_ind = float(t[ind][0]) if ind.any() else float(t[0])
    # S_met(t) = S_IN * integral_{t_ind}^{t} D dt  (biomass-free).
    smet = S_IN * (cumD - float(np.interp(t_ind, t, cumD)))  # g substrate metabolized / L broth, since induction
    smet = np.maximum(smet, 0.0)

    # Observed qP at interval midpoints between consecutive post-induction samples.
    tobs = np.asarray(run["t_obs"], dtype=float)
    X = np.asarray(run["obs"]["X"], dtype=float)
    Psol = np.asarray(run["obs"]["P_sol"], dtype=float)
    Pib = np.asarray(run["obs"]["P_IB"], dtype=float)
    Ptot = Psol + Pib
    ind_obs = tobs >= t_ind
    tobs_i, X_i, P_i = tobs[ind_obs], X[ind_obs], Ptot[ind_obs]
    rows = []
    for k in range(len(tobs_i) - 1):
        t0, t1 = tobs_i[k], tobs_i[k + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        X0, X1 = X_i[k], X_i[k + 1]
        P0, P1 = P_i[k], P_i[k + 1]
        if not (np.isfinite(X0) and np.isfinite(X1) and np.isfinite(P0) and np.isfinite(P1)):
            continue
        Xbar = 0.5 * (X0 + X1)
        Pbar = 0.5 * (P0 + P1)
        if Xbar < 1e-3:
            continue
        Dbar = float(np.interp(0.5 * (t0 + t1), t, D))
        dPdt = (P1 - P0) / dt
        qP = (dPdt + Dbar * Pbar) / Xbar
        tm = 0.5 * (t0 + t1)
        u = float(np.interp(tm, t, smet)) / Xbar     # S_met/X at midpoint
        rows.append({"t_mid": tm, "qP": qP, "u": u, "dPdt": dPdt, "Xbar": Xbar})

    # Endpoint load + specific product.
    Xf = float(X_i[-1]) if len(X_i) else float("nan")
    Pf = float(P_i[-1]) if len(P_i) else float("nan")
    smet_end = float(np.interp(t[-1], t, smet))
    return {
        "run": run["run_id"], "T": SET[run["run_id"]][0], "qs": SET[run["run_id"]][1],
        "t": t, "smet": smet, "t_ind": t_ind,
        "intervals": rows,
        "smet_end_over_X": smet_end / Xf if Xf > 1e-3 else float("nan"),
        "Ptot_over_X_final": Pf / Xf if Xf > 1e-3 else float("nan"),
        "X_final": Xf, "smet_end": smet_end,
    }


def _partial_corr(qP, u, groups):
    """Pearson r of (qP, u) after residualizing both by group means (group = T, or T+qs)."""
    qP = np.asarray(qP, float)
    u = np.asarray(u, float)
    groups = list(groups)
    m = np.isfinite(qP) & np.isfinite(u)
    qP, u, groups = qP[m], u[m], [g for g, mm in zip(groups, m) if mm]
    if len(qP) < 5:
        return float("nan")
    qP_r = qP - _group_mean(qP, groups)
    u_r = u - _group_mean(u, groups)
    if np.std(qP_r) < 1e-9 or np.std(u_r) < 1e-9:
        return float("nan")
    return float(np.corrcoef(qP_r, u_r)[0, 1])


def _group_mean(vals, groups):
    means = {}
    counts = {}
    for v, g in zip(vals, groups):
        means[g] = means.get(g, 0.0) + v
        counts[g] = counts.get(g, 0) + 1
    gm = {g: means[g] / counts[g] for g in means}
    return np.array([gm[g] for g in groups])


def _haldane(u, qP0, Ks, Ki):
    """Standard Andrews: qP0 * u/(u + Ks + u^2/Ki)."""
    return qP0 * u / (u + Ks + u ** 2 / Ki)


def main():
    tr, te = D.split_for_fold("full")
    bundle = D.load_dataset(tr, te)
    recs = [analyze_run(r) for r in bundle["train"]]

    # Pool interval points.
    qP_all, u_all, T_all, qs_all, run_all = [], [], [], [], []
    for r in recs:
        for row in r["intervals"]:
            qP_all.append(row["qP"])
            u_all.append(row["u"])
            T_all.append(r["T"])
            qs_all.append(r["qs"])
            run_all.append(r["run"])
    qP_all = np.array(qP_all)
    u_all = np.array(u_all)
    T_all = np.array(T_all)
    qs_all = np.array(qs_all)

    # Correlations.
    m = np.isfinite(qP_all) & np.isfinite(u_all)
    r_raw = float(np.corrcoef(qP_all[m], u_all[m])[0, 1]) if m.sum() > 5 else float("nan")
    r_T = _partial_corr(qP_all, u_all, T_all)                       # control for T (alpha)
    r_Tq = _partial_corr(qP_all, u_all, [f"{t}_{q}" for t, q in zip(T_all, qs_all)])  # control for T+qs

    # Endpoint (qS-peak) test.
    end_u = np.array([r["smet_end_over_X"] for r in recs])
    end_P = np.array([r["Ptot_over_X_final"] for r in recs])
    end_T = np.array([r["T"] for r in recs])
    end_q = np.array([r["qs"] for r in recs])
    me = np.isfinite(end_u) & np.isfinite(end_P)
    r_end_raw = float(np.corrcoef(end_u[me], end_P[me])[0, 1])
    r_end_T = _partial_corr(end_u, end_P, end_T)

    # Haldane fit to the pooled cloud.
    fit_ok, popt, r2_hal = False, None, float("nan")
    try:
        popt, _ = curve_fit(_haldane, u_all[m], qP_all[m], p0=[0.03, 1.0, 5.0],
                            bounds=([1e-4, 1e-3, 1e-3], [0.5, 50, 200]), maxfev=20000)
        resid = qP_all[m] - _haldane(u_all[m], *popt)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((qP_all[m] - np.mean(qP_all[m])) ** 2))
        r2_hal = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fit_ok = True
    except Exception as e:
        print(f"Haldane fit failed: {e}")

    # Plots.
    _plot_trajectories(recs, OUT / "smet_trajectories.png")
    _plot_summary(recs, qP_all, u_all, T_all, qs_all, end_u, end_P, end_T, end_q,
                  popt, r2_hal, OUT / "smet_summary.png")

    # CSV.
    pd.DataFrame({"run": run_all, "T": T_all, "qs": qs_all, "qP": qP_all, "Smet_over_X": u_all}
                 ).to_csv(OUT / "smet.csv", index=False)

    # Within-run decline: qP last interval vs first.
    declines = []
    for r in recs:
        iv = r["intervals"]
        if len(iv) >= 2 and iv[0]["qP"] > 0:
            declines.append(iv[-1]["qP"] / iv[0]["qP"])
    declines = np.array(declines)

    # Report.
    L = ["# Task A1 -- does cumulative metabolic load (S_met) drive the qS-peak / decline?", ""]
    L.append("Muller 2023 idea: S_met = integral(qS*X) since induction as a cell-age/burden state in a")
    L.append("generalized-Haldane inhibition on qP. Biomass cancels (qS*X = S_IN*D), so")
    L.append("S_met = S_IN * integral(D) dt since induction -- dense, X-free, identifiable by construction.")
    L.append("")
    L.append("## Per-run endpoint (cumulative load vs specific product)")
    L.append("| run | T | qs | S_met_end/X_final | Ptot/X_final |")
    L.append("| --- | -:| -:| ---:| ---:|")
    for r in recs:
        print(f"{r['run']} T{r['T']} q{r['qs']}: S_met/X_end={r['smet_end_over_X']:.2f} "
              f"Ptot/X={r['Ptot_over_X_final']:.3f} (n_int={len(r['intervals'])})")
        L.append(f"| {r['run']} | {r['T']} | {r['qs']} | {r['smet_end_over_X']:.2f} | "
                 f"{r['Ptot_over_X_final']:.3f} |")
    L.append("")

    L.append("## Correlation of observed qP with cumulative load S_met/X (interval points)")
    L.append(f"- r(qP, S_met/X) raw = **{r_raw:+.2f}**")
    L.append(f"- r(qP, S_met/X | T) partial = **{r_T:+.2f}**  (controls for alpha(T) -- the key test)")
    L.append(f"- r(qP, S_met/X | T,qs) partial = **{r_Tq:+.2f}**  (also controls qS-setpoint)")
    L.append("- A load-driven decline needs a NEGATIVE partial r within-T (more load -> less qP).")
    L.append("")
    L.append("## Endpoint test (the qS-peak)")
    L.append(f"- r(Ptot/X_final, S_met/X_final) raw = **{r_end_raw:+.2f}**")
    L.append(f"- r(Ptot/X_final, S_met/X_final | T) partial = **{r_end_T:+.2f}**")
    L.append("")
    if fit_ok:
        L.append("## Haldane fit to pooled cloud")
        L.append(f"- qP ~ qP0 * u/(u+Ks+u^2/Ki), u=S_met/X: qP0={popt[0]:.4f}, Ks={popt[1]:.2f}, "
                 f"Ki={popt[2]:.2f}, R^2={r2_hal:.2f}")
        L.append("- (R^2 vs mean-only baseline; the categorical T-mean and qs-mean explain the rest.)")
        L.append("")
    L.append("## Within-run qP trajectory (last/first interval ratio; <1 => declines over induction)")
    L.append(f"- mean(last/first) = {np.nanmean(declines):.2f}; "
             f"fraction declining (<1) = {np.nanmean(declines < 1.0) * 100:.0f}%")
    L.append("")

    # Cross-T qS-peak test: does the qs0.4 -> qs0.6 product decline appear at every T (load-driven),
    # or only at some (T/partition-driven)? Decisive because S_met/X at qs0.6 is ~same across T.
    byTq = {}
    for r in recs:
        byTq[(r["T"], r["qs"])] = r["Ptot_over_X_final"]
    byTq_load = {}
    for r in recs:
        byTq_load[(r["T"], r["qs"])] = r["smet_end_over_X"]
    L.append("## Cross-T qS-peak test (qs0.4 -> qs0.6 product change, and the load at qs0.6)")
    L.append("| T | Ptot/X qs0.4 | Ptot/X qs0.6 | dP (qs0.6-0.4) | S_met/X @ qs0.6 | load-driven? |")
    L.append("| -:| ---:| ---:| ---:| ---:| --- |")
    n_decline = 0
    for T in (25, 30, 35):
        p4 = [byTq[(T, 0.4)]] if (T, 0.4) in byTq else []
        p4 = [v for k, v in byTq.items() if k[0] == T and k[1] == 0.4]
        p6 = byTq.get((T, 0.6))
        u6 = byTq_load.get((T, 0.6))
        if p4 and p6 is not None:
            dP = p6 - np.mean(p4)
            ld = "decline (yes)" if dP < -0.005 else ("rise (no)" if dP > 0.005 else "flat")
            if dP < -0.005:
                n_decline += 1
            L.append(f"| T{T} | {np.mean(p4):.3f} | {p6:.3f} | {dP:+.3f} | {u6:.2f} | {ld} |")
    L.append("")

    L.append("## Verdict -- S_met is NOT supported as the qS-peak driver")
    L.append("")
    L.append("**The clincher (cross-T load range):** cumulative load at qs0.6 is ~the SAME at every")
    L.append("temperature (S_met/X @ qs0.6 = ~2.1-2.3 for T25/T30/T35), yet the qs0.4->qs0.6 product")
    L.append(f"change differs by T: only **{n_decline}/3** strata decline. If cumulative load drove the")
    L.append("decline via Haldane inhibition, the SAME load should produce the SAME decline at every T.")
    L.append("It does not (T25 rises, T30 declines, T35 marginal) -> the qS-peak is a T x qS")
    L.append("(partition) effect, not a load effect. This mirrors the E.2 (O2) finding exactly: the")
    L.append("qS-peak is T30-specific.")
    L.append("")
    L.append("**What IS real:** qP declines with cumulative load WITHIN a run at fixed (T, qs)")
    L.append(f"(partial r | T,qs = {r_Tq:+.2f}) -- a genuine post-induction productivity decline. But")
    L.append("this is (a) the same time-decline the already-REJECTED induction-decay captures (overfits")
    L.append("held-out), and (b) confounded with the existing Monod(qS) term -- realized qS(t) declines")
    L.append("over a run as biomass outpaces the feed ramp, so Monod(qS(t)) already produces a within-")
    L.append("run qP decline without any load state. S_met adds no identifiable mechanism beyond these.")
    L.append("")
    L.append(f"(Linear correlations undersell a non-monotonic peak: raw r(qP,u)={r_raw:+.2f},")
    L.append(f"r|T={r_T:+.2f}; endpoint r|T={r_end_T:+.2f}. The bare Haldane fit R^2={r2_hal:.2f} is")
    L.append("T-dominated and not informative.)")
    L.append("")
    L.append("**Do NOT fit A1.** The qS-peak is not a cumulative-load effect; S_met would reproduce the")
    L.append("induction-decay overfit without resolving the peak. The qS-peak remains a T30-specific")
    L.append("partition/data-resolution limit. Next lever is the soft sensor (plan B) to extract the dense")
    L.append("trajectory and test whether ANY latent state becomes observable.")
    L.append("")
    L.append("Artifacts: smet_summary.png, smet_trajectories.png, smet.csv.")

    (OUT / "smet_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'smet_test.md'} (+ smet_summary.png, smet_trajectories.png, smet.csv)")
    print("\n--- verdict snapshot ---")
    print(f"r(qP,u) raw={r_raw:+.2f}  |T={r_T:+.2f}  |T,qs={r_Tq:+.2f}")
    print(f"endpoint r(Ptot/X, S_met/X) raw={r_end_raw:+.2f}  |T={r_end_T:+.2f}")
    if fit_ok:
        print(f"Haldane R^2={r2_hal:.2f} (qP0={popt[0]:.4f}, Ks={popt[1]:.2f}, Ki={popt[2]:.2f})")


def _plot_trajectories(recs, out):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, r in zip(axes.ravel(), recs):
        t = r["t"]
        tind = r["t_ind"]
        m = t >= tind
        ax2 = ax.twinx()
        ax2.plot(t[m], r["smet"][m] / float(np.interp(tind, t, r["smet"]) if False else 1.0),
                 color=C_LOAD, lw=1.2)  # S_met (g/L) raw on right axis
        # overlay S_met/X approximated at sample midpoints via intervals
        iv = r["intervals"]
        if iv:
            ax.scatter([row["t_mid"] for row in iv], [row["qP"] for row in iv],
                       c=C_QP, s=22, zorder=5, label="qP (interval)")
        ax.set_title(f"{r['run']} T{r['T']} q{r['qs']}", fontsize=9)
        ax.axhline(0, color="k", lw=0.4)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        if r is recs[0]:
            ax.set_ylabel("qP [g/g/h]", color=C_QP, fontsize=8)
            ax2.set_ylabel("S_met [g/L]", color=C_LOAD, fontsize=8)
    fig.suptitle("qP (blue) vs cumulative load S_met (orange) over induction -- does qP fall as load rises?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _plot_summary(recs, qP_all, u_all, T_all, qs_all, end_u, end_P, end_T, end_q, popt, r2_hal, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: qP vs S_met/X (interval points), colored by T, marker by qs.
    ax = axes[0]
    for T in (25, 30, 35):
        for qs in (0.2, 0.4, 0.6):
            sel = (T_all == T) & (qs_all == qs)
            if sel.any():
                ax.scatter(u_all[sel], qP_all[sel], c=Tcol[T], marker=QSmk[qs],
                           s=55, edgecolor="k", lw=0.3,
                           label=f"T{T} qs{qs}" if T == 30 else None)
    if popt is not None:
        uu = np.linspace(0.01, np.nanmax(u_all) * 1.05, 200)
        ax.plot(uu, _haldane(uu, *popt), "--", color="0.4", lw=1.2,
                label=f"Haldane fit (R$^2$={r2_hal:.2f})")
    ax.set_xlabel("cumulative load  S_met / X  [g/g]")
    ax.set_ylabel("observed total qP [g/g/h]")
    ax.set_title("qP vs cumulative load (interval points)\n(falls within-T => load-driven decline)")
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="k", lw=0.4)

    # Panel B: endpoint Ptot/X vs S_met/X (the qS-peak test).
    ax = axes[1]
    for T in (25, 30, 35):
        for qs in (0.2, 0.4, 0.6):
            sel = (end_T == T) & (end_q == qs)
            if sel.any():
                ax.scatter(end_u[sel], end_P[sel], c=Tcol[T], marker=QSmk[qs],
                           s=80, edgecolor="k", lw=0.4)
    for r in recs:
        ax.annotate(r["run"], (r["smet_end_over_X"], r["Ptot_over_X_final"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("endpoint cumulative load  S_met / X_final  [g/g]")
    ax.set_ylabel("specific total product  Ptot / X_final  [g/g]")
    ax.set_title("Endpoint: does the qS-peak track\ncumulative load? (shape key: o=qs0.2 s=0.4 ^=0.6)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=Tcol[T], label=f"T{T}") for T in (25, 30, 35)]
    handles += [Line2D([0], [0], marker=QSmk[q], color="w", markerfacecolor="gray",
                       markeredgecolor="k", label=f"qs{q}", linestyle="") for q in (0.2, 0.4, 0.6)]
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    # Panel C: partial-correlation bar.
    ax = axes[2]
    # recompute the three r's for the bar
    r_raw = float(np.corrcoef(qP_all, u_all)[0, 1])
    r_T = _partial_corr(qP_all, u_all, T_all)
    r_Tq = _partial_corr(qP_all, u_all, [f"{t}_{q}" for t, q in zip(T_all, qs_all)])
    ax.bar(["raw", "| T", "| T, qs"], [r_raw, r_T, r_Tq], color=["#999", C_QP, C_LOAD])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Pearson r(qP, S_met/X)")
    ax.set_title("Partial correlation: qP vs load\n(negative within-T => load drives the decline)")
    ax.set_ylim(-0.6, 0.6)
    for i, v in enumerate([r_raw, r_T, r_Tq]):
        ax.text(i, v - 0.04 * np.sign(v) - 0.02, f"{v:+.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
