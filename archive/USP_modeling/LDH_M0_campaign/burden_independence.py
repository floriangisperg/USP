"""Independence test: is offgas burden an IB-partition driver *beyond* qS (load)?

The burden-IB correlation (r~0.98 endpoint; r~0.6-0.77 time-resolved within 30 C,
see burden_ib_timeresolved.py) could be a genuine metabolic driver OR a confound of
qS: burden (extra CO2 from expression) and IB fraction both rise with expression
load, so a raw correlation cannot tell them apart. This asks the sharper question:

    Does burden explain IB-partition variance AFTER controlling for qS(t)?

Three lenses, all descriptive (the 30 C stratum has 5 runs; one of the three
constant-qS=0.4 replicates, F14, has broken gas -> low statistical power):

  1. Endpoint (n=5): partial Pearson r(burden, IB | qS0) across the 30 C runs.
  2. Time-resolved (n~20): partial r(burden, IB | qS_dyn) and a multiple
     regression IB ~ burden + qS_dyn (+ biomass), using every post-induction
     sample point in the 30 C stratum.
  3. Within-qS=0.4 (F13/F14/F15): the ONLY place qS is held constant, so any IB
     spread must come from something other than qS. Does burden track it?

qS_dyn(t) = feed_rate(t) * S_IN / (X(t) * V(t)) is the model's dynamic feed-supply
proxy (same driver the partition uses), capped at QS_CAP.

Output: results/offgas_diagnostics/burden_independence.{md,png}
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

S_IN = 400.0
QS_CAP = 2.0
RUNS_30 = ["F17", "F13", "F14", "F15", "F18"]          # the 30 C mixed stratum (IB spread)
REPS_040 = ["F13", "F14", "F15"]                       # constant-qS=0.4 center replicates


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xm, ym = x[m], y[m]
    if xm.std() < 1e-12 or ym.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(xm, ym)[0, 1])


def _partial_pearson(x, y, z):
    """Partial correlation r(x,y | z) from pairwise Pearson correlations."""
    rxy, rxz, ryz = _pearson(x, y), _pearson(x, z), _pearson(y, z)
    if any(np.isnan(v) for v in (rxy, rxz, ryz)):
        return float("nan")
    den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    return (rxy - rxz * ryz) / den


def collect(runs):
    """Per post-induction sample point: (run, t_ind, ibfrac, burden, qs_dyn, X)."""
    rows = []
    for run in runs:
        d = od.process_run(df, run)
        t = d["t"]; cer = d["cer"]; X = d["X_meas"]; ind = d["ind"]; pre = d["pre"]
        a, b = d["a_maint"], d["b_maint"]
        mu = np.gradient(np.log(np.maximum(X, 1e-3)), t)
        qco2 = cer / np.maximum(X, 1e-3)

        # Dynamic qS proxy from feed + volume (mirror offgas_diagnostics).
        r = df[df["run_name"] == run].sort_values(od._c(df, "Process time"))
        tt = r[od._c(df, "Process time")].to_numpy(float)
        feed = np.nan_to_num(r[od._c(df, "Feed [mL]")].to_numpy(float), nan=0.0)
        V = np.nan_to_num(r[od._c(df, "Volume corrected [mL]")].to_numpy(float) / 1000.0, nan=1.0)
        fr = np.zeros_like(feed)                                   # feed rate [L/h]
        dt = np.diff(tt)
        fr[:-1] = np.where(dt > 0, np.diff(feed) / dt, 0.0)
        fr[-1] = fr[-2] if len(fr) > 1 else 0.0
        fr = np.maximum(fr / 1000.0, 0.0)
        qs_dyn = np.minimum(fr * S_IN / np.maximum(V, 1e-3) / np.maximum(X, 1e-3), QS_CAP)

        # Offline sample product fractions.
        Ps = np.nan_to_num(r[od._c(df, "Protein soluble")].to_numpy(float), nan=0.0)
        Pb = np.nan_to_num(r[od._c(df, "Protein IB")].to_numpy(float), nan=0.0)
        is_s = r[od._c(df, "is_sample")].fillna(False).to_numpy(bool)
        t0ind = t[ind][0] if ind.any() else np.nan
        for ti, pso, pbo, sm in zip(tt, Ps, Pb, is_s):
            if not sm:
                continue
            ptot = pso + pbo
            if not np.isfinite(ptot) or ptot < 0.2:               # need measurable product
                continue
            ibf = pbo / ptot
            qsi = float(np.interp(ti, t, qs_dyn))
            burden = float(np.interp(ti, t, qco2) - (a * float(np.interp(ti, t, mu)) + b)) \
                if np.isfinite(a) else float("nan")
            xi = float(np.interp(ti, t, X))
            rows.append((run, ti - t0ind, ibf, burden, qsi, xi))
    return pd.DataFrame(rows, columns=["run", "t_ind", "ibfrac", "burden", "qs_dyn", "X"])


def _ols(df_, ycol, xcols):
    """Ordinary least squares; return (coefs dict, R^2, n)."""
    fin = df_[[ycol] + xcols].dropna()
    n = len(fin)
    if n < len(xcols) + 2:
        return None, float("nan"), n
    A = np.column_stack([fin[c].to_numpy(float) for c in xcols] + [np.ones(n)])
    sol, *_ = np.linalg.lstsq(A, fin[ycol].to_numpy(float), rcond=None)
    pred = A @ sol
    ss_res = float(np.sum((fin[ycol].to_numpy(float) - pred) ** 2))
    ss_tot = float(np.sum((fin[ycol].to_numpy(float) - fin[ycol].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return dict(zip(xcols, sol[:-1])), r2, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sub = collect(RUNS_30)

    # ---- Lens 1: endpoint (per-run final IB fraction vs median burden/qS) ----
    eps = []
    for run, g in sub.groupby("run"):
        ib_end = float(g.sort_values("t_ind")["ibfrac"].iloc[-1])      # last post-ind sample
        eps.append((run, ib_end, float(g["burden"].median()),
                    float(g["qs_dyn"].median()), od.RUN_INFO[run][1]))
    ep = pd.DataFrame(eps, columns=["run", "ib_end", "burden_med", "qs_med", "qs0"])
    r_b = _pearson(ep["burden_med"], ep["ib_end"])
    r_q = _pearson(ep["qs_med"], ep["ib_end"])
    r_bq = _pearson(ep["burden_med"], ep["qs_med"])
    pr_b = _partial_pearson(ep["burden_med"], ep["ib_end"], ep["qs_med"])
    pr_q = _partial_pearson(ep["qs_med"], ep["ib_end"], ep["burden_med"])

    # ---- Lens 2: time-resolved partial correlation + multiple regression ----
    r_b_t = _pearson(sub["burden"], sub["ibfrac"])
    r_q_t = _pearson(sub["qs_dyn"], sub["ibfrac"])
    pr_b_t = _partial_pearson(sub["burden"], sub["ibfrac"], sub["qs_dyn"])
    pr_q_t = _partial_pearson(sub["qs_dyn"], sub["ibfrac"], sub["burden"])
    coefs_bq, r2_bq, n_bq = _ols(sub, "ibfrac", ["burden", "qs_dyn"])
    coefs_bqx, r2_bqx, n_bqx = _ols(sub, "ibfrac", ["burden", "qs_dyn", "X"])
    # burden-only and qS-only baselines (does the other add anything?)
    coefs_b_only, r2_b_only, _ = _ols(sub, "ibfrac", ["burden"])
    coefs_q_only, r2_q_only, _ = _ols(sub, "ibfrac", ["qs_dyn"])

    # ---- Lens 3: within-qS=0.4 replicates (constant qS -> burden must explain IB) ----
    rep_rows = []
    for run in REPS_040:
        g = sub[sub["run"] == run]
        rep_rows.append((run, float(g["ibfrac"].iloc[-1]), float(g["burden"].median()),
                         od.RUN_INFO[run][2] if False else None))
    rep = pd.DataFrame(rep_rows, columns=["run", "ib_end", "burden_med", "_"])
    # RQ-good fraction per replicate (F14 gas is broken -> its burden is unreliable).
    rgood = {run: float(np.mean([od.process_run(df, run)["rq_good_frac"]])) for run in REPS_040}

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for run, g in sub.groupby("run"):
        ax.scatter(g["burden"], g["ibfrac"], label=run, s=45)
    ax.set_xlabel("burden qCO2 (mmol/g/h, maintenance-corrected)")
    ax.set_ylabel("IB fraction at sample time")
    ax.set_title(f"30 C time-resolved\nr(burden,IB)={r_b_t:.2f}, "
                 f"partial r|qS={pr_b_t:.2f}")
    ax.legend(fontsize=8)
    ax = axes[1]
    for run, g in sub.groupby("run"):
        ax.scatter(g["qs_dyn"], g["ibfrac"], label=run, s=45)
    ax.set_xlabel("dynamic qS (feed proxy) [g/g/h]")
    ax.set_ylabel("IB fraction at sample time")
    ax.set_title(f"30 C time-resolved\nr(qS,IB)={r_q_t:.2f}, "
                 f"partial r|burden={pr_q_t:.2f}")
    ax.legend(fontsize=8)
    ax = axes[2]
    x = np.arange(len(rep))
    ax.bar(x - 0.2, rep["ib_end"], 0.4, color="#444", label="final IB frac")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, rep["burden_med"], 0.4, color="#D55E00", label="median burden")
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n(RQgood {rgood[r]*100:.0f}%)" for r in rep["run"]])
    ax.set_ylabel("final IB fraction", color="#444")
    ax2.set_ylabel("median burden qCO2", color="#D55E00")
    ax.set_title("Within-qS=0.4 replicates\n(constant qS -> burden must track IB)")
    fig.tight_layout(); fig.savefig(OUT / "burden_independence.png", dpi=120); plt.close(fig)

    # ---- Markdown ----
    L = ["# Burden vs IB — independence from qS", "",
         "Does offgas burden (qCO2 above the growth baseline) drive the IB partition **beyond**",
         "what the dynamic feed-supply qS already explains? Raw burden-IB correlation is",
         "confounded by qS (both rise with expression load). Three lenses:", ""]
    L.append("## Lens 1 — 30 C endpoints (n=5)")
    L.append("| run | qS0 | IB_end | burden_med | qS_med |")
    L.append("| --- | -:| -:| -:| -:|")
    for _, r in ep.iterrows():
        L.append(f"| {r['run']} | {r['qs0']} | {r['ib_end']:.3f} | {r['burden_med']:.2f} | {r['qs_med']:.2f} |")
    L.append("")
    L.append(f"- r(burden, IB) = {r_b:.2f};  r(qS, IB) = {r_q:.2f};  r(burden, qS) = {r_bq:.2f}")
    L.append(f"- **partial r(burden, IB | qS) = {pr_b:.2f}**;  partial r(qS, IB | burden) = {pr_q:.2f}")
    L.append(f"- n=5 endpoints (1 per run) -> essentially no residual power; descriptive only.")
    L.append("")
    L.append("## Lens 2 — 30 C time-resolved (n=%d post-induction sample points)" % len(sub))
    L.append(f"- r(burden, IB) = {r_b_t:.2f};  r(qS, IB) = {r_q_t:.2f}")
    L.append(f"- **partial r(burden, IB | qS) = {pr_b_t:.2f}**;  partial r(qS, IB | burden) = {pr_q_t:.2f}")
    L.append(f"- OLS IB ~ burden + qS:  burden coef={coefs_bq['burden']:.2f}, qS coef={coefs_bq['qs_dyn']:.2f}, R^2={r2_bq:.2f} (n={n_bq})")
    L.append(f"- OLS IB ~ burden + qS + X: burden coef={coefs_bqx['burden']:.2f}, qS coef={coefs_bqx['qs_dyn']:.2f}, X coef={coefs_bqx['X']:.2f}, R^2={r2_bqx:.2f} (n={n_bqx})")
    L.append(f"- baselines: IB~burden only R^2={r2_b_only:.2f};  IB~qS only R^2={r2_q_only:.2f}")
    L.append("- Delta R^2 (adding burden to qS) and (adding qS to burden) tells which carries the unique signal.")
    L.append("")
    L.append("## Lens 3 — within-qS=0.4 replicates (F13/F14/F15, qS held constant)")
    L.append("| run | final IB frac | median burden | RQ-good % |")
    L.append("| --- | -:| -:| -:|")
    for _, r in rep.iterrows():
        L.append(f"| {r['run']} | {r['ib_end']:.3f} | {r['burden_med']:.2f} | {rgood[r['run']]*100:.0f} |")
    L.append("")
    L.append("- Constant qS => any IB spread must be driven by something other than qS.")
    L.append("- F14's gas is broken (RQ-good ~1%), so its burden estimate is unreliable; the reliable")
    L.append("  contrast is essentially F13 vs F15 (IB 0.123 vs 0.116, burden 1.11 vs 1.03) — too close to call.")
    L.append("")
    L.append("## Read")
    L.append("- If partial r(burden, IB | qS) collapses toward 0 and adding burden to qS barely raises R^2,")
    L.append("  burden is a **proxy for qS**, not an independent driver -> the dynamic burden state is")
    L.append("  expected to be non-identifiable beyond the a_qS term (consistent with the qP-collinearity caveat).")
    L.append("- If partial r stays substantial and burden adds R^2 on top of qS, burden carries independent")
    L.append("  signal -> worth the profile-likelihood test of a_b in M0_aload_burden.")
    (OUT / "burden_independence.md").write_text("\n".join(L))
    print(f"saved {OUT/'burden_independence.png'} and burden_independence.md")
    print("\n".join(L))


if __name__ == "__main__":
    main()
