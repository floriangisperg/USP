"""Offgas diagnostics for the LDH playground.

Computes gas-only quantities that are independent of the offline biomass/protein
measurements, to test whether the offgas is consistent with the rest of the data
and whether it carries product/IB-relevant signal:

1. RQ quality + O2-enrichment regime per run (which windows of the gas trace are
   trustworthy; flags broken/offset sensors and O2-transfer-limited plateaus).
2. Offgas-implied biomass X_gas(t) from the cumulative carbon balance
   (C_biomass = C_fed - C_CO2 - C_product), independent of offline X.
3. Pre-induction Y_xs from gas (feed r_S + CER only; no biomass used) vs the
   biomass-fit value 0.351.
4. Post-induction "burden" respiration (qCO2 above growth stoichiometry) and its
   correlation with the IB fraction -- a candidate metabolic signature of IB
   aggregation burden.

Outputs: plots/offgas_<run>.png, OFFGAS_DIAGNOSTICS.md (per-run table + the
burden-vs-IB correlation).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR.parent / "LDH_consolidated_results_processed.pkl"
OUT_DIR = SCRIPT_DIR / "results" / "offgas_diagnostics"

S_IN = 400.0
MW_S, MW_X, MW_P = 30.03, 26.565, 22.58  # g / C-mol
RQ_GOOD = (0.8, 1.2)  # trustworthy RQ window

# Verified FCCD setpoints (T, qS0) and final IB fractions (from the data).
RUN_INFO = {
    "F07": (35, 0.2, 1.000), "F09": (25, 0.2, 0.000), "F10": (25, 0.6, 0.000),
    "F12": (35, 0.6, 1.000), "F13": (30, 0.4, 0.123), "F14": (30, 0.4, 0.147),
    "F15": (30, 0.4, 0.116), "F16": (25, 0.4, 0.018), "F17": (30, 0.2, 0.021),
    "F18": (30, 0.6, 0.189), "F19": (35, 0.4, 1.000),
}


def _c(df, base):
    if base in df.columns:
        return base
    b = base.split(" [")[0].lower()
    for c in df.columns:
        if c.split(" [")[0].lower() == b:
            return c
    raise KeyError(base)


def process_run(df: pd.DataFrame, run: str) -> dict:
    r = df[df["run_name"] == run].sort_values(_c(df, "Process time"))
    t = r[_c(df, "Process time [h]")].to_numpy(float)
    feed = r[_c(df, "Feed [mL]")].to_numpy(float)
    feed = np.nan_to_num(feed, nan=0.0)
    V = np.nan_to_num(r[_c(df, "Volume corrected [mL]")].to_numpy(float) / 1000.0, nan=1.0)
    Tc = np.nan_to_num(r[_c(df, "Temperature [°C]")].to_numpy(float), nan=30.0)
    cer = r[_c(df, "CER [mmol/L/h]")].to_numpy(float)
    our = r[_c(df, "OUR [mmol/L/h]")].to_numpy(float)
    inO2 = np.nan_to_num(r[_c(df, "Inlet O2 [%]")].to_numpy(float), nan=21.0)
    ind = (np.nan_to_num(r[_c(df, "Is induced [-]")].to_numpy(float), nan=0.0) > 0.5)
    X = r[_c(df, "Biomass [g/L]")].to_numpy(float)
    Psol = r[_c(df, "Protein soluble [g/L]")].to_numpy(float)
    Pib = r[_c(df, "Protein IB [g/L]")].to_numpy(float)
    Ptot = np.nan_to_num(Psol, nan=0.0) + np.nan_to_num(Pib, nan=0.0)

    # feed rate [L/h] (forward diff + clip)
    fr = np.zeros_like(feed)
    dt = np.diff(t)
    fr[:-1] = np.where(dt > 0, np.diff(feed) / dt, 0.0)
    fr[-1] = fr[-2] if len(fr) > 1 else 0.0
    fr = np.maximum(fr / 1000.0, 0.0)  # mL/h -> L/h

    feed_on = feed > 0
    pre = feed_on & (~ind)
    rS = fr * S_IN / np.maximum(V, 1e-3)            # g substrate / L / h (uptake = supply)
    rq = cer / np.where(np.abs(our) > 1e-6, our, np.nan)
    enrich = inO2 - 21.0

    # (1) RQ quality: fraction of fed phase with sane RQ; max enrichment.
    fed = feed_on
    rq_finite = rq[np.isfinite(rq) & fed]
    rq_good_frac = float(np.mean((rq_finite >= RQ_GOOD[0]) & (rq_finite <= RQ_GOOD[1]))) if len(rq_finite) else 0.0
    max_enrich = float(np.nanmax(enrich))

    # (2) Gas-implied biomass, RATE-BASED, FED PHASE ONLY, anchored at the measured
    # feed-onset biomass. The cumulative carbon balance is NOT usable during batch:
    # the data has no substrate-uptake tracking then ("Glucose consumed" = all batch
    # glucose upfront at t=0; "Glucose consumed rate" = 0 in batch), so a cumulative
    # balance starts at "~all batch glucose is biomass" and decreases as CO2 rises.
    # Instead: gas-implied biomass formation rate rX = MW_X*(rS/MW_S - CER/1000)
    # [g/L/h], with rS = feed-supply uptake (valid only once feeding starts), and
    # integrate total biomass B from the measured feed-onset biomass B0.
    Ptot_online = np.interp(t, t[np.isfinite(Ptot)], np.nan_to_num(Ptot[np.isfinite(Ptot)], nan=0.0))
    rS = fr * S_IN / np.maximum(V, 1e-3)                                   # g substrate / L / h
    rX = MW_X * (rS / MW_S - np.nan_to_num(cer, nan=0.0) / 1000.0)         # g biomass / L / h (from gas)
    X_gas = np.full_like(t, np.nan)
    fonset = np.where(feed_on)[0]
    if len(fonset) and np.isfinite(X).any():
        i0 = fonset[0]
        Xm = np.interp(t, t[np.isfinite(X)], X[np.isfinite(X)])           # offline X on the grid
        B0 = Xm[i0] * V[i0]                                                # anchor: measured biomass at feed onset
        dt_full = np.concatenate([[0.0], np.diff(t)])
        cs = np.cumsum(rX * V * dt_full)
        B = B0 + cs - cs[i0]                                               # total biomass (g) from gas-implied growth
        X_gas = B / np.maximum(V, 1e-3)
        X_gas[:i0] = np.nan                                                # not computable before feed

    # gas-implied biomass at induction onset (compare to nearest measured X)
    ind_idx = np.where(ind)[0]
    if len(ind_idx):
        i1 = ind_idx[0]
        meas_near = X[np.isfinite(X)]
        X_meas_at_ind = float(np.interp(t[i1], t[np.isfinite(X)], meas_near)) if len(meas_near) else np.nan
        X_gas_at_ind = float(X_gas[i1])
    else:
        X_meas_at_ind = X_gas_at_ind = np.nan

    # (3) Pre-induction Y_xs from gas: Yxs = MW_X/MW_S - MW_X*CER/(1000*rS).
    Yxs_gas = MW_X / MW_S - MW_X * cer / (1000.0 * np.where(rS > 1e-6, rS, np.nan))
    m = pre & (rS > 1e-3) & np.isfinite(Yxs_gas) & np.isfinite(rq) & (rq >= RQ_GOOD[0]) & (rq <= RQ_GOOD[1]) & (Yxs_gas > 0.05) & (Yxs_gas < 0.95)
    Yxs_pre = Yxs_gas[m]
    yxs_gas_med = float(np.median(Yxs_pre)) if len(Yxs_pre) else np.nan

    # (4) Post-induction burden: qCO2_excess = qCO2_obs - mu/Y_XCO2_growth.
    # Growth-expected qCO2 from measured biomass: mu = d(ln X)/dt ; qCO2_growth = mu / Y_XCO2
    # with Y_XCO2 chosen so that pre-induction it matches (calibrate per run from pre-phase).
    X_interp = np.interp(t, t[np.isfinite(X)], meas_near) if len(meas_near) else np.full_like(t, np.nan)
    mu = np.gradient(np.log(np.maximum(X_interp, 1e-3)), t) if len(meas_near) else np.full_like(t, np.nan)
    qco2 = cer / np.maximum(X_interp, 1e-3)  # mmol CO2 / gX / h
    # apparent yield of biomass on CO2 (C-mol): from pre-induction, qCO2_growth = mu/MW_X*1000*(... )
    # Simplify: pre-induction qCO2 ≈ mu/MW_X*1000*(MW_X/MW_S... ) -- instead calibrate qCO2 = a*mu + b (maintenance)
    mp = pre & np.isfinite(mu) & np.isfinite(qco2) & (mu > -0.5)
    a_maint = b_maint = np.nan
    if mp.sum() > 20:
        A = np.vstack([mu[mp], np.ones(mp.sum())]).T
        sol, *_ = np.linalg.lstsq(A, qco2[mp], rcond=None)
        a_maint, b_maint = sol  # qCO2 = a_maint*mu + b_maint (b_maint = maintenance respiration)
    # burden during induction = observed qCO2 - (a_maint*mu + b_maint)
    indf = ind & np.isfinite(mu) & np.isfinite(qco2) & np.isfinite(a_maint)
    burden = (qco2[indf] - (a_maint * mu[indf] + b_maint)) if np.isfinite(a_maint) else np.array([])
    burden_med = float(np.nanmedian(burden)) if len(burden) else np.nan

    Tset, qs0, ibfrac = RUN_INFO[run]
    return dict(run=run, T=Tset, qs0=qs0, ibfrac=ibfrac, t=t, cer=cer, our=our, rq=rq,
                enrich=enrich, X_meas=X_interp, X_gas=X_gas, ind=ind, pre=pre, feed_on=feed_on,
                rq_good_frac=rq_good_frac, max_enrich=max_enrich, X_gas_at_ind=X_gas_at_ind,
                X_meas_at_ind=X_meas_at_ind, yxs_gas_med=yxs_gas_med, n_yxs=int(m.sum()),
                burden_med=burden_med, a_maint=a_maint, b_maint=b_maint)


def plot_run(d: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    good = np.isfinite(d["rq"]) & (d["rq"] >= RQ_GOOD[0]) & (d["rq"] <= RQ_GOOD[1]) & d["feed_on"]
    ax.scatter(d["t"][good], d["rq"][good], s=3, c="#2ca02c", label=f"RQ in {RQ_GOOD}")
    bad = np.isfinite(d["rq"]) & d["feed_on"] & ~good
    ax.scatter(d["t"][bad], d["rq"][bad], s=3, c="#d62728", label="RQ outside (untrusted)")
    ax.set_ylim(0, 2.5); ax.set_title(f"{d['run']} RQ  (good frac={d['rq_good_frac']:.2f}, maxEnr={d['max_enrich']:.0f}%)")
    ax.axhline(1.0, color="k", lw=0.5); ax.set_ylabel("RQ = CER/OUR"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(d["t"], d["cer"], c="#0072B2", lw=0.8, label="CER")
    ax.plot(d["t"], d["our"], c="#D55E00", lw=0.8, label="OUR")
    ax.set_title(f"{d['run']} offgas rates"); ax.set_ylabel("mmol/L/h"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(d["t"], d["X_gas"], c="#0072B2", lw=1.2, label="X from gas (C-balance)")
    m = np.isfinite(d["X_meas"])
    ax.scatter(d["t"][m], d["X_meas"][m], c="k", s=20, zorder=5, label="X measured")
    ax.set_title(f"{d['run']} biomass: gas-implied vs measured"); ax.set_ylabel("Biomass [g/L]"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(d["t"], d["enrich"], c="#9467bd", lw=0.8)
    ax.set_title(f"{d['run']} O2 enrichment (Inlet O2 - 21%)"); ax.set_ylabel("%"); ax.set_xlabel("Process time [h]")
    for a in axes.flat:
        if d["ind"].any():
            a.axvline(d["t"][d["ind"]][0], color="#999", ls="--", lw=0.7)
    # Omit the batch phase entirely: start the x-axis at feed onset (the gas analysis
    # is fed-phase only; the model and the gas-implied biomass both start there).
    if d["feed_on"].any():
        fo = d["t"][d["feed_on"]][0]
        for a in axes.flat:
            a.set_xlim(left=fo)
    fig.tight_layout()
    fig.savefig(out, dpi=120); plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(DATA_FILE)
    recs = [process_run(df, r) for r in RUN_INFO]
    for d in recs:
        plot_run(d, OUT_DIR / f"offgas_{d['run']}.png")

    # Markdown summary.
    L = ["# Offgas diagnostics", "",
         "## Per-run summary",
         "| run | T | qS0 | IBfrac | Yxs_gas(pre) | n_yxs | RQ_good% | maxEnrich | Xgas@ind | Xmeas@ind | burden qCO2 | a_maint | b_maint(maint) |",
         "| --- | -:| -:| -:| ---:| -:| -:| -:| ---:| ---:| ---:| ---:| ---:|"]
    for d in recs:
        L.append(f"| {d['run']} | {d['T']} | {d['qs0']} | {d['ibfrac']:.3f} | {d['yxs_gas_med']:.3f} | {d['n_yxs']} | "
                 f"{d['rq_good_frac']*100:.0f} | {d['max_enrich']:.0f} | {d['X_gas_at_ind']:.1f} | {d['X_meas_at_ind']:.1f} | "
                 f"{d['burden_med']:.2f} | {d['a_maint']:.2f} | {d['b_maint']:.2f} |")
    L.append("")
    L.append("- Yxs_gas(pre): pre-induction apparent yield from gas (feed r_S + CER only; no biomass). biomass-fit value = 0.351.")
    L.append("- RQ_good%: fraction of fed phase with RQ in [0.8,1.2] (trustworthy). Low => sensor artifact / O2-limitation.")
    L.append("- Xgas@ind vs Xmeas@ind: gas-implied vs measured biomass at induction (independent check).")
    L.append("- burden qCO2: median post-induction qCO2 above the pre-induction mu-linear baseline (maintenance-corrected).")

    # Burden-vs-IB correlation (only the 30C mixed stratum has IB spread).
    L.append("")
    L.append("## Burden vs IB fraction (30 °C mixed stratum: F17/F13/F14/F15/F18)")
    sub = [d for d in recs if d["T"] == 30]
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [d["burden_med"] for d in sub]; ys = [d["ibfrac"] for d in sub]
    ax.scatter(xs, ys, c="#0072B2")
    for d in sub:
        ax.annotate(d["run"], (d["burden_med"], d["ibfrac"]), fontsize=9)
    ax.set_xlabel("post-induction burden qCO2 (mmol/g/h, maintenance-corrected)")
    ax.set_ylabel("final IB fraction")
    ax.set_title("IB fraction vs offgas burden (30 C)")
    xsa = np.array(xs, float); ysa = np.array(ys, float)
    fin = np.isfinite(xsa) & np.isfinite(ysa)
    if fin.sum() >= 3:
        r = float(np.corrcoef(xsa[fin], ysa[fin])[0, 1])
        ax.set_title(f"IB fraction vs offgas burden (30 C)  Pearson r={r:.2f}")
    fig.tight_layout(); fig.savefig(OUT_DIR / "burden_vs_IB.png", dpi=120); plt.close(fig)
    if fin.sum() >= 3:
        L.append(f"Pearson r (burden qCO2 vs IB fraction, 30 C) = {r:.2f}")
    (OUT_DIR / "OFFGAS_DIAGNOSTICS.md").write_text("\n".join(L))
    print(f"wrote {OUT_DIR/'OFFGAS_DIAGNOSTICS.md'} and {len(recs)} run plots + burden_vs_IB.png")


if __name__ == "__main__":
    main()
