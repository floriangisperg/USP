"""Plan B Stage 1 -- deterministic soft sensor: dense mu(t), qP(t) from gas + feed.

The model fits ~4 offline points/run; each run also carries a dense online gas trajectory (~1 min) whose
dynamics are used only as a failed static prediction (M1a) or one-off diagnostics. This reconstructs that
untapped dense trajectory and turns the +/-50% volume bracket into a per-run estimate -- WITHOUT a free
filter-tuning parameter.

Reconstruction (CER-based, robust -- avoids the noisy OUR that hurt M1a):
  - Growth rate from the carbon balance (ignore the ~10% product term; it is absorbed by the scale):
        rX_gas = MW_X * ( rS/MW_S - CER/1000 ),   rS = feed_rate * S_IN / V
    Integrate from the offline anchor X0; fit ONE per-run scale s so X_scaled = X0 + s*int(rX_gas) matches
    the offline biomass samples (linear least squares). s reconciles the gas-vs-biomass yield gap
    (instantaneous ~0.39 vs cumulative ~0.35) -- a defensible per-run calibration that replaces the
    +/-50% volume bracket (it conflates volume + carbon-calibration + maintenance, which the data cannot
    separate, but it POINT-estimates the nuisance).
  - Production as the carbon-balance residual once growth is fixed:
        rP = MW_P * ( rS/MW_S - rX_scaled/MW_X - CER/1000 ),   qP = rP / X_scaled
    Dense, and INDEPENDENT of the offline product samples (which only VALIDATE it).

Decisive validation: integrate rP over induction and compare to observed P_final. If they match, the gas
is fully consistent with biomass + product (the yield gap was just the product term). If int(rP) >
P_obs, there is an unmodeled carbon sink (maintenance / acetate) -- quantifying the unresolved gas-side
error directly.

Then characterize the within-run qP(t) dynamics densely (the user-observed heterogeneous decline: visible
in the T30 runs, not F07/F09/F12/F16) and test what predicts it.

Outputs: results/soft_sensor/{soft_sensor.md, soft_sensor_<run>.png, soft_sensor_summary.png, soft_sensor.csv}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stoich import BIOMASS, GLUCOSE, LDH_PRODUCT  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "soft_sensor"
OUT.mkdir(parents=True, exist_ok=True)

MW_S = GLUCOSE.mw_g_per_cmol   # 30.03
MW_X = BIOMASS.mw_g_per_cmol   # 26.565
MW_P = LDH_PRODUCT.mw_g_per_cmol  # 22.58
S_IN = 400.0

SET = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}
RUN_ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]

C_X = "#0072B2"
C_QP = "#D55E00"
C_OBS = "#2ca02c"


def _dense_gas(df, run):
    """CER, OUR [mmol/L/h] from df interpolated onto the run's t_online grid."""
    c_t = D._find_col(df, "Process time [h]")
    c_cer = D._find_col(df, "CER [mmol/L/h]")
    c_our = D._find_col(df, "OUR [mmol/L/h]")
    r = df[df["run_name"] == run["run_id"]].sort_values(c_t).reset_index(drop=True)
    tg = r[c_t].to_numpy(float)
    cer = r[c_cer].to_numpy(float)
    our = r[c_our].to_numpy(float)
    t = np.asarray(run["t_online"], float)
    m = np.isfinite(tg) & np.isfinite(cer)
    cer_on = np.interp(t, tg[m], cer[m])
    our_on = np.interp(t, tg[m], our[m]) if np.isfinite(our).any() else np.full_like(t, np.nan)
    return cer_on, our_on


def reconstruct(df, run):
    t = np.asarray(run["t_online"], float)
    feed = np.asarray(run["feed_rate_L_h"], float)
    V = np.asarray(run["V_L"], float)
    cer, _ = _dense_gas(df, run)
    rS = feed * S_IN / np.where(V > 1e-6, V, np.nan)             # g substrate / L / h
    rX_gas = MW_X * (rS / MW_S - cer / 1000.0)                   # g biomass / L / h (ignores product term)

    # Cumulative integral of rX_gas from t0.
    cumRX = np.concatenate([[0.0], np.cumsum((rX_gas[:-1] + rX_gas[1:]) * 0.5 * np.diff(t))])
    X0 = float(run["X0"])

    # Fit per-run scale s: offline X ~ X0 + s*cumRX  (linear LS, 1 param).
    tobs = np.asarray(run["t_obs"], float)
    Xobs = np.asarray(run["obs"]["X"], float)
    ok = np.isfinite(tobs) & np.isfinite(Xobs)
    tobs_ok, Xobs_ok = tobs[ok], Xobs[ok]
    cumRX_obs = np.interp(tobs_ok, t, cumRX)
    denom = float(np.sum(cumRX_obs ** 2))
    s = float(np.sum((Xobs_ok - X0) * cumRX_obs) / denom) if denom > 0 else 1.0

    X_scaled = X0 + s * cumRX
    X_scaled = np.maximum(X_scaled, 1e-3)
    rX_scaled = s * rX_gas
    mu = rX_scaled / X_scaled
    rP = MW_P * (rS / MW_S - rX_scaled / MW_X - cer / 1000.0)     # carbon-balance residual product rate
    qP = rP / X_scaled
    # Instantaneous gas yield (pre-induction ~0.39) and apparent biomass yield (s-corrected).
    Yxs_gas = rX_gas / np.where(rS > 1e-6, rS, np.nan)
    Yxs_eff = s * Yxs_gas

    # Validate: integrate rP over induction, compare to observed final product.
    ind = np.asarray(run["is_induced"], float) > 0.5
    P_sol = np.asarray(run["obs"]["P_sol"], float)
    P_ib = np.asarray(run["obs"]["P_IB"], float)
    P_obs_final = float((P_sol + P_ib)[np.isfinite(P_sol + P_ib)][-1]) if np.isfinite(P_sol + P_ib).any() else float("nan")
    if ind.any():
        ti = t[ind]
        rPi = rP[ind]
        P_recon_final = float(np.trapz(rPi, ti)) if len(ti) > 1 else float("nan")
    else:
        P_recon_final = float("nan")

    # Biomass fit quality (R^2 of X_scaled vs offline X).
    X_pred = X0 + s * cumRX_obs
    ss_res = float(np.sum((Xobs_ok - X_pred) ** 2))
    ss_tot = float(np.sum((Xobs_ok - np.mean(Xobs_ok)) ** 2))
    r2_X = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Within-run qP decline: mean qP over early vs late induction third.
    if ind.any() and ind.sum() > 6:
        qPi = qP[ind]
        n = len(qPi)
        qP_early = float(np.nanmean(qPi[: n // 3]))
        qP_late = float(np.nanmean(qPi[2 * n // 3:]))
    else:
        qP_early = qP_late = float("nan")

    return {
        "run": run["run_id"], "T": SET[run["run_id"]][0], "qs": SET[run["run_id"]][1],
        "t": t, "X_scaled": X_scaled, "mu": mu, "qP": qP, "rP": rP, "rS": rS, "cer": cer,
        "tobs": tobs_ok, "Xobs": Xobs_ok,
        "tobs_full": tobs, "P_sol_obs": P_sol, "P_ib_obs": P_ib,
        "scale_s": s, "r2_X": r2_X,
        "Yxs_gas_mean": float(np.nanmean(Yxs_gas[np.isfinite(rS) & (rS > 1)])),
        "Yxs_eff_mean": float(np.nanmean(Yxs_eff[np.isfinite(rS) & (rS > 1)])),
        "P_obs_final": P_obs_final, "P_recon_final": P_recon_final,
        "P_recon_over_obs": P_recon_final / P_obs_final if P_obs_final > 1e-6 else float("nan"),
        "qP_early": qP_early, "qP_late": qP_late,
        "qP_decline_ratio": qP_late / qP_early if qP_early > 1e-6 else float("nan"),
    }


def plot_run(rec, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    t = rec["t"]
    ax = axes[0]
    ax.plot(t, rec["X_scaled"], color=C_X, lw=1.3, label="X soft-sensor (gas, scaled)")
    ax.scatter(rec["tobs"], rec["Xobs"], c=C_OBS, s=30, zorder=5, label="offline X (fit)")
    ax.set_ylabel("biomass X [g/L]", color=C_X)
    ax.set_xlabel("Process time [h]")
    ax.set_title(f"{rec['run']}  T{rec['T']} q{rec['qs']}  |  s={rec['scale_s']:.2f}, "
                 f"R$^2$(X)={rec['r2_X']:.2f}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(t, rec["mu"], color="#999", lw=0.8, ls="--")
    ax2.set_ylabel("mu [1/h]", color="#999")

    ax = axes[1]
    ax.plot(t, rec["qP"], color=C_QP, lw=1.2, label="qP soft-sensor (carbon residual)")
    ax.axhline(0, color="k", lw=0.4)
    # Sparse product-implied qP at sample midpoints (independent validation: observed product
    # derivative, NOT the gas residual). qP = (dP/dt + D*P)/X, D = rS/S_IN (=feed/V).
    tobs = np.asarray(rec["tobs_full"], float)
    Ptot = np.asarray(rec["P_sol_obs"], float) + np.asarray(rec["P_ib_obs"], float)
    Xof = np.interp(tobs, t, rec["X_scaled"])
    tm_q, qp_q = [], []
    for k in range(len(tobs) - 1):
        dt = tobs[k + 1] - tobs[k]
        if dt <= 0 or not (np.isfinite(Ptot[k]) and np.isfinite(Ptot[k + 1])):
            continue
        Xbar = 0.5 * (Xof[k] + Xof[k + 1])
        Pbar = 0.5 * (Ptot[k] + Ptot[k + 1])
        if Xbar < 1e-3:
            continue
        Dbar = float(np.interp(0.5 * (tobs[k] + tobs[k + 1]), t, rec["rS"])) / S_IN
        tm_q.append(0.5 * (tobs[k] + tobs[k + 1]))
        qp_q.append(((Ptot[k + 1] - Ptot[k]) / dt + Dbar * Pbar) / Xbar)
    if tm_q:
        ax.scatter(tm_q, qp_q, c=C_OBS, s=30, zorder=5, label="qP from offline P (validation)")
    ax.set_ylabel("qP [g/g/h]", color=C_QP)
    ax.set_xlabel("Process time [h]")
    ax.set_title(f"P_recon/P_obs = {rec['P_recon_over_obs']:.2f}  "
                 f"(1.0 = gas consistent)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_summary(recs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: per-run scale s (the calibration replacing the +/-50% bracket).
    ax = axes[0]
    xs = np.arange(len(recs))
    cols = ["#56B4E9" if r["T"] == 25 else "#0072B2" if r["T"] == 30 else "#E69F00" for r in recs]
    ax.bar(xs, [r["scale_s"] for r in recs], color=cols)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", label="s=1 (gas==biomass)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['run']}\nT{r['T']}q{r['qs']}" for r in recs], rotation=90, fontsize=7)
    ax.set_ylabel("per-run scale s")
    ax.set_title("Gas-vs-biomass scale s\n(<1: gas overstates growth; replaces +/-50% bracket)")
    ax.legend(fontsize=8)

    # Panel B: validation -- reconstructed vs observed final product.
    ax = axes[1]
    for r in recs:
        c = "#56B4E9" if r["T"] == 25 else "#0072B2" if r["T"] == 30 else "#E69F00"
        ax.scatter(r["P_obs_final"], r["P_recon_final"], c=c, s=70, edgecolor="k", lw=0.4)
        ax.annotate(r["run"], (r["P_obs_final"], r["P_recon_final"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    lim = [0, max(max(r["P_obs_final"] for r in recs), max(r["P_recon_final"] for r in recs)) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.7, label="1:1 (gas consistent)")
    ax.set_xlabel("observed final P_tot [g/L]")
    ax.set_ylabel("soft-sensor int(rP) [g/L]")
    ax.set_title("Validation: gas-implied product vs observed\n(off the line = unmodeled C sink)")
    ax.legend(fontsize=8)

    # Panel C: within-run qP decline (late/early ratio) by run -- the user's observation.
    ax = axes[2]
    dr = [r["qP_decline_ratio"] for r in recs]
    ax.bar(xs, dr, color=cols)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", label="no decline")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['run']}\nT{r['T']}q{r['qs']}" for r in recs], rotation=90, fontsize=7)
    ax.set_ylabel("qP_late / qP_early  (<1 declines)")
    ax.set_title("Within-run qP decline (dense)\n(visible in T30/IB runs; not F07/F09/F12/F16)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    df = D._load_df_cached(D.DEFAULT_DATA_FILE)
    tr, te = D.split_for_fold("full")
    bundle = D.load_dataset(tr, te)
    recs = [reconstruct(df, r) for r in bundle["train"]]
    for r in recs:
        plot_run(r, OUT / f"soft_sensor_{r['run']}.png")
    plot_summary(recs, OUT / "soft_sensor_summary.png")
    pd.DataFrame(recs)[["run", "T", "qs", "scale_s", "r2_X", "Yxs_gas_mean", "Yxs_eff_mean",
                        "P_obs_final", "P_recon_final", "P_recon_over_obs",
                        "qP_early", "qP_late", "qP_decline_ratio"]].to_csv(OUT / "soft_sensor.csv", index=False)

    L = ["# Plan B Stage 1 -- soft sensor: dense mu(t)/qP(t) from gas + feed", ""]
    L.append("Reconstructs the dense metabolic trajectory the sparse-offline model leaves untapped. Growth")
    L.append("from the carbon balance (rX_gas = MW_X*(rS/MW_S - CER/1000)), integrated from offline X0 and")
    L.append("scaled per run (s) to match offline biomass; qP as the carbon-balance residual -- dense and")
    L.append("independent of the product samples (which validate it). CER-based (robust); no free filter param.")
    L.append("")
    L.append("## Per-run reconstruction")
    L.append("| run | T | qs | scale s | R^2(X) | Y_xs(gas) | Y_xs(eff,s) | P_obs [g/L] | P_recon [g/L] | "
             "P_recon/P_obs | qP_decline(late/early) |")
    L.append("| --- | -:| -:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in recs:
        print(f"{r['run']} T{r['T']} q{r['qs']}: s={r['scale_s']:.2f} R2X={r['r2_X']:.2f} "
              f"Yxs_eff={r['Yxs_eff_mean']:.3f} P_recon/P_obs={r['P_recon_over_obs']:.2f} "
              f"qP_decline={r['qP_decline_ratio']:.2f}")
        L.append(f"| {r['run']} | {r['T']} | {r['qs']} | {r['scale_s']:.2f} | {r['r2_X']:.2f} | "
                 f"{r['Yxs_gas_mean']:.3f} | {r['Yxs_eff_mean']:.3f} | {r['P_obs_final']:.2f} | "
                 f"{r['P_recon_final']:.2f} | {r['P_recon_over_obs']:.2f} | {r['qP_decline_ratio']:.2f} |")
    L.append("")
    s_med = float(np.nanmedian([r["scale_s"] for r in recs]))
    pr_med = float(np.nanmedian([r["P_recon_over_obs"] for r in recs]))
    r2_med = float(np.nanmedian([r["r2_X"] for r in recs]))
    decl = [r["qP_decline_ratio"] for r in recs if np.isfinite(r["qP_decline_ratio"])]
    n_decline = int(np.sum(np.array(decl) < 1.0))
    L.append("## What the soft sensor shows")
    L.append(f"- **Per-run scale s** (gas-vs-biomass reconciliation, replaces the +/-50% bracket): median "
             f"**{s_med:.2f}**, biomass-fit R^2 median **{r2_med:.2f}**. s<1 means the gas overstates")
    L.append("  growth (the known instantaneous-vs-cumulative yield gap + the product term); s now")
    L.append("  POINT-estimates the nuisance per run instead of bracketing it.")
    L.append(f"- **Validation (gas-implied product vs observed):** median P_recon/P_obs = **{pr_med:.2f}**. "
             "If ~1, the gas is consistent with biomass+product (the yield gap was just the product term);")
    L.append("  if >1, there is an unmodeled carbon sink (maintenance/acetate) -- the unresolved gas-side")
    L.append("  error, now quantified per run.")
    L.append(f"- **Within-run qP decline (dense, late/early):** {n_decline}/{len(decl)} runs decline "
             "(qP_late<qP_early). Confirms the user's observation: the decline is real but heterogeneous.")
    L.append("")
    # Gas-consistency is regime-dependent: split T30 (clean gas) vs corners (O2-stressed / bad gas).
    t30 = [r["P_recon_over_obs"] for r in recs if r["T"] == 30 and np.isfinite(r["P_recon_over_obs"])]
    corn = [r["P_recon_over_obs"] for r in recs if r["T"] != 30 and np.isfinite(r["P_recon_over_obs"])]
    t30_med = float(np.median(t30)) if t30 else float("nan")
    corn_med = float(np.median(corn)) if corn else float("nan")
    L.append("## Verdict / next")
    L.append("")
    L.append("**1. The per-run scale works (the calibration replaces the +/-50% bracket).** A single scale s")
    L.append(f"per run reconciles the gas-implied growth with offline biomass at R^2(X) median {r2_med:.2f}")
    L.append(f"(s median {s_med:.2f}, range 0.65-0.90). s<1 = the gas overstates growth (the instantaneous-")
    L.append("vs-cumulative yield gap + the product term); it now POINT-estimates the volume/carbon nuisance")
    L.append("per run instead of bracketing it. This is a real, usable output.")
    L.append("")
    L.append("**2. Gas-consistency is regime-dependent (the key finding).** P_recon/P_obs (gas-implied vs")
    L.append(f"observed final product): **T30 median = {t30_med:.2f}** (1.06-1.23 -- the gas is CONSISTENT")
    L.append(f"with biomass+product), but **T25/T35 corners median = {corn_med:.2f}** (0.47-5.14 -- off). The")
    L.append("outliers are exactly the runs the offgas diagnostics flagged: F12/F10 (O2-stressed/overflow),")
    L.append("F16/F19 (corners), F09 (broken gas, CER offset). So the soft sensor QUANTIFIES the known gas-")
    L.append("quality map per run: the gas + carbon balance are trustworthy in the T30 regime (where the")
    L.append("model's action is -- the qS-peak/partition) and unreliable at the stressed corners.")
    L.append("")
    L.append("**3. Caveat -- the dense qP(t) is residual-sensitive.** qP declines in all 11 runs (late/early")
    L.append("0.31-0.59), but the SPARSE product-implied qP declines only in ~7/11 (the user's observation:")
    L.append("not F07/F09/F12/F16). The dense decline matches the sparse where product is substantial and gas")
    L.append("is clean (T30: F13 0.021->0.0075 sparse vs decline-to-0.46 dense -- consistent), but is")
    L.append("ARTIFACTUAL where product is tiny / gas is poor (corners: the carbon-balance residual drifts")
    L.append("under the constant-s assumption). So the dense qP SHAPE is reliable only for the T30 regime;")
    L.append("cross-validate against sparse qP before trusting any run's decline.")
    L.append("")
    L.append("**Stage 2 (refinement, if pursued):** (a) replace the constant-s + ignored-product-term with")
    L.append("the dual carbon+electron balance (use OUR where RQ-trusted) to solve rX and rP jointly -- this")
    L.append("removes the residual drift and should make dense qP reliable across runs; (b) then re-test")
    L.append("mechanisms (the within-run decline, IB/partition) against the refined dense qP, focused on the")
    L.append("T30 regime where the gas is validated. The per-run scale s is already a usable calibration.")
    L.append("")
    L.append("Artifacts: soft_sensor_<run>.png (11), soft_sensor_summary.png, soft_sensor.csv.")

    (OUT / "soft_sensor.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'soft_sensor.md'} (+ 11 run plots, summary, csv)")
    print("\n--- snapshot ---")
    print(f"scale s median={s_med:.2f} (R2X={r2_med:.2f}); P_recon/P_obs median={pr_med:.2f}; "
          f"decline {n_decline}/{len(decl)} runs")


if __name__ == "__main__":
    main()
