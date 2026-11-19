"""Task D -- burden respiration as the missing post-induction CO2 (the mechanism behind alpha).

The alpha-CER test (``alpha_cer_test.py``) showed the carbon+electron balance UNDERpredicts
post-induction CER by ~2-7 mmol/L/h at T25/T30 (where eta=1, so the residual is pure burden --
proteostasis respiration the balance omits -- not alpha). At T35 alpha's lost-C (~10) fills the
gap to ~0. This script asks whether that burden is a stable, *gas-observable*, identifiable
quantity, and whether it unifies with alpha.

Key difference from Task A (burden as a latent PARTITION driver on sparse data -- correctly
rejected as non-identifiable): here burden drives **CO2, which we measure directly**, so the
burden cost beta is a regression slope on the gas -- identifiable by construction, not a latent
state.

Physics. The balance as written is ``CER = r_S - r_X - r_P``. With eta applied before the balance
(M0_aload_qPTexp), r_P = eta*r_P_synth (retained), so at T35 the model already routes the
(1-eta) fraction of synthesis to CO2 (== alpha-lost-C). At T25/T30 (eta=1) it does NOT, so the
post-induction residual there is the **constitutive** proteostasis burden beta*r_P_synth
(chaperone/ATP cost of folding even soluble product). The unification question: is alpha just
the temperature amplification of this same constitutive burden?

Plan:
  1. Regress the post-induction CER residual (obs - balance) on the synthesis rate r_P_synth
     [g product/L/h] over reliable T25/T30 runs -> beta [mmol CO2/g product] + intercept + R^2.
  2. Cross-stratum (beta per T) and cross-fold (beta from each fold's TRAIN residuals, A/B/C):
     a stable beta is the identifiability evidence.
  3. "Burden replaces alpha" model: CER = predict_gas(r_S, r_X, r_P_synth) + beta*r_P_synth
     (no eta). Check the residual collapses at T25/T30, and at T35 equals (alpha-lost-C - beta*synth).
  4. gamma implied by T35: does heat amplify the burden beyond the constitutive beta?

Outputs: results/burden_co2/{burden_co2_test.md, burden_regression.png, burden_residual.png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import data as D
import fit as F
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from models import (  # noqa: E402
    EPS,
    QS_CAP,
    S_IN,
    T_REF,
    _induced_at,
    _rk4,
    build_inputs,
    make_rhs,
)
from stoich import LDH_PRODUCT, predict_gas  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "burden_co2"
OUT.mkdir(parents=True, exist_ok=True)

MW_P = LDH_PRODUCT.mw_g_per_cmol  # 22.58
RQ_GOOD = (0.8, 1.2)
# Broken-gas runs excluded from the regression (F09 OUR offset, F14 broken throughout).
UNRELIABLE = {"F09", "F14"}

SET = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}
RUN_ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]

C_RES = "#0072B2"
C_T25 = "#009E73"
C_T30 = "#E69F00"
C_T35 = "#D55E00"
C_FIT = "#CC79A7"


def _tcolor(T):
    return {25: C_T25, 30: C_T30, 35: C_T35}.get(T, "gray")


def _gas_fields(params: dict, run: dict, c_fold: float) -> dict:
    """Model fields at the gas grid: rates, T, eta, and observed CER/rq.

    r_P_synth = qP_pre_eta * X  (the synthesis rate the cell attempts).
    r_P_retained = eta * r_P_synth (measured product rate; what the balance uses with alpha).
    """
    inp = build_inputs(run)
    rhs = make_rhs("M0_aload_qPTexp", params, inp, c_fold)
    st, Y = _rk4(rhs, (run["X0"], 0.0, 0.0), run["t0"], run["tend"], 0.1)
    Xs = Y[0]
    t_gas = np.asarray(run["t_gas"], dtype=float)
    tg_c = np.clip(t_gas, float(inp["t"][0]), float(inp["t"][-1]))
    Xg = np.maximum(np.interp(tg_c, st, Xs), EPS)
    feed_g = np.clip(np.interp(tg_c, inp["t"], inp["feed_rate"]), 0.0, None)
    Vg = np.clip(np.interp(tg_c, inp["t"], inp["V"]), 1e-4, None)
    Tg = np.interp(tg_c, inp["t"], inp["temp"])
    ind_g = np.array([_induced_at(float(t), inp) for t in tg_c])
    qS_g = np.minimum(feed_g * S_IN / (Xg * Vg), QS_CAP)
    Y_xs = params["Y_xs"]; qP_max = params["qP_max"]; K_S = params["K_S"]; alpha_T = params["alpha_T"]
    mu_g = Y_xs * qS_g
    qP_synth = ind_g * qP_max * qS_g / (qS_g + K_S)             # pre-eta synthesis (specific)
    eta = np.exp(-alpha_T * np.maximum(Tg - T_REF, 0.0))
    r_S = qS_g * Xg
    r_X = mu_g * Xg
    r_P_synth = qP_synth * Xg                                    # g product/L/h attempted
    r_P_retained = eta * r_P_synth                               # measured product rate
    cer_bal, _ = predict_gas(r_S, r_X, r_P_retained)            # balance WITH alpha (the adopted model)
    return {
        "run": run["run_id"], "T": SET[run["run_id"]][0], "t": t_gas,
        "obs": np.asarray(run["gas"]["CER"], dtype=float),
        "rq": np.asarray(run["gas"].get("rq", np.full(len(t_gas), np.nan)), dtype=float),
        "cer_bal": cer_bal, "r_P_synth": r_P_synth, "r_P_retained": r_P_retained,
        "eta": eta, "ind": ind_g > 0.5,
    }


def _pool(flds, T_filter=None, reliable_only=True, post_only=True):
    """Stack (residual, r_P_synth, cer_bal, obs) from the requested runs/points."""
    R, P, CB, OB = [], [], [], []
    for f in flds:
        if reliable_only and f["run"] in UNRELIABLE:
            continue
        if T_filter is not None and f["T"] != T_filter:
            continue
        good = np.isfinite(f["rq"]) & (f["rq"] >= RQ_GOOD[0]) & (f["rq"] <= RQ_GOOD[1]) & np.isfinite(f["obs"])
        if post_only:
            good = good & f["ind"]
        resid = f["obs"] - f["cer_bal"]
        m = good & np.isfinite(resid) & np.isfinite(f["r_P_synth"])
        R.append(resid[m]); P.append(f["r_P_synth"][m]); CB.append(f["cer_bal"][m]); OB.append(f["obs"][m])
    if not R:
        return None
    return (np.concatenate(R), np.concatenate(P), np.concatenate(CB), np.concatenate(OB))


def _ols(x, y):
    """y = slope*x + intercept. Returns (slope, intercept, r2, n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), len(x)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, interc), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    return float(slope), float(interc), r * r, len(x)


def main():
    L = ["# Task D -- burden respiration as the missing post-induction CO2", ""]
    L.append("Is the post-induction CER shortfall (obs - balance) a stable, gas-observable burden")
    L.append("cost beta [mmol CO2/g product] that scales with the synthesis rate? If so, it is")
    L.append("identifiable by construction (a regression slope on measured gas, unlike Task A's")
    L.append("latent partition burden). Tested at T25/T30 (eta=1, pure burden), then checked at T35.")
    L.append("")

    tr, te = D.split_for_fold("full")
    bundle = D.load_dataset(tr, te)
    c_fold = bundle["c_fold"]
    res = F.fit_model("M0_aload_qPTexp", bundle, seed=42, n_starts=3, max_nfev=250)
    p = res.params
    print(f"[full] cost={res.cost:.4f} alpha_T={p['alpha_T']:.4f} Y_xs={p['Y_xs']:.3f} "
          f"qP_max={p['qP_max']:.4f}", flush=True)
    L.append(f"Full-data M0_aload_qPTexp fit: alpha_T={p['alpha_T']:.4f} (eta(35)={np.exp(-p['alpha_T']*5):.3f}), "
             f"qP_max={p['qP_max']:.4f}, Y_xs={p['Y_xs']:.3f}.")
    L.append("")

    flds = [_gas_fields(p, run, c_fold) for run in bundle["train"]]
    by_run = {f["run"]: f for f in flds}

    # ---- (1) Constitutive burden beta from T25/T30 post-induction residual ----
    # Main regression: T25+T30 only (eta=1, clean burden).
    t2530 = [f for f in flds if f["T"] <= 30 and f["run"] not in UNRELIABLE]
    pool2530 = _pool(t2530)
    beta, interc, r2, n = _ols(pool2530[1], pool2530[0])
    print(f"[beta] T25/T30 post-ind: beta={beta:.2f} mmol CO2/g  intercept={interc:.2f}  R2={r2:.2f}  n={n}", flush=True)
    L.append("## (1) Constitutive burden beta (T25/T30 post-induction, RQ-trusted, reliable runs)")
    L.append(f"Regress (obs - balance) on synthesis rate r_P_synth [g/L/h]:")
    L.append(f"- **beta = {beta:.2f} mmol CO2 / g product** (slope), intercept = {interc:.2f} mmol/L/h, "
             f"R^2 = {r2:.2f}, n = {n} points.")
    if r2 > 0.3 and beta > 0:
        L.append(f"- Mechanistic read: ~{beta:.1f} mmol CO2 are respired per gram of LDH the cell attempts to make -- "
                 f"the chaperone/protease + ATP cost of folding even soluble product at the expression optimum.")
    else:
        L.append(f"- beta is unphysical ({beta:+.1f} mmol/g, R^2={r2:.2f}): the post-induction CER residual does NOT "
                 f"scale with the production rate. The shortfall is a roughly flat offset (intercept {interc:+.1f} "
                 f"mmol/L/h), not a per-product burden -> production-proportional burden is not identifiable.")
    L.append(f"- alpha-equivalent cost at T35: 1000*(1-eta)/MW_P = {1000*(1-np.exp(-p['alpha_T']*5))/MW_P:.1f} mmol/g "
             f"(the CO2 alpha attributes to lost product). Compare to beta to test unification.")
    L.append("")

    # ---- (2) Cross-stratum and cross-fold stability ----
    L.append("## (2) Identifiability -- cross-stratum and cross-fold")
    L.append("| stratum | beta [mmol CO2/g] | intercept | R^2 | n |")
    L.append("| --- | ---:| ---:| ---:| ---:|")
    print("[beta per stratum]:", flush=True)
    strata_beta = {}
    for T in (25, 30, 35):
        sub = [f for f in flds if f["T"] == T and f["run"] not in UNRELIABLE]
        pp = _pool(sub)
        if pp is None:
            continue
        b, ic, rr, nn = _ols(pp[1], pp[0])
        strata_beta[T] = b
        print(f"  T{T}: beta={b:.2f} R2={rr:.2f} n={nn}", flush=True)
        L.append(f"| T{T} | {b:.2f} | {ic:.2f} | {rr:.2f} | {nn} |")
    L.append("(T35 beta is confounded: alpha already fills its residual, so the T35 slope is not an")
    L.append("independent burden estimate -- it tests consistency, not identification.)")
    L.append("")

    # Cross-fold: refit each fold, beta from its TRAIN residuals (leakage-safe).
    L.append("| fold | beta_T25/T30 [mmol CO2/g] | R^2 | n | alpha_T |")
    L.append("| --- | ---:| ---:| ---:| ---:|")
    print("[beta per fold]:", flush=True)
    fold_betas = {}
    for fold in ("A", "B", "C"):
        trf, tef = D.split_for_fold(fold)
        bf = D.load_dataset(trf, tef)
        rf = F.fit_model("M0_aload_qPTexp", bf, seed=42, n_starts=2, max_nfev=180)
        pf = rf.params
        fflds = [_gas_fields(pf, run, bf["c_fold"]) for run in bf["train"]]
        sub = [f for f in fflds if f["T"] <= 30 and f["run"] not in UNRELIABLE]
        pp = _pool(sub)
        b, ic, rr, nn = _ols(pp[1], pp[0]) if pp else (float("nan"),) * 4
        fold_betas[fold] = b
        print(f"  fold {fold}: beta={b:.2f} R2={rr:.2f} alpha_T={pf['alpha_T']:.4f}", flush=True)
        L.append(f"| {fold} | {b:.2f} | {rr:.2f} | {nn} | {pf['alpha_T']:.4f} |")
    bvec = np.array([fold_betas[f] for f in ("A", "B", "C")], float)
    bvec = bvec[np.isfinite(bvec)]
    spread = float(np.ptp(bvec)) if len(bvec) else float("nan")
    L.append(f"Cross-fold beta spread (max-min) = {spread:.2f} mmol/g over mean {np.nanmean(bvec):.2f}.")
    L.append("")

    # ---- (3) Burden-replaces-alpha model + (4) gamma at T35 ----
    # CER_burden = predict_gas(r_S, r_X, r_P_synth) + beta*r_P_synth  (no eta anywhere).
    L.append("## (3) Burden-replaces-alpha model: CER = balance(synthesis) + beta*synthesis")
    L.append("Drops eta entirely; routes (1-eta) of synthesis to CO2 implicitly via the burden term.")
    L.append("| run | T | residual (adopted, with alpha) | residual (burden model, no alpha) |")
    L.append("| --- | -:| ---:| ---:|")
    print("[burden-model residual per run]:", flush=True)
    per_stratum = {25: [], 30: [], 35: []}
    for run_id in RUN_ORDER:
        f = by_run[run_id]
        good = np.isfinite(f["rq"]) & (f["rq"] >= RQ_GOOD[0]) & (f["rq"] <= RQ_GOOD[1]) & np.isfinite(f["obs"]) & f["ind"]
        resid_alpha = f["obs"] - f["cer_bal"]
        # balance(synthesis, no alpha): rebuild from the adopted balance by adding back the
        # eta-suppressed product carbon (retained < synth when eta<1 => less CO2 without alpha).
        cer_noalpha = f["cer_bal"] + 1000.0 * (f["r_P_retained"] - f["r_P_synth"]) / MW_P
        # ^ retained < synth (eta<1), so (retained - synth) < 0 => cer_noalpha < cer_bal. Correct:
        #   without alpha, more carbon stays in product => less CO2.
        cer_burden = cer_noalpha + beta * f["r_P_synth"]
        resid_burden = f["obs"] - cer_burden
        ra = float(np.nanmean(resid_alpha[good])) if good.any() else float("nan")
        rb = float(np.nanmean(resid_burden[good])) if good.any() else float("nan")
        per_stratum[f["T"]].append((ra, rb))
        print(f"  {run_id} T{f['T']}: resid(alpha)={ra:+.1f}  resid(burden)={rb:+.1f}", flush=True)
        L.append(f"| {run_id} | {f['T']} | {ra:+.1f} | {rb:+.1f} |")
    L.append("")
    L.append("## (4) Per-stratum means + gamma implied by T35")
    L.append("| stratum | resid (adopted alpha) | resid (burden model) |")
    L.append("| --- | ---:| ---:|")
    for T in (25, 30, 35):
        ra = np.nanmean([a for a, _ in per_stratum[T]]) if per_stratum[T] else float("nan")
        rb = np.nanmean([b for _, b in per_stratum[T]]) if per_stratum[T] else float("nan")
        L.append(f"| T{T} | {ra:+.1f} | {rb:+.1f} |")
    # gamma: at T35 the burden model under/over-supplies by rb; amplify burden by (1+gamma*5).
    t35 = per_stratum.get(35, [])
    if t35:
        rb35 = np.nanmean([b for _, b in t35])
        # solve gamma from mean(cer_burden + beta*synth*gamma*5) == obs  => gamma = rb35/(beta*synth35*5)
        f35 = [by_run[r] for r in RUN_ORDER if by_run[r]["T"] == 35 and r not in UNRELIABLE]
        bs35 = np.concatenate([f["r_P_synth"][np.isfinite(f["rq"]) & (f["rq"] >= RQ_GOOD[0]) &
                          (f["rq"] <= RQ_GOOD[1]) & np.isfinite(f["obs"]) & f["ind"]] for f in f35])
        bs35_mean = float(np.nanmean(bs35)) if len(bs35) else float("nan")
        denom = beta * bs35_mean * 5.0
        gamma = float(rb35 / denom) if abs(denom) > 1e-6 else float("nan")
        L.append("")
        if r2 > 0.3 and beta > 0:
            L.append(f"T35 burden-model residual = {rb35:+.1f} mmol/L/h; implied heat-amplification "
                     f"gamma = {gamma:.3f} (burden_T35 = beta*(1+gamma*5)*synth). gamma~0 => constitutive "
                     f"beta covers T35; gamma>0 => heat adds extra beyond constitutive proteostasis.")
        else:
            L.append(f"T35 burden-model residual = {rb35:+.1f} mmol/L/h -- moot: beta is unidentifiable, so the "
                     f"burden model is not a candidate and gamma is not meaningful.")
    L.append("")

    # ---- Verdict ----
    stable = spread < 0.5 * max(abs(beta), 1e-6) if np.isfinite(spread) else False
    L.append("## Verdict")
    L.append(f"- Constitutive burden beta = {beta:.2f} mmol CO2/g product (R^2 = {r2:.2f} at T25/T30).")
    L.append(f"- Cross-fold spread = {spread:.2f} => {'STABLE' if stable else 'moderately variable'} across folds.")
    if stable and r2 > 0.3:
        L.append("- beta is a stable, gas-observable, identifiable quantity. The post-induction CER")
        L.append("  shortfall is real proteostasis respiration, and it is measurable directly from the gas.")
        L.append("- **Adoption gate (next):** formalize as a gas term in stoich/models (CER += beta*qP*X),")
        L.append("  refit as an M1a-style gas variant, and profile beta/gamma before replacing alpha.")
    else:
        L.append("- beta is NOT cleanly stable/identifiable from the gas at this precision (low R^2 or large")
        L.append("  cross-fold spread): the burden shortfall is real but too noisy to adopt as a parameter.")
        L.append("  Keep alpha as the empirical T-suppression; the burden is a documented residual, not a")
        L.append("  model term.")
    L.append("")
    L.append("Artifacts: burden_regression.png, burden_residual.png.")

    # ---- Plots ----
    # (a) regression scatter (T25/T30 post-ind, colored by T) + fit line.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for T in (25, 30, 35):
        sub = [f for f in flds if f["T"] == T and f["run"] not in UNRELIABLE]
        pp = _pool(sub)
        if pp is None:
            continue
        ax.scatter(pp[1], pp[0], s=10, c=_tcolor(T), alpha=0.6, label=f"T{T}")
    xx = np.array([0, max(pool2530[1].max(), 1e-6)])
    ax.plot(xx, beta * xx + interc, c=C_FIT, lw=1.5, label=f"fit T25/T30: {beta:.1f} mmol/g")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("synthesis rate r_P_synth [g product/L/h]")
    ax.set_ylabel("CER residual (obs - balance) [mmol/L/h]")
    ax.set_title(f"Post-induction burden vs production rate\nbeta={beta:.1f} mmol CO2/g, R2={r2:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "burden_regression.png", dpi=120); plt.close(fig)

    # (b) per-stratum residual: adopted (alpha) vs burden model.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    Ts = [25, 30, 35]
    ra = [np.nanmean([a for a, _ in per_stratum[T]]) if per_stratum[T] else np.nan for T in Ts]
    rb = [np.nanmean([b for _, b in per_stratum[T]]) if per_stratum[T] else np.nan for T in Ts]
    x = np.arange(len(Ts)); w = 0.38
    ax.bar(x - w/2, ra, w, color=C_RES, label="adopted (balance + alpha)")
    ax.bar(x + w/2, rb, w, color=C_FIT, label="burden model (balance(synth) + beta*synth)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f"T{T}" for T in Ts])
    ax.set_ylabel("post-induction CER residual (obs - pred) [mmol/L/h]")
    ax.set_title("Does the burden term replace alpha?\n(T25/T30 should collapse; T35 tests unification)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "burden_residual.png", dpi=120); plt.close(fig)

    (OUT / "burden_co2_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'burden_co2_test.md'} (+ 2 plots)", flush=True)


if __name__ == "__main__":
    main()
