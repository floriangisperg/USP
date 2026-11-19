"""Test what alpha (qP(T) suppression) means, via the CER carbon-balance residual at T35.

The adopted model ``M0_aload_qPTexp`` suppresses *measured* product above 30 C:

    qP *= exp(-alpha_T * max(T-30, 0)),   alpha_T ~ 0.096/C   (eta(35) = exp(-0.48) = 0.62)

It was fit to SPARSE product only, so its CER is a zero-parameter prediction from the
carbon + electron balance (``stoich.predict_gas``):

    CER = 1000 * ( r_S/MW_S - r_X/MW_X - r_P_measured/MW_P )

Crucially, ``simulate_run`` applies eta to ``qP`` *before* the balance, so the carbon
that alpha "removed" from measured product is -- by construction of this equation --
routed to CO2. That makes the model's CER at T35 a falsifiable prediction, and the two
physical readings of alpha predict *different* observed CER at T35:

  - DEGRADATION (real heat-shock biology): the lost protein is synthesized, then
    oxidized to CO2. Carbon closes as feed - biomass - measured_product, so the
    model's CER (== CER_deg) should MATCH observed CER at T35.

  - UNDER-RECOVERY (IB assay misses ~38%): the "lost" protein is actually retained as
    unmeasured IB (true product = measured / eta) and is NOT oxidized. Observed CER
    should then be LOWER than the model by the lost-product carbon:
        CER_under = 1000*( r_S/MW_S - r_X/MW_X - (r_P_measured/eta)/MW_P )
    i.e. the model OVERPREDICTS CER at T35.

T25/T30 (eta = 1, no suppression) are the control: there CER_deg == CER_under, and the
CER residual is the model's baseline gas accuracy. If the T35 residual is ~0 (like the
baseline) alpha is consistent with real oxidative degradation; if it is systematically
negative (model overpredicts, and observed sits near CER_under) alpha is partly/mostly
an assay artifact. This directly addresses the HANDOFF's open caveat on alpha.

Outputs: results/alpha_cer/{cer_alpha_test.md, cer_<run>.png, cer_bias_summary.png}.
"""

from __future__ import annotations

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

OUT = Path(__file__).resolve().parent / "results" / "alpha_cer"
OUT.mkdir(parents=True, exist_ok=True)

MW_S = 30.03
MW_X = 26.565
MW_P = LDH_PRODUCT.mw_g_per_cmol  # 22.58

RQ_GOOD = (0.8, 1.2)  # trustworthy RQ window (drops F09/F14 + O2-enriched points)

# Verified FCCD setpoints (T, qS0).
SET = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}
RUN_ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]

# Palette (matches offgas_diagnostics.py).
C_DEG = "#0072B2"     # model CER (degradation hypothesis) -- blue
C_UNDER = "#D55E00"   # under-recovery CER -- orange
C_OBS_GOOD = "#2ca02c"
C_OBS_BAD = "#d62728"
C_LOST = "#9467bd"    # lost-product carbon band -- purple


def _gas_rates(params: dict, run: dict, c_fold: float) -> dict:
    """Model instantaneous rates at the run's gas time grid.

    Mirrors the gas-evaluation block of ``models.simulate_run`` so that CER_deg computed
    here is identical to the model's CER prediction. Also returns eta(T) and the
    retained (post-eta) product rate r_P, which the under-recovery CER needs.
    """
    inp = build_inputs(run)
    rhs = make_rhs("M0_aload_qPTexp", params, inp, c_fold)
    t0, tend = run["t0"], run["tend"]
    st, Y = _rk4(rhs, (run["X0"], 0.0, 0.0), t0, tend, 0.1)
    Xs = Y[0]

    t_gas = np.asarray(run["t_gas"], dtype=float)
    tg_c = np.clip(t_gas, float(inp["t"][0]), float(inp["t"][-1]))
    Xg = np.maximum(np.interp(tg_c, st, Xs), EPS)
    feed_g = np.clip(np.interp(tg_c, inp["t"], inp["feed_rate"]), 0.0, None)
    Vg = np.clip(np.interp(tg_c, inp["t"], inp["V"]), 1e-4, None)
    Tg = np.interp(tg_c, inp["t"], inp["temp"])
    ind_g = np.array([_induced_at(float(t), inp) for t in tg_c])

    qS_g = np.minimum(feed_g * S_IN / (Xg * Vg), QS_CAP)
    Y_xs = params["Y_xs"]
    qP_max = params["qP_max"]
    K_S = params["K_S"]
    alpha_T = params["alpha_T"]
    mu_g = Y_xs * qS_g
    qP_g = ind_g * qP_max * qS_g / (qS_g + K_S)           # Monod (pre-eta)
    eta = np.exp(-alpha_T * np.maximum(Tg - T_REF, 0.0))  # folding efficiency eta(T)
    qP_retained = qP_g * eta                               # post-eta == measured product rate
    r_S = qS_g * Xg
    r_X = mu_g * Xg
    r_P = qP_retained * Xg                                 # retained (measured) product rate
    return {"t": t_gas, "r_S": r_S, "r_X": r_X, "r_P": r_P, "T": Tg, "eta": eta, "ind": ind_g}


def _bias(obs: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(obs) & np.isfinite(pred)
    return float(np.mean(obs[m] - pred[m])) if m.any() else float("nan")


def _frac_closer(obs, a, b, mask):
    """Fraction of masked points where obs is closer to a (degradation) than b (under-recovery)."""
    m = mask & np.isfinite(obs) & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    da = np.abs(obs[m] - a[m])
    db = np.abs(obs[m] - b[m])
    return float(np.mean(da <= db))


def analyze_run(params, run, c_fold) -> dict:
    rates = _gas_rates(params, run, c_fold)
    obs = np.asarray(run["gas"]["CER"], dtype=float)
    rq = np.asarray(run["gas"].get("rq", np.full(len(obs), np.nan)), dtype=float)
    t = rates["t"]
    eta = rates["eta"]
    ind = rates["ind"] > 0.5

    cer_deg, _ = predict_gas(rates["r_S"], rates["r_X"], rates["r_P"])            # degradation
    # Under-recovery: true retained product = measured/eta (assay under-counts by eta).
    r_P_true = rates["r_P"] / np.where(eta > 1e-6, eta, 1.0)
    cer_under, _ = predict_gas(rates["r_S"], rates["r_X"], r_P_true)
    lost_c = cer_deg - cer_under   # CO2 [mmol/L/h] the model attributes to lost product

    good = np.isfinite(rq) & (rq >= RQ_GOOD[0]) & (rq <= RQ_GOOD[1]) & np.isfinite(obs)
    post = good & ind
    pre = good & ~ind

    rec = {
        "run": run["run_id"], "T": SET[run["run_id"]][0], "qs": SET[run["run_id"]][1],
        "t": t, "obs": obs, "rq": rq, "good": good, "ind": ind,
        "cer_deg": cer_deg, "cer_under": cer_under, "lost_c": lost_c,
        "n_post": int(post.sum()), "n_pre": int(pre.sum()),
        "bias_pre": _bias(obs, cer_deg, pre),
        "bias_post_deg": _bias(obs, cer_deg, post),
        "bias_post_under": _bias(obs, cer_under, post),
        "lost_c_mean_post": float(np.nanmean(lost_c[post])) if post.any() else float("nan"),
        "frac_closer_deg_post": _frac_closer(obs, cer_deg, cer_under, post),
        # integrated lost product over induction (g/L protein + its carbon as CO2 mmol/L)
        "lost_protein_gL": float(np.trapz(np.nan_to_num(rates["r_P"] * (1.0 / np.where(eta > 1e-6, eta, 1.0) - 1.0)), t)) if len(t) > 1 else float("nan"),
    }
    return rec


def plot_run(rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = rec["t"]
    # lost-product carbon band between under-recovery and degradation CER.
    lo = np.minimum(rec["cer_deg"], rec["cer_under"])
    hi = np.maximum(rec["cer_deg"], rec["cer_under"])
    ax.fill_between(t, lo, hi, color=C_LOST, alpha=0.15, label="CO2 alpha attributes to lost product")
    ax.plot(t, rec["cer_deg"], color=C_DEG, lw=1.4, label="model CER (degradation: lost protein -> CO2)")
    ax.plot(t, rec["cer_under"], color=C_UNDER, lw=1.2, ls="--", label="CER if under-recovery (lost protein retained)")
    g = rec["good"]
    ax.scatter(t[g], rec["obs"][g], s=8, c=C_OBS_GOOD, label="observed CER (RQ trusted)", zorder=5)
    b = np.isfinite(rec["obs"]) & ~g
    ax.scatter(t[b], rec["obs"][b], s=8, c=C_OBS_BAD, label="observed CER (RQ untrusted)", zorder=4)
    if rec["ind"].any():
        ax.axvline(t[rec["ind"]][0], color="#999", ls="--", lw=0.8)
    title = f"{rec['run']}  T={rec['T']}  qs={rec['qs']}"
    if rec["T"] == 35:
        title += f"  | post-ind bias {rec['bias_post_deg']:+.1f} mmol/L/h (T35 -- alpha test)"
        ax.set_title(title, color=C_OBS_BAD, fontsize=10)
    else:
        ax.set_title(title + f"  | post-ind bias {rec['bias_post_deg']:+.1f}", fontsize=10)
    ax.set_ylabel("CER [mmol/L/h]")
    ax.set_xlabel("Process time [h]")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_summary(recs, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: post-induction CER bias (deg) vs T -- the decisive plot.
    ax = axes[0]
    for r in recs:
        c = C_OBS_BAD if r["T"] == 35 else C_DEG
        ax.scatter(r["T"], r["bias_post_deg"], c=c, s=60, zorder=5)
        ax.annotate(r["run"], (r["T"], r["bias_post_deg"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color="k", lw=0.7)
    # under-recovery reference for T35 (where obs would sit if alpha is an artifact):
    t35 = [r for r in recs if r["T"] == 35]
    if t35:
        ax.scatter([35] * len(t35), [r["bias_post_under"] for r in t35],
                   marker="x", c=C_UNDER, s=70, zorder=6, label="bias vs under-recovery CER (artifact case)")
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("post-induction CER bias (obs - model) [mmol/L/h]")
    ax.set_title("CER residual vs T\n(neg at T35 => alpha over-routes carbon to CO2 => artifact)")
    ax.set_xticks([25, 30, 35])
    ax.legend(fontsize=8)

    # Right: per-stratum means with the lost-product carbon magnitude.
    ax = axes[1]
    strata = [(25, "T25"), (30, "T30"), (35, "T35")]
    xs = np.arange(3)
    bias_deg = [np.nanmean([r["bias_post_deg"] for r in recs if r["T"] == T]) for T, _ in strata]
    bias_under = [np.nanmean([r["bias_post_under"] for r in recs if r["T"] == T]) for T, _ in strata]
    lost = [np.nanmean([r["lost_c_mean_post"] for r in recs if r["T"] == T]) for T, _ in strata]
    w = 0.35
    ax.bar(xs - w / 2, bias_deg, w, color=C_DEG, label="bias vs model CER (degradation)")
    ax.bar(xs + w / 2, bias_under, w, color=C_UNDER, label="bias vs under-recovery CER")
    ax.plot(xs, lost, "o-", color=C_LOST, label="lost-product CO2 magnitude (model)")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([s for _, s in strata])
    ax.set_ylabel("post-induction CER [mmol/L/h]")
    ax.set_title("Per-stratum: observed sits nearer which hypothesis?")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    L = ["# alpha (qP(T)) meaning test -- CER carbon-balance residual at T35", ""]
    L.append("Does the alpha-removed product carbon really appear as CO2 (real oxidative")
    L.append("degradation / heat-shock biology), or is it retained as under-recovered IB")
    L.append("(assay artifact)? The model's CER is a zero-parameter prediction (fit to sparse")
    L.append("product only). If it OVERPREDICTS CER at T35 (residual < 0 vs a ~0 T25/T30")
    L.append("baseline), alpha is partly/mostly an assay artifact. If it MATCHES, alpha is")
    L.append("consistent with real degradation. See module docstring for the two-hypothesis logic.")
    L.append("")

    tr, te = D.split_for_fold("full")
    bundle = D.load_dataset(tr, te)
    c_fold = bundle["c_fold"]
    res = F.fit_model("M0_aload_qPTexp", bundle, seed=42, n_starts=3, max_nfev=300)
    p = res.params
    a = p["alpha_T"]
    print(f"M0_aload_qPTexp full-data fit: cost={res.cost:.4f}  alpha_T={a:.4f}  eta(35)={np.exp(-a*5):.3f}")
    print(f"  Y_xs={p['Y_xs']:.3f}  qP_max={p['qP_max']:.4f}  K_S={p['K_S']:.3f}  a_qS={p['a_qS']:.2f}")
    L.append(f"M0_aload_qPTexp full-data fit: alpha_T = {a:.4f}/C (eta(35) = {np.exp(-a*5):.3f}), "
             f"qP_max = {p['qP_max']:.4f}, Y_xs = {p['Y_xs']:.3f}.")
    L.append("")

    recs = [analyze_run(p, run, c_fold) for run in bundle["train"]]
    by_id = {r["run"]: r for r in recs}
    for r in recs:
        plot_run(r, OUT / f"cer_{r['run']}.png")
    plot_summary(recs, OUT / "cer_bias_summary.png")

    # Per-run table.
    L.append("## Per-run CER residual (post-induction, RQ-trusted)")
    L.append("| run | T | qs | n_post | pre-ind bias | post bias (deg) | post bias (under) | lost-C | frac closer to deg | lost protein [g/L] |")
    L.append("| --- | -:| -:| -:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for run in RUN_ORDER:
        r = by_id[run]
        print(f"{run} T{r['T']} qs{r['qs']}: pre {r['bias_pre']:+.1f}  post(deg) {r['bias_post_deg']:+.1f}  "
              f"post(under) {r['bias_post_under']:+.1f}  lostC {r['lost_c_mean_post']:.1f}  "
              f"closer-deg {r['frac_closer_deg_post']:.2f}")
        L.append(f"| {run} | {r['T']} | {r['qs']} | {r['n_post']} | {r['bias_pre']:+.1f} | "
                 f"{r['bias_post_deg']:+.1f} | {r['bias_post_under']:+.1f} | {r['lost_c_mean_post']:.1f} | "
                 f"{r['frac_closer_deg_post']:.2f} | {r['lost_protein_gL']:.2f} |")
    L.append("")
    L.append("- bias = observed - predicted CER [mmol/L/h]. Negative => model OVERPREDICTS CER.")
    L.append("- post bias (deg): vs the model CER (degradation hypothesis; lost protein -> CO2).")
    L.append("- post bias (under): vs the under-recovery CER (lost protein retained as unmeasured IB).")
    L.append("- lost-C: mean CO2 [mmol/L/h] the model attributes to lost product over induction.")
    L.append("- frac closer to deg: fraction of post-ind points where obs is nearer the degradation CER.")
    L.append("")

    # Stratum summary.
    L.append("## Stratum summary (mean over runs)")
    L.append("| stratum | pre-ind bias | post bias (deg) | post bias (under) | lost-C | frac closer to deg |")
    L.append("| --- | ---:| ---:| ---:| ---:| ---:|")
    print("\nstratum summary:")
    for T, name in [(25, "T25 (eta=1)"), (30, "T30 (eta=1)"), (35, "T35 (eta<1, alpha active)")]:
        sub = [r for r in recs if r["T"] == T]
        bd = np.nanmean([r["bias_post_deg"] for r in sub])
        bu = np.nanmean([r["bias_post_under"] for r in sub])
        bp = np.nanmean([r["bias_pre"] for r in sub])
        lc = np.nanmean([r["lost_c_mean_post"] for r in sub])
        fc = np.nanmean([r["frac_closer_deg_post"] for r in sub])
        print(f"  {name}: pre {bp:+.1f}  post(deg) {bd:+.1f}  post(under) {bu:+.1f}  lostC {lc:.1f}  closer-deg {fc:.2f}")
        L.append(f"| {name} | {bp:+.1f} | {bd:+.1f} | {bu:+.1f} | {lc:.1f} | {fc:.2f} |")
    L.append("")

    # Verdict.
    t35 = [r for r in recs if r["T"] == 35]
    base = [r for r in recs if r["T"] != 35]
    t35_deg = np.nanmean([r["bias_post_deg"] for r in t35])
    t35_under = np.nanmean([r["bias_post_under"] for r in t35])
    base_deg = np.nanmean([r["bias_post_deg"] for r in base])
    t35_frac = np.nanmean([r["frac_closer_deg_post"] for r in t35])
    L.append("## Verdict")
    L.append(f"- T25/T30 baseline post-ind CER bias (deg): **{base_deg:+.1f} mmol/L/h** (model's gas accuracy where alpha is inactive).")
    L.append(f"- T35 post-ind CER bias vs degradation CER: **{t35_deg:+.1f} mmol/L/h**.")
    L.append(f"- T35 post-ind CER bias vs under-recovery CER: **{t35_under:+.1f} mmol/L/h**.")
    L.append(f"- T35 points closer to degradation than under-recovery: **{t35_frac*100:.0f}%**.")
    L.append("")
    if t35_deg < base_deg - 5 and t35_frac < 0.5:
        L.append("The model OVERPREDICTS CER at T35 and observed sits nearer the under-recovery curve:")
        L.append("**alpha is partly/mostly an assay-under-recovery artifact**, not clean oxidative")
        L.append("degradation -- the alpha-removed product carbon is NOT appearing as CO2.")
    elif abs(t35_deg - base_deg) < 5 or t35_frac >= 0.5:
        L.append("The model's CER at T35 matches observation about as well as the T25/T30 baseline,")
        L.append("and observed sits at least half the time nearer the degradation curve:")
        L.append("**alpha is consistent with real oxidative degradation** (heat-shock biology) -- the")
        L.append("lost product carbon does appear as CO2. The under-recovery artifact is not supported")
        L.append("(or is smaller than the gas noise).")
    else:
        L.append("The T35 CER residual is ambiguous at the precision of the gas (effect size comparable")
        L.append("to noise): the data cannot cleanly separate degradation from under-recovery. Report alpha")
        L.append("as 'what the measured data require' with the biological interpretation unresolved.")
    L.append("")
    L.append("Artifacts: per-run cer_<run>.png, cer_bias_summary.png.")

    (OUT / "cer_alpha_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'cer_alpha_test.md'} (+ {len(recs)} run plots, summary)")


if __name__ == "__main__":
    main()
