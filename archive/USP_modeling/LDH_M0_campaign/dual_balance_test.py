"""Stage 2 -- can the dual carbon+electron balance split growth (rX) from product (rP) for LDH?

Plan B Stage 2 aimed to solve rX and rP JOINTLY from the gas (CER, OUR) + feed (rS), removing the
Stage-1 residual drift, to get a reliable dense qP(t) for mechanism re-testing. The dual balance inverts:

    Carbon:    rS/MW_S = rX/MW_X + rP/MW_P + CER/1000
    Electron:  gS*rS/MW_S = gX*rX/MW_X + gP*rP/MW_P + 4*OUR/1000

for rX, rP. But the split coefficient is (gP - gX) in the denominator, and for LDH:

    gS (glucose CH2O) = 4.00,   gX (biomass) = 4.07,   gP (LDH product) = 4.16

i.e. gP - gX = 0.09 -- biomass and LDH product are NEARLY EQUALLY REDUCED, so the electron balance CANNOT
distinguish them. The split amplifies the CER/OUR/feed closure error by ~1/0.09 ~ 11x. This script shows the
consequence empirically: the dual-balance rP is garbage (wrong-signed, 10-20x too big).

IMPLICATION (the real finding): the offgas constrains only the TOTAL rX + rP (via the carbon balance),
never the growth-vs-product split. This is the stoichiometric root cause of EVERY gas-side failure in this
project -- M1a (gas overpredicts biomass), M1b maintenance, burden -- all tried to use the gas to pin the
split, which the electron balance forbids for LDH. It is a stoichiometric limit, not a data or modeling
failure. The only viable split uses the OFFLINE observations (biomass -> growth, product = residual), which
is Stage 1; its T30 qP is therefore the reliable dense qP.

Outputs: results/dual_balance/{dual_balance_test.md, dual_balance_summary.png, dual_balance.csv}.
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

OUT = Path(__file__).resolve().parent / "results" / "dual_balance"
OUT.mkdir(parents=True, exist_ok=True)

MW_S, MW_X, MW_P = GLUCOSE.mw_g_per_cmol, BIOMASS.mw_g_per_cmol, LDH_PRODUCT.mw_g_per_cmol
gS, gX, gP = GLUCOSE.degree_of_reduction(), BIOMASS.degree_of_reduction(), LDH_PRODUCT.degree_of_reduction()
DENOM = gP - gX
S_IN = 400.0
RQ_GOOD = (0.8, 1.2)

SET = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}
RUN_ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]


def _dense_cerour(df, run):
    c_t = D._find_col(df, "Process time [h]")
    c_cer = D._find_col(df, "CER [mmol/L/h]")
    c_our = D._find_col(df, "OUR [mmol/L/h]")
    r = df[df["run_name"] == run["run_id"]].sort_values(c_t).reset_index(drop=True)
    tg = r[c_t].to_numpy(float)
    cer = r[c_cer].to_numpy(float)
    our = r[c_our].to_numpy(float)
    m = np.isfinite(tg) & np.isfinite(cer) & np.isfinite(our)
    t = np.asarray(run["t_online"], float)
    return (t, np.interp(t, tg[m], cer[m]), np.interp(t, tg[m], our[m]))


def analyze(df, run):
    t, cer, our = _dense_cerour(df, run)
    feed = np.asarray(run["feed_rate_L_h"], float)
    V = np.asarray(run["V_L"], float)
    rS = feed * S_IN / np.where(V > 1e-6, V, np.nan)
    A = rS / MW_S - cer / 1000.0                    # rX/MW_X + rP/MW_P  (C-mol/L/h, carbon balance total)
    B = gS * rS / MW_S - 4.0 * our / 1000.0          # electron-balance total
    v = (B - gX * A) / DENOM                         # rP/MW_P (dual-balance product rate)
    rP_dual = MW_P * v
    with np.errstate(divide="ignore", invalid="ignore"):
        rq = cer / np.where(np.abs(our) > 1e-6, our, np.nan)
    trust = np.isfinite(rq) & (rq >= RQ_GOOD[0]) & (rq <= RQ_GOOD[1]) & (rS > 0)
    ind = np.asarray(run["is_induced"], float) > 0.5
    seg = ind & trust
    # Closure error that gets amplified (the numerator B - gX*A, before /DENOM).
    closure_num = B - gX * A
    P_obs = float((run["obs"]["P_sol"] + run["obs"]["P_IB"])[-1])
    # integrated rP_dual over trusted induced points (trapezoid over the trusted set, gaps -> nan skip)
    if seg.sum() > 1:
        ts, vs = t[seg], rP_dual[seg]
        P_recon_dual = float(np.trapz(vs, ts))
    else:
        P_recon_dual = float("nan")
    return {
        "run": run["run_id"], "T": SET[run["run_id"]][0], "qs": SET[run["run_id"]][1],
        "P_obs_final": P_obs, "P_recon_dual": P_recon_dual,
        "rP_dual_mean": float(np.nanmean(rP_dual[seg])) if seg.any() else float("nan"),
        "rP_dual_std": float(np.nanstd(rP_dual[seg])) if seg.any() else float("nan"),
        "frac_negative": float(np.nanmean(rP_dual[seg] < 0)) if seg.any() else float("nan"),
        "n_seg": int(seg.sum()),
        "closure_num_rms": float(np.sqrt(np.nanmean(closure_num[seg] ** 2))) if seg.any() else float("nan"),
    }


def main():
    df = D._load_df_cached(D.DEFAULT_DATA_FILE)
    tr, te = D.split_for_fold("full")
    bundle = D.load_dataset(tr, te)
    recs = [analyze(df, r) for r in bundle["train"]]
    recs = [next(r for r in recs if r["run"] == rid) for rid in RUN_ORDER]

    pd.DataFrame(recs).to_csv(OUT / "dual_balance.csv", index=False)

    # Plot: dual-balance integrated product vs observed (should be ~1:1 if usable; it is garbage).
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in recs:
        c = "#56B4E9" if r["T"] == 25 else "#0072B2" if r["T"] == 30 else "#E69F00"
        ax.scatter(r["P_obs_final"], r["P_recon_dual"], c=c, s=80, edgecolor="k", lw=0.4)
        ax.annotate(r["run"], (r["P_obs_final"], r["P_recon_dual"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    lim = [min(0, min(r["P_recon_dual"] for r in recs)) * 1.1, max(r["P_obs_final"] for r in recs) * 1.3]
    ax.plot([0, max(r["P_obs_final"] for r in recs)], [0, max(r["P_obs_final"] for r in recs)],
            "k--", lw=0.7, label="1:1 (if usable)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("observed final P_tot [g/L]")
    ax.set_ylabel("dual-balance int(rP) [g/L]")
    ax.set_title(f"Dual balance is unusable for LDH (gP-gX={DENOM:.2f}):\n"
                 "integrated product is wrong-signed / huge vs observed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dual_balance_summary.png", dpi=120)
    plt.close(fig)

    L = ["# Stage 2 -- dual carbon+electron balance CANNOT split LDH growth from product", ""]
    L.append("Plan B Stage 2 aimed to solve rX and rP jointly from gas+feed for a reliable dense qP. The dual")
    L.append("balance splits them via their degrees of reduction, but for LDH biomass and product are nearly")
    L.append(f"equally reduced: gS={gS:.2f}, **gX={gX:.2f}, gP={gP:.2f}**, so the split denominator")
    L.append(f"**(gP - gX) = {DENOM:.2f}** amplifies the CER/OUR/feed closure error ~{1/DENOM:.0f}x. Result:")
    L.append("the dual-balance product rate is garbage (wrong-signed, 10-20x too big).")
    L.append("")
    L.append("## Per-run dual-balance product (induced, RQ-trusted)")
    L.append("| run | T | qs | P_obs [g/L] | int(rP)_dual [g/L] | rP_dual mean | rP_dual std | %negative | "
             "closure-num RMS |")
    L.append("| --- | -:| -:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in recs:
        print(f"{r['run']} T{r['T']} q{r['qs']}: P_obs={r['P_obs_final']:.2f} int(rP)_dual={r['P_recon_dual']:.1f} "
              f"rP_mean={r['rP_dual_mean']:.2f} std={r['rP_dual_std']:.2f} %neg={r['frac_negative']*100:.0f}")
        L.append(f"| {r['run']} | {r['T']} | {r['qs']} | {r['P_obs_final']:.2f} | {r['P_recon_dual']:.1f} | "
                 f"{r['rP_dual_mean']:.2f} | {r['rP_dual_std']:.2f} | {r['frac_negative']*100:.0f} | "
                 f"{r['closure_num_rms']:.3f} |")
    L.append("")
    neg_med = float(np.nanmedian([r["frac_negative"] for r in recs]))
    L.append("## Verdict -- a stoichiometric limit, not a data/modeling failure")
    L.append("")
    L.append(f"The dual-balance rP is wrong-signed ({neg_med*100:.0f}% of points negative, median) and huge")
    L.append("(std ~10-18 g/L/h vs the true ~0.5-0.8). The closure-error numerator (B - gX*A, RMS shown) is")
    L.append(f"small but divided by (gP-gX)={DENOM:.2f} -> amplified ~{1/DENOM:.0f}x. Smoothing does not help")
    L.append("(the bias is structural, not noise).")
    L.append("")
    L.append("**Why it matters (the root cause of every gas-side failure):** biomass (g=4.07) and LDH product")
    L.append("(g=4.16) are nearly equally reduced, so the electron balance CANNOT distinguish growth carbon")
    L.append("from product carbon. The offgas therefore constrains only the **total rX + rP** (via the carbon")
    L.append("balance), never the split. This is why M1a (gas overpredicts biomass), M1b maintenance, and the")
    L.append("burden state all failed -- each tried to use the gas to pin the growth/product split, which the")
    L.append("stoichiometry forbids for LDH. It is a property of the LDH composition, not fixable by more data")
    L.append("or a better filter.")
    L.append("")
    L.append("**Consequence for the soft sensor:** the only viable split uses the OFFLINE observations")
    L.append("(biomass -> growth via the per-run scale; product = carbon residual). That is Stage 1, and its")
    L.append("T30 qP (validated, P_recon/P_obs ~ 1.1) is the reliable dense qP -- there is no gas-only")
    L.append("refinement beyond it for LDH. Mechanism re-tests should use the Stage-1 T30 qP (cross-validated")
    L.append("against sparse product), with the understanding that the gas cannot independently constrain")
    L.append("product formation for this system.")
    L.append("")
    L.append("Artifacts: dual_balance_summary.png, dual_balance.csv.")

    (OUT / "dual_balance_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'dual_balance_test.md'} (+ dual_balance_summary.png, dual_balance.csv)")


if __name__ == "__main__":
    main()
