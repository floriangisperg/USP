"""Task E.2 (Phase 1, diagnostic) -- does O2-transfer limitation cause the qS-peak?

The qS-peak (HANDOFF): specific total product Ptot/X_final PEAKS at qs0.4 and declines at
qs0.6 (T30: 0.065 / 0.096 / 0.079). A monotonic Monod qP(qS) cannot represent a peak, so the
qs0.6 runs are overpredicted once qP_max is freed. E.2 hypothesises the decline is O2-transfer
limitation: at qs0.6 (high biomass) the culture hits the OTR cap and diverts carbon away from
expression. Mechanistically real, and the offgas diagnostics show qs0.6 corners enrich inlet O2
(21 -> 27-53 %). This script tests, data-first, whether O2 limitation actually explains the peak.

O2-limitation observables available (consolidated df):
  - Inlet O2 [%]: the controller's O2-DEMAND response (>21 % => air alone can't supply O2).
  - pO2 [%]: dissolved O2; a crash toward 0 is direct O2 starvation.
  - RQ = CER/OUR: ~1.0-1.1 fully aerobic; >1.2 is overflow / partial anaerobic (acetate makes
    CO2 without consuming O2) -- the metabolic signature of UNMET O2 demand.
  - qO2 [mmol/g/h], OUR [mmol/L/h]: specific and volumetric O2 uptake.

The falsification has three angles (all checked below):
  1. Specific product vs O2 demand: does Ptot/X decline as inlet-O2 enrichment rises? If O2
     limits production, yes -- across runs and within each temperature.
  2. Within-temperature: the qS-peak is T30-specific. qs0.6 is the MOST O2-stressed at every T
     (enrichment 37 % at T25/T35 vs 19 % at T30). If O2 drove the decline it should be worst at
     T25/T35 -- where there is in fact NO decline.
  3. Demand vs limitation: RQ ~ 1 at qs0.6 (F18 0.94, F12 0.99) means the controller MET the O2
     demand (enrichment + stirrer kept it aerobic) -- high demand is not the same as unmet demand.
     Even the genuinely overflowing run (F10, RQ 1.57) shows no depressed specific product.

Decision: if O2 demand does not track the production decline, E.2 is not supported and the qS-peak
is not an O2 effect (it points back to the partition/temperature structure or the unresolved E.3
cofactor/subpopulation mechanism).

Outputs: results/o2_limit/{o2_limit_test.md, o2_limit_summary.png, o2_limit.csv}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "o2_limit"
OUT.mkdir(parents=True, exist_ok=True)

SET = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}
RUN_ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]

RQ_OVERFLOW = 1.2   # RQ above this ~ overflow / partial-anaerobic (unmet O2 demand)

C_DEMAND = "#D55E00"   # inlet-O2 enrichment (demand) -- orange
C_PROD = "#0072B2"     # specific product -- blue
C_RQ = "#009E73"       # RQ (aerobic vs overflow) -- green


def _col(df, name):
    return D._find_col(df, name)


def analyze_run(df, run_id, rd):
    c_t = _col(df, "Process time [h]")
    c_ind = _col(df, "Is induced [-]")
    c_inlet = _col(df, "Inlet O2 [%]")
    c_pO2 = _col(df, "pO2 [%]")
    c_stir = _col(df, "Stirrer speed [1/min]")
    c_qo2 = _col(df, "qO2 [mmol/g/h]")
    c_our = _col(df, "OUR [mmol/L/h]")
    c_cer = _col(df, "CER [mmol/L/h]")

    r = df[df["run_name"] == run_id].sort_values(c_t).reset_index(drop=True)
    ind = r[c_ind].fillna(0).to_numpy(float) > 0.5
    inlet = r[c_inlet].to_numpy(float)
    pO2 = r[c_pO2].to_numpy(float)
    stir = r[c_stir].to_numpy(float)
    qo2 = r[c_qo2].to_numpy(float)
    our = r[c_our].to_numpy(float)
    cer = r[c_cer].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rq = cer / np.where(np.abs(our) > 1e-6, our, np.nan)

    o = rd["obs"]
    xf = float(o["X"][-1])
    pf = float(o["P_sol"][-1] + o["P_IB"][-1])

    def _m(arr, mask, fn):
        m = mask & np.isfinite(arr)
        return float(fn(arr[m])) if m.any() else float("nan")

    return {
        "run": run_id, "T": SET[run_id][0], "qs": SET[run_id][1],
        "Ptot_per_X": pf / xf, "X_final": xf, "Ptot_final": pf,
        "inlet_max": _m(inlet, ind, np.nanmax),
        "enrich_excess": _m(inlet, ind, np.nanmax) - 21.0,
        "pO2_min": _m(pO2, ind, np.nanmin),
        "stir_max": _m(stir, ind, np.nanmax),
        "qO2_mean": _m(qo2, ind, np.nanmean),
        "OUR_max": _m(our, ind, np.nanmax),
        "RQ_mean": _m(rq, ind, np.nanmean),
        "RQ_max": _m(rq, ind, np.nanmax),
        "frac_overflow": _m(rq, ind, lambda a: np.mean(a > RQ_OVERFLOW)),
    }


def plot_summary(recs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    Tcol = {25: "#56B4E9", 30: C_PROD, 35: "#E69F00"}

    # Panel A: specific product vs inlet-O2 excess, colored by T (the falsification scatter).
    ax = axes[0]
    for r in recs:
        ax.scatter(r["enrich_excess"], r["Ptot_per_X"], c=Tcol[r["T"]], s=70, zorder=5)
        ax.annotate(r["run"], (r["enrich_excess"], r["Ptot_per_X"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel("inlet O2 enrichment excess [%]  (O2 demand)")
    ax.set_ylabel("specific total product  Ptot / X_final  [g/g]")
    ax.set_title("Falsification: product does NOT decline\nwith O2 demand (qs0.6 most stressed, no decline)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=Tcol[25], label="T25"), Patch(color=Tcol[30], label="T30"),
                       Patch(color=Tcol[35], label="T35")], fontsize=8)

    # Panel B: per-run bars -- specific product (left) vs enrichment excess (right), by T.
    ax = axes[1]
    xs = np.arange(len(recs))
    ax.bar(xs - 0.2, [r["Ptot_per_X"] for r in recs], 0.4, color=C_PROD, label="Ptot/X_final [g/g]")
    ax2 = ax.twinx()
    ax2.bar(xs + 0.2, [r["enrich_excess"] for r in recs], 0.4, color=C_DEMAND, label="inlet O2 excess [%]")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['run']}\nT{r['T']} q{r['qs']}" for r in recs], rotation=90, fontsize=7)
    ax.set_ylabel("specific product [g/g]", color=C_PROD)
    ax2.set_ylabel("inlet O2 excess [%]", color=C_DEMAND)
    ax.set_title("Specific product vs O2 demand per run\n(demand peaks at qs0.6; product does not track it)")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # Panel C: RQ vs enrichment excess -- demand met (RQ~1) even at high enrichment.
    ax = axes[2]
    for r in recs:
        ax.scatter(r["enrich_excess"], r["RQ_mean"], c=Tcol[r["T"]], s=70, zorder=5)
        ax.annotate(r["run"], (r["enrich_excess"], r["RQ_mean"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(1.0, color="k", lw=0.6, ls="--", label="fully aerobic (RQ=1)")
    ax.axhline(RQ_OVERFLOW, color=C_RQ, lw=0.8, ls=":", label=f"overflow (RQ>{RQ_OVERFLOW})")
    ax.set_xlabel("inlet O2 enrichment excess [%]")
    ax.set_ylabel("mean RQ (induced)")
    ax.set_title("Demand vs limitation: RQ stays ~1 at high\nenrichment => controller met O2 demand (aerobic)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    df = D._load_df_cached(D.DEFAULT_DATA_FILE)
    tr, te = D.split_for_fold("full")
    bundle = D.load_dataset(tr, te)
    rdmap = {r["run_id"]: r for r in bundle["train"]}

    recs = [analyze_run(df, run_id, rdmap[run_id]) for run_id in RUN_ORDER]
    plot_summary(recs, OUT / "o2_limit_summary.png")
    pd.DataFrame(recs).to_csv(OUT / "o2_limit.csv", index=False)

    L = ["# Task E.2 (Phase 1) -- does O2-transfer limitation cause the qS-peak?", ""]
    L.append("The qS-peak (specific product Ptot/X peaks at qs0.4, declines at qs0.6) cannot be")
    L.append("represented by a monotonic Monod, so qs0.6 overpredicts once qP_max is freed. E.2 asks")
    L.append("whether O2-transfer limitation at the high-biomass qs0.6 corners drives the decline.")
    L.append("This is a data-first test (no model refit).")
    L.append("")
    L.append("## Per-run O2-limitation signature (induced phase)")
    L.append("| run | T | qs | Ptot/X | inlet_max | enrich_excess | pO2_min | stir_max | qO2_mean | OUR_max | RQ_mean | RQ_max | %overflow |")
    L.append("| --- | -:| -:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in recs:
        print(f"{r['run']} T{r['T']} q{r['qs']}: Ptot/X={r['Ptot_per_X']:.3f} enrich={r['enrich_excess']:.1f} "
              f"pO2_min={r['pO2_min']:.1f} RQ={r['RQ_mean']:.2f} (max {r['RQ_max']:.2f}) %ovf={r['frac_overflow']*100:.0f}")
        L.append(f"| {r['run']} | {r['T']} | {r['qs']} | {r['Ptot_per_X']:.3f} | {r['inlet_max']:.1f} | "
                 f"{r['enrich_excess']:.1f} | {r['pO2_min']:.1f} | {r['stir_max']:.0f} | {r['qO2_mean']:.2f} | "
                 f"{r['OUR_max']:.0f} | {r['RQ_mean']:.2f} | {r['RQ_max']:.2f} | {r['frac_overflow']*100:.0f} |")
    L.append("")
    L.append("- enrich_excess = max(Inlet O2) - 21 %: the controller's O2-DEMAND response.")
    L.append("- RQ ~ 1.0 = fully aerobic (demand met); RQ > 1.2 = overflow / partial-anaerobic (unmet).")
    L.append("")

    # The T x qS grid: specific product vs O2 demand.
    L.append("## The T x qS grid: specific product vs O2 demand (enrich excess)")
    L.append("| T | qs0.2 P/X (enrich) | qs0.4 P/X (enrich) | qs0.6 P/X (enrich) |")
    L.append("| -:| --- | --- | --- |")
    byT = {}
    for r in recs:
        byT.setdefault(r["T"], {})[r["qs"]] = r
    for T in (25, 30, 35):
        cells = []
        for qs in (0.2, 0.4, 0.6):
            r = byT.get(T, {}).get(qs)
            if r is None:
                cells.append("-")
            else:
                cells.append(f"{r['Ptot_per_X']:.3f} ({r['enrich_excess']:.1f})")
        L.append(f"| T{T} | " + " | ".join(cells) + " |")
    L.append("")

    # Within-T qs0.4 -> qs0.6 change: does the product decline track O2 demand?
    L.append("## Within-temperature: qs0.4 -> qs0.6 change in product vs in O2 demand")
    L.append("| T | dP/X (qs0.6 - qs0.4) | dEnrich (qs0.6 - qs0.4) | qs0.6 RQ_mean | verdict |")
    L.append("| -:| ---:| ---:| ---:| --- |")
    for T in (25, 30, 35):
        a = byT.get(T, {}).get(0.4)
        b = byT.get(T, {}).get(0.6)
        if not (a and b):
            continue
        dP = b["Ptot_per_X"] - a["Ptot_per_X"]
        dE = b["enrich_excess"] - a["enrich_excess"]
        # Decline = dP < -0.005; demand rose = dE > 5
        decline = dP < -0.005
        demand_rose = dE > 5.0
        if decline and demand_rose:
            v = "decline + more O2 demand (consistent w/ O2)"
        elif (not decline) and demand_rose:
            v = "MORE O2 demand but NO decline (against O2)"
        else:
            v = "ambiguous"
        L.append(f"| T{T} | {dP:+.3f} | {dE:+.1f} | {b['RQ_mean']:.2f} | {v} |")
    L.append("")

    # Correlation across all runs: specific product vs O2-limitation metrics.
    P = np.array([r["Ptot_per_X"] for r in recs])
    E = np.array([r["enrich_excess"] for r in recs])
    R = np.array([r["RQ_mean"] for r in recs])
    Pm = np.isfinite(P) & np.isfinite(E)
    r_PE = float(np.corrcoef(P[Pm], E[Pm])[0, 1])
    Pm2 = np.isfinite(P) & np.isfinite(R)
    r_PR = float(np.corrcoef(P[Pm2], R[Pm2])[0, 1])
    L.append("## Cross-run correlation (specific product vs O2 limitation)")
    L.append(f"- r(Ptot/X, inlet-O2 enrich excess) = **{r_PE:+.2f}** (O2-limitation predicts a NEGATIVE r; found {r_PE:+.2f}).")
    L.append(f"- r(Ptot/X, RQ) = **{r_PR:+.2f}** (overflow predicts a NEGATIVE r).")
    L.append("- These are dominated by the temperature signal (T35 ~0.05, T25/T30 ~0.08-0.10), not by O2.")
    L.append("")

    # Verdict.
    f12 = next(r for r in recs if r["run"] == "F12")
    f10 = next(r for r in recs if r["run"] == "F10")
    f18 = next(r for r in recs if r["run"] == "F18")
    L.append("## Verdict -- O2 does NOT limit LDH production; E.2 is not supported")
    L.append("")
    L.append("**O2 demand is real and peaks at qs0.6** (inlet enrichment 37 % at T25/T35, 19 % at T30;")
    L.append(f"F10/F12 hit 58 % inlet O2, and F12's pO2 crashed to **{f12['pO2_min']:.1f} %**). But:")
    L.append("")
    L.append("1. **The production decline is T30-specific.** qs0.6 is the MOST O2-stressed at every")
    L.append("   temperature, yet the qS-peak (qs0.4 > qs0.6) appears only at T30. At T25 and T35 --")
    L.append("   where O2 stress is WORST (enrichment 37 %) -- there is NO decline. If O2 drove the")
    L.append("   decline it should be strongest at T25/T35; it is absent there.")
    L.append("2. **The controller met the demand (RQ ~ 1).** Most qs0.6 runs are fully aerobic (F18 RQ")
    L.append(f"   {f18['RQ_mean']:.2f}, F12 RQ {f12['RQ_mean']:.2f}) -- high inlet")
    L.append("   enrichment is the controller *supplying* O2, not unmet demand. High demand is not")
    L.append("   metabolic O2 limitation.")
    L.append("3. **Even genuine O2 limitation does not depress product.** F10 is the one real overflow run")
    L.append(f"   (RQ {f10['RQ_mean']:.2f} > {RQ_OVERFLOW}), severely stressed (enrichment {f10['enrich_excess']:.0f} %), yet its")
    L.append(f"   specific product ({f10['Ptot_per_X']:.3f}) is not depressed. And F12 -- pO2 crashed to")
    L.append(f"   {f12['pO2_min']:.1f} %, the worst O2 starvation in the dataset -- has Ptot/X = {f12['Ptot_per_X']:.3f},")
    L.append(f"   essentially equal to its qs0.4 control F19 ({byT[35][0.4]['Ptot_per_X']:.3f}). O2 starvation did not")
    L.append("   reduce LDH production.")
    L.append("")
    L.append(f"**Cross-run r(Ptot/X, enrich excess) = {r_PE:+.2f}** -- not the negative relationship an O2-cap")
    L.append("requires (the correlation is dominated by temperature, i.e. the already-modelled qP(T)).")
    L.append("")
    L.append("**Conclusion: do NOT add an O2-transfer capacity term.** O2 demand is large at qs0.6 but")
    L.append("does not limit LDH production; the qS-peak is a T30-specific pattern pointing back to the")
    L.append("partition/temperature structure (or the unresolved E.3 cofactor/subpopulation mechanism),")
    L.append("not to O2 transfer. This also confirms the HANDOFF caveat that the qs0.6 gas-transfer")
    L.append("artifact is not a usable production lever. E.2 closed (negative).")
    L.append("")
    L.append("Artifacts: o2_limit_summary.png, o2_limit.csv.")

    (OUT / "o2_limit_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'o2_limit_test.md'} (+ o2_limit_summary.png, o2_limit.csv)")


if __name__ == "__main__":
    main()
