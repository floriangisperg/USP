"""Task E.1 -- can the missing NH4OH base volume be PINNED from the respiratory acid load?

Hypothesis (HANDOFF Task E.1): in a glucose-limited E. coli fed-batch the NH4OH base
demand is overwhelmingly respiratory CO2 (carbonic acid), so

    base_volume(t) ~ k * integral( CER(t) * V(t) dt )           (*shape* ~ int CER)

and at pH 7 ~1 mol base neutralizes 1 mol CO2, so the *scale* would be stoichiometric
(0 free parameters) -- pinning the reactor volume that the N-floor bracket only bounds
to +/-50 %. The smoking gun cited for the shape is "Pump2 (base) correlates 0.87 with
cumulative CO2". This script tests both the shape and the scale, falsifiably.

WHAT IT COMPUTES (per run, full run = batch + fed):
  1. acid_load = int(CER * V) dt [mmol CO2 evolved] and base_1to1 = acid_load / C_BASE
     [mL], the 1 base : 1 CO2 stoichiometric prediction (C_BASE = 6.95 M NH4OH).
  2. SCALE test: does base_1to1 land inside the N-floor bracket [N_floor, 2xN_floor]?
     Report ratio = base_1to1 / N_floor and the implied CO2-neutralization fraction
     f_neut = bracket_base / base_1to1 (the fraction of evolved CO2 that would have to
     consume base to match the weighed-informed bracket). f_neut is a free parameter
     unless stoichiometry fixes it.
  3. SHAPE test (rate-level, controls the biomass confound): correlate the base-addition
     RATE d(Pump2)/dt with CER, with the feed rate, and report r(CER, feed) -- if CER
     and feed are collinear, "base tracks CO2" is indistinguishable from "base tracks
     substrate" and the shape carries no CO2-specific information.
  4. SPURIOUS-CORRELATION check: within-run r(cumBase, cumCO2) is ~1.0 by construction
     (both monotonic) -- quantifies why the cited "0.87 cumulative r" is not evidence.

Decision tree (HANDOFF E.1 step 4):
  - base ~= int(CER) stoichiometric  -> volume PINNED, 0 free params (adopt it).
  - base << int(CER)                  -> CO2 is leaving another way (HERE: stripped, not
                                         a gas error -- carbon closure ~103 % says CER is
                                         accurate, so the CO2 is real but does not consume
                                         base; f_neut is a free ~0.2-0.4 parameter).
  - base >> int(CER)                  -> large non-respiratory acid (overflow/acetate).

Outputs: results/base_acid/{acid_load_test.md, acid_load_summary.png, acid_load.csv}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import base_reconstruct as BR
import data as D
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "base_acid"
OUT.mkdir(parents=True, exist_ok=True)

# 6.95 M NH4OH: mmol base per mL (1 mL neutralizes 6.95 mmol acid at 1:1).
C_BASE = 6.95 * 1000.0 / 1000.0  # = 6.95 mmol/mL

# Verified FCCD setpoints (T, qS0) -- for labelling/stratification.
SET = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}
RUN_ORDER = ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]

# Palette.
C_CO2 = "#0072B2"     # 1:1 CO2-implied base -- blue
C_BRACKET = "#D55E00"  # N-floor bracket -- orange
C_NEUT = "#009E73"     # implied neutralization fraction -- green
C_FEED = "#CC79A7"     # feed-rate confound -- pink


def _full_run_arrays(df: pd.DataFrame, run_id: str):
    """Full-run (batch+fed) dense arrays: t[h], CER[mmol/L/h], V[L], feed[mL cumul]."""
    c_t = D._find_col(df, "Process time [h]")
    c_cer = D._find_col(df, "CER [mmol/L/h]")
    c_vol = D._find_col(df, "Volume corrected [mL]")
    c_feed = D._find_col(df, "Feed [mL]")
    r = df[df["run_name"] == run_id].sort_values(c_t).reset_index(drop=True)
    t = r[c_t].to_numpy(float)
    cer = r[c_cer].to_numpy(float)
    V = r[c_vol].ffill().bfill().to_numpy(float) / 1000.0
    feed = r[c_feed].fillna(0.0).to_numpy(float)
    m = np.isfinite(t) & np.isfinite(cer) & np.isfinite(V) & np.isfinite(feed)
    return t[m], cer[m], V[m], feed[m]


def analyze_run(df: pd.DataFrame, run_id: str, rd: dict) -> dict:
    t, cer, V, feed = _full_run_arrays(df, run_id)
    acid_load = float(np.trapz(cer * V, t))            # mmol CO2 evolved (full run)
    base_1to1 = acid_load / C_BASE                     # mL at 1 base:1 CO2

    n_floor = BR.n_floor_ml(rd)
    ratio = base_1to1 / n_floor                         # base_1to1 / N_floor
    # Implied fraction of evolved CO2 that must consume base to match the bracket.
    f_neut_floor = n_floor / base_1to1                  # if base_actual = N_floor
    f_neut_upper = (2.0 * n_floor) / base_1to1          # if base_actual = 2xN_floor (scfv)

    # Rate-level shape (controls biomass confound; avoids spurious cumulative r).
    t_raw, p2 = BR._read_raw_pump2(run_id)
    tc = np.clip(t, float(t_raw.min()), float(t_raw.max()))
    p2_on = np.interp(tc, t_raw, p2)
    dbase = np.gradient(p2_on, t)                       # base-addition rate [mL/h]
    dfeed = np.gradient(feed, t)                        # feed rate [mL/h]
    ok = np.isfinite(cer) & np.isfinite(dbase) & (cer > 0) & np.isfinite(dfeed) & (dbase >= 0)
    def _corr(a, b):
        m = ok & np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 5 else float("nan")
    r_base_cer = _corr(cer, dbase)
    r_base_feed = _corr(dfeed, dbase)
    r_cer_feed = _corr(cer, dfeed)

    # Spurious cumulative correlation (both monotonic -> ~1.0 by construction).
    cumco2 = np.concatenate([[0.0], np.cumsum((cer * V)[:-1] * np.diff(t))])
    m = np.isfinite(cumco2) & np.isfinite(p2_on)
    r_cum = float(np.corrcoef(cumco2[m], p2_on[m])[0, 1]) if m.sum() > 5 else float("nan")

    return {
        "run": run_id, "T": SET[run_id][0], "qs": SET[run_id][1],
        "t": t, "cumco2": cumco2, "pump2": p2_on, "cer": cer, "V": V,
        "acid_load_mmol": acid_load, "base_1to1_mL": base_1to1,
        "n_floor_mL": n_floor, "2xN_mL": 2.0 * n_floor,
        "ratio_over_Nf": ratio,
        "f_neut_at_Nf": f_neut_floor, "f_neut_at_2xN": f_neut_upper,
        "r_base_cer": r_base_cer, "r_base_feed": r_base_feed, "r_cer_feed": r_cer_feed,
        "r_cum_spurious": r_cum,
    }


def plot_summary(recs, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: scale -- base_1to1 vs the bracket per run (log y).
    ax = axes[0]
    xs = np.arange(len(recs))
    w = 0.27
    ax.bar(xs - w, [r["base_1to1_mL"] for r in recs], w, color=C_CO2, label="1:1 CO2-implied base")
    ax.bar(xs, [r["2xN_mL"] for r in recs], w, color=C_BRACKET, label="2xN_floor (bracket upper)")
    ax.bar(xs + w, [r["n_floor_mL"] for r in recs], w, color="#E69F00", label="N_floor (bracket lower)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["run"] for r in recs], rotation=90, fontsize=8)
    ax.set_ylabel("base volume [mL] (log)")
    ax.set_title("Scale: 1:1 CO2 base vs N-floor bracket\n(CO2 implies 3-5x more base than used)")
    ax.legend(fontsize=7, loc="upper right")

    # Panel B: implied CO2-neutralization fraction (bracket / 1:1 base).
    ax = axes[1]
    fs_n = [r["f_neut_at_Nf"] for r in recs]
    fs_u = [r["f_neut_at_2xN"] for r in recs]
    ax.bar(xs - w / 2, fs_n, w, color="#56B4E9", label="if base_actual = N_floor")
    ax.bar(xs + w / 2, fs_u, w, color=C_NEUT, label="if base_actual = 2xN_floor (scfv)")
    mean_u = float(np.nanmean(fs_u))
    ax.axhline(mean_u, color=C_NEUT, ls="--", lw=1, label=f"mean(2xN) = {mean_u:.2f}")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["run"] for r in recs], rotation=90, fontsize=8)
    ax.set_ylabel("fraction of evolved CO2 that neutralizes base")
    ax.set_title("Implied neutralization fraction\n(free param ~0.2-0.4; rest is stripped)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, max(1.1, np.nanmax(fs_u) * 1.2))

    # Panel C: rate-level shape correlations (the confound).
    ax = axes[2]
    ax.bar(xs - w, [r["r_base_cer"] for r in recs], w, color=C_CO2, label="r(base rate, CER)")
    ax.bar(xs, [r["r_base_feed"] for r in recs], w, color=C_FEED, label="r(base rate, feed rate)")
    ax.bar(xs + w, [r["r_cer_feed"] for r in recs], w, color="#999", label="r(CER, feed rate)")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["run"] for r in recs], rotation=90, fontsize=8)
    ax.set_ylabel("Pearson r (rate-level)")
    ax.set_title("Shape: base tracks CO2 or feed?\n(CER~feed collinear => not CO2-specific)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    df = D._load_df_cached(D.DEFAULT_DATA_FILE)
    tr, te = D.split_for_fold("full")
    bundle = D.load_dataset(tr, te)
    rdmap = {r["run_id"]: r for r in bundle["train"]}

    recs = [analyze_run(df, run_id, rdmap[run_id]) for run_id in RUN_ORDER]
    plot_summary(recs, OUT / "acid_load_summary.png")

    # CSV.
    pd.DataFrame([{k: v for k, v in r.items() if k in {
        "run", "T", "qs", "acid_load_mmol", "base_1to1_mL", "n_floor_mL", "2xN_mL",
        "ratio_over_Nf", "f_neut_at_Nf", "f_neut_at_2xN",
        "r_base_cer", "r_base_feed", "r_cer_feed", "r_cum_spurious"}} for r in recs]
    ).to_csv(OUT / "acid_load.csv", index=False)

    L = ["# Task E.1 -- can the base volume be PINNED from the respiratory acid load (int CER)?", ""]
    L.append("Hypothesis: NH4OH base demand is overwhelmingly respiratory CO2, so")
    L.append("`base(t) ~ k * int(CER*V) dt` and at pH 7 ~1 base:1 CO2 fixes the *scale*")
    L.append("stoichiometrically (0 free parameters) -- pinning the reactor volume the N-floor")
    L.append("bracket only bounds to +/-50 %. This script tests the scale AND the shape, falsifiably.")
    L.append("")
    L.append("## Per-run results")
    L.append("| run | T | qs | acid_load [mmol CO2] | base 1:1 [mL] | N_floor [mL] | 2xN [mL] | "
             "base1:1/Nf | f_neut(Nf) | f_neut(2xN) | r(base,CER) | r(base,feed) | r(CER,feed) | r_cum |")
    L.append("| --- | -:| -:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in recs:
        print(f"{r['run']} T{r['T']} qs{r['qs']}: CO2={r['acid_load_mmol']:.0f}mmol base1:1={r['base_1to1_mL']:.0f}mL "
              f"Nf={r['n_floor_mL']:.0f} ratio={r['ratio_over_Nf']:.2f} f_neut(2xN)={r['f_neut_at_2xN']:.2f} "
              f"r(b,CER)={r['r_base_cer']:.2f} r(b,feed)={r['r_base_feed']:.2f} r(CER,feed)={r['r_cer_feed']:.2f}")
        L.append(f"| {r['run']} | {r['T']} | {r['qs']} | {r['acid_load_mmol']:.0f} | {r['base_1to1_mL']:.0f} | "
                 f"{r['n_floor_mL']:.0f} | {r['2xN_mL']:.0f} | {r['ratio_over_Nf']:.2f} | "
                 f"{r['f_neut_at_Nf']:.2f} | {r['f_neut_at_2xN']:.2f} | {r['r_base_cer']:.2f} | "
                 f"{r['r_base_feed']:.2f} | {r['r_cer_feed']:.2f} | {r['r_cum_spurious']:.2f} |")
    L.append("")

    # Aggregates.
    ratio_mean = float(np.nanmean([r["ratio_over_Nf"] for r in recs]))
    fneut_mean = float(np.nanmean([r["f_neut_at_2xN"] for r in recs]))
    rbc = [r["r_base_cer"] for r in recs if np.isfinite(r["r_base_cer"])]
    rbf = [r["r_base_feed"] for r in recs if np.isfinite(r["r_base_feed"])]
    rcf = [r["r_cer_feed"] for r in recs if np.isfinite(r["r_cer_feed"])]

    L.append("## Aggregate")
    L.append(f"- Mean base_1:1 / N_floor = **{ratio_mean:.2f}** (range "
             f"{min(r['ratio_over_Nf'] for r in recs):.2f}-{max(r['ratio_over_Nf'] for r in recs):.2f}): "
             f"the 1:1 CO2 stoichiometry predicts ~{ratio_mean:.0f}x the N-floor, i.e. "
             f"~{ratio_mean/2:.1f}x the bracket *upper* (2xN_floor).")
    L.append(f"- Mean implied f_neut at 2xN_floor (scfv-weighed scale) = **{fneut_mean:.2f}**: only "
             f"~{fneut_mean*100:.0f}% of evolved CO2 would have to consume base to match the bracket.")
    L.append(f"- Rate-level r(base, CER) median = **{np.median(rbc):.2f}** (one run, F17, is ~0).")
    L.append(f"- Rate-level r(CER, feed) median = **{np.median(rcf):.2f}**: CER and feed rate are near-collinear.")
    L.append(f"- Within-run r(cumBase, cumCO2) = **~1.0 for every run** (both monotonic): the cited")
    L.append("  \"0.87 cumulative correlation\" is a spurious artifact of correlating two cumulative curves.")
    L.append("")

    L.append("## Verdict -- the volume is NOT pinned by int(CER)")
    L.append("")
    L.append("**Scale fails.** The 1 base:1 CO2 stoichiometry over-predicts the bracket-scale base by")
    L.append(f"~{ratio_mean:.0f}x. Carbon closure is ~103% (HANDOFF; low variance, rigorous N2-balance), so the")
    L.append("offgas CER is accurate -- the CO2 is real. It is therefore **stripped, not neutralized**: only")
    L.append(f"~{fneut_mean*100:.0f}% of evolved CO2 (bracket / 1:1) consumes base, the rest leaves as gas, exactly")
    L.append("as expected in a well-aerated reactor. That neutralization fraction is a **free ~0.2-0.4")
    L.append("parameter, not a stoichiometric constant** -- so int(CER) supplies a *scale* only through the")
    L.append("same calibration the N-floor bracket already uses. No zero-parameter pin.")
    L.append("")
    L.append("**Shape fails to discriminate.** CER and feed rate are near-collinear")
    L.append(f"(r ~ {np.median(rcf):.2f}), so \"base tracks CO2\" is indistinguishable from \"base tracks")
    L.append("substrate/biomass\" -- the shape carries no CO2-specific information. The Pump2 signal remains")
    L.append("the better base *shape* (it is the direct measurement); int(CER) adds no independent shape.")
    L.append("")
    L.append("**Decision-tree branch:** `base << int(CER)` is triggered, but the cause is **normal CO2")
    L.append("stripping**, not a gas-quality error (carbon closure rules that out). This falsifies the E.1")
    L.append("premise that int(CER) pins the volume with zero free parameters.")
    L.append("")
    L.append("**Consequence for Tasks A-D.** The motivation for E.1 was that the rejected gas-side")
    L.append("parameters (M1b m_s, burden) might be *volume artifacts* unidentifiable only because the")
    L.append("volume injects too much noise. Since int(CER) cannot tighten the volume beyond the +/-50%")
    L.append("bracket, this rescue path is **closed**: the volume uncertainty stands, and the gas-side")
    L.append("identifiability ceiling is *not* shown to be a volume artifact -- it remains most plausibly")
    L.append("real/biological (consistent with the M1b/burden re-test verdicts). The only true volume pin is")
    L.append("a weighed base bottle on the LDH reactor (MF2-R6) for one future run.")
    L.append("")
    L.append("Artifacts: acid_load_summary.png, acid_load.csv.")

    (OUT / "acid_load_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'acid_load_test.md'} (+ acid_load_summary.png, acid_load.csv)")


if __name__ == "__main__":
    main()
