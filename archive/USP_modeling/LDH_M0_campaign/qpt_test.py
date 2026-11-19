"""Test the qP(T) temperature-suppression variant (M0_aload_qPT).

Does adding a T-suppression of total production (hinge at 30 C, 1 param k_T):
  (a) fix the dominant T35 overprediction (every T35 run overpredicted +1.2..+2.0)?
  (b) help the center (T30/qs0.4) underprediction?
  (c) is k_T identifiable (profile likelihood)?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import fit as F
from models import simulate_run

OUT = Path(__file__).resolve().parent / "results" / "qpt"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_pickle(Path(__file__).resolve().parent.parent / "LDH_consolidated_results_processed.pkl")
SET = {'F07':(35,0.2),'F09':(25,0.2),'F10':(25,0.6),'F12':(35,0.6),'F13':(30,0.4),'F14':(30,0.4),
       'F15':(30,0.4),'F16':(25,0.4),'F17':(30,0.2),'F18':(30,0.6),'F19':(35,0.4)}


def _col(b):
    for c in df.columns:
        if c.split(' [')[0].lower() == b.lower():
            return c


def _ptot_obs(run_id):
    r = df[df.run_name == run_id].sort_values(_col('Process time'))
    return (np.nan_to_num(r[_col('Protein soluble')].to_numpy(float), nan=0)[-1]
            + np.nan_to_num(r[_col('Protein IB')].to_numpy(float), nan=0)[-1])


def _biases(model, params, runs, c_fold):
    out = {}
    for run in runs:
        pred = simulate_run(model, params, run, c_fold, eval_gas=False)
        pr = pred['obs']['P_sol'][-1] + pred['obs']['P_IB'][-1]
        out[run['run_id']] = pr - _ptot_obs(run['run_id'])
    return out


def main():
    L = ["# qP(T) T-suppression variant (M0_aload_qPT)", ""]

    # ---- Full-data fit: M0_aload vs M0_aload_qPT per-run Ptot bias ----
    tr, te = D.split_for_fold("full")
    b = D.load_dataset(tr, te)
    r0 = F.fit_model("M0_aload", b, seed=42, n_starts=3, max_nfev=300)
    r1 = F.fit_model("M0_aload_qPT", b, seed=42, n_starts=3, max_nfev=300)
    bi0 = _biases("M0_aload", r0.params, b['train'], b['c_fold'])
    bi1 = _biases("M0_aload_qPT", r1.params, b['train'], b['c_fold'])
    p = r1.params
    print(f"M0_aload      full cost={r0.cost:.4f}  Y_xs={r0.params['Y_xs']:.3f}  qP_max={r0.params['qP_max']:.3f}")
    print(f"M0_aload_qPT  full cost={r1.cost:.4f}  Y_xs={p['Y_xs']:.3f}  qP_max={p['qP_max']:.3f}  k_T={p['k_T']:.3f}")
    print(f"  (k_T={p['k_T']:.3f} -> T35 factor={max(1-p['k_T']*5,0.05):.2f})")
    print()
    print("run  T qs | M0_aload bias -> qPT bias")
    L.append("## Full-data per-run Ptot bias (pred-obs)")
    L.append("| run | T | qs | M0_aload | M0_aload_qPT |")
    L.append("| --- | -:| -:| ---:| ---:|")
    for run in ['F09','F16','F10','F17','F13','F14','F15','F18','F07','F19','F12']:
        T, qs = SET[run]
        print(f"{run} {T} {qs} | {bi0[run]:+.2f} -> {bi1[run]:+.2f}")
        L.append(f"| {run} | {T} | {qs} | {bi0[run]:+.2f} | {bi1[run]:+.2f} |")
    # T35 summary
    t35_0 = np.mean([bi0[r] for r in ['F07','F12','F19']])
    t35_1 = np.mean([bi1[r] for r in ['F07','F12','F19']])
    ctr_1 = np.mean([bi1[r] for r in ['F13','F14','F15']])
    print(f"\nT35 mean bias: {t35_0:+.2f} -> {t35_1:+.2f}   center(T30/0.4) mean bias: {ctr_1:+.2f}")
    L.append(f"\nT35 mean bias: {t35_0:+.2f} -> {t35_1:+.2f};  center(T30/0.4) mean bias: {ctr_1:+.2f}")
    L.append("")

    # ---- Fold B held-out center ----
    tr, te = D.split_for_fold("B")
    bB = D.load_dataset(tr, te)
    rB = F.fit_model("M0_aload_qPT", bB, seed=42, n_starts=3, max_nfev=300)
    pB = rB.params
    print(f"\nfold B qPT: cost={rB.cost:.4f} k_T={pB['k_T']:.3f} Y_xs={pB['Y_xs']:.3f}")
    L.append("## Fold B held-out center (F13/F14/F15)")
    L.append("| run | Ptot obs/pred | IB obs/pred |")
    L.append("| --- | --- | --- |")
    for run in bB['test']:
        pred = simulate_run("M0_aload_qPT", pB, run, bB['c_fold'], eval_gas=False)
        o = _ptot_obs(run['run_id']); pr = pred['obs']['P_sol'][-1] + pred['obs']['P_IB'][-1]
        io = run['obs']['P_IB'][-1] / max(run['obs']['P_sol'][-1] + run['obs']['P_IB'][-1], 1e-6)
        ip = pred['obs']['P_IB'][-1] / max(pred['obs']['P_sol'][-1] + pred['obs']['P_IB'][-1], 1e-6)
        print(f"  {run['run_id']}: Ptot {o:.2f}/{pr:.2f}  IB {io:.3f}/{ip:.3f}")
        L.append(f"| {run['run_id']} | {o:.2f}/{pr:.2f} | {io:.3f}/{ip:.3f} |")
    L.append("")

    # ---- k_T profile (identifiability) on full data ----
    grid = np.array([0.0, 0.03, 0.06, 0.10, 0.14, 0.18])
    prof = []
    for v in grid:
        rr = F.fit_model("M0_aload_qPT", b, seed=42, n_starts=1, fixed={"k_T": float(v)}, max_nfev=250)
        prof.append((float(v), float(rr.cost) if rr else float("inf")))
    base = min(c for _, c in prof)
    print("\nk_T profile (full data):")
    L.append("## k_T profile likelihood (full data)")
    L.append("| k_T | train cost | delta_cost |")
    L.append("| ---:| ---:| ---:|")
    for v, c in prof:
        print(f"  k_T={v:.2f}: cost={c:.4f} dcost={c-base:+.4f}")
        L.append(f"| {v:.2f} | {c:.4f} | {c-base:+.4f} |")
    (OUT / "qpt_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'qpt_test.md'}")


if __name__ == "__main__":
    main()
