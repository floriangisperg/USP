"""Test qPT + Haldane (substrate inhibition) for the residual qS-peak.

After qP(T), all qs0.6 runs overpredict (F10 +0.80, F18 +0.60, F12 +0.54) and qP_max
is interior (0.046), so substrate-inhibition may now be identifiable (it was rejected
before only because qP_max sat at its bound). Haldane: qP = qP_max*qS/(K_S+qS+qS^2/Ki),
peak at qS = sqrt(Ki*K_S). Check: qs0.6 overprediction shrinks? Ki interior? peak in-data?
"""

from __future__ import annotations

from pathlib import Path

import data as D
import fit as F
import numpy as np
import pandas as pd
from models import simulate_run

OUT = Path(__file__).resolve().parent / "results" / "qpt"
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
        out[run['run_id']] = pred['obs']['P_sol'][-1] + pred['obs']['P_IB'][-1] - _ptot_obs(run['run_id'])
    return out


def main():
    tr, te = D.split_for_fold("full")
    b = D.load_dataset(tr, te)
    r = F.fit_model("M0_aload_qPT_haldane", b, seed=42, n_starts=4, max_nfev=350)
    p = r.params
    Ki, K_S = p["Ki"], p["K_S"]
    qpeak = np.sqrt(Ki * K_S)
    print(f"M0_aload_qPT_haldane full: cost={r.cost:.4f} (qPT was 0.5273)")
    print(f"  qP_max={p['qP_max']:.3f}  K_S={K_S:.3f}  Ki={Ki:.3f}  k_T={p['k_T']:.3f}  Y_xs={p['Y_xs']:.3f}")
    print(f"  Haldane peak at qS = sqrt(Ki*K_S) = {qpeak:.3f}  (realized qS ~0.17/0.27/0.30)")
    bi = _biases("M0_aload_qPT_haldane", p, b['train'], b['c_fold'])
    print("\nrun  T qs | qPT bias -> qPT+Haldane bias")
    qpt_biases = {'F09':-0.50,'F16':+0.72,'F10':+0.80,'F17':+0.34,'F13':-0.56,'F14':-0.07,
                  'F15':-0.50,'F18':+0.60,'F07':+0.28,'F19':+0.04,'F12':+0.54}
    for run in ['F09','F16','F10','F17','F13','F14','F15','F18','F07','F19','F12']:
        T, qs = SET[run]
        print(f"{run} {T} {qs} | {qpt_biases[run]:+.2f} -> {bi[run]:+.2f}")
    q6 = np.mean([bi[r] for r in ['F10','F18','F12']])
    ctr = np.mean([bi[r] for r in ['F13','F14','F15']])
    print(f"\nqs0.6 mean bias: +0.65 -> {q6:+.2f}   center(T30/0.4) mean bias: {ctr:+.2f}")

    # Ki profile
    grid = np.array([0.05, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0])
    prof = []
    for v in grid:
        rr = F.fit_model("M0_aload_qPT_haldane", b, seed=42, n_starts=1, fixed={"Ki": float(v)}, max_nfev=250)
        prof.append((float(v), float(rr.cost) if rr else float("inf")))
    base = min(c for _, c in prof)
    print("\nKi profile:")
    for v, c in prof:
        qp = np.sqrt(v * K_S)
        print(f"  Ki={v:5.2f} (peak qS={qp:.3f}): cost={c:.4f} dcost={c-base:+.4f}")
    print(f"\nsaved {OUT/'haldane_test.md'}")


if __name__ == "__main__":
    main()
