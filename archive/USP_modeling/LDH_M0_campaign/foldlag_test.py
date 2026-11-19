"""Test the soluble folding-transit delay (M0_aload_qPTexp_foldlag).

Does adding a soluble folding-transit delay tau_fold (IB immediate, soluble emerges from
a folding pool after tau_fold) improve the IB-fraction DYNAMICS and the endpoint IB
predictions? Compare to qPTexp (constant partition). Check tau_fold identifiability.
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


def _col(b):
    for c in df.columns:
        if c.split(' [')[0].lower() == b.lower():
            return c


def _obs_traj(run_id):
    """observed (t_since_ind, P_sol, P_IB, ibfrac) at sample points during induction."""
    r = df[df.run_name == run_id].sort_values(_col('Process time'))
    t = r[_col('Process time')].to_numpy(float)
    Ps = np.nan_to_num(r[_col('Protein soluble')].to_numpy(float), nan=0)
    Pb = np.nan_to_num(r[_col('Protein IB')].to_numpy(float), nan=0)
    is_s = r[_col('is_sample')].fillna(False).to_numpy(bool)
    ind = np.nan_to_num(r[_col('Is induced')].to_numpy(float), nan=0) > 0.5
    if not ind.any():
        return []
    t_ind = t[ind][0]
    out = []
    for ti, ps, pb, sm in zip(t, Ps, Pb, is_s):
        if sm and (ps + pb) >= 0.1:
            out.append((ti - t_ind, ps, pb, pb / (ps + pb)))
    return out


def _ibfrac_traj(model, params, run, c_fold):
    pred = simulate_run(model, params, run, c_fold, eval_gas=False)
    st, sPs, sPb = pred['sol_t'], pred['sol_Psol'], pred['sol_Pib']
    ind = run['is_induced'] > 0.5
    t_ind = run['t_online'][ind][0] if ind.any() else st[0]
    out = []
    for dh, ps_o, pb_o, ibf_o in _obs_traj(run['run_id']):
        ti = t_ind + dh
        ps = float(np.interp(ti, st, sPs)); pb = float(np.interp(ti, st, sPb))
        out.append((dh, ibf_o, pb / (ps + pb + 1e-9)))
    return out


def main():
    tr, te = D.split_for_fold("full")
    b = D.load_dataset(tr, te)
    r0 = F.fit_model("M0_aload_qPTexp", b, seed=42, n_starts=3, max_nfev=300)
    r1 = F.fit_model("M0_aload_qPTexp_foldlag", b, seed=42, n_starts=4, max_nfev=350)
    print(f"qPTexp        : cost={r0.cost:.4f}")
    print(f"qPTexp_foldlag: cost={r1.cost:.4f}  tau_fold={r1.params['tau_fold']:.2f}h  "
          f"alpha={r1.params['alpha_T']:.3f}  qP_max={r1.params['qP_max']:.3f}  "
          f"a_qS={r1.params['a_qS']:.2f}  theta_T={r1.params['theta_T']:.2f}")

    # IB-fraction trajectory RMSE (at sample times) per run
    print("\nIB-fraction trajectory RMSE at sample times (qPTexp -> foldlag):")
    L = ["# foldlag (soluble transit delay) test", "",
         f"qPTexp cost={r0.cost:.4f}; foldlag cost={r1.cost:.4f} tau_fold={r1.params['tau_fold']:.2f}h", "",
         "| run | IB-RMSE qPTexp | IB-RMSE foldlag | endpoint IB obs/qPT/fold |"]
    L.append("| --- | ---:| ---:| --- |")
    runs = {r['run_id']: r for r in b['train']}
    for rid in ['F17', 'F13', 'F14', 'F15', 'F18']:
        run = runs[rid]
        t0 = _ibfrac_traj("M0_aload_qPTexp", r0.params, run, b['c_fold'])
        t1 = _ibfrac_traj("M0_aload_qPTexp_foldlag", r1.params, run, b['c_fold'])
        rmse0 = float(np.sqrt(np.mean([(a[1] - a[2]) ** 2 for a in t0])))
        rmse1 = float(np.sqrt(np.mean([(a[1] - a[2]) ** 2 for a in t1])))
        eo = t0[-1][1]; e0 = t0[-1][2]; e1 = t1[-1][2]
        print(f"  {rid}: {rmse0:.3f} -> {rmse1:.3f}   endpoint IB obs/qPT/fold = {eo:.3f}/{e0:.3f}/{e1:.3f}")
        L.append(f"| {rid} | {rmse0:.3f} | {rmse1:.3f} | {eo:.3f}/{e0:.3f}/{e1:.3f} |")

    # tau_fold profile
    grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    prof = []
    for v in grid:
        rr = F.fit_model("M0_aload_qPTexp_foldlag", b, seed=42, n_starts=1, fixed={"tau_fold": float(v)}, max_nfev=250)
        prof.append((float(v), float(rr.cost) if rr else float("inf")))
    base = min(c for _, c in prof)
    print("\ntau_fold profile:")
    L.append("")
    L.append("| tau_fold | cost | delta |")
    L.append("| ---:| ---:| ---:|")
    for v, c in prof:
        print(f"  tau_fold={v:.1f}h: cost={c:.4f} dcost={c-base:+.4f}")
        L.append(f"| {v:.1f} | {c:.4f} | {c-base:+.4f} |")
    (OUT / "foldlag_test.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'foldlag_test.md'}")


if __name__ == "__main__":
    main()
