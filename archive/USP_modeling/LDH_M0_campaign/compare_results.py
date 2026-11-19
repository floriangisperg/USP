"""Aggregate mechanistic-playground results into a comparison table.

Reads every results/<model>/<split>/metrics.json and prints:
  - held-out overall NRMSE_range / NLL (primary, scale-free)
  - per-state X / P_sol / P_IB RMSE and bias (held-out)
  - key parameters Y_xs, theta_T (+ m_s if present) with seed spread
  - final IB fraction observed vs predicted for held-out runs

Usage:
  python compare_results.py            # all results
  python compare_results.py M0 M1a     # subset of models
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

RESULTS = Path(__file__).resolve().parent / "results"


def load_all() -> List[dict]:
    out = []
    for mj in sorted(RESULTS.glob("*/*/metrics.json")):
        try:
            out.append(json.loads(mj.read_text()))
        except Exception:
            pass
    return out


def _seed_spread(all_seeds, key):
    vals = [s["params"].get(key) for s in all_seeds if s["params"].get(key) is not None]
    if not vals:
        return float("nan"), float("nan")
    return min(vals), max(vals)


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "  -- "
    return f"{x:.{p}f}"


def main(models: List[str]):
    recs = load_all()
    if models:
        recs = [r for r in recs if r["model"] in models]
    if not recs:
        print("No results found under", RESULTS)
        return

    print("=" * 110)
    print("HELD-OUT OVERALL (primary = NRMSE_range, scale-free) + per-state RMSE / bias (obs-pred)")
    print("=" * 110)
    hdr = f"{'model':6} {'fold':4} {'test':10} {'ovRMSE':>7} {'ovNRMSE':>8} {'ovNLL':>8} | {'X_RMSE':>7} {'X_bias':>7} {'Psol_RMSE':>9} {'Psol_bias':>9} {'Pib_RMSE':>8} {'Pib_bias':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        t = r.get("test")
        if not t:
            continue
        ov = t["overall"]
        ps = t["per_state"]
        def s(st, m):
            return ps.get(st, {}).get(m, float("nan"))
        print(f"{r['model']:6} {r['fold']:4} {','.join(r['test_run_ids']):10} "
              f"{fmt(ov.get('RMSE')):>7} {fmt(ov.get('NRMSE_range')):>8} {fmt(ov.get('NLL'),1):>8} | "
              f"{fmt(s('X','RMSE')):>7} {fmt(s('X','Bias')):>7} {fmt(s('P_sol','RMSE')):>9} {fmt(s('P_sol','Bias')):>9} "
              f"{fmt(s('P_IB','RMSE')):>8} {fmt(s('P_IB','Bias')):>8}")

    print()
    print("=" * 110)
    print("KEY PARAMETERS (best seed) + seed spread [min..max]  (identifiability proxy)")
    print("=" * 110)
    keys = ["Y_xs", "qP_max", "K_S", "theta_0", "theta_T", "a_qS", "m_s"]
    hdr = f"{'model':6} {'fold':4} " + " ".join(f"{k:>20}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        bp = r["best_params"]
        parts = []
        for k in keys:
            if k in bp:
                lo, hi = _seed_spread(r["all_seeds"], k)
                parts.append(f"{fmt(bp[k],3)}[{fmt(lo,2)}..{fmt(hi,2)}]".rjust(20))
            else:
                parts.append(" ".rjust(20))
        print(f"{r['model']:6} {r['fold']:4} " + " ".join(parts))

    print()
    print("=" * 110)
    print("FINAL IB FRACTION (held-out): observed vs predicted  [+ biomass / total product final]")
    print("=" * 110)
    for r in recs:
        ep = r.get("test_endpoints") or []
        if not ep:
            continue
        print(f"-- {r['model']} fold {r['fold']} (held-out {r['test_run_ids']}) --")
        for e in ep:
            print(f"   {e['run_id']}: IBfrac obs={fmt(e['IBfrac_final_obs'],3)} pred={fmt(e['IBfrac_final_pred'],3)}  |  "
                  f"X final obs={fmt(e['X_final_obs'],1)} pred={fmt(e['X_final_pred'],1)}  |  "
                  f"Ptot final obs={fmt(e['Ptot_final_obs'],2)} pred={fmt(e['Ptot_final_pred'],2)}")
    print()
    print("Bias sign: positive = model UNDERpredicts (obs > pred). R2 omitted (unreliable for near-zero pools).")


if __name__ == "__main__":
    main(sys.argv[1:])
