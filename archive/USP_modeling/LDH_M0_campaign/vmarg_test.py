"""H2 — volume-marginalized fitting (2026-09-05 overnight program).

The ±50% base-volume bracket has always been COMMITTED to (N-floor default); the
base_volume_refit bracket table showed params move with it (Y_xs 0.351->0.368).
Hypothesis: part of the cross-fold parameter instability (a_qS swinging 0.5->4,
theta_T at bounds) is bracket noise; fitting against predictions AVERAGED over
the bracket (marginalizing the nuisance) should stabilize parameters and improve
transfer.

Implementation: duplicate each train/test run at the two bracket ends (N-floor,
2x N-floor; base_reconstruct pump-shaped) and fit the adopted model on the
duplicated bundle -- the residual sum then averages over the bracket. Obs are
assay values (unchanged); only V_L differs.

Compare per fold: parameters (esp. a_qS, theta_T) and pooled held-out RMSE vs
the standard N-floor fits (results/M0_aload_qPTexp/).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import base_reconstruct as BR
import data as D
import fit as F
from fit import gather_predictions

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "vmarg"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "M0_aload_qPTexp"
FOLDS = ["A", "B", "C"]
MULTS = (1.0, 2.0)  # bracket ends: N-floor, 2x N-floor


def marginalized_bundle(train_ids: List[str], test_ids: List[str]) -> dict:
    D.CORRECT_BASE_VOLUME = False
    try:
        bundle = D.load_dataset(train_ids, test_ids)
    finally:
        D.CORRECT_BASE_VOLUME = True
    for part in ("train", "test"):
        runs2 = []
        for run in bundle[part]:
            for mult in MULTS:
                r2 = copy.deepcopy(run)
                try:
                    total = BR.n_floor_ml(r2)
                    r2["V_L"] = BR.reconstruct_v(r2, total * mult)
                except Exception:
                    pass  # leave V unchanged if pump trace unavailable
                runs2.append(r2)
        bundle[part] = runs2
    return bundle


def pooled(obs: np.ndarray, pred: np.ndarray):
    ok = np.isfinite(obs) & np.isfinite(pred)
    e = obs[ok] - pred[ok]
    return float(np.sqrt(np.mean(e**2))), float(np.mean(e))


def main() -> None:
    lines = ["# H2 — volume-marginalized fits (bracket ends as duplicated runs)", ""]
    for fold in FOLDS:
        train_ids, test_ids = D.split_for_fold(fold)
        bundle = marginalized_bundle(train_ids, test_ids)
        res = F.fit_model(MODEL, bundle, seed=42, n_starts=3)
        print(f"fold {fold}: cost {res.cost:.4f} params={ {k: round(v,4) for k,v in res.params.items()} }", flush=True)
        # Held-out: predict the REAL (N-floor) test runs with the marginalized fit.
        real_test = D.load_dataset(train_ids, test_ids)["test"]
        preds = gather_predictions(MODEL, res.params, real_test, bundle["c_fold"])
        row: Dict[str, tuple] = {}
        for s in ("X", "P_sol", "P_IB"):
            o = np.concatenate([np.asarray(r["run"]["obs"][s], dtype=float) for r in preds])
            p = np.concatenate([np.asarray(r["pred"]["obs"][s], dtype=float) for r in preds])
            row[s] = pooled(o, p)
        # Reference: stored standard fit.
        ref_dir = HERE / "results" / "M0_aload_qPTexp" / f"holdout_{''.join(test_ids)}"
        ref = json.loads((ref_dir / "metrics.json").read_text())
        lines.append(f"## fold {fold} (held-out {', '.join(test_ids)})")
        lines.append(f"- params: { {k: round(v,4) for k,v in res.params.items()} }")
        for s in ("X", "P_sol", "P_IB"):
            ref_rmse = ref["test"]["per_state"][s]["RMSE"]
            ref_bias = ref["test"]["per_state"][s]["Bias"]
            lines.append(f"- {s}: vmarg RMSE {row[s][0]:.3f} (bias {row[s][1]:+.3f})"
                         f" vs standard {ref_rmse:.3f} ({ref_bias:+.3f})")
        lines.append("")
        (OUT / "vmarg_partial.md").write_text("\n".join(lines) + "\n")
    (OUT / "vmarg_test.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {OUT/'vmarg_test.md'}")


if __name__ == "__main__":
    main()
