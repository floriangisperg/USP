"""Posterior-ensemble predictive bands (Laplace) for the LDH playground (2026-09-04).

The paper's hybrid reports ensemble mean +/- spread over 8 seeds; the parametric
model so far reports only MAP curves. This adds the honest analogue: at each
fold's stored best fit, a Laplace approximation of the parameter posterior --

    cov = s^2 (J^T J)^{-1},   s^2 = r^T r / (n - p)

with J the finite-difference Jacobian of the *actual weighted residuals*
(fit.residuals) at the optimum (pinv for the sloppy/rank-deficient directions),
200 parameter samples clipped to the bounds, held-out predictions per sample.

Outputs per fold: held-out RMSE of the ensemble MEAN trajectory vs the MAP
prediction (does posterior-mean prediction beat the MAP point?), and 90 percent
predictive bands per state -> results/posterior_ensemble/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import data as D
import fit as F
from models import PARAM_SPECS, unpack


def split_tag(fold: str) -> str:
    """fold -> holdout_<ids> output-dir tag (same rule as run_playground)."""
    _, test_ids = D.split_for_fold(fold)
    return f"holdout_{''.join(test_ids)}" if test_ids else "full"

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "M0_aload_qPTexp"
OUT = HERE / "results" / "posterior_ensemble"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "M0_aload_qPTexp"
N_SAMP = 200
SEED = 42

FOLDS = ["A", "B", "C", "LOGO_F07", "LOGO_F09", "LOGO_F10", "LOGO_F12", "LOGO_F16", "LOGO_F19"]
STATES = ("X", "P_sol", "P_IB")


def laplace_samples(model: str, params: Dict[str, float], bundle: dict, rng) -> List[Dict[str, float]]:
    """Sample the Laplace posterior at the given optimum (finite-difference J)."""
    channels = F._channels_for(model)
    names = list(PARAM_SPECS[model].keys())
    lo = {k: PARAM_SPECS[model][k][1] for k in names}
    hi = {k: PARAM_SPECS[model][k][2] for k in names}

    def _res(pvec: Dict[str, float]) -> np.ndarray:
        vec = np.array([pvec[k] for k in PARAM_SPECS[model].keys()], dtype=float)
        return F.residuals(vec, model, bundle, channels)

    r0 = _res(params)
    n, p = r0.size, len(names)
    s2 = float(r0 @ r0) / max(n - p, 1)

    # Finite-difference Jacobian (central differences; respect bounds).
    J = np.zeros((n, p))
    for j, nm in enumerate(names):
        h = max(1e-5, 1e-4 * abs(params[nm]))
        up, dn = dict(params), dict(params)
        up[nm] = min(params[nm] + h, hi[nm])
        dn[nm] = max(params[nm] - h, lo[nm])
        h_eff = up[nm] - dn[nm]
        if h_eff <= 0:
            J[:, j] = 0.0
            continue
        J[:, j] = (_res(up) - _res(dn)) / h_eff

    cov = s2 * np.linalg.pinv(J.T @ J, rcond=1e-10)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 0.0, None)
    L = eigvec @ np.diag(np.sqrt(eigval))

    p0 = np.array([params[nm] for nm in names])
    samples = []
    for _ in range(N_SAMP):
        z = rng.standard_normal(p)
        pv = p0 + L @ z
        pv = np.clip(pv, [lo[nm] for nm in names], [hi[nm] for nm in names])
        samples.append(unpack(model, pv))
    return samples


def main() -> None:
    import fit as _F  # gather_predictions
    rng = np.random.default_rng(SEED)
    report = ["# Posterior-ensemble predictive bands (Laplace, 200 samples/fold)", ""]
    report += ["| fold | held-out | state | MAP RMSE | ensemble-mean RMSE | mean-band width (90%) |",
               "| --- | --- | --- | --- | --- | --- |"]
    for fold in FOLDS:
        mj = json.loads((RESULTS / split_tag(fold) / "metrics.json").read_text())
        params = mj["best_params"]
        train_ids, test_ids = D.split_for_fold(fold)
        bundle = D.load_dataset(train_ids, test_ids)
        samples = laplace_samples(MODEL, params, bundle, rng)

        map_pred = _F.gather_predictions(MODEL, params, bundle["test"], bundle["c_fold"])
        ens = {r["run_id"]: {s: [] for s in STATES} for r in bundle["test"]}
        for pv in samples:
            preds = _F.gather_predictions(MODEL, pv, bundle["test"], bundle["c_fold"])
            for r_s in preds:
                for s in STATES:
                    ens[r_s["run_id"]][s].append(np.asarray(r_s["pred"]["obs"][s], dtype=float))
        for run in bundle["test"]:
            for s in STATES:
                obs = np.asarray(run["obs"][s], dtype=float)
                arr = np.array(ens[run["run_id"]][s])  # [n_samp, n_points]
                mapv = None
                for r_map in map_pred:
                    if r_map["run_id"] == run["run_id"]:
                        mapv = np.asarray(r_map["pred"]["obs"][s], dtype=float)
                def _rmse(v):
                    ok = np.isfinite(obs) & np.isfinite(v)
                    return float(np.sqrt(np.mean((obs[ok] - v[ok]) ** 2)))
                lo_b = np.percentile(arr, 5, axis=0)
                hi_b = np.percentile(arr, 95, axis=0)
                width = float(np.mean(hi_b - lo_b))
                report.append(
                    f"| {fold} | {run['run_id']} | {s} | {_rmse(mapv):.3f} | {_rmse(np.mean(arr, axis=0)):.3f} | {width:.3f} |"
                )
        # --- plots: obs vs MAP vs ensemble mean with 90% band, per held-out run ---
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots = OUT / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        for rec_p in map_pred:
            run = rec_p["run"]
            t_obs = run["t_obs"]
            fig, axes = plt.subplots(1, len(STATES), figsize=(13, 3.6), sharex=True)
            for ax, s in zip(np.atleast_1d(axes), STATES):
                obs = np.asarray(run["obs"][s], dtype=float)
                arr = np.array(ens[run["run_id"]][s])
                mapv = np.asarray(rec_p["pred"]["obs"][s], dtype=float)
                lo_b = np.percentile(arr, 5, axis=0)
                hi_b = np.percentile(arr, 95, axis=0)
                ax.fill_between(t_obs, lo_b, hi_b, alpha=0.25, label="90% band", color="#4E79A7")
                ax.plot(t_obs, mapv, "k--", lw=1.2, label="MAP")
                ax.plot(t_obs, np.mean(arr, axis=0), color="#4E79A7", lw=2, label="ens. mean")
                ax.plot(t_obs, obs, "o", color="#2ca02c", ms=6, label="obs")
                ax.set_title(f"{s}  (obs−MAP RMSE {np.sqrt(np.nanmean((obs-mapv)**2)):.2f})", fontsize=10)
                ax.set_xlabel("t [h]")
            np.atleast_1d(axes)[0].set_ylabel("g/L")
            np.atleast_1d(axes)[0].legend(fontsize=8)
            fig.suptitle(f"{run['run_id']} — fold {fold} held-out posterior ensemble (Laplace, {N_SAMP} samples)", fontsize=11)
            fig.tight_layout()
            fig.savefig(plots / f"ens_{fold}_{run['run_id']}.png", dpi=140)
            plt.close(fig)
    text = "\n".join(report) + "\n"
    (OUT / "posterior_ensemble.md").write_text(text)
    print(text)
    print(f"saved {OUT/'posterior_ensemble.md'}")


if __name__ == "__main__":
    main()
