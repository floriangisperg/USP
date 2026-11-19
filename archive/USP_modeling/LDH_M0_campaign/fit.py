"""Fitting driver: range-weighted residuals, multistart, profile likelihood.

Uses scipy.optimize.least_squares (bounded, trust-region) on range-weighted
residuals  (obs - pred) / range_train[channel].  Minimising 0.5*sum(resid^2) is
equivalent to a Gaussian NLL with the (train-derived) range scales as per-channel
weights -- this is the "range-weighted MSE" that stays meaningful for near-zero
pools (unlike R^2).

All data-derived constants (ranges, c_fold) come from TRAIN runs only (leakage-safe).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import metrics as M
import numpy as np
from models import (
    GAS_STATES,
    OBS_CHANNELS,
    PARAM_SPECS,
    SPARSE_STATES,
    simulate_run,
    unpack,
)
from scipy.optimize import least_squares

# Re-export for convenience
SPARSE_STATES = SPARSE_STATES
GAS_STATES = GAS_STATES


def _channels_for(model: str) -> Tuple[str, ...]:
    return OBS_CHANNELS[model]


# Declared log-normal priors (MAP penalty, unit weight) for weakly-informed decline
# timescales (M0_aload_qPTexp_decl). Centers come from the data: qp_decline_test
# (observed qP falls to ~0.54 of early over the ~9 h induction window -> tau ~ 12 h)
# and the decoupled-v2 LDH clock (tau ~ 0.7-2 h). sigma is in ln-space (0.7 ~ factor-2
# spread). Purpose: keep the decline scales physically plausible on the flat ridge
# where 4-5 sample points per run cannot pin them -- the failure mode that sank the
# free-fit decline mechanisms. Each penalty enters the residual vector as ONE unit-
# weight pseudo-observation: (ln(tau) - ln(center)) / sigma.
# Per-model prior exemptions (the unregularized turnover test): M0_kinpart_deg_free
# fits k_deg0 WITHOUT its prior — does the data alone pull a P_IB turnover out of
# ~1 sample interval?
PRIORS_SKIP: Dict[str, set] = {"M0_kinpart_deg_free": {"k_deg0"}}

PRIOR_LOGNORMAL: Dict[str, Tuple[float, float]] = {
    "tau_P": (12.0, 0.7),
    "tau_IBd": (2.0, 0.7),
    # Decoupled-v2 IB clock (same param name): documented LDH decline-basin tau ~ 0.7-1.5 h
    # but multimodal (0.7 / 1.6 / 4.1 at dcost < 0.06) -- the prior anchors the ridge.
    # NOTE: applies wherever tau_IB is a free parameter (also the ibtransient variants).
    "tau_IB": (1.5, 0.7),
    # Kinetic partition (M0_kinpart): proteolysis heat-shock amplification -- the molecular
    # counterpart of the CER-validated alpha (measured-product T-suppression ~9%/C above 30).
    "gamma_p": (0.12, 0.7),
    # v3 aggregate proteolysis: mass-balance deficit bounds the T30 need at ~0-0.05/h
    # (dilution alone covers the observed mild declines); center below the deficit ceiling.
    "k_deg0": (0.04, 0.7),
}


def residuals(
    vec: np.ndarray,
    model: str,
    bundle: dict,
    channels: Tuple[str, ...],
    fixed: Optional[Dict[str, float]] = None,
    runs: Optional[List[dict]] = None,
) -> np.ndarray:
    """Range-weighted residuals across the given runs (default: train runs)."""
    params = unpack(model, vec)
    if fixed:
        params.update(fixed)
    runs = bundle["train"] if runs is None else runs
    eval_gas = any(c in GAS_STATES for c in channels)
    rng = bundle["range_train"]
    parts: List[np.ndarray] = []
    for run in runs:
        try:
            pred = simulate_run(model, params, run, bundle["c_fold"], eval_gas=eval_gas)
        except Exception:
            # Penalise crashes with a large residual.
            for ch in channels:
                src = run["obs"] if ch in SPARSE_STATES else run["gas"]
                n = len(src[ch])
                parts.append(np.full(n, 1e3))
            continue
        for ch in channels:
            if ch in SPARSE_STATES:
                obs = np.asarray(run["obs"][ch], dtype=float)
                p = np.asarray(pred["obs"][ch], dtype=float)
                mask = np.isfinite(obs) & np.isfinite(p)
            else:
                obs = np.asarray(run["gas"][ch], dtype=float)
                p = np.asarray(pred["gas"][ch], dtype=float)
                mask = np.isfinite(obs) & np.isfinite(p)
                # Trustworthy-gas mask: only RQ in [0.8,1.2] (drops broken sensors,
                # e.g. F09 OUR offset / F14, and O2-transfer-limited/enriched points).
                rq = np.asarray(run["gas"].get("rq", np.full(len(obs), np.nan)), dtype=float)
                mask = mask & np.isfinite(rq) & (rq >= 0.8) & (rq <= 1.2)
            scale = rng[ch][2] if rng[ch][2] > 1e-12 else 1.0
            parts.append((obs[mask] - p[mask]) / scale)
    return np.concatenate(parts) if parts else np.zeros(1)


def _free_masks(model: str, fixed: Dict[str, float]):
    names = list(PARAM_SPECS[model].keys())
    free_idx = [i for i, n in enumerate(names) if n not in fixed]
    return names, free_idx


def fit_model(
    model: str,
    bundle: dict,
    seed: int = 42,
    n_starts: int = 3,
    fixed: Optional[Dict[str, float]] = None,
    max_nfev: int = 400,
    verbose: bool = False,
):
    """Multistart bounded least-squares fit. Returns the best scipy OptimizeResult.

    `fixed` pins named params (used by profile likelihood); those are removed
    from the optimisation vector and substituted into residuals.
    """
    fixed = fixed or {}
    channels = _channels_for(model)
    init, lo, hi = _vectors(model, fixed)
    names, free_idx = _free_masks(model, fixed)
    rng = np.random.default_rng(seed)

    def _res(free_vec):
        full = _scatter(model, free_vec, fixed)
        r = residuals(full, model, bundle, channels, fixed=fixed)
        # MAP prior penalties (unit-weight pseudo-observations, see PRIOR_LOGNORMAL).
        pv = unpack(model, full)
        pens = [
            np.atleast_1d((np.log(pv[nm]) - np.log(mu)) / sd)
            for nm, (mu, sd) in PRIOR_LOGNORMAL.items()
            if nm in pv and pv[nm] > 0.0 and (not fixed or nm not in fixed)
            and nm not in PRIORS_SKIP.get(model, set())
        ]
        return np.concatenate([r] + pens) if pens else r

    starts = [init]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.uniform(lo, hi))
    best = None
    for x0 in starts:
        x0 = np.clip(x0, lo, hi)
        try:
            r = least_squares(_res, x0, bounds=(lo, hi), method="trf",
                              max_nfev=max_nfev, x_scale="jac")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  start failed: {e}")
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is not None:
        best.x_full = _scatter(model, best.x, fixed)
        best.params = unpack(model, best.x_full)
    return best


def _vectors(model: str, fixed: Dict[str, float]):
    specs = PARAM_SPECS[model]
    names = list(specs.keys())
    init = np.array([specs[n][0] for n in names if n not in fixed], dtype=float)
    lo = np.array([specs[n][1] for n in names if n not in fixed], dtype=float)
    hi = np.array([specs[n][2] for n in names if n not in fixed], dtype=float)
    return init, lo, hi


def _scatter(model: str, free_vec: np.ndarray, fixed: Dict[str, float]) -> np.ndarray:
    """Expand a free-only vector back to the full parameter vector."""
    names = list(PARAM_SPECS[model].keys())
    full = np.zeros(len(names))
    fi = iter(free_vec)
    for i, n in enumerate(names):
        if n in fixed:
            full[i] = fixed[n]
        else:
            full[i] = next(fi)
    return full


# ---------------------------------------------------------------------------
# Evaluation -> per-cell metrics + raw predictions (for plotting)
# ---------------------------------------------------------------------------
def gather_predictions(model: str, params: Dict[str, float], runs: List[dict], c_fold: float) -> List[dict]:
    """Simulate each run; attach obs + pred arrays for all channels + trajectory."""
    channels = _channels_for(model)
    eval_gas = any(c in GAS_STATES for c in channels)
    out = []
    for run in runs:
        pred = simulate_run(model, params, run, c_fold, eval_gas=eval_gas)
        rec = {"run_id": run["run_id"], "run": run, "pred": pred}
        out.append(rec)
    return out


def cells_from_predictions(
    model: str, preds: List[dict], range_train: Dict[str, List[float]]
) -> Dict[tuple, Dict[str, float]]:
    """Build the {(run_id, state): cell_metrics} dict for the metrics module."""
    cells: Dict[tuple, Dict[str, float]] = {}
    channels = _channels_for(model)
    for rec in preds:
        run = rec["run"]
        pred = rec["pred"]
        for ch in channels:
            if ch in SPARSE_STATES:
                t = run["t_obs"]
                obs = np.asarray(run["obs"][ch], dtype=float)
                p = np.asarray(pred["obs"][ch], dtype=float)
            else:
                t = run["t_gas"]
                obs = np.asarray(run["gas"][ch], dtype=float)
                p = np.asarray(pred["gas"][ch], dtype=float)
            rng = range_train[ch][2]
            cells[(rec["run_id"], ch)] = M.cell_metrics(t, obs, p, rng, ch)
    return cells


# ---------------------------------------------------------------------------
# Profile likelihood (identifiability probe)
# ---------------------------------------------------------------------------
def profile_likelihood(
    model: str,
    bundle: dict,
    base_params: Dict[str, float],
    param_name: str,
    grid: np.ndarray,
    seed: int = 42,
) -> List[Tuple[float, float, Optional[Dict[str, float]]]]:
    """Fix param_name at each grid value, refit the rest, record the TRAIN cost.

    Returns [(value, cost, refit_params)]. A bounded, parabolic profile around
    the MLE => identifiable; flat or monotone => not identifiable.
    """
    channels = _channels_for(model)
    results = []
    for v in grid:
        fixed = {param_name: float(v)}
        r = fit_model(model, bundle, seed=seed, n_starts=1, fixed=fixed, max_nfev=200)
        if r is None:
            results.append((float(v), float("inf"), None))
        else:
            results.append((float(v), float(r.cost), dict(r.params)))
    return results
