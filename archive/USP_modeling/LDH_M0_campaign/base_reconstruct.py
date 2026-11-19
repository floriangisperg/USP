"""Reconstruct the missing NH4OH base volume for LDH runs.

The base (NH4OH, 6.95 M) is the one volume term the consolidated pipeline drops
(total_added_g = 0 -> base = 0). It is not weighed for LDH, but the raw bioreactor
CSV logs the base pump (Pump2.Duration, cumulative). We reconstruct the base
*trajectory* from that signal and scale it to a per-run total that is bounded:

  - N-floor (lower bound): the stoichiometric NH3 needed to fix the N in the final
    biomass + product, minus the batch (NH4)2HPO4 pulse. The feed is N-free, so all
    post-batch N must enter via base -> base >= N-floor. (scfv shows actual base is
    ~1-2x this, the excess being acid-neutralization.)
  - upper bound: 2x the N-floor, motivated by the scfv base/N-demand ratio (~1.1-1.9x;
    ferm_06 qs0.4/T30 = 1.9x). The scfv k-calibration itself does NOT transfer to LDH
    (different reactors: scfv R1/R3/R4 vs LDH MF2-R6; even the feed-pump calibration
    differs), so we bracket rather than point-estimate.

Volume correction: V_recon(t) = Volume_corrected(t) + base(t), with base(t) shaped by
the normalized Pump2.Duration trace (preserves WHEN base was added within each run).
Observations (X, P g/L) are assay values and do not change; only the model forcing
(qS = F*S_IN/(X*V), D = F/V) is affected.
"""

from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# This folder was moved out of USP/examples/LDH/modeling_playground (Nov 2026).
# Raw bioreactor exports and the pump2 pickle cache still live with the LDH
# analyses in the USP repo; the cache is mtime-keyed against those raw files.
LDH_DIR = Path(
    r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH"
)

# Physical constants (from the run/media config).
NFRAC_X = 0.1264          # g N / g biomass (CH1.77N0.24, 26.565 g/C-mol)
NFRAC_P = 0.1647          # g N / g LDH product (run config nitrogen_content_gN_gP)
NPERML_BASE = 6.95 * 14 / 1000   # g N per mL of 6.95 M NH4OH = 0.0973
BATCH_N_GL = 4.0 * 28 / 132.06   # (NH4)2HPO4 4 g/L -> g N/L in batch medium = 0.848
V0_DEFAULT = 1.1          # L, initial batch volume

# Base-level multipliers on the N-floor for the bracket (acid-neutralization excess).
LEVELS = {"none": 0.0, "floor": 1.0, "mid": 1.5, "upper": 2.0}


def _read_raw_pump2(run_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_hours, pump2_duration) from a run's raw bioreactor file.

    Cached to pickle (across processes) + in-memory (within a process), keyed by the
    raw file's mtime, so the default pipeline (which now calls this on every load)
    stays fast: xlsx reads are ~seconds each, the cache makes repeat loads instant.
    """
    if run_id in _PUMP2_MEM:
        return _PUMP2_MEM[run_id]
    files = glob.glob(str(LDH_DIR / run_id / "data" / "Rawdata*"))
    if not files:
        raise FileNotFoundError(f"no raw file for {run_id}")
    f = files[0]
    mtime = os.path.getmtime(f)
    pkl = _load_pump2_pkl()
    entry = pkl.get(run_id)
    if entry and abs(entry.get("mtime", 0) - mtime) < 1e-3:
        t, p2 = entry["t"], entry["p2"]
    else:
        if f.endswith(".csv"):
            raw = pd.read_csv(f, sep=";", skiprows=1, low_memory=False)
        else:
            raw = pd.read_excel(f, skiprows=1)
        t = pd.to_numeric(raw["Batch Time, sec"], errors="coerce").to_numpy(float) / 3600.0
        p2col = [c for c in raw.columns if "Pump2.Duration" in c][0]
        p2 = pd.to_numeric(raw[p2col], errors="coerce").to_numpy(float)
        m = np.isfinite(t) & np.isfinite(p2)
        t, p2 = t[m], p2[m]
        pkl[run_id] = {"mtime": mtime, "t": t, "p2": p2}
        _save_pump2_pkl(pkl)
    _PUMP2_MEM[run_id] = (t, p2)
    return t, p2


_PUMP2_MEM: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_PUMP2_PKL = LDH_DIR / "pump2_raw_cache.pkl"


def _load_pump2_pkl() -> dict:
    try:
        with open(_PUMP2_PKL, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return {}


def _save_pump2_pkl(d: dict) -> None:
    try:
        with open(_PUMP2_PKL, "wb") as fh:
            pickle.dump(d, fh)
    except Exception:
        pass


def _pump2_shape_on_tonline(run_id: str, t_online: np.ndarray) -> np.ndarray:
    """Normalized cumulative base fraction on the run's t_online grid in [0,1].

    base(t)/base_total = (Pump2(t) - Pump2(t_raw_min)) / (Pump2 range).
    Carries the batch-phase base into V at t0 (Pump2 is cumulative from t=0).
    """
    t_raw, p2 = _read_raw_pump2(run_id)
    lo, hi = float(np.nanmin(t_raw)), float(np.nanmax(t_raw))
    # Clamp t_online into the raw span, then interpolate Pump2 onto it.
    tc = np.clip(t_online, lo, hi)
    p2_on = np.interp(tc, t_raw, p2)
    p2_min, p2_max = float(p2[0]), float(p2[-1])
    span = p2_max - p2_min
    if span <= 0:
        return np.zeros_like(t_online)
    return (p2_on - p2_min) / span


def n_floor_ml(run: dict) -> float:
    """N-balance lower bound on base volume [mL] for one run."""
    obs = run["obs"]
    Xf = float(obs["X"][-1])
    Pf = float(obs["P_sol"][-1] + obs["P_IB"][-1])
    Vf = float(run["V_L"][-1])  # current (Volume corrected) final volume, L
    n_demand = NFRAC_X * Xf * Vf + NFRAC_P * Pf * Vf        # g N in final X+P
    n_batch = BATCH_N_GL * V0_DEFAULT                        # g N from batch (NH4)2HPO4
    return max((n_demand - n_batch) / NPERML_BASE, 0.0)


def reconstruct_v(run: dict, total_base_ml: float) -> np.ndarray:
    """Corrected reactor volume [L] on run['t_online']: V_corrected + base(t)."""
    t_online = np.asarray(run["t_online"], dtype=float)
    shape = _pump2_shape_on_tonline(run["run_id"], t_online)
    base_ml = total_base_ml * shape                      # cumulative mL added
    v_corr = np.asarray(run["V_L"], dtype=float)         # current Volume corrected, L
    return v_corr + base_ml / 1000.0


def apply_volume(bundle: dict, level: str) -> dict:
    """Return a bundle copy with run['V_L'] overridden by the reconstructed volume.

    `level` in {none, floor, mid, upper}; multiplies each run's N-floor.
    'none' leaves volumes unchanged (base=0, the current model)."""
    mult = LEVELS[level]
    import copy
    new = copy.deepcopy(bundle)
    for run in list(new["train"]) + list(new["test"]):
        if mult == 0.0:
            continue
        total = n_floor_ml(run) * mult
        run["V_L"] = reconstruct_v(run, total)
    return new


def report_n_floors(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        nf = n_floor_ml(run)
        rows.append({"run": run["run_id"], "n_floor_ml": round(nf, 1),
                     "V_final_L": round(float(run["V_L"][-1]), 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import data as D
    tr, te = D.split_for_fold("full")
    b = D.load_dataset(tr, te)
    runs = b["train"]
    print(report_n_floors(runs).to_string(index=False))
    # sanity: shape for F13
    r13 = next(r for r in runs if r["run_id"] == "F13")
    sh = _pump2_shape_on_tonline("F13", np.asarray(r13["t_online"], float))
    print(f"\nF13 pump2_shape: t0={sh[0]:.3f} mid={sh[len(sh)//2]:.3f} end={sh[-1]:.3f} "
          f"(monotone incr: {bool(np.all(np.diff(sh) >= -1e-9))})")
    print(f"F13 V(t0): current={r13['V_L'][0]:.3f} L -> +floor({n_floor_ml(r13):.0f}mL)={reconstruct_v(r13, n_floor_ml(r13))[0]:.3f} L")
