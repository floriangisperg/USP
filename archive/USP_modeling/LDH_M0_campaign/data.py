"""Data loading for the LDH mechanistic modeling playground.

Reads the consolidated processed-modeling Excel produced by
``examples/LDH/consolidate_results.py`` and turns it into per-run model-ready
arrays: dense online forcing inputs (feed_rate, V, T, is_induced) plus sparse
offline observations (X, P_sol, P_IB at sample times) and dense online offgas
observations (CER, OUR on a downsampled grid).

Mirrors the hybrid model's grouped train/test split and its leakage-safe
data-derived constants (train-only ranges, reference load C_fold).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import base_reconstruct as BR  # NH4OH base-volume reconstruction (N-floor + pump signal)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# Moved here (from USP/examples/LDH/modeling_playground) in Nov 2026; the
# consolidated results file is bundled under data/.
DEFAULT_DATA_FILE = SCRIPT_DIR / "data" / "LDH_consolidated_results_processed.xlsx"
DATA_SHEET = "All_Processed"

# All 11 FCCD runs.
ALL_RUN_IDS: List[str] = [
    "F07", "F09", "F10", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
]

# Grouped leave-one-30C-regime-out folds (identical to the hybrid model).
# Fold B always holds all three center replicates out TOGETHER.
#
# LOGO_*: complete-design leave-one-group-out folds (Tier-2 validation, mirroring
# the Fermentation paper's methods): nine hold-out groups cover all eleven runs —
# the 30 °C isotherm folds A/B/C (center triplicate always held out together)
# plus the six single-run groups below, which hold out the temperature-edge
# fermentations one at a time. Those regimes are partitioning-saturated (IBfrac
# pinned at 0 or 1), so they probe endpoint generalization, not the partition.
FOLDS: Dict[str, List[str]] = {
    "full": [],
    "A": ["F17"],
    "B": ["F13", "F14", "F15"],
    "C": ["F18"],
    "LOGO_F07": ["F07"],
    "LOGO_F09": ["F09"],
    "LOGO_F10": ["F10"],
    "LOGO_F12": ["F12"],
    "LOGO_F16": ["F16"],
    "LOGO_F19": ["F19"],
}

# Known feed glucose concentration [g/L] (mechanistic constant, not fitted).
S_IN = 400.0

# Gas downsampling grid spacing [h]. The raw online grid is ~1 min; we keep
# ~15-min knots for the offgas observations so the optimiser stays tractable.
GAS_DT = 0.25

# Feed-rate cleaning (ported from the hybrid model's m1_loadpartition.py).
FEED_RATE_MEDIAN_WINDOW = 5
FEED_RATE_MA_WINDOW = 1

# Apply the N-floor NH4OH base-volume reconstruction to the reactor volume by default.
# Base is not logged for LDH (total_added_g = 0 -> base = 0, V under-counted); the
# reconstruction adds the N-balance lower bound (pump-shaped) to V_corrected. See
# base_reconstruct.py and HANDOFF (2026-07-24). Set False to recover the base=0 model.
CORRECT_BASE_VOLUME = True


def _find_col(df: pd.DataFrame, target: str) -> str:
    """Resolve a column name, tolerating degree-symbol encoding differences.

    Matches exact, then the base name before ' ['.
    """
    if target in df.columns:
        return target
    base = target.split(" [")[0].lower()
    for c in df.columns:
        if c.split(" [")[0].lower() == base:
            return c
    raise KeyError(f"Column not found (base '{base}'). Available: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Feed-rate cleaning (ported from the hybrid model)
# ---------------------------------------------------------------------------
def _median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w <= 1:
        return x
    if w % 2 == 0:
        raise ValueError("Median filter window must be odd.")
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=float)
    for i in range(out.size):
        out[i] = float(np.median(xp[i : i + w]))
    return out


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=float) / float(w)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xp, kernel, mode="valid")
    return y[: x.shape[0]]


def _clean_feed_rate(t_h: np.ndarray, feed_mL_cumul: np.ndarray) -> np.ndarray:
    """Forward-diff the cumulative feed [mL], median-smooth, clip >= 0 [mL/h]."""
    feed_rate = np.zeros_like(feed_mL_cumul, dtype=float)
    if len(t_h) > 1:
        dt = np.diff(t_h)
        d_feed = np.diff(feed_mL_cumul)
        feed_rate[:-1] = np.where(dt > 0, d_feed / dt, 0.0)
        feed_rate[-1] = feed_rate[-2] if len(feed_rate) > 1 else 0.0
    feed_rate = np.maximum(feed_rate, 0.0)
    if FEED_RATE_MEDIAN_WINDOW > 1:
        feed_rate = _median_filter_1d(feed_rate, FEED_RATE_MEDIAN_WINDOW)
    if FEED_RATE_MA_WINDOW > 1:
        feed_rate = _moving_average_1d(feed_rate, FEED_RATE_MA_WINDOW)
    return np.maximum(feed_rate, 0.0)


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------
def _extract_run(df: pd.DataFrame, run_id: str) -> dict:
    """Build the model-ready arrays for one run.

    Returns a dict with:
      run_id, t0, tend, X0
      t_online, feed_rate_L_h, V_L, temp, is_induced  (dense forcing grid)
      t_obs, obs: {X, P_sol, P_IB}  (sparse sample-point observations)
      t_gas, gas: {CER, OUR}  (downsampled online offgas)
      T_induction (median T while induced)
    """
    c_time = _find_col(df, "Process time [h]")
    c_temp = _find_col(df, "Temperature [°C]")
    c_feed = _find_col(df, "Feed [mL]")
    c_vol = _find_col(df, "Volume corrected [mL]")
    c_ind = _find_col(df, "Is induced [-]")
    c_indt = _find_col(df, "Induction time [h]")
    c_sample = _find_col(df, "is_sample")
    c_x = _find_col(df, "Biomass [g/L]")
    c_psol = _find_col(df, "Protein soluble [g/L]")
    c_pib = _find_col(df, "Protein IB [g/L]")
    c_cer = _find_col(df, "CER [mmol/L/h]")
    c_our = _find_col(df, "OUR [mmol/L/h]")
    c_po2 = _find_col(df, "pO2 [%]")

    r = df[df["run_name"] == run_id].sort_values(c_time).reset_index(drop=True)

    t = r[c_time].to_numpy(dtype=float)
    feed_mL = r[c_feed].fillna(0.0).to_numpy(dtype=float)
    V_mL = r[c_vol].ffill().bfill().to_numpy(dtype=float)
    temp = r[c_temp].ffill().bfill().to_numpy(dtype=float)
    is_ind = (r[c_ind].fillna(0).to_numpy(dtype=float) > 0.5).astype(float)
    ind_time = r[c_indt].fillna(0.0).to_numpy(dtype=float)
    is_sample = r[c_sample].fillna(False).to_numpy(dtype=bool)
    po2 = r[c_po2].ffill().bfill().fillna(40.0).to_numpy(dtype=float)

    feed_rate_mLh = _clean_feed_rate(t, feed_mL)
    feed_rate_L_h = feed_rate_mLh / 1000.0
    V_L = V_mL / 1000.0

    # Simulation window: start at the first sample (≈ feed onset), where the
    # feed-supply model is valid. Batch-phase growth is absorbed into X0.
    sample_mask = is_sample & np.isfinite(r[c_x].to_numpy(dtype=float))
    if not sample_mask.any():
        raise ValueError(f"{run_id}: no biomass samples.")
    t0 = float(t[sample_mask][0])
    X0 = float(r[c_x].to_numpy(dtype=float)[sample_mask][0])
    tend = float(t[-1])
    win = (t >= t0) & (t <= tend)

    # Sparse observations (sample points).
    sm = is_sample & win
    t_obs = t[sm]
    obs = {
        "X": r[c_x].to_numpy(dtype=float)[sm],
        "P_sol": r[c_psol].to_numpy(dtype=float)[sm],
        "P_IB": r[c_pib].to_numpy(dtype=float)[sm],
    }

    # Downsampled gas grid within the simulation window (keep finite values).
    tg = t[win]
    cer = r[c_cer].to_numpy(dtype=float)[win]
    our = r[c_our].to_numpy(dtype=float)[win]
    finite = np.isfinite(cer) & np.isfinite(our)
    tg, cer, our = tg[finite], cer[finite], our[finite]
    if len(tg) > 0:
        bins = np.floor((tg - t0) / GAS_DT).astype(np.int64)
        keep = np.concatenate(([True], np.diff(bins) != 0))
        tg, cer, our = tg[keep], cer[keep], our[keep]
    t_gas = tg
    # RQ at each gas point (for trustworthy-window masking; CER/OUR finite here).
    with np.errstate(divide="ignore", invalid="ignore"):
        rq = cer / np.where(np.abs(our) > 1e-6, our, np.nan)
    gas = {"CER": cer, "OUR": our, "rq": rq}

    T_ind = float(np.nanmedian(temp[is_ind > 0.5])) if (is_ind > 0.5).any() else float(np.nanmedian(temp))

    run = {
        "run_id": run_id,
        "t0": t0,
        "tend": tend,
        "X0": X0,
        "t_online": t[win],
        "feed_rate_L_h": feed_rate_L_h[win],
        "V_L": V_L[win],
        "temp": temp[win],
        "is_induced": is_ind[win],
        "ind_time": ind_time[win],
        "po2": po2[win],
        "t_obs": t_obs,
        "obs": obs,
        "t_gas": t_gas,
        "gas": gas,
        "T_induction": T_ind,
    }
    if CORRECT_BASE_VOLUME:
        run = _apply_base_volume(run)
    return run


def _apply_base_volume(run: dict) -> dict:
    """Add the N-floor NH4OH base volume (pump-shaped) to the reactor volume.

    N-floor is computed from the uncorrected V (no circularity: it is a ~5-10 %
    correction). Falls back to the uncorrected volume if the raw pump trace is
    unavailable.
    """
    try:
        total = BR.n_floor_ml(run)
        run["V_L"] = BR.reconstruct_v(run, total)
        run["base_added_ml"] = total
    except Exception:
        pass  # leave V_L unchanged
    return run


def load_dataset(
    train_run_ids: List[str],
    test_run_ids: List[str],
    data_file: Optional[Path] = None,
) -> dict:
    """Load runs and compute train-only leakage-safe constants.

    Returns a bundle with 'train' and 'test' run dicts plus:
      range_train: {state: (min, max, range)} over TRAIN observations, for
        range-weighted residuals and metrics normalization.
      c_fold: train-only median observed specific total production rate.
    """
    data_file = Path(data_file) if data_file else DEFAULT_DATA_FILE
    df = _load_df_cached(data_file)

    def _runs(ids):
        return [_extract_run(df, r) for r in ids]

    train = _runs(train_run_ids)
    test = _runs(test_run_ids)

    # Train-only ranges for normalization (sparse states + gas channels).
    range_train: Dict[str, List[float]] = {}
    for state in ("X", "P_sol", "P_IB"):
        vals = np.concatenate([np.asarray(rd["obs"][state]) for rd in train])
        vals = vals[np.isfinite(vals)]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        range_train[state] = [lo, hi, hi - lo]
    for state, key in (("CER", "CER"), ("OUR", "OUR")):
        vals = np.concatenate([np.asarray(rd["gas"][key]) for rd in train])
        vals = vals[np.isfinite(vals)]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        range_train[state] = [lo, hi, hi - lo]

    c_fold = _estimate_c_fold(train)

    return {
        "train": train,
        "test": test,
        "range_train": range_train,
        "c_fold": c_fold,
    }


def _load_df_cached(data_file: Path) -> pd.DataFrame:
    """Read the All_Processed sheet, caching to pickle for fast repeat loads.

    openpyxl reads of the 7 MB xlsx take ~60 s; pickle is ~0.5 s. The cache is
    invalidated when the source xlsx is newer. Pickle is used (not parquet) so no
    extra dependency (pyarrow/fastparquet) is required.
    """
    cache = data_file.with_suffix(".pkl")
    if cache.exists() and cache.stat().st_mtime >= data_file.stat().st_mtime:
        try:
            return pd.read_pickle(cache)
        except Exception:
            pass
    df = pd.read_excel(data_file, sheet_name=DATA_SHEET)
    try:
        df.to_pickle(cache)
    except Exception:
        pass
    return df


def _estimate_c_fold(train_runs: List[dict]) -> float:
    """Train-only median observed specific total production rate [gP/gX/h].

    Used only as a fixed scale inside the optional load-dependent partition
    term (absorbed into theta_0). Computed from TRAIN runs only to avoid leakage.
    """
    qps: List[float] = []
    for rd in train_runs:
        t = rd["t_obs"]
        p = rd["obs"]["P_sol"] + rd["obs"]["P_IB"]
        x = rd["obs"]["X"]
        ok = np.isfinite(t) & np.isfinite(p) & np.isfinite(x) & (x > 1e-3) & (p > 1e-6)
        if ok.sum() < 2:
            continue
        t, p, x = t[ok], p[ok], x[ok]
        dt = np.diff(t)
        dpdt = np.diff(p) / np.where(dt > 0, dt, np.nan)
        x_mid = 0.5 * (x[:-1] + x[1:])
        qp = dpdt / x_mid
        qp = qp[np.isfinite(qp) & (qp > 0)]
        qps.extend(qp.tolist())
    return float(max(np.median(qps), 1e-6)) if qps else 0.005


def split_for_fold(fold: str) -> tuple[List[str], List[str]]:
    """Return (train_ids, test_ids) for a named fold."""
    test = list(FOLDS[fold])
    train = [r for r in ALL_RUN_IDS if r not in test]
    return train, test
