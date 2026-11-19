"""Evaluation metrics for the LDH mechanistic playground.

Conventions follow the hybrid model's test_metrics.md:
  - error definition:  e = y - y_hat  (observed - predicted)
  - bias = mean(e)     (positive => model underpredicts)
  - range normalization uses TRAIN-derived ranges:  NRMSE_range = RMSE / range_train
  - nISE_range = ( (1/T) * trapz(e^2, t) ) / range_train^2
  - NLL: exact Gaussian NLL with a cv + floor noise model

Aggregation:
  - per (run, state) cell: micro over timepoints
  - per-state summary: macro over runs (each run equal weight)
  - overall (PRIMARY): macro over (run, state) cells

R^2 is reported but flagged unreliable for near-zero pools (P_sol / P_IB at
25/35 C), where the observed variance is tiny and R^2 explodes. Range-weighted
RMSE/MAE and bias are the primary tools, per the identifiability-first plan.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# Noise model for NLL reporting (cv relative to magnitude + absolute floor).
NOISE_CV: Dict[str, float] = {"X": 0.06, "P_sol": 0.085, "P_IB": 0.085, "CER": 0.05, "OUR": 0.05}
NOISE_FLOOR: Dict[str, float] = {"X": 0.01, "P_sol": 0.01, "P_IB": 0.01, "CER": 1.0, "OUR": 1.0}

METRIC_NAMES = ["RMSE", "MAE", "Bias", "NRMSE_range", "NMAE_range", "nISE_range", "R2", "NLL"]


def _sigma(y: np.ndarray, state: str) -> np.ndarray:
    cv = NOISE_CV.get(state, 0.08)
    floor = NOISE_FLOOR.get(state, 0.01)
    return np.sqrt((cv * np.abs(y)) ** 2 + floor**2)


def cell_metrics(
    t: np.ndarray,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    range_state: float,
    state: str,
) -> Dict[str, float]:
    """Metrics for one (run, state) cell, micro over finite timepoints."""
    t = np.asarray(t, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(t)
    n = int(finite.sum())
    out: Dict[str, float] = {"n_pts": n, "T": float(t[finite].max() - t[finite].min()) if n >= 2 else 0.0}
    if n == 0:
        for m in METRIC_NAMES:
            out[m] = float("nan")
        return out
    y_o = y_obs[finite]
    y_p = y_pred[finite]
    t_f = t[finite]
    e = y_o - y_p
    rng = range_state if range_state and range_state > 1e-12 else 1.0

    out["RMSE"] = float(np.sqrt(np.mean(e**2)))
    out["MAE"] = float(np.mean(np.abs(e)))
    out["Bias"] = float(np.mean(e))
    out["NRMSE_range"] = out["RMSE"] / rng
    out["NMAE_range"] = out["MAE"] / rng
    # Integrated squared error (needs sorted time).
    order = np.argsort(t_f)
    ts, es = t_f[order], e[order] ** 2
    if n >= 2:
        Tspan = ts[-1] - ts[0]
        Tspan = Tspan if Tspan > 1e-12 else 1.0
        # np.trapezoid (numpy>=2.0) replaces np.trapz; support both.
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        ise = _trapz(es, ts) / Tspan
    else:
        ise = float(es[0])
    out["nISE_range"] = ise / rng**2
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y_o - np.mean(y_o)) ** 2))
    out["R2"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    sigma = _sigma(y_o, state)
    out["NLL"] = float(np.sum(0.5 * (e / sigma) ** 2 + 0.5 * np.log(2.0 * np.pi * sigma**2)))
    return out


def _macro(values: Sequence[float]) -> float:
    v = [x for x in values if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def aggregate(cells: Dict[tuple, Dict[str, float]]) -> Dict[str, object]:
    """Aggregate a dict keyed by (run_id, state) -> cell metrics.

    Returns overall (macro over cells, primary), per-state (macro over runs),
    and the raw cells.
    """
    by_state: Dict[str, List[float]] = {m: [] for m in METRIC_NAMES}
    overall: Dict[str, List[float]] = {m: [] for m in METRIC_NAMES}
    for (_run, state), mets in cells.items():
        for m in METRIC_NAMES:
            overall[m].append(mets[m])
            if m == "Bias":
                by_state.setdefault(f"{state}", [])
        # per-state macro: collect per (state, run) is already one value per cell
    # per-state macro over runs
    per_state: Dict[str, Dict[str, float]] = {}
    state_runs: Dict[str, Dict[str, List[float]]] = {}
    for (run, state), mets in cells.items():
        d = state_runs.setdefault(state, {m: [] for m in METRIC_NAMES})
        for m in METRIC_NAMES:
            d[m].append(mets[m])
    for state, d in state_runs.items():
        per_state[state] = {m: _macro(d[m]) for m in METRIC_NAMES}
    overall_macro = {m: _macro(overall[m]) for m in METRIC_NAMES}
    return {"overall": overall_macro, "per_state": per_state, "cells": cells}


def render_metrics_md(
    title: str,
    cells: Dict[tuple, Dict[str, float]],
    range_train: Dict[str, Sequence[float]],
    model_label: str = "",
) -> str:
    """Render a markdown report mirroring the hybrid test_metrics.md layout."""
    agg = aggregate(cells)
    lines: List[str] = []
    lines.append(f"=== {title} ===")
    if model_label:
        lines.append(f"Model: {model_label}")
    lines.append("")
    lines.append("Error definition: e = y - y_hat (observed - predicted). Bias positive => underprediction.")
    lines.append("Normalization: NRMSE_range = RMSE / range_train[state]. R2 flagged unreliable for near-zero pools.")
    lines.append("")
    lines.append("## TRAIN normalization constants")
    lines.append("| State | y_min_train | y_max_train | range_train |")
    lines.append("|---|---:|---:|---:|")
    for s in ("X", "P_sol", "P_IB", "CER", "OUR"):
        if s in range_train:
            lo, hi, rng = range_train[s]
            lines.append(f"| {s} | {lo:.4g} | {hi:.4g} | {rng:.4g} |")
    lines.append("")
    lines.append("## Overall: macro over (run,state) cells (PRIMARY)")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for m in METRIC_NAMES:
        lines.append(f"| {m} | {agg['overall'][m]:.6g} |")
    lines.append("")
    lines.append("## Per-state summary: macro over runs")
    lines.append("| State | " + " | ".join(METRIC_NAMES) + " |")
    lines.append("|---|" + "---:|" * len(METRIC_NAMES))
    for s in ("X", "P_sol", "P_IB", "CER", "OUR"):
        if s in agg["per_state"]:
            row = agg["per_state"][s]
            lines.append("| " + s + " | " + " | ".join(f"{row[m]:.4g}" for m in METRIC_NAMES) + " |")
    lines.append("")
    lines.append("## Per-run details (micro over timepoints within each state)")
    for (run, state), mets in cells.items():
        if mets["n_pts"] == 0:
            continue
        lines.append(f"### {run} / {state}  (n={mets['n_pts']})")
        lines.append("| " + " | ".join(["Metric"] + METRIC_NAMES) + " |")
        lines.append("|" + "---:|" * (len(METRIC_NAMES) + 1))
        lines.append("| value | " + " | ".join(f"{mets[m]:.4g}" for m in METRIC_NAMES) + " |")
        lines.append("")
    return "\n".join(lines)
