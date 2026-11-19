"""X-conditioned product diagnostic (2026-09-04, user-driven improvement thread).

Question: how much of the held-out P_sol / P_IB error is INHERITED from biomass
error? The adopted model consumes *predicted* X twice in the product equations
(formation qP*X, and the substrate proxy qS = F*S_IN/(X*V)); the biomass RMSE
(1.45 g/L, mixed signs = the volume bracket) therefore propagates into both.

Test: re-simulate ONLY the two product ODEs with X(t) = interpolation of the
run's *measured* offline X (held-out runs use their own X samples -- the
soft-sensor framing: biomass is measured during the run, products are forecast
conditioned on it), keeping all parameters at each fold's stored baseline fit
(M0_aload_qPTexp, corrected data). Same RK4 grid and equations as models.py.

    dP_sol/dt = (1-phi) * qP * Xc - D * P_sol
    dP_IB/dt  = phi     * qP * Xc - D * P_IB
    qS  = cap(F*S_IN/(Xc*V));  qP = is_ind*qP_max*qS/(qS+K_S)*exp(-alpha*max(T-30,0))
    phi = sigmoid(theta_0 + theta_T*(T-30) + a_qS*log(qS/0.4));  D = F/V

Output: per-group + aggregate held-out RMSE/bias vs the unconditioned baseline
(results/logo_summary/), plus the T35 endpoint ordering check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import data as D
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "M0_aload_qPTexp"
OUT = HERE / "results" / "xcond"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "M0_aload_qPTexp"
S_IN = 400.0
QS_CAP = 2.0
QS_REF = 0.4
T_REF = 30.0
DT = 0.1  # match models._rk4 step

FOLDS = ["A", "B", "C", "LOGO_F07", "LOGO_F09", "LOGO_F10", "LOGO_F12", "LOGO_F16", "LOGO_F19"]
ROLE = {"LOGO_F07": "corner", "LOGO_F09": "corner", "LOGO_F10": "corner", "LOGO_F12": "corner",
        "A": "axial", "C": "axial", "LOGO_F16": "axial", "LOGO_F19": "axial", "B": "center"}


def _sig(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def simulate_products_cond(run: dict, p: Dict[str, float]):
    """Integrate the two product ODEs with X(t) = measured-X interpolation."""
    t = run["t_online"]
    t0, tend = float(t[0]), float(t[-1])
    t_obs = run["t_obs"]
    x_obs = np.asarray(run["obs"]["X"], dtype=float)
    ok = np.isfinite(x_obs) & np.isfinite(t_obs)
    t_x, x_x = t_obs[ok], x_obs[ok]

    def Xc(tc: float) -> float:
        return float(np.interp(tc, t_x, x_x))

    grid = np.arange(t0, tend + 1e-9, DT)
    P = np.zeros(2)  # [P_sol, P_IB]
    out_t, out_P = [], []
    for i, tc in enumerate(grid):
        out_t.append(tc)
        out_P.append(P.copy())
        if i == len(grid) - 1:
            break
        for _ in range(1):  # RK4 substeps at DT
            ks = []
            for th in (0.0, 0.5 * DT, 0.5 * DT, DT):
                s = tc + th
                x = Xc(s)
                F_rate = float(np.interp(s, t, run["feed_rate_L_h"]))
                V = float(np.interp(s, t, run["V_L"]))
                T = float(np.interp(s, t, run["temp"]))
                # ZOH induction gate (same as models._induced_at)
                idx = np.searchsorted(t, s, side="right") - 1
                idx = 0 if idx < 0 else (len(t) - 1 if idx >= len(t) else idx)
                is_ind = float(run["is_induced"][idx])
                V = max(V, 1e-4)
                qs = min(F_rate * S_IN / (x * V), QS_CAP)
                Dv = F_rate / V
                qP = is_ind * p["qP_max"] * qs / (qs + p["K_S"])
                qP *= float(np.exp(-p["alpha_T"] * max(T - T_REF, 0.0)))
                phi = _sig(p["theta_0"] + p["theta_T"] * (T - T_REF)
                           + p["a_qS"] * np.log(max(qs, 1e-6) / QS_REF))
                phi = min(max(phi, 0.0), 1.0)
                f1 = (1.0 - phi) * qP * x - Dv * P[0]
                f2 = phi * qP * x - Dv * P[1]
                ks.append(np.array([f1, f2]))
            P = P + DT / 6.0 * (ks[0] + 2 * ks[1] + 2 * ks[2] + ks[3])
    return np.array(out_t), np.array(out_P)


def pooled(obs: np.ndarray, pred: np.ndarray):
    ok = np.isfinite(obs) & np.isfinite(pred)
    e = obs[ok] - pred[ok]
    return float(np.sqrt(np.mean(e**2))), float(np.mean(e))


def split_tag(fold: str) -> str:
    _, test_ids = D.split_for_fold(fold)
    return f"holdout_{''.join(test_ids)}" if test_ids else "full"


def main() -> None:
    per_fold = {}
    for fold in FOLDS:
        mj = json.loads((RESULTS / split_tag(fold) / "metrics.json").read_text())
        params = mj["best_params"]
        train_ids, test_ids = D.split_for_fold(fold)
        bundle = D.load_dataset(train_ids, test_ids)
        acc = {"P_sol": [], "P_IB": [], "P_tot": []}
        endpoints = {}
        for run in bundle["test"]:
            tt, PP = simulate_products_cond(run, params)
            for s in ("P_sol", "P_IB"):
                obs = np.asarray(run["obs"][s], dtype=float)
                pred = np.interp(run["t_obs"], tt, PP[:, 0 if s == "P_sol" else 1])
                acc[s].append((obs, pred))
            o_tot = np.asarray(run["obs"]["P_sol"]) + np.asarray(run["obs"]["P_IB"])
            p_tot = PP[:, 0] + PP[:, 1]
            acc["P_tot"].append((o_tot, np.interp(run["t_obs"], tt, p_tot)))
            if len(test_ids) == 1:
                i_end = min(np.searchsorted(tt, run["t_obs"][-1]), len(tt) - 1)
                endpoints[run["run_id"]] = (o_tot[-1], float(p_tot[i_end]))
        row = {"fold": fold, "role": ROLE[fold]}
        for s in acc:
            o = np.concatenate([a[0] for a in acc[s]])
            pr = np.concatenate([a[1] for a in acc[s]])
            row[s] = pooled(o, pr)
        row["endpoints"] = endpoints
        per_fold[fold] = row

    # Baseline reference (stored unconditioned numbers).
    base = {}
    with (HERE / "results" / "logo_summary" / "logo_summary.csv").open() as fh:
        import csv as _csv
        for r in _csv.DictReader(fh):
            base[r["fold"]] = r

    lines = [
        "# X-conditioned product diagnostic (measured-X in the product ODEs)",
        "",
        "Products re-simulated with X(t) = measured offline X (soft-sensor framing);",
        "params = each fold's stored baseline fit. Comparison: unconditioned baseline.",
        "e = obs - pred (bias > 0 => underprediction). RMSE in g/L, pooled per group.",
        "",
        "| fold | held-out | P_sol cond (base) | P_IB cond (base) | P_tot cond (base) |",
        "| --- | --- | --- | --- | --- |",
    ]
    agg = {"cond": {s: [] for s in ("P_sol", "P_IB", "P_tot")},
           "base": {s: [] for s in ("P_sol", "P_IB", "P_tot")}}
    for fold in FOLDS:
        r = per_fold[fold]
        b = base[fold]
        cells = []
        for s in ("P_sol", "P_IB", "P_tot"):
            rmse, bias = r[s]
            b_rmse = float(b[f"{s}_rmse"])
            agg["cond"][s].append(rmse)
            agg["base"][s].append(b_rmse)
            cells.append(f"{rmse:.3f} ({bias:+.2f}) vs {b_rmse:.3f}")
        lines.append(f"| {fold} | {', '.join(D.split_for_fold(fold)[1])} | " + " | ".join(cells) + " |")

    lines += ["", "## Aggregate over the 9 groups (mean RMSE, g/L)", "",
              "| State | conditioned | baseline (uncond.) |", "| --- | --- | --- |"]
    for s in ("P_sol", "P_IB", "P_tot"):
        lines.append(f"| {s} | {np.mean(agg['cond'][s]):.3f} | {np.mean(agg['base'][s]):.3f} |")

    lines += ["", "## T35 endpoint ordering (single-run folds)", "",
              "| run | Ptot obs | Ptot cond pred |"]
    for fold in ("LOGO_F07", "LOGO_F19", "LOGO_F12"):
        r = per_fold[fold]
        for rid, (o, p) in r["endpoints"].items():
            lines.append(f"| {rid} | {o:.2f} | {p:.2f} |")

    text = "\n".join(lines) + "\n"
    (OUT / "xcond_test.md").write_text(text)
    print(text)
    print(f"saved {OUT/'xcond_test.md'}")


if __name__ == "__main__":
    main()
