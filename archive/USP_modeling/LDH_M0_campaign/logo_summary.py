"""Tier-2 complete-design leave-one-group-out (LOGO) summary for the playground.

Mirrors the Fermentation paper's complete-design validation (``tab:fulldesign``):
nine hold-out groups cover all eleven fermentations — the 30 degC isotherm folds
A/B/C (center triplicate always held out together) plus the six single-run
LOGO_F* folds — with the canonical ``M0_aload_qPTexp`` retrained per fold.

For each fold the stored best parameters (``metrics.json``) are re-simulated on
the held-out runs through the *exact* runner code path
(``fit.gather_predictions``), and pooled held-out RMSE/Bias are computed per
state (X, P_sol, P_IB) plus the derived total product (P_sol + P_IB at the
sample points). Entries in the aggregate table are mean +/- SD of the
per-fold pooled RMSEs across folds in each stratum (the paper's +/- is across
8-seed ensemble members; here seeds are deterministic and identical within a
fold, so the reported spread is across folds).

Error convention: e = obs - pred (Bias > 0 => underprediction), as everywhere
in the playground.

Outputs: results/logo_summary/logo_summary.md (+ .csv) and a console table.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import data as D
import fit as F
import numpy as np

HERE = Path(__file__).resolve().parent

MODEL = sys.argv[1] if len(sys.argv) > 1 else "M0_aload_qPTexp"
RESULTS = HERE / "results" / MODEL
OUT = HERE / "results" / ("logo_summary" if MODEL == "M0_aload_qPTexp" else f"logo_summary_{MODEL}")
OUT.mkdir(parents=True, exist_ok=True)

# The nine hold-out groups of the complete design, with the held-out run's
# FCCD role (factorial corner / axial face-center / interior center).
FOLDS: List[str] = [
    "A", "B", "C",
    "LOGO_F07", "LOGO_F09", "LOGO_F10", "LOGO_F12", "LOGO_F16", "LOGO_F19",
]
ROLE: Dict[str, str] = {
    "LOGO_F07": "corner", "LOGO_F09": "corner", "LOGO_F10": "corner", "LOGO_F12": "corner",
    "A": "axial", "C": "axial", "LOGO_F16": "axial", "LOGO_F19": "axial",
    "B": "center",
}
STATES = ("X", "P_sol", "P_IB", "P_tot")


def _pooled(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    """Pooled RMSE, Bias (obs - pred), n over finite points."""
    ok = np.isfinite(obs) & np.isfinite(pred)
    e = obs[ok] - pred[ok]
    return float(np.sqrt(np.mean(e**2))), float(np.mean(e)), int(ok.sum())


def fold_split_tag(fold: str) -> str:
    _, test_ids = D.split_for_fold(fold)
    return f"holdout_{''.join(test_ids)}" if test_ids else "full"


def evaluate_fold(fold: str) -> dict:
    """Re-simulate the fold's held-out runs with its stored best parameters."""
    tag = fold_split_tag(fold)
    mj = json.loads((RESULTS / tag / "metrics.json").read_text())
    params = mj["best_params"]
    train_ids, test_ids = D.split_for_fold(fold)
    bundle = D.load_dataset(train_ids, test_ids)
    preds = F.gather_predictions(MODEL, params, bundle["test"], bundle["c_fold"])

    arrays: Dict[str, list] = {s: ([], []) for s in STATES}  # state -> (obs[], pred[])
    per_run = []
    for rec in preds:
        run, pred = rec["run"], rec["pred"]
        row = {"run": rec["run_id"]}
        for s in ("X", "P_sol", "P_IB"):
            o = np.asarray(run["obs"][s], dtype=float)
            p = np.asarray(pred["obs"][s], dtype=float)
            arrays[s][0].append(o)
            arrays[s][1].append(p)
            rmse, bias, n = _pooled(o, p)
            row[s] = (rmse, bias, n)
        o_tot = np.asarray(run["obs"]["P_sol"], dtype=float) + np.asarray(run["obs"]["P_IB"], dtype=float)
        p_tot = np.asarray(pred["obs"]["P_sol"], dtype=float) + np.asarray(pred["obs"]["P_IB"], dtype=float)
        arrays["P_tot"][0].append(o_tot)
        arrays["P_tot"][1].append(p_tot)
        rmse, bias, n = _pooled(o_tot, p_tot)
        row["P_tot"] = (rmse, bias, n)
        per_run.append(row)

    fold_res = {"fold": fold, "role": ROLE[fold], "test": test_ids, "per_run": per_run}
    for s in STATES:
        o = np.concatenate(arrays[s][0])
        p = np.concatenate(arrays[s][1])
        fold_res[s] = _pooled(o, p)  # pooled over all held-out points of the group
    return fold_res


def _agg(folds: List[dict], state: str) -> tuple[float, float]:
    vals = [f[state][0] for f in folds]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def main() -> None:
    res = [evaluate_fold(f) for f in FOLDS]

    strata = {
        "Isotherm A-C": [r for r in res if r["fold"] in ("A", "B", "C")],
        "Full design (9)": res,
        "Corners (4)": [r for r in res if r["role"] == "corner"],
        "Axial (4)": [r for r in res if r["role"] == "axial"],
        "Center (1)": [r for r in res if r["role"] == "center"],
    }

    lines = [
        "# Complete-design LOGO validation — M0_aload_qPTexp (corrected data, 2026-09-04)",
        "",
        "Nine hold-out groups covering all eleven fermentations (isotherm folds A/B/C +",
        "single-run LOGO_F07/F09/F10/F12/F16/F19; center triplicate always held out",
        "together). Held-out RMSE/Bias pooled per group; P_tot = P_sol + P_IB at sample",
        "points. e = obs - pred (Bias > 0 => underprediction). Aggregate entries are",
        "mean +/- SD of the per-group RMSEs across folds (seeds are deterministic and",
        "identical within each fold, so there is no ensemble spread to report).",
        "",
        "## Per-group held-out performance",
        "",
        "| fold | role | held-out | " + " | ".join(f"{s} RMSE (bias)" for s in STATES) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in STATES) + " |",
    ]
    csv_rows = [["fold", "role", "held_out"] + [f"{s}_rmse" for s in STATES] + [f"{s}_bias" for s in STATES]]
    for r in res:
        cells = []
        for s in STATES:
            rmse, bias, _ = r[s]
            cells.append(f"{rmse:.3f} ({bias:+.3f})")
        lines.append(
            f"| {r['fold']} | {r['role']} | {', '.join(r['test'])} | " + " | ".join(cells) + " |"
        )
        csv_rows.append([r["fold"], r["role"], ";".join(r["test"])]
                        + [f"{r[s][0]:.4f}" for s in STATES] + [f"{r[s][1]:.4f}" for s in STATES])

    lines += ["", "## Aggregate (mean +/- SD of per-group RMSE, g/L)", "",
              "| State | " + " | ".join(strata.keys()) + " |",
              "| --- | " + " | ".join("---" for _ in strata) + " |"]

    def _row(label: str, states: tuple[str, ...]) -> str:
        cells = []
        for folds in strata.values():
            vals = [np.mean([f[s][0] for s in states]) for f in folds]
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cells.append(f"{m:.2f} +/- {sd:.2f}" if len(folds) > 1 else f"{m:.2f}")
        return f"| {label} | " + " | ".join(cells) + " |"

    lines.append(_row("Biomass (X)", ("X",)))
    lines.append(_row("Soluble (P_sol)", ("P_sol",)))
    lines.append(_row("IB (P_IB)", ("P_IB",)))
    lines.append(_row("Total (P_tot)", ("P_tot",)))
    lines.append(_row("Protein states", ("P_sol", "P_IB")))
    lines.append(_row("Macro-average (measured)", ("X", "P_sol", "P_IB")))

    # Endpoint detail for the six new single-run folds (from the stored runner outputs).
    lines += ["", "## Endpoint detail — single-run LOGO folds", "",
              "| fold | X final (obs/pred) | Ptot final (obs/pred) | IBfrac final (obs/pred) |",
              "| --- | --- | --- | --- |"]
    for fold in ("LOGO_F07", "LOGO_F09", "LOGO_F10", "LOGO_F12", "LOGO_F16", "LOGO_F19"):
        mj = json.loads((RESULTS / fold_split_tag(fold) / "metrics.json").read_text())
        for ep in mj["test_endpoints"]:
            lines.append(
                f"| {fold} ({ep['run_id']}) | {ep['X_final_obs']:.1f} / {ep['X_final_pred']:.1f}"
                f" | {ep['Ptot_final_obs']:.2f} / {ep['Ptot_final_pred']:.2f}"
                f" | {ep['IBfrac_final_obs']:.3f} / {ep['IBfrac_final_pred']:.3f} |"
            )

    (OUT / "logo_summary.md").write_text("\n".join(lines) + "\n")
    with (OUT / "logo_summary.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)

    print("\n".join(lines))
    print(f"\nsaved {OUT/'logo_summary.md'} (+ .csv)")


if __name__ == "__main__":
    main()
