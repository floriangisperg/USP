"""CLI driver for the LDH mechanistic modeling playground.

Fits a mechanistic model variant on a grouped fold, evaluates on the held-out
run(s), and writes a self-contained result directory:

  results/<model>/<split>/
    params_all_seeds.json     all seeds' fitted parameters + train cost
    train_metrics.md          TRAIN metrics (primary reference structure)
    test_metrics.md           held-out metrics
    identifiability.md        parameter spread across seeds/folds + profile notes
    plots/
      traj_<run>.png          per-run observed-vs-predicted (X, P_sol, P_IB, CER, OUR, IB frac)
      parity.png              observed-vs-predicted scatter, all channels, train vs test
      final_summary.png       final biomass / total product / IB fraction per run
      profile_<param>.png     profile-likelihood curves (with --profile)

Example:
  python run_playground.py --fold A --model M0 --seeds 42 43 44
  python run_playground.py --fold B --model M1a --seeds 42 --profile
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import data as D  # noqa: E402
import fit as F  # noqa: E402
import metrics as M  # noqa: E402
from models import GAS_STATES, OBS_CHANNELS, PARAM_SPECS, SPARSE_STATES, param_names  # noqa: E402

# Verified FCCD setpoints (T [C], initial qS target [g/g/h]) for plot labels.
RUN_SETPIONTS = {
    "F07": (35, 0.2), "F09": (25, 0.2), "F10": (25, 0.6), "F12": (35, 0.6),
    "F13": (30, 0.4), "F14": (30, 0.4), "F15": (30, 0.4), "F16": (25, 0.4),
    "F17": (30, 0.2), "F18": (30, 0.6), "F19": (35, 0.4),
}

# Restrained palette (observed = near-black; predicted = blue; gas = orange; test = red title).
C_OBS, C_PRED, C_GAS_OBS, C_GAS_PRED, C_TEST = "#222222", "#0072B2", "#888888", "#D55E00", "#B22222"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _induction_time(run: dict) -> float:
    """First process time at which is_induced turns on (for the vline)."""
    ind = run["is_induced"]
    t = run["t_online"]
    on = np.where(ind > 0.5)[0]
    return float(t[on[0]]) if len(on) else float("nan")


def plot_run_trajectory(rec: dict, model: str, out_path: Path, is_test: bool) -> None:
    """Per-run observed-vs-predicted figure (X, P_sol, P_IB, CER, OUR, IB fraction)."""
    run, pred = rec["run"], rec["pred"]
    t_obs = run["t_obs"]
    t_gas = run["t_gas"]
    st, sX = pred["sol_t"], pred["sol_X"]
    sPsol = pred.get("sol_Psol", np.full_like(st, np.nan))
    sPib = pred.get("sol_Pib", np.full_like(st, np.nan))
    t_ind = _induction_time(run)
    channels = OBS_CHANNELS[model]
    has_gas = any(c in GAS_STATES for c in channels)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    Tset, qs = RUN_SETPIONTS.get(run["run_id"], (run["T_induction"], None))
    tag = "TEST (held-out)" if is_test else "train"
    fig.suptitle(f"{run['run_id']}  —  T={Tset:.0f}°C  qS₀={qs}  [{model}, {tag}]", color=C_TEST if is_test else "black")

    # Biomass (dense solution trajectory + obs points).
    ax = axes[0, 0]
    ax.plot(st, sX, color=C_PRED, lw=1.5, label="pred")
    ax.scatter(t_obs, run["obs"]["X"], color=C_OBS, s=28, zorder=5, label="obs")
    ax.set_ylabel("Biomass [g/L]"); ax.set_title("Biomass X")
    # Product soluble / IB: dense trajectory (correctly 0 before induction) + obs points.
    ax = axes[0, 1]
    ax.plot(st, sPsol, color=C_PRED, lw=1.5)
    ax.scatter(t_obs, run["obs"]["P_sol"], color=C_OBS, s=28, zorder=5)
    ax.set_ylabel("Protein sol [g/L]"); ax.set_title("Soluble product")
    ax = axes[0, 2]
    ax.plot(st, sPib, color=C_PRED, lw=1.5)
    ax.scatter(t_obs, run["obs"]["P_IB"], color=C_OBS, s=28, zorder=5)
    ax.set_ylabel("Protein IB [g/L]"); ax.set_title("IB product")
    # Gas (if model predicts it).
    ax = axes[1, 0]
    if has_gas and len(t_gas):
        ax.plot(t_gas, pred["gas"]["CER"], color=C_GAS_PRED, lw=1.3, label="pred")
        ax.scatter(t_gas, run["gas"]["CER"], color=C_GAS_OBS, s=6, alpha=0.6, label="obs")
    ax.set_ylabel("CER [mmol/L/h]"); ax.set_title("CER"); ax.set_xlabel("Process time [h]")
    ax = axes[1, 1]
    if has_gas and len(t_gas):
        ax.plot(t_gas, pred["gas"]["OUR"], color=C_GAS_PRED, lw=1.3, label="pred")
        ax.scatter(t_gas, run["gas"]["OUR"], color=C_GAS_OBS, s=6, alpha=0.6, label="obs")
    ax.set_ylabel("OUR [mmol/L/h]"); ax.set_title("OUR"); ax.set_xlabel("Process time [h]")
    # IB fraction on the dense grid (undefined before product exists).
    ax = axes[1, 2]
    ptot_o = run["obs"]["P_sol"] + run["obs"]["P_IB"]
    ptot_p = sPsol + sPib
    frac_o = np.where(ptot_o > 1e-6, run["obs"]["P_IB"] / np.maximum(ptot_o, 1e-6), np.nan)
    frac_p = np.where(ptot_p > 1e-6, sPib / np.maximum(ptot_p, 1e-6), np.nan)
    ax.plot(st, frac_p, color=C_PRED, lw=1.5)
    ax.scatter(t_obs, frac_o, color=C_OBS, s=28, zorder=5)
    ax.set_ylabel("IB fraction [-]"); ax.set_title("IB fraction (P_IB / P_total)"); ax.set_xlabel("Process time [h]")
    ax.set_ylim(-0.03, 1.03)

    for ax in axes.flat:
        if np.isfinite(t_ind):
            ax.axvline(t_ind, color="#999999", ls="--", lw=0.8)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_parity(preds_train: List[dict], preds_test: List[dict], model: str, out_path: Path) -> None:
    """Observed-vs-predicted scatter for every channel; train vs test coloured."""
    channels = OBS_CHANNELS[model]
    n = len(channels)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow), squeeze=False)
    for idx, ch in enumerate(channels):
        ax = axes[idx // ncol][idx % ncol]
        for recs, color, label in ((preds_train, C_PRED, "train"), (preds_test, C_TEST, "test")):
            xs, ys = [], []
            for rec in recs:
                run, pred = rec["run"], rec["pred"]
                if ch in SPARSE_STATES:
                    obs = np.asarray(run["obs"][ch]); p = np.asarray(pred["obs"][ch])
                else:
                    obs = np.asarray(run["gas"][ch]); p = np.asarray(pred["gas"][ch])
                fin = np.isfinite(obs) & np.isfinite(p)
                xs.extend(obs[fin]); ys.extend(p[fin])
            if xs:
                ax.scatter(xs, ys, color=color, s=16, alpha=0.6, label=f"{label} ({len(xs)})")
        if xs:
            lo = min(min(xs), min(ys) if ys else min(xs)); hi = max(max(xs), max(ys) if ys else max(xs))
            ax.plot([lo, hi], [lo, hi], color="#333333", ls="--", lw=0.8)
        ax.set_xlabel(f"observed {ch}"); ax.set_ylabel(f"predicted {ch}"); ax.set_title(ch)
        ax.legend(fontsize=8); ax.set_aspect("equal", adjustable="datalim")
    for j in range(len(channels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{model} — parity (observed vs predicted)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_final_summary(preds_train: List[dict], preds_test: List[dict], model: str, out_path: Path) -> None:
    """Final biomass, total product, and IB fraction per run (obs vs pred)."""
    runs_order = [r["run_id"] for r in preds_train] + [r["run_id"] for r in preds_test]
    is_test = [False] * len(preds_train) + [True] * len(preds_test)
    allp = preds_train + preds_test
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, kind, title in zip(
        axes,
        ("X", "Ptot", "IBfrac"),
        ("Final biomass [g/L]", "Final total product [g/L]", "Final IB fraction [-]"),
    ):
        obs_v, pred_v, xlbl = [], [], []
        for rec, testf in zip(allp, is_test):
            run, pred = rec["run"], rec["pred"]
            xo = run["obs"]["X"]; xp = pred["obs"]["X"]
            pso_o, pso_p = run["obs"]["P_sol"], pred["obs"]["P_sol"]
            pib_o, pib_p = run["obs"]["P_IB"], pred["obs"]["P_IB"]
            if kind == "X":
                o, p = xo[-1], xp[-1]
            elif kind == "Ptot":
                o, p = (pso_o[-1] + pib_o[-1]), (pso_p[-1] + pib_p[-1])
            else:
                to, tp = (pso_o[-1] + pib_o[-1]), (pso_p[-1] + pib_p[-1])
                o = pib_o[-1] / to if to > 1e-6 else np.nan
                p = pib_p[-1] / tp if tp > 1e-6 else np.nan
            obs_v.append(o); pred_v.append(p)
        x = np.arange(len(allp))
        w = 0.38
        ax.bar(x - w / 2, obs_v, w, color="#444444", label="observed")
        ax.bar(x + w / 2, pred_v, w, color=C_PRED, label="predicted")
        for xi, tv in zip(x, is_test):
            if tv:
                ax.axvspan(xi - 0.5, xi + 0.5, color=C_TEST, alpha=0.08)
        ax.set_xticks(x); ax.set_xticklabels(runs_order, rotation=45, ha="right", fontsize=8)
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle(f"{model} — final endpoint summary (red shade = held-out run)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_profile(param: str, profile: list, mle_val: float, out_path: Path) -> None:
    vals = [p[0] for p in profile if np.isfinite(p[1])]
    costs = [p[1] for p in profile if np.isfinite(p[1])]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals, costs, "-o", color=C_PRED, ms=3)
    ax.axvline(mle_val, color=C_TEST, ls="--", lw=1, label=f"MLE={mle_val:.4g}")
    ax.set_xlabel(param); ax.set_ylabel("train cost (0.5 * sum resid^2)")
    ax.set_title(f"Profile likelihood: {param}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _endpoints(preds: List[dict]) -> List[dict]:
    """Final biomass, total product, and IB fraction (obs vs pred) per run."""
    out = []
    for rec in preds:
        run, pred = rec["run"], rec["pred"]
        xo = run["obs"]["X"]; xp = pred["obs"]["X"]
        pso_o, pso_p = run["obs"]["P_sol"], pred["obs"]["P_sol"]
        pib_o, pib_p = run["obs"]["P_IB"], pred["obs"]["P_IB"]
        to = pso_o[-1] + pib_o[-1]; tp = pso_p[-1] + pib_p[-1]
        out.append({
            "run_id": rec["run_id"],
            "X_final_obs": float(xo[-1]), "X_final_pred": float(xp[-1]),
            "Ptot_final_obs": float(to), "Ptot_final_pred": float(tp),
            "IBfrac_final_obs": float(pib_o[-1] / to) if to > 1e-6 else float("nan"),
            "IBfrac_final_pred": float(pib_p[-1] / tp) if tp > 1e-6 else float("nan"),
        })
    return out



def run_one(args) -> Path:
    t_start = time.time()
    train_ids, test_ids = D.split_for_fold(args.fold)
    print(f"Fold {args.fold}: train={train_ids} | test={test_ids}")
    bundle = D.load_dataset(train_ids, test_ids, data_file=args.data)

    split_tag = f"holdout_{''.join(test_ids)}" if test_ids else "full"
    out_dir = Path(args.out) / args.model / split_tag
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # --- Multistart fits ---
    all_seeds = []
    best = None
    for seed in args.seeds:
        t0 = time.time()
        res = F.fit_model(args.model, bundle, seed=seed, n_starts=args.n_starts, max_nfev=args.max_nfev)
        dt = time.time() - t0
        if res is None:
            print(f"  seed {seed}: FAILED")
            continue
        print(f"  seed {seed}: train cost={res.cost:.4f}  params={ {k: round(v,4) for k,v in res.params.items()} }  ({dt:.0f}s)")
        all_seeds.append({"seed": seed, "cost": float(res.cost), "params": res.params})
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        print("All seeds failed."); sys.exit(1)

    params = best.params
    (out_dir / "params_all_seeds.json").write_text(json.dumps(all_seeds, indent=2))

    # --- Predictions + metrics (train + test) ---
    preds_train = F.gather_predictions(args.model, params, bundle["train"], bundle["c_fold"])
    preds_test = F.gather_predictions(args.model, params, bundle["test"], bundle["c_fold"])
    cells_train = F.cells_from_predictions(args.model, preds_train, bundle["range_train"])
    cells_test = F.cells_from_predictions(args.model, preds_test, bundle["range_train"])

    def _jsonable(cells):
        agg = M.aggregate(cells)
        return {
            "overall": agg["overall"],
            "per_state": agg["per_state"],
            "cells": {f"{r}|{s}": v for (r, s), v in cells.items()},
        }
    metrics_json = {
        "model": args.model, "fold": args.fold, "test_run_ids": test_ids,
        "best_seed_cost": float(best.cost), "best_params": params,
        "all_seeds": all_seeds,
        "train": _jsonable(cells_train),
        "test": _jsonable(cells_test) if preds_test else None,
        "test_endpoints": _endpoints(preds_test),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    (out_dir / "train_metrics.md").write_text(
        M.render_metrics_md(f"TRAIN metrics — {args.model} fold {args.fold} ({split_tag})",
                            cells_train, bundle["range_train"], args.model))
    if preds_test:
        (out_dir / "test_metrics.md").write_text(
            M.render_metrics_md(f"TEST (held-out) metrics — {args.model} fold {args.fold} ({split_tag})",
                                cells_test, bundle["range_train"], args.model))

    # --- Plots ---
    plots = out_dir / "plots"
    for rec in preds_train:
        plot_run_trajectory(rec, args.model, plots / f"traj_{rec['run_id']}_train.png", is_test=False)
    for rec in preds_test:
        plot_run_trajectory(rec, args.model, plots / f"traj_{rec['run_id']}_TEST.png", is_test=True)
    plot_parity(preds_train, preds_test, args.model, plots / "parity.png")
    plot_final_summary(preds_train, preds_test, args.model, plots / "final_summary.png")

    # --- Identifiability: seed spread + optional profile likelihood ---
    id_lines = [f"# Identifiability — {args.model} fold {args.fold}", ""]
    id_lines.append("## Parameter spread across seeds")
    id_lines.append("| param | " + " | ".join(f"seed{s['seed']}" for s in all_seeds) + " | min | max | spread |")
    id_lines.append("|---|" + "---:|" * (len(all_seeds) + 3))
    for p in param_names(args.model):
        vals = [s["params"].get(p, float("nan")) for s in all_seeds]
        vmin, vmax = min(vals), max(vals)
        spread = (vmax - vmin) / max(abs(np.mean(vals)), 1e-9)
        id_lines.append(f"| {p} | " + " | ".join(f"{v:.4g}" for v in vals) + f" | {vmin:.4g} | {vmax:.4g} | {spread:.2f} |")
    id_lines.append("")
    id_lines.append("spread = (max-min)/|mean|; small => stable across seeds. Interior (not at bound) required for 'identified'.")
    (out_dir / "identifiability.md").write_text("\n".join(id_lines))

    if args.profile:
        # Probe the identifiability-relevant params (curated; a_qS/K_S/theta_T/Y_xs,
        # plus the burden params a_b/tau_b when the model carries them).
        probed = [p for p in ["a_qS", "K_S", "theta_T", "Y_xs", "a_b", "tau_b"]
                  if p in PARAM_SPECS[args.model]]
        prof_lines = [f"# Profile likelihood — {args.model} fold {args.fold}", ""]
        for p in probed:
            mle = params[p]
            specs = PARAM_SPECS[args.model][p]
            lo, hi = specs[1], specs[2]
            span = min(abs(mle - lo), abs(hi - mle))
            span = max(span, abs(mle) * 0.5, 1e-3)
            grid = np.linspace(max(lo, mle - 2 * span), min(hi, mle + 2 * span), 7)
            prof = F.profile_likelihood(args.model, bundle, params, p, grid, seed=args.seeds[0])
            plot_profile(p, prof, mle, plots / f"profile_{p}.png")
            prof_lines.append(f"## {p}  (MLE={mle:.4g}, bounds=[{lo},{hi}])")
            prof_lines.append("| value | train cost | delta_cost |")
            prof_lines.append("|---:|---:|---:|")
            base = min(c for _, c, _ in prof if np.isfinite(c))
            for v, c, _ in prof:
                prof_lines.append(f"| {v:.4g} | {c:.4g} | {c - base:.4g} |")
            prof_lines.append("")
        (out_dir / "profile_likelihood.md").write_text("\n".join(prof_lines))

    print(f"\nDone {args.model}/{split_tag} in {time.time()-t_start:.0f}s -> {out_dir}")
    print(f"  Best seed {all_seeds[0]['seed'] if len(all_seeds)==1 else min(all_seeds,key=lambda s:s['cost'])['seed']} "
          f"train cost={best.cost:.4f}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="LDH mechanistic playground")
    ap.add_argument("--fold", choices=list(D.FOLDS.keys()), default="A")
    ap.add_argument("--model", choices=list(PARAM_SPECS.keys()), default="M0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-starts", type=int, default=3, help="multistart restarts per seed")
    ap.add_argument("--max-nfev", type=int, default=400)
    ap.add_argument("--profile", action="store_true", help="run profile-likelihood identifiability probes")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
