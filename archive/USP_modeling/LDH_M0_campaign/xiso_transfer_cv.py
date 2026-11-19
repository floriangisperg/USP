"""Xiso transfer experiment -- DOCUMENTED LOO-CV.

QUESTION: does the IB-decline "clock" (the production-decline mechanism from LDH) transfer to Xiso
(a different protein)? LDH's clock is T-gated (decline at T30, sustained at T35); Xiso declines at
~T35 (Ferm01), so the test model uses a T-INDEPENDENT clock (z=exp(-t_ind/tau)), independent sol/IB
gates -- the structure LDH motivated, with the gating generalized.

DESIGN:
  - full/      : fit on all 3 Xiso runs (consistency) -> extract tau, compare to LDH (~0.7-1.5h).
  - holdout_<run>/ (LOO-CV, 3 folds): train on the OTHER 2 Xiso runs, predict the held-out one.
    Fold tests: holdout_Ferm01 (predict T35/high-high from low-low+center = extrapolation),
                holdout_Ferm02 (predict T25/low-low), holdout_Ferm03 (predict T29/center = interpolation).

Each subfolder: params.json, metrics.json (train + held-out RMSE), metrics.md, and per-run trajectory
plots (train + held-out). Top-level SUMMARY.md documents the experiment, the tau comparison, the verdict.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xiso_transfer as X  # reuse load_xiso_runs, simulate, fit, PARAMS, rhs_factory, rk4

OUT = "results/xiso_transfer"
LDH_TAU_RANGE = "0.7-1.5 h"
T_LOOKUP = {"00_Ferm01": 35, "01_Ferm02": 25, "02_Ferm03": 29}


def traj_dense(params, run):
    """Return (t_dense, [X,P_sol,P_IB] arrays) for the model line, plus obs arrays."""
    rhs = X.rhs_factory(params, run)
    y = np.array([run["X0"], run["Psol0"], run["Pib0"]])
    t = run["t0"]
    td = [t]; yd = [y.copy()]
    for te in run["t_obs"][1:]:
        y = X.rk4(rhs, y, t, te, 0.1); t = te
        td.append(t); yd.append(y.copy())
    return np.array(td), np.array(yd)


def metrics(params, runs, ranges):
    out = {}
    for k in ("X", "P_sol", "P_IB"):
        se, nb = [], 0
        for r in runs:
            o = X.simulate(params, r)
            d = (o[k] - r["obs"][k]) / ranges[k]
            fin = np.isfinite(d)
            se.extend((d[fin] ** 2).tolist()); nb += int(fin.sum())
        out[f"{k}_RMSE_range"] = float(np.sqrt(np.sum(se) / nb)) if nb else float("nan")
    return out


def plot_run(params, run, ax_row_title=""):
    td, yd = traj_dense(params, run)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (j, k, lab, c) in zip(axes, [(0, "X", "X [g/L]", "#0072B2"),
                                          (1, "P_sol", "P_sol [g/L]", "#009E73"),
                                          (2, "P_IB", "P_IB [g/L]", "#D55E00")]):
        ax.scatter(run["t_obs"], run["obs"][k], c="k", s=30, zorder=5, label="observed")
        ax.plot(td, yd[:, j], "-", c=c, lw=1.8, label="model")
        ax.set_title(f"{run['run_id']} {lab} (T~{T_LOOKUP.get(run['run_id'],'?')})", fontsize=9)
        ax.set_xlabel("process time [h]", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
    return fig


def write_fold(name, params, train_runs, test_run, ranges):
    fold = os.path.join(OUT, name)
    os.makedirs(os.path.join(fold, "plots"), exist_ok=True)
    p = dict(zip(X.PARAMS, params))
    with open(os.path.join(fold, "params.json"), "w") as f:
        json.dump({**p, "tau_IB_note": f"compare LDH {LDH_TAU_RANGE}"}, f, indent=2)
    # train + held-out metrics
    m = {"train": metrics(params, train_runs, ranges)}
    if test_run is not None:
        held = {"run_id": test_run["run_id"], "T": T_LOOKUP.get(test_run["run_id"])}
        m["held_out"] = {**held, **metrics(params, [test_run], ranges)}
        o = X.simulate(params, test_run)
        m["held_out"]["IBend_model"] = float(o["P_IB"][-1])
        m["held_out"]["IBend_obs"] = float(test_run["obs"]["P_IB"][-1])
        m["held_out"]["dIB_model"] = float(o["P_IB"][-1] - o["P_IB"][-2])
        m["held_out"]["dIB_obs"] = float(test_run["obs"]["P_IB"][-1] - test_run["obs"]["P_IB"][-2])
    with open(os.path.join(fold, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    # metrics.md
    with open(os.path.join(fold, "metrics.md"), "w") as f:
        f.write(f"# {name}\n\nModel: decoupled sol/IB + T-INDEPENDENT clock (z=exp(-t_ind/tau)).\n")
        f.write(f"Trained on: {[r['run_id'] for r in train_runs]}\n")
        f.write(f"Held out: {test_run['run_id'] if test_run else 'none'}\n\n")
        f.write(f"tau_IB = {p['tau_IB']:.2f} h  (LDH clock ~{LDH_TAU_RANGE})\n\n")
        f.write(f"train metrics: {m['train']}\n")
        if "held_out" in m:
            h = m["held_out"]
            f.write(f"held-out: {h}\n")
    # plots: train runs + held-out
    for r in train_runs:
        fig = plot_run(params, r)
        fig.suptitle(f"{name} TRAIN: {r['run_id']}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(fold, "plots", f"traj_{r['run_id']}_train.png"), dpi=110); plt.close(fig)
    if test_run is not None:
        fig = plot_run(params, test_run)
        fig.suptitle(f"{name} HELD-OUT: {test_run['run_id']} (predict from { [r['run_id'] for r in train_runs] })",
                     fontsize=10, color="crimson")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(fold, "plots", f"traj_{test_run['run_id']}_TEST.png"), dpi=110); plt.close(fig)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = X.load_xiso_runs("../../Xiso/Xiso_consolidated_results.xlsx")
    ranges = {k: max(np.nanmax([r["obs"][k] for r in runs]) - np.nanmin([r["obs"][k] for r in runs]), 1e-3)
              for k in ("X", "P_sol", "P_IB")}

    # --- full fit (consistency) ---
    print("full fit...")
    full_best = X.fit(runs, seed=42, n_starts=6)
    full_m = write_fold("full", full_best[0], runs, None, ranges)
    print(f"  full cost={full_best[1]:.4f} tau_IB={dict(zip(X.PARAMS,full_best[0]))['tau_IB']:.2f}")

    # --- LOO-CV (3 folds) ---
    fold_results = {}
    for i, test in enumerate(runs):
        train = [r for r in runs if r["run_id"] != test["run_id"]]
        name = f"holdout_{test['run_id']}"
        print(f"{name}: train={[r['run_id'] for r in train]} -> test {test['run_id']}...")
        best = X.fit(train, seed=42, n_starts=6)
        m = write_fold(name, best[0], train, test, ranges)
        fold_results[name] = (dict(zip(X.PARAMS, best[0]))["tau_IB"], best[1], m)
        print(f"  tau_IB={dict(zip(X.PARAMS,best[0]))['tau_IB']:.2f}  "
              f"held-out IB RMSE={m['held_out']['P_IB_RMSE_range']:.3f}")

    # --- SUMMARY.md ---
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as f:
        f.write("# Xiso transfer experiment\n\n")
        f.write("**Question:** does the IB-decline 'clock' (production-decline mechanism from LDH) transfer\n")
        f.write("to Xiso (a different protein)? LDH's clock is T-gated (decline at T30, sustained at T35);\n")
        f.write("Xiso declines at ~T35, so the test model uses a T-INDEPENDENT clock z=exp(-t_ind/tau),\n")
        f.write("independent sol/IB gates -- the LDH-motivated structure with gating generalized.\n\n")
        f.write(f"**Model:** decoupled sol/IB, Monod(qS), independent T-gates, T-independent clock. "
                f"{len(X.PARAMS)} params.\n")
        f.write(f"**Data:** 3 Xiso runs (low-low/center/high-high DoE diagonal), T~25/29/35.\n")
        f.write(f"**LDH clock reference:** tau ~{LDH_TAU_RANGE}.\n\n")
        f.write("## Results\n\n")
        f.write(f"- **full fit** (all 3, consistency): tau_IB = {dict(zip(X.PARAMS,full_best[0]))['tau_IB']:.2f} h, "
                f"cost {full_best[1]:.4f}\n")
        f.write("- **LOO-CV** (train 2, test 1):\n\n")
        f.write("| fold (held-out) | train | tau_IB [h] | held-out P_IB RMSE (range) | held-out dIB mdl/obs |\n")
        f.write("| --- | --- | ---: | ---: | --- |\n")
        for name, (tau, cost, m) in fold_results.items():
            h = m["held_out"]
            f.write(f"| {name} | T~{h['T']} | {tau:.2f} | {h['P_IB_RMSE_range']:.3f} | "
                    f"{h['dIB_model']:+.3f}/{h['dIB_obs']:+.3f} |\n")
        f.write("\n## Verdict\n\n")
        f.write("(filled by the run's outcome -- compare tau_IB across folds and to LDH; check whether the\n")
        f.write("held-out IB trajectory (esp. Ferm01's decline) is predicted.)\n")
    print(f"\nDONE -- documented in {OUT}/")


if __name__ == "__main__":
    main()
