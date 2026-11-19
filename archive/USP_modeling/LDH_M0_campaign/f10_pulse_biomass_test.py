"""F10 glucose-pulse biomass test (2026-09-26, user-driven).

Cheap check proposed in the F10 pulse discussion: does the modeled biomass X
overshoot observed X inside F10's glucose-accumulation window (~22.5-28 h)
and converge afterwards?

Background: F10 is the only FCCD run with measurable residual glucose -- the
offline assay shows 11.05 g/L at the first post-induction sample (t=24.0 h),
6.58 g/L at 26.0 h, back at the 0.01 g/L floor by 27.9 h (pool mass ~14 g,
about a third of the feed delivered over the same stretch; F10 fed ~48 mL/h
= ~19 g glucose/h, 2.5x the T25 siblings). The model's substrate driver is
the feed-supply proxy qS(t) = F_rate*S_IN/(X*V) (no pool state), so during
the accumulation it overstates true uptake by exactly the piling-up rate;
growth mu = Y_xs*qS should then run hot in the window -> predicted X
overshoots, and rejoins once the pool clears.

Decision rule (from the discussion):
  overshoot mid-window + convergence after -> pulse real, transient-only;
  pred X matches obs even mid-window -> pulse inert for this structure.
Controls: F09/F16 (T25, no accumulation) -- any analogous hump there is NOT
pulse-attributable. Each run is simulated with its own LOGO hold-out fit
(trained without that run), i.e. honest out-of-sample predictions.

Added here: a pool-corrected COUNTERFACTUAL -- the same fit re-simulated with
the feed forcing rescaled by (1 - d(pool)/dt / feed-glucose-rate): uptake =
feed while the pool grows, feed + pool-burn while it clears (total glucose
conserved). If the overshoot IS the phantom glucose, the counterfactual
should land on the observations.

NOTE (bug fixed while writing this test): models.py rhs line ~593 assigned a
local `k_deg` inside the kinpart branch, shadowing the closure variable and
raising UnboundLocalError for every NON-kinpart model at the shared
k_ref/k_deg tail -- silently swallowed by simulate_run's blanket except into
all-NaN predictions. Fixed by renaming the local (k_deg_T); behavior of all
stored results unchanged (verified: recomputed LOGO_F10 X RMSE 2.175 /
bias -1.615 == stored test_metrics.md).

Artifacts: results/f10_pulse/{f10_pulse_biomass.png, residuals.csv, SUMMARY.md}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import data as D
import fit as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "f10_pulse"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = ("M0_aload_qPTexp", "M0_kinpart_deg")  # empirical reference + primary mechanistic
FOLDS = {"F10": "LOGO_F10", "F09": "LOGO_F09", "F16": "LOGO_F16"}
WIN = (22.5, 28.0)  # the accumulation window from the discussion

S_IN = 400.0  # feed glucose concentration [g/L] (models.py)


def stored_params(model: str, fold: str) -> dict:
    """Best stored parameters for a fold, exactly as logo_summary.py reads them."""
    tag = f"holdout_{''.join(D.split_for_fold(fold)[1])}"
    mj = json.loads((HERE / "results" / model / tag / "metrics.json").read_text())
    return mj["best_params"]


def pool_series_f10() -> pd.DataFrame:
    """Residual-glucose pool mass [g] at F10's sample points (pipeline data)."""
    pkl = HERE.parent / "LDH_consolidated_results_processed.pkl"
    df = pd.read_pickle(pkl)
    f10 = df[(df["run_name"] == "F10") & (df["is_sample"] == True)]  # noqa: E712
    f10 = f10.sort_values("Process time [h]")
    out = pd.DataFrame({
        "t": f10["Process time [h]"].to_numpy(),
        "glc_g_l": f10["Glucose [g/L]"].to_numpy(),
        "V_L": f10["Volume corrected [mL]"].to_numpy() / 1000.0,
    })
    out["pool_g"] = out["glc_g_l"] * out["V_L"]
    return out


def pool_rate_fn(t: np.ndarray, pool: pd.DataFrame) -> np.ndarray:
    """Piecewise-linear d(pool)/dt [g/h]; zero outside the sampled span."""
    tp = pool["t"].to_numpy()
    pp = pool["pool_g"].to_numpy()
    rate = np.zeros_like(t, dtype=float)
    for i in range(len(tp) - 1):
        m = (t >= tp[i]) & (t <= tp[i + 1])
        rate[m] = (pp[i + 1] - pp[i]) / (tp[i + 1] - tp[i])
    return rate


def pulse_context(pool: pd.DataFrame) -> str:
    """Supporting numbers: how much fed glucose piled up, and when."""
    s = pool[pool["glc_g_l"] > 0.1]  # samples with a real pool
    lines = [
        "F10 pulse context (pipeline data; pool = Glucose[g/L] x V_corr):",
        f"  pool: {pool['pool_g'].iloc[0]:.2f} g at t={pool['t'].iloc[0]:.2f} h ->",
    ]
    for _, r in s.iterrows():
        lines.append(f"    {r['pool_g']:.2f} g at t={r['t']:.2f} h ({r['glc_g_l']:.2f} g/L)")
    lines.append(f"  cleared back to {pool['pool_g'].iloc[-1]:.2f} g by t={pool['t'].iloc[-1]:.2f} h")
    lines.append("")
    return "\n".join(lines)


def counterfactual_run(run: dict, pool: pd.DataFrame) -> dict:
    """Copy of `run` with the feed forcing rescaled to pool-corrected uptake.

    scale(t) = 1 - (dPool/dt) / (feed_glucose_rate); clipped at >= 0.
    While the pool grows (rate > 0) uptake < feed -> scale < 1; while it
    clears (rate < 0) uptake > feed -> scale > 1. Integral of the correction
    over the window ~ 0 (the pool returns to its floor), so total glucose is
    conserved -- only its timing changes.
    """
    t = np.asarray(run["t_online"], dtype=float)
    feed = np.asarray(run["feed_rate_L_h"], dtype=float).copy()
    gluco = np.maximum(feed * S_IN, 1e-9)
    scale = 1.0 - pool_rate_fn(t, pool) / gluco
    run2 = dict(run)
    run2["feed_rate_L_h"] = feed * np.clip(scale, 0.0, None)
    return run2


def main() -> None:
    pool = pool_series_f10()
    print(pulse_context(pool))

    rows = []
    summaries = []
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for ax, (run_id, fold) in zip(axes, FOLDS.items()):
        train_ids, test_ids = D.split_for_fold(fold)
        bundle = D.load_dataset(train_ids, test_ids)
        run = bundle["test"][0]
        t_obs = np.asarray(run["t_obs"], dtype=float)
        x_obs = np.asarray(run["obs"]["X"], dtype=float)
        ind = np.asarray(run["ind_time"], dtype=float)
        t_on = run["t_online"]
        t_flip = float(t_on[np.argmax(ind > 0)]) if (ind > 0).any() else float("nan")
        run_cf = counterfactual_run(run, pool) if run_id == "F10" else None

        ax.plot(t_obs, x_obs, "ko", ms=6, label="observed X", zorder=6)
        for model in MODELS:
            params = stored_params(model, fold)
            pred = F.gather_predictions(model, params, bundle["test"], bundle["c_fold"])[0]["pred"]
            x_pred = np.asarray(pred["obs"]["X"], dtype=float)
            st, sx = np.asarray(pred["sol_t"]), np.asarray(pred["sol_X"])
            in_win = (t_obs >= WIN[0]) & (t_obs <= WIN[1])
            e = x_pred - x_obs  # pred - obs; positive = model OVERpredicts
            for t, o, p, m in zip(t_obs, x_obs, x_pred, in_win):
                rows.append(dict(run=run_id, model=model, t=t, X_obs=o, X_pred=p,
                                 overpred=p - o, window=bool(m)))
            over_win = float(np.max(e[in_win]))
            after = t_obs > WIN[1]
            e_after = float(np.mean(e[after])) if after.any() else float("nan")
            e_pre = float(np.mean(e[t_obs < WIN[0]]))
            summaries.append(dict(run=run_id, model=model,
                                  overpred_in_window=over_win,
                                  mean_after=e_after, mean_pre=e_pre))
            print(f"{run_id} {model:18s} flip@{t_flip:5.2f}h | pre {e_pre:+.2f} | "
                  f"max overpred in window {over_win:+.2f} | mean after {e_after:+.2f} g/L")
            (line,) = ax.plot(st, sx, lw=1.8, label=model)
            if run_cf is not None:
                pred_cf = F.gather_predictions(model, params, [run_cf], bundle["c_fold"])[0]["pred"]
                x_cf = np.asarray(pred_cf["obs"]["X"], dtype=float)
                for t, p_cf in zip(t_obs, x_cf):
                    obs_t = float(np.interp(t, t_obs, x_obs))
                    rows.append(dict(run=run_id, model=model + " CF(pool-corr)", t=t,
                                     X_obs=obs_t, X_pred=p_cf, overpred=p_cf - obs_t,
                                     window=bool(WIN[0] <= t <= WIN[1])))
                ax.plot(st, np.interp(st, t_obs, x_cf), "--", color=line.get_color(),
                        lw=1.2, alpha=0.8, label=model + " pool-corrected")
                d = x_pred - x_cf
                m = in_win
                print(f"{run_id} {model:18s} counterfactual shifts X by "
                      f"{np.max(d[m]):+.2f} g/L in-window (phantom-glucose signature)")

        ax.axvspan(*WIN, color="tab:red", alpha=0.10, label="glucose pool window")
        ax.axvline(t_flip, color="gray", ls=":", lw=1)
        ax.set_title(f"{run_id} (LOGO hold-out, induction flip {t_flip:.1f} h)")
        ax.set_xlabel("Process time [h]")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Biomass X [g/L]")
    fig.suptitle("F10 pulse check: modeled vs observed biomass (shaded = residual-glucose window)")
    fig.tight_layout()
    fig.savefig(RESULTS / "f10_pulse_biomass.png", dpi=160)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(RESULTS / "residuals.csv", index=False)

    md = ["# F10 pulse biomass test — results", "", pulse_context(pool), "",
          "## Per-sample biomass residuals (pred − obs; positive = overprediction)", ""]
    for run_id in FOLDS:
        sub = tbl[tbl["run"] == run_id]
        for model in MODELS:
            m = sub[sub["model"].isin([model, model + " CF(pool-corr)"])]
            md.append(f"### {run_id} — {model}")
            md.append("")
            md.append("| t [h] | variant | X_obs | X_pred | overpred |")
            md.append("|---:|---|---:|---:|---:|")
            for _, r in m.iterrows():
                md.append(f"| {r['t']:.2f} | {r['model'].replace(model, 'base' if r['model'] == model else 'CF')}"
                          f" | {r['X_obs']:.2f} | {r['X_pred']:.2f} | {r['overpred']:+.2f} |")
            md.append("")
    summ = pd.DataFrame(summaries)
    md += ["## Summary (pred − obs)", "", summ.to_string(index=False, float_format="+.2f"), ""]
    (RESULTS / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"artifacts -> {RESULTS}")


if __name__ == "__main__":
    main()
