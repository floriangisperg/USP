"""Xiso transfer (consistency check): fit a T-INDEPENDENT production clock on Xiso's IB decline
and compare the timescale to LDH's. LDH's v2 clock is T-GATED (decline at T30, sustained at T35) --
but Xiso declines at ~35C (Ferm01), so the gate would suppress the clock exactly where Xiso needs it.
So the test model uses a T-INDEPENDENT clock (z=exp(-t_ind/tau)), independent sol/IB gates, fit on
the 3 Xiso runs. If tau lands near LDH's ~0.7-1.5h, the clock is shared (general induction burden)."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

S_IN = 400.0
T_REF = 30.0
QS_CAP = 2.0
EPS = 1e-6


def load_xiso_runs(path):
    df = pd.read_excel(path, sheet_name="All_Time_Points")
    Tcol = [c for c in df.columns if "Temperature" in c][0]
    runs = []
    for rid, g in df.groupby("run_name"):
        g = g.sort_values("Process time [h]").reset_index(drop=True)
        t = g["Process time [h]"].to_numpy(float)
        feed_mL = g["Feed [mL]"].fillna(0).to_numpy(float)
        fr = np.zeros_like(feed_mL)
        dt = np.diff(t)
        fr[:-1] = np.where(dt > 0, np.diff(feed_mL) / dt, 0.0)
        fr = np.maximum(fr, 0.0) / 1000.0  # L/h
        V = g["Volume corrected [mL]"].ffill().bfill().to_numpy(float) / 1000.0
        T = g[Tcol].to_numpy(float)
        ind = g["Is induced [-]"].fillna(0).to_numpy(float) > 0.5
        tind = np.zeros_like(t)
        if ind.any():
            t0i = float(t[ind][0])
            tind = np.where(t >= t0i, t - t0i, 0.0)
        else:
            t0i = float(t[0])
        sm = np.isfinite(g["Protein soluble [g/L]"])
        to = t[sm]
        X = g["Biomass [g/L]"].to_numpy(float)[sm]
        Ps = g["Protein soluble [g/L]"].to_numpy(float)[sm]
        Pi = g["Protein IB [g/L]"].to_numpy(float)[sm]
        ok = np.isfinite(X) & np.isfinite(Ps) & np.isfinite(Pi)
        runs.append(dict(run_id=str(rid), t0=float(to[ok][0]), tend=float(t[-1]),
                         X0=float(X[ok][0]), Psol0=float(Ps[ok][0]), Pib0=float(Pi[ok][0]),
                         t_online=t, feed_rate_L_h=fr, V_L=V, temp=T,
                         is_induced=ind.astype(float), ind_time=tind,
                         t_obs=to[ok], obs=dict(X=X[ok], P_sol=Ps[ok], P_IB=Pi[ok]),
                         T_induction=float(t0i)))
    return runs


# Decoupled model with a T-INDEPENDENT time clock (z=exp(-t_ind/tau)), independent sol/IB gates.
PARAMS = ["Y_xs", "qP_sol_max", "qP_IB_max", "K_S",
          "theta_sol_0", "theta_sol_T", "theta_ib_0", "theta_ib_T", "tau_IB"]
INIT = [0.45, 0.03, 0.03, 0.5, 2.0, 2.0, 2.0, 2.0, 3.0]
LO = [0.20, 1e-4, 1e-4, 1e-3, -4.0, 0.5, -4.0, 0.5, 0.5]
HI = [0.80, 0.10, 0.30, 2.0, 6.0, 8.0, 6.0, 8.0, 20.0]


def rhs_factory(p, run):
    t_arr, feed_arr, V_arr, T_arr, tind_arr = (run["t_online"], run["feed_rate_L_h"],
                                                run["V_L"], run["temp"], run["ind_time"])
    t_lo, t_hi = float(t_arr[0]), float(t_arr[-1])

    def rhs(t, y):
        X = y[0] if y[0] > EPS else EPS
        tc = t if t_lo <= t <= t_hi else (t_lo if t < t_lo else t_hi)
        V = np.interp(tc, t_arr, V_arr)
        if V < 1e-4:
            V = 1e-4
        feed = max(np.interp(tc, t_arr, feed_arr), 0.0)
        T = np.interp(tc, t_arr, T_arr)
        tind = np.interp(tc, t_arr, tind_arr)
        is_ind = 1.0 if tind > 0 else 0.0
        qs = min(feed * S_IN / (X * V), QS_CAP)
        D = feed / V
        mu = p[0] * qs
        gs = 1.0 / (1.0 + np.exp(-(p[4] - p[5] * (T - T_REF))))
        gib = 1.0 / (1.0 + np.exp(-(p[6] + p[7] * (T - T_REF))))
        mon = qs / (qs + p[3])
        z = float(np.exp(-max(tind, 0.0) / p[8]))
        qPs = p[1] * mon * gs
        qPi = p[2] * mon * gib * z
        return (mu * X - D * X, is_ind * qPs * X - D * y[1], is_ind * qPi * X - D * y[2])
    return rhs


def rk4(rhs, y0, t0, tend, h=0.1):
    n = max(int(np.ceil((tend - t0) / h)), 1)
    hh = (tend - t0) / n
    y = np.array(y0, float)
    for i in range(n):
        t = t0 + i * hh
        k1 = np.asarray(rhs(t, y))
        k2 = np.asarray(rhs(t + hh / 2, y + hh / 2 * k1))
        k3 = np.asarray(rhs(t + hh / 2, y + hh / 2 * k2))
        k4 = np.asarray(rhs(t + hh, y + hh * k3))
        y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def simulate(p, run):
    rhs = rhs_factory(p, run)
    obs_t = run["t_obs"]
    out = {k: np.full(len(obs_t), np.nan) for k in ("X", "P_sol", "P_IB")}
    # integrate to each obs time, starting AT the first sample with its X/P_sol/P_IB as the IC
    y = np.array([run["X0"], run["Psol0"], run["Pib0"]])
    t = run["t0"]
    for i, te in enumerate(obs_t):
        if te <= t:
            y_at = y if te == t else np.array([run["X0"], 0.0, 0.0])
        else:
            try:
                y = rk4(rhs, y, t, te, 0.1)
                t = te
                y_at = y
            except Exception:
                y_at = np.array([np.nan] * 3)
        out["X"][i], out["P_sol"][i], out["P_IB"][i] = y_at
    return out


def fit(runs, seed=42, n_starts=4):
    rng = np.random.default_rng(seed)
    ranges = {k: max(np.nanmax([r["obs"][k] for r in runs]) - np.nanmin([r["obs"][k] for r in runs]), 1e-3)
              for k in ("X", "P_sol", "P_IB")}

    def resid(vec):
        if np.any(vec < LO) or np.any(vec > HI):
            return np.full(999, 1e3)
        r = []
        for run in runs:
            o = simulate(vec, run)  # vec is a list/array; rhs_factory indexes it
            for k in ("X", "P_sol", "P_IB"):
                d = (o[k] - run["obs"][k]) / ranges[k]
                r.extend(d[np.isfinite(d)])
        return np.array(r) if r else np.full(999, 1e3)

    best = None
    for s in range(n_starts):
        x0 = np.array(INIT) * (1 + 0.3 * rng.standard_normal(len(INIT)))
        x0 = np.clip(x0, LO, HI)
        try:
            res = least_squares(resid, x0, bounds=(LO, HI), method="trf", max_nfev=400)
            cost = float(np.mean(res.fun ** 2))
            if best is None or cost < best[1]:
                best = (res.x, cost)
        except Exception:
            pass
    return best


if __name__ == "__main__":
    runs = load_xiso_runs("../../Xiso/Xiso_consolidated_results.xlsx")
    print(f"loaded {len(runs)} Xiso runs: {[(r['run_id'], len(r['t_obs'])) for r in runs]}")
    best = fit(runs, n_starts=5)
    p = dict(zip(PARAMS, best[0]))
    print(f"\nXiso fit: cost={best[1]:.4f}")
    print("params:", {k: round(v, 4) for k, v in p.items()})
    print(f"\n=> tau_IB (Xiso) = {p['tau_IB']:.2f} h   (LDH v2 clock ~0.7 h; raw IB decline ~1.3-1.9 h)")
    for run in runs:
        o = simulate(best[0], run)
        dIB = o["P_IB"][-1] - o["P_IB"][-2]
        dIB_obs = run["obs"]["P_IB"][-1] - run["obs"]["P_IB"][-2]
        print(f"  {run['run_id']}: IBend {o['P_IB'][-1]:.2f}/{run['obs']['P_IB'][-1]:.2f}  "
              f"dIB {dIB:+.3f}/{dIB_obs:+.3f}  | solend {o['P_sol'][-1]:.2f}/{run['obs']['P_sol'][-1]:.2f}")

    # Plot: 3 Xiso runs x (P_sol, P_IB), obs vs fitted model
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Tcol_lookup = {"00_Ferm01": 35, "01_Ferm02": 25, "02_Ferm03": 29}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, run in enumerate(runs):
        o = simulate(best[0], run)
        # dense trajectory for the model line
        rhs = rhs_factory(best[0], run)
        y = np.array([run["X0"], run["Psol0"], run["Pib0"]])
        t = run["t0"]
        td = [t]; yd = [y.copy()]
        for te in run["t_obs"][1:]:
            y = rk4(rhs, y, t, te, 0.1); t = te
            td.append(t); yd.append(y.copy())
        td = np.array(td); yd = np.array(yd)
        for row, (k, lab) in enumerate([("P_sol", "P_sol [g/L]"), ("P_IB", "P_IB [g/L]")]):
            ax = axes[row][col]
            ax.scatter(run["t_obs"], run["obs"][k], c="k", s=30, zorder=5, label="observed")
            ax.plot(td, yd[:, 1 + row], "-", c="#009E73" if row == 0 else "#D55E00", lw=1.7,
                    label="model (T-indep clock)")
            ax.set_title(f"{run['run_id']}  (T~{Tcol_lookup.get(run['run_id'], '?')})", fontsize=9)
            if col == 0:
                ax.set_ylabel(lab, fontsize=9)
            if row == 1:
                ax.set_xlabel("process time [h]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Xiso transfer: decoupled + T-INDEPENDENT clock  (tau_IB={p['tau_IB']:.2f} h, "
                 f"vs LDH ~0.7-1.5 h)\nFerm01 IB observed declines (-0.16) -- does the model capture it?",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("results/xiso_transfer.png", dpi=120)
    plt.close(fig)
    print("saved results/xiso_transfer.png")
    print("DONE")
