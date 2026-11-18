"""
Partition-IB RSM variants for the LDH FCCD (11 runs).

Diagnoses whether qs really matters once the model is allowed to see it:

  Model 1  raw, full quadratic            (baseline; buries the qs effect)
  Model 2  raw, augmented with x1^2 * x2  (lets qs act only in the center-T band)
  Model 3  logit(partition), full quadratic
  Model 4  logit(partition), augmented    (bounded + qs-localised)

Plots surfaces for Model 2 and Model 4; prints a 4-model comparison and the
predicted-vs-observed 3x3 (T x qs) grid for the two augmented models.

Factors coded:  x1 = (T-30)/5 in [-1,1],  x2 = (qs-0.4)/0.2 in [-1,1].
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

FILE = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\LDH_consolidated_results.xlsx"
OUT_AUG = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\partition_response_surface_augmented.png"
OUT_LOGIT = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\partition_response_surface_logit.png"

EPS = 0.01  # clip for logit transform at exact 0/1

# ---- data ----
df = pd.read_excel(FILE, sheet_name="Run_Summary")
qs_col = [c for c in df.columns if "qs_initial" in c][0]
tset_col = [c for c in df.columns if "T_setpoint" in c][0]
ib_col, sol_col = "Protein IB [g/L]_final", "Protein soluble [g/L]_final"
df["partition_IB"] = df[ib_col] / (df[ib_col] + df[sol_col])
df = df.dropna(subset=[tset_col, qs_col, "partition_IB"]).reset_index(drop=True)

T = df[tset_col].values.astype(float)
qs = df[qs_col].values.astype(float)
P = df["partition_IB"].values.astype(float)
names = df["run_name"].values
x1 = (T - 30.0) / 5.0
x2 = (qs - 0.4) / 0.2

# ---- model machinery ----
TERMS = {
    "1": lambda a, b: np.ones_like(a),
    "x1": lambda a, b: a,
    "x2": lambda a, b: b,
    "x1^2": lambda a, b: a**2,
    "x2^2": lambda a, b: b**2,
    "x1*x2": lambda a, b: a * b,
    "x1^2*x2": lambda a, b: (a**2) * b,
}
QUAD = ["1", "x1", "x2", "x1^2", "x2^2", "x1*x2"]
AUG = ["1", "x1", "x2", "x1^2", "x2^2", "x1*x2", "x1^2*x2"]


def design(a, b, keys):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.column_stack([TERMS[k](a, b) for k in keys])


def fit_eval(a, b, y, keys):
    X = design(a, b, keys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot
    press = 0.0
    n = len(y)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        bb, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        press += (y[i] - X[i] @ bb) ** 2
    q2 = 1 - press / ss_tot
    return beta, r2, q2


def predict_grid(beta, keys, logit=False):
    Tg, Qg = np.meshgrid(np.linspace(24, 36, 220), np.linspace(0.15, 0.65, 220))
    eta = (
        design(
            ((Tg - 30) / 5).ravel(),
            ((Qg - 0.4) / 0.2).ravel(),
            AUG if "x1^2*x2" in keys else keys,
        )
        @ beta
    )
    eta = eta.reshape(Tg.shape)
    return 1 / (1 + np.exp(-eta)) if logit else np.clip(eta, 0, 1), Tg, Qg


def pred_at(Tv, qv, beta, keys, logit=False):
    eta = design(np.array([(Tv - 30) / 5]), np.array([(qv - 0.4) / 0.2]), keys) @ beta
    eta = float(np.ravel(eta)[0])
    return 1 / (1 + np.exp(-eta)) if logit else float(min(max(eta, 0.0), 1.0))


# ---- fits ----
b_q_raw, r2_q_raw, q2_q_raw = fit_eval(x1, x2, P, QUAD)
b_a_raw, r2_a_raw, q2_a_raw = fit_eval(x1, x2, P, AUG)
Pc = np.clip(P, EPS, 1 - EPS)
Plog = np.log(Pc / (1 - Pc))
b_q_log, r2_q_log, q2_q_log = fit_eval(x1, x2, Plog, QUAD)
b_a_log, r2_a_log, q2_a_log = fit_eval(x1, x2, Plog, AUG)

# ---- observed 3x3 grid ----
obs = {}
for tv, qv, pv in zip(T, qs, P):
    obs.setdefault((round(tv), round(qv, 1)), []).append(pv)
obs = {k: float(np.mean(v)) for k, v in obs.items()}
grid_T = [25, 30, 35]
grid_Q = [0.2, 0.4, 0.6]


def print_grid(title, predfn):
    print(f"\n{title}\n            " + "  ".join(f"qs={q:<3}" for q in grid_Q))
    for tv in grid_T:
        row = "  ".join(f"{predfn(tv, q):6.3f}" for q in grid_Q)
        ob = "  ".join(f"{obs.get((tv, q), float('nan')):6.3f}" for q in grid_Q)
        print(f"  T={tv}°C  pred: {row}   | obs: {ob}")


# ================= comparison table =================
print("=" * 78)
print("MODEL COMPARISON  (R2/Q2 on the scale the model was fit on)")
print("=" * 78)
print(f"{'model':24s} {'terms':>6} {'R2':>8} {'Q2(LOO)':>9}")
print("-" * 78)
for nm, r2, q2, k in [
    ("raw quadratic", r2_q_raw, q2_q_raw, QUAD),
    ("raw augmented (+x1^2x2)", r2_a_raw, q2_a_raw, AUG),
    ("logit quadratic", r2_q_log, q2_q_log, QUAD),
    ("logit augmented", r2_a_log, q2_a_log, AUG),
]:
    print(f"{nm:24s} {len(k):>6d} {r2:>8.4f} {q2:>9.4f}")

print("\n" + "-" * 78)
print("AUGMENTED RAW coefficients:")
for k, c in zip(AUG, b_a_raw):
    print(f"  {k:8s} = {c:+.4f}")
print("\nAUGMENTED LOGIT coefficients (logit scale):")
for k, c in zip(AUG, b_a_log):
    print(f"  {k:8s} = {c:+.4f}")

print("\n" + "-" * 78)
print("Key diagnostic -- qs coefficient b2 (the partition-slope vs qs at 30 C = x1=0):")
print(f"  raw quadratic   b2 = {b_q_raw[2]:+.4f}   (forced global avg -> buries qs)")
print(
    f"  raw augmented   b2 = {b_a_raw[2]:+.4f}   (true center-T qs slope; x1^2*x2 = {b_a_raw[6]:+.4f} suppresses it at T extremes)"
)
print(f"  logit augmented b2 = {b_a_log[2]:+.4f} (logit scale)")

print_grid(
    "Predicted vs observed partition  [raw augmented]",
    lambda tv, qv: pred_at(tv, qv, b_a_raw, AUG, logit=False),
)
print_grid(
    "Predicted vs observed partition  [logit augmented]",
    lambda tv, qv: pred_at(tv, qv, b_a_log, AUG, logit=True),
)


# ================= plots =================
def surface(beta, keys, logit, out, title):
    Pg, Tg, Qg = predict_grid(beta, keys, logit=logit)
    fig, ax = plt.subplots(figsize=(10, 7))
    cs = ax.contourf(Tg, Qg, Pg, levels=np.linspace(0, 1, 21), cmap="RdYlBu_r")
    plt.colorbar(cs, ax=ax, label="Predicted partition IB")
    ax.contour(
        Tg,
        Qg,
        Pg,
        levels=[0.1, 0.5, 0.9],
        colors="black",
        linewidths=0.8,
        linestyles="--",
    )
    ax.scatter(
        T,
        qs,
        c=P,
        cmap="RdYlBu_r",
        s=170,
        edgecolors="black",
        linewidths=1.5,
        vmin=0,
        vmax=1,
        zorder=5,
    )
    for ti, qi, nm in zip(T, qs, names):
        ax.annotate(nm, (ti, qi), textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel("Induction temperature [°C]", fontsize=14, fontweight="bold")
    ax.set_ylabel("qs,initial [g/g/h]", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


r2a = r2_a_raw
q2a = q2_a_raw
surface(
    b_a_raw,
    AUG,
    False,
    OUT_AUG,
    f"Partition IB response surface — augmented (+x1²·qs)\nR² = {r2a:.3f},  Q²(LOO) = {q2a:.3f}  [raw, clipped 0–1]",
)
surface(
    b_a_log,
    AUG,
    True,
    OUT_LOGIT,
    f"Partition IB response surface — logit-transformed, augmented\nR² = {r2_a_log:.3f},  Q²(LOO) = {q2_a_log:.3f}  [logit scale, eps={EPS}]",
)

print("\nSaved:", OUT_AUG, "\n      ", OUT_LOGIT)
