"""
2-factor view of partition IB for the LDH FCCD (11 runs).

Figure 1: partition vs temperature, points colored by qs  -> shows that the
          30 C mixed band fans out with qs while 25/35 C are pinned.
Figure 2: full-quadratic response surface over (T, qs)     -> formal 2-factor
          model (the textbook FCCD analysis); reports R2 and LOO-Q2.

Reads Run_Summary from LDH_consolidated_results.xlsx (needs T_setpoint,
qs_initial, Protein IB/soluble finals -- produced by consolidate_results.py).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

FILE = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\LDH_consolidated_results.xlsx"
OUT1 = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\partition_vs_T_colored_by_qs.png"
OUT2 = r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH\partition_response_surface_T_qs.png"

df = pd.read_excel(FILE, sheet_name='Run_Summary')

qs_col  = [c for c in df.columns if 'qs_initial' in c][0]
tset_col = [c for c in df.columns if 'T_setpoint' in c][0]
ib_col, sol_col = 'Protein IB [g/L]_final', 'Protein soluble [g/L]_final'

df['partition_IB'] = df[ib_col] / (df[ib_col] + df[sol_col])
df = df.dropna(subset=[tset_col, qs_col, 'partition_IB']).reset_index(drop=True)

T = df[tset_col].values.astype(float)
qs = df[qs_col].values.astype(float)
P = df['partition_IB'].values.astype(float)
names = df['run_name'].values

# =========================================================================
# FIGURE 1 -- partition vs T, colored by qs
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(T, P, c=qs, cmap='viridis', s=220, edgecolors='black',
                linewidths=1.5, zorder=5, vmin=0.2, vmax=0.6)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('qs,initial [g/g/h]', fontsize=13, fontweight='bold')

for ti, pi, nm in zip(T, P, names):
    ax.annotate(nm, (ti, pi), textcoords='offset points', xytext=(8, 7),
                fontsize=9, fontweight='bold')

# faint T-only sigmoid: what the 1-D model sees (ignores qs)
def sigmoid(T, s_T, T_50):
    return 1 / (1 + np.exp(-s_T * (T - T_50)))
popt, _ = curve_fit(sigmoid, T, P, p0=[2, 30], maxfev=10000)
xf = np.linspace(23, 37, 200)
ax.plot(xf, sigmoid(xf, *popt), '--', color='gray', linewidth=2.2, alpha=0.6,
        label='T-only sigmoid (ignores qs)')

ax.set_xlabel('Induction temperature [°C]', fontsize=14, fontweight='bold')
ax.set_ylabel('Partition IB  =  IB / (IB + soluble)', fontsize=14, fontweight='bold')
ax.set_title('Partition IB vs temperature — colored by qs\n'
             '30 °C band fans out with qs; 25 °C and 35 °C are pinned',
             fontsize=13.5, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(23, 37)
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig(OUT1, dpi=300, bbox_inches='tight')
plt.close()

# =========================================================================
# FIGURE 2 -- full-quadratic response surface over (T, qs)
# =========================================================================
# code factors to [-1, 1]:  x1 = (T-30)/5,  x2 = (qs-0.4)/0.2
x1 = (T - 30.0) / 5.0
x2 = (qs - 0.4) / 0.2

def design(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return np.column_stack([np.ones_like(a), a, b, a**2, b**2, a * b])

X = design(x1, x2)
beta, *_ = np.linalg.lstsq(X, P, rcond=None)
P_hat = X @ beta
ss_res = float(np.sum((P - P_hat) ** 2))
ss_tot = float(np.sum((P - np.mean(P)) ** 2))
r2 = 1 - ss_res / ss_tot

# leave-one-out Q2
press = 0.0
n = len(P)
for i in range(n):
    mask = np.ones(n, dtype=bool); mask[i] = False
    b, *_ = np.linalg.lstsq(X[mask], P[mask], rcond=None)
    row = np.array([1.0, x1[i], x2[i], x1[i] ** 2, x2[i] ** 2, x1[i] * x2[i]])
    press += (P[i] - row @ b) ** 2
q2 = 1 - press / ss_tot

# prediction grid (real units)
Tg, Qg = np.meshgrid(np.linspace(24, 36, 220), np.linspace(0.15, 0.65, 220))
Pg = design(((Tg - 30.0) / 5.0).ravel(), ((Qg - 0.4) / 0.2).ravel()) @ beta
Pg = np.clip(Pg.reshape(Tg.shape), 0, 1)

fig, ax = plt.subplots(figsize=(10, 7))
cs = ax.contourf(Tg, Qg, Pg, levels=np.linspace(0, 1, 21), cmap='RdYlBu_r')
plt.colorbar(cs, ax=ax, label='Predicted partition IB')
ax.contour(Tg, Qg, Pg, levels=[0.1, 0.5, 0.9], colors='black',
           linewidths=0.8, linestyles='--')
ax.scatter(T, qs, c=P, cmap='RdYlBu_r', s=170, edgecolors='black',
           linewidths=1.5, vmin=0, vmax=1, zorder=5)
for ti, qi, nm in zip(T, qs, names):
    ax.annotate(nm, (ti, qi), textcoords='offset points', xytext=(7, 5), fontsize=8)
ax.set_xlabel('Induction temperature [°C]', fontsize=14, fontweight='bold')
ax.set_ylabel('qs,initial [g/g/h]', fontsize=14, fontweight='bold')
ax.set_title(f'Partition IB response surface — full quadratic\n'
             f'R² = {r2:.3f},  Q²(LOO) = {q2:.3f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT2, dpi=300, bbox_inches='tight')
plt.close()

# =========================================================================
print("=" * 70)
print("FULL QUADRATIC RSM   (coded:  x1 = (T-30)/5,  x2 = (qs-0.4)/0.2)")
print("=" * 70)
for lab, c in zip(['intercept b0', 'b1  (T)', 'b2  (qs)', 'b11 (T²)',
                   'b22 (qs²)', 'b12 (T·qs)'], beta):
    print(f"  {lab:14s} = {c:+.4f}")
print(f"\n  R² (fit) = {r2:.4f}")
print(f"  Q² (LOO) = {q2:.4f}")
print("\nObservations:")
print("  • b1 (T) is by far the largest term -> temperature is the primary switch.")
print("  • The qs terms (b2, b22, b12) are small but nonzero; they capture the")
print("    secondary modulation that is visible only inside the 30 °C mixed band.")
