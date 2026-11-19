"""Test M0_aload_qPTexp_yxt (T-dependent biomass yield): does Y_xs(T)=Y_xs*exp(-beta_T*max(T-30,0))
fix the systematic T35 biomass over-prediction (F07 +4.0, F19/F12 +1.3), AND does the fit keep CER
matching (alpha_T and beta_T both route carbon to CO2 at T35 -> coupled; if CER bias grows, the T35
biomass gap is a volume effect, not yield)?"""
import data as D
import fit as F
import numpy as np
from models import simulate_run

TSET = {"F09": 25, "F16": 25, "F10": 25, "F17": 30, "F13": 30, "F14": 30, "F15": 30,
        "F18": 30, "F07": 35, "F19": 35, "F12": 35}
b = D.load_dataset(*D.split_for_fold("full"))
rdm = {r["run_id"]: r for r in b["train"]}

print("fitting baseline (M0_aload_qPTexp)...")
base = F.fit_model("M0_aload_qPTexp", b, seed=42, n_starts=2, max_nfev=400)
print("fitting M0_aload_qPTexp_yxt...")
yxt = F.fit_model("M0_aload_qPTexp_yxt", b, seed=42, n_starts=3, max_nfev=400)
print("\nbaseline:", {k: round(v, 4) for k, v in base.params.items()})
print("yxt     :", {k: round(v, 4) for k, v in yxt.params.items()})
print(f"\ncost: baseline {base.cost:.4f}  yxt {yxt.cost:.4f}")

print(f"\n{'run':>4}{'T':>3}  X_resid_base  X_resid_yxt   (T35 should improve)")
for rid in ["F09", "F16", "F10", "F17", "F13", "F14", "F15", "F18", "F07", "F19", "F12"]:
    r = rdm[rid]
    rb = simulate_run("M0_aload_qPTexp", base.params, r, b["c_fold"], eval_gas=False)["obs"]
    ry = simulate_run("M0_aload_qPTexp_yxt", yxt.params, r, b["c_fold"], eval_gas=False)["obs"]
    xb = rb["X"][-1] - r["obs"]["X"][-1]
    xy = ry["X"][-1] - r["obs"]["X"][-1]
    print(f"{rid:>4}{TSET[rid]:>3}   {xb:+7.2f}      {xy:+7.2f}")

# CER check: model vs observed CER post-induction, RQ-trusted, by temperature.
print("\nCER bias (model - obs, mmol/L/h, median, RQ-trusted post-induction):")
for tag, res, mdl in [("baseline", base, "M0_aload_qPTexp"), ("yxt", yxt, "M0_aload_qPTexp_yxt")]:
    print(f"  {tag}:")
    for T in [25, 30, 35]:
        biases = []
        for rid in [k for k, v in TSET.items() if v == T]:
            r = rdm[rid]
            sim = simulate_run(mdl, res.params, r, b["c_fold"], eval_gas=True)
            tg = np.asarray(r["t_gas"], float)
            obs_cer = np.asarray(r["gas"]["CER"], float)
            obs_our = np.asarray(r["gas"]["OUR"], float)
            m_cer = sim["gas"]["CER"]
            tind = float(r["T_induction"])
            mask = (tg >= tind) & np.isfinite(obs_cer) & np.isfinite(obs_our) & np.isfinite(m_cer)
            rq = np.where(np.abs(obs_our) > 1e-6, obs_cer / obs_our, np.nan)
            mask &= (rq > 0.8) & (rq < 1.2) & (obs_cer > 10)
            if mask.sum() > 0:
                biases.extend((m_cer[mask] - obs_cer[mask]).tolist())
        if biases:
            print(f"    T={T}: median {np.median(biases):+.2f}  (n={len(biases)})")
print("DONE")
