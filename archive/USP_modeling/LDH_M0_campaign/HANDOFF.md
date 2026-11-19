# LDH mechanistic modeling playground — handoff

Last updated: 2026-09-26

Self-contained context for the fully-mechanistic (no neural network) ODE model
of the LDH fermentation built in the USP repository. Read this before changing
the model or interpreting results. The companion **hybrid** ODE-ML model lives
in a separate repo; see "Relationship to the hybrid model" below.

## Purpose and status

Build a self-contained, **fully-mechanistic** (parametric, no NN) dynamic model
of the LDH *E. coli* fed-batch that:

1. reuses the hybrid model's **grouped train/test split** (folds A/B/C/full);
2. models biomass, soluble product, and IB product mechanistically;
3. **uses offgas (CER/OUR)** — a dense online observable the hybrid model does
   not use — as additional information to identify biomass stoichiometry;
4. is **identifiability-first**: no parameter or latent state is added that the
   data cannot identify; every extension must survive a profile-likelihood +
   cross-fold-stability check.

Guiding principle, stated by the user: *focus on identifiability — do not add
parameters / latent states that cannot be identified.*

**Update (2026-09-04):** two pipeline data corrections (induction timing in all 11
runs; F19 IB artifact) were applied, and the canonical model was **re-fit and
re-validated on the corrected data — every verdict stands**: Y_xs ≈ 0.363 stable,
α refined 0.096 → 0.090 (η(35)=0.64, still CER-validated biology), held-out
metrics unchanged, adequacy verdict unchanged. Full result in the findings-log
section *"Data corrections (2026-09-04) — inputs changed; refit COMPLETED"* at the
end of this document; pre-correction artifacts archived at
`results/M0_aload_qPTexp_PRECORRECTION_20260724/`. **Specific `M0_aload_qPTexp`
numbers in this document are superseded by that section** except where marked.

Status (2026-07-24): framework (`data.py`, `stoich.py`, `metrics.py`, `models.py`,
`fit.py`, `run_playground.py`, `base_reconstruct.py`) implemented and smoke-tested.
**Adopted model: `M0_aload_qPTexp`** (sparse obs, no gas) on the **N-floor base-volume
correction** (default in `data.py`). Two structural gains this session:

1. **Base/NH₄OH volume reconstruction** — the missing base (unlogged) is reconstructed
   from the raw Pump2 signal (shape) × the N-balance floor (scale). Corrects the
   under-counted reactor volume → **Y_xs refined 0.351 → ~0.36**, and **reconciles the
   gas-vs-biomass yield conflict** (the M1a overprediction was a volume artifact; offgas
   no longer rejected for biomass overprediction).
2. **qP(T) temperature suppression of total production** — `qP *= exp(−α·max(T−30,0))`,
   α≈0.096/°C (first-order proteostasis / σ³² heat-shock burden). Identifiable; fixes the
   model's largest error (every T35 run was overpredicted +1.2–2.0 g/L); 35% cost cut.

**Validated this session (2026-07-24):** α (qP(T)) is **real biology, not an IB-under-recovery
artifact** — the CER carbon-balance test (`alpha_cer_test.py`) shows the α-removed product carbon
appears as CO₂ (model CER matches observed at T35; under-recovery would leave it ~10 mmol/L/h low;
under-recovery ruled out, only the within-biology split — proteolysis vs stress respiration — is
unresolvable by the carbon balance).

Rejected (identifiability-first, all logged below): offgas gas observables M1a (reconciled under
corrected volume but still adds noise), **M1b maintenance (re-tested 2026-07-24: in-sample
identifiable but OVERFITS — held-out F17 collapses +0.42→−4.58; m_s=0.076 absorbs the carbon-balance
systematic error, not real maintenance)**, burden state (collinear with qS-load),
**production-proportional burden β·qP·X (Task D: R²≈0, non-identifiable)**, Haldane substrate-
inhibition (Ki→bound; qS compressed), induction-decay (overfits), soluble folding-transit delay
τ_fold (→0; qS-varying φ already captures the dynamics). Residual, data-limited: the qS-peak
(corner runs), partition identifiability (a_qS/θ_T, only T30 informs it), ±50% volume, and the T35
IB-under-recovery caveat on α (now resolved — see above).

**Where this leaves it (the key reframe):** the **parameter-space** of the current structure is
exhausted — every metabolic-burden/maintenance parameter fails because the gas balance carries a
~10% systematic carbon-accounting error dominated by the **±50% base-volume bracket**. The hope was
that those rejections might be **volume artifacts** — `m_s`/burden real but unidentifiable *because*
the volume injects too much noise — and that pinning the volume from the respiratory acid load ∫CER
(Task E.1) would un-freeze them. **That lever is now tested and CLOSED (2026-07-24):** ∫CER does NOT
pin the volume — the 1:1 CO₂ stoichiometry over-predicts the bracket-scale base by ~6×, carbon
closure (~103%) says the CO₂ is real, so ~65% is **stripped, not neutralized**, and the
neutralization fraction is a free ~0.2–0.4 parameter (CER also tracks feed near-collinearly, so the
shape carries no CO₂-specific signal). The volume therefore stays at its ±50% bracket, and the
gas-side identifiability ceiling is **not** shown to be a volume artifact — it remains most plausibly
real/biological. The only true volume pin is a weighed base bottle on the LDH reactor (MF2-R6).
Findings are logged at the bottom; **Decision** below summarises current verdicts.

## File map

```
examples/LDH/modeling_playground/
  HANDOFF.md          this document (living: update the findings log after each run)
  data.py             load consolidated processed xlsx; per-run arrays; feed-rate
                      cleaning; D derivation; train-only leakage-safe constants;
                      N-floor base-volume correction (CORRECT_BASE_VOLUME, default on)
  base_reconstruct.py NH4OH base reconstruction: Pump2 signal (shape) x N-balance
                      floor (scale); pump-shape caching; bracket helpers
  stoich.py           fixed composition + degree-of-reduction; carbon+electron
                      balance -> predicted CER/OUR (zero free parameters)
  metrics.py          metric suite + macro aggregation (range-weighted, bias, NLL)
  models.py           M0/M1a/M1b/+aload/+qPT/+foldlag RHS, parameter specs, simulate_run()
  fit.py              range-weighted residual objective, multistart, profile likelihood
  run_playground.py   CLI + results output (params.json, metrics.md, plots, profile)
  burden_independence.py   Task A: burden-vs-IB independence from qS (diagnostic)
  burden_identifiability.py / burden_plots.py  Task A: a_b profile + verdict plot
  base_volume_refit.py / gas_volume_check.py   base bracket re-fit + gas-conflict check
  qpt_test.py / haldane_test.py / foldlag_test.py   production-model variant tests
  alpha_cer_test.py        (2026-07-24) α biology test: CER residual at T35 rules out IB under-recovery
  burden_co2_test.py       (2026-07-24) Task D: production-proportional burden β·qP·X — non-identifiable
  m1b_maintenance_test.py  (2026-07-24) M1b Pirt maintenance re-test — in-sample identifiable, overfits
  base_acid_reconstruct.py (2026-07-24) Task E.1: base-volume from respiratory acid load ∫CER — TESTED, NOT pinned (CO2 stripped)
  o2_limit_test.py        (2026-07-24) Task E.2: O2-transfer limitation as qS-peak driver — TESTED, NOT supported (demand met, RQ~1)
  smet_test.py            (2026-07-24) S_met cumulative-load Haldane (Müller 2023, A1) — TESTED, NOT supported (qS-peak is T-driven)
  soft_sensor.py          (2026-07-24) Plan B Stage 1: dense μ/qP from gas + per-run scale s — T30 gas validated; dense qP residual-sensitive
  dual_balance_test.py    (2026-07-24) Plan B Stage 2: dual C+e⁻ balance — DEAD (γX≈γP≈4.1, can't split growth/product)
  noise_floor_test.py     (2026-07-24) F13/F14/F15 triplicate noise floor — qS-peak is a dilution artifact (Ptot flat, X rises)
  model_adequacy_test.py  (2026-07-24) Option A: M0_aload_qPTexp adequate — qs0.6 product within noise; biomass resid = volume
  psplit_test.py          (2026-07-27) Late P_IB decline: refold ruled out (no sol boost); ibdeg weak; dilution ruled out
  hillib_test.py          (2026-07-27) Cooperative (Hill) IB formation — RULED OUT; IB decline = total-production-front-loading symptom
  qp_decline_test.py      (2026-07-27) Production decline is REAL & beyond qS, in LDH + Xiso -> general induction burden
  ibtransient_test.py     (2026-07-27) IB-specific transient decline — RULED OUT (5th IB mechanism); trajectory not identifiable
  nn_qp_extract.py        (2026-07-27) Extracted the hybrid NN (m13) learned qP_sol/qP_IB -> oracle for the parametric form
  smet_independent_test.py (2026-07-27) M0_smet_independent: independent S_met-driven rates (NN port) — TESTED
  smet_ibdeg_test.py        (2026-07-28) M0_smet_independent_ibdeg: net-negative IB channel (opt 1+3) — REJECTED (overfits)
  explore_psol_pib.py       (2026-07-28) data look: sol sustained T<=30/absent T35, IB T30-transient/T35-sustained
  decoupled_test.py         (2026-07-28) M_decoupled_ibtransient (v1, complementary gates) — REJECTED (starves IB)
  decoupled_v2_test.py      (2026-07-28) M_decoupled_v2 (independent gates + T30 clock) — captures T30 IB decline
  decoupled_v3_test.py      (2026-07-28) M_decoupled_v3 (delayed shutoff) — peak+decline in-sample, overfits
  explore_psol_pib.py       (2026-07-28) μ-vs-qS growth-burden check + feed-shape (μ decline = linear feed, not burden)
  yxt_test.py               (2026-07-28) M0_aload_qPTexp_yxt (T-dependent Y_xs) — REJECTED (too blunt)
  plot_decoupled_comparison.py / plot_biomass.py  (2026-07-28) comparison + biomass-X plots
  proteostasis_test.py      (2026-09-04) Proteostasis v1 (rescue capacity): T-half SUPPORTED (T50_m≈30.3, σ=0.66≡η(35)=0.64); capacity FALSIFIED (C→0)
  proteostasis_v2_test.py   (2026-09-04) Proteostasis v2 (capacity-limited k_f): REJECTED (degenerate no-IB corner, 10× cost)
  f10_pulse_biomass_test.py (2026-09-26) F10 glucose-pulse biomass check: overshoot CONFIRMED + pool-quantified via counterfactual; fixed rhs k_deg shadowing bug
  results/            gitignored outputs (results/<model>/<split>/...)
```

Input data (already processed; regenerate with `examples/LDH/consolidate_results.py`):

```
examples/LDH/LDH_consolidated_results_processed.xlsx   sheet "All_Processed"
```

## Dataset and FCCD design (11 runs)

| Run | T [°C] | qS₀ target [g/g/h] | Role                |
| --- | ------:| ------------------:| ------------------- |
| F09 | 25     | 0.2                | corner              |
| F10 | 25     | 0.6                | corner              |
| F16 | 25     | 0.4                | face center         |
| F17 | 30     | 0.2                | low-qS face center  |
| F13 | 30     | 0.4                | center replicate    |
| F14 | 30     | 0.4                | center replicate    |
| F15 | 30     | 0.4                | center replicate    |
| F18 | 30     | 0.6                | high-qS face center |
| F07 | 35     | 0.2                | corner              |
| F12 | 35     | 0.6                | corner              |
| F19 | 35     | 0.4                | face center         |

## Train/test split (identical to the hybrid model)

`FOLDS = {full:[], A:[F17], B:[F13,F14,F15], C:[F18]}`. Fold B holds **all three
center replicates out together** to prevent center-condition leakage. Fold B is
the interpolation test between low and high feed regimes; Folds A and C are
boundary tests. **No split can identify the unsampled 30–35 °C temperature
transition** — report regime-level uncertainty, not a precise transition curve.

All data-derived constants (train ranges for normalization, the reference load
`c_fold`) are computed from TRAIN runs only (leakage-safe).

## Critical terminology (do not blur these)

- **qS target vs qS supply.** The experiment fixes only an *initial qS target*
  encoded in the feed policy. The model uses the dynamic **feed-supply proxy**
  `qS(t) = F_rate(t)·S_IN/(X(t)·V(t))`, `S_IN = 400 g/L`, capped at 2.0. This is
  *supply*, not measured uptake; interpreting it as uptake assumes carbon
  limitation with negligible accumulation/overflow/maintenance.
- **phi_IB flux vs pool fraction.** `phi_IB(t)` is the instantaneous
  production-flux allocation to IB; the measured `P_IB/(P_sol+P_IB)` is the
  accumulated pool fraction. They differ whenever allocation changes within a run.
- **Glucose is at the floor.** Measured glucose is pinned at 0.01 g/L throughout,
  so an explicit-glucose-state Monod model is not viable; the feed proxy is the
  only substrate driver (same as the hybrid model).
- **qS/mu/qP columns are smoothed diagnostics**, built from interpolated offline
  data. The model fits *raw* observations only (X, P_sol, P_IB at samples; CER/OUR
  online), never these derivatives.

## Mechanistic model

States (g/L): `X`, `P_sol`, `P_IB`. Forcing inputs (online): `feed_rate [L/h]`,
`V [L]`, `T [°C]`, `is_induced`. **V is the N-floor-corrected volume** (`data.py:
CORRECT_BASE_VOLUME = True`): `V = Volume_corrected + base(t)`, base shaped by the raw
Pump2 signal × the per-run N-balance floor (`base_reconstruct.py`).

```
qS(t)   = F_rate·S_IN/(X·V)                            # capped at 2.0
D(t)    = F_rate/V
mu      = Y_xs·qS                                       (feed-limited; no T term)
phi_IB  = sigmoid(theta_0 + theta_T·(T−30) + a_qS·log(qS/0.4))   # sol/IB partition
qP_tot  = is_induced · qP_max · qS/(qS + K_S) · exp(−α·max(T−30, 0))   # Monod × qP(T)
dX/dt     = mu·X − D·X
dP_sol/dt = (1−phi_IB)·qP_tot·X − D·P_sol
dP_IB/dt  = phi_IB·qP_tot·X − D·P_IB
```

`qP(T) = exp(−α·max(T−30,0))` is the **first-order folding-efficiency decline** above the
expression optimum (σ³² heat-shock proteostasis burden); α≈0.096/°C, η(35)=exp(−0.48)≈0.62.
Mechanistic read: η(T) is the fraction of expression-directed substrate carbon that
becomes *measurable* product (soluble + recovered IB); the rest is respired as CO₂ via the
σ³² proteostasis response — misfolded-protein proteolysis (Lon/Clp/HslUV) and/or the ATP
cost of chaperone/protease cycling + reduced net synthesis (carbon diverted to stress
proteins). It is real CO₂, **not** unmeasured retained protein (IB under-recovery is ruled
out by the CER test; see findings log 2026-07-24). The carbon balance cannot further split
proteolysis from stress respiration. Paired with θ_T (sol→IB shift at high T) = two
signatures of one heat-shock stress. (Equivalent empirical hinge `M0_aload_qPT`, k_T≈0.076, also retained.)

### Offgas prediction — the carbon + electron balance (zero new parameters)

```
r_S = qS·X,  r_X = mu·X,  r_P = qP_tot·X   [g/L/h]
CER = 1000·( r_S/30.03 − r_X/26.565 − r_P/22.58 )                       [mmol/L/h]
OUR = 1000·( 4·r_S/30.03 − 4.07·r_X/26.565 − 4.16·r_P/22.58 ) / 4       [mmol/L/h]
```

All coefficients are fixed physical constants (composition below). Because the
carbon-balance CO₂ depends only on the (fit) growth rate, **maintenance is
already captured implicitly** by M1a: it shows up as the residual carbon that is
not biomass or product. M1b's `m_s` only tests whether the *apparent* yield is
qS-dependent (lower at low qS), which is the only thing a single yield cannot
express. This is why M1a adds information without parameters.

Composition (C-mol basis; degree of reduction γ relative to NH₃, so the unlogged
NH₄OH/N-source drops out of the electron balance):

| species     | formula (per C)  | g/C-mol | γ    |
| ----------- | ---------------- | -------:| ----:|
| glucose     | CH₂O             | 30.03   | 4.00 |
| biomass     | CH₁.₇₇N₀.₂₄O₀.₄₉ | 26.565  | 4.07 |
| LDH product | CH₁.₅₉N₀.₂₇O₀.₃₁ | 22.58   | 4.16 |

Sanity check: pure aerobic growth (Y_xs=0.45, qS=0.4, X=30) predicts RQ≈1.02 and
CER/OUR≈196/193 mmol/L/h — in the observed range (e.g. F13: CER≈172, OUR≈167).

## Model hierarchy (each step must be identifiable; verdicts logged)

| Model | Obs channels | Params (vs M0) | Verdict |
| --- | --- | --- | --- |
| **M0** | X, P_sol, P_IB | Y_xs, qP_max, K_S, theta_0, theta_T | NN-free baseline; T-only partition |
| **M0_aload** | sparse | + a_qS (qS-load partition) | captures the 30 °C qS-ordering |
| **M0_aload_qPTexp** ← **ADOPTED** | sparse | + α (qP(T) T-suppression) | fixes the dominant T35 error; α identifiable |
| M1a / M1a_aload | + CER, OUR | 0 | **gas reconciled** under corrected volume (was rejected for biomass overprediction — a volume artifact); still adds noise, not re-adopted |
| M1b | M1a | + m_s | **rejected** (non-identifiable without gas) |
| M0_aload_burden | sparse | + a_b, τ_b (Task A) | **rejected** (collinear with a_qS; a_b at bound, sign-wrong) |
| M0_aload_qPT_haldane | sparse | + Ki | **rejected** (Ki→bound; qS compressed) |
| M0_aload_qPTexp_foldlag | sparse | + τ_fold (soluble transit) | **rejected** (τ_fold→0; qS-varying φ already captures dynamics) |
| M0_decay / M0_aload_haldane | sparse | + τ_burden / Ki | **rejected** (overfit / beyond-data; earlier session) |
| M0_smet_independent | sparse | independent qP_sol/qP_IB + S_met state (NN port) | **descriptive only** (fits trajectory in-sample, overfits held-out) |
| M0_smet_independent_ibdeg | sparse | + k_deg (net-negative IB, opt 1+3) | **rejected** (in-sample identifiable, overfits held-out; M1b trap) |
| M_decoupled_ibtransient (v1) | sparse | independent sol/IB, complementary gates + clock | **rejected** (g_ib(T30)→0.05 starves IB; τ→20, clock off) |
| **M_decoupled_v2** | sparse | independent sol/IB, **independent** gates + T30 clock | **captures the T30 IB decline** (first); τ~0.7 identifiable; peak under-predicted (qP_IB_max bound) |
| M_decoupled_v3 | sparse | + t_delay (delayed shutoff) | **captures peak+decline in-sample** (cost 0.417) but overfits (qP_IB_max bound, α_T→0.45); fold-C level under |
| M0_aload_qPTexp_yxt | sparse | + β_T (T-dependent Y_xs) | **rejected** (β_T→0; too blunt — fixes F07, over-corrects F19/F12; biomass↔product zero-sum) |
| M0_proteostasis | sparse | replaces θ₀/θ_T/a_qS/α with fold-race + rescue capacity (7 params) | **split** (T-switch + σ loss SUPPORTED ≡ α; rescue capacity falsified) |
| M0_proteostasis_v2 | sparse | capacity limits the folding rate k_f | **rejected** (degenerate no-IB corner, 10× cost; load-ratio arithmetic) |

`M0_aload_qPTexp` = `_BASE` (Y_xs, qP_max, K_S, theta_0, theta_T) + `_ALOAD` (a_qS) +
`_QPT_EXP` (alpha_T). The empirical hinge `M0_aload_qPT` (+`_QPT` k_T) is equivalent.

## Parameter specs (init / bounds)

| param   | init  | bounds       | role                                             | identified? |
| ------- | -----:| ------------ | ------------------------------------------------ | ----------- |
| Y_xs    | 0.45  | (0.20, 0.80) | apparent biomass yield on feed substrate [gX/gS] | **tight** (~0.36, corrected) |
| qP_max  | 0.010 | (1e-4, 0.10) | max specific total production [gP/gX/h]          | sloppy ratio with K_S |
| K_S     | 0.10  | (1e-3, 2.0)  | Monod half-saturation in qS [gS/gX/h]            | weak (near-saturation) |
| theta_0 | −1.91 | (−6, 6)      | partition logit intercept                        | weak |
| theta_T | 2.01  | (−1, 8)      | partition logit temperature slope [1/°C]          | weak (hits 8 bound in fold A) |
| a_qS    | 0.0   | (−4, 4)      | qS-load coefficient on log(qS/0.4) (partition)   | weak (swing 0.5–2.8 across folds) |
| **alpha_T** | 0.10 | (0.0, 0.50) | qP(T) first-order folding-loss [1/°C] (qPTexp) | **tight** (~0.08–0.10) |
| k_T     | 0.10  | (0.0, 0.20)  | qP(T) linear hinge [1/°C] (qPT; equiv to alpha_T) | tight |
| m_s / a_b / tau_b / Ki / tau_fold | — | — | **rejected** params (M1b/burden/Haldane/foldlag) | non-identifiable |

`theta_0/theta_T` priors anchor the known ~31 °C switch (sigmoid s_T=2.01, T50=30.95).
**Well-identified: Y_xs (~0.36) and alpha_T (~0.096).** Everything else (K_S, theta_0,
theta_T, a_qS) is weakly identified — report as ranges; only the 30 °C stratum informs the
partition, and only the T35 vs T25/30 contrast informs alpha_T.

## Identifiability discipline

1. Multistart (≥3 seeds) per fold; report parameter spread across seeds **and** folds.
2. **Profile likelihood** per scalar: fix on a grid, refit the rest, report the
   train-cost interval; flat/asymmetric ⇒ not identifiable.
3. A parameter is "identified" only if interior (not at a bound), stable across
   folds, with a bounded profile.
4. Specifically probe: `K_S` vs `qP_max` (Monod near-saturation ⇒ sloppy),
   `Y_xs` (should tighten sharply under M1a), `m_s` (M1b go/no-go), `a_qS`
   (supported effect but weakly identified — only the 30 °C stratum informs it).

## Metrics and caveats

`e = y − ŷ` (observed − predicted); positive bias ⇒ underprediction. Per state:
RMSE, MAE, Bias, **NRMSE_range** (= RMSE/range_train), nISE_range, NLL (Gaussian,
cv+floor noise), R² (reported but **flagged unreliable for near-zero P_sol/P_IB**
— use range-weighted RMSE/MAE and bias instead). Normalization uses TRAIN ranges.
Aggregation: micro over timepoints within (run,state); **macro over (run,state)
cells = primary**; per-state macro over runs; full per-run/per-state breakdown.

**Plots are the primary model-assessment tool.** Per run: observed-vs-predicted
trajectories for X, P_sol, P_IB, CER, OUR, and IB fraction (induction time
marked; held-out runs titled in red). Plus parity, final-endpoint summary, and
profile-likelihood plots.

## Known limitations

- **NH₄OH base unlogged → now reconstructed** (was `total_added_g: 0`). The N-floor
  base-volume correction (`base_reconstruct.py`, default on) recovers V to ±~50% (bounded
  by the N-balance floor and the scfv ~2× acid-neutralization excess). Pinning it needs a
  weighed base bottle on the LDH reactor (MF2-R6) — not available. The redox balance still
  sidesteps N via NH₃-referenced degree of reduction.
- **qS-peak unresolved** (corner runs qs0.2/qs0.6 mis-predicted): the realized dynamic qS
  compresses the setpoint spread (0.2/0.4/0.6 → ~0.17/0.27/0.30), so no qS-based mechanism
  is identifiable. Needs qS levels between/above the FCCD points.
- **Partition weakly identified** (a_qS, θ_T): only the 30 °C stratum informs it; θ_T hits
  its 8 bound in fold A. The corner IB fractions (F17, F18) are poorly predicted.
- **α mechanism — under-recovery ruled out (2026-07-24 CER test)**: the α-removed product
  carbon really does appear as CO₂ (model CER matches observed at T35; under-recovery would
  leave it ~10 mmol/L/h lower). α is biology, not an assay artifact. Unresolved is only the
  *within-biology* split — proteolysis vs stress-response respiration vs reduced net synthesis
  — which the carbon balance cannot separate. Effect size (~10 mmol/L/h) is modest vs gas noise;
  F12 (qs0.6/high-biomass, O₂-transfer-limited gas) is the lone contradicting run.
- **Biomass ash fraction (0.08) not applied** by the USP pipeline → biomass carbon
  over-counted by ~8% (rescales Y_xs; tested, does not reconcile gas — see findings log).
- **Below-LOQ near-0/100% soluble/IB fractions** are not censored (LOQ undocumented) —
  judge near-zero RMSE/bias, not R².
- **Unsampled 30–35 °C transition** is not identifiable by any split.
- Carbon closure ≈ 103% (100–107%, low variance) → CER/OUR trustworthy; LDH uses
  the rigorous N₂-balance method (`use_n2_balance: true`, wet gas).

## Relationship to the hybrid model

The hybrid model (`HybridModel/examples/LDH_FCCD`, JAX+Diffrax+Equinox) models
the same states with a **neural network** production rate and fits **only sparse
offline** data. Its mechanistic-NN-free variant (`m1_loadpartition_mechanistic.py`)
is the direct ancestor of this playground's M0 (same Monod-in-feed-proxy
production, temperature sigmoid partition, Y_xs·qS growth). The playground's
contributions: (a) it lives in the USP repo with a scipy stack (self-contained);
(b) it **adds offgas** as a zero-parameter observable via the C+e⁻ balance; (c)
it makes identifiability the explicit selection criterion. The hybrid's reference
M3-P1-E held-out numbers (X macro RMSE ≈ 1.3–1.5; final center IB fractions ≈
0.12–0.15) are the sanity-check ballpark for M0.

## Reproducibility

```
cd examples/LDH/modeling_playground
# adopted model (canonical):
python run_playground.py --fold B --model M0_aload_qPTexp --seeds 42 43 --n-starts 3
# variants (tested/rejected) are all selectable via --model:
python run_playground.py --fold B --model M0_aload_qPTexp_foldlag --seeds 42 --profile
# recover the base=0, no-qPT baseline by toggling the volume correction off:
CORRECT_BASE_VOLUME=0 is set in data.py (module flag); set False to revert.
```

pandas/numpy/scipy/matplotlib only; reads one xlsx + per-run raw bioreactor files
(for Pump2); deterministic seeds; cached pickle loads. Canonical results live in
`results/M0_aload_qPTexp/holdout_*/`; comparison artifacts in `results/base_volume/`,
`results/qpt/`, `results/offgas_diagnostics/`, `results/M0_aload_burden/`.

## Findings log

(updated as runs complete)

### Performance / engineering (done)

- openpyxl read of the 7 MB xlsx ≈ 60 s → cached to pickle (`LDH_consolidated_results_processed.pkl`), repeat loads ≈ 1 s.
- adaptive `solve_ivp` ≈ 286 ms/solve → fixed-step RK4 (dt=0.1 h) ≈ 100 ms/solve; residual eval (9 runs) 7 s → 0.5 s. One fold-seed (3 starts) ≈ 80–300 s.

### M0 (NN-free, temperature-only partition) — fold A (held-out F17)

- **Y_xs ≈ 0.354, rock-stable across all seeds** (0.3535 in seeds 42/44) — matches the hybrid model's stable ~0.352–0.357. Strong identifiability for the biomass yield.
- **theta_T poorly identified**: seed 43 hit the upper bound (8.0) at a worse local min, while seeds 42/44 settled at theta_T≈0.63. This is the expected temperature-only-partition identifiability limit (25/35 °C runs are saturated, so only the 30 °C stratum informs the slope). Offgas (M1a) constrains biomass/production, not partition, so partition identifiability will remain limited — this is a data limit, not a model flaw.
- Held-out F17: X RMSE 0.93 g/L (bias +0.69, slight underpredict), P_sol RMSE 0.29 (underpredict), **P_IB RMSE 0.19 (bias −0.148 → overpredicts IB)** — the same F17 IB overprediction the hybrid documents. R² on P_IB is −49 (useless for the near-zero pool) — confirming the decision to judge by range-weighted RMSE/bias, not R².
- Sanity vs hybrid: held-out X RMSE ≈ 0.93 is in the hybrid M0 ballpark (~0.96–1.0). M0 is a sound baseline.

### qS dependence of the partition — CONFIRMED from data (2026-07-23)

Final IB fraction `P_IB/(P_sol+P_IB)` by run (verified from the consolidated data):

| T | run | qS₀ | IB frac |
| --- | --- | --- | --- |
| 25 °C | F09/F16/F10 | 0.2/0.4/0.6 | 0.00 / 0.02 / 0.00 (all soluble) |
| 30 °C | F17 | 0.2 | **0.021** |
| 30 °C | F13/F14/F15 | 0.4 | **0.123 / 0.147 / 0.116** |
| 30 °C | F18 | 0.6 | **0.189** |
| 35 °C | F07/F19/F12 | 0.2/0.4/0.6 | 1.00 / 1.00 / 1.00 (all IB) |

Temperature sets the **regime** (soluble at 25 °C, IB at 35 °C); within the 30 °C
mixed band, **qS (expression load) modulates the partition magnitude** with a
clear low < center < high ordering. A temperature-only partition assigns all
30 °C runs the same φ and **cannot** reproduce this — so the partition must
include the load term. This matches the hybrid model's M3-P1 result (qS-dependent
allocation supported; full-data ensemble reproduced the 30 °C ordering).

**Consequence for the hierarchy:** temperature-only (M0/M1a) is the *baseline that
is expected to fail at 30 °C*; the load-dependent partition `phi_IB = sigmoid(
theta_0 + theta_T·(T−30) + a_qS·log(qS/0.4))` (the `*_aload` variants, qS-based
load — the qP-based form saturates and hits the bound) is the scientifically
correct model. `a_qS`'s exact numerical value is weakly identified (only the
30 °C stratum informs the slope; 25/35 °C are saturated) — report it as a supported
effect with uncertainty, not a physiological constant. The dynamic load `qS(t)`
(not the constant initial-qS target) is the correct driver, since realized qS
evolves during induction.

### M1a (offgas observables, zero new params) — NEGATIVE RESULT

Grouped folds A/B/C completed (M0, M0_aload, M1a, M1a_aload, seeds 42/43). The
offgas hypothesis — that adding CER/OUR as zero-parameter observables (predicted
from the C+electron balance) improves biomass identifiability — is **NOT supported
as formulated**. Adding gas degrades biomass prediction:

| fold | model | X_RMSE | X_bias | P_sol_RMSE | P_IB_RMSE |
| --- | --- | ---:| ---:| ---:| ---:|
| A (F17) | M0_aload | 0.93 | +0.69 | 0.19 | 0.11 |
| A (F17) | M1a_aload | 1.64 | **−1.40** | 0.41 | 0.15 |
| C (F18) | M0_aload | 1.27 | +0.21 | 0.33 | 0.46 |
| C (F18) | M1a_aload | 2.28 | **−1.96** | 0.57 | 0.51 |

(bias = observed − predicted; negative ⇒ overprediction). Adding gas **flips the
biomass bias from under- to over-prediction and raises X_RMSE**, because:

- gas pulls **Y_xs up (0.351 → 0.388)** to fit the CO₂ magnitude, and that higher
  yield overpredicts biomass (the no-gas Y_xs≈0.351 is stable across seeds, folds,
  and the hybrid model — it is the trusted value);
- gas pushes **production to the K_S=2.0 bound** (qP_max≈0.1), i.e. the Monod
  collapses to `qP ∝ qS` — only the ratio qP_max/K_S is identified when gas is
  present (the K_S/qP_max sloppy confound, now exposed by gas);
- the C+electron-balance gas prediction reflects the **instantaneous** pre-induction
  yield (~0.39, from CER/feed) which exceeds the **cumulative** biomass-fit yield
  (0.351); fitting the gas rates pulls Y_xs toward 0.39, overpredicting the offline
  biomass trajectory (which reflects 0.351). The gas also inherits carbon-accounting
  uncertainty (monohydrate/anhydrous ~10 %) and noisy/negative OUR; the corrected
  gas-implied biomass is ~within that uncertainty of offline X (finding 2), so this is
  a yield-shape effect, not a biomass-level bias.

**Rescuing offgas would require correcting the data errors first** (apply the 8 %
biomass ash fraction; reconstruct NH₄OH/base volume; investigate the ~3 %
over-recovery; filter negative OUR). That is future work; for now the defensible
mechanistic model uses **sparse biomass + protein only (no gas)**.

### M0_aload (qS+T partition, no gas) — BEST MODEL

- Captures the 30 °C qS ordering: held-out center F13/F14/F15 IB fractions
  predicted **0.123 / 0.131 / 0.126 vs observed 0.123 / 0.147 / 0.116** (fold B,
  the qS interpolation test). Temperature-only M0 gives all 30 °C runs the same φ
  and predicts ~0.14 for all — wrong for F17 (0.021) and F18 (0.189).
- Biomass held-out RMSE 0.93 (A) / 1.43 (B) / 1.27 (C) g/L — good; total product
  underpredicted (~4.4→3.2 at center), the same production-model limitation the
  hybrid reports.
- **Y_xs ≈ 0.351–0.354, stable across all seeds and folds** (matches hybrid).
- **a_qS: supported effect, weakly identified value** — interior and seed-stable
  within each fold, but varies across folds (B=1.68, A=0.94, C=0.49; sign-flips in
  the gas models). This reproduces the hybrid's finding: the qS partition is real
  but its coefficient is not a physiological constant — report it with uncertainty.
- `theta_T` occasionally hits the 8.0 bound in a worse seed-43 local min (seeds 42
  settle at ≈0.63–0.80); the temperature slope is weakly constrained because only
  the 30 °C stratum is unsaturated (25/35 °C are at the 0/1 floors).

### Decision

- **Primary model: M0_aload** (qS+T partition, sparse obs, no gas), with the
  **N-floor base-volume correction applied** (see findings log, 2026-07-24). Reported
  **Y_xs ≈ 0.36–0.37** (corrected), not 0.351.
- **Production model extended with qP(T)** — mechanistic form `M0_aload_qPTexp`:
  `qP *= exp(−α·max(T−30, 0))`, **α ≈ 0.096/°C** (~9% fractional loss of measurable product
  per °C above the 30 °C optimum; η(35)≈0.62) = first-order folding-efficiency decline
  (proteostasis / σ³² heat-shock burden). Identifiable (tight profile); fixes the dominant
  T35 overprediction (+1.53→+0.29), 35% cost cut, partly closes the center gap. Equivalent
  empirical hinge `M0_aload_qPT` (k_T≈0.076) also retained. Haldane re-rejected on this
  baseline (Ki→bound; qS-peak is a data-resolution limit). Caveat: α may partly fit T35 IB
  under-recovery, not only real heat-stress suppression — unresolved.
- **M1a/M1a_aload gas overprediction was a volume artifact, now resolved.** Under the
  N-floor volume correction the gas model's held-out biomass bias collapses (+2.46 →
  +0.29 g/L) and Y_xs converges with the sparse fit — the offgas is consistent with the
  biomass once volume is right. Gas revisitable; still adds noise and doesn't fix the
  production limitation, so M0_aload remains primary.
- **M1b (maintenance) — RE-TESTED and rejected (overfits), 2026-07-24**: with gas reconciled
  (volume corrected, RQ-masked), re-fit `M1b_aload_qPTexp` on the adopted α-baseline. m_s *is*
  in-sample identifiable (clean parabolic profile, MLE ≈ 0.08, Δcost(m_s=0)=+0.59, train cost
  2.40→1.81) — but it **OVERFITS**: held-out F17 biomass bias collapses +0.42 → −4.58 (same
  lower-train/worse-held-out trap as induction-decay). The fitted Y_xs=0.50 / m_s=0.076 are
  unphysical (true yield > gas-implied instantaneous ~0.39; m_s ≫ typical E. coli ~0.02–0.05) →
  m_s is absorbing the carbon-balance systematic error (volume ±50%, monohydrate/anhydrous ~10%,
  the flat CER offset), not real maintenance. The gas-vs-biomass yield gap is a carbon-accounting
  artifact, not Pirt maintenance. **Keep μ = Y_xs·qS.** See findings log + `m1b_maintenance_test.py`.
- **Reject M0_aload_burden (Task A)**: the offgas-burden partition driver is not
  identifiable beyond the qS-load term (burden ≈ qS proxy; a_b profile monotone,
  sign-wrong, destabilizes a_qS, no held-out gain). See findings log, 2026-07-24.

### Open identifiability questions (profile likelihood, next)

- a_qS, K_S, theta_T profiles on M0_aload (best model) — quantify how weakly each is identified.
- Whether correcting biomass ash / base volume rescues the offgas hypothesis (future).

### Profile-likelihood identifiability — M0_aload fold B (2026-07-23)

Train-cost profile (Δcost = cost − min over the grid; small ⇒ flat ⇒ weakly identified):

| param | MLE | Δcost away from MLE | verdict |
| --- | ---:| --- | --- |
| Y_xs   | 0.351 | sharp parabolic: 1.57 @0.20, 0.30 @0.28, **0 @0.37**, 0.68 @0.45, 2.33 @0.53, 8.57 @0.70 | **well identified** (~0.35–0.37) |
| theta_T | 0.744 | asymmetric: explodes left (4.66 @−1, 4.01 @−0.13), flat plateau right (0.085 @1.6, 0.107 @2.5, 0.108 @4.2) | identified lower side (~0.7); weak upper tail (why seed-43 hits the 8.0 bound) |
| a_qS   | 1.677 | **broad/flat**: 0.13 @0.52, **0 @1.68**, 0.0095 @2.84, 0.025 @4.0; 0.19 @−2.97 | **weakly identified** — 0.5–4 nearly equivalent (matches cross-fold 0.49–1.68 swing) |
| K_S    | 0.258 | shallow: 0.099 @0.001, 0.011 @0.13, **0 @0.26**, 0.012 @0.51, 0.028 @0.77 | weakly identified (Monod near-saturation sloppiness; only qP_max/K_S ratio matters) |

**Bottom line:** Y_xs is the only tightly identified parameter. The partition
coefficients (a_qS, theta_T upper tail) and K_S are weakly constrained — report
them as ranges, not point estimates. This is consistent with only the 30 °C
stratum being informative for the partition (25/35 °C saturated) and with the
hybrid model's own instability findings for these quantities.

### Offgas diagnostics (2026-07-24) — `offgas_diagnostics.py`

Script + `results/offgas_diagnostics/` (per-run plots + `OFFGAS_DIAGNOSTICS.md`).
Four findings, the first two of which **reframe the offgas negative result**:

**1. Gas quality varies enormously by run (fit only the reliable runs/windows).**
Fraction of the fed phase with trustworthy RQ∈[0.8,1.2]:

| good | runs |
| --- | --- |
| ~95–99 % | F10, F12, F16, F18, F19 |
| ~80–86 % | F07, F17 |
| ~38–46 % | F13, F15 |
| **~1–29 %** | **F14 (1 %), F09 (29 %)** — essentially unusable |

Causes (confirmed from raw data): F09 has a systematic **OUR offset** (RQ 0.34–0.85
throughout; offgas O₂ reads low); F14's gas is broken nearly throughout; and at
high biomass the controller cranks **O₂ enrichment** (Inlet O₂ 21→27–53 %), where
the N₂-balance CER/OUR becomes ill-conditioned and the culture hits O₂-transfer
limitation (OUR plateaus, then spikes). ⇒ F09/F14 should be dropped from any
gas-informed fit, and other runs restricted to the RQ∈[0.8,1.2], low-enrichment window.

**2. Offgas-implied biomass — RATE-BASED, FED-PHASE ONLY (corrected twice, 2026-07-24).**
Two earlier cumulative-balance attempts were wrong:
- (a) feed-only carbon → X_gas **negative during batch** (no feed in batch);
- (b) using "Glucose consumed [g]" → X_gas **started at ~20 g/L and decreased**, because
  that column counts the *entire* initial batch glucose as consumed at t=0 and the
  "Glucose consumed rate" is 0 throughout batch — i.e. **the data has no substrate-
  uptake tracking during batch**, so a cumulative carbon balance is not computable there.

The correct construction is **rate-based, fed-phase only, anchored at the measured
feed-onset biomass**: integrate the gas-implied growth rate
`rX = MW_X·(rS/MW_S − CER/1000)` (rS = feed-supply uptake, valid once feeding starts)
from the measured biomass at feed onset. This is physical: it starts at the measured
feed-onset X (~8–9 g/L) and rises monotonically.

Result: gas and measured biomass **agree at induction** (within ~1–3 g/L), then
**diverge upward by final**:

| run | X_gas@ind | X_meas@ind | X_gas@final | X_meas@final |
| --- | ---:| ---:| ---:| ---:|
| F13 | 23.1 | 23.5 | 55.8 | 44.3 |
| F17 | 28.6 | 26.1 | 49.5 | 41.0 |
| F07 | 31.3 | 33.3 | 50.7 | 41.2 |
| F18 | 24.0 | 21.7 | 72.6 | 55.6 |

The late divergence is **not a biomass-level bias** — it is the gas-implied yield
(~0.39) outpacing the measured apparent yield (~0.35) over the fed phase (the same gap
as finding 3). So the offgas does not reveal that offline biomass is biased; it reveals
a yield discrepancy (instantaneous gas yield > cumulative biomass yield), which
accumulates. Both prior "offline biomass biased high/low" claims are retracted.

**3. Pre-induction Y_xs from gas ≈ 0.39–0.40** (feed r_S + CER only; no biomass) vs
biomass-fit 0.351. This is the **instantaneous** pre-induction yield; the biomass-fit
0.351 is a **cumulative apparent** yield, and finding 2 shows the gas-implied (higher)
yield accumulates more biomass than measured over the fed phase. Per-run: F09 0.353
(few points — bad gas), F13/F15 0.36–0.37, F17/F18/F19 0.40–0.43.

**4. IB formation has a metabolic signature in the offgas (burden respiration).**
Post-induction, the **specific CO₂ production rises above the pre-induction
(growth) baseline**, and that rise correlates almost perfectly with the IB
fraction across the 30 °C stratum:

| run | IB frac | ΔqCO₂ (post−pre) |
| --- | ---:| ---:|
| F17 | 0.021 | −0.67 |
| F13 | 0.123 | 1.11 |
| F15 | 0.116 | 1.03 |
| F14 | 0.147 | 1.24 |
| F18 | 0.189 | 2.76 |

Pearson **r = 0.98** (pre/post qCO₂; maintenance-corrected burden r = 0.95; RQ-shift
r = 0.22 — so it's the CO₂ burden, not RQ). Mechanistic read: IB aggregation
imposes a measurable metabolic cost (wasted translation, chaperone/ATP demand)
seen as extra CO₂. Caveat: n = 5 and qS co-varies (burden and IB both rise with
expression load), so this is a strong correlation, not yet a proven independent
driver — but it is a candidate **offgas-derived metabolic state** for IB, and a
possible alternative partition driver to explore (φ_IB ~ burden instead of qS).

**Action items from the diagnostics:** (a) drop F09/F14 and RQ-filter the rest for
any gas-informed fit; (b) correct offline X for ash/bound water and re-test whether
gas and biomass then agree; (c) explore burden qCO₂ as an IB driver.

### Total-product underprediction investigation (2026-07-24)

M0_aload underpredicts **center** (qS=0.4) total product (F13 obs 4.40 vs pred ~3.2)
while fitting F17 (low) and F18 (high). Diagnosis + falsification tests:

**Diagnosis — specific production is non-monotonic in qS.** Per-biomass product
(Ptot/X_final): low **0.065**, center **0.096**, high **0.079**. Specific production
**peaks at moderate qS and declines at high qS** (high-qS cells grow more but express
less per biomass). A monotonic Monod `qP_max·qS/(qS+K_S)` cannot represent a peak.

**Falsified fixes (each tested, each rejected — identifiability-consistent):**
- **Induction decay** `qP·exp(−t/τ)` (motivated by observed front-loaded qP, peak ~0.03
  decaying to ~0.006): LOWER train cost (0.33 vs 0.44) but **WORSE** held-out center
  Ptot (2.86 vs 3.17) — it overfits training shape at the expense of the center
  magnitude. (Same trap as the hybrid's flexible M6.) ⇒ drop.
- **Substrate-inhibition (Haldane)** `qP_max·qS/(K_S+qS+qS²/Ki)`: on fold B (center
  held out) `Ki`→ upper bound (collapses to Monod — no peak info without center). On
  **full data** `Ki=0.296` (interior) but `qP_max=0.1` (bound) and the fitted peak
  lands at qS≈0.66 (beyond the data), so center is still underpredicted (F13 3.52 vs 4.40).
- **Full-data Monod** also underpredicts center (F13 3.48) with F17/F18 well-fit, and
  `qP_max` at/near its bound — the optimizer wants more production than the cap allows
  but cannot get center without overpredicting high.

**Conclusion:** the center underprediction is **not fixable by an identifiable
production mechanism** — added parameters either overfit (decay), are rejected
(Haldane on held-out), or push the peak beyond the data (Haldane on full). It mirrors
the hybrid model's M6 result exactly. Report it as predictive uncertainty at the
moderate-qS production optimum.

**Synthesis — likely contributors:**
1. **Missing NH₄OH/base volume inflates measured concentrations.** Base is unlogged
   (`total_added_g: 0`), so recorded V is too low → g/L concentrations (Ptot, X) are
   too high. Reconstructing base volume would lower observed Ptot toward the prediction,
   partially closing the gap. *(No base mass / NH₃ conc. available → parked.)*
2. **Genuine production optimum at moderate qS** unresolved by the 3-level FCCD — would
   need qS levels between 0.2/0.4/0.6 to identify a peak.
3. *(Previously listed "measured biomass inflated by ash" — **retracted**: the corrected
   gas-implied biomass is actually ~above measured, within carbon-accounting uncertainty,
   so there is no clear biomass-inflation signal.)*

**Next concrete step:** reconstruct base volume + apply ash, then re-check whether the
center underprediction and the gas-vs-biomass discrepancy shrink together.

### Reconstruct NH₄OH volume (method)

- **(a) Stoichiometric N-demand (no extra data):** biomass CH₁.₇₇N₀.₂₄ (~12.6 % N) +
  LDH product (16.47 % N) accumulated → integrate N into X and P over the run = minimum
  NH₄OH for assimilation (lower bound; ignores the pH-neutralization excess).
- **(b) From logged `Base pump duration` × calibration:** the pump-duration trace is
  logged; the USP volume module scales it to `total_added_g` (currently 0 in configs →
  disabled). Supplying the per-run total base mass/volume (bioreactor log/scale) scales
  the trace to the full base volume (N-assimilation + acid-neutralization).

> **Status (2026-07-24):** no base mass and no NH₃ concentration are available, so base
> reconstruction is **parked**. Ash (8 %) was tested (see below).

### Burden respiration — mechanism and IB specificity (2026-07-24)

The post-induction burden (qCO₂ above the pre-induction growth baseline) is the
metabolic cost of recombinant expression: transcription/translation (ATP/GTP), folding
(chaperones), amino-acid biosynthesis, and — for IB — wasted translation of misfolded
protein + aggregation/stress-response handling.

**Dominated by expression load, with an IB premium.** Extra CO₂ per gram of total
product (mmol/g), by temperature at matched qS:

| qS | 25 °C (soluble) | 35 °C (pure IB) |
| --:| ---:| ---:|
| 0.4 | 116 | **183** |
| 0.6 | 224 | **276** |

At every matched qS, IB-producing runs cost more CO₂/g than soluble runs ⇒ **IB carries
a distinct extra metabolic cost** on top of generic expression. But burden rises steeply
with qS at all temperatures, so it is **primarily an expression-load signal, with a
secondary IB premium** — not a clean standalone IB marker. The r=0.98 burden–IB
correlation holds *within* the 30 °C band (where qS and IB co-vary); across all
temperatures the picture is mixed (the 30 °C mixed stratum has the lowest cost/g).
Caveats: n small; the maintenance baseline is noisy (negative burden at low qS).

**Time-resolved test (`burden_ib_timeresolved.py`, 2026-07-24) tempers the endpoint r.**
Pairing burden with IB fraction at every post-induction sample point (not just final):

| scope | raw qCO₂ | burden (a·μ+b) | burden (pre-median) | n |
| --- | ---:| ---:| ---:| ---:|
| 30 °C stratum, time-resolved | 0.63 | 0.69 | **0.77** | 20 |
| all runs, time-resolved | 0.11 | 0.11 | 0.24 | 44 |
| 30 °C, final endpoint (prior) | — | 0.95 | 0.98 | 5 |

The headline r≈0.98 was partly an **aggregation artifact** of final endpoints. The
genuine *dynamic* link is **moderate (r≈0.6–0.77) within the 30 °C stratum** and **weak
(r≈0.1–0.24) across all temperatures** (IB is pinned at 0/1 in the saturated 25/35 °C
regimes, so there is no within-run co-variation there). The link is real and
**regime-specific** (the 30 °C mixed band) but noisier than the endpoint suggested. The
scatter is consistent with IB at time *t* depending on the **history** of burden
(accumulation), not its instantaneous value — which is the mechanistic motivation for a
filtered/accumulated burden state (Task A option 2), though collinearity with the qS-load
term remains the identifiability risk. Artifacts: `burden_vs_IB_timeresolved.png`,
`burden_ib_timeresolved_30C.csv`.

### Ash correction — tested; rescales Y_xs, does not reconcile gas vs biomass (2026-07-24)

Applying the 8 % biomass ash fraction (offline X × 0.92) just **rescales** Y_xs
proportionally (0.354 dry → 0.325 organic); biomass fit quality is unchanged. It does
**not** reconcile the gas-vs-biomass yield gap (instantaneous gas Y_xs ~0.39 vs biomass
0.354/0.325). Note the corrected gas-implied biomass is now ~above measured (finding 2),
so the residual gap is most plausibly the **glucose-carbon accounting** (monohydrate/
anhydrous ~10 % over-count) and the instantaneous-vs-cumulative yield distinction, not
ash and not (testably) acetate (**no acetate column**). ⇒ apply ash for correctness
(report organic Y_xs), but it will not rescue the offgas on its own.

### Split-fitting gas on reliable runs only — viable for noise, not for bias

Yes: include CER/OUR residuals only for runs/windows with trustworthy gas (drop F09/F14;
RQ∈[0.8,1.2] window), while all runs contribute sparse (X, P_sol, P_IB). The gas then
informs the shared growth/production params (Y_xs, qP) from clean runs; partition params
(theta, a_qS) come from every run's sparse data. This removes **sensor-noise** leverage
but not the **systematic** carbon-balance discrepancy (gas still implies Y_xs≈0.39 vs
biomass 0.35/0.33). So split-fitting gives cleaner gas params + honest uncertainty, but
is not expected to flip the negative offgas result until the systematic gap (overflow/
CO₂ bias) is resolved.

**Implemented + tested (2026-07-24):** an RQ-trustworthy mask is now in
`fit.residuals` (gas residuals only where RQ∈[0.8,1.2]; this silently drops F09/F14's
gas and the O₂-enriched/transfer-limited points). Re-fitting M1a_aload fold A with it:
Y_xs=0.382 (vs 0.385 unfiltered), F17 biomass bias −2.5 (still overpredicting).
⇒ **Confirmed: cleaning gas noise does NOT remove the systematic gas-biomass conflict.**
The negative offgas result is robust to gas cleaning. The conflict is systematic
(carbon-balance gap, likely omitted overflow/acetate + CO₂ bias), not sensor noise.
Keep the RQ mask for honest uncertainty; do not expect it to rescue offgas.

### Task A — burden as IB-partition driver: NON-IDENTIFIABLE, closed (2026-07-24)

Two-part test (observational diagnostic + dynamic model). Both point the same way;
the burden state does **not** add identifiable information beyond the qS-load term.

**Part 1 — independence diagnostic (`burden_independence.py`):** does observed
offgas burden explain IB-partition variance *after* controlling for the dynamic
feed-supply qS(t)? At the sample-point level (30 °C stratum, n=20 post-induction
points) burden is a **proxy for qS**: partial r(burden, IB | qS) = **0.12** (vs raw
0.69); OLS burden coefficient → **0.00** once qS + biomass are included (R²=0.80
from qS+X alone); burden-only and qS-only fits are indistinguishable (R² 0.47 vs
0.49). The endpoint view (n=5) looks stronger (partial r 0.84) only because the
constant-qS replicates F13/F14/F15 inflate it — and there the only IB spread with
reliable gas is F13 vs F15 (0.123 vs 0.116), too close to call (F14's gas is broken).
⇒ burden carries no qS-independent signal the model could exploit.

**Part 2 — dynamic burden model `M0_aload_burden`:** latent state
`db/dt = qP·X − b/τ_b` (filtered production; **offline obs only** — b is model-derived,
so F14's broken gas does not contaminate the fit), partition `+= a_b·b`, tested
alongside the qS-load term. Gold-standard evidence on fold B (the qS-interpolation
test):

- **Multistart (8 starts):** `a_qS` pinned at its **+4 bound in every start**;
  `a_b` at a bound (±4) in every start, the better minimum at the **unphysical
  `a_b=−4`** (burden *decreasing* IB); the physically-correct +4 fits worse.
- **a_b profile likelihood** (fix a_b, refit a_qS free):

  | a_b | train cost | Δcost | a_qS refit |
  | ---:| ---:| ---:| ---:|
  | −4 | 0.373 | 0.000 | 4.0 (bound) |
  | −2 | 0.376 | 0.003 | 3.38 |
  | 0 | 0.440 | 0.066 | **1.68** |
  | +2 | 0.452 | 0.079 | 1.58 |
  | +4 | 0.464 | 0.090 | 1.54 |

  **Monotone (no interior minimum) and shallow** (Δcost < 0.1 across ±4) ⇒ a_b has
  no finite MLE ⇒ **not identifiable**. At a_b=0 the refit recovers a_qS=1.68 —
  exactly M0_aload — so a_b and a_qS are one sloppy ridge: moving a_b just shuffles
  the partition between two collinear terms, with a slight overfit preference for
  the wrong sign.
- **Held-out center IB (fold B):** the burden model is *worse* than M0_aload
  (F13/F14/F15: 0.098/0.112/0.100 vs M0_aload 0.123/0.131/0.126 vs obs
  0.123/0.147/0.116) — the bound-hitting MLE overfits train (cost 0.373 vs 0.44)
  at the expense of held-out partition. Biomass/product held-out RMSE unchanged.
- **Cross-fold:** a_b at/near −4 in all folds (B −4, A −2.71, C −4); tau_b swings
  1.7–3.5; fails the cross-fold-stability criterion.

**Verdict:** the burden state is **not identifiable beyond the qS-load term** —
confirmed by both the observational diagnostic (burden ≈ qS proxy) and the
model-based profile (monotone, sign-wrong, destabilizes a_qS, no held-out gain).
This matches and sharpens the hybrid-model expectation. **Drop the burden state;
M0_aload remains the locked primary model.** Artifacts: `burden_independence.{md,png}`,
`results/M0_aload_burden/{identifiability_summary.md,burden_verdict.png}`.
`M0_aload_burden` is retained in `PARAM_SPECS` as a tested-and-rejected variant
(like M1a / Haldane / decay). Task A closed.

### Base/NH₄OH volume reconstruction — the missing volume reconciles gas with biomass (2026-07-24)

The base (NH₄OH, 6.95 M) is the one volume term the consolidated pipeline drops
(`total_added_g = 0`). It is not weighed for LDH, so V is under-counted → qS and D
biased. Reconstructable only to a bounded bracket (no weighed base on the LDH reactor),
but the bracket is informative enough to change two prior conclusions.

**Reconstruction (bounded; `base_reconstruct.py`):**
- **N-balance floor.** Feed is N-free; all post-batch N enters via base. So
  `base ≥ (N in biomass + N in product − batch (NH₄)₂HPO₄ pulse) / titrant-N`.
  Per run: F13/14/15 ~75 mL, F17 ~64, F18 ~107, F10/F12 ~100–114 mL (range 54–114).
  Batch (NH₄)₂HPO₄ = 4 g/L (~0.85 g N/L) is the only other N source.
- **scfv reference** (same strain/process, different protein, base weighed): base
  65–180 g, ~2× the N-floor — base is **acid-neutralization-dominated, not N-driven**
  (the base/biomass ratio varies 0.68–2.35 g/g). The scfv pump-calibration
  `k = base_mL / Pump2.Duration` has CV 0.39 and is **reactor-dependent** (scfv R1/R3/R4
  vs LDH MF2-R6; even the feed-pump calibration differs) → does **not** transfer to LDH.
  (ferm_05 is stoichiometrically impossible — base N < biomass N — so that scfv feed
  likely contained N.) ⇒ we bracket, not point-estimate.
- The raw **Pump2.Duration** signal gives the per-run base SHAPE (correlates with
  biomass 0.94, cumulative CO₂ 0.87). Bracket: base ∈ [N_floor, 2×N_floor], pump-shaped.
  Observations (X, P g/L) are assay values and do not change; only qS = F·S_IN/(X·V)
  and D = F/V are affected.

**Bracket re-fit (M0_aload, fold B):**

| base level | Y_xs | held-out Ptot bias | held-out IB rmse |
| --- | ---:| ---:| ---:|
| none (0) | 0.351 | −1.06 | 0.011 |
| floor (1×N) | 0.359 | −1.06 | 0.011 |
| mid (1.5×N) | 0.364 | −1.06 | 0.011 |
| upper (2×N) | 0.368 | −1.06 | 0.011 |

- **Y_xs reconciles with gas** — rises 0.351 → 0.368 across the bracket (recovers ~40%
  of the gap to the gas-implied ~0.39). The missing volume was depressing the biomass-fit
  yield (`Y_xs = μ_obs/qS`; too-low V ⇒ too-high qS ⇒ too-low Y_xs).
- **Center total-product underprediction is ROBUST (−1.06 g/L, unchanged)** ⇒ it is the
  production-model limitation (non-monotonic specific production), NOT a volume artifact.
- **Partition robust** (a_qS 1.68→1.71, theta_T stable).

**Gas-biomass conflict — RESOLVED at the N-floor (the main update).** M1a_aload (gas
model), fold A (holdout F17):

| base level | Y_xs (gas fit) | F17 X bias (pred−obs) |
| --- | ---:| ---:|
| none (0) | 0.382 | **+2.46** (the M1a overprediction that got it rejected) |
| floor (1×N) | 0.369 | **+0.29** (≈ resolved) |
| upper (2×N) | 0.356 | −1.72 (flips to underpredict) |

Correcting the volume to the N-floor **collapses the gas model's biomass overprediction
from +2.46 to +0.29 g/L**, and Y_xs converges (gas 0.369 ≈ sparse 0.368). **The missing
base volume was a major driver of the gas-vs-biomass yield conflict** — the offgas is
consistent with the biomass once the volume is right. This **removes the rationale used
to reject M1a** (gas no longer overpredicts biomass).

**Decisions / updates:**
- **Adopt the N-floor volume correction** as the defensible volume (physically grounded
  and reconciles the gas). Report **Y_xs ≈ 0.36–0.37** (corrected), not 0.351.
- **The offgas (M1a) is no longer rejected for biomass overprediction** — that conflict
  was a volume artifact. The gas is consistent with biomass under corrected volume;
  revisitable, though it still adds gas noise and does not fix the production limitation.
- **The center total-product underprediction is unchanged** → still the production-model
  limitation (Task B, parked on data).
- Volume remains a bounded uncertainty (±~50%, [N_floor, 2×N_floor]); to PIN it, weigh
  the base bottle on the LDH reactor (MF2-R6) for one future run.
Artifacts: `base_reconstruct.py`, `base_volume_refit.py`, `gas_volume_check.py`,
`results/base_volume/{bracket_refit.md, gas_volume_check.md}`.

**Canonical regeneration (2026-07-24).** The N-floor correction is now the DEFAULT
volume (`data.py: CORRECT_BASE_VOLUME = True`, applied in `_extract_run` via
`base_reconstruct`, raw Pump2 reads pkl-cached at `examples/LDH/pump2_raw_cache.pkl`).
M0_aload regenerated on folds A/B/C confirms the bracket: **Y_xs ≈ 0.36 stable across
folds** (A 0.362 / B 0.359 / C 0.362; was 0.351 at base=0); a_qS cross-fold instability
preserved (A 0.95 / B 1.69 / C 0.50 ≈ base=0); center Ptot underprediction unchanged
(−1.06 g/L); boundary partition errors (F17 overpredicts IB, F18 underpredicts)
unchanged. Set `CORRECT_BASE_VOLUME = False` to recover the base=0 model for comparison.

### Center total-product underprediction — reframed & partly resolved (qP(T)) (2026-07-24)

The center (T30/qs0.4) total-product underprediction survived the volume bracket
(−1.06 g/L unchanged) → it is a production-model issue, not a concentration artifact.
Reframing it on the (T × qS) grid (observed specific product Ptot/X_final, and the
time-resolved qP(t) = (dP + D·P)/X over induction) revealed it is a **T × qS pattern**,
not the simple "qS peak" previously assumed:

| | qs0.2 | qs0.4 | qs0.6 |
| --- | ---:| ---:| ---:|
| T25 | 0.085 | 0.072 | 0.079 |
| T30 | 0.065 | **0.096** | 0.079 |
| T35 | 0.042 | 0.054 | 0.050 |

The dominant signal is **temperature**: total specific product is ~40% lower at T35
across all qS, and is suppressed *throughout* induction (time-resolved qP ~0.005–0.015
at T35 vs 0.012–0.024 at T25/T30). But the model's `qP = qP_max·qS/(qS+K_S)` has **no
temperature dependence** (only the sol/IB partition `theta_T` depends on T). Full-data
fit confirms the consequence: **every T35 run is overpredicted +1.2 to +2.0 g/L** — the
single biggest error in the model.

**Fix — qP(T) (`M0_aload_qPT`):** add an asymmetric T-suppression of total production,
`qP *= max(1 − k_T·max(T−30, 0), 0.05)` (hinge at 30 °C; flat for T≤30, drops above —
matches the T35-only drop). 1 new parameter. Results (full data):

| model | cost | T35 mean bias | center mean bias | qP_max |
| --- | ---:| ---:| ---:| ---:|
| M0_aload | 0.811 | +1.53 | −0.75 | 0.031 |
| **M0_aload_qPT** | **0.527** | **+0.29** | **−0.38** | 0.046 |

- **k_T = 0.076** → T35 production ~62% of T30 (matches the observed ~40% suppression).
- **k_T is tightly identified**: clean parabolic profile (Δcost +0.46 at k_T=0, +1.94 at
  0.18) — the best-identified parameter after Y_xs.
- qP_max freed (0.031→0.046, interior — no longer dragged down by T35), which **partly
  closes the center gap** (−0.75 → −0.38). Fold B F13 −1.24 → −0.85.

**Residual qS-peak — NOT identifiable (Haldane rejected, re-tested on the qPT baseline).**
With qP_max freed, all qs0.6 runs now overpredict (F10 +0.80, F18 +0.60, F12 +0.54) — the
production peak at qs0.4 vs qs0.6. `M0_aload_qPT_haldane` (substrate inhibition,
`qP = qP_max·qS/(K_S+qS+qS²/Ki)`) gives **Ki = 10 (upper bound), zero cost improvement**
(profile monotone to the bound; peak qS = √(Ki·K_S) = 2.47, far beyond the data). Cause:
the **realized dynamic qS compresses the setpoint spread** (0.2/0.4/0.6 → ~0.17/0.27/0.30),
so the qs0.6-vs-qs0.4 production difference tracks the *setpoint*, not the realized qS, and
no qS-based mechanism can resolve it. Same data-resolution limit the hybrid model hits.

**Mechanistic form — first-order folding efficiency (`M0_aload_qPTexp`).** The hinge is
empirical; the mechanistic realization is that measured product = synthesis × folding
efficiency η(T), with η declining exponentially above the expression optimum (first-order
proteostasis rate): `qP *= exp(−α·max(T−30, 0))`. Equivalent fit to the hinge over T25/30/35
(cost 0.5271 vs 0.5273; identical qP_max/Y_xs/T35+center bias — 3 T levels cannot distinguish
linear from exponential), but α has a first-order (rate-based) reading: **α = 0.096/°C ≈ 9 %
fractional loss of measurable product per °C above 30 °C** (η(35)=exp(−0.48)=0.62). α is
tightly identified (profile cost 1.02→0.53→1.61 over α=0→0.10→0.30).

**Interpretation:** η(T) is the productive-folding efficiency — the fraction of synthesized
protein that becomes measurable product (soluble + recovered IB), ≈1 at T ≤ 30 °C (the
recombinant-expression optimum) and declining above it as the σ³² heat-shock regulon diverts
chaperones/proteases and faster translation drives misfolding/aggregation/degradation (the
classic reason pET/E. coli expression is run at 18–25 °C). Paired with the partition θ_T
(sol→IB shift at high T), high T shows two signatures of one proteostasis stress: more IB
partition AND lower total product. α quantifies the burden (~38 % of expression lost at 35 °C).

**Mechanism of k_T / α — resolved 2026-07-24 (CER carbon-balance test, `alpha_cer_test.py`).**
The IB-under-recovery reading is **ruled out**. `simulate_run` applies η *before* the carbon
balance, so the model routes the α-removed product carbon to CO₂ — a falsifiable prediction:
under-recovery (lost protein retained as unmeasured IB) would leave observed CER ~10 mmol/L/h
*below* the model, whereas real oxidative biology would match it. Result (post-induction,
RQ-trusted): model CER bias **T25 −2.3 / T30 +7.5 / T35 +0.3 mmol/L/h** — T35 matches and sits
inside the baseline range; observed is ~10 mmol/L/h *above* the under-recovery curve (69% of
T35 points nearer the degradation prediction). F07/F19 clearly support biology; F12 (qs0.6,
O₂-transfer-limited gas) is the lone contradiction. **So α is biology, not artifact.** What
remains unresolved is *which* biology: the carbon balance cannot separate (a) proteolysis of
misfolded LDH by σ³²-upregulated AAA+ proteases (Lon/ClpXP/HslUV) with amino-acid carbon
oxidized, (b) ATP cost of the chaperone/protease response, and (c) reduced net synthesis /
carbon diverted to stress proteins — all route carbon to CO₂. Read η(T) as "fraction of
expression carbon that becomes measurable product; the rest is respired." Effect size (~10
mmol/L/h, ~6% of CER) is modest vs per-run gas noise. Artifacts: `results/alpha_cer/`.

**Decision:** adopt **qP(T)** as the production-model extension (identifiable, fixes the
dominant T35 error, helps the center). The residual qS-peak is a documented, unresolved
data-resolution limit (needs qS levels between/above 0.2/0.4/0.6, or an independent IB
recovery factor). Artifacts: `qpt_test.py`, `haldane_test.py`, `results/qpt/`.

**Held-out caveat (canonical regen, `M0_aload_qPTexp`, folds A/B/C).** Net in-sample win
(cost A 0.80→0.51, B 0.44→0.29, C 0.76→0.50; T35 + center fixed), BUT the freed qP_max
(0.031→0.046) overpredicts the **corner** runs once the T35 drag is removed — exposing the
qS-peak the monotonic Monod cannot represent:
- fold A (F17, qs0.2): held-out IB 0.081→**0.181** (worse; θ_T sits at its 8.0 bound —
  systematic, all 3 seeds identical, not multistart noise).
- fold C (F18, qs0.6): held-out Ptot +0.04→**+0.69** (worse), IB 0.076→0.122 (better).
- fold B (center, the target): held-out Ptot −1.06→**−0.85** (better).

So qP(T) trades the dominant T35/center errors for sharper corner (qs0.2/qs0.6)
mis-prediction — the qS-peak is now the visible residual, plus the partition's pre-existing
weak identifiability (a_qS/θ_T; θ_T hits its bound in fold A). The production model is
improved; the **partition is now the weak link** (next focus if pursued). θ_T=8.0 is largely
benign (T25/T35 are already saturated at 0/1, so the steep step doesn't change them; the
corner IB error is driven by θ_0/a_qS, not θ_T).

### α mechanism — CER test rules out IB under-recovery (2026-07-24)

`alpha_cer_test.py` + `results/alpha_cer/`. The model routes the α-removed product carbon to
CO₂ (η applied before the carbon balance), so under-recovery (lost protein retained) is
falsifiable: it would leave observed CER below the model. Post-induction CER bias (RQ-trusted):
T25 −2.3 / T30 +7.5 / **T35 +0.3 mmol/L/h** — T35 matches, ~10 above the under-recovery curve
(69% of T35 points nearer degradation). **Under-recovery ruled out; α is biology.** The
within-biology split (proteolysis vs stress respiration vs reduced synthesis) is not resolvable
by the carbon balance. Bonus signal: the model *underpredicts* T25/T30 post-induction CER by
~2–7 (a shortfall the balance omits); α's lost-C (~10) over-compensates at T35. That shortfall was
tested as a production-proportional burden β·qP·X (Task D) — **non-identifiable** (R²≈0; it is a flat
offset, likely maintenance-like, not per-product). α stays empirical. F12 (qs0.6) contradicts the
CER match but is O₂-transfer-limited gas.

### Dynamic IB/sol partition (soluble folding-transit delay) — REJECTED (2026-07-24)

Observed dynamics: the IB fraction is **not constant** — it peaks ~2–4 h post-induction then
declines (F18 monotone 0.33→0.19; F13/14/15 rise-then-fall). Hypothesised cause: IB is a
fast kinetic aggregation (instant), soluble is chaperone-limited folding (lags). Built a
biologically-consistent two-pathway model `M0_aload_qPTexp_foldlag`: φ·qP·X → IB instantly,
(1−φ)·qP·X → folding pool N_fold → soluble after maturation τ_fold (mass conserved;
verified). Result: **rejected**.
- Cost worse (0.527 → 0.579); **τ_fold → 0.5 h (lower bound)**, profile monotone increasing
  (wants τ_fold→0, i.e. no delay → collapses to qPTexp).
- No improvement in IB-fraction trajectory RMSE or endpoint IB.
- **Reason:** the model's **qS-varying partition already captures most of the dynamics**
  (φ changes as qS(t) evolves — F18 trajectory already declines 0.42→0.18). The explicit
  delay is redundant and non-identifiable. The residual (model overpredicts early IB for
  F18; F13/14/15 rise-then-fall vs model monotone) is too subtle for 4 sample points/run.
⇒ Per identifiability discipline: do not add τ_fold. The dynamics are real but not
resolvable with an identifiable mechanism from this data. Artifacts: `foldlag_test.py`,
`results/qpt/foldlag_test.md`.

### M1b maintenance re-test on corrected volume — overfits, REJECTED (2026-07-24)

`m1b_maintenance_test.py` + `results/m1b_maintenance/`. Premise: with gas reconciled (volume
corrected → M1a held-out F17 bias +2.46→+0.29; RQ mask drops broken gas) and Task D showing the
post-induction CER shortfall is a flat maintenance-like offset, a Pirt term `mu = Y_xs*(qS - m_s)`
might now identify and explain the gas-vs-biomass yield gap (instantaneous ~0.39 vs cumulative ~0.35).
Added `M1a_aload_qPTexp` / `M1b_aload_qPTexp` gas variants (gas on the adopted α-baseline) to
`models.py`.

**Result: m_s is in-sample identifiable but overfits — rejected.** Full-data fit: M1b train cost
2.40→1.81, m_s=0.076, Y_xs jumps 0.369→**0.504** (above the gas-implied instantaneous ~0.39 —
unphysical), qP_max comes off its 0.10 bound (→0.072). m_s profile is cleanly parabolic (MLE≈0.08,
Δcost(m_s=0)=+0.59, rises to +8.5 at 0.30). **But held-out F17 (fold A) collapses: X bias +0.42 →
−4.58** (M1a was the resolved +0.42). Same lower-train-cost / worse-held-out trap as induction-decay.

**Read:** m_s is absorbing the carbon-balance systematic error (volume ±50%, monohydrate/anhydrous
~10%, the flat CER offset), not real maintenance. m_s=0.076 ≫ typical *E. coli* ~0.02–0.05; the
true-yield 0.50 > the instantaneous gas yield 0.39 is mechanistically inconsistent (instantaneous
already includes maintenance, so true must be ≥ but the fit overshoots). The gas-vs-biomass yield
gap is a **carbon-accounting artifact, not Pirt maintenance** — confirming the long-standing
handoff conclusion with the corrected-volume + reliable-gas machinery. **Verdict: keep
mu = Y_xs·qS; do not adopt m_s.** This closes the last gas-side thread; the offgas is consistent
with the model but cannot add an identifiable parameter. Lesson reaffirmed: a bounded in-sample
profile is necessary but not sufficient — adoption requires held-out generalization (no degradation).

### Task E.1 — pin the volume from the respiratory acid load (∫CER) — CLOSED, NOT pinned (2026-07-24)

`base_acid_reconstruct.py` + `results/base_acid/`. The hoped-for zero-free-parameter volume pin:
NH₄OH base demand ~ respiratory CO₂ (carbonic acid), so `base ~ ∫CER·V` and ~1 base:1 CO₂ at pH 7
fixes the scale stoichiometrically. **Tested on all 11 runs (full run, batch+fed) and falsified on
both axes:**

- **Scale fails.** The 1:1 CO₂ stoichiometry predicts **base ≈ 360–680 mL/run, i.e. ~5.7× the
  N-floor (~2.8× the bracket *upper* 2×N_floor)**, consistently across runs (ratio 4.8–7.1). Carbon
  closure ≈ 103% (low variance, rigorous N₂-balance) says the offgas CER is **accurate** — the CO₂
  is real. It is therefore **stripped, not neutralized**: only **~35% of evolved CO₂** (bracket/1:1,
  range 0.28–0.42 at 2×N_floor) consumes base; the rest leaves as gas, as expected in a well-aerated
  reactor. That neutralization fraction is a **free ~0.2–0.4 parameter, not a stoichiometric
  constant**, so ∫CER supplies a *scale* only through the same calibration the N-floor bracket
  already uses. **No zero-parameter pin.**
- **Shape fails to discriminate.** CER and feed rate are near-collinear (rate-level **r ≈ 0.92**),
  so "base tracks CO₂" is indistinguishable from "base tracks substrate/biomass" — the shape carries
  no CO₂-specific information. (Rate-level r(base, CER) is only moderate, median 0.63; F17 ≈ 0.04.)
  The Pump2 signal remains the better base *shape* (direct measurement); ∫CER adds no independent
  shape. Also: within-run r(cumBase, cumCO₂) = **1.00 for every run** (both monotonic) — the
  previously-cited "0.87 cumulative correlation" is a spurious artifact of correlating two
  cumulative curves, not evidence.
- **Decision-tree branch:** `base ≪ ∫CER` is triggered, but the cause is **normal CO₂ stripping**
  (not a gas-quality error — carbon closure rules that out). Falsifies the E.1 premise.

**Consequence (closes the rescue path for Tasks A–D).** The motivation was that the rejected
gas-side parameters (M1b `m_s`, burden) might be *volume artifacts* — real but unidentifiable because
the ±50% volume bracket injects too much noise. Since ∫CER **cannot** tighten the volume beyond that
bracket, this rescue path is **closed**: the volume uncertainty stands, and the gas-side
identifiability ceiling is **not** shown to be a volume artifact — it remains most plausibly
real/biological (consistent with the M1b/burden re-test verdicts). The only true volume pin is a
weighed base bottle on the LDH reactor (MF2-R6) for one future run. Artifacts: `acid_load_test.md`,
`acid_load_summary.png`, `acid_load.csv`.

### Task E.2 — O₂-transfer limitation as the qS-peak driver — CLOSED, NOT supported (2026-07-24)

`o2_limit_test.py` + `results/o2_limit/`. Hypothesis: the qs0.6 specific-product decline (the qS-peak)
is O₂-transfer limitation — at high biomass the culture hits the OTR cap and diverts carbon from
expression. The consolidated df carries the full O₂ signal (Inlet O₂, pO₂, RQ, qO₂, OUR), so this is a
clean data-first test (no refit). **O₂ demand is real and peaks at qs0.6 — but it does NOT limit LDH
production:**

- **O₂ demand peaks at qs0.6 at every temperature.** Inlet-O₂ enrichment excess (max Inlet O₂ − 21 %):
  T25 qs0.2/0.4/0.6 = 0.3 / 9.9 / **37.3**; T30 = 2.6 / 10.2 / **18.8**; T35 = 6.9 / 9.0 / **37.5**.
  F10/F12 hit 58 % inlet O₂; F12's pO₂ crashed to **0.6 %** (near-anaerobic).
- **But the production decline is T30-only.** qs0.4→qs0.6 specific-product change: T25 **+0.006** (no
  decline) with +27 enrichment; T30 **−0.023** (decline) with only +8.6; T35 **−0.004** (no decline) with
  +28.5. The only decline (T30) has the *smallest* O₂-demand rise; T25/T35 have the worst O₂ stress and
  no decline. If O₂ drove the decline it would be strongest at T25/T35.
- **The controller met the demand (RQ ~ 1).** F18 RQ 0.94, F12 RQ 0.99 — fully aerobic; the high inlet
  enrichment is the controller *supplying* O₂, not unmet demand. High demand ≠ metabolic limitation.
- **Even genuine O₂ limitation doesn't depress product.** F10 is the one real overflow run (RQ 1.57), yet
  Ptot/X = 0.079 (not depressed). F12 — pO₂ crashed to 0.6 % — has Ptot/X = 0.050 ≈ its qs0.4 control
  F19 (0.054). O₂ starvation did not reduce LDH production.
- **Cross-run r(Ptot/X, enrich excess) = −0.14** — weak, temperature-dominated (T35 ~0.05 vs T25/T30
  ~0.08–0.10), not the strong negative an O₂-cap requires.

**Verdict: do NOT add an O₂-transfer capacity term** (`qP_eff = min(qP_Monod, OTR/Y_PO)`). O₂ demand is
large at qs0.6 but does not limit LDH production; the qS-peak is a T30-specific pattern pointing back to
the partition/temperature structure (or the unresolved E.3 cofactor/subpopulation mechanism), not O₂
transfer. This confirms the HANDOFF caveat that the qs0.6 gas-transfer artifact is not a usable
production lever. Artifacts: `o2_limit_test.md`, `o2_limit_summary.png`, `o2_limit.csv`.

### S_met cumulative-load Haldane (Müller 2023, "A1") — CLOSED, NOT supported (2026-07-24)

`smet_test.py` + `results/smet/`. Inspired by Müller et al. 2023 (ESCAPE33 — same E. coli/LDH/fed-batch
lineage, TU Wien/CHASE): use the cumulative metabolized substrate since induction `S_met = ∫qS·X dt` as a
cell-age/burden state in a generalized-Haldane inhibition on qP. This was the one structural lever that
could represent the qS-peak *identifiably*, because (unlike the rejected Haldane-in-instantaneous-qS, which
failed since realized qS compresses the setpoint spread) S_met is the time-integral of load and is not
compressed. **Key simplification:** qS·X = S_IN·D, so the biomass cancels and `S_met = S_IN·∫D dt` — dense,
X-free, identifiable by construction (state-determined, not a free latent).

**Result: S_met does NOT explain the qS-peak.** The clincher is the cross-temperature load range: cumulative
load at qs0.6 is ~the SAME at every T (S_met/X @ qs0.6 ≈ 2.1–2.3 for T25/T30/T35), yet the qs0.4→qs0.6
specific-product change differs by T — only **1/3 strata decline** (T25 *rises* +0.007, T30 *declines*
−0.016, T35 marginal −0.004). If cumulative load drove the decline via Haldane inhibition, the SAME load
should produce the SAME decline at every T; it does not ⇒ the qS-peak is a T×qS (partition) effect, not a
load effect (mirrors the E.2/O₂ finding: the qS-peak is T30-specific).

**What IS real but not useful:** qP declines with cumulative load *within a run* at fixed (T, qs) (partial
r | T,qs = −0.66) — a genuine post-induction productivity decline. But this is (a) the same time-decline the
already-**rejected induction-decay** captures (overfits held-out), and (b) confounded with the existing
Monod(qS) term — realized qS(t) declines over a run as biomass outpaces the feed ramp, so Monod(qS(t))
already produces a within-run qP decline without any load state. S_met adds no identifiable mechanism beyond
these. (Linear correlations undersell a non-monotonic peak: raw r(qP,u)=−0.17, r|T=−0.19, endpoint
r|T=+0.26; the bare Haldane fit R²=0.03 is T-dominated and not informative.)

**Verdict: do NOT fit A1.** The qS-peak is not a cumulative-load effect; S_met would reproduce the
induction-decay overfit without resolving the peak. Other Müller ideas noted but not pursued: Moser/Hill
partition in qP (swap for the sigmoid — low priority, current partition works), viable/dead split (needs
flow cytometry — not available), structural-identifiability sensitivity-rank gate (cheap workflow add).
Artifacts: `smet_test.md`, `smet_summary.png`, `smet_trajectories.png`, `smet.csv`.

### Plan B Stage 1 — soft sensor: dense μ(t)/qP(t) from gas + per-run scale (2026-07-24)

`soft_sensor.py` + `results/soft_sensor/`. Reconstructs the dense trajectory the sparse-offline model
leaves untapped. Growth from the carbon balance (`rX_gas = MW_X·(rS/MW_S − CER/1000)`) integrated from the
offline anchor X0, with ONE per-run scale `s` fit to the offline biomass; qP as the carbon-balance residual
(`rP = MW_P·(rS/MW_S − rX_scaled/MW_X − CER/1000)`), dense and independent of the product samples (which
validate it). CER-based (robust, avoids the noisy OUR).

**Three findings:**
1. **The per-run scale works (replaces the ±50% volume bracket).** A single `s` per run (0.65–0.90, median
   0.81) reconciles gas-implied growth with offline biomass at **R²(X) median 0.99**. s<1 = gas overstates
   growth (instantaneous-vs-cumulative yield gap + the product term); it now POINT-estimates the
   volume/carbon nuisance per run instead of bracketing it. Usable calibration.
2. **Gas-consistency is regime-dependent (the key result).** P_recon/P_obs (gas-implied vs observed final
   product): **T30 median ≈ 1.1** (1.06–1.23 — gas CONSISTENT with biomass+product) vs **T25/T35 corners
   median ≈ 3.1** (0.47–5.14 — off). The outliers are exactly the offgas-diagnostics-flagged runs (F12/F10
   O₂-stressed/overflow, F16/F19 corners, F09 broken gas). ⇒ the gas + carbon balance are **trustworthy in
   the T30 regime** (where the model's action is — qS-peak/partition) and unreliable at the stressed
   corners. The soft sensor quantifies the gas-quality map per run.
3. **Caveat — the dense qP(t) is residual-sensitive.** qP declines in all 11 runs (late/early 0.31–0.59),
   but the SPARSE product-implied qP declines only in ~7/11 (not F07/F09/F12/F16). The dense decline matches
   the sparse where product is substantial + gas clean (T30 — consistent), but is ARTIFACTUAL at the corners
   (the carbon-balance residual drifts under the constant-s assumption). ⇒ dense qP shape is reliable only
   for T30; cross-validate against sparse before trusting any run.

**Stage 2 (next, if pursued):** replace the constant-s + ignored-product-term with the **dual carbon+
electron balance** (use OUR where RQ-trusted) to solve rX and rP jointly — removes the residual drift and
should make dense qP reliable across runs; then re-test mechanisms (within-run decline, IB/partition)
against the refined dense qP, focused on the T30 regime. Separately, the per-run `s` could feed back as a
calibration (a per-run volume/carbon scale) to tighten the model's nuisance. Artifacts: `soft_sensor.md`,
`soft_sensor_<run>.png` (11), `soft_sensor_summary.png`, `soft_sensor.csv`.

### Plan B Stage 2 — dual carbon+electron balance — DEAD (stoichiometric limit, 2026-07-24)

`dual_balance_test.py` + `results/dual_balance/`. Stage 2 aimed to solve rX and rP JOINTLY from gas+feed
(inverting the carbon + electron balances) for a reliable dense qP. **It is unusable for LDH**, and the
reason is fundamental: the split coefficient is (γP − γX), and biomass (γ=4.07) and LDH product (γ=4.16)
are **nearly equally reduced** — γP − γX = 0.09 — so the electron balance CANNOT distinguish growth carbon
from product carbon. The dual-balance product rate is garbage: wrong-signed (44–100 % of points negative)
and 10–20× too big (std ~10–18 g/L/h vs true ~0.5–0.8); integrated product is −160 to +90 g/L vs observed
1.7–4.4, with no consistency even within T30 (F13 −10, F14 +90, F15 +82). Smoothing does not help (the bias
is structural: the CER/OUR/feed closure error is amplified ~1/0.09 ≈ 11×).

**This is the stoichiometric root cause of every gas-side failure.** The offgas constrains only the TOTAL
rX + rP (via the carbon balance), never the growth/product split. M1a (gas overpredicts biomass), M1b
maintenance, and the burden state all tried to use the gas to pin the split — which the LDH stoichiometry
forbids. It is a property of the LDH composition (γX≈γP), not fixable by more data or a better filter.
**Consequence:** the only viable split uses the OFFLINE observations (Stage 1: biomass → growth, product =
residual); its validated T30 qP is the reliable dense qP, and there is no gas-only refinement beyond it.
Artifacts: `dual_balance_test.md`, `dual_balance_summary.png`, `dual_balance.csv`.

### Noise-floor reframe — the qS-peak is a dilution artifact, not a production decline (2026-07-24)

`noise_floor_test.py` + `results/noise_floor/`. Stepping outside the "find a mechanism" frame: the FCCD's
one pure-error estimate is the F13/F14/F15 triplicate (T30/qs0.4). Using it as the noise floor (CV: X 1%,
Ptot 8%, Psol 10%, Pib 5%, IBfrac 13%, Ptot_X 9%), every effect in SD units reframes the qS-peak hunt:

- **The qS-peak in specific product is a dilution artifact.** At T30, total product Ptot is **FLAT** from
  qs0.4 to qs0.6 (4.21→4.37 g/L, **0.5 SD** — nothing) while **biomass rises 18 SD** (44→56 g/L). So Ptot/X
  declines (0.096→0.079) purely because the extra substrate at qs0.6 is routed to **biomass, not product**.
  There is no "production decline at qs0.6" to mechanise — above qs0.4, production has saturated and the
  additional substrate grows cells.
- **The effects the model captures are the robust ones:** biomass-vs-qS (18 SD; μ∝qS linear growth), IB
  fraction-vs-qS (3.7–6.6 SD; the a_qS partition term — the *real* qS effect), IB fraction-vs-T (60 SD; T25
  soluble→T35 pure IB — α(T)+θ_T). These are all in `M0_aload_qPTexp`.
- **What is marginal/noise-limited (and was over-hunted as a "mechanism"):** Ptot_X qS-peak (1.9 SD —
  dilution + within noise, n=1/corner); α(T) on Ptot_X (2.1 SD — the 60-SD IB-fraction shift is the real T
  effect, not total-product suppression).

**Implication.** Every mechanism hunted for the qS-peak (Haldane, induction-decay, S_met, O₂) was chasing a
dilution artifact within triplicate noise — which is *why* none identified. The model's structure already
captures the robust phenomena. The one genuine production feature is **total-product saturation by qs0.4**
(Ptot flat while biomass rises) — qualitatively what a saturating Monod qP already describes (realized qS
0.27 vs 0.30 → Monod differs only ~6 %, consistent with flat Ptot). **Action:** verify the adopted model's
predicted Ptot/X at qs0.6 against the noise floor — if within noise, the model is adequate and the qS-peak
"problem" was overstated; the only mechanistic refinement with a real (if small) target is a
growth/expression **allocation** structure (production capacity saturates; excess substrate→growth).
Artifacts: `noise_floor_test.md`, `noise_floor_summary.png`.

### Option A — model adequacy: M0_aload_qPTexp is ADEQUATE (2026-07-24)

`model_adequacy_test.py` + `results/model_adequacy/`. Fit on all 11 runs (cost 0.527; Y_xs 0.363, qP_max
0.046, K_S 0.563, alpha_T 0.096, a_qS 1.18 — canonical). Residuals (pred − obs) in F13/F14/F15
triplicate-SD units settle the question "is the qS-peak a real misfit or noise?":

- **Specific product (Ptot/X) at qs0.6 is WITHIN NOISE:** F10 +1.6, F18 +1.6, F12 +0.9 SD. The qS-peak is a
  dilution artifact (above) and the model tracks it within noise. **The "qS-peak problem" was overstated.**
- **Total Ptot at qs0.6 mildly overpredicted** (+1.6 to +2.3 SD) — the Monod wants Ptot rising when it is
  flat (saturation); marginal, within the range a saturating qP already describes.
- **The largest residuals are BIOMASS, not product:** F07 +6.3, F14 −4.0, F18 −3.4 SD — scattered across
  corners (mixed signs, different T/qS) = the ±50 % base-volume bracket (qS→μ→X) + run variability, NOT a
  growth/production mechanism. Known, un-pinnable (E.1), surfaced here as the biomass residual.

**Verdict: the model is adequate for this dataset.** The qS-peak hunt ends — it was an artifact within
triplicate noise; the model captures the robust phenomena (biomass-vs-qS, IB-fraction-vs-qS/T) within
noise. Option B (growth/expression allocation) is NOT motivated (product residual within/marginal noise;
qP_max/K_S already sloppy). Declare `M0_aload_qPTexp` the final model for the existing data, with
documented uncertainties (±50 % volume → biomass residuals; γX≈γP gas-split ceiling; 3-level qS). Further
progress needs new data, not new structure. Artifacts: `model_adequacy_test.md`, `model_adequacy_summary.png`,
`model_adequacy.csv`.

### Late P_IB decline — refolding ruled out; weak IB-degradation; not dilution (2026-07-27)

`psplit_test.py` + `results/psplit_test.png`. Observation (LDH + Xiso): in the T30 mixed regime P_sol grows
strictly but P_IB plateaus/declines late (F13 0.71→0.54, F18 0.94→0.83; stronger in Xiso). The adopted model
predicts P_IB keeps *rising* (+0.02 to +0.05) — it misses the decline. Tested three mechanisms by fitting
variants at the adopted (N-floor) volume:

- **IB→sol refolding (k_ref) — RULED OUT.** The model already *over-predicts* the late soluble gain
  (F13: model dSol +0.86 vs obs +0.61), so there is no "missing soluble boost" for refolding to provide;
  k_ref=0.015, Δcost ≈ −0.004 (nothing). The IB decline is NOT mass transferred to soluble.
- **IB degradation (k_deg) — weakly supported.** k_deg≈0.10/h, Δcost ≈ −0.023; captures the mild decline
  (F18 model dIB −0.03 vs obs −0.03) but NOT the sharp F13 drop (still +0.04 vs obs −0.17). Modest (~4% cost),
  likely marginal — needs profile-likelihood + held-out before trusting (cf induction-decay/M1b overfit trap).
- **Dilution — RULED OUT.** Baseline at 2×N volume still predicts P_IB rising (+0.05) and barely moves cost
  (Δcost −0.007). More dilution does not produce the decline.

**Deeper issue surfaced:** the model *over-predicts late TOTAL production* (F13: model dTot +0.91 vs obs
+0.44) — so the late IB/sol dynamics are entangled with a late production over-prediction (realized-qS
decline / induction-decay territory), not cleanly an IB-split problem. Net: refolding and dilution are out;
a small IB-degradation term is the only IB-specific lever but is marginal and doesn't capture the sharp F13
drop. New model variants `M0_aload_qPTexp_{refold,ibdeg,refoldibdeg}` retained in `PARAM_SPECS` as tested.
Artifacts: `results/psplit_test.png`.

### Hill-IB (cooperative formation) — RULED OUT; IB decline is a total-production-front-loading symptom (2026-07-27)

`hillib_test.py` + `results/hillib_test.png`. Tested Müller's cooperative (Moser/Hill) IB formation
(`M0_qPTexp_hillib`: `qP_IBmax·cap(T)·Hill_N(qP)`, soluble = total − IB, mass-conserving) — the hypothesis
that the front-loaded IB peak/decline is nucleation-dependent cooperative aggregation. **Result: does not
work.** Cost 0.526 vs baseline 0.527 (Δcost −0.001); `qP_IBmax` at its 0.08 upper bound (non-identified),
N_hill≈1.5 (weakly cooperative, near-Michaelis); IB RMSE 0.227 (slightly *worse* than baseline 0.219);
still does not capture the decline (F13 model dIB +0.078 vs obs −0.170). Soluble unchanged (the mass-conserving
formulation did NOT break sol — the separation concern is addressed — but it didn't help IB either).

**Synthesis — the IB decline is NOT an IB-specific mechanism.** This is the 4th IB-specific lever ruled out
(refold, ibdeg, dilution, Hill-IB). Per-sample diagnostics show the real pattern: observed production is
**front-loaded** (high early, declining — IB gains +0.34→+0.12→−0.17 for F13; soluble milder +0.91→+0.45→+0.61),
while the model's qP·X is **back-loaded** (rises with biomass X; the X growth offsets the qS decline). So the
IB peak/decline is a *symptom* of total production being front-loaded in reality but back-loaded in the model;
reshaping only the IB share (Hill/refold/deg/dilution) cannot fix it. The root is the **total specific
production qP declining faster post-induction than the smooth Monod captures** — i.e. induction-decay /
burden territory (induction-decay was tested & rejected for overfitting) or a realized-qS discrepancy under
the open-loop feed. Variant `M0_qPTexp_hillib` retained as tested-and-rejected. Artifacts: `results/hillib_test.png`.

### Production decline is REAL and beyond qS — general across LDH + Xiso (2026-07-27)

`qp_decline_test.py` + `results/qp_decline_test.png`. Settles whether the post-induction production front-loading
is qS-driven (model right) or an EXTRA decline (real burden): compare observed `qP_obs(t) = (dP/dt + D·P)/X`
decline vs `qS(t)` and `Monod(qS)` decline (late/early factors, K_S-free vs raw qS is decisive). **Result:
observed qP declines FASTER than qS explains, in BOTH datasets:**
- LDH (11 runs): median f_qP=**0.54** vs f_qS=0.69 vs f_Monod=0.77 (7/11 "extra"); observed qP falls below
  even raw qS.
- Xiso (3 runs): median f_qP=**0.75** vs f_qS=0.74 vs f_Monod=0.81 (2/3 "extra").

⇒ the production decline is a **real extra burden, not a qS-modeling artifact**, and **consistent across two
unrelated proteins** ⇒ a GENERAL induction mechanism. This is the root of the front-loading / IB peak-decline
/ model late over-prediction. The qS-consistent exceptions (F07/F09/F16) are low-product / bad-gas runs.
**Implication:** the smooth `Monod(qS)` model misses a real production decline. The endpoint is still adequate
(Option A) but the TRAJECTORY carries a genuine burden the model lacks. Capturing it: induction-decay (free
time-constant) overfits (tested); the untested mechanistic candidate is a **burden-driven qP decline tied to
the offgas ΔqCO₂** (the dense, observable post-induction metabolic cost — distinct from the latent partition
burden Task A rejected). Artifacts: `results/qp_decline_test.png`.

### IB-transient decline — RULED OUT (5th IB mechanism); trajectory dynamics not identifiable (2026-07-27)

`ibtransient_test.py` + `results/ibtransient_test.png`. A `qP_decomp` split (sol vs IB apparent rates)
showed the decline is **IB-specific** (qP_IB → ~0 late while qP_sol stays up — cleanly in Xiso), so the
obvious mechanism is an IB-formation transient: `qP_IB *= exp(−t_ind/tau_IB)`, soluble = total − decayed-IB
(mass-conserving). Variant `M0_aload_qPTexp_ibtransient` (+1 param tau_IB). **Result: decisively rejected.**
- Full-data fit pushes **tau_IB to its 20 h upper bound (= no decay)**; the profile is **monotonically
  decreasing** (tau=1 → cost 3.27; tau=20 → 0.54) — the fit wants NO decay; cost is *worse* than baseline
  (0.545 vs 0.527).
- Held-out fold B identical (tau=20; IB endpoint model 0.35–0.38 vs obs 0.51–0.56).

**Why it fails (the real diagnosis):** the model **UNDER-predicts IB** (level ~0.35 vs obs ~0.54; it never
reaches the observed peak), so reducing IB formation (decay) makes it *worse* — it needs MORE IB, not less.
And the full-ODE fit (using all points) does not support an IB decline: the apparent "qP_IB → 0" from the
sparse decomposition was the **noisy last interval** (1 point), not a robust formation shutdown.

**Synthesis — 5th IB-specific mechanism ruled out** (refold, ibdeg, dilution, Hill-IB, ibtransient). The IB
peak/decline is not capturable by any IB-specific term because (a) the model under-predicts IB level
(partition θ, weakly identified) and (b) the decline signal is sparse/noisy (4 sample points/run). The robust
production finding (front-loaded total, beyond qS, cross-protein LDH+Xiso) stands, but every form that
captures it overfits (induction-decay) or is rejected (ibtransient). **Conclusion: the IB/production
TRAJECTORY dynamics are not identifiable from the sparse LDH data; the model is adequate on ENDPOINTS
(Option A, within triplicate noise). The trajectory shape (front-loading / IB peak) is a documented limit,
not a modeling deficit fixable from this data.** Variant `M0_aload_qPTexp_ibtransient` retained as
tested-and-rejected. Artifacts: `results/ibtransient_test.png`.

### NN oracle (hybrid m13) reveals the structure — and vindicates S_met (2026-07-27)

The user noted a hybrid NN (`HybridModel/examples/LDH_FCCD/m13_independent_qp_deep8_staged`, full-data
seed 42) **does** capture the trajectory (sol monotonic rise, IB peak-then-decline). Extracting its learned
rates (`nn_qp_extract.py`, from `independent_qp_diagnostics.csv`) reveals *why* — and it overturns the
"trajectory not identifiable" conclusion above:

- **Independent qP_sol and qP_IB** (two separate NNs), **not a partition**. This is why every φ-split variant
  failed: the partition forces soluble and IB to share one total-production shape.
- **Both rates driven by (T, cumulative_substrate_per_X0, t_since_induction)** — i.e. **cumulative load
  (S_met) + time**, *not* instantaneous qS. Both **decline as cumulative load accumulates**: qP_sol falls
  from a burst to a sustained **floor** (~0.009 g/g/h); qP_IB falls **through zero to net-negative**
  (net IB degradation). T gates *which* product (T25→sol, T35→IB).
- Oracle shapes (e.g. F13 T30): qP_sol 0.020→0.008 (floor); qP_IB 0.0065→−0.0009 (transient).

**This vindicates the S_met / burden mechanism rejected in A1.** A1 found the within-run decline
(r|T,qs=−0.66) but dismissed it as "induction-decay overfit." The NN confirms cumulative metabolic load
suppressing production **is** the trajectory mechanism — it is Müller's S_met form, applied to *independent*
rates (not the shared-total partition I tested). The error was the partition frame, not the mechanism.

**Ported as `M0_smet_independent`** (`smet_independent_test.py`): 4 states (X, P_sol, P_IB, S_met with
dS_met/dt = S_IN·D); independent `qP_sol = qP_sol_max·(1−φ_T)·Monod(qS)·η(T)` (sustained) and
`qP_IB = qP_IB_max·φ_T·Monod(qS)·η(T)·exp(−S_met/K_IB)` (transient); 8 params. φ_T = T-only sol/IB gate.
**Result (2026-07-27): the function exists but OVERFITS.** In-sample cost 0.459 vs baseline 0.527 (**−13%**
— the parametric form does describe the data better, validating the user's point that a function exists).
BUT `qP_IB_max` at its 0.1 upper bound (non-identified); F13's IB *decline* still not captured in-sample
(dIB +0.06 vs obs −0.17); and **held-out fold B collapses** (F13/F14/F15 IB endpoint 0.18–0.22 vs obs
0.51–0.56 — worse than baseline's ~0.35). So smet fits the training trajectories but does not generalize —
the same trap as induction-decay/M1b/ibtransient, and exactly the NN's "at least in a full training run"
caveat. **Refined conclusion:** the trajectory dynamics ARE describable by a parametric function (smet/NN fit
them), but that function is NOT identifiable/generalizable from 4 sample points/run — it overfits. The
endpoint model (Option A) remains the *predictive* one; smet/NN are *descriptive* (reproduce the data,
incl. the IB peak/decline, but don't extrapolate). Variant `M0_smet_independent` retained as
tested (partial: better in-sample, overfits held-out). Artifacts: `results/smet_independent_test.png`.

**Canonical run (2026-07-27, `run_playground.py`, folds A/B/C/full, seed 42, 3 starts)** →
`results/M0_smet_independent/{full,holdout_F17,holdout_F13F14F15,holdout_F18}/` (params.json, metrics.json,
identifiability.md, test/train_metrics.md, plots/). Held-out per-state RMSE vs the adopted `M0_aload_qPTexp`:
A(F17) X 0.85/P_sol 0.17/P_IB 0.25 (vs 0.93/0.29/0.19); B(F13/14/15) 1.37/0.57/0.27 (vs 1.43/0.33/~0.11);
C(F18) 1.36/0.64/0.32 (vs 1.27/0.33/0.46). **Mixed-to-worse: soluble degrades in B/C (0.57, 0.64 vs 0.33).**
Overfit confirmed: `qP_IB_max` at 0.1 bound in 3/4 folds, `theta_T` at 8.0 bound in B (swings 1.4→8.0). So the
S_met port does NOT generalize better than the adopted partition model — same wall the NN hits. **Decision:
keep `M0_aload_qPTexp` as the predictive model; `M0_smet_independent` retained as a documented descriptive
variant (NN-confirmed mechanism, captures trajectory in-sample, not identifiable from 4 pts/run).** The NN's
own held-out (`HybridModel/.../holdout_F13F14F15_seed42`: P_sol RMSE 1.07 bias −0.85, P_IB 0.12; `holdout_F17`:
P_sol 0.69 bias +0.46) is likewise ~0.7–1.0 RMSE / inconsistent bias — "not unusable" but not generalizing.
**Bottom line: trajectory dynamics are describable (NN + parametric port fit them on full data) but NOT
generalizable from this sparse data — confirmed by both.** (Also fixed `metrics.py` np.trapezoid→trapz fallback.)

### M0_smet_independent does NOT robustly capture the DECLINING P_IB — structural reason + next-session directions (2026-07-27)

User-identified from the F18 full-fit trajectory (`results/M0_smet_independent/full/plots/traj_F18_train.png`):
the model does not properly represent the declining P_IB — F18 at best plateaus / marginally declines
(model ΔP_IB last interval ≈ 0; obs −0.03), and F13 still *rises* (model +0.057 vs obs −0.17). It reaches
the IB peak but cannot draw it back down.

**Structural reason.** `qP_IB = qP_IB_max·φ_T·Monod(qS)·η(T)·exp(−S_met/K_IB)` is **always ≥ 0** — the
`exp(−S_met/K_IB)` only *reduces* formation, it never shuts it off or reverses it. P_IB declines only when
formation drops below dilution (`qP_IB·X < D·P_IB` ⇒ `qP_IB < ~0.002 g/g/h`). The fit chose **K_IB ≈ 118
(gentle decline)** and **qP_IB_max at its 0.1 bound** (wants *high* formation to reach the observed peak),
so late-induction `qP_IB ≈ 0.003` sits just *above* the dilution threshold ⇒ P_IB plateaus, does not decline.
The NN escapes this because its `qP_IB` uses a **tanh output → goes negative** (net degradation, learned to
≈ −0.001 late); the parametric port has no such channel.

**To capture the decline, IB formation must drop below dilution or go net-negative. Three next-session options:**
1. **Net-degradation term (NN-style):** let `qP_IB` go negative — e.g. `qP_IB_net = formation·exp(−S_met/K) −
   k_deg·(P_IB/X)`. (ibdeg ALONE was weak in the earlier test, but combined with the S_met formation decline
   it may work — the NN uses both.)
2. **Sharper cumulative-load shutoff:** replace the gentle `exp(−S_met/K_IB)` with a Haldane-in-S_met form
   (sharper; drives formation well below dilution), or constrain K_IB smaller.
3. **Constrain `qP_IB_max` off its 0.1 bound** (it pins there = non-identified) so the formation-vs-shutoff
   trade-off is better posed.

**Options 1+3 — TESTED & REJECTED (2026-07-28, `M0_smet_independent_ibdeg`):** the net-negative IB channel
(`k_deg·P_IB → CO₂`) is the NN's own recipe and IS in-sample identifiable (shallow interior profile,
k_deg≈0.23) — it captures the F18 decline and cuts in-sample cost 0.527→0.413. But it OVERFITS held-out
(qP_IB_max collapses 0.30→0.087 across train sets; F13 decline still missed; in-sample win doesn't transfer).
Option 3 failed — qP_IB_max pins at the *raised* 0.30 bound too (raising just moved the pin); and K_IB→bound
shows the S_met formation decline is redundant with degradation (the fit abandons it). See the 2026-07-28
findings-log entry. Option 2 deprioritized (same ceiling). Variant retained as tested-and-rejected.

**Identifiability caveat (the real ceiling):** the P_IB decline is NOT identifiable from 4 sample points/run —
the fit cannot distinguish "plateau" from "decline," so it chooses the gentler plateau. The NN (200 params,
tanh→negative) captures the decline in full training but **overfits held-out** (~0.7–1.0 RMSE, inconsistent
bias across folds). So capturing the decline robustly likely needs **denser P_IB sampling** (more offline
points), not only a better mechanism — any mechanism that fits the decline from 4 points will tend to overfit.
**This is the key open issue for the next session**, along with the Xiso transfer (cumulative-load qP as the
shared mechanism + LDH priors).

### M0_smet_independent_ibdeg (net-negative IB channel, option 1+3) — in-sample identifiable, OVERFITS, rejected (2026-07-28)

`smet_ibdeg_test.py` + `results/smet_ibdeg_test.png`. Tested the NN's actual recipe — a **net-negative
IB channel** (first-order proteolysis `k_deg·P_IB → CO₂`) on top of the smet-independent S_met formation
decline — so qP_IB can drop below dilution (the parametric port's structural block: the gentle
`exp(−S_met/K_IB)` only *reduces* formation ≥0, so it plateaus P_IB instead of declining it). Option 3
bundled: qP_IB_max upper bound raised 0.10→0.30 so formation can reach the observed peak interiorly (it
pinned at 0.10 wanting *more*; smet under-predicts IB ~0.35 vs obs ~0.54 — so "off the bound" = raise it,
not lower; lowering would worsen the under-prediction).

**In-sample: the channel works descriptively, but option 3 fails.**

| model | cost | IB RMSE | F13 dIB (mdl/obs) | F18 dIB (mdl/obs) |
| --- | ---:| ---:| --- | --- |
| M0_aload_qPTexp | 0.527 | 0.150 | +0.051 / −0.170 | +0.018 / −0.028 |
| M0_smet_independent | 0.459 | 0.148 | +0.057 / −0.170 | 0.000 / −0.028 |
| **M0_smet_independent_ibdeg** | **0.413** | **0.102** | +0.047 / −0.170 | **−0.030 / −0.028** |

- The net-negative channel captures the **F18 (gradual) decline** cleanly — but **NOT the F13 sharp drop**
  (still +0.047 vs obs −0.170). The IB-RMSE win is mostly level/shape elsewhere, not the F13 decline.
- **Option 3 FAILED**: qP_IB_max pins at the **raised** 0.30 bound at every profile point and in the full
  fit. Raising the bound just moved the pin 0.10→0.30; formation wants more still and is non-identified.
- **The S_met mechanism was abandoned**: as k_deg rises, K_IB drifts up (110 → 500 → 1000 bound),
  neutralizing `exp(−S_met/K_IB)` → 1 (off). The fit did NOT use the NN's *combination* (S_met decline +
  degradation); it traded the S_met formation decline for pure degradation. The two decline channels are
  **redundant** — the optimizer picks one and kills the other.
- alpha_T destabilizes 0.096 → 0.371 (T35 production suppressed to 16% vs 62%) — the extra freedom (k_deg +
  unbounded qP_IB_max) absorbs variance via the T-term. Red flag.

**k_deg profile (full data) — shallow interior minimum (in-sample identifiable, weakly):**

| k_deg | 0.00 | 0.05 | 0.10 | 0.20 | ~0.23 (MLE) | 0.30 | 0.40 | 0.50 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| cost | 0.452 | 0.433 | 0.422 | 0.413 | **0.4125** | 0.414 | 0.421 | 0.433 |

Interior minimum at k_deg≈0.23 (Δcost ~0.04 to k_deg=0; ~0.02 to the 0.50 bound) — k_deg **is** in-sample
identifiable (weakly; comparable to M1b's m_s), NOT a bound-pinner. BUT qP_IB_max=0.30 (bound) and K_IB
trades off at every point — the (qP_IB_max, k_deg, K_IB) triple is a sloppy compensating ridge.

**Held-out fold B (F13/F14/F15) — OVERFITS (the trap):**

| fit | qP_IB_max | k_deg | F13 IB endp (mdl/obs) | F13 dIB (mdl/obs) |
| --- | ---:| ---:| --- | --- |
| full-data | 0.300 | 0.229 | — | — |
| fold-B train | **0.087** | 0.206 | 0.35 / 0.54 | +0.035 / −0.170 |

qP_IB_max **collapses 0.30 → 0.087** across train sets (non-identified, unstable); k_deg is stable
(0.229 → 0.206) but the F13 decline is STILL missed held-out (+0.035 vs −0.170); held-out IB level
0.35–0.36 ≈ baseline (~0.35), so the in-sample win (cost −21%, IB RMSE −32%) does NOT transfer.

**Verdict: REJECTED for the predictive model — the M1b trap.** k_deg is in-sample identifiable (shallow
interior profile) and the net-negative channel is the NN's own recipe, so the trajectory IS describable
by it (in-sample cost 0.527 → 0.413). But it OVERFITS: qP_IB_max non-identified (pins at bound), K_IB
trades off (S_met abandoned), alpha_T destabilizes, and the held-out decline is still missed. Same
lower-train-cost / worse-held-out wall as smet_independent / induction-decay / M1b / ibtransient. **This
is the 7th trajectory/IB mechanism tested** (refold, ibdeg, dilution, Hill-IB, ibtransient,
smet_independent, smet_ibdeg) — including the NN's actual recipe — and every one that captures the
decline overfits from 4 sample points/run. The trajectory shape (front-loading, IB peak/decline) is NOT
identifiable/generalizable from the sparse LDH data; the endpoint model M0_aload_qPTexp remains the
predictive one. **Option 2 (Haldane-in-S_met sharper shutoff) is now deprioritized**: K_IB→bound shows the
S_met formation decline is already redundant with degradation, so a sharper S_met shutoff would meet the
same ceiling. Variant `M0_smet_independent_ibdeg` retained as tested-and-rejected. Artifacts:
`results/smet_ibdeg_test.png`. Remaining live thread: **Xiso transfer** (cumulative-load qP as a shared
mechanism across two datasets — the one thing that could add real identifying information).

### Decoupled sol/IB models — the partition was the wrong frame; decoupling captures the dynamics (2026-07-28)

`explore_psol_pib.py`, `decoupled_test.py`, `decoupled_v2_test.py`, `results/{explore_psol_pib,decoupled_test,decoupled_v2_test}.png`.
A user-driven reframe: **P_sol and P_IB have different dynamics** (confirmed from data):
- **P_sol**: a steady accumulator — sustained qP_sol ~0.008–0.016, present T≤30, **exactly 0 at T35**, never declines.
- **P_IB**: **T30-transient, T35-sustained** — at T30 it peaks then declines (F13 0.71→0.54, F18 0.94→0.83; apparent qP_IB→0 late); at T35 it keeps rising (F07 1.35→1.74, F19→2.45, F12→2.85). Absent at T25. The "decline" is **regime-specific (T30), not a universal IB property** — at T35 IB accumulates like sol. So forcing sol+IB through a shared `qP·X` partition (or shared Monod·η) is the wrong frame; they are independent processes.

**v1 — `M_decoupled_ibtransient` (complementary gates + T30-gated clock) — FAILED (diagnostic).** Independent
qP_sol/qP_IB amplitudes but **complementary** T-gates (g_sol+g_ib=1). The clock `z=exp(-t_ind·g_sol/τ)` runs only
at T30 (g_sol~1), off at T35. Result: clock **rejected** (τ→20, the 20 h upper bound — wants no decay), qP_IB_max
pinned at 0.10, T30 IB still rises. **Diagnosis:** complementary gates force a sol-vs-IB tradeoff at T30 — the fit
landed g_sol(T30)=0.95, **g_ib(T30)=0.05**, starving IB formation → qP_IB_max pins at its bound trying to
compensate → still under-predicts the peak → **can't afford to shut formation off** → clock pulled to τ=∞. Same
trap as original ibtransient, recreated by the complementary-gate partition. Cost only 0.514 (−2.5% vs 0.527).

**v2 — `M_decoupled_v2` (INDEPENDENT gates + T30-gated clock) — the dynamics are captured (first time).**
The fix: **non-complementary** gates — g_sol(T)=sigmoid(θ_sol0−θ_solT·(T−30)) (on T≤30), g_ib(T)=sigmoid(θ_ib0+
θ_ibT·(T−30)) (on T≥30), so BOTH can be ~1 at T30 (data has both). Clock z=exp(-t_ind·g_sol/τ) unchanged. Full-data fit:

| | baseline | v1 | **v2** |
| --- | ---:| ---:| ---:|
| cost | 0.527 | 0.514 | **0.448** (−15%) |
| τ_IB | — | 20 (off) | **~0.7** (engaged) |
| gates @T30 | — | g_sol .95 / g_ib .05 | g_sol 1.0 / g_ib .31 (independent) |
| F18 dIB (mdl/obs) | +.018/−.028 | +.034/−.028 | **−.027/−.028** (matches) |
| F15 dIB | +.051/−.010 | +.084/−.010 | **−.019/−.010** (matches) |
| T35 F07 dIB | +.490/+.372 | +.512/+.372 | +.216/+.372 (sustained ✓) |

- **τ_IB is identifiable** — interior minimum at ~0.7 h (profile cost: 0.501@0.5 → **0.448@0.7** → 0.495@1.0; rises
  both sides). Not pinned. The clock timescale is real (~42 min IB-formation fatigue at T30).
- **Fold C (holdout F18): the decline direction generalizes** (held-out dIB −0.017 vs obs −0.028 — the clock
  transfers to an unseen run) — the first trajectory mechanism whose *dynamics* extrapolate.
- **Remaining issue — peak height, not overfitting:** qP_IB_max pins at its 0.10 bound → IB **peak under-predicted**
  (full F13 0.33/0.54; fold-C F18 0.43/0.83). A fast clock (τ≈0.7) builds the peak in ~1–2 h then kills formation,
  so a high peak needs high formation, which the 0.10 cap forbids (fold-C train pushed τ→0.5 and qP_IB_max→0.10,
  both bounds). This is a **peak-vs-decline trade-off**, distinct from the overfit wall that sank the prior 7
  mechanisms: the dynamics (decline) are real and identified; only the magnitude is capped.
- **Test (2026-07-28): qP_IB_max bound raised 0.10→0.30 — revealed a peak-vs-decline TRADE-OFF, not a bound
  limit.** Multistart now lands in one of two basins: (a) **decline-basin** τ≈0.7, qP_IB_max→0.10, captures the
  decline, under-predicts peak (F13 0.33/0.54), cost **0.448** (best found); (b) **peak-basin** τ≈4.2,
  qP_IB_max≈0.034 (interior), matches the peak (F13 0.48/0.54, F18 0.83/0.83), **no decline** (F13 dIB +0.036),
  cost 0.483. Fixing τ=0.7 with qP_IB_max free → 0.072 (interior), still under-predicts peak (0.37/0.54, cost 0.499).
  **Cause:** the immediate-decay clock `z=exp(−t/τ)` suppresses formation from t=0 (z≈0.26 at t=1 h for τ=0.7), so a
  fast-enough clock to decline also starves the peak. The 10-param landscape is multimodal → 2-start fits unstable.
  The dynamics are real; the immediate-decay clock is the wrong *shape*.
- **Next (proposed, not yet built): delayed shutoff** `z = 1 for t<t_delay, else exp(−(t−t_delay)/τ)` — full
  formation until t_delay (peak builds to observed height), then shutoff (decline). Decouples peak (formation ×
  delay window) from decline (τ after delay) → can capture both. Biologically the stress response runs at full
  strength then chaperone-mediated adaptation shuts off aggregation after a delay. +1 param (t_delay).

### Decoupled v3 (delayed shutoff) — captures peak+decline in-sample but OVERFITS (params sloppy) (2026-07-28)

`decoupled_v3_test.py`, `plot_decoupled_comparison.py`, `results/decoupled_comparison.png`. Built v3
(`M_decoupled_v3`, smooth-softplus delayed shutoff; the hard `max()` kink stalled the optimizer's t_delay
Jacobian >1h, smoothing fixed it to ~5 min/fit). **Captures BOTH peak and decline in-sample — best cost yet:**

| model | cost | F18 IBend (mdl/obs) | F18 dIB |
| --- | ---:| --- | --- |
| baseline | 0.527 | 0.90/0.83 | +.018/−.028 |
| v2 (decline-basin) | 0.448 | 0.69/0.83 | −.027/−.028 |
| **v3 (delayed)** | **0.417** | **0.76/0.83** | **−.029/−.028** |

t_delay≈3.2 h, τ≈0.7. Peak and decline together for the first time. **BUT the params are the overfit signature:**
qP_IB_max pins at its 0.30 bound, g_ib(T30) collapses to 0.03 (sloppy g_ib·qP_IB_max ridge), **alpha_T destabilizes
0.096→0.449** (sacrificing the CER-validated T-suppression to absorb variance), sol leaks into T35. **Fold C
(holdout F18): decline direction generalizes (dIB −0.021/−0.028) but IB level under-predicted (0.54/0.83), params
still sloppy.** Same overfit wall as the prior 7 mechanisms — now reached via decoupling. v3 fits the trajectory by
spending non-identified freedom; it does not generalize the magnitude. Variant retained as tested (descriptive only).
The comparison plot (`results/decoupled_comparison.png`) is the cleanest visual: the grey partition baseline cannot
decline IB at all; v2 (blue) declines from a low peak; v3 (orange) peaks+declines (closest to obs) but leaks T35 sol.

### Growth-burden hypothesis ruled out — μ decline is the LINEAR FEED, not product burden (2026-07-28)

A mechanism from a feed-only/IB-only study (burden reduces growth capacity, not production directly) was tested
against the data. μ does decline faster than qS (median f_μ≈0.5 vs f_qS≈0.7), BUT: (a) the decline does NOT track
IB burden (F17 with ~0 IB shows it strongly; F12 with max IB shows none) — it tracks LOW qS instead; (b) the decisive
check μ/(Y_xs·qS) ≈ 1 for F13/F18/F12 — **μ tracks the declining qS**, which is the feed policy (the feed is
roughly exponential-but-under-ramped, so qS declines as X·V outgrows F). So the μ decline is feed-limitation, already
in the model (μ=Y_xs·qS), NOT burden. **Conclusion: the burden→growth route is out; the IB decline is unambiguously a
production-side phenomenon (the v2/v3 clock).** F07 is the one exception (μ/(Y_xs·qS)≈0.65, a multi-stressor corner).

### Biomass (X) improvement — Y_xs(T) tested and rejected (too blunt); deviation is run-specific/volume (2026-07-28)

`yxt_test.py`, `plot_biomass.py`, `results/biomass_x.png`. The biomass residual is systematic at T35: F07 +4.0,
F19 +1.3, F12 +1.3 g/L over-predicted; apparent Y_xs drops 0.28 (T25/30) → 0.21 (T35). Tested `M0_aload_qPTexp_yxt`
(Y_xs(T)=Y_xs·exp(−β_T·max(T−30,0)), mirrors α_T). **Rejected: β_T→0** (fit moved it down from init 0.10). Forcing the
profile shows why — it's too blunt:

| β_T | cost | F07 | F19 | F12 |
| ---:| ---:| ---:| ---:| ---:|
| 0.00 | 0.527 | +3.97 | +1.26 | +1.30 |
| 0.05 | 0.621 | −0.43 | −2.77 | −3.62 |
| 0.10 | 0.896 | −5.55 | −7.71 | −9.90 |

β_T=0.05 fixes F07 but over-corrects F19/F12 (F07 wants ~0.05; F19/F12 want ~0.01) — a single T-yield factor can't fit
all three T35 runs. And cost rises because lowering T35 biomass also lowers T35 product (qP·X) — a **zero-sum biomass↔
product trade**. The T35 over-prediction is **run-specific (F07 corner: qs0.2 + O₂-stressed + high IB), not a uniform
T-yield effect**, consistent with the CER-validated constant yield. The deviation is the ±50% base-volume bracket +
run variability; the only true fix is a weighed base bottle (MF2-R6), not a model term. `M0_aload_qPTexp_yxt` retained
as tested-and-rejected.

**Verdict so far:** decoupling (the user's insight) was correct — independent gates + a T30-gated clock is the
first structure to capture the IB decline with an identifiable timescale and a generalizing decline direction.
The partition/coupling frame was genuinely the obstacle. Open: the IB peak height (qP_IB_max bound/identifiability)
and full fold-A/C/multistart confirmation (fits are ~5 min/start; background-task runtime limits interrupted the
full eval). Variants `M_decoupled_ibtransient` (v1, rejected) and `M_decoupled_v2` (adopted direction) retained.

### CORRECTION (2026-07-28, 8-start canonical run): v2 is DESCRIPTIVE, NOT identified — τ is multimodal/sloppy

The canonical `run_playground.py --model M_decoupled_v2 --n-starts 8` (full fold) revealed v2 is NOT the clean,
identified result implied above. τ_IB swings across multistart: 4.06 (3-start, cost 0.483) → 0.70 (4-start lean,
0.448) → **1.56 (8-start canonical, 0.425, best)** — all within Δcost≈0.02-0.06, i.e. a **flat/multimodal
landscape**, not an identified parameter. The best fit is still sloppy: qP_IB_max=0.30 (bound), alpha_T=0.39
(**destabilized** vs the CER-validated 0.096), theta_sol_0=6.0 (bound). So **v2 captures the IB-decline SHAPE
(trajectories do decline — visible in results/M_decoupled_v2/full/plots/) but its parameters are NOT robustly
identifiable** — same overfit/non-identification signature as v3/smet/the other mechanisms. The earlier "τ≈0.7
identifiable, generalizes held-out" framing was based on a single 4-start fit and was OVERSTATED. Honest read:
**decoupling captures the decline direction (real), but no decoupled variant has identifiable params** — they all
trade a trajectory fit for sloppy/bound-hitting/alpha_T-destabilizing params. This STRENGTHENS the overall
conclusion: identifiability is the wall for the partition frame AND the decoupled frame AND cross-protein
(Xiso transfer, below). The trajectory dynamics are not identifiable from this sparse data, period.

### Xiso transfer experiment — DOCUMENTED LOO-CV, NEGATIVE (2026-07-28)

`xiso_transfer_cv.py`, `results/xiso_transfer/{SUMMARY.md, full/, holdout_*/}`. Tested whether the IB-decline
clock transfers across proteins (LDH -> Xiso). Xiso = 3 runs on the DoE diagonal (low-low Ferm02 T25, center
Ferm03 T29, high-high Ferm01 T35), sol+IB measured. Model: decoupled sol/IB + **T-INDEPENDENT** clock
(z=exp(-t_ind/tau), independent gates) — because Xiso declines at T35 (where LDH's clock is gated off), so a
T-gated clock can't apply to both. Full fit (consistency) + LOO-CV (train 2, test 1, 3 folds):

| fold (held-out) | tau_IB [h] | held-out P_IB RMSE (range) | held-out dIB mdl/obs |
| --- | ---: | ---: | --- |
| full (all 3) | 5.13 | — | — |
| Ferm01 (T35/high-high, the decline run) | 3.34 | **0.468** | +0.026 / -0.157 |
| Ferm02 (T25/low-low, ~no IB) | 5.21 | 0.052 (trivial) | +0.000 / -0.063 |
| Ferm03 (T29/center) | 5.21 | **0.377** | +0.203 / +0.115 |

**Verdict: the clock does NOT transfer.** (1) tau_IB ~3.3-5.2 h on Xiso vs LDH ~0.7-1.5 h (~3-7x slower). (2) No
within-Xiso generalization: held-out RMSE good only for Ferm02 (trivial, ~no IB); FAILS for the two IB-bearing
runs (Ferm01 the decline run, Ferm03 the center) — trained on low-IB runs, the model cannot predict the high-IB
T35 trajectory or its decline. Root cause = sparsity (3 runs, no replicates, ~6 samples each). The one robust
transferable quantity is Y_xs (~0.36, strain/process level). This confirms the trajectory mechanism is NOT a
shared, protein-independent, identifiable clock — the ceiling holds cross-protein. Full writeup + per-fold
params/metrics/plots (train + red held-out TEST trajectory) in `results/xiso_transfer/`.

## Open tasks for continuation

### Task A — burden qCO₂ as an IB-partition driver — CLOSED (non-identifiable, 2026-07-24)
**Resolved:** tested via the independence diagnostic + the `M0_aload_burden` dynamic
model (see findings log, 2026-07-24). Burden is a qS proxy (partial r=0.12); the a_b
profile is monotone with no interior minimum, sign-wrong, destabilizes a_qS, and yields
no held-out gain. **Drop the burden state.** The original hypothesis write-up is kept
below for reference.

The post-induction burden respiration (ΔqCO₂, pre-vs-post) correlates r≈0.98 with IB
fraction within the 30 °C stratum (and IB carries a distinct extra cost/g vs soluble at
matched qS). Test whether an offgas-derived burden state drives partitioning better
than the qS-load term. Options, in increasing complexity/identifiability risk:
1. **Static, observational (diagnostic only):** regress final IB fraction on the
   measured median burden qCO₂ across the 30 °C runs — confirms/quantifies the link,
   not a dynamic model.
2. **Quasi-mechanistic:** add a latent burden state `b` driven by the model production
   rate, `db/dt = (qP_tot·X·k_b − b)/τ_b`, and let `phi_IB = sigmoid(θ₀ + θ_T·(T−30) +
   a_b·b)`. This is mechanistically a "production-cost → aggregation" story. Risk:
   `b` is essentially a filtered version of qP, so it is largely collinear with the
   existing `a_qS·log(qS)` term ⇒ likely not independently identifiable from these data
   (analogous to the hybrid's rejected M4/M5 latent states). Profile-likelihood `a_b`
   before trusting it.
3. **Use measured burden as an input** (not mechanistic, leaks observation into the RHS):
   only as a sensitivity check, not a defensible model.
Recommended order: do (1) first (cheap, quantifies the relationship), then (2) only if
(1) is strong AND `a_b` passes profile likelihood. **Expect (2) to be non-identifiable**
given the qP-collinearity; if so, the honest conclusion is "burden tracks IB but adds no
identifiable information beyond the qS-load term."

### Task B — production-form for the center underprediction (parked)
Substrate-inhibition (Haldane) and induction-decay were tested and rejected (overfit /
beyond-data / qP_max at bound). The remaining lever is the omitted acetate/overflow
carbon (no acetate column) and the uncorrected base volume (no base mass) — both need
data we don't have. Revisit only if acetate or base-volume data becomes available.

### Task C — gas as a biomass constraint / maintenance — CLOSED (overfits, 2026-07-24)
The corrected gas-implied biomass is ~within carbon-accounting uncertainty of offline X
(finding 2, corrected), so there is no large systematic biomass gap to "rescue." The gas-vs-biomass
**yield gap** (instantaneous ~0.39 vs cumulative ~0.35) was tested as a Pirt maintenance term `m_s`
(`m1b_maintenance_test.py`): m_s is in-sample identifiable but **overfits** — held-out F17 collapses
+0.42→−4.58, with unphysical Y_xs=0.50/m_s=0.076 that absorb the carbon-balance systematic error,
not real maintenance. **Conclusion: the yield gap is a carbon-accounting artifact (volume ±50%,
monohydrate/anhydrous ~10%), not a maintenance/biomass-bias term.** This closes the last gas-side
thread; the gas side is at its identifiability ceiling. Revisit only if glucose-carbon accounting
is firmly resolved, a weighed base bottle pins the volume, or an independent biomass measure appears.

### Task D — burden respiration as the missing post-induction CO₂ — CLOSED (non-identifiable, 2026-07-24)
**Resolved (negative).** `burden_co2_test.py` + `results/burden_co2/`. The α-CER test surfaced a
gas-observable gap (balance underpredicts post-induction CER by ~2–7 mmol/L/h at T25/T30), so the
hypothesis was: that shortfall is a production-proportional proteostasis burden β·qP·X (the
mechanism behind α), identifiable by construction as a regression slope on the measured gas
(unlike Task A's latent *partition* burden). **Tested and rejected.** Regressing the post-induction
CER residual (obs − balance, RQ-trusted, F09/F14 dropped) on the synthesis rate r_P_synth over
T25/T30: **β = −42 mmol CO₂/g, R² = 0.01, n = 177** — no correlation. Per-stratum β sign-flips
(T25 +193 / T30 −52 / T35 −69, all R²<0.2); per-fold β is unstable (A +0.5 / B −0.1 / C −50.8 — the
C swing is O₂-transfer-limitation contamination at the high-biomass qs0.6 corners). The shortfall is
a roughly **flat offset** (intercept ~+23 mmol/L/h), not a per-product cost.

**Why it failed (and what it means):** (i) little within-run leverage — qP·X is fairly steady during
induction, so pooled points have no slope to identify; (ii) the high-biomass corners hit O₂
enrichment / transfer limitation, biasing observed CER low and manufacturing a spurious negative
slope; (iii) the shortfall, if systematic, is **maintenance-like (∝ biomass), not production-like**
— which is the M1b/Task-C question, not a per-gram-product burden. **Verdict: do NOT add a
production-proportional burden term.** α stays as the empirical T-suppression (biology-validated by
the CER test); the post-induction CER shortfall is a documented residual, not a model term. The
remaining gas thread is **M1b maintenance** (does the flat offset scale with biomass and survive a
corrected-volume + reliable-gas refit?) — low priority, likely small. Artifacts:
`results/burden_co2/{burden_co2_test.md, burden_regression.png, burden_residual.png}`.

### Task E — structural alternatives (E.1 tested 2026-07-24: volume NOT pinned by ∫CER; E.2 is now the lead)

**Motivation / reframe.** Tasks A–D exhausted the *parameter-space* of the current structure: every
metabolic-burden/maintenance parameter is non-identifiable or overfits, all tracing to the same
~10% carbon-balance systematic error dominated by the **±50% base-volume bracket**. The working
hypothesis going forward: **those rejections may be volume artifacts, not absent biology** — `m_s`
and burden could be real but unidentifiable *because* the volume injects too much noise. The lever
is therefore not another parameter but a **mechanistic restructuring of the volume**, which (if it
pins the volume) may un-freeze the gas-side identifiability. α is already validated as biology
(CER test), so the model is sound; this is about removing the accounting noise.

**E.1 (lead) Base-volume from the respiratory acid load — `base_acid_reconstruct.py`. RESULT (2026-07-24): CLOSED, NOT pinned** (see findings-log entry above for the full result): the 1:1 CO₂ stoichiometry over-predicts the bracket-scale base ~6×; carbon closure ≈103% ⇒ the CO₂ is real but ~65% is **stripped, not neutralized** (free ~0.2–0.4 neutralization fraction, not stoichiometric); and CER tracks feed near-collinearly (r≈0.92) ⇒ no CO₂-specific shape. The volume stays at its ±50% bracket; the gas-side identifiability ceiling is not shown to be a volume artifact. *The hypothesis + plan below are retained as historical (pre-test) reference.*
Mechanism: in glucose-limited *E. coli* fed-batch the NH₄OH base demand is overwhelmingly **acid
neutralization**, and the dominant acid is respiratory CO₂ (carbonic acid). So
`base_volume(t) ≈ k · ∫ CER(t)·V(t) dt` (+ smaller organic-acid / N-assimilation terms). This is
**pH-stat stoichiometry, not a free bracket**: NH₄OH 6.95 M neutralizes carbonic acid, and at pH 7
≈1 mol base per mol CO₂. Empirical smoking gun already in hand: **the Pump2 (base) signal
correlates 0.87 with cumulative CO₂** (base_volume section) — so the *shape* of base is ∫CER; only
the *scale* is in question, and the scale is stoichiometrically constrained.

Plan (diagnostic-first, uses existing data, no refit needed to test):
1. Per run, compute `acid_load(t) = ∫ CER·V dt` [mol CO₂] and predict `base_pred = acid_load /
   C_NaOH_eq` (stoichiometric, ~1 base:1 CO₂ at pH 7; sweep the pH/pKa assumption 0.7–1.0).
2. Compare `base_pred` to (a) the **N-floor** (lower bound) and (b) the **Pump2-derived scale**
   (shape×scale). Does stoichiometric ∫CER land inside the [N_floor, 2×N_floor] bracket, and where?
3. **Residual = overflow/acetate:** `non_respiratory_acid = base_actual − base_pred` ≈ organic
   acids (acetate). This quantifies overflow **without an acetate column** — directly reviving
   Task B (the qS-peak / center underprediction). Check whether the implied acetate tracks qs0.6
   (where the qS-peak lives).
4. **Decision tree:**
   - If `base ≈ stoichiometric ∫CER` → the volume is **pinned with ~zero free parameters**. Adopt
     it as the default volume (replace the N-floor bracket in `base_reconstruct.py`), then
     **re-test M1b (`m_s`) and Task D burden** — if they now identify *and generalize held-out*,
     the earlier rejections were volume artifacts (real biology rescued); if they still fail, the
     gas-side ceiling is real and biological.
   - If `base ≪ ∫CER-stoichiometric` → the CO₂ is leaving another way (mass-transfer/calibration,
     or the wet-gas N₂-balance over-reads CER) — a decisive finding about gas quality that reframes
     all offgas use.
   - If `base ≫ ∫CER` → large non-respiratory acid (overflow); model it explicitly (acetate state)
     — but note carbon closure ~103% bounds this.

**E.2 O₂-transfer-limited production — RESULT (2026-07-24): CLOSED, NOT supported.** See findings log: O₂ demand peaks at qs0.6 (inlet O₂→58 %, F12 pO₂→0.6 %) but does NOT limit LDH production (RQ~1 = controller met demand; qS-peak is T30-only, absent at T25/T35 where O₂ stress is worse; F12 starvation didn't depress product; r(Ptot/X, enrich)=−0.14). Do not add an OTR capacity term. *(Original hypothesis retained below.)* **Historical:** the offgas diagnostics show the
high-biomass corners (qs0.6) hit O₂ enrichment and transfer limitation (OUR plateaus, inlet O₂
21→53%). A *capacity* structure `qP_eff = min(qP_Monod, OTR/Y_PO)` (or μ capped by OTR) is
mechanistically real and explains the **qs0.6 production plateau / qS-peak** using measured OUR +
inlet O₂ — distinct from the supply-proxy Monod. Test after E.1 (cleaner volume first). Caveat:
confound with the gas-transfer artifact itself.

**E.3 (speculative) Structured biomass / cofactor limitation.** The qS-peak (specific production
peaks at qs0.4, lower at qs0.6) is the signature of a non-producing subpopulation (overflow → cells
grow but don't express) or **cofactor/redox saturation** (LDH is a flavoprotein needing FAD/NAD⁺;
at high expression the cofactor pool caps soluble folding → excess aggregates). A producing-fraction
or cofactor-pool state is mechanistically grounded and LDH-specific, but likely data-limited (latent
state, 4 sample points/run) — profile-likelihood before trusting.

**E.4 (quick swap) Luedeking-Piret production** `qP = α·μ + β` instead of Monod-in-feed-proxy —
different qP(qS) shape, possibly better conditioned; probably still qS-proxy-limited but cheap to
test.

**Recommended order / strategic pivot (2026-07-24):** E.1 (∫CER volume), E.2 (O₂ cap), and A1 (S_met
load-Haldane, Müller-inspired) are all DONE and all NEGATIVE — every mechanistic lever for the qS-peak and
the gas-side identifiability has been ruled out (the qS-peak is consistently T30-specific = a partition
effect; the volume is un-pinnable from existing data). **Expert assessment of "is the data exhausted?": the
measured variables are final, but the *information* is not** — the model fits ~4 offline points/run while
each run carries ~80 dense online points (CER/OUR/pO₂/flows) whose *dynamic trajectory* is used only as a
failed static prediction (M1a) or one-off diagnostics. The under-exploited vein is **state estimation**:
fuse sparse offline anchors with dense online gas to reconstruct μ(t)/qP(t) at dense resolution and estimate
the volume/carbon-calibration nuisance as a per-run latent state (rather than the ±50 % bracket). This
differs from M1a (which tried to match gas *magnitude* to one global yield and failed): a filter uses gas for
the *dynamics* and estimates the nuisance, separating the two failures that doomed M1a.

**Next lever (plan B): soft sensor — Stage 1 DONE (2026-07-24), Stage 2 next.** Stage 1 (`soft_sensor.py`)
succeeded on two of three fronts: (i) a per-run scale `s` reconciles gas-implied growth with offline biomass
at R²≈0.99 — a defensible per-run calibration that point-estimates the volume/carbon nuisance (replacing the
±50% bracket); (ii) it VALIDATED the gas for the T30 regime (P_recon/P_obs ≈ 1.1 there) while quantifying
that the O₂-stressed/bad-gas corners are off (0.5–5×) — i.e. the gas + carbon balance are trustworthy exactly
where the model's action is. The third front — dense qP(t) — is residual-sensitive (reliable for T30,
artifactual at corners under the constant-s assumption).

**Stage 2 — DEAD (stoichiometric limit, 2026-07-24):** the dual carbon+electron balance cannot split LDH
growth from product because γX≈γP≈4.1 (equally reduced; denominator 0.09 amplifies the closure error ~11× →
garbage rP). This is the root cause of ALL gas-side failures (M1a/M1b/burden): the gas constrains only the
total rX+rP, never the split. Not fixable by data or filters — it is a property of the LDH composition.

**Where this leaves the investigation.** The gas side has hit a fundamental stoichiometric ceiling for LDH.
The reliable dense qP is Stage 1's T30 residual (offline-splits the gas-constrained total; validated). The
qS-peak remains a T30-specific partition effect (not load, not O₂, not gas-split-resolvable). Realistic
remaining levers, all lower-probability: (i) feed the per-run Stage-1 scale `s` back as a volume/carbon
calibration and re-test gas-side params — but without expecting the split to resolve (it structurally
cannot); (ii) E.3/E.4 (structured biomass / cofactor / Luedeking-Piret) — only testable against the Stage-1
T30 qP, and likely data-limited; (iii) **new data** (weighed base bottle MF2-R6, acetate column,
intermediate qS levels 0.2/0.4/0.6) — still the only thing that would actually move the identifiability. The
adopted `M0_aload_qPTexp` stands; the documented limits (qS-peak, ±50 % volume, gas split) are
stoichiometric/data ceilings, not modeling deficits.

**Option A — DONE (2026-07-24): the model is ADEQUATE.** `model_adequacy_test.py` shows qs0.6 specific
product is within triplicate noise (+0.9–1.6 SD); the qS-peak was a dilution artifact all along (Ptot flat,
biomass rises). The larger residuals are biomass (F07 +6.3 SD etc., scattered) = the ±50 % volume bracket,
not a production mechanism. **`M0_aload_qPTexp` is the final model for this dataset.** Options B/C/D
(allocation model / optimal-experimental-design / hierarchical-Bayes uncertainty) are available but not
required for correctness — B is unmotivated (product residual within noise); C (design the next run to
resolve saturation/volume) and D (deliver honest predictive uncertainty) are the only forward directions
with real value, and both are about *next steps* (new data / uncertainty quantification) rather than fixing
the current model. **The investigation has reached a principled, defensible conclusion.**

## Data corrections (2026-09-04) — inputs changed; refit COMPLETED, verdicts stand

Two corrections were applied to the pipeline data after the modeling conclusions above were
reached. **All `M0_aload_qPTexp` fits, metrics, and verdicts in this document predate them.**

1. **Induction timing (all 11 runs).** The SCADA phase flip into 'induced Fedbatch' *leads* the
   physical IPTG bolus (operator Evaluation sheets): +6–28 min in F07–F15, **65 min (F17/F18/F19)
   and 85 min (F16)** — in F16–F19 the end-of-fedbatch sample S2 was drawn *after* the phase flip
   and was misflagged as post-induction. Fixed by `processing.induction_time_h` in every run config
   (new `correct_induction_time` step in the package); all runs + consolidations regenerated.
   Verified side effect: post-induction samples now sit on a clean ~2.0 h grid in every run.
   Modeling impact: qP activation starts 1–1.4 h later in F16–F19; S2 correctly pre-induction.
2. **F19 Protein IB was an artifact — corrected.** The campaign series (1.2537/2.2297/2.3852/
   2.4537 g/L) is the sheet's own IB/Sol ratio column pasted into the IB column during the
   14.04.2026 below-LOD rework (sol retracted to "n.d."; identity `canonical_IB ÷ pasted =
   retracted sol` holds 4/4 to six digits; the true titers predate it in the 23.03 MODDE import).
   Corrected to **1.1073153 / 2.2941798 / 2.7247912 / 3.1947748 g/L** (matches the publication)
   in `F19/data/samples.xlsx`; F19 + consolidations regenerated. Full provenance:
   `examples/LDH/F19/CORRECTION_F19_IB.md`. Modeling impact: T35/qs0.4 endpoint +0.74 g/L
   (2.454→3.195); the T35 group is now F07 1.744 / **F19 3.195** / F12 2.848 — a mid-qs hump
   like T30 (was monotone-rising). Directly touches the T35 stratum that informs `alpha_T`,
   and the "T35 total product ~40 % below T30" magnitude behind qP(T).

**REFIT COMPLETED (2026-09-04): the model stands — every verdict survives the corrected data.**
Canonical refit of `M0_aload_qPTexp` (folds A/B/C, seeds 42+43 × 3 starts; seeds identical
within each fold; the pre-correction canonical runs were seed 42 only). Pre-correction
artifacts archived at `results/M0_aload_qPTexp_PRECORRECTION_20260724/`; the numbers below
supersede all earlier `M0_aload_qPTexp` results in this document. The pkl cache
(`LDH_consolidated_results_processed.pkl`) regenerated with the xlsx, so the fits consumed
the corrected data (mtime-checked, `data.py:_load_df_cached`).

**Parameters (pre → post correction):**

| fold | train cost | Y_xs | qP_max | K_S | theta_0 | theta_T | a_qS | alpha_T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A (holdout F17) | 0.513→0.502 | 0.362→0.362 | 0.043→0.051 | 0.50→0.57 | −2.50→−2.29 | 8.0 (bound)→8.0 (bound) | −1.05→−1.01 | **0.098→0.090** |
| B (holdout F13–15) | 0.286→0.328 | 0.360→0.360 | 0.034→0.035 | 0.39→0.33 | −1.45→−1.10 | 1.20→1.28 | 2.79→**4.0 (bound)** | **0.073→0.070** |
| C (holdout F18) | 0.501→0.494 | 0.362→0.362 | 0.049→0.048 | 0.60→0.51 | −2.01→−2.01 | 1.23→1.37 | 0.86→0.71 | **0.096→0.090** |

- **Y_xs unchanged** (0.360–0.363 across folds; full-data 0.363) — the corrections do not
  touch the growth side.
- **alpha_T down in all three folds (per-fold mean 0.089→0.083; full-data 0.096→0.090,
  η(35)=exp(−5α)=0.62→0.64)** — exactly the predicted direction: F19's true T35/qs0.4
  endpoint is +30 % above the artifact, so less T-suppression is needed.
- **Partition still the weak link**: fold A keeps θ_T at its 8.0 bound with a_qS negative
  (−1.01); fold B's a_qS now pins at its +4 bound (was interior 2.79). Identifiability
  verdicts unchanged — report a_qS/θ_T as ranges, not constants.

**Held-out (test) metrics — no material change (macro RMSE g/L, bias in parens; pre → post):**

| fold | X | P_sol | P_IB |
| --- | --- | --- | --- |
| A (F17) | 0.88→0.89 (+0.64→+0.64) | 0.22→0.39 (+0.16→+0.30) | 0.30→0.28 (−0.24→−0.21) |
| B (F13–15) | 1.39→1.40 (+1.04→+1.05) | 0.70→0.66 (+0.55→+0.48) | 0.15→0.11 (+0.11→+0.07) |
| C (F18) | 1.39→1.39 (+0.26→+0.27) | 0.54→0.46 (−0.40→−0.25) | 0.27→0.35 (+0.21→+0.27) |

All shifts ≤ 0.16 g/L; X stable everywhere; product channels mixed small changes
(B/C soluble and fold-B IB better; fold-A soluble and fold-C IB slightly worse).
**No conclusion flips.**

**α-CER biology test re-run (`alpha_cer_test.py`, full-data α=0.0901):** T35 stratum
post-induction CER bias **+6.0 mmol/L/h under the degradation reading vs +16.3 under
under-recovery** (obs sits ~10 mmol/L/h above the under-recovery curve; 80 % of T35
points closer to degradation). F19 on its corrected series now clearly *supports*
degradation (+5.2 vs +14.8, 0.86); F12 remains the lone contradiction (0.55,
O₂-transfer-limited gas). Note the T30 control baseline is +12.2 — the T35 residual
sits *below* the T25/T30 baseline (+4.7 / +12.2), i.e. no excess unexplained CER at
T35 at all. **Under-recovery stays ruled out; α is biology** (within-biology split
proteolysis vs stress respiration remains unresolvable, as before).

**Adequacy re-run (`model_adequacy_test.py`, full data):** verdict snapshot unchanged —
qs0.6 max |product residual| 3.2 SD (within triplicate noise), max biomass residual
6.3 SD (the ±50 % volume signature). New nuance: **F19 (T35/qs0.4) is now mildly
UNDER-predicted (−0.69 g/L, −2.0 SD)** — with the corrected endpoint the T35 group
shows the same mid-qS hump as T30 (F07 1.744 / **F19 3.195** / F12 2.848), i.e.
total-product saturation by qs0.4 is now visible at BOTH temperatures, strengthening
the "Ptot flat above qs0.4; extra substrate → biomass" reading. The mild qs0.6
overprediction persists (F10 +3.2, F12 +2.3, F18 +1.5 SD).

**Full-data canonical parameters (corrected data; identical fit from both test
scripts):** cost 0.525 (was 0.527); Y_xs 0.363, qP_max 0.0493, K_S 0.541,
a_qS 0.77 (was 1.18), **alpha_T 0.0901 (η(35)=0.637)**.

**F19 trajectory check:** with the corrected series the gate opens at the operator
induction (~23.0 h), S2 sits at zero pre-induction, and the trajectory tracks the
sustained titers (final 3.19 obs vs ~2.55 pred — the familiar moderate-qS shortfall,
now mirrored in the T35 arm; the old artifact series *decelerated* to a stall:
increments 1.25→0.98→0.16→0.07 g/L).

**Data caveat recorded:** F16's export ends at 29.43 h but the operator logged S6 at
30.68 h — the last F16 sample is missing from the pipeline data (truncated export;
not fixable in-repo).

**Verdict: `M0_aload_qPTexp` remains the final model, now validated on the corrected
data.** All structural/identifiability conclusions (α = biology, Y_xs, qS-peak as
dilution artifact, gas-side γX≈γP ceiling, ±50 % volume bracket,
trajectory-not-identifiable) are unchanged. The trajectory-thread numbers
(decoupled/smet/IB-decline entries of 2026-07-27/28) predate the corrections — F19
endpoint citations there (e.g. "F19→2.45") refer to the old artifact series — but
their verdicts (descriptive, non-identified, overfitting) are robust to held-out
shifts of ≤ 0.16 g/L; those threads stay closed.

### Complete-design LOGO validation (Tier 2, paper-style) — accuracy matches the hybrid across the full FCCD (2026-09-04)

`logo_summary.py` + `results/logo_summary/{logo_summary.md, logo_summary.csv}`; fold
definitions in `data.py:FOLDS` (`LOGO_F07/F09/F10/F12/F16/F19` added). Mirrors the
Fermentation paper's two-tier validation design (methods + `tab:fulldesign`): Tier 1 is
the three 30 °C isotherm folds A/B/C (above); Tier 2 is a complete-design
**leave-one-group-out** CV — **nine hold-out groups covering all eleven fermentations**
(T center triplicate always held out together), each fold refitting the canonical
`M0_aload_qPTexp` from scratch (seeds 42+43 × 3 starts, deterministic and identical
within folds). Aggregate entries are mean ± SD of per-group pooled RMSEs across folds
(the paper's ± is across 8-seed ensemble members; there is no ensemble spread here).

| State | Isotherm A–C | Full design (9) | Corners (4) | Axial (4) | Center (1) |
| --- | --- | --- | --- | --- | --- |
| Biomass (X) | 1.24 ± 0.30 | 1.45 ± 0.62 | 1.60 ± 0.78 | 1.31 ± 0.61 | 1.43 |
| Soluble (P_sol) | 0.51 ± 0.15 | 0.34 ± 0.27 | 0.27 ± 0.31 | 0.32 ± 0.22 | 0.67 |
| IB (P_IB) | 0.25 ± 0.12 | 0.31 ± 0.33 | 0.33 ± 0.43 | 0.35 ± 0.30 | 0.12 |
| Total (P_tot) | 0.47 ± 0.25 | 0.55 ± 0.21 | 0.60 ± 0.21 | 0.46 ± 0.20 | 0.75 |
| Protein states | 0.38 ± 0.04 | 0.33 ± 0.09 | 0.30 ± 0.11 | 0.33 ± 0.07 | 0.39 |
| Macro-average (measured) | 0.66 ± 0.13 | 0.70 ± 0.18 | 0.73 ± 0.23 | 0.66 ± 0.17 | 0.74 |

**The 8-parameter mechanistic model matches the hybrid NN ensemble's complete-design
accuracy almost exactly** (paper `tab:fulldesign`: full-design macro 0.71 ± 0.22, X
1.47 ± 0.56, P_sol 0.45 ± 0.41, P_IB 0.21 ± 0.20, P_tot 0.62 ± 0.32) — the paper's core
Tier-2 conclusion replicates for the parametric model: **accuracy is uniform across the
held-out run's design role**, and holding out a factorial corner (genuine design-space
extrapolation — the remaining runs no longer span that vertex) does not degrade it
(corners 0.73 vs isotherm 0.66, within SD). Known residual structure persists and
mirrors the pre-correction findings: center P_tot underpredicted (fold B +0.56 bias),
T35 IB endpoints scattered around the corrected mid-qS hump (F07 +0.76 / F19 −1.04 /
F12 +1.67 g/L endpoint error — the qs0.4-vs-0.6 realized-qS compression limit), X
residuals scattered = the ±50 % volume signature. The paper's "opposing patterns at the
25 °C edge, no single directional bias" also replicates (F09 P_sol under +0.40 vs F10/F16
over −0.29/−0.32). Structural zeros behave correctly (P_sol ≡ 0 at T35, P_IB ≡ 0 at T25
→ 0.000 RMSE in those cells). Per-group detail incl. endpoints in
`results/logo_summary/logo_summary.md`. Caveat for comparisons: single-run folds make
the group RMSE equal that run's RMSE, so our fold-level SDs are wider than the paper's
member-averaged SDs; seeds contribute no spread (deterministic).

### Decline-revisit (2026-09-04, user-driven): the product curves ARE too straight — and 9 approaches to bending them now all fail held-out

User observation from the LOGO plots (correct): the adopted model's P_sol/P_IB trajectories are
quasi-straight/convex while the data is concave (front-loaded) — F13 P_IB peaks 0.70→declines to
0.54 vs a model that stagnates at ~0.42; F18 peaks 0.94→0.82 vs a model still climbing at 0.55;
P_sol increments decelerate (+0.94→+0.42) vs a convex accelerating model curve. Structural cause:
near-saturated Monod(qS) × rising X = accelerating `qP·X`. The mismatch is real; the question was
whether any identifiable mechanism can afford the bend. **Two new variants were built and tested;
both fail — completing the strongest possible version of the not-identifiable verdict:**

1. **`M0_aload_qPTexp_decl` (shared-pipe decline + priors) — REJECTED, dominated in-sample.**
   Backbone + `qP *= exp(−t_ind/τ_P)` (both products) and an extra IB shutoff `exp(−t_ind/τ_IBd)`
   (`_DECL` in models.py). Priors (`fit.py:PRIOR_LOGNORMAL`, log-normal MAP penalties, unit
   weight): τ_P ~ LN(12 h, 0.7) from qp_decline's f_qP≈0.54 over ~9 h; τ_IBd ~ LN(2 h, 0.7) from
   the decoupled clock. Result: 8-start-stable cost **1.62 vs 0.525 baseline** (not a bad basin);
   qP_max pins at 0.10, α halves to 0.04 (steals T35 suppression); and the decisive
   **frozen-backbone diagnostic**: with all baseline params fixed and only the τ's free, cost is
   **2.16** — bending the curves *at correct production levels* costs more than any reallocation
   recovers. The sparse range-weighted objective *prefers straight* when one production channel
   feeds everything. Mechanically clean rejection: it's the coupling, not the decline.

2. **`M_decoupled_v2` + anchored clock prior (the hybrid's own structure, regularized) — REJECTED
   on held-out.** v2 = independent sol/IB gates + T30-gated shutoff clock `tau_IB`; prior
   τ_IB ~ LN(1.5 h, 0.7) anchors the documented 0.7/1.6/4.1 multimodality — and it *works*:
   τ lands at 1.487–1.507 in all 9 folds (interior, prior-centered). In-sample cost **0.407 =
   best yet** (vs 0.425 free-fit, 0.527 baseline). But the freed amplitude/gates are where the
   instability went: `qP_IB_max` at its 0.30 bound in 6/9 train sets, gate params at bounds in 4/9,
   and **α destabilized in every fold (0.141–0.396; CER-validated value is 0.090)**. Held-out
   (`results/logo_summary_M_decoupled_v2/`, 9 groups vs the straight baseline):

   | State (full design, 9) | baseline `M0_aload_qPTexp` | v2 + prior |
   | --- | --- | --- |
   | X | 1.45 ± 0.62 | 1.45 ± 0.63 |
   | P_sol | 0.34 ± 0.27 | 0.34 ± 0.19 |
   | **P_IB** | **0.31 ± 0.33** | **0.37 ± 0.34** |
   | P_tot | 0.55 ± 0.21 | 0.62 ± 0.36 |
   | Macro | 0.70 ± 0.18 | 0.72 ± 0.21 |

   Isotherm folds are the tell: P_IB **0.55 ± 0.41 vs 0.25 ± 0.12** — fold C (F18 held out)
   **explodes** (predicted IB spike to ~2.2 g/L vs obs 0.63; P_IB RMSE 1.03, bias −0.83) while
   fold B's fit collapses `theta_ib_0` to its −4 bound and **starves** F13's IB (rises to ~0.13 vs
   obs 0.71). Opposite failure modes from the same structure: the prior fixed the *timescale*
   instability and the *amplitude/gate* instability took its place, and held-out it is worse than
   the straight baseline's honest flat-IB error.

**Verdict (upgraded from 2026-07-28).** Nine distinct approaches to the P_curvature — refold,
ibdeg, dilution, Hill-IB, ibtransient, smet_independent, smet_ibdeg, decoupled v1/v2/v3, and now
prior-regularized shared-pipe decline AND prior-anchored decoupled — all fail to improve held-out
product prediction at ~4–5 sample points/run. The straightness is the honest optimum for this
sampling density, not a fixable model deficit; the adopted `M0_aload_qPTexp` remains final. The
hybrid NN's ability to *describe* the decline in-sample (user-confirmed: its held-out P_IB error
is low, ~6 % NRMSE in fold B) comes from ~60–200 parameters whose staged training acts as implicit
regularization; its held-out P_sol error (≈1.07 vs our 0.66) pays for it, and the parametric port
of the same recipe does not transfer. **What would actually make the curves bend identifiably:
denser off-line sampling in the induction window (~every 2 h)** — then the decline shape is
measured, not inferred, and any of these mechanisms becomes testable. Artifacts:
`models.py:_DECL`, `fit.py:PRIOR_LOGNORMAL`, `results/logo_summary_M_decoupled_v2/`,
`results/M_decoupled_v2/holdout_*/` (9 folds, TEST plots show the F18 explosion and F13
starvation), `logo_summary.py <model>` (now model-parametrized).

### Improvement program on frozen data (2026-09-04 cont.): X-conditioning negative; posterior-ensemble bands added, P_IB −10 % pooled

User directive: no new samples — improve with the data at hand. Two directions tested/delivered:

1. **X-conditioning diagnostic (`xcond_test.py` + `results/xcond/`) — NEGATIVE, informative.**
   Products re-simulated with X(t) = measured-X interpolation (soft-sensor framing; params =
   each fold's stored fit). Aggregates: P_sol 0.312 vs 0.337 uncond., P_IB 0.318 vs 0.314,
   P_tot 0.532 vs 0.552; the T35 ordering stays scrambled (F07 2.46 / F19 2.14 / F12 4.50 vs
   obs 1.74 / 3.19 / 2.85). **Product error is NOT inherited biomass error** — with perfect X
   the same structure makes the same product mistakes. Closes the whole "use X better" class;
   independently re-confirms the production-structure ceiling.

2. **Posterior-ensemble predictive bands (`posterior_ensemble.py` +
   `results/posterior_ensemble/`) — DELIVERED, and the mean wins on P_IB.** Laplace
   approximation at each fold's stored optimum (finite-difference Jacobian of the actual
   weighted residuals, cov = s²(JᵀJ)⁻¹ with pinv for the sloppy directions), 200 samples/fold,
   held-out predictions. Results (pooled over the 13 run×state rows per state):
   **P_IB 0.276 → 0.249 RMSE (−10 %)** — wins exactly where the MAP overpredicts (convexity on
   the sloppy ridge): F12 0.903 → **0.597 (−34 %)**, fold-B P_IB all better (F13 0.170→0.142,
   F14 0.106→0.086, F15 0.057→0.033). X unchanged (1.438 vs 1.442); P_sol ≈ equal pooled (MAP
   keeps the clean structural zeros the samples leak: T35-sol / T25-IB ≈ 0.03–0.07 g/L).
   90 % bands: X ≈ 1.1–1.5 g/L wide (the ±50 % volume bracket surfaced as honest uncertainty),
   P_sol 0.2–0.4, P_IB 0.2–0.5 — and F12's 1.52 g/L band correctly flags the O₂-stressed corner
   as the least-trusted prediction.

**Recommendation:** report **ensemble mean ± 90 % band** for the product states (the parametric
analogue of the paper's 8-seed ensemble presentation), MAP for X and the structural-zero states.
Remaining honest improvement direction: the **soft-sensor layer** (within-run estimation —
per-run nuisance calibration + updating against that run's own offline points), the only place
where curves that track the observed bends are deliverable; cross-run product accuracy is at
ceiling (mechanisms, X-channel, and now sampling-of-parameters all exhausted).

### Proteostasis folding-competition module — v1 split / v2 rejected; the a_qS surface stands (2026-09-04)

User-driven "mechanism, not calibration" thread: replace the adopted model's two phenomenological
pieces (the logit partition θ₀/θ_T/a_qS and the α_T hinge) with ONE mechanism — the fold/misfold
race with a finite folding capacity, and proteolysis of the misfolded flux → CO₂ (the α channel).
`proteostasis_test.py` (v1) + `proteostasis_v2_test.py` (v2); variants `M0_proteostasis` /
`M0_proteostasis_v2` in PARAM_SPECS; 7 free params each (same count as the adopted model).
Artifacts: `results/proteostasis_test{,_v2}.png` + logs.

**v1 (rescue-capacity form: folding ≤ capacity C_chap, overflow → IB/CO₂) — split verdict.**

- SUPPORTED — the T-proteostasis half. T50_m ≈ 30.3 °C (bounded profile: min 30.3–31.5, rising
  to 0.91/1.68 at 30/35) and σ_IB = 0.66 (interior, parabolic: 1.77/0.58/0.61 at σ 0.2/0.6/0.8).
  σ implies a **34 % T35 proteolytic loss — independently reproducing α's η(35) = 0.64 via a
  different parameterization** (α suppresses the measurable flux; σ splits the misfolded flux;
  two routes, same number). Held-out fold A improved (F17 IB RMSE 0.119 vs canonical 0.181).
- FALSIFIED — the rescue-capacity half. C_chap → 3e-4 (lower bound), profile monotone (no rescue
  saturation at q_syn ~ 0.05 gP/gX/h); θ_m → 6.0 bound (wants a pure step). T30 qS-ordering
  compressed (model 0.086–0.127 vs obs 0.021–0.189); full-data cost 0.572 vs 0.525; fold-C F18
  IB RMSE 0.470 (canonical 0.122).

**v2 (capacity-limited folding RATE: k_f_eff = k_f·(1 − q_syn/C_cap)) — REJECTED decisively.**
Full-data cost 5.24 (10× baseline); multistart escapes to a degenerate no-IB corner (T50_m → 24
bound, θ_m → 6 bound, σ_IB → 0.07, φ_m ≈ 0 everywhere — T35 IBfrac predicted 0.000 vs obs 1.000);
all three structural profiles at/beyond bounds. The knife-edge rate form gives all-or-nothing IB
where the data has graded fractions.

**Why no single-parameter capacity form can work (the arithmetic):** the realized T30
synthesis-load range is only ~1.4× (q_syn ≈ 0.007–0.010 gP/gX/h, F17→F18) while the IB-fraction
range is ~9× (0.021→0.189). Reproducing that needs misfolding odds ∝ load^n with n ≈ 6 — which
is mathematically the a_qS logit surface re-expressed. **The T30 load-ordering is a response
surface at this data density; that is a finding, not a missed mechanism.** (v1 also confirms the
guardrail: load may only multiply T — T25 stays soluble at any load, IBfrac = 0.00 at T25/qs0.6.)

**Net verdict:** `M0_aload_qPTexp` stands unchanged. The a_qS·log(qS) term remains the
best-supported representation of the T30 ordering (a range, the identifiability weak link —
unchanged). The SUPPORTED mechanistic knowledge from this thread is the **T-proteostasis core**:
a race switch at T50_m ≈ 30.3 °C plus a σ ≈ 0.66 IB-vs-proteolysis split (α corroborated by two
routes; the α-CER validation logic extends unchanged to σ — the `simulate_run` gas path routes
(1−σ)·misfolded → CO₂). Not adopted as a replacement (equal params, worse fit, θ_m at bound);
retained as tested-and-split, like the hinge↔exp pair.


### H3 — measured-pO2 stress forcing — CLOSED, NULL (2026-09-05, overnight program)

`_O2` in models.py, variant `M0_aload_qPTexp_o2`. Motivation: all runs hold a 40 % pO2
setpoint and the excursions below it map onto the worst prediction errors (F12's anoxia
crash to 0.6 %). A Monod-in-pO2 factor on synthesis (1 param, data-forcing like feed/T —
distinct from E.2's rejected OTR *capacity* term). Result: cost 0.512 vs 0.525 — a
cosmetic −0.013 bought by pinning K_o2 at its 40 bound and qP_max at 0.1 (uniform halving
absorbed by doubling the amplitude); **F12's endpoint unmoved** (3.67 vs ~3.66 baseline,
obs 2.85). The crash window is too short / too little product forms during it. **E.2's
"O2 does not limit LDH production" survives re-test in forcing form.** F12's residual
stays corner-attributed (run-specific, volume bracket). Variant retained as tested-and-null.

### H2 — volume marginalization / bracket-level scan — CLOSED, negative with insight (2026-09-05, overnight program)

`vmarg_test.py` + `results/vmarg/`. (1) **Partition instability is NOT bracket noise**: fitting
the adopted model against predictions averaged over the bracket ends (runs duplicated at
N-floor and 2xN-floor) leaves the parameters essentially unchanged (a_qS still -1.01 / +4.0
bound / 0.72 across folds A/B/C) — the a_qS/theta_T cross-fold instability is in the
data/structure, not the volume nuisance. (2) **No global volume level wins**: scanning stored
fits at 1x/1.5x/2x over all 9 groups — aggregate X 1.451 / 1.487 / 1.586 (N-floor stays best);
the per-run effective volume is heterogeneous within the bracket (overpredicted corners improve
with larger V, underpredicted runs worsen). (3) The vmarg refit's apparent held-out X gains on
A/B/C do not replicate as a prediction protocol (bracket-end averaging with the standard params
is *worse*: X 1.451 -> 1.486) — they were fold-fit parameter wiggle. **Only a weighed base
bottle (MF2-R6) can improve the volume; no fitting protocol extracts more from the bracket.**

### H1 — KINETIC PARTITION of the unfolded pool (`M0_kinpart`) — smoke PROMISING, held-out gauntlet running (2026-09-05, overnight program)

User challenge ("there IS a mechanism for the sol/IB partition — yes or no") was correct: the
nine falsified forms were empirical redistributions of a fixed partition; the *molecular*
mechanism was untested. `M0_kinpart` = kinetic partitioning of the unfolded protein pool
(QSSA — folding/aggregation relax in minutes, samples are hours apart):

    P_syn = is_ind*qP_max*Monod(qS)*X          # total synthesis
    b     = k_f(T) + k_p(T) + D
    U*    = (sqrt(b^2 + 4*k_a(T)*P_syn) - b) / (2*k_a(T))      # unfolded pool, QSSA
    dP_sol/dt = k_f(T)*U* - D*P_sol            # first-order chaperone folding
    dP_IB/dt  = k_a(T)*U*^2 - D*P_IB           # SECOND-order aggregation
    (P_syn - k_f*U* - k_a*U*^2 = k_p(T)*U* -> CO2   # the CER-validated proteolysis channel)
    k_a(T) = k_a0*exp(delta_a*(T-30));  k_f(T) = k_f0*exp(-delta_f*max(T-30,0));  k_p(T) = k_p0*exp(gamma_p*max(T-30,0))

The load-dependence of the partition EMERGES from the rate-law orders (no a_qS/theta_0/theta_T);
the 25-soluble / 30-mixed / 35-IB switch from the rates' T-dependence; the IB deceleration from
U* collapsing as synthesis front-loads and declines. 9 molecular params (vs 7 adopted), prior:
gamma_p ~ LN(0.12, 0.7) (the CER-validated alpha's molecular counterpart).

**v1 (k_f const) FAILED in-sample** (cost 0.947; k_f/delta_a at bounds): second-order aggregation
leaked the load-dependence into T25 (data: fully soluble at every load) — the diagnosis that
folding capacity itself must collapse under heat shock. **v2 (+delta_f) cost 0.545 vs adopted
0.525** — baseline-level fit from a mechanism. IB trajectories: F13 0.12/0.25/0.39/**0.53** vs
obs 0.25/0.59/0.71/**0.54** (first model to reproduce the deceleration-plateau); F18 ends exactly
at obs 0.83; T35 endpoints improved (F07 2.17 vs ~2.5, F12 3.89 vs ~4.5). gamma_p landed at 0.121
~ prior center. Costs: k_f0/delta_f at bounds (wants slower/steeper folding collapse);
F17 low-qS boundary overprediction persists; early F18 peak under-shot.

**All 9 hold-out groups running** (`results/M0_kinpart/holdout_*`). The discriminating questions:
(a) does held-out P_IB/P_sol beat or match the adopted model (fold A: F17's overprediction —
does the intrinsic load-dependence fix the boundary?); (b) do the kinetic params transfer across
train sets (the property all nine empirical forms lacked)? Verdict to follow.

**H1 held-out VERDICT (9 groups, seeds 42+43 x 2 starts; `results/M0_kinpart/`, `results/logo_summary_M0_kinpart/`):**

| Held-out | adopted `M0_aload_qPTexp` | `M0_kinpart` |
| --- | --- | --- |
| **Isotherm A-C** P_sol / P_IB / protein / macro | 0.51 / 0.25 / 0.38 / 0.66 | **0.42 / 0.23 / 0.32 / 0.63** |
| **Full design (9)** P_sol / P_IB / protein / macro | **0.34 / 0.31 / 0.33 / 0.70** | 0.27 / 0.39 / 0.33 / 0.71 |

- **The partition tier (isotherm, where partitioning is informative) goes to the MECHANISM**:
  P_sol -18 %, P_IB -8 % with tighter spread, protein-states 0.38 -> 0.32, macro 0.66 -> 0.63.
  Fold A (F17 held out): **P_IB 0.281 -> 0.160 (-43 %)** — the boundary overprediction the
  empirical sigmoid could never fix is substantially fixed by the intrinsic second-order
  load-dependence, with NO fitted partition slope.
- **Trajectory curvature transfers held out**: F13 TEST (fold B) shows strong deceleration
  flattening toward the observed plateau (pred 0.47 at the last point vs obs 0.54) — the first
  model whose held-out IB *bends*; all nine empirical forms rise linearly.
- **Parameter transfer**: k_a0 0.045-0.080 across all 9 train sets (±25 %, no bounds) vs a_qS
  swinging -1.0 -> +4.0; gamma_p 0.120-0.121 (prior-anchored at the CER-validated alpha);
  k_f0/delta_f pinned at bounds IDENTICALLY in every fold (a consistent regime, not
  instability — but the bounds are doing work: identifiability caveat).
- **Where the mechanism loses**: the extreme T35 corners held out (F12 1.14 vs 0.90, F19 0.81
  vs 0.74) plus a small T25 IB leak (F10 0.31 vs 0.00 — second-order aggregation gives
  tiny-but-nonzero IB where the sigmoid's hard gate gives exactly 0). The F12-holdout train
  fit is pathological (qP_max at bound on 2 starts; refit with 4 starts x 3 seeds pending).
- **H5 model averaging (adopted + kinpart, prediction level)**: no domination — a compromise
  everywhere (P_sol 0.297, P_IB 0.349, both between the singles). The models are complementary:
  adopted owns the structural zeros and T35 corners; kinpart owns T30, P_sol, and the curvature.
  Not adopted.

**Standing verdict (2026-09-05 dawn):** the user's challenge is answered — yes, a mechanism for
the sol/IB partition exists (kinetic folding-vs-aggregation competition), it is now identified
and tested, and it BEATS the empirical partition on the isotherm tier while matching elsewhere,
with molecular parameters that transfer. It does not dominate the adopted model on the full
design (corners), so the adopted model stays the overall-predictive champion; `M0_kinpart` is
adopted as the mechanistic partition model alongside it. Refinements queued: F12-holdout refit
(multistart artifact?); bounds/identifiability check on k_f0/delta_f; ensemble bands for
kinpart.

**F12-holdout refit (2026-09-05 dawn): NOT a multistart artifact.** 3 seeds x 4 starts all
converge to the identical optimum (cost 0.3631, qP_max at its 0.1 bound, delta_a 0.40); held-out
F12 P_IB 1.143 unchanged. Trained without F12, the T35 production amplitude genuinely walks to
the bound — the corner residual is structural (F12's titer is the lowest T35 despite qs0.6; the
model cannot anticipate a run's corner character), inherited by the mechanism as by every other
form. **Overnight program complete: H1 verdict stands** — kinpart wins the isotherm/partition
tier with transferable molecular parameters and derived IB curvature; adopted stays the
overall-predictive champion; corners remain the shared data limit (weighed base bottle +
intermediate qS setpoints remain the levers). All variants retained in PARAM_SPECS; scripts:
`kinpart` (models.py `_KIN`), `M0_aload_qPTexp_o2` (`_O2`), `vmarg_test.py`, `xcond_test.py`,
`posterior_ensemble.py`, `logo_summary.py <model>`.

### The turnover question — CLOSED with the free-k_deg exhibit (2026-09-05, user-driven: "those are only straight lines")

User critique of `results/ib_shape_comparison.png`: all model P_IB curves are straight lines;
the observed peak-decline is not in any prediction. Correct — and now proven definitive:

1. **`M0_kinpart_deg`** (k_deg prior-anchored at 0.04): cost **0.481 — below the adopted model
   (0.525)**; all T30 endpoints land (F13 0.53/0.54, F18 0.83/0.83, F15 0.55/0.51); k_deg stays
   at the prior; curves monotone.
2. **`M0_kinpart_deg_free`** (k_deg0 prior-FREE, bounds 0-0.5): cost **0.4236 — best mechanistic
   fit in playground history**. The data pulls k_deg0 to **0.106/h** (2.6x the prior center) and
   drives k_p0 -> 0 (degradation acts on aggregates, not free chains — mechanistically sensible
   Lon/Clp logic) — **and the curves are STILL monotone** (F13 0.17/0.32/0.44/0.54; F18 ends
   0.85/0.83). Reproducing the peak-decline would fit F13's peak+final exactly, but the shared
   parameters must also fit F14/F15 — **and the triplicates disagree about the decline** (last
   interval: F13 -0.17 g/L, F14 ~flat, F15 +0.02). The least-squares optimum is the monotone
   compromise.

**Three independent proofs that the near-straight lines are the honest optimum**: (i) mass
balance — dilution (0.011-0.027/h) alone covers the observed mild declines; (ii) free-fit
behavior — unregularized removal still chooses monotone; (iii) replicate inconsistency — the
turnover is within triplicate scatter at this sampling density. **The P_IB peak-decline is not
identifiable AND not reproducible across replicates; straight lines are final for P_IB shape
until denser induction-window sampling exists.** The kinpart family's value is elsewhere and
real: molecular parameters, isotherm-tier accuracy, F17 boundary fix, endpoint precision
(v3-deg: 0.481 in-sample, gauntlet running). `results/ib_shape_comparison.png` now shows all
four curves incl. the free fit.

**`M0_kinpart_deg` held-out VERDICT (9 groups, seeds 42+43 x 2 starts; `results/M0_kinpart_deg/`,
`results/logo_summary_M0_kinpart_deg/`) — TIES the adopted model overall, WINS the isotherm tier:**

| Held-out | adopted | kinpart | **kinpart_deg** |
| --- | --- | --- | --- |
| Isotherm P_sol / P_IB / P_tot / protein / macro | 0.51 / 0.25 / 0.47 / 0.38 / 0.66 | 0.42 / 0.23 / 0.48 / 0.32 / 0.63 | **0.42 / 0.21 / 0.46 / 0.31 / 0.62** |
| Full design P_sol / P_IB / P_tot / protein / macro | 0.34 / **0.31** / 0.55 / 0.33 / **0.70** | **0.27** / 0.39 / 0.57 / 0.33 / 0.71 | **0.27** / 0.36 / **0.55** / **0.32** / **0.70** |

- **Full-design macro ties the adopted model (0.70 vs 0.70) while winning the isotherm tier
  (0.62 vs 0.66; protein states 0.31 vs 0.38)** — with 9 molecular parameters and a turnover
  channel (`k_deg0`, prior-anchored at 0.041 — the data cannot pull it higher, see the
  free-fit closure above). F17 boundary P_IB stays fixed (0.159 vs adopted 0.281); the corner
  losses vs plain kinpart shrink (F12 1.14->1.02, F19 0.81->0.75, F07 0.38->0.35); the T25 IB
  leak narrows (F10 0.31->0.30, F09 0.056). Adopted keeps the single best P_IB corner cells
  (F12 0.90 structural-zero machinery), kinpart_deg takes everything else.
- **Parameter transfer is clean**: k_a0 0.055-0.084, k_p0 0.042-0.097 (interior, no bounds),
  gamma_p 0.1221-0.1228 and k_deg0 0.0410-0.0411 (prior-anchored, dead stable) across all 9
  train sets; k_f0/delta_f remain identically bound-pinned (the known caveat).

**FINAL STANDING (2026-09-05):** `M0_kinpart_deg` is adopted as the primary mechanistic model —
equal overall accuracy to the empirical champion, better on the partition tier, better
endpoints, interpretable and transferable rates (folding collapse delta_f, aggregation
delta_a/k_a0, proteolysis gamma_p/k_p0/k_deg0), P_IB trajectory curvature derived from the rate
physics. The adopted `M0_aload_qPTexp` stays as the empirical reference (best single corner
cells). The P_IB turnover itself remains unidentifiable and replicate-inconsistent (free-fit
closure above) — that result stands.

### F10 glucose-pulse biomass check — overshoot CONFIRMED and pool-quantified; rhs shadowing bug FIXED (2026-09-26, user-driven)

User asked to settle the F10 pulse question directly (arising from the induction-lag /
stage-refinement discussion): does modeled X overshoot observed X inside F10's
glucose-accumulation window (22.5–28 h) and converge afterwards? F10 is the only FCCD run with
residual glucose: pool = 11.05 g/L × V ≈ **14.1 g at 24.0 h** (≈ a third of the ~19 g glucose/h
fed since induction; F10 fed ~48 mL/h vs F09's ~16), 8.8 g at 26.0 h, cleared by 27.9 h.
Test: `f10_pulse_biomass_test.py` — each run simulated with its own LOGO hold-out fit (trained
without that run), both standing models, plus a **pool-corrected counterfactual**: the feed
forcing rescaled by `1 − (dPool/dt)/(feed-glucose-rate)` (uptake = feed while the pool grows,
feed + pool-burn while it clears; total glucose conserved, only its timing corrected).

Result (pred − obs X, adopted model; `M0_kinpart_deg` identical to <0.02 g/L — both share
μ = Y_xs·qS and Y_xs agrees to 3 decimals):

| t [h] | phase | X obs | X pred | overpred | pool-corrected pred |
| --- | --- | ---: | ---: | ---: | ---: |
| 20.88 | pre-window | 22.37 | 23.23 | +0.86 | (+0.86) |
| 23.98 | pool peak 14.1 g | 30.07 | 33.96 | **+3.89** | 31.52 (+1.45) |
| 25.98 | pool 8.8 g | 38.10 | 41.45 | **+3.35** | 40.18 (+2.08) |
| 27.93 | pool cleared | 47.10 | 47.90 | +0.80 | (+1.07) |
| 29.80 | final | 52.67 | 53.45 | +0.78 | (+1.04) |

**Verdict: the overshoot branch — a +3–4 g/L hump rides on the run's +0.8 g/L baseline offset
(present before the window and after), exactly where the pool sits, and is gone once the pool
clears.** The counterfactual attributes **~2.4 g/L of the peak to the measured pool**
(≈ pool × Y_xs 0.365 / V, partially self-limiting through qS ∝ 1/X); the remainder is the
run's pre-existing baseline offset. Controls: F09 (no pool) has no hump (−0.6/−0.8/−0.2,
underpredicts throughout); F16 shows a pool-free +4.4 g/L transient that starts PRE-induction
and REVERSES sign by the end — an early-window X overshoot per se is not pulse-specific
(volume-bracket/run-specific transients exist), and the pulse cannot explain F16 — confirmed.

**Implications.** (a) The uptake transient is real, transient-only, and visible in X: F10's qS
forcing was inflated ~25 % inside the window (+4.5 g/h pool growth vs 19 g/h feed), which also
feeds the Monod production term — so F10's early product residual carries a substrate-transient
confounder that F16's (pool-free) does not; the production-lag direction stays attributable for
F16 but is confounded in F10's first ~4 h. (b) No model change: the pool exists in one corner
run for ~4 h; an explicit pool state would be unidentifiable from one occurrence (identifiability
discipline), but the window should be kept in mind for F10-only features.

**Bug fixed while writing this test (`models.py` rhs, 2026-09-26):** the kinpart branch assigned
a local `k_deg` (the T-scaled aggregate-proteolysis rate), shadowing the closure-level `k_deg`
extracted at the top of `make_rhs` — every NON-kinpart model then hit `UnboundLocalError` at the
shared `k_ref/k_deg` tail, which `simulate_run`'s blanket `except` silently converted into
all-NaN predictions. Fixed by renaming the local to `k_deg_T` (behavior-identical for kinpart).
Verified against stored artifacts: recomputed LOGO_F10 adopted-model X RMSE 2.175 / bias −1.615
== `results/M0_aload_qPTexp/holdout_F10/test_metrics.md` exactly. All stored results predate the
bug. Lesson: `simulate_run`'s silent exception path hides breakage — it surfaced only because a
downstream script printed NaNs; treat all-NaN predictions from it as a crash, not a result.
Artifacts: `results/f10_pulse/{f10_pulse_biomass.png, residuals.csv, SUMMARY.md}`.
