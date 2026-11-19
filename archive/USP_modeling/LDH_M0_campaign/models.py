"""Mechanistic ODE models for the LDH playground.

States (all g/L): X (biomass), P_sol (soluble product), P_IB (inclusion body).
Forcing inputs (online, interpolated): feed_rate [L/h], V [L], T [C], is_induced.

Core equations (NN-free):

    qS(t)   = F_rate*S_IN / (X*V)                      # feed-supply proxy, capped at QS_CAP
    D(t)    = F_rate / V
    mu      = Y_xs * qS                                # M0/M1a
    mu      = Y_xs * (qS - m_s)                        # M1b (Pirt maintenance -> qS-dependent yield)
    phi_IB  = sigmoid(theta_0 + theta_T*(T-30) [+ a_L*log(qP/c_fold)])
    qP_tot  = is_induced * qP_max * qS/(qS + K_S)      # Monod-in-feed-proxy total production
    dX/dt     = mu*X - D*X
    dP_sol/dt = (1-phi_IB)*qP_tot*X - D*P_sol
    dP_IB/dt  = phi_IB*qP_tot*X - D*P_IB

Offgas (CER, OUR) is predicted from the carbon + electron balance (see stoich.py)
using the model's instantaneous rates. This adds information, not parameters.

Model variants (each is a strict superset of the previous; every added
parameter must pass a profile-likelihood + cross-fold identifiability check):

  M0 : [Y_xs, qP_max, K_S, theta_0, theta_T]   obs = X, P_sol, P_IB (sparse)
  M1a: same params as M0                         obs = M0 + CER, OUR (online)  [0 new params]
  M1b: M1a + m_s                                 [1 new param]
  aload variants (+a_L on the partition) are tested only as an identifiability probe.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from stoich import predict_gas

S_IN = 400.0          # feed glucose concentration [g/L]
T_REF = 30.0          # reference temperature [C] (FCCD center)
QS_CAP = 2.0          # cap on the qS feed proxy [g/g/h]
QS_REF = 0.4          # centering reference for the log-qS partition term (FCCD center)
EPS = 1e-8

# Sparse (offline) states always observed; gas channels observed only in M1a+.
SPARSE_STATES = ("X", "P_sol", "P_IB")
GAS_STATES = ("CER", "OUR")

# ---------------------------------------------------------------------------
# Parameter specifications: name -> (init, lo, hi)
# ---------------------------------------------------------------------------
_BASE = {
    "Y_xs":   (0.45, 0.20, 0.80),
    "qP_max": (0.010, 1e-4, 0.10),
    "K_S":    (0.10, 1e-3, 2.0),
    "theta_0": (-1.91, -6.0, 6.0),
    "theta_T": (2.01, -1.0, 8.0),
}
_M1B = {"m_s": (0.02, 0.0, 0.50)}
_ALOAD = {"a_qS": (0.0, -4.0, 4.0)}
_DECAY = {"tau_burden": (4.0, 0.5, 20.0)}  # induction production decay [h]
_HALDANE = {"Ki": (1.0, 0.05, 10.0)}  # substrate-inhibition constant [g/g/h]
# Offgas-derived metabolic burden state b (Task A): a low-pass-filtered production
# rate whose steady state is qP*X*tau_b (so a_b*tau_b are coupled at steady state;
# only the transient/lag separates them -> profile-likelihood probes the sloppiness).
_BURDEN = {"a_b": (0.0, -4.0, 4.0), "tau_b": (4.0, 0.5, 20.0)}
# Temperature suppression of total production: observed total specific product drops
# ~40% at 35 C vs 25/30 C (full-data fit overpredicts every T35 run by +1.2..+2.0 g/L).
# The drop is asymmetric (T25 ~= T30, only T35 low) -> hinge at T_REF, flat below.
_QPT = {"k_T": (0.10, 0.0, 0.20)}   # qP factor = max(1 - k_T*max(T-30,0), 0.05) [1/C]
# Mechanistic (first-order proteostasis) T-decline of total production: the measured
# product = synthesis x folding efficiency eta(T); eta declines exponentially above the
# expression optimum (sigma-32 heat-shock burden -> misfolding/aggregation/degradation).
# qP *= exp(-alpha_T*max(T-30,0)). Equivalent fit to the hinge over T25/30/35 but with a
# first-order (rate-based) reading: alpha_T = fractional loss of measurable product /C.
_QPT_EXP = {"alpha_T": (0.10, 0.0, 0.50)}   # [1/C]
# T-dependent biomass yield: Y_xs(T) = Y_xs*exp(-beta_T*max(T-30,0)). Same heat stress as alpha_T
# (product folding loss) but on biomass -- at T35 more substrate carbon goes to stress respiration
# (CO2/proteostasis) and less to biomass, so the constant-yield model over-predicts T35 biomass.
# Fixes the systematic T35 biomass over-prediction (F07 +4.0, F19/F12 +1.3 g/L).
_YXT = {"beta_T": (0.10, 0.0, 0.50)}   # biomass yield loss above 30 C [1/C]
# Soluble folding-transit delay (dynamic IB/sol partition): translated protein either
# aggregates immediately -> IB (kinetic) or enters a folding pool N_fold and emerges as
# soluble after maturation time tau_fold (chaperone-assisted). Reproduces the observed
# early-IB / late-soluble dynamics (IB fraction peaks ~2-4h post-induction then declines).
# States become (X, P_sol, P_IB, N_fold); N_fold latent. Preserves total production mass.
_FOLDLAG = {"tau_fold": (3.0, 0.5, 12.0)}   # soluble folding maturation time [h]
# IB -> soluble refolding / resolubilization (mass-conserving transfer; chaperone-mediated
# resolubilization of aggregates). Observed: in the T30 mixed regime P_IB plateaus/declines late
# while P_sol keeps rising -- a transfer IB->sol would both flatten P_IB and boost P_sol beyond
# production-minus-dilution. First-order in P_IB. [1/h]
_REFOLD = {"k_ref": (0.05, 0.0, 0.50)}
# IB degradation / proteolysis (mass LOST -> CO2, not recovered as soluble). Same late-P_IB decline
# signature but WITHOUT a soluble boost. First-order in P_IB. [1/h]
_IBDEG = {"k_deg": (0.02, 0.0, 0.50)}
# Cooperative (Moser/Hill) IB formation -- aggregation is nucleation-dependent (cooperative), so IB
# formation is maxed at high expression load (qP) and drops steeply as qP falls (open-loop qS decay)
# -> IB peaks then declines (the observed front-loaded IB the smooth linear-phi split misses). Soluble
# = total - IB by subtraction (mass-conserving). Replaces the a_qS log-qS load term (qS-load now via
# qP). T-dependence stays in theta (IB capacity: T25 soluble, T35 IB).
_HILLIB = {"qP_IBmax": (0.020, 1e-4, 0.080),  # max specific IB formation rate [g/g/h]
           "K_IB": (0.020, 1e-3, 0.080),       # Hill half-saturation in qP [g/g/h]
           "N_hill": (2.0, 1.0, 4.0)}          # cooperativity (1=Michaelis; >1 cooperative)
# IB-specific transient decline: IB formation is an early-stress response that decays over induction
# (qP_IB flux *= exp(-t_ind/tau_IB)), while soluble is sustained. Observed (LDH + Xiso): qP_IB -> ~0
# late (IB shuts off) while qP_sol stays up -- the decline is IB-specific, not total-qP. Soluble =
# total - decayed-IB (mass-conserving; sol rises as IB shuts off). Distinct from total induction-decay.
_IBTRANS = {"tau_IB": (3.0, 0.5, 20.0)}   # IB formation decay time over induction [h]

# Hierarchical-decline variant (2026-09-04, user-driven revisit of the trajectory thread).
# Front-loaded production: total formation decays over induction (tau_P -- both products;
# matches the decelerating P_sol and qp_decline's f_qP ~ 0.54 beyond qS), and IB formation
# shuts off FASTER (tau_IBd < tau_P: aggregation machinery fatigues / chaperone adaptation),
# so P_IB can peak and then decline via dilution -- the F13/F18 pattern the straight
# baseline cannot produce. Named tau_IBd (not tau_IB) to avoid the use_ibtransient flag
# collision. The two timescales are weakly informed by 4-5 points/run, so they carry
# declared log-normal priors in fit.py (MAP penalty) instead of free-fit bound collapse.
_DECL = {"tau_P": (12.0, 2.0, 48.0), "tau_IBd": (2.0, 0.3, 12.0)}   # [h], (init, lo, hi)

# Kinetic partitioning of the unfolded pool (2026-09-04, user challenge: "there IS a
# mechanism for the sol/IB partition"). The molecular mechanism the sigmoid partition
# was shadowing: nascent chains FOLD (chaperone-assisted, first-order k_f) or AGGREGATE
# (bimolecular k_a*U -- the load-dependence of the partition emerges from the rate-law
# orders, replacing the a_qS/theta_0/theta_T patch), surplus is PROTEOLYZED -> CO2
# (k_p -- the CER-validated alpha channel). Quasi-steady-state pool (folding/aggregation
# relax in minutes; samples are hours apart), so no latent state:
#
#   P_syn = is_ind*qP_max*Monod(qS)*X          [g/L/h]  total synthesis
#   b     = k_f + k_p(T) + D
#   U*    = (sqrt(b^2 + 4*k_a(T)*P_syn) - b) / (2*k_a(T))      [g/L]
#   dP_sol/dt = k_f*U* - D*P_sol
#   dP_IB/dt  = k_a(T)*U*^2 - D*P_IB
#   (P_syn - k_f*U* - k_a(T)*U*^2 = k_p(T)*U* -> CO2)
#
#   k_a(T) = k_a0*exp(delta_a*(T-30))     aggregation accelerates
#   k_f(T) = k_f0*exp(-delta_f*max(T-30,0))  folding collapses above the optimum
#       (sigma32: chaperones saturated/diverted -- v1 (no k_f(T)) failed in-sample,
#        cost 0.95, k_f/delta_a at bounds: second-order aggregation leaked the
#        load-dependence into T25, where the data is fully soluble at every load;
#        the folding collapse gates the load-dependence to T>=30)
#   k_p(T) = k_p0*exp(gamma_p*max(T-30,0))  proteostasis/proteolysis heat-shock (sigma32)
#
# 9 params (vs 7 adopted), all molecular. Prior: gamma_p ~ the CER-validated alpha.
_KIN = {
    "k_f0":    (20.0, 0.5, 200.0),    # chaperone-assisted folding at <=30 C [1/h]
    "delta_f": (0.5, 0.0, 1.0),       # folding collapse above 30 C [1/C]
    "k_a0":    (50.0, 1e-2, 1e4),     # aggregation at 30 C [L/g/h] (second-order)
    "delta_a": (0.20, 0.0, 0.60),     # aggregation T-steepness [1/C]
    "k_p0":    (0.5, 0.0, 20.0),      # proteolysis at <=30 C [1/h] (-> CO2)
    "gamma_p": (0.09, 0.0, 0.50),     # proteolysis heat-shock amplification [1/C]
}
_KIN_BASE = {"Y_xs": (0.45, 0.20, 0.80), "qP_max": (0.010, 1e-4, 0.10), "K_S": (0.10, 1e-3, 2.0)}

# v3: aggregate proteolysis — the sigma32 response degrades not only unfolded chains
# (k_p) but also aggregates (Lon/Clp on P_IB): dP_IB/dt = k_a*U*^2 - k_deg(T)*P_IB - D*P_IB.
# Mass balance: dilution (0.011-0.027 /h) alone explains the observed mild T30 declines
# (-0.014..-0.033 g/L/h, single-interval), so k_deg is PERMITTED, not required — prior
# from the deficit numbers. This is the turnover lever: production-side collapse alone
# can only asymptote to dilution, never cross it.
_KINDEG = {"k_deg0": (0.05, 0.0, 0.30)}   # aggregate proteolysis at <=30 C [1/h] (shares gamma_p)
# Free-k_deg variant (no prior): does the DATA, unregularized, pull turnover out of
# one sample interval? k_deg0 wider bounds, fit.PRIORS_SKIP exempts it from the prior.
_KINDEG_FREE = {"k_deg0": (0.10, 0.0, 0.50)}

# Measured-pO2 stress forcing (2026-09-04 overnight, H3): all runs hold a 40% pO2
# setpoint; the EXCURSIONS below it are the stress signal and map onto the worst
# prediction errors (F12's anoxia crash to 0.6%; cf. o2_limit_test, which ruled out a
# steady OTR *capacity* term -- this is different: the measured excursion itself
# throttles synthesis, Monod in dissolved O2, 1 param). Data-forcing like feed/T/gate,
# not a fitted mechanism.
_O2 = {"K_o2": (5.0, 0.5, 40.0)}   # Monod half-saturation in pO2 [%]

# Proteostasis folding-competition module (2026-09-04, "mechanism not calibration" thread):
# REPLACES the logit partition (theta_0/theta_T/a_qS) AND the alpha_T hinge with ONE mechanism.
# Synthesis makes unfolded chain at q_syn; each chain folds or misfolds (a race: phi_m = k_m/(k_f+k_m),
# the Arrhenius rate ratio -> logistic in T); the chaperone system rescues misfolded chains up to a
# finite folding-assistance capacity C_chap [gP/gX/h]; the unrescued overflow splits into recoverable
# IB (sigma_IB) vs proteolysis -> CO2 (1-sigma_IB -- the alpha_T channel, CER-validated biology).
# Consequences PRODUCED, not fitted: the T regime (phi_m(25)~0 -> soluble, phi_m(35)~1 -> IB), the
# T30 qS-ordering (capacity overflow grows with synthesis load -- replaces a_qS*log(qS)), and the T35
# total-product loss with its CO2 signature (replaces alpha_T). Same parameter count as the adopted
# M0_aload_qPTexp (7) but every parameter has units and a physical job. Sub-variant with theta_m
# FIXED (fit_model(fixed={"theta_m": 1.5}); 3 T strata cannot pin a slope) drops to 6 free params.
_PROTEO = {
    "Y_xs":      (0.45, 0.20, 0.80),
    "q_syn_max": (0.070, 1e-4, 0.20),   # max total synthesis flux [gP/gX/h] (pre-folding)
    "K_S":       (0.10, 1e-3, 2.0),     # Monod half-saturation of synthesis in qS [g/g/h]
    "C_chap":    (0.004, 1e-4, 0.05),   # chaperone folding-assistance capacity [gP/gX/h]
    "T50_m":     (31.5, 24.0, 45.0),    # misfolding midpoint (k_m = k_f) [degC]
    "theta_m":   (1.5, 0.3, 6.0),       # misfolding-ratio slope ln(k_m/k_f) [1/degC]
    "sigma_IB":  (0.60, 0.0, 1.0),      # unrescued-misfolded split: IB (recoverable) vs proteolysis->CO2
}

# v2 (load-coupled race, 2026-09-04): the SAME mechanism but the capacity acts on the FOLDING
# RATE, not as a rescue flux: k_f_eff = k_f*(1 - q_syn/C_cap), so the race is lost where synthesis
# load approaches the proteome folding capacity. v1's verdict drove this: rescue-capacity C_chap
# -> 0 (monotone profile; no rescue saturation at these rates) and the T30 qS-ordering stayed
# compressed (F18 0.127 vs obs 0.189) -- the load signal must enter through the race itself.
# Guardrail preserved: the e^(theta_m*(T50_m-T)) ratio keeps T25 soluble at ANY load (data:
# T25 qs0.6 IBfrac = 0.00), i.e. load only matters where the race is already contested.
_PROTEO2 = {
    "Y_xs":      (0.45, 0.20, 0.80),
    "q_syn_max": (0.060, 1e-4, 0.20),
    "K_S":       (0.10, 1e-3, 2.0),
    "C_cap":     (0.019, 0.004, 0.08),  # proteome folding capacity [gP/gX/h] (k_f -> 0 at q_syn -> C_cap)
    "T50_m":     (31.5, 24.0, 45.0),
    "theta_m":   (3.0, 0.3, 6.0),
    "sigma_IB":  (0.60, 0.0, 1.0),
}

PARAM_SPECS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "M0": dict(_BASE),
    "M1a": dict(_BASE),
    "M1b": {**_BASE, **_M1B},
    "M0_aload": {**_BASE, **_ALOAD},
    "M1a_aload": {**_BASE, **_ALOAD},
    "M1b_aload": {**_BASE, **_M1B, **_ALOAD},
    # Production-decay variants: qP burst at induction then exp(-t/tau) decline
    # (observed qP is front-loaded, peaking ~0.03 at induction, decaying to ~0.006 by 8h).
    "M0_decay": {**_BASE, **_DECAY},
    "M0_aload_decay": {**_BASE, **_ALOAD, **_DECAY},
    "M1a_aload_decay": {**_BASE, **_ALOAD, **_DECAY},
    # Substrate-inhibition (Haldane) production: qP peaks at moderate qS then declines
    # (specific production observed to peak at qS~0.4, lower at qS=0.6 -> targets the
    # center total-product underprediction that monotonic Monod cannot capture).
    "M0_aload_haldane": {**_BASE, **_ALOAD, **_HALDANE},
    # Offgas-burden-driven partition (Task A): latent burden state b (filtered
    # production) enters the partition alongside the qS-load term. Tests whether the
    # post-induction burden respiration is an IB driver identifiable beyond a_qS.
    # b is latent (not observed); sparse obs only, no gas.
    "M0_aload_burden": {**_BASE, **_ALOAD, **_BURDEN},
    # Temperature-suppressed total production: qP *= max(1 - k_T*max(T-30,0), 0.05).
    # Targets the dominant full-data mis-fit (every T35 run overpredicted +1.2..+2.0).
    "M0_aload_qPT": {**_BASE, **_ALOAD, **_QPT},
    # qPT + substrate-inhibition (Haldane): with qP_max freed by qPT, test whether the
    # residual qS-peak (all qs0.6 runs overpredict after qPT) is now identifiable.
    "M0_aload_qPT_haldane": {**_BASE, **_ALOAD, **_QPT, **_HALDANE},
    # Mechanistic qPT: first-order (exponential) folding-efficiency decline above 30 C.
    "M0_aload_qPTexp": {**_BASE, **_ALOAD, **_QPT_EXP},
    # Front-loaded production with IB-specific shutoff (see _DECL) -- the one untested
    # answer to the "P_curves too straight / P_IB stagnates" residual; timescales are
    # prior-regularized (fit.py PRIOR_LOGNORMAL) rather than free-fit.
    "M0_aload_qPTexp_decl": {**_BASE, **_ALOAD, **_QPT_EXP, **_DECL},
    # Kinetic partitioning of the unfolded pool (see _KIN): the molecular sol/IB
    # mechanism -- folding vs (second-order) aggregation vs proteolysis. Replaces the
    # empirical sigmoid partition (theta_0/theta_T/a_qS) and alpha_T.
    "M0_kinpart": {**_KIN_BASE, **_KIN},
    # v3: + aggregate proteolysis (see _KINDEG) — the turnover lever for P_IB.
    "M0_kinpart_deg": {**_KIN_BASE, **_KIN, **_KINDEG},
    # v3-free: same, but k_deg0 prior-free (see _KINDEG_FREE) — the unregularized
    # turnover test.
    "M0_kinpart_deg_free": {**_KIN_BASE, **_KIN, **_KINDEG_FREE},
    # Adopted model + measured-pO2 stress forcing (see _O2): targets the F12/F07/F10
    # excursions below the 40% setpoint.
    "M0_aload_qPTexp_o2": {**_BASE, **_ALOAD, **_QPT_EXP, **_O2},
    # Proteostasis folding-competition module (see _PROTEO): ONE mechanism replaces the
    # logit partition (theta_0/theta_T/a_qS) + the alpha_T hinge. theta_m can be pinned via
    # fit_model(fixed={"theta_m": ...}) for the 6-free-param sub-variant.
    "M0_proteostasis": dict(_PROTEO),
    "M0_proteostasis_v2": dict(_PROTEO2),
    # Adopted baseline + T-dependent biomass yield (2026-07-28): fixes the systematic T35 biomass
    # over-prediction (apparent Y_xs drops 0.28->0.21 at T35; same heat-stress biology as alpha_T).
    "M0_aload_qPTexp_yxt": {**_BASE, **_ALOAD, **_QPT_EXP, **_YXT},
    # qPTexp + soluble folding-transit delay (dynamic partition; fixes the early-IB/
    # late-soluble dynamics the constant-phi model cannot represent).
    "M0_aload_qPTexp_foldlag": {**_BASE, **_ALOAD, **_QPT_EXP, **_FOLDLAG},
    # qPTexp + IB->soluble refolding (mass-conserving) / IB degradation (->CO2): test whether the
    # observed late P_IB plateau/decline (T30 mixed regime, model misses it) is a refolding boost to
    # soluble (k_ref) or a loss (k_deg), vs the baseline dilution/partition.
    "M0_aload_qPTexp_refold": {**_BASE, **_ALOAD, **_QPT_EXP, **_REFOLD},
    "M0_aload_qPTexp_ibdeg": {**_BASE, **_ALOAD, **_QPT_EXP, **_IBDEG},
    "M0_aload_qPTexp_refoldibdeg": {**_BASE, **_ALOAD, **_QPT_EXP, **_REFOLD, **_IBDEG},
    # Cooperative (Moser/Hill) IB formation (Muller 2023 form): IB rate = qP_IBmax*Hill_N(qP), soluble
    # = total - IB. Tests whether the observed front-loaded IB (peak then decline) is cooperative
    # aggregation rather than the smooth linear-phi split. Drops a_qS (qS-load now via qP/Hill).
    "M0_qPTexp_hillib": {**_BASE, **_QPT_EXP, **_HILLIB},
    # IB-specific transient decline (IB formation front-loaded then decays over induction, soluble
    # sustained, mass-conserving). Targets the observed IB-specific shutdown (qP_IB -> 0 late, qP_sol
    # up) that Hill-IB (tied to total qP) could not capture. +1 param (tau_IB).
    "M0_aload_qPTexp_ibtransient": {**_BASE, **_ALOAD, **_QPT_EXP, **_IBTRANS},
    # Independent S_met-driven rates (NN m13 oracle port): drop the partition; qP_sol and qP_IB are
    # independent, both Monod(qS)*eta(T) for formation+T-gate, but qP_IB carries an EXTRA cumulative-load
    # decline exp(-S_met/K_IB) (the burden -> IB transient) while qP_sol does not (-> sustained/floor).
    # S_met = integral(qS*X) dt = S_IN*integral(D) dt is a 4th state (biomass-free). Captures the
    # NN-validated trajectory (sol monotonic rise, IB peak-then-decline) the partition model cannot.
    "M0_smet_independent": {
        "Y_xs": (0.45, 0.20, 0.80),
        "qP_sol_max": (0.030, 1e-4, 0.10),    # soluble formation amplitude [g/g/h]
        "qP_IB_max": (0.030, 1e-4, 0.10),     # IB formation amplitude [g/g/h] (independent of sol)
        "K_S": (0.10, 1e-3, 2.0),             # Monod half-sat in qS [g/g/h]
        "theta_0": (-1.91, -6.0, 6.0),        # T-gate logit intercept (sol<->IB switch)
        "theta_T": (2.01, -1.0, 8.0),         # T-gate logit slope [1/C]
        "alpha_T": (0.10, 0.0, 0.50),         # total-production folding loss above 30 C [1/C]
        "K_IB": (100.0, 5.0, 1000.0),         # IB cumulative-load decline constant [g substrate/L broth]
    },
    # Independent S_met-driven rates + net IB degradation (option 1+3, 2026-07-27). Same as
    # M0_smet_independent but the IB flux can go NET-NEGATIVE: first-order IB proteolysis
    # (k_deg*P_IB -> CO2) on top of the S_met formation decline. This is the NN m13 recipe
    # (tanh -> negative qP_IB, learned ~ -0.001 late): the gentle exp(-S_met/K_IB) alone only
    # REDUCES formation (>=0), so with K_IB~118 and qP_IB_max pinned it leaves qP_IB just above
    # the dilution threshold late -> P_IB plateaus instead of declining. The degradation channel
    # lets net IB flux drop below dilution -> real decline (F13 obs dIB -0.17).
    # NB ibdeg ALONE (on the partition base) was weak (psplit_test, dcost ~-0.02); the point here
    # is the COMBINATION with the S_met formation decline -- the NN uses both.
    # Option 3: qP_IB_max upper bound raised 0.10 -> 0.30 so formation can reach the observed IB
    # peak interiorly (it pinned at 0.10 wanting MORE -- the smet model under-predicts IB ~0.35 vs
    # obs ~0.54); the k_deg profile is the identifiability arbiter for the formation-vs-degradation
    # trade-off.
    "M0_smet_independent_ibdeg": {
        "Y_xs": (0.45, 0.20, 0.80),
        "qP_sol_max": (0.030, 1e-4, 0.10),    # soluble formation amplitude [g/g/h]
        "qP_IB_max": (0.030, 1e-4, 0.30),     # IB formation amplitude [g/g/h] (raised 0.10->0.30, opt 3)
        "K_S": (0.10, 1e-3, 2.0),             # Monod half-sat in qS [g/g/h]
        "theta_0": (-1.91, -6.0, 6.0),        # T-gate logit intercept (sol<->IB switch)
        "theta_T": (2.01, -1.0, 8.0),         # T-gate logit slope [1/C]
        "alpha_T": (0.10, 0.0, 0.50),         # total-production folding loss above 30 C [1/C]
        "K_IB": (100.0, 5.0, 1000.0),         # IB cumulative-load decline constant [g substrate/L broth]
        "k_deg": (0.02, 0.0, 0.50),           # first-order IB proteolysis [1/h] (-> CO2, not conserved)
    },
    # Decoupled independent sol/IB (2026-07-28, per the data reframe: sol and IB have DIFFERENT
    # dynamics -- sol a steady accumulator present T<=30, IB a T30-transient/T35-sustained pool).
    # Drops the partition: qP_sol and qP_IB are independent, each amplitude x Monod(qS) x a
    # complementary T-gate (same theta) x eta(T). sol has NO decline channel; IB declines iff its
    # formation drops below dilution. IB carries a T-gated clock z=exp(-t_ind*gSol/tau_IB) (option A):
    # aggregation fatigues on a clock at T30 (gSol~1) -> shutoff -> decline; z=1 at T35 (gSol~0) ->
    # sustained (matches F07/F19/F12 rising). 3 states (no latent). Re-tests ibtransient where
    # qP_IB_max is free to reach the peak (the partition-base version failed by under-predicting IB).
    "M_decoupled_ibtransient": {
        "Y_xs": (0.45, 0.20, 0.80),
        "qP_sol_max": (0.030, 1e-4, 0.10),    # soluble formation amplitude [g/g/h] (sustained)
        "qP_IB_max": (0.030, 1e-4, 0.10),     # IB formation amplitude [g/g/h] (independent of sol)
        "K_S": (0.10, 1e-3, 2.0),             # Monod half-sat in qS [g/g/h] (shared sol/IB)
        "theta_0": (-1.91, -6.0, 6.0),        # T-gate logit intercept (sol<->IB switch)
        "theta_T": (2.01, -1.0, 8.0),         # T-gate logit slope [1/C]
        "alpha_T": (0.10, 0.0, 0.50),         # total-production folding loss above 30 C [1/C]
        "tau_IB": (3.0, 0.5, 20.0),           # IB clock-transient time [h] (T-gated to T30)
    },
    # Decoupled v2: INDEPENDENT (non-complementary) T-gates (2026-07-28). v1's complementary gates
    # (g_sol+g_ib=1) starved IB at T30 (g_ib->0.05) -> qP_IB_max pinned -> clock rejected (tau->20).
    # Independent gates let both products be fully on at T30 (data has both), so qP_IB_max is interior
    # and the T30-gated clock can engage to decline IB. Separate gate params for sol (on T<=30) and IB
    # (on T>=30); clock z=exp(-t_ind*g_sol/tau) runs at T30 (sol made -> cell adapts), off at T35.
    # 10 params, 3 states.
    "M_decoupled_v2": {
        "Y_xs": (0.45, 0.20, 0.80),
        "qP_sol_max": (0.030, 1e-4, 0.10),    # soluble formation amplitude [g/g/h] (sustained)
        "qP_IB_max": (0.030, 1e-4, 0.30),     # IB formation amplitude [g/g/h] (raised 0.10->0.30: pins at 0.10
                                              #   in v2 -> peak under-predicted; fast clock needs high formation)
        "K_S": (0.10, 1e-3, 2.0),             # Monod half-sat in qS [g/g/h] (shared sol/IB)
        "theta_sol_0": (2.0, -4.0, 6.0),      # sol T-gate intercept (sol on at T<=30)
        "theta_sol_T": (2.0, 0.5, 8.0),       # sol T-gate slope (->off above 30) [1/C]
        "theta_ib_0": (2.0, -4.0, 6.0),       # IB T-gate intercept (IB on at T>=30)
        "theta_ib_T": (2.0, 0.5, 8.0),        # IB T-gate slope (->on above 30) [1/C]
        "alpha_T": (0.10, 0.0, 0.50),         # total-production folding loss above 30 C [1/C]
        "tau_IB": (3.0, 0.5, 20.0),           # IB clock-transient time [h] (T-gated to T30 via g_sol)
    },
    # Decoupled v3: delayed shutoff (2026-07-28). v2's immediate-decay clock exp(-t*g_sol/tau) suppresses
    # formation from t=0, so a fast-enough clock to decline also starves the peak -> a peak-vs-decline
    # trade-off (multistart lands in a decline-basin OR a peak-basin, never both). v3 lets formation run
    # at full strength until t_delay, THEN shuts off: z=1 for t<t_delay, else exp(-(t-t_delay)*g_sol/tau).
    # Decouples peak (full formation x the delay window) from decline (tau after the delay) -> both.
    # Biologically: stress response runs at full strength, then chaperone-mediated adaptation shuts off
    # aggregation after a delay. Same independent gates as v2; +1 param (t_delay). 11 params, 3 states.
    "M_decoupled_v3": {
        "Y_xs": (0.45, 0.20, 0.80),
        "qP_sol_max": (0.030, 1e-4, 0.10),
        "qP_IB_max": (0.030, 1e-4, 0.30),
        "K_S": (0.10, 1e-3, 2.0),
        "theta_sol_0": (2.0, -4.0, 6.0),
        "theta_sol_T": (2.0, 0.5, 8.0),
        "theta_ib_0": (2.0, -4.0, 6.0),
        "theta_ib_T": (2.0, 0.5, 8.0),
        "alpha_T": (0.10, 0.0, 0.50),
        "t_delay": (2.0, 0.0, 8.0),           # IB full-formation delay before shutoff [h]
        "tau_IB": (3.0, 0.5, 20.0),           # IB shutoff time after t_delay [h] (T-gated to T30 via g_sol)
    },
    # Gas + adopted qPTexp baseline (for the M1b maintenance re-test on corrected volume,
    # 2026-07-24): the gas observables on top of the adopted sparse model, with/without a
    # Pirt maintenance term m_s (mu = Y_xs*(qS - m_s) -> qS-dependent apparent yield + maintenance CO2).
    "M1a_aload_qPTexp": {**_BASE, **_ALOAD, **_QPT_EXP},
    "M1b_aload_qPTexp": {**_BASE, **_ALOAD, **_QPT_EXP, **_M1B},
}

# Which observation channels each model fits.
OBS_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "M0": SPARSE_STATES,
    "M0_aload": SPARSE_STATES,
    "M0_decay": SPARSE_STATES,
    "M0_aload_decay": SPARSE_STATES,
    "M0_aload_haldane": SPARSE_STATES,
    "M0_aload_burden": SPARSE_STATES,
    "M0_aload_qPT": SPARSE_STATES,
    "M0_aload_qPT_haldane": SPARSE_STATES,
    "M0_aload_qPTexp": SPARSE_STATES,
    "M0_aload_qPTexp_decl": SPARSE_STATES,
    "M0_kinpart": SPARSE_STATES,
    "M0_kinpart_deg": SPARSE_STATES,
    "M0_kinpart_deg_free": SPARSE_STATES,
    "M0_aload_qPTexp_o2": SPARSE_STATES,
    "M0_proteostasis": SPARSE_STATES,
    "M0_proteostasis_v2": SPARSE_STATES,
    "M0_aload_qPTexp_yxt": SPARSE_STATES,
    "M0_aload_qPTexp_foldlag": SPARSE_STATES,
    "M0_aload_qPTexp_refold": SPARSE_STATES,
    "M0_aload_qPTexp_ibdeg": SPARSE_STATES,
    "M0_aload_qPTexp_refoldibdeg": SPARSE_STATES,
    "M0_qPTexp_hillib": SPARSE_STATES,
    "M0_aload_qPTexp_ibtransient": SPARSE_STATES,
    "M0_smet_independent": SPARSE_STATES,
    "M0_smet_independent_ibdeg": SPARSE_STATES,
    "M_decoupled_ibtransient": SPARSE_STATES,
    "M_decoupled_v2": SPARSE_STATES,
    "M_decoupled_v3": SPARSE_STATES,
    "M1a": SPARSE_STATES + GAS_STATES,
    "M1a_aload": SPARSE_STATES + GAS_STATES,
    "M1a_aload_decay": SPARSE_STATES + GAS_STATES,
    "M1b": SPARSE_STATES + GAS_STATES,
    "M1b_aload": SPARSE_STATES + GAS_STATES,
    "M1a_aload_qPTexp": SPARSE_STATES + GAS_STATES,
    "M1b_aload_qPTexp": SPARSE_STATES + GAS_STATES,
}


def param_names(model: str) -> List[str]:
    return list(PARAM_SPECS[model].keys())


def spec_arrays(model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (init, lo, hi) vectors for a model."""
    specs = PARAM_SPECS[model]
    init = np.array([specs[k][0] for k in specs], dtype=float)
    lo = np.array([specs[k][1] for k in specs], dtype=float)
    hi = np.array([specs[k][2] for k in specs], dtype=float)
    return init, lo, hi


def unpack(model: str, vec: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(PARAM_SPECS[model].keys(), vec)}


# ---------------------------------------------------------------------------
# Forcing-input arrays for one run (raw arrays + np.interp in the RHS; no class)
# ---------------------------------------------------------------------------
def build_inputs(run: dict) -> Dict[str, np.ndarray]:
    """Return the raw online forcing arrays (already windowed to [t0, tend])."""
    return {
        "t": np.asarray(run["t_online"], dtype=float),
        "feed_rate": np.asarray(run["feed_rate_L_h"], dtype=float),
        "V": np.asarray(run["V_L"], dtype=float),
        "temp": np.asarray(run["temp"], dtype=float),
        "is_induced": np.asarray(run["is_induced"], dtype=float),
        "t_ind": np.asarray(run.get("ind_time", np.zeros_like(run["t_online"])), dtype=float),
        "po2": np.asarray(run.get("po2", np.full(len(run["t_online"]), 40.0)), dtype=float),
    }


def _induced_at(t: float, inp: Dict[str, np.ndarray]) -> float:
    """Zero-order-hold (previous-value) lookup of is_induced at scalar t."""
    idx = int(np.searchsorted(inp["t"], t, side="right")) - 1
    idx = 0 if idx < 0 else (len(inp["is_induced"]) - 1 if idx >= len(inp["is_induced"]) else idx)
    return float(inp["is_induced"][idx])


# ---------------------------------------------------------------------------
# RHS and fixed-step RK4 simulation
# ---------------------------------------------------------------------------
def make_rhs(model: str, params: Dict[str, float], inp: Dict[str, np.ndarray], c_fold: float):
    """Return an ODE RHS f(t, y) for the given model/params (np.interp forcing)."""
    Y_xs = params["Y_xs"]
    qP_max = params.get("qP_max", 0.0)  # absent in M0_smet_independent (uses qP_sol_max/qP_IB_max)
    K_S = params["K_S"]
    theta_0 = params.get("theta_0", 0.0)  # absent in M_decoupled_v2 (uses theta_sol_*/theta_ib_*)
    theta_T = params.get("theta_T", 0.0)
    m_s = params.get("m_s", 0.0)
    a_qS = params.get("a_qS", 0.0)
    tau_burden = params.get("tau_burden", None)
    Ki = params.get("Ki", None)
    a_b = params.get("a_b", 0.0)
    tau_b = params.get("tau_b", None)
    k_T = params.get("k_T", 0.0)
    alpha_T = params.get("alpha_T", 0.0)
    beta_T = params.get("beta_T", 0.0)  # T-dependent biomass yield loss above 30 C (Y_xs(T))
    tau_fold = params.get("tau_fold", None)
    k_ref = params.get("k_ref", 0.0)
    k_deg = params.get("k_deg", 0.0)
    qP_IBmax = params.get("qP_IBmax", None)
    K_IB = params.get("K_IB", None)
    N_hill = params.get("N_hill", None)
    tau_IB = params.get("tau_IB", None)
    tau_P = params.get("tau_P", None)      # total production decline [h] (_DECL)
    tau_IBd = params.get("tau_IBd", None)  # IB-specific extra shutoff [h] (_DECL)
    q_syn_max = params.get("q_syn_max", None)   # proteostasis module (_PROTEO/_PROTEO2):
    C_chap = params.get("C_chap", None)         #   v1: folding-assistance (rescue) capacity
    C_cap = params.get("C_cap", None)           #   v2: proteome folding capacity (rate-limited)
    T50_m = params.get("T50_m", None)           #   misfolding midpoint
    theta_m = params.get("theta_m", None)       #   misfolding-ratio slope
    sigma_IB = params.get("sigma_IB", None)     #   IB vs proteolysis split of misfolded flux
    qP_sol_max = params.get("qP_sol_max", None)
    qP_IB_max_s = params.get("qP_IB_max", None)
    K_IB_smet = params.get("K_IB", None)
    theta_sol_0 = params.get("theta_sol_0", None)
    theta_sol_T = params.get("theta_sol_T", None)
    theta_ib_0 = params.get("theta_ib_0", None)
    theta_ib_T = params.get("theta_ib_T", None)
    t_delay = params.get("t_delay", 0.0)  # IB delayed-shutoff onset [h] (v3; 0 = immediate, v2)
    k_f = params.get("k_f0", None)        # kinetic partition: folding at <=30 C [1/h]
    delta_f = params.get("delta_f", 0.0)  # folding collapse above 30 C [1/C]
    k_a0 = params.get("k_a0", None)       # kinetic partition: aggregation at 30 C [L/g/h]
    delta_a = params.get("delta_a", 0.0)  # aggregation T-steepness [1/C]
    k_p0 = params.get("k_p0", 0.0)        # proteolysis at <=30 C [1/h]
    gamma_p = params.get("gamma_p", 0.0)  # proteolysis heat-shock amplification [1/C]
    use_kinpart = "k_f0" in PARAM_SPECS[model]
    k_deg0 = params.get("k_deg0", 0.0)    # aggregate proteolysis at <=30 C [1/h] (v3)
    K_o2 = params.get("K_o2", None)       # pO2 stress half-saturation [%] (_O2)
    use_o2 = "K_o2" in PARAM_SPECS[model]
    use_load = "a_qS" in PARAM_SPECS[model]
    use_decay = "tau_burden" in PARAM_SPECS[model]
    use_burden = "a_b" in PARAM_SPECS[model]
    use_qPT = "k_T" in PARAM_SPECS[model]
    use_qPTexp = "alpha_T" in PARAM_SPECS[model]
    use_foldlag = "tau_fold" in PARAM_SPECS[model]
    use_hillib = "N_hill" in PARAM_SPECS[model]
    use_ibtransient = "tau_IB" in PARAM_SPECS[model]
    use_decl = "tau_P" in PARAM_SPECS[model]
    use_smetindep = "qP_sol_max" in PARAM_SPECS[model] and "K_IB" in PARAM_SPECS[model]
    use_decoupled_v2 = "theta_sol_0" in PARAM_SPECS[model]
    use_proteo = "q_syn_max" in PARAM_SPECS[model]
    use_proteo2 = "C_cap" in PARAM_SPECS[model]
    use_decoupled = ("qP_sol_max" in PARAM_SPECS[model] and "K_IB" not in PARAM_SPECS[model]
                     and not use_decoupled_v2)
    t_arr = inp["t"]
    feed_arr = inp["feed_rate"]
    V_arr = inp["V"]
    T_arr = inp["temp"]
    tind_arr = inp["t_ind"]
    t_lo, t_hi = float(t_arr[0]), float(t_arr[-1])

    def rhs(t, y):
        X = y[0] if y[0] > EPS else EPS
        tc = t if t_lo <= t <= t_hi else (t_lo if t < t_lo else t_hi)
        V = np.interp(tc, t_arr, V_arr)
        if V < 1e-4:
            V = 1e-4
        feed_rate = np.interp(tc, t_arr, feed_arr)
        if feed_rate < 0.0:
            feed_rate = 0.0
        T = np.interp(tc, t_arr, T_arr)
        is_ind = _induced_at(tc, inp)

        qs = feed_rate * S_IN / (X * V)
        if qs > QS_CAP:
            qs = QS_CAP
        D = feed_rate / V
        mu = Y_xs * (qs - m_s) if m_s > 0 else Y_xs * qs
        if beta_T != 0.0:
            # T-dependent biomass yield (heat stress -> more carbon to CO2/proteostasis, less to biomass).
            # Mirrors alpha_T (product folding loss); beta_T=0 -> constant yield (baseline).
            mu = mu * float(np.exp(-beta_T * max(T - T_REF, 0.0)))

        qP = is_ind * qP_max * qs / (qs + K_S)
        if Ki is not None:
            qP = is_ind * qP_max * qs / (K_S + qs + qs * qs / Ki)  # Haldane substrate-inhibition
        if use_decay:
            t_ind = np.interp(tc, t_arr, tind_arr)
            qP = qP * float(np.exp(-max(t_ind, 0.0) / tau_burden))
        if use_qPT:
            # Asymmetric T-suppression of total production: flat for T<=30, drops above.
            qP = qP * max(1.0 - k_T * max(T - T_REF, 0.0), 0.05)
        if use_qPTexp:
            # First-order (exponential) folding-efficiency decline above the optimum.
            qP = qP * float(np.exp(-alpha_T * max(T - T_REF, 0.0)))
        if use_o2:
            # Measured-pO2 stress forcing: Monod in dissolved O2. At the 40% setpoint
            # this is a near-constant factor (absorbed into qP_max); excursions below
            # it (F12's crash to 0.6%) transiently throttle synthesis.
            po2_t = float(np.interp(tc, t_arr, inp["po2"]))
            qP = qP * (po2_t / (po2_t + K_o2))

        if use_kinpart:
            # Kinetic partitioning of the unfolded pool (see _KIN). QSSA pool: folding/
            # aggregation relax in minutes, samples are hours apart. Aggregation is
            # SECOND-order (k_a*U^2) -> the sol/IB partition rises with expression load
            # (the a_qS effect emerges from the rate-law orders); proteolysis k_p(T)*U*
            # routes the surplus to CO2 (the CER-validated alpha channel).
            q_syn = is_ind * qP_max * qs / (qs + K_S)          # specific synthesis [g/gX/h]
            P_syn = q_syn * X                                   # total synthesis [g/L/h]
            k_a = k_a0 * float(np.exp(delta_a * (T - T_REF)))
            k_fT = k_f * float(np.exp(-delta_f * max(T - T_REF, 0.0)))
            k_p = k_p0 * float(np.exp(gamma_p * max(T - T_REF, 0.0)))
            b = k_fT + k_p + D
            U_star = (-b + float(np.sqrt(b * b + 4.0 * k_a * P_syn))) / (2.0 * k_a)
            r_sol = k_fT * U_star                               # folded -> soluble [g/L/h]
            r_ib = k_a * U_star * U_star                        # aggregated -> IB [g/L/h]
            k_deg_T = k_deg0 * float(np.exp(gamma_p * max(T - T_REF, 0.0)))
            dX = mu * X - D * X
            dP_sol = r_sol - D * y[1]
            dP_IB = r_ib - (k_deg_T + D) * y[2]
            return (dX, dP_sol, dP_IB)

        if use_decoupled_v2:
            # Decoupled v2: INDEPENDENT (non-complementary) T-gates -- the fix for v1's failure.
            # v1 used complementary gates (g_sol+g_ib=1), which forced a sol-vs-IB tradeoff at T30
            # (g_ib(T30)->0.05), starving IB formation -> qP_IB_max pinned at bound -> couldn't reach
            # the peak -> clock pulled to tau=inf. Independent gates let BOTH be ~1 at T30 (data: T30
            # has substantial sol AND IB), so a moderate qP_IB_max reaches the peak and the clock can
            # engage. g_sol: ~1 T25/T30, ~0 T35 (sol present iff T<=30). g_ib: ~0 T25, ~1 T30/T35
            # (IB present iff T>=30). Clock z=exp(-t_ind*g_sol/tau): runs where sol is also made (T30,
            # proteostasis working -> cell adapts and shuts off aggregation); z=1 at T35 (sol fails,
            # overwhelmed -> IB sustained, matches F07/F19/F12 rising).
            g_sol = 1.0 / (1.0 + np.exp(-(theta_sol_0 - theta_sol_T * (T - T_REF))))
            g_ib = 1.0 / (1.0 + np.exp(-(theta_ib_0 + theta_ib_T * (T - T_REF))))
            eta = float(np.exp(-alpha_T * max(T - T_REF, 0.0)))
            monod_qs = qs / (qs + K_S)
            t_ind = np.interp(tc, t_arr, tind_arr)
            # Delayed shutoff (v3): full formation (z~1) until t_delay, then smooth decay. SMOOTH-softplus
            # form (differentiable -- avoids the max() kink that stalled the optimizer's t_delay Jacobian;
            # t_delay=0 -> ~immediate decay = v2). logaddexp is numerically stable.
            _shutoff = np.logaddexp(0.0, 2.0 * (t_ind - t_delay)) / 2.0  # smooth max(t-t_delay, 0)
            z = float(np.exp(-_shutoff * g_sol / tau_IB))
            qP_sol = qP_sol_max * monod_qs * g_sol * eta
            qP_IB = qP_IB_max_s * monod_qs * g_ib * eta * z
            dX = mu * X - D * X
            dP_sol = is_ind * qP_sol * X - D * y[1]
            dP_IB = is_ind * qP_IB * X - D * y[2]
            return (dX, dP_sol, dP_IB)

        if use_proteo:
            # Proteostasis folding competition (see _PROTEO/_PROTEO2): one mechanism, three
            # consequences. The fold/misfold race phi_m = k_m/(k_f+k_m) sets the T regime
            # (T25 soluble, T35 IB); the load-ordering enters through the race itself (v2:
            # k_f capacity-limited) replacing a_qS*log(qS); proteolysis of the misfolded flux
            # is the alpha_T channel (T35 total loss -> CO2, CER-checkable via sigma_IB).
            q_syn0 = q_syn_max * qs / (qs + K_S)          # specific synthesis rate (pre-induction)
            if use_proteo2:
                # v2: capacity limits the FOLDING RATE: k_f_eff = k_f*(1 - q_syn/C_cap).
                kf_fac = 1.0 - q_syn0 / C_cap
                if kf_fac < 1e-6:
                    kf_fac = 1e-6                          # capacity collapsed -> race lost
                eta_m = 1.0 / (1.0 + float(np.exp(theta_m * (T - T50_m))) * kf_fac)
                q_syn = is_ind * q_syn0
                qP_sol = (1.0 - eta_m) * q_syn             # folded -> soluble
                qP_IB = sigma_IB * eta_m * q_syn           # misfolded -> recoverable IB
            else:
                eta_m = 1.0 / (1.0 + np.exp(-theta_m * (T - T50_m)))
                q_syn = is_ind * q_syn0
                q_mis = eta_m * q_syn
                rescued = q_mis if q_mis < C_chap else C_chap   # chaperone rescue (capacity-limited)
                overflow = q_mis - rescued
                qP_sol = (1.0 - eta_m) * q_syn + rescued        # folded + rescued -> soluble
                qP_IB = sigma_IB * overflow                     # aggregates -> recoverable IB
            dX = mu * X - D * X
            dP_sol = qP_sol * X - D * y[1]
            dP_IB = qP_IB * X - D * y[2]
            return (dX, dP_sol, dP_IB)

        if use_decoupled:
            # Decoupled independent sol/IB (NO partition): each its own amplitude x Monod(qS) x
            # complementary T-gate x eta(T). sol is a pure accumulator (no decline channel -- matches
            # data: sol never declines); IB declines iff its formation drops below dilution. IB carries
            # a T-gated clock transient z = exp(-t_ind*gSol/tau_IB): the aggregation machinery fatigues
            # on a clock, but only in the T30 regime (gSol~1, where the cell also folds soluble) -- so
            # IB formation shuts off late at T30 -> P_IB declines via dilution; at T35 (gSol~0,
            # proteostasis overwhelmed) z=1 -> IB sustained (matches data: T35 IB keeps rising).
            phi_logit = theta_0 + theta_T * (T - T_REF)
            g_sol = 1.0 / (1.0 + np.exp(phi_logit))    # sol T-gate: ~1 T25, ~0 T35
            g_ib = 1.0 / (1.0 + np.exp(-phi_logit))     # IB T-gate (complementary): ~0 T25, ~1 T35
            eta = float(np.exp(-alpha_T * max(T - T_REF, 0.0)))
            monod_qs = qs / (qs + K_S)
            t_ind = np.interp(tc, t_arr, tind_arr)
            z = float(np.exp(-max(t_ind, 0.0) * g_sol / tau_IB))
            qP_sol = qP_sol_max * monod_qs * g_sol * eta
            qP_IB = qP_IB_max_s * monod_qs * g_ib * eta * z
            dX = mu * X - D * X
            dP_sol = is_ind * qP_sol * X - D * y[1]
            dP_IB = is_ind * qP_IB * X - D * y[2]
            return (dX, dP_sol, dP_IB)

        if use_smetindep:
            # Independent S_met-driven rates (NN m13 port): qP_sol sustained, qP_IB transient.
            # S_met (y[3]) = cumulative substrate load since induction; only qP_IB carries the decline.
            phi_T = 1.0 / (1.0 + np.exp(-(theta_0 + theta_T * (T - T_REF))))
            eta = float(np.exp(-alpha_T * max(T - T_REF, 0.0)))
            monod_qs = qs / (qs + K_S)
            qP_sol = qP_sol_max * (1.0 - phi_T) * monod_qs * eta
            qP_IB = qP_IB_max_s * phi_T * monod_qs * eta * float(np.exp(-y[3] / K_IB_smet))
            dX = mu * X - D * X
            dP_sol = is_ind * qP_sol * X - D * y[1]
            # Net IB flux can go negative: formation - first-order proteolysis (k_deg, -> CO2) - dilution.
            # k_deg=0 for plain M0_smet_independent (no-op); >0 only in M0_smet_independent_ibdeg.
            dP_IB = is_ind * qP_IB * X - k_deg * y[2] - D * y[2]
            dSmet = S_IN * D * is_ind
            return (dX, dP_sol, dP_IB, dSmet)

        if use_hillib:
            # Cooperative (Hill/Moser) IB formation in qP; soluble = total - IB (mass-conserving).
            # IB capacity is T-dependent (theta): low at T25 (soluble), high at T35 (IB).
            cap = 1.0 / (1.0 + np.exp(-(theta_0 + theta_T * (T - T_REF))))
            qP_ib = qP_IBmax * cap * (qP ** N_hill) / (qP ** N_hill + K_IB ** N_hill)
            qP_sol = qP - qP_ib
            if qP_sol < 0.0:
                qP_sol = 0.0
            dX = mu * X - D * X
            dP_sol = qP_sol * X - D * y[1]
            dP_IB = qP_ib * X - D * y[2]
            return (dX, dP_sol, dP_IB)

        if use_load:
            # qS-based load term (hybrid M3-P1 form): qS discriminates the 30C
            # feed regimes far better than the saturated qP. QS_REF=0.4 centers it.
            logit = theta_0 + theta_T * (T - T_REF) + a_qS * np.log(max(qs, 1e-6) / QS_REF)
        else:
            logit = theta_0 + theta_T * (T - T_REF)
        if use_burden:
            # Latent burden state b (y[3]) shifts the partition toward IB as the
            # accumulated metabolic cost of expression builds up. db/dt below.
            logit = logit + a_b * y[3]
        phi = 1.0 / (1.0 + np.exp(-logit))
        if phi < 0.0:
            phi = 0.0
        elif phi > 1.0:
            phi = 1.0

        if use_ibtransient:
            # IB formation is transient: front-loaded then decays over induction (early-stress
            # aggregation that subsides); soluble = total - decayed-IB (mass-conserving, sol rises
            # as IB shuts off). Distinct from total induction-decay (this acts on IB only).
            t_ind = np.interp(tc, t_arr, tind_arr)
            decay = float(np.exp(-max(t_ind, 0.0) / tau_IB))
            qP_ib = phi * qP * decay
            qP_sol = qP - qP_ib
            if qP_sol < 0.0:
                qP_sol = 0.0
            dX = mu * X - D * X
            dP_sol = qP_sol * X - D * y[1]
            dP_IB = qP_ib * X - D * y[2]
            return (dX, dP_sol, dP_IB)

        if use_decl:
            # Front-loaded production (see _DECL): total formation decays over induction
            # (tau_P -> decelerating P_sol, matching the data), IB formation additionally
            # shuts off with tau_IBd (< tau_P) so P_IB peaks and declines once formation
            # drops below dilution. Mass-conserving split (sol = total - IB).
            t_ind = np.interp(tc, t_arr, tind_arr)
            dec_P = float(np.exp(-max(t_ind, 0.0) / tau_P))
            dec_IBd = float(np.exp(-max(t_ind, 0.0) / tau_IBd))
            qP_ib = phi * qP * dec_P * dec_IBd
            qP_sol_d = qP * dec_P - qP_ib
            if qP_sol_d < 0.0:
                qP_sol_d = 0.0
            dX = mu * X - D * X
            dP_sol = qP_sol_d * X - D * y[1]
            dP_IB = qP_ib * X - D * y[2]
            return (dX, dP_sol, dP_IB)

        dX = mu * X - D * X
        if use_foldlag:
            # Soluble emerges from the folding pool N_fold (y[3]) after maturation tau_fold;
            # nascent soluble (1-phi)*qP*X feeds N_fold. IB (phi*qP*X) is immediate. This
            # conserves total product mass but delays soluble -> early-IB/late-soluble dynamics.
            dN = (1.0 - phi) * qP * X - y[3] / tau_fold - D * y[3]
            dP_sol = y[3] / tau_fold - D * y[1]
            dP_IB = phi * qP * X - D * y[2]
            return (dX, dP_sol, dP_IB, dN)
        dP_sol = (1.0 - phi) * qP * X - D * y[1]
        dP_IB = phi * qP * X - D * y[2]
        if k_ref > 0.0 or k_deg > 0.0:
            # IB -> soluble refolding (mass-conserving, k_ref) + IB degradation/proteolysis (-> CO2, k_deg).
            dP_IB = dP_IB - (k_ref + k_deg) * y[2]
            dP_sol = dP_sol + k_ref * y[2]
        if use_burden:
            # b is a first-order low-pass of the production rate (gain tau_b):
            # before induction qP=0 so b decays to 0; during induction b -> qP*X*tau_b.
            db = qP * X - y[3] / tau_b
            return (dX, dP_sol, dP_IB, db)
        return (dX, dP_sol, dP_IB)

    return rhs


def _rk4(rhs, y0, t0, tend, h):
    """Fixed-step RK4 over [t0, tend]. Returns (t_grid[n+1], Y[dim, n+1]).

    Dimension is inferred from y0, so latent-state variants (e.g. the burden
    state b in M0_aload_burden) work without specialization.
    """
    n = max(int(np.ceil((tend - t0) / h)), 1)
    hh = (tend - t0) / n
    ts = np.empty(n + 1)
    Y = np.empty((len(y0), n + 1))
    y = np.array(y0, dtype=float)
    ts[0] = t0
    Y[:, 0] = y
    for i in range(n):
        t = t0 + i * hh
        k1 = np.asarray(rhs(t, y))
        k2 = np.asarray(rhs(t + hh / 2, y + hh / 2 * k1))
        k3 = np.asarray(rhs(t + hh / 2, y + hh / 2 * k2))
        k4 = np.asarray(rhs(t + hh, y + hh * k3))
        y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ts[i + 1] = t + hh
        Y[:, i + 1] = y
    return ts, Y


def simulate_run(
    model: str,
    params: Dict[str, float],
    run: dict,
    c_fold: float,
    eval_gas: bool = True,
    dt: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Integrate one run with fixed-step RK4; return predictions at obs/gas times.

    Returns dict with:
      obs: {X, P_sol, P_IB} arrays aligned with run['t_obs']
      gas: {CER, OUR} arrays aligned with run['t_gas'] (empty if not eval_gas)
      sol_t, sol_X  (full solution grid, for plotting)
    """
    inp = build_inputs(run)
    rhs = make_rhs(model, params, inp, c_fold)
    t0, tend = run["t0"], run["tend"]

    # Latent-state variants (burden b / folding pool N_fold / S_met) start the extra state at 0.
    # qP_sol_max counts as 4-state only for the smet variants (which carry S_met via K_IB); the
    # decoupled sol/IB model (qP_sol_max, no K_IB) is 3-state.
    _four = ("a_b" in PARAM_SPECS[model] or "tau_fold" in PARAM_SPECS[model]
             or ("qP_sol_max" in PARAM_SPECS[model] and "K_IB" in PARAM_SPECS[model]))
    y0 = (run["X0"], 0.0, 0.0, 0.0) if _four else (run["X0"], 0.0, 0.0)
    try:
        st, Y = _rk4(rhs, y0, t0, tend, dt)
    except Exception:
        nan = np.full(1, np.nan)
        return {
            "obs": {"X": np.full(len(run["t_obs"]), np.nan),
                    "P_sol": np.full(len(run["t_obs"]), np.nan),
                    "P_IB": np.full(len(run["t_obs"]), np.nan)},
            "gas": {"CER": np.full(len(run["t_gas"]), np.nan),
                    "OUR": np.full(len(run["t_gas"]), np.nan)},
            "sol_t": np.array([t0, tend]), "sol_X": np.array([run["X0"], np.nan]),
        }
    if not np.all(np.isfinite(Y)):
        nan_obs = np.full(len(run["t_obs"]), np.nan)
        nan_gas = np.full(len(run["t_gas"]), np.nan)
        return {"obs": {"X": nan_obs, "P_sol": nan_obs.copy(), "P_IB": nan_obs.copy()},
                "gas": {"CER": nan_gas, "OUR": nan_gas.copy()}, "sol_t": st, "sol_X": Y[0]}

    Xs, Psol_s, Pib_s = Y[0], Y[1], Y[2]
    t_obs = np.asarray(run["t_obs"], dtype=float)
    out = {
        "obs": {"X": np.interp(t_obs, st, Xs),
                "P_sol": np.interp(t_obs, st, Psol_s),
                "P_IB": np.interp(t_obs, st, Pib_s)},
        "sol_t": st,
        "sol_X": Xs,
        "sol_Psol": Psol_s,
        "sol_Pib": Pib_s,
        "sol_b": Y[3] if Y.shape[0] > 3 else None,
    }

    t_gas = np.asarray(run["t_gas"], dtype=float) if eval_gas else np.array([])
    if eval_gas and len(t_gas) > 0:
        Xg = np.maximum(np.interp(t_gas, st, Xs), EPS)
        tg_c = np.clip(t_gas, float(inp["t"][0]), float(inp["t"][-1]))
        feed_g = np.clip(np.interp(tg_c, inp["t"], inp["feed_rate"]), 0.0, None)
        Vg = np.clip(np.interp(tg_c, inp["t"], inp["V"]), 1e-4, None)
        Tg = np.interp(tg_c, inp["t"], inp["temp"])
        ind_g = np.array([_induced_at(float(t), inp) for t in tg_c])
        qS_g = np.minimum(feed_g * S_IN / (Xg * Vg), QS_CAP)
        Y_xs = params["Y_xs"]; qP_max = params.get("qP_max", 0.0); K_S = params["K_S"]; m_s = params.get("m_s", 0.0)
        Ki_g = params.get("Ki", None)
        mu_g = Y_xs * (qS_g - m_s) if m_s > 0 else Y_xs * qS_g
        if Ki_g is not None:
            qP_g = ind_g * qP_max * qS_g / (K_S + qS_g + qS_g * qS_g / Ki_g)   # Haldane
        else:
            qP_g = ind_g * qP_max * qS_g / (qS_g + K_S)
        if "tau_burden" in PARAM_SPECS[model]:
            tind_g = np.interp(tg_c, inp["t"], inp["t_ind"])
            qP_g = qP_g * np.exp(np.maximum(tind_g, 0.0) / params["tau_burden"] * -1.0)
        if "tau_P" in PARAM_SPECS[model]:
            tind_g = np.interp(tg_c, inp["t"], inp["t_ind"])
            qP_g = qP_g * np.exp(np.maximum(tind_g, 0.0) / params["tau_P"] * -1.0)
        if "k_T" in PARAM_SPECS[model]:
            k_T = params["k_T"]
            Tg_factor = np.maximum(1.0 - k_T * np.maximum(Tg - T_REF, 0.0), 0.05)
            qP_g = qP_g * Tg_factor
        if "alpha_T" in PARAM_SPECS[model]:
            Tg_factor = np.exp(-params["alpha_T"] * np.maximum(Tg - T_REF, 0.0))
            qP_g = qP_g * Tg_factor
        if "q_syn_max" in PARAM_SPECS[model]:
            # Proteostasis module: gas sees the MEASURABLE flux (folded + IB); the proteolyzed
            # misfolded share (1-sigma_IB) routes to CO2 -- the same convention as alpha_T
            # (loss applied before the balance; the alpha_cer_test logic applies unchanged).
            qsyn0_g = params["q_syn_max"] * qS_g / (qS_g + params["K_S"])
            if "C_cap" in PARAM_SPECS[model]:
                kfac_g = np.maximum(1.0 - qsyn0_g / params["C_cap"], 1e-6)
                eta_g = 1.0 / (1.0 + np.exp(params["theta_m"] * (Tg - params["T50_m"])) * kfac_g)
                qsyn_g = ind_g * qsyn0_g
                qP_g = ((1.0 - eta_g) * qsyn_g
                        + params["sigma_IB"] * eta_g * qsyn_g)
            else:
                eta_g = 1.0 / (1.0 + np.exp(-params["theta_m"] * (Tg - params["T50_m"])))
                qsyn_g = ind_g * qsyn0_g
                qmis_g = eta_g * qsyn_g
                resc_g = np.minimum(qmis_g, params["C_chap"])
                qP_g = ((1.0 - eta_g) * qsyn_g + resc_g
                        + params["sigma_IB"] * (qmis_g - resc_g))
        cer, our = predict_gas(qS_g * Xg, mu_g * Xg, qP_g * Xg)
        out["gas"] = {"CER": cer, "OUR": our}
    else:
        out["gas"] = {"CER": np.array([]), "OUR": np.array([])}
    return out
