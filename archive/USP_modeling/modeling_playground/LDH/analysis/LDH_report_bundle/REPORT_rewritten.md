# Temperature and Feeding Strategy Effects on LDH Solubility: Re-analysis Report (Corrected)

**Date:** 2026-01-05  
**System:** *E. coli* BL21(DE3) (pET), recombinant LDH  
**Dataset:** `LDH_consolidated_results.xlsx`  

This document replaces the previous report and is rewritten from scratch to be statistically and scientifically sound.

---

## 1. Executive summary (what the data actually supports)

### Main outcome (robust)
Across the sampled design space, **temperature is the dominant driver of LDH partitioning**:

- At **25 °C**, the product is **fully soluble** at all induction sampling points (**IB fraction = 0%** in both runs).
- At **35 °C**, the product is **fully insoluble** (**IB fraction = 100%** in both runs).
- At **30 °C** (center condition, **3 replicate runs**), LDH is **mostly soluble**, with a **small but consistent IB fraction (~11–15%)**.

### What cannot be concluded from this dataset (important)
Because **qs was not varied at 30 °C** (only qs_target = 0.4 there), and because the response is **saturated** at 25 °C (0%) and 35 °C (100%), this dataset **cannot** answer whether qs influences IB formation *near the transition region* (where the response is sensitive).

Correct statement:

> **Within the tested extremes (25 °C and 35 °C), changing nominal initial qs does not alter the temperature-determined solubility regime.  
> A qs effect may still exist near ~30–33 °C, but the present design does not test it.**

---

## 2. Data and preprocessing (critical for correctness)

### 2.1 What is measured
The file contains dense process signals (temperature, biomass, feed, volume, etc.) and sparse analytical samples for protein.

- `Protein soluble [g/L]`, `Protein IB [g/L]` are present **only at sampling times** (42 rows total).
- Many dense time points contain **zeros** in `Protein total calculated [g/L]` due to table filling; these zeros are **not measurements** and must not be used for kinetics.

All protein time-course plots and endpoints in this report use **only true analytical sampling rows**.

### 2.2 Induction alignment
For each run, the induction start time **t₀** is defined as the first time point with `Is induced [-] = 1`.  
All induction dynamics are plotted versus:

t_since_ind = t − t₀

### 2.3 “Actual qs” during induction (derived)
Even with approximately constant feed rate, **actual qs decreases** as biomass increases.

To quantify this, qs is estimated as:

qs(t) = mS_dot(t) / (X(t) · V(t))

where substrate mass flow is estimated from feed rate.

**Assumption (from the original report):** feed contains **400 g/L glucose** (= **0.4 g/mL**).  
Thus:

mS_dot = F_dot [mL/h] × 0.4 [g/mL]

This provides a consistent qs proxy for comparing runs. If the true feed concentration differs, the qs curves scale accordingly but the **decline trend** remains.

---

## 3. Experimental design (what was actually run)

Two factors were intended:

- Temperature setpoint: 25 °C vs 35 °C (with a 30 °C center)
- Nominal initial qs target: 0.2 vs 0.6 (with a 0.4 center)

The realized design contains **5 populated cells** (corners + center), with replication only at the center (30 °C, qs_target=0.4, n=3).

![](Fig1_design_matrix.png)

---

## 4. Primary endpoint results (end of induction)

The endpoint for each run is defined as the **last analytical sample during induction** (~8.2 h after induction).

### 4.1 Run-level endpoint table

| run_name | Temp_set_C | qs_target | Temp_measured_C | t_end_h | qs_est_initial | qs_est_end | qs_est_decline_pct | P_tot_end_gL | P_sol_end_gL | P_IB_end_gL | Sol_frac_end_pct | IB_frac_end_pct | productivity_total_gL_h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ferm_02 | 25 | 0.200 | 25.010 | 8.250 | 0.196 | 0.145 | 26.124 | 3.087 | 3.087 | 0.000 | 100.000 | 0.000 | 0.374 |
| ferm_03 | 25 | 0.600 | 24.980 | 8.250 | 0.528 | 0.243 | 53.953 | 4.143 | 4.143 | 0.000 | 100.000 | 0.000 | 0.502 |
| ferm_05 | 30 | 0.400 | 30.015 | 8.217 | 0.355 | 0.210 | 40.778 | 4.403 | 3.861 | 0.542 | 87.690 | 12.310 | 0.536 |
| ferm_06 | 30 | 0.400 | 30.000 | 8.217 | 0.368 | 0.207 | 43.832 | 3.813 | 3.252 | 0.561 | 85.284 | 14.716 | 0.464 |
| ferm_07 | 30 | 0.400 | 30.005 | 8.200 | 0.371 | 0.222 | 40.079 | 4.426 | 3.913 | 0.513 | 88.411 | 11.589 | 0.540 |
| ferm_01 | 35 | 0.200 | 34.980 | 8.233 | 0.212 | 0.170 | 19.835 | 1.744 | 0.000 | 1.744 | 0.000 | 100.000 | 0.212 |
| ferm_04 | 35 | 0.600 | 34.997 | 8.250 | 0.502 | 0.237 | 52.796 | 2.848 | 0.000 | 2.848 | 0.000 | 100.000 | 0.345 |

**Notes**
- `qs_est_*` columns are derived from feed + biomass + volume signals (Section 2.3).
- `productivity_total_gL_h = P_tot_end / t_end` is an average productivity over induction (not an instantaneous rate).

### 4.2 Summary by temperature

| Temp_set_C | n | IB_mean | IB_sd | Ptot_mean | Ptot_sd | Prod_mean | Prod_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25.000 | 2.000 | 0.000 | 0.000 | 3.615 | 0.746 | 0.438 | 0.090 |
| 30.000 | 3.000 | 12.871 | 1.637 | 4.214 | 0.348 | 0.513 | 0.043 |
| 35.000 | 2.000 | 100.000 | 0.000 | 2.296 | 0.780 | 0.279 | 0.094 |

At 25 °C and 35 °C the IB fraction is exactly 0% or 100% (no within-temperature variation in this dataset).  
At 30 °C, the replicate SD in IB fraction at the endpoint is ~1.64 percentage points.

---

## 5. Time-course results during induction (analytical samples only)

### 5.1 IB fraction over time
![](Fig2_IB_fraction_timecourse.png)

### 5.2 Total protein over time
![](Fig3_total_protein_timecourse.png)

### 5.3 Estimated qs trajectories
![](Fig4_qs_est_trajectories.png)

---

## 6. Factor-effect interpretation (correct level of claim)

### 6.1 Endpoint IB fraction vs temperature
![](Fig5_endpoint_IB_vs_temp.png)

### 6.2 Endpoint protein partitioning
![](Fig6_endpoint_partitioning.png)

---

## 7. Planning future experiments (power and design recommendations)

### 7.1 Detectable qs effect at ~30 °C vs sample size
![](Fig7_power_curve_30C.png)

Approximate planning table (two-sided α=0.05, 80% power, normal approximation):

| n_per_group | min_detectable_delta_pct |
| --- | --- |
| 1.00 | 6.48 |
| 2.00 | 4.58 |
| 3.00 | 3.74 |
| 5.00 | 2.90 |
| 8.00 | 2.29 |
| 10.00 | 2.05 |
| 15.00 | 1.67 |
| 20.00 | 1.45 |

---

## Appendix A. Center-point replicate variability (30 °C, qs_target=0.4)

| run_name | IB_frac_end_pct | P_tot_end_gL | productivity_total_gL_h |
| --- | --- | --- | --- |
| ferm_05 | 12.310 | 4.403 | 0.536 |
| ferm_06 | 14.716 | 3.813 | 0.464 |
| ferm_07 | 11.589 | 4.426 | 0.540 |
