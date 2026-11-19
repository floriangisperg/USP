# Temperature and Feeding Strategy Effects on LDH Solubility (LDH campaign)  
## Exhaustive corrected re-analysis report

**Date:** 2026-01-05  
**Input data:** `LDH_consolidated_results.xlsx`  
**Main response:** LDH partitioning into soluble vs inclusion bodies (IB), reported as **IB fraction [%]**  
**Scope:** Verify what the data supports, correct invalid statistics, and provide scientifically sound conclusions + next-step design.

---

## 1. What the data supports (bottom line)

### 1.1 Strong, directly-supported finding
Across the executed design space, **temperature dominates LDH partitioning**:

- **25 °C:** LDH is **fully soluble** at every induction sampling point (**IB fraction = 0%**) in both runs at 25 °C.
- **35 °C:** LDH is **fully insoluble** at every induction sampling point (**IB fraction = 100%**) in both runs at 35 °C.
- **30 °C (center, n=3):** LDH is mostly soluble, with a small but consistent IB fraction (**~11–15%**).

This is visible in the endpoint design matrix:

![](Fig1_design_matrix.png)

…and in the endpoint scatter:

![](Fig5_endpoint_IB_vs_temp.png)

### 1.2 What the dataset *does not* identify
This campaign **does not identify a qs effect on IB formation near the transition region**, because:

- the response is **saturated** at the 25 °C and 35 °C corners (0% and 100%),
- qs is **not varied at 30 °C** (only qs_target=0.4 there),
- there is **no replication** at the corners (n=1 per corner cell).

Correct claim level:

> **Within the tested extremes (25 °C and 35 °C), changing nominal initial qs does not change the temperature-determined solubility regime.  
> A qs effect may still exist near the transition (~30–33 °C), but the present design does not test it.**

---

## 2. Data audit and preprocessing (what was checked)

### 2.1 Protein measurements are sparse; dense zeros are not measurements
The spreadsheet contains dense process signals plus sparse protein analytics.

Critical correction vs the previous report:  
`Protein total calculated [g/L]` contains many **zeros at dense time points**, which are **not analytical measurements**.  
Analytical protein samples are identified by non-null values in `Protein soluble [g/L]` or `Protein IB [g/L]`.

- Total analytical protein rows in file: **42**
- Induction analytical samples per run: **4** (total 28 points during induction)

Sampling schedule per run (hours since induction start):

| run_name | induction_sample_times_h |
| --- | --- |
| ferm_01 | 2.22, 4.23, 6.18, 8.23 |
| ferm_02 | 2.25, 4.25, 6.20, 8.25 |
| ferm_03 | 2.43, 4.43, 6.38, 8.25 |
| ferm_04 | 2.48, 4.52, 6.50, 8.25 |
| ferm_05 | 2.23, 4.23, 6.25, 8.22 |
| ferm_06 | 2.38, 4.38, 6.40, 8.22 |
| ferm_07 | 2.22, 4.22, 6.23, 8.20 |

### 2.2 Induction alignment
For each run, induction start **t₀** is defined as the first time point with `Is induced [-]=1`.  
All induction plots use `t_since_ind = t − t₀`.

### 2.3 Temperature stability during induction (process check)
Temperature is stable within each run during induction:

| run_name | Temp_set_C | Temp_mean_C | Temp_sd_C |
| --- | --- | --- | --- |
| ferm_02 | 25 | 24.997 | 0.031 |
| ferm_03 | 25 | 25.113 | 0.465 |
| ferm_05 | 30 | 30.007 | 0.057 |
| ferm_06 | 30 | 30.027 | 0.090 |
| ferm_07 | 30 | 30.017 | 0.060 |
| ferm_01 | 35 | 35.015 | 0.038 |
| ferm_04 | 35 | 35.025 | 0.108 |

### 2.4 Feed profile during induction is (approximately) linear (process check)
A linear fit of cumulative feed vs time since induction gives R² very close to 1, consistent with a linear feed ramp:

| run_name | Temp_set_C | qs_target | feed_slope_mL_h | feed_intercept_mL | feed_linear_R2 |
| --- | --- | --- | --- | --- | --- |
| ferm_02 | 25 | 0.2000 | 16.8346 | 149.7210 | 1.0000 |
| ferm_03 | 25 | 0.6000 | 47.6824 | 133.7157 | 1.0000 |
| ferm_05 | 30 | 0.4000 | 31.8267 | 125.0597 | 1.0000 |
| ferm_06 | 30 | 0.4000 | 31.0422 | 115.9635 | 1.0000 |
| ferm_07 | 30 | 0.4000 | 32.9628 | 130.4165 | 1.0000 |
| ferm_01 | 35 | 0.2000 | 23.6250 | 199.4529 | 1.0000 |
| ferm_04 | 35 | 0.6000 | 58.9432 | 157.9022 | 1.0000 |

---

## 3. Design description and limitations

### 3.1 Intended factors and realized design
Factors:
- Temperature setpoint: 25 vs 35 (center 30)
- Nominal initial qs target: 0.2 vs 0.6 (center 0.4)

Realized design: **5 populated cells**, replication only at the center (30 °C, qs_target=0.4, n=3).  
This matters because it prevents valid inference on interactions and provides limited precision on qs effects.

### 3.2 Consequence of response saturation
At 25 °C the response is pinned at 0% IB, and at 35 °C pinned at 100% IB.  
When a response is saturated:

- Correlations with other variables (qs, productivity) are **not meaningful**.
- Linear models / ANOVA can report “large effects” simply because the mean difference is huge, not because the design provides real uncertainty quantification.
- Within-run correlations are undefined if IB fraction has zero variance.

---

## 4. Primary results (analytical samples only)

## 4.1 IB fraction dynamics
![](Fig2_IB_fraction_timecourse.png)

The within-run trajectories confirm:
- 25 °C runs: constant 0% IB
- 35 °C runs: constant 100% IB
- 30 °C runs: low, stable IB fraction

### 4.2 Total protein dynamics
![](Fig3_total_protein_timecourse.png)

Total protein differs between runs, but corner cells have no replication → treat as descriptive only.

### 4.3 Endpoint protein partitioning (soluble vs IB)
![](Fig6_endpoint_partitioning.png)

---

## 5. Derived “actual qs” dynamics (and why it’s not constant)

### 5.1 Why qs declines
Even with constant feed rate, qs declines because biomass increases.

Biomass increase during induction (based on `Biomass [g/L]` × `Volume corrected`):

| run_name | Temp_set_C | qs_target | X_start_gL | X_end_gL | V_start_L | V_end_L | biomass_mass_start_g | biomass_mass_end_g | biomass_mass_increase_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ferm_02 | 25 | 0.200 | 28.267 | 36.500 | 1.210 | 1.208 | 34.195 | 44.110 | 28.994 |
| ferm_03 | 25 | 0.600 | 30.067 | 52.667 | 1.193 | 1.447 | 35.884 | 76.198 | 112.344 |
| ferm_05 | 30 | 0.400 | 29.700 | 44.267 | 1.185 | 1.306 | 35.201 | 57.827 | 64.279 |
| ferm_06 | 30 | 0.400 | 28.200 | 44.700 | 1.176 | 1.291 | 33.165 | 57.700 | 73.981 |
| ferm_07 | 30 | 0.400 | 29.567 | 43.467 | 1.191 | 1.321 | 35.202 | 57.408 | 63.080 |
| ferm_01 | 35 | 0.200 | 35.333 | 41.233 | 1.259 | 1.314 | 44.500 | 54.176 | 21.746 |
| ferm_04 | 35 | 0.600 | 38.100 | 57.067 | 1.218 | 1.563 | 46.401 | 89.205 | 92.248 |

### 5.2 Estimated qs trajectories
**Assumption (from previous report):** feed glucose concentration = 400 g/L (= 0.4 g/mL).  
Then substrate flow = feed_rate[mL/h] × 0.4 [g/mL], and qs is computed as substrate flow divided by biomass mass.

![](Fig4_qs_est_trajectories.png)

### 5.3 IB fraction vs estimated qs across all induction sample points
This plot explains why “qs correlations” are misleading here: points cluster into saturated regimes.

![](Fig8_IB_vs_qs_samples.png)

---

## 6. Correct statistical interpretation (and what was wrong before)

This section explicitly answers: *“are the used statistical measures correct, and do the data support the conclusions?”*

### 6.1 Correlation matrix on 7 endpoints (previous report)
A correlation matrix on a 7-run, partially-replicated design with saturated outputs is **not an inferential statistic**.

- It will show very high correlation between Temperature and IB fraction simply because the design includes 25/35 extremes.
- It cannot disentangle qs effects because the design does not provide data in the sensitive region where qs could matter.
- Correlation does not handle the “step-like” / bounded nature of IB fraction well.

**Correct use:** as a *descriptive visualization only*, with explicit caveats.

### 6.2 Two-way ANOVA in the previous report
The previous report’s two-way ANOVA output is **not statistically valid** for this dataset as executed, because:

1. **No replication at the corner cells** → residual variance is not estimable from independent replicates there.  
2. If time points were used as replicates, that is **pseudo-replication** (time points within one run are not independent experimental units).  
3. The response is **bounded (0–100%)** and saturated → linear-normal ANOVA assumptions are violated.

Therefore, claims like “p < 0.001” or “R² ≈ 0.97” for factor effects are **not supported** by a valid model of the experiment.

### 6.3 Within-run “correlations” (previous report panels C/D in Figure 3)
At 25 °C and 35 °C, IB fraction is constant across induction samples, so variance is zero → correlation coefficients are undefined (NaN).  
This is not a bug; it means you cannot learn a relationship from a constant response.

### 6.4 “Fisher information” / “identifiability” language (previous report)
Fisher information requires:
- a **parameterized model** (e.g., logistic response curve vs T and qs), and
- an explicit **measurement noise model**.

With only three temperatures and saturation at the extremes, the current dataset supports only:

- **Practical identifiability of temperature regime** (yes),
- **Non-identifiability of qs effect near transition** (because untested).

It does **not** support a meaningful “FIM sensitivity curve” in the mechanistic sense.

---

## 7. Scientifically sound conclusions

1. **Temperature drives solubility regime** in this dataset: 25 °C → soluble, 35 °C → IB, 30 °C → mostly soluble with small IB fraction.
2. **qs_target does not change the regime at the extremes** (25 and 35), but this is a weak statement because the response is saturated there.
3. **qs effects near the transition cannot be concluded** (design limitation).
4. Total protein differs by run; without replication at the corners, treat these as descriptive observations only:

![](Fig9_endpoint_total_vs_temp.png)

---

## 8. What to do next (power + design that actually answers the qs question)

The only replicated condition is the center (30 °C, qs_target=0.4, n=3).  
Endpoint IB fraction SD at 30 °C is:

- **SD ≈ 1.64 percentage points**

Using that as a local noise estimate, the approximate minimum detectable qs effect at 30 °C (two-sided α=0.05, 80% power) is:

| n_per_group | min_detectable_delta_pct |
| --- | --- |
| 1 | 6.48 |
| 2 | 4.58 |
| 3 | 3.74 |
| 5 | 2.90 |
| 8 | 2.29 |
| 10 | 2.05 |
| 15 | 1.67 |
| 20 | 1.45 |

### Recommended next DOE (minimal but informative)
To test qs properly, you need to vary qs in the **transition temperature band**:

- Temperature: 30, 31, 32, 33 °C (or narrower if feasible)
- qs_target: at least 0.2 and 0.6 (ideally include 0.4 as well)
- Replication: ≥3 per cell near transition

This design converts qs from “not identifiable” into “testable”.

---

## Appendix: Run-level endpoints table (for reference)

| run_name | Temp_set_C | qs_target | Temp_measured_C | t_end_h | qs_est_initial | qs_est_end | qs_est_decline_pct | P_tot_end_gL | P_sol_end_gL | P_IB_end_gL | Sol_frac_end_pct | IB_frac_end_pct | productivity_total_gL_h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ferm_02 | 25 | 0.200 | 25.010 | 8.250 | 0.196 | 0.145 | 26.124 | 3.087 | 3.087 | 0.000 | 100.000 | 0.000 | 0.374 |
| ferm_03 | 25 | 0.600 | 24.980 | 8.250 | 0.528 | 0.243 | 53.953 | 4.143 | 4.143 | 0.000 | 100.000 | 0.000 | 0.502 |
| ferm_05 | 30 | 0.400 | 30.015 | 8.217 | 0.355 | 0.210 | 40.778 | 4.403 | 3.861 | 0.542 | 87.690 | 12.310 | 0.536 |
| ferm_06 | 30 | 0.400 | 30.000 | 8.217 | 0.368 | 0.207 | 43.832 | 3.813 | 3.252 | 0.561 | 85.284 | 14.716 | 0.464 |
| ferm_07 | 30 | 0.400 | 30.005 | 8.200 | 0.371 | 0.222 | 40.079 | 4.426 | 3.913 | 0.513 | 88.411 | 11.589 | 0.540 |
| ferm_01 | 35 | 0.200 | 34.980 | 8.233 | 0.212 | 0.170 | 19.835 | 1.744 | 0.000 | 1.744 | 0.000 | 100.000 | 0.212 |
| ferm_04 | 35 | 0.600 | 34.997 | 8.250 | 0.502 | 0.237 | 52.796 | 2.848 | 0.000 | 2.848 | 0.000 | 100.000 | 0.345 |

---

## Generated figures
- `Fig1_design_matrix.png`
- `Fig2_IB_fraction_timecourse.png`
- `Fig3_total_protein_timecourse.png`
- `Fig4_qs_est_trajectories.png`
- `Fig5_endpoint_IB_vs_temp.png`
- `Fig6_endpoint_partitioning.png`
- `Fig7_power_curve_30C.png`
- `Fig8_IB_vs_qs_samples.png`
- `Fig9_endpoint_total_vs_temp.png`
