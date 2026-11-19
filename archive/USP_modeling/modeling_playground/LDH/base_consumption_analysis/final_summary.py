"""
BASE CONSUMPTION SUMMARY - FINAL

Key limitation: No direct calibration available for current pump.
Results below use an old calibration of unknown applicability.

OLD CALIBRATION DATA:
- Pump duration: 0 → 79,065 units
- Mass: 0 → 308.39 g
- Factor: 0.00390 g/pump-unit

CAVEAT: If old pump has different flow rate, these estimates may be
significantly off (could be 2x higher or lower). Treat as rough estimates only.
"""

import pandas as pd
from pathlib import Path

# Old calibration factor
OLD_CALIBRATION_G_PER_UNIT = 308.39 / 79065  # 0.00390 g/pump-unit

# Base solution properties (from run_config.yaml)
BASE_DENSITY_G_ML = 0.947
BASE_MOLARITY_MOL_L = 6.95
MOLAR_MASS_NH3 = 17.031

# Load data
BASE_DIR = Path(r"C:\Users\fgisperg\OneDrive - TU Wien\01_Uni\01_PhD\05_Projects\USP\examples\LDH")
runs = ['ferm_02', 'ferm_03', 'ferm_04', 'ferm_05', 'ferm_06', 'ferm_07']

results = []
for run in runs:
    df = pd.read_excel(BASE_DIR / run / 'results' / 'processed_data.xlsx')

    pump_dur_initial = df['Base pump duration [-]'].iloc[0]
    pump_dur_final = df['Base pump duration [-]'].iloc[-1]
    pump_dur_change = pump_dur_final - pump_dur_initial

    biomass = df['Biomass [g/L]'].max()
    feed_g = df['Feed [g]'].sum()

    # Using old calibration
    base_mass_g = pump_dur_change * OLD_CALIBRATION_G_PER_UNIT
    base_vol_ml = base_mass_g / BASE_DENSITY_G_ML
    nh3_mol = (base_vol_ml / 1000) * BASE_MOLARITY_MOL_L

    results.append({
        'run': run,
        'pump_duration_initial': pump_dur_initial,
        'pump_duration_final': pump_dur_final,
        'pump_duration_change': pump_dur_change,
        'biomass_final_gL': biomass,
        'feed_total_g': feed_g,
        'base_mass_g_est': base_mass_g,
        'base_vol_ml_est': base_vol_ml,
        'nh3_mol_est': nh3_mol,
    })

df = pd.DataFrame(results)

print("="*80)
print("BASE CONSUMPTION SUMMARY")
print("="*80)
print("\nWARNING: Using old calibration of unknown applicability.")
print("Treat values as rough estimates only. True values could differ significantly.\n")

print("RAW DATA (no calibration assumption needed):")
print("-"*80)
print(df[['run', 'pump_duration_change', 'biomass_final_gL', 'feed_total_g']].to_string(index=False))

print("\n\nESTIMATED VALUES (using old calibration):")
print("-"*80)
summary = df[['run', 'base_vol_ml_est', 'nh3_mol_est']].copy()
summary.columns = ['Run', 'Base Volume [mL] (est.)', 'NH3 [mol] (est.)']
print(summary.to_string(index=False))

print("\n\nTOTALS:")
print("-"*80)
print(f"Total base volume (est.): {df['base_vol_ml_est'].sum():.1f} mL")
print(f"Total NH3 (est.): {df['nh3_mol_est'].sum():.2f} mol")
print(f"Average per run: {df['nh3_mol_est'].mean():.2f} mol NH3")

# Save to file
OUTPUT_DIR = BASE_DIR / "base_consumption_analysis"

# Save raw data (reliable)
df[['run', 'pump_duration_change', 'biomass_final_gL', 'feed_total_g']].to_excel(
    OUTPUT_DIR / "base_pump_duration_raw.xlsx", index=False)

# Save estimated values (with caveat)
df[['run', 'base_vol_ml_est', 'nh3_mol_est']].to_excel(
    OUTPUT_DIR / "base_consumption_estimates_old_calibration.xlsx", index=False)

# Save full summary
df.to_excel(OUTPUT_DIR / "base_consumption_full_summary.xlsx", index=False)

print(f"\n\nFiles saved to {OUTPUT_DIR}:")
print("  - base_pump_duration_raw.xlsx           (Raw pump duration - reliable)")
print("  - base_consumption_estimates_old_calibration.xlsx  (Estimates - uncertain)")
print("  - base_consumption_full_summary.xlsx    (All data)")
