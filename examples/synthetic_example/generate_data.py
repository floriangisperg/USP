"""Generate the synthetic fermentation dataset for this example.

Writes ``data/rawdata.csv`` (online, DASGIP-style headers, German CSV
format) and ``data/samples.xlsx`` (offline measurements) from a simple
process model — deterministic (seeded RNG), so regenerating reproduces
the committed files byte-for-byte.

Run from anywhere::

    python generate_data.py
"""

import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2026-01-15 09:00")
DT_MIN = 2.0
T_END_H = 28.0
T_PREP_H = 0.25
V0_L = 1.5
X0 = 0.5
MU_BATCH = 0.28
MU_FED = 0.08
Y_XS = 0.45
G0 = 20.0
S_FEED_G_ML = 0.4  # 400 g/L glucose feed
FEED_DENSITY = 1.15  # g/mL, matches run_config.yaml feed_definitions
BALANCE_START_G = 800.0
RQ = 1.0
Y_O2_CMOL = 1.0  # mol O2 per C-mol biomass
MW_X = 26.565  # g per C-mol biomass
F_AIR = 1.0  # L/min
MOLAR_VOLUME = 22.414

rng = np.random.default_rng(SEED)


# Model biomass as piecewise exponential (batch / fed), matching the phase
# switches below. Vessel volume for gas rates also grows with feed — the
# pipeline recomputes it, we only need plausible offgas %.
def biomass(t_h):
    x = np.where(
        t_h < 10.0,
        X0 * np.exp(MU_BATCH * np.clip(t_h, 0, None)),
        X0 * np.exp(MU_BATCH * 10.0) * np.exp(MU_FED * (t_h - 10.0)),
    )
    return np.where(t_h < 0, X0 * 0.2, x)  # pre-inoculation: low carryover


def dxdt(t_h):
    mu = np.where(t_h < 10.0, MU_BATCH, MU_FED)
    return mu * biomass(t_h)


def main() -> None:
    # --- time grid ---------------------------------------------------------
    t = np.arange(-T_PREP_H, T_END_H + DT_MIN / 60, DT_MIN / 60.0)
    stamps = START + pd.to_timedelta(t, unit="h")
    phase = np.where(
        t < 0,
        "Preparation",
        np.where(t < 10, "Batch", np.where(t < 16, "Fedbatch", "induced Fedbatch")),
    )
    x = biomass(t)

    # --- balance: flat until feed start, then decreasing by cumulative feed
    # Feed supplies glucose for growth at yield Y_XS; the balance senses the
    # pumped MASS (mL/h * feed density), the vessel volume the pumped VOLUME.
    feed_glu_g_h = np.maximum(dxdt(t), 0) * V0_L / Y_XS  # glucose g/h
    feed_ml_h = feed_glu_g_h / S_FEED_G_ML  # 400 g/L feed
    feed_rate_g_h = np.where(t >= 10, feed_ml_h * FEED_DENSITY, 0.0)
    dt_h = DT_MIN / 60
    feed_cum_g = np.cumsum(feed_rate_g_h) * dt_h
    feed_cum_ml = feed_cum_g / FEED_DENSITY
    noise_bal = rng.normal(0, 0.05, t.size)
    balance = BALANCE_START_G - feed_cum_g + noise_bal

    # --- gas: OUR/CER from growth, converted to exit % at constant air flow
    our_mmol_l_h = np.where(t >= 0, dxdt(t) / MW_X * Y_O2_CMOL * 1000, 0.0)
    volume_l = V0_L + feed_cum_ml / 1000
    f_mol_h = F_AIR * 60 / MOLAR_VOLUME
    d_o2_pct = our_mmol_l_h * volume_l / (f_mol_h * 1000) * 100 * 1.02
    cer_mmol_l_h = our_mmol_l_h * RQ
    d_co2_pct = cer_mmol_l_h * volume_l / (f_mol_h * 1000) * 100
    exit_o2 = 20.9 - d_o2_pct + rng.normal(0, 0.03, t.size)
    exit_co2 = 0.04 + d_co2_pct + rng.normal(0, 0.02, t.size)

    # --- offline-measurable online signals ---------------------------------
    po2 = np.where(
        (t > 9.2) & (t < 10.4), 12.0, np.where(t < 0, 45.0, 38.0)
    ) + rng.normal(0, 2.0, t.size)
    ph = np.where(t < 16, 7.0, 7.1) + rng.normal(0, 0.02, t.size)
    od600 = x / 0.35 * (1 + rng.normal(0, 0.03, t.size))
    od600 = np.clip(od600, 0.05, None)

    # --- pumps --------------------------------------------------------------
    dur = np.maximum(t, 0) * 3600  # cumulative pump seconds
    pump2_dur = np.where(t < 0, np.nan, dur)
    pump4_dur = np.where(t < 10, np.nan, (t - 10) * 3600)

    online = pd.DataFrame(
        {
            "Date Time UTC": stamps.strftime("%d.%m.%Y %H:%M"),
            "Batch Time, sec": (t * 3600).round(0).astype(int),
            "Phase": phase,
            "Phase Time, sec": (dur % 3600).astype(int),
            "AirFlow, l/min": np.round(F_AIR + rng.normal(0, 0.02, t.size), 3),
            "TotalFlow, l/min": np.round(F_AIR + rng.normal(0, 0.02, t.size), 3),
            "GasMix": 21,
            "Pump2.Duration, -": np.round(pump2_dur, 0),
            "Pump4.Duration, -": np.round(pump4_dur, 0),
            "Stirrer, 1/min": np.round(1200 + rng.normal(0, 8, t.size), 0),
            "Temperature, °C": np.round(37.0 + rng.normal(0, 0.05, t.size), 2),
            "Balance, g": np.round(balance, 3),
            "pH, -": np.round(ph, 3),
            "pO2, %": np.round(po2, 1),
            "GenericGasAnalyser.ExitCO2, %": np.round(exit_co2, 3),
            "GenericGasAnalyser.ExitO2, %": np.round(exit_o2, 3),
            "OD600": np.round(od600, 3),
        }
    )

    # --- write German-format CSV (decimal comma, ';' separator, preamble) --
    def de(col):
        s = online[col].copy()
        if s.dtype.kind in "fc":
            return s.map(lambda v: "" if pd.isna(v) else f"{v:.3f}".replace(".", ","))
        return s

    out = pd.DataFrame({c: de(c) for c in online.columns})
    # Preamble line + header => matches parsing.skiprows: 1 in the defaults
    with open("data/rawdata.csv", "w", encoding="utf-8", newline="") as f:
        f.write("DASGIP-Export: synthetic fermenter SIM01 (generated data)\n")
        out.to_csv(f, sep=";", index=False)

    # --- offline samples -----------------------------------------------------
    s_t = np.array([1.5, 4.0, 7.0, 9.5, 12.0, 15.0, 19.0, 23.0, 26.5])
    s_x = biomass(s_t) * (1 + rng.normal(0, 0.04, s_t.size))
    s_glu = np.where(s_t < 10, np.maximum(G0 - (s_x - X0) / Y_XS, 0.15), 0.3)
    s_glu = s_glu * (1 + rng.normal(0, 0.08, s_t.size))
    s_ace = np.clip(
        0.35 * np.maximum(0, s_t - 6) / 4 - 0.15 * np.maximum(0, s_t - 12), 0, 1.2
    )
    s_prot_t = np.where(s_t > 16, 0.12 * (s_x - biomass(16.0)), 0.0)
    offline = pd.DataFrame(
        {
            "Process Time [h]": s_t,
            "OD": np.round(s_x / 0.35, 2),
            "Biomass [g/L]": np.round(s_x, 3),
            "Glucose [g/L]": np.round(s_glu, 3),
            "Acetate [g/L]": np.round(s_ace, 3),
            "Protein concentration [g/L]": np.round(s_prot_t, 3),
            "Protein soluble [g/L]": np.round(s_prot_t * 0.6, 3),
            "Protein IB [g/L]": np.round(s_prot_t * 0.4, 3),
            "Sample volume [mL]": 10.0,
        }
    )
    offline.to_excel("data/samples.xlsx", index=False)
    print(
        f"wrote data/rawdata.csv ({len(online)} rows) and data/samples.xlsx ({len(offline)} samples)"
    )


if __name__ == "__main__":
    main()
