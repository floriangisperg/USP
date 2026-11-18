"""Prepare the hGH example data (runs R5_lowlow / R6_highhigh).

Run first, before ``analyze_hgh_run.py``:

    python prepare_hgh_data.py

What it does (idempotent; re-running overwrites the generated files):

1. **Caps both runs at the successful inoculation**: the first inoculation
   failed (wrong pre-culture pH, see 00_Weekly_Summary.txt); both vessels
   were re-inoculated ~170 mL of OD 5 pre-culture on 10-21 late morning
   (R5 11:41, R6 11:52 - Samplingsheet_*.pdf). Everything before that is
   dead holding time (the ~24 h cold 18 degC phase) and is dropped;
   'Process time [h]' zero = successful inoculation. The failed-batch
   residue sample #1 (R5 0.1 g/L, R6 6.9 g/L, drawn just before
   re-inoculation) falls outside the window and is excluded from
   samples.xlsx. The uncapped raw exports stay untouched at the example
   root.
2. **R5_lowlow**: the R5 export is split across two files (a program
   restart at the Batch->Fedbatch seam), and its balance channel reads all
   zeros (feed bottle scale never wired into the export). This script
   - merges the two files on wall-clock time (dropping the stray
     Preparation row at the seam),
   - rebuilds ``Batch Time, sec`` / ``Batch Time (since inoc.), sec`` from
     the local timestamp (the restart zeroes the original counters),
   - reconstructs a *synthetic* ``Balance, g`` series: feed-start anchor
     weight from the manual scale readings (``FeedpumpCalibration.xlsx`` /
     ``Feed_Waagenwerte_R5``), then feed mass integrated from ``Pump4, %``
     times the pump calibration slope, rescaled by a single factor fitted
     to the manual scale anchor deltas (the fresh-water calibration
     overestimates in-process delivery by ~10-13% due to backpressure).
   R6 has a real, scale-verified balance and is only truncated + re-timed.
3. Builds per-run ``data/samples.xlsx`` from the analytics workbook
   (``20261109_Analytik_highhigh_lowlow.xlsx``): RP-HPLC soluble/IB hGH,
   biomass, sample volumes per the sampling sheets. Sample process times =
   successful-inoculation offset + workbook "hours after induction".
4. Prints diagnostics: induction offsets for the run configs, balance fit
   residuals, feed totals vs. manual scale readings.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# --- Raw files (kept at the example root as source of truth) ---
RAW_R6 = HERE / "SNO_hgh_highhigh.csv"
RAW_R5_BATCH = HERE / "SNO_hgh_lowlow.csv"
RAW_R5_FEDBATCH = HERE / "SNO_hgh_lowlow_FedBatch.csv"
ANALYTICS_XLSX = HERE / "20261109_Analytik_highhigh_lowlow.xlsx"
CALIBRATION_XLSX = HERE / "FeedpumpCalibration.xlsx"

RUN_DIRS = {"R5": HERE / "R5_lowlow", "R6": HERE / "R6_highhigh"}

# SNO export dialect (matches the parsing overrides in both run_config.yaml)
SEP = ";"
DATE_FMT = "%m/%d/%Y %I:%M:%S %p"

# Operator-verified IPTG bolus moments (wall clock, local time; 2026-10-22).
# The SCADA flips into 'inducedFedbatch' LEAD these by 16 min (R5) / 40 min
# (R6), so 'processing.induction_time_h' in the run configs must use these.
INDUCTION_WALLCLOCK = {
    "R5": pd.Timestamp("2026-10-22 11:15:00"),
    "R6": pd.Timestamp("2026-10-22 11:18:00"),
}

# Successful re-inoculation moments (wall clock, local time; 2026-10-21) from
# Samplingsheet_R5_lowlow.pdf / Samplingsheet_R6_highhigh.pdf. Both exports
# are CAPPED here: process-time zero = successful inoculation, the failed
# first batch (~24 h cold holding) is dropped. These are also the time zero
# for every sample process time and for 'processing.induction_time_h'.
PROCESS_TIME_ZERO = {
    "R5": pd.Timestamp("2026-10-21 11:41:00"),
    "R6": pd.Timestamp("2026-10-21 11:52:00"),
}

SAMPLE_VOLUME_ML_DEFAULT = 20.0  # 'Probenvolumen Reaktor' for the analytics samples

# Batch-phase samples #2/#3, transcribed from Samplingsheet_R5_lowlow.pdf /
# Samplingsheet_R6_highhigh.pdf. Process hours relative to PROCESS_TIME_ZERO
# (the successful re-inoculation). Sample #1 (drawn just BEFORE re-inoculation,
# biomass 0.1 g/L R5 / 6.9 g/L R6 = residue of the failed first batch) is
# outside the capped window and intentionally excluded; the KW43 workbook
# still holds its triplicates if ever needed.
BATCH_SAMPLES = {
    "R5": [  # zero = 2026-10-21 11:41 (re-inoculation)
        {"sample": "Sample 2", "t_process_h": 5.8167, "volume_ml": 10.0, "od": 20.2},
        {"sample": "Sample 3", "t_process_h": 20.1500, "volume_ml": 10.0, "od": 45.8},
    ],
    "R6": [  # zero = 2026-10-21 11:52 (re-inoculation)
        {"sample": "Sample 2", "t_process_h": 6.1333, "volume_ml": 10.0, "od": 19.2},
        {"sample": "Sample 3", "t_process_h": 22.0000, "volume_ml": 10.0, "od": 46.7},
    ],
}

# Analytics samples #4-#7: '2x10 mL Product + 5 mL DCW (~30 mL)' per the
# sampling sheets (operator's total estimate; the analytics workbook's
# 'Probenvolumen Reaktor' = 20 mL is only the product-vial volume used in its
# protein calcs - the pipeline's sample volume is the physical withdrawal).
ANALYTICS_SAMPLE_VOLUME_ML = 30.0

DCW_XLSX = HERE / "20261020_DCW_KW43.xlsx"

SAMPLES_COLUMNS = [
    "Process Time [h]",
    "OD",
    "Glucose [g/L]",
    "Acetate [g/L]",
    "Protein IB [g/L]",
    "Protein soluble [g/L]",
    "Protein concentration [g/L]",
    "Biomass [g/L]",
    "Viability [%]",
    "Sample volume [mL]",
]

REACTOR_DIRS = {"R5": "R5", "R6": "R6"}  # analytics sheets -> RUN_DIRS keys


def fmt_num(x: float) -> str:
    """Compact number formatting for rebuilt CSV cells (ints stay ints)."""
    if x == int(x):
        return str(int(x))
    return f"{x:.2f}"


def cap_at_inoculation(df: pd.DataFrame, t_cap: pd.Timestamp, label: str):
    """Truncate an export at the successful re-inoculation and re-time it.

    Drops everything before ``t_cap`` (the failed first batch + cold holding),
    rebuilds both Batch-time counters from the local timestamp so the cleaner
    anchors 'Process time [h]' at the first kept row (= process-time zero).
    Returns (capped_df, n_dropped).
    """
    dt = pd.to_datetime(df["Date Local Time"], format=DATE_FMT)
    n_before = int((dt < t_cap).sum())
    df = df[dt >= t_cap].copy().reset_index(drop=True)
    dt = pd.to_datetime(df["Date Local Time"], format=DATE_FMT)
    assert (
        df["Phase"].str.strip() == "Preparation"
    ).sum() == 0, f"{label}: Preparation rows after the cap - cap time is too early"
    df["Batch Time, sec"] = [fmt_num(v) for v in (dt - dt.iloc[0]).dt.total_seconds()]
    df["Batch Time (since inoc.), sec"] = [
        fmt_num(v) for v in (dt - dt.iloc[0]).dt.total_seconds()
    ]
    return df, n_before


def write_export(df: pd.DataFrame, version_line: str, out_path: Path) -> None:
    """Write a processed export in the original SNO dialect (version line first)."""
    with open(out_path, "w", encoding="cp1252", newline="") as fh:
        fh.write(version_line + "\n")
        df.to_csv(fh, sep=SEP, index=False, na_rep="", lineterminator="\n")


def read_export(path: Path) -> pd.DataFrame:
    """Read one SNO export keeping every cell as its original string."""
    return pd.read_csv(
        path, sep=SEP, skiprows=1, dtype=str, keep_default_na=False, low_memory=False
    )


def load_slope_g_per_pct(sheet: pd.DataFrame) -> float:
    """Pump calibration slope (g/h per %), least squares through the origin.

    Sheet layout 'R5'/'R6': col 0 = Speed%, col 5 = 'g Feed/h'. The workbook
    also stores its own slope (col 7); we refit so the method is explicit.
    """
    pts = sheet.iloc[:, [0, 5]].apply(pd.to_numeric, errors="coerce").dropna()
    pts.columns = ["pct", "g_per_h"]
    pts = pts[(pts["pct"] > 0) & (pts["g_per_h"] > 0)]
    slope = float((pts["pct"] * pts["g_per_h"]).sum() / (pts["pct"] ** 2).sum())
    return slope


def load_scale_anchors(sheet: pd.DataFrame) -> pd.DataFrame:
    """Manual feed-bottle scale readings: timestamp + weight.

    Sheet layout 'Feed_Waagenwerte_R5'/'..._R6': col 0 = Date (only on the
    first row), col 1 = time of day, col 4 = Gewicht (g). Rows without an
    explicit time of day are skipped (ambiguous 'hours since start' formats).
    """
    s = sheet.iloc[:, [0, 1, 4]].copy()
    s.columns = ["date", "time", "weight"]
    s["date"] = s["date"].ffill()
    rows = []
    for _, r in s.iterrows():
        t_str, w = str(r["time"]).strip(), str(r["weight"]).strip()
        if not t_str or not re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", t_str):
            continue
        d_str = str(r["date"]).strip()[:10]
        try:
            ts = pd.Timestamp(f"{d_str} {t_str}")
            rows.append({"time": ts, "weight": float(w)})
        except ValueError:
            continue
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def prepare_r5() -> None:
    """Merge the two R5 exports, rebuild time columns, synthesize the balance."""
    out_dir = RUN_DIRS["R5"] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_batch = read_export(RAW_R5_BATCH)
    df_fb = read_export(RAW_R5_FEDBATCH)

    # --- Merge on wall clock -------------------------------------------------
    # The FedBatch file restarts the SCADA program: drop its stray Preparation
    # row (the pipeline would filter it anyway) and re-anchor all time columns.
    n_prep = int((df_fb["Phase"].str.strip() == "Preparation").sum())
    df_fb = df_fb[df_fb["Phase"].str.strip() != "Preparation"].copy()

    for df in (df_batch, df_fb):
        df["_t"] = pd.to_datetime(df["Date Local Time"], format=DATE_FMT)

    merged = pd.concat([df_batch, df_fb], ignore_index=True)
    merged = merged.sort_values("_t").reset_index(drop=True)
    dup = merged["_t"].duplicated().sum()
    if dup:
        raise SystemExit(f"R5 merge: {dup} duplicate timestamps found - aborting.")
    gap = merged["_t"].diff().max()
    print(
        f"R5 merge: {len(df_batch)} + {len(df_fb) - n_prep} rows "
        f"(dropped {n_prep} stray Preparation rows); "
        f"largest timestamp gap {gap}"
    )

    # --- Cap at the successful re-inoculation --------------------------------
    merged, n_dropped = cap_at_inoculation(merged, PROCESS_TIME_ZERO["R5"], "R5")
    t_cap = pd.to_datetime(merged["Date Local Time"], format=DATE_FMT).iloc[0]
    print(
        f"R5 capped at successful re-inoculation {t_cap}: "
        f"dropped {n_dropped} rows of the failed first batch "
        f"({len(merged)} rows kept)"
    )
    merged["_t"] = pd.to_datetime(merged["Date Local Time"], format=DATE_FMT)

    # --- Synthetic balance ----------------------------------------------------
    # R5's export has no balance signal (all zeros). Feed clearly happened
    # (Pump4 ran ~15.5 h); the feed bottle sat on a manual scale, recorded in
    # FeedpumpCalibration.xlsx. Reconstruct:
    #   Balance(t) = w_start - f * integral(Pump4% * slope) dt
    # with w_start the manual reading at feed start and f a single scale
    # factor fitted to all manual anchor deltas (fresh-water calibration
    # overestimates in-process delivery through backpressure).
    calib = pd.read_excel(CALIBRATION_XLSX, sheet_name="R5", header=0)
    slope = load_slope_g_per_pct(calib)
    anchors = load_scale_anchors(
        pd.read_excel(CALIBRATION_XLSX, sheet_name="Feed_Waagenwerte_R5", header=0)
    )
    w_start, t_start = anchors.iloc[0]["weight"], anchors.iloc[0]["time"]
    print(
        f"R5 pump calibration slope {slope:.4f} g/h/%; "
        f"{len(anchors)} manual scale readings, start {w_start} g at {t_start}"
    )

    fb_row = merged.index[merged["Phase"].str.strip() == "Fedbatch"][0]
    t_feed_start = merged.loc[fb_row, "_t"]
    if abs((t_feed_start - t_start).total_seconds()) > 120:
        print(
            f"WARNING: SCADA feed start {t_feed_start} vs manual anchor "
            f"{t_start} differ by more than 2 min."
        )

    pump4 = pd.to_numeric(merged["Pump4, %"], errors="coerce").fillna(0.0)
    before = merged["_t"] < t_feed_start
    rate = pump4 * slope  # g/h
    rate[before] = 0.0  # batch-phase % is setpoint echo, no delivery
    dt_h = merged["_t"].diff().dt.total_seconds().div(3600).fillna(0.0)
    cum_feed = (rate * dt_h).cumsum()  # g pumped since feed start
    cum_since_start = cum_feed - cum_feed.iloc[fb_row]

    # Fit f on the post-start anchors: minimize sum((w_start - f*cum - w)^2)
    a_times, a_weights = anchors["time"].iloc[1:], anchors["weight"].iloc[1:]
    cum_at = np.array(
        [
            np.interp(t.value, merged["_t"].values.astype(np.int64), cum_since_start)
            for t in a_times
        ]
    )
    deltas = w_start - a_weights.to_numpy()
    f_fit = float((cum_at * deltas).sum() / (np.square(cum_at).sum()))
    resid = f_fit * cum_at - deltas
    print(
        f"R5 balance fit: scale factor f = {f_fit:.4f} "
        f"(in-process slope = {f_fit * slope:.3f} g/h/% vs {slope:.3f} calibrated)"
    )
    for t, d, e in zip(a_times, deltas, resid):
        model_d = np.interp(
            t.value, merged["_t"].values.astype(np.int64), cum_since_start
        )
        print(
            f"  anchor {t}: manual -{d:6.1f} g, model -{f_fit * model_d:6.1f} g, "
            f"resid {e:+.1f} g"
        )

    balance = w_start - f_fit * cum_since_start
    merged["Balance, g"] = [fmt_num(v) for v in balance]

    # --- Write merged export (keep the version line first) -------------------
    version_line = RAW_R5_BATCH.read_text(encoding="cp1252").splitlines()[0]
    out_path = out_dir / "SNO_hgh_lowlow_merged.csv"
    write_export(merged.drop(columns="_t"), version_line, out_path)
    print(f"R5 merged export written: {out_path.relative_to(HERE)}")

    # --- Diagnostics vs. the manual end reading ------------------------------
    total_model = f_fit * cum_since_start.iloc[-1]
    total_manual = w_start - anchors.iloc[-1]["weight"]
    print(
        f"R5 total feed at end of export: model -{total_model:.1f} g vs "
        f"manual scale -{total_manual:.1f} g "
        f"({anchors.iloc[-1]['time']}, export ends {merged['_t'].iloc[-1]})"
    )

    # Induction offset for run_config.yaml (relative to the capped zero)
    offset_h = (
        INDUCTION_WALLCLOCK["R5"] - PROCESS_TIME_ZERO["R5"]
    ).total_seconds() / 3600.0
    fb_idx = merged.index[merged["Phase"].str.strip() == "inducedFedbatch"][0]
    flip = (merged.loc[fb_idx, "_t"] - PROCESS_TIME_ZERO["R5"]).total_seconds() / 3600.0
    print(
        f"R5 induction_time_h = {offset_h:.4f} "
        f"(bolus {INDUCTION_WALLCLOCK['R5']}; SCADA flip at {flip:.4f} h, "
        f"delta {offset_h - flip:+.2f} h)\n"
    )


def prepare_r6() -> None:
    """Cap the continuous R6 export at the successful inoculation."""
    out_dir = RUN_DIRS["R6"] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_export(RAW_R6)

    # Cross-check the recorded balance against the manual scale readings
    df["_t"] = pd.to_datetime(df["Date Local Time"], format=DATE_FMT)
    balance = pd.to_numeric(df["Balance, g"], errors="coerce")
    anchors = load_scale_anchors(
        pd.read_excel(CALIBRATION_XLSX, sheet_name="Feed_Waagenwerte_R6", header=0)
    )
    print(
        f"R6 manual scale readings: {len(anchors)}, "
        f"start {anchors.iloc[0]['weight']} g, end {anchors.iloc[-1]['weight']} g"
    )
    print(
        f"R6 export balance: starts {balance.iloc[0]:.1f} g, "
        f"ends {balance.iloc[-1]:.1f} g (drop "
        f"{balance.iloc[0] - balance.iloc[-1]:.1f} g)"
    )

    # Cap at the successful re-inoculation; keep the real balance column
    df, n_dropped = cap_at_inoculation(df, PROCESS_TIME_ZERO["R6"], "R6")
    df["_t"] = pd.to_datetime(df["Date Local Time"], format=DATE_FMT)
    print(
        f"R6 capped at successful re-inoculation {df['_t'].iloc[0]}: "
        f"dropped {n_dropped} rows of the failed first batch "
        f"({len(df)} rows kept)"
    )

    version_line = RAW_R6.read_text(encoding="cp1252").splitlines()[0]
    out_path = out_dir / RAW_R6.name
    write_export(df.drop(columns="_t"), version_line, out_path)
    print(f"R6 export written: {out_path.relative_to(HERE)}")

    # Induction offset for run_config.yaml (relative to the capped zero)
    offset_h = (
        INDUCTION_WALLCLOCK["R6"] - PROCESS_TIME_ZERO["R6"]
    ).total_seconds() / 3600.0
    fb_idx = df.index[df["Phase"].str.strip() == "inducedFedbatch"]
    flip = (df.loc[fb_idx[0], "_t"] - PROCESS_TIME_ZERO["R6"]).total_seconds() / 3600.0
    print(
        f"R6 induction_time_h = {offset_h:.4f} "
        f"(bolus {INDUCTION_WALLCLOCK['R6']}; SCADA flip at {flip:.4f} h, "
        f"delta {offset_h - flip:+.2f} h)\n"
    )


def load_dcw_means(reactor: str) -> pd.DataFrame:
    """Mean DCW (g/L) per sample from 20261020_DCW_KW43.xlsx triplicates.

    Sheet 'Reaktor5': only 'empty'/'full' Eppi weights (1 mL volume assumed,
    matching the analytics workbook's 'Volumen (mL) = 1'). Sheet 'Reaktor6':
    'Tara [g]'/'Auswaage [g]'. Zelldichte = (full - empty) * 1000 for 1 mL.
    """
    sheet = pd.read_excel(DCW_XLSX, sheet_name=f"Reaktor{reactor[-1]}", header=0)
    sheet = sheet.iloc[:, :3]
    sheet.columns = ["eppi", "empty", "full"]
    sheet["empty"] = pd.to_numeric(sheet["empty"], errors="coerce")
    sheet["full"] = pd.to_numeric(sheet["full"], errors="coerce")
    sheet = sheet.dropna(subset=["empty", "full"])
    m = sheet["eppi"].astype(str).str.match(r"^\d+[A-C]$")
    sheet = sheet[m].copy()
    sheet["n"] = sheet["eppi"].str.extract(r"^(\d+)").astype(int)
    sheet["g_l"] = (sheet["full"] - sheet["empty"]) * 1000.0
    means = sheet.groupby("n")["g_l"].agg(["mean", "count"]).reset_index()
    means.columns = ["n", "biomass", "n_eppis"]
    if not (means["n_eppis"] == 3).all():
        print(
            f"WARNING {reactor}: DCW triplicates incomplete: "
            f"{means[['n', 'n_eppis']].to_dict('records')}"
        )
    means["sample"] = "Sample " + means["n"].astype(str)
    return means[["sample", "biomass"]]


def build_samples(reactor: str) -> None:
    """Build data/samples.xlsx for one run from the analytics workbook.

    RP-HPLC 'Menge im Reaktor (g/L)' (Calculations_Protein sheet) feeds the
    Protein soluble/IB columns; biomass from Biomass_Summary. Sample process
    time = operator bolus (as process-time h) + workbook 'hours after
    induction'. Run ``prepare`` first: it prints the exact offsets used here.
    """
    # Column positions in Calculations_Protein:
    #   0 Reaktor | 2 Sampletype | 3 Sample # | 4 h after induction
    #   5 Probenvolumen Reaktor | 9 Menge im Reaktor (g/L)
    calc = pd.read_excel(ANALYTICS_XLSX, sheet_name="Calculations_Protein", header=0)
    calc = calc.iloc[:, [0, 2, 3, 4, 5, 9]]
    calc.columns = ["reactor", "sampletype", "sample", "h_after_indu", "vol_ml", "g_l"]
    # Fill only the grouping columns; blank separator rows then lack values and
    # are dropped below instead of becoming phantom duplicates of the last
    # sample of each block.
    for col in ("reactor", "sampletype"):
        calc[col] = calc[col].ffill()
    calc = calc[calc["sample"].astype(str).str.match(r"Sample \d+$")].copy()
    calc = calc.dropna(subset=["h_after_indu", "g_l"])
    for col in ("h_after_indu", "vol_ml", "g_l"):
        calc[col] = pd.to_numeric(calc[col], errors="coerce")
    calc = calc[calc["reactor"] == reactor]
    if not (calc["vol_ml"].dropna() == SAMPLE_VOLUME_ML_DEFAULT).all():
        raise SystemExit(
            f"{reactor}: unexpected sample volume(s) in workbook: "
            f"{sorted(calc['vol_ml'].dropna().unique())}"
        )

    sol = calc[calc["sampletype"] == "soluable Protein"].set_index("sample")
    ib = calc[calc["sampletype"] == "IBs"].set_index("sample")
    if not sol.index.equals(ib.index):
        raise SystemExit(
            f"{reactor}: soluble/IB sample sets differ: "
            f"{list(sol.index)} vs {list(ib.index)}"
        )
    if not np.allclose(sol["h_after_indu"], ib["h_after_indu"]):
        raise SystemExit(f"{reactor}: soluble/IB 'hours after induction' differ.")

    # Biomass_Summary: 0 Reaktor | 2 Sample # | 4 Biomasse (g/L)
    bio = pd.read_excel(ANALYTICS_XLSX, sheet_name="Biomass_Summary", header=0)
    bio = bio.iloc[:, [0, 2, 4]]
    bio.columns = ["reactor", "sample", "biomass"]
    bio["reactor"] = bio["reactor"].ffill()
    bio = bio[bio["sample"].astype(str).str.match(r"Sample \d+$")].copy()
    bio = bio.dropna(subset=["biomass"])
    bio["biomass"] = pd.to_numeric(bio["biomass"], errors="coerce")
    bio = bio[bio["reactor"] == reactor].set_index("sample")
    if not sol.index.equals(bio.index):
        raise SystemExit(
            f"{reactor}: biomass sample sets differ: "
            f"{list(sol.index)} vs {list(bio.index)}"
        )

    # Cross-check: KW43 workbook means for the analytics samples (#4-#7)
    # against the Biomass_Summary values used above
    dcw = load_dcw_means(reactor).set_index("sample")
    chk = bio.join(dcw, lsuffix="_summary", rsuffix="_dcw")
    dev = (
        chk.loc[sol.index, "biomass_summary"] - chk.loc[sol.index, "biomass_dcw"]
    ).abs()
    print(
        f"{reactor} DCW cross-check (KW43 vs Biomass_Summary, #4-#7): "
        f"max |dev| = {dev.max():.2f} g/L"
    )

    # Recompute the induction offset the same way prepare_r5/r6 report it
    t_zero = PROCESS_TIME_ZERO[reactor]
    offset_h = (INDUCTION_WALLCLOCK[reactor] - t_zero).total_seconds() / 3600.0

    analytics = pd.DataFrame(
        {
            "Process Time [h]": offset_h + sol["h_after_indu"].to_numpy(),
            "OD": np.nan,
            "Glucose [g/L]": np.nan,
            "Acetate [g/L]": np.nan,
            "Protein IB [g/L]": ib["g_l"].to_numpy(),
            "Protein soluble [g/L]": sol["g_l"].to_numpy(),
            "Protein concentration [g/L]": (sol["g_l"] + ib["g_l"]).to_numpy(),
            "Biomass [g/L]": bio["biomass"].to_numpy(),
            "Viability [%]": np.nan,
            "Sample volume [mL]": ANALYTICS_SAMPLE_VOLUME_ML,
        }
    )

    # --- Batch-phase samples #1-#3 (DCW only, from the sampling sheets) ------
    batch = pd.DataFrame(BATCH_SAMPLES[reactor])
    dcw = load_dcw_means(reactor).set_index("sample")
    bio_vals = [dcw.loc[s, "biomass"] for s in batch["sample"]]
    samples = pd.DataFrame(
        {
            "Process Time [h]": batch["t_process_h"],
            "OD": batch["od"],
            "Glucose [g/L]": np.nan,
            "Acetate [g/L]": np.nan,
            "Protein IB [g/L]": np.nan,
            "Protein soluble [g/L]": np.nan,
            "Protein concentration [g/L]": np.nan,
            "Biomass [g/L]": bio_vals,
            "Viability [%]": np.nan,
            "Sample volume [mL]": batch["volume_ml"],
        }
    )
    samples = pd.concat([samples, analytics], ignore_index=True)
    samples = samples.sort_values("Process Time [h]").reset_index(drop=True)

    out_dir = RUN_DIRS[REACTOR_DIRS[reactor]] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        samples.to_excel(xw, sheet_name="Sheet1", index=False)
    print(
        f"{reactor} samples written: {out_path.relative_to(HERE)} "
        f"({len(samples)} samples, first at "
        f"{samples['Process Time [h]'].iloc[0]:.3f} h)"
    )


def main() -> None:
    for raw in (
        RAW_R6,
        RAW_R5_BATCH,
        RAW_R5_FEDBATCH,
        ANALYTICS_XLSX,
        CALIBRATION_XLSX,
        DCW_XLSX,
    ):
        if not raw.exists():
            raise SystemExit(f"Missing raw file: {raw}")

    prepare_r6()
    prepare_r5()
    build_samples("R5")
    build_samples("R6")
    print("\nDone. Now run: python analyze_hgh_run.py R5_lowlow  (and R6_highhigh)")


if __name__ == "__main__":
    sys.exit(main())
