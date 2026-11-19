"""Audit: operator-noted induction time vs. SCADA phase flip for the scFv DoE.

For every ferment under `examples/doe_example/all_data/` (dir with a raw CSV +
`*_Auswertung.xlsx`), extract:

- the operator's `Induction` time [h] from the 'Time-Rownr' sheet (same event
  table layout as the LDH `Evaluation_*.xlsx` workbooks),
- the SCADA phase-flip time [h] (first sample whose `Phase` enters the induced
  phase; process time = `Batch Time (since inoc.), sec` / 3600),
- the gap (flip - operator; negative = flip LEADS the physical bolus),
- offline sample times falling inside the gap (the F16-F19 failure mode: a
  sample drawn after the flip but before inducer was added, misflagged as
  post-induction by phase-based logic).

Read-only audit; writes nothing.
"""

import sys
from pathlib import Path

import openpyxl
import pandas as pd

ALL_DATA = Path(__file__).parent / "all_data"
INDUCED_PAT = "induc"  # phase-name substring, case-insensitive


def operator_events(xlsx: Path):
    """Return (inoculation_h, induction_h, [(label, time_h), ...]) from Time-Rownr."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["Time-Rownr"]
    inoculation, induction, samples = None, None, []
    for row in ws.iter_rows(max_row=20, max_col=4, values_only=True):
        label, t = row[0], row[3]  # cols: Sample | Hour | Minute | Time
        if label is None or not isinstance(t, (int, float)):
            continue
        label = str(label).strip()
        if label.lower() == "inoculation":
            inoculation = float(t)
        elif label.lower() == "induction":
            induction = float(t)
        elif label.upper().startswith("S") and label[1:].isdigit():
            samples.append((label, float(t)))
    wb.close()
    return inoculation, induction, samples


def csv_phase_flip(csv: Path):
    """Return (flip_op_base_h, phase_name, phases, t0_feedstart_h) for the first
    induced-phase row.

    The scFv exports start at inoculation, so batch coords ~= operator base.
    Two export conventions exist across campaigns: post-inoculation batch
    growth labelled 'Preparation' (Run1) or 'Batch'. The operator-base flip is
    therefore batch_flip - operator_inoculation; t0 (first non-Preparation
    batch time) is reported to expose the convention.
    """
    df = pd.read_csv(
        csv,
        sep=";",
        decimal=",",
        skiprows=1,  # version banner line: '1.121.957;de-DE'
        encoding="cp1252",
        usecols=["Phase", "Batch Time, sec"],
    )
    phases = df["Phase"].astype(str).str.strip()
    unique = sorted(phases.dropna().unique())
    batch_s = pd.to_numeric(df["Batch Time, sec"], errors="coerce")
    induced = phases.str.contains(INDUCED_PAT, case=False, na=False)
    kept = phases.ne("Preparation") & phases.notna()
    t0 = float(batch_s[kept].dropna().iloc[0]) / 3600.0 if kept.any() else None
    if not induced.any():
        return None, None, unique, t0
    first = induced.idxmax()
    return float(batch_s.iloc[first]) / 3600.0, phases.iloc[first], unique, t0


def main() -> None:
    rows = []
    for csv in sorted(ALL_DATA.rglob("*.csv")):
        xlsx_candidates = sorted(csv.parent.glob("*_Auswertung.xlsx"))
        if not xlsx_candidates:
            print(f"SKIP (no Auswertung): {csv.relative_to(ALL_DATA)}")
            continue
        xlsx = xlsx_candidates[0]
        campaign = csv.relative_to(ALL_DATA).parts[0]
        ferment = csv.parent.name

        inoculation, induction, samples = operator_events(xlsx)
        flip_batch, phase_name, unique, t0 = csv_phase_flip(csv)
        if induction is None or flip_batch is None:
            print(f"SKIP (no induction event / phase flip): {ferment}")
            continue

        # exports start at inoculation -> operator-base flip = batch_flip - inoc
        offset_h = (flip_batch - inoculation) if inoculation is not None else flip_batch
        gap = offset_h - induction  # negative = flip leads the bolus
        lo, hi = sorted((induction, offset_h))
        in_gap = [lbl for lbl, t in samples if lo < t < hi]
        rows.append(
            {
                "campaign": campaign,
                "ferment": ferment,
                "phases": "|".join(unique),
                "operator_inoc_h": inoculation,
                "operator_ind_h": induction,
                "flip_batch_h": flip_batch,
                "flip_op_h": offset_h,
                "gap_min": gap * 60.0,
                "samples_in_gap": ", ".join(in_gap) if in_gap else "-",
            }
        )
        print(
            f"{campaign} | {ferment}\n"
            f"    phases [{', '.join(unique)}] | inoc {inoculation:.3f} h | "
            f"operator {induction:.4f} h | flip {offset_h:.4f} h ('{phase_name}') "
            f"| gap {gap * 60:+.0f} min | samples in gap: {in_gap or '-'}"
        )

    if not rows:
        print("Nothing audited.")
        return
    summary = pd.DataFrame(rows)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        summary.to_csv(out, index=False, sep=";", decimal=",")
        print(f"\nWrote {out}")
    print(
        f"\n{len(rows)} ferments audited | flip leads in "
        f"{(summary['gap_min'] < 0).sum()}/{len(rows)} | "
        f"lead range {summary['gap_min'].min():+.0f} .. {summary['gap_min'].max():+.0f} min"
    )


if __name__ == "__main__":
    main()
