# CORRECTION — F19 Protein IB series (2026-09-04)

## What was wrong

`data/samples.xlsx` carried a **corrupted Protein IB series** for F19. The values
1.2537211 / 2.2297262 / 2.3851574 / 2.4536884 g/L (samples at 25.02 / 27.02 / 29.02 /
31.02 h) are **not titers** — they are the sheet's own `IB/Sol` **ratio column**, pasted
into the "Protein IB (g/L)" column during a manual rework of `Evaluation_F19.xlsx`.

## Provenance / how it happened (settled in the paper-vs-data audit, 2026-09-04)

1. **23.03.2026** — the lab measured F19: soluble 0.883 / 1.029 / 1.142 / 1.302 g/L and
   IB **1.1073153 / 2.2941798 / 2.7247912 / 3.1947748 g/L** (these titers are preserved in
   the 23.03 MODDE import and in `LDH_DoE_Summary.xlsx`, and match the publication).
2. Same day, the **soluble series was retracted as below-LOD** (the paper prints "n.d.").
3. **14.04.2026 16:49** — rework of `Evaluation_F19.xlsx`: soluble hardcoded to 0 (typed
   constants), and the **ratio column got pasted into the IB column** (rows 14/17/20/23 of
   the Sampling sheet are typed constants amid otherwise live formulas; AA is the live
   `=Y/Z` ratio formula). Identity check, 4/4 exact:
   `canonical_IB ÷ pasted_value = 0.883223 / 1.028907 / 1.142395 / 1.302030`
   = the retracted soluble series. Chronology rules out the reverse direction.

## What was changed here

`data/samples.xlsx`, rows 4–7 (columns E "Protein IB [g/L]" and G
"Protein concentration [g/L]"; soluble F stays 0/n.d.):

| t [h] | old (artifact, = IB/Sol) | **new (canonical titer)** |
| -----:| ------------------------:| -------------------------:|
| 25.02 | 1.2537211                | **1.1073153**             |
| 27.02 | 2.2297262                | **2.2941798**             |
| 29.02 | 2.3851574                | **2.7247912**             |
| 31.02 | 2.4536884                | **3.1947748**             |

The canonical values are the 23.03 measurement / summary cells that the publication uses.

## Files NOT changed here

- `data/Evaluation_F19.xlsx` — this repo's copy still contains the corrupted rework
  (Z14/17/20/23). It is kept as received; **the lab (M.E.) must fix the file of record**,
  after which this copy can be replaced. Nothing in this repo's pipeline reads it
  (`samples.xlsx` is the sole pipeline input).
- The campaign workbook (outside this repo) carries the same artifact; fix handled
  separately.

## Downstream

Per-run F19 results and both consolidated workbooks (+ the modeling-playground pickle)
were regenerated after this correction. Modeling note: the T35/qs0.4 endpoint rises
+0.74 g/L (2.454 → 3.195); the T35 group becomes F07 1.744 / F19 3.195 / F12 2.848 —
a mid-qs hump like T30. Any `M0_aload_qPTexp` fit that predates 2026-09-04 used the old
values (see `modeling_playground/HANDOFF.md`).
