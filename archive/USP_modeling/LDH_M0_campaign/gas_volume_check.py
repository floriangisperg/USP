"""Gas-biomass conflict under corrected volume (M1a_aload).

The bracket re-fit showed correcting the missing base volume raises the sparse
biomass-fit Y_xs (0.351 -> 0.368), recovering ~40% of the gap to the gas-implied
~0.39. This asks whether that reconciliation is enough to rescue the offgas model:
does M1a_aload (gas + sparse) still overpredict held-out biomass under corrected
volume, or does the bias shrink toward zero?

Fold A (holdout F17) -- the gas-conflict test fold from the HANDOFF (M1a_aload gave
F17 X_bias -1.40 unfiltered / -2.5 RQ-filtered at base=0).
"""

from __future__ import annotations

from pathlib import Path

import data as D
import fit as F
import numpy as np
from base_reconstruct import apply_volume

OUT = Path(__file__).resolve().parent / "results" / "base_volume"


def main():
    tr, te = D.split_for_fold("A")
    base_bundle = D.load_dataset(tr, te)
    c_fold = base_bundle["c_fold"]
    L = ["# M1a_aload gas-biomass conflict under corrected volume (fold A, holdout F17)", ""]
    print(f"{'level':6} {'cost':>8} {'Y_xs':>6} {'F17 X_obs':>9} {'F17 X_pred':>10} {'X_bias':>8}")
    L.append("| level | train cost | Y_xs | F17 X obs | F17 X pred | X bias (pred-obs) |")
    L.append("| --- | ---:| ---:| ---:| ---:| ---:|")
    for level in ["none", "floor", "upper"]:
        bundle = apply_volume(base_bundle, level)
        res = F.fit_model("M1a_aload", bundle, seed=42, n_starts=2, max_nfev=300)
        p = res.params
        from models import simulate_run
        run17 = bundle["test"][0]
        pred = simulate_run("M1a_aload", p, run17, c_fold, eval_gas=True)
        xo = run17["obs"]["X"][-1]; xp = pred["obs"]["X"][-1]
        print(f"{level:6} {res.cost:8.4f} {p['Y_xs']:6.3f} {xo:9.2f} {xp:10.2f} {xp-xo:+8.2f}")
        L.append(f"| {level} | {res.cost:.4f} | {p['Y_xs']:.3f} | {xo:.2f} | {xp:.2f} | {xp-xo:+.2f} |")
    L.append("")
    L.append("HANDOFF reference (base=0): M1a_aload F17 X_bias = -1.40 (unfiltered) / -2.5 (RQ-filtered).")
    L.append("Negative bias => overprediction. If the bias shrinks toward 0 under corrected volume,")
    L.append("the missing base volume was partly driving the gas-biomass conflict.")
    (OUT / "gas_volume_check.md").write_text("\n".join(L))
    print(f"\nsaved {OUT/'gas_volume_check.md'}")


if __name__ == "__main__":
    main()
