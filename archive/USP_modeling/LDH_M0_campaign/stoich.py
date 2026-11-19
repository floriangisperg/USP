"""Fixed elemental composition and the carbon + electron (redox) balance.

The offgas observables CER and OUR are predicted from the model's instantaneous
rates (substrate uptake, biomass growth, product formation) using a fixed
stoichiometry and the degree-of-reduction (electron) balance. This adds
**information, not parameters**: every coefficient below is a known physical
constant, not a fitted value.

Conventions
-----------
- C-mol basis: 1 C-mol CO2 = 1 mol CO2 = 1000 mmol CO2.
- Degree of reduction (DoR, gamma) is defined relative to NH3 (N oxidation
  state = -3), H = +1, C = +4, O = -2. Because the nitrogen source (NH4OH) is
  the reference, the unlogged NH4OH flux drops out of the electron balance --
  this is why the redox balance is still usable even though N is untracked.
- Biomass ash fraction (0.08) is NOT applied by the USP pipeline; biomass
  carbon is therefore over-counted by ~8%. We keep this default for
  consistency with the processed data and flag it as a known limitation.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Composition(NamedTuple):
    """Elemental composition on a C-mol basis (per 1 C atom)."""
    C: float
    H: float
    N: float
    O: float
    mw_g_per_cmol: float

    def degree_of_reduction(self) -> float:
        """DoR relative to NH3: gamma = 4C + H - 3N - 2O (with C normalized to 1)."""
        return 4.0 * self.C + 1.0 * self.H - 3.0 * self.N - 2.0 * self.O


# Fixed compositions (from src/upstream_analysis/core/constants.py + LDH run config).
GLUCOSE = Composition(C=1.0, H=2.0, N=0.0, O=1.0, mw_g_per_cmol=30.03)   # CH2O
BIOMASS = Composition(C=1.0, H=1.77, N=0.24, O=0.49, mw_g_per_cmol=26.565)  # E. coli
LDH_PRODUCT = Composition(C=1.0, H=1.59, N=0.27, O=0.31, mw_g_per_cmol=22.58)

GAMMA_S = GLUCOSE.degree_of_reduction()      # = 4.0
GAMMA_X = BIOMASS.degree_of_reduction()      # ~ 4.07
GAMMA_P = LDH_PRODUCT.degree_of_reduction()  # ~ 4.16


def predict_gas(
    r_S: np.ndarray,
    r_X: np.ndarray,
    r_P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict CER and OUR [mmol/L/h] from volumetric rates [g/L/h].

    Carbon balance (C-mol/L/h):  r_S/mw_S = r_X/mw_X + r_P/mw_P + r_CO2
        -> CER = 1000 * (r_S/mw_S - r_X/mw_X - r_P/mw_P)    [mmol CO2/L/h]

    Electron balance (e- eq/L/h): gamma_S*r_S/mw_S = gamma_X*r_X/mw_X + gamma_P*r_P/mw_P + 4*r_O2
        -> OUR = 1000 * (gamma_S*r_S/mw_S - gamma_X*r_X/mw_X - gamma_P*r_P/mw_P) / 4  [mmol O2/L/h]

    r_S = qS*X  (substrate uptake), r_X = mu*X (gross growth), r_P = qP*X (product).
    """
    mw_S, mw_X, mw_P = GLUCOSE.mw_g_per_cmol, BIOMASS.mw_g_per_cmol, LDH_PRODUCT.mw_g_per_cmol
    c_in = r_S / mw_S - r_X / mw_X - r_P / mw_P          # C-mol CO2 / L / h
    e_in = GAMMA_S * r_S / mw_S - GAMMA_X * r_X / mw_X - GAMMA_P * r_P / mw_P  # e- eq / L / h
    cer = 1000.0 * c_in
    our = 1000.0 * e_in / 4.0
    return cer, our


def print_composition_summary() -> None:
    """Human-readable stoichiometry summary (for the HANDOFF / logs)."""
    for name, comp in (("glucose", GLUCOSE), ("biomass", BIOMASS), ("LDH product", LDH_PRODUCT)):
        print(
            f"  {name:11s}: C{comp.C:.2f}H{comp.H:.2f}N{comp.N:.2f}O{comp.O:.2f}  "
            f"{comp.mw_g_per_cmol:.3f} g/Cmol  gamma={comp.degree_of_reduction():.3f}"
        )


if __name__ == "__main__":
    print("Composition + DoR (NH3-referenced):")
    print_composition_summary()
    # Quick sanity: pure aerobic growth (no product), Y_xs=0.45 -> RQ and yields.
    qS = np.array([0.4]); X = np.array([30.0]); Y = 0.45
    r_S = qS * X; r_X = Y * qS * X; r_P = np.array([0.0])
    cer, our = predict_gas(r_S, r_X, r_P)
    print(f"\nSanity (Y_xs=0.45, qS=0.4, X=30): CER={cer[0]:.1f} OUR={our[0]:.1f} RQ={cer[0]/our[0]:.3f}")
