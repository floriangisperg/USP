"""Smoke test: run the pipeline on synthetic data and assert sane results.

Executed in CI (see .github/workflows/ci.yml) and runnable locally::

    python verify_results.py

The assertions double as documentation of what "working" means: phase
filtering, feed from the balance, volume continuity, positive gas rates,
physiological RQ, sample alignment, induction timing, protein fractions.
"""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_synthetic_run import run_analysis  # noqa: E402


def main() -> int:
    result = run_analysis(EXAMPLE_DIR)
    df = result.processed

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    t = df["Process time [h]"]

    print("synthetic example smoke test:")
    check("at least 800 online rows", len(df) >= 800)
    check("'Preparation' phase filtered, time starts at 0", t.min() >= -1e-9)
    check(
        "no raw 'Preparation' rows survive",
        not (df["Process phase [-]"] == "Preparation").any(),
    )

    feed_final = df["Feed [g]"].dropna().iloc[-1]
    check(f"cumulative feed plausible ({feed_final:.1f} g)", 150 < feed_final < 300)

    vol_end = df["Volume corrected [mL]"].dropna().iloc[-1]
    check(f"volume continuity ({vol_end:.0f} mL at end)", 1550 < vol_end < 1700)

    grown = df[t > 2]
    check("OUR positive and large during growth", grown["OUR [mmol/L/h]"].mean() > 5)
    check("CER positive during growth", grown["CER [mmol/L/h]"].mean() > 5)
    rq = df["RQ [-]"].replace([np.inf, -np.inf], np.nan).dropna()
    check(f"RQ physiological ({rq.mean():.2f} mean)", 0.5 < rq.mean() < 1.5)

    samples = df[df["is_sample"] == True].drop_duplicates(
        subset=["Process time [h]"]
    )  # noqa: E712
    check(f"9 offline samples aligned ({len(samples)} found)", len(samples) == 9)

    induced = df[df["Is induced [-]"] > 0]["Process time [h]"]
    check(
        f"induction flags at ~16 h ({induced.min():.2f} h)",
        len(induced) > 0 and abs(induced.min() - 16.0) < 0.34,
    )

    late = samples[samples["Process time [h]"] > 16]
    frac = (late["Protein IB [g/L]"] / late["Protein [g/L]"]).dropna()
    check(
        f"IB fraction ~0.4 after induction ({frac.iloc[-1]:.2f})",
        len(frac) > 0 and 0.3 < frac.iloc[-1] < 0.5,
    )

    failed = [name for name, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
