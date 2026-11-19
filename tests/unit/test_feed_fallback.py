"""Feed calculation edge case: feed bottle placed on the balance after feeding started."""

import numpy as np
import pandas as pd

from upstream_analysis.data.calculations.feed import calculate_feed

CONFIG = {
    "processing": {
        "processing_parameters": {
            "feed_definitions": {"feed_1": {"density_g_ml": 1.15}},
            "balance_mode": "feed_source",
        }
    }
}


def _run_df():
    t = np.arange(0.0, 20.0, 0.1)
    phase = np.where(t < 5, "Batch", "Fedbatch")
    # Balance reads 0 through batch and early fed-batch (bottle not placed),
    # then jumps to ~1654 g at t=8 and declines as feed is consumed.
    balance = np.where(t < 8, 0.0, np.maximum(1654.0 - (t - 8.0) * 20.0, 1000.0))
    return pd.DataFrame(
        {"Process time [h]": t, "Process phase [-]": phase, "Balance [g]": balance}
    )


def test_feed_fallback_when_bottle_placed_late():
    out = calculate_feed(_run_df(), CONFIG)
    # Balance declines 1654 -> ~1414 from placement (t=8) to t=20: ~240 g consumed.
    feed_max = out["Feed [g]"].max()
    assert (
        200 < feed_max < 300
    ), f"expected feed from the post-placement tare, got {feed_max}"


def test_feed_normal_case_unaffected():
    t = np.arange(0.0, 20.0, 0.1)
    phase = np.where(t < 5, "Batch", "Fedbatch")
    # Bottle on balance from the start: flat during batch, declining while feeding.
    balance = np.where(t < 5, 1773.0, np.maximum(1773.0 - (t - 5.0) * 20.0, 800.0))
    df = pd.DataFrame(
        {"Process time [h]": t, "Process phase [-]": phase, "Balance [g]": balance}
    )
    out = calculate_feed(df, CONFIG)
    # Batch flat at 1773, feeding from t=5 declining 20 g/h -> ~300 g by t=20.
    feed_max = out["Feed [g]"].max()
    assert 250 < feed_max < 350, f"normal tare path changed: {feed_max}"
