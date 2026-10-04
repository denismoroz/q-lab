"""Every research run of a detector variant goes on record (qlab.detector_trials)."""

from __future__ import annotations

import pandas as pd

from qlab import detector_trials as dt

HOLDOUT_START = pd.Timestamp("2024-01-01", tz="UTC")


def log(family: str, variant: str, window, metrics: dict, script: str,
        chosen: bool = False) -> None:
    """Record one scored variant; the period follows from the window: from
    2024-01-01 on it is the holdout."""
    start = pd.Timestamp(window[0])
    start = start if start.tzinfo else start.tz_localize("UTC")
    period = dt.HOLDOUT if start >= HOLDOUT_START else dt.DEVELOPMENT
    dt.record(family, variant, period, window, metrics, script, chosen=chosen)
