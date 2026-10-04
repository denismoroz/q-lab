"""Detector variants are on record (docs/REGIME_DETECT.md)."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab import detector_trials as dt
from qlab.registry.models import Base, DetectorTrial


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def test_variants_and_holdout_looks_are_counted(session) -> None:
    dev = (pd.Timestamp("2020-04-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    hold = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
    for v in ("a", "b", "c"):
        dt.record("level1", v, dt.DEVELOPMENT, dev, {"acc": 0.6}, "x.py", session=session)
    dt.record("level1", "b", dt.HOLDOUT, hold, {"acc": 0.62}, "x.py", chosen=True,
              session=session)
    dt.record("level1", "a", dt.HOLDOUT, hold, {"acc": 0.61}, "x.py", session=session)
    (s,) = dt.summary(session)
    assert (s.family, s.development_variants, s.holdout_looks, s.chosen) == ("level1", 3, 2, ["b"])
    row = session.query(DetectorTrial).filter_by(period="holdout", chosen=True).one()
    assert str(row.data_start) == "2024-01-01" and row.metrics == {"acc": 0.62}
    with pytest.raises(ValueError, match="period"):
        dt.record("level1", "a", "later", dev, {}, "x.py", session=session)
