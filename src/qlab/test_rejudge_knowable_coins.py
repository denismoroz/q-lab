"""The dry run and the rewrite of scripts/rejudge_knowable_coins.py on a made-up lab root."""

from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qlab import knowable_coins as kc
from qlab.pipeline.spec import load_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "specs" / "generated" / "stage2" / "bv3-research-fast-release-hl.yaml"
CONFIG = kc.KnowableConfig(top_k=2, window_days=5, raw_dir=Path("data/raw"), interval="1d",
                           source="night/knowable_coins.yaml")


def _as_before_the_rejudge(path: Path) -> None:
    """The repo's spec was itself rewritten by the script on 2026-11-04 (commit 8f61c02): put back
    the hand date it carried before, so the test does not depend on the repo's current state."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.pop("coins_knowable", None)
    raw["judged_from"] = "2026-10-03"
    raw["instruments_chosen_at"] = "2026-10-03"
    raw.setdefault("sources", {})
    raw["sources"]["instruments_chosen_at"] = (
        "specs/research/bv2/bv2-research-btceth.yaml: params_fixed_at")
    raw["sources"]["judged_from"] = (
        "docs/RETUNE.md: a hand-picked list is judged from the day it was known")
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


@pytest.fixture()
def script():
    spec = importlib.util.spec_from_file_location(
        "rejudge_knowable_coins", ROOT / "scripts" / "rejudge_knowable_coins.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def lab(tmp_path: Path, monkeypatch) -> Path:
    """A root with one second-stage spec of an open idea and a registry that says it is open."""
    target = tmp_path / "specs" / "generated" / "stage2"
    target.mkdir(parents=True)
    shutil.copy(SPEC, target / SPEC.name)
    _as_before_the_rejudge(target / SPEC.name)
    (tmp_path / "data").mkdir()
    with sqlite3.connect(tmp_path / "data" / "qlab.db") as conn:
        conn.execute("create table idea (id text, status text)")
        conn.execute("insert into idea values ('bv3-research-fast-release-hl', 'bench')")
    spec = load_spec(SPEC)
    index = pd.date_range(pd.Timestamp(spec.data.start) - pd.Timedelta(days=30), spec.data.end)
    volume = pd.DataFrame({"BTC": 1000.0, "ETH": 900.0, "F1": 1.0}, index=index)
    monkeypatch.setattr(kc, "raw_volume", lambda venue, config=None: volume)
    return tmp_path


def test_the_dry_run_lists_the_cut_and_writes_nothing(script, lab: Path) -> None:
    before = (lab / "specs/generated/stage2" / SPEC.name).read_text(encoding="utf-8")
    rows = script.scan(lab, lab / "data" / "qlab.db", CONFIG)
    assert len(rows) == 1 and rows[0].changed
    assert rows[0].before == date(2026, 10, 3) and rows[0].after is None
    out = script.render(rows)
    assert "2026-10-03 -> data start" in out and "bv3-research-fast-release-hl" in out
    assert (lab / "specs/generated/stage2" / SPEC.name).read_text(encoding="utf-8") == before


def test_a_closed_idea_is_not_listed(script, lab: Path) -> None:
    with sqlite3.connect(lab / "data" / "qlab.db") as conn:
        conn.execute("update idea set status = 'rejected'")
    assert script.scan(lab, lab / "data" / "qlab.db", CONFIG) == []


def test_apply_rewrites_the_dates_and_the_spec_still_loads(script, lab: Path) -> None:
    rows = script.scan(lab, lab / "data" / "qlab.db", CONFIG)
    assert script.apply(rows, lab, CONFIG) == [rows[0].path]
    spec = load_spec(lab / rows[0].path)
    assert spec.judged_from is None and spec.instruments_chosen_at is None
    assert spec.coins_knowable.from_start and spec.coins_knowable.venue == "hyperliquid"
    # a second pass finds nothing left to change
    again = script.scan(lab, lab / "data" / "qlab.db", CONFIG)
    assert not any(row.changed for row in again)
