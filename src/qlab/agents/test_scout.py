"""Tests for the scout's token economy and guards."""

from __future__ import annotations

import json
from datetime import date

import qlab.agents.scout as scout_mod
from qlab.budget import ExplicitNoBudget


def test_no_structural_event_spends_no_tokens(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scout_mod, "QLAB_ROOT", tmp_path)
    (tmp_path / "data/events").mkdir(parents=True)
    (tmp_path / "data/events/2026-10-03.json").write_text(json.dumps(
        [{"source": "hyperliquid", "kind": "funding-extreme", "instrument": "X", "detail": ""}]))

    def no_call(*a, **k):
        raise AssertionError("the agent must not be called")

    monkeypatch.setattr(scout_mod, "run_agent", no_call)
    out = scout_mod.scout(date(2026, 10, 3), budget=ExplicitNoBudget("test"), session=None)
    assert out.reply.startswith("NO CANDIDATE")
