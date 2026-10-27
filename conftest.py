"""Tests never call the real agent: the `claude` binary is replaced by `false` for
every test. A test that exercises the agent plumbing sets its own fake binary
(`QLAB_CLAUDE_BIN`) over this. A real call from a test would spend the owner's tokens
and rewrite memos on disk (it did, 2026-10-25, when the nightly researcher became the
default of `qlab.night.run`)."""

from __future__ import annotations

import pytest

from qlab.budget.agent import CLAUDE_BIN_ENV


@pytest.fixture(autouse=True)
def _no_real_agent(monkeypatch) -> None:
    monkeypatch.setenv(CLAUDE_BIN_ENV, "/usr/bin/false")


@pytest.fixture(autouse=True)
def _noise_in_process(monkeypatch) -> None:
    """Noise books run in this process in tests (qlab.calibration.run); the one
    test of the worker pool sets its own count."""
    from qlab.calibration.run import NOISE_WORKERS_ENV

    monkeypatch.setenv(NOISE_WORKERS_ENV, "1")


@pytest.fixture(autouse=True)
def _no_real_agent_ledgers(monkeypatch) -> None:
    """A test that runs the whole night never reaches the programmer's queue or the
    cataloguer's ledger on disk: both stages read and write the owner's real files
    (data/programmer/queue.json, data/cataloguer/ledger.json). Until 2026-10-27 every test
    run appended a budget error to the real queue. A test of a stage sets its own over this."""
    import qlab.night as night

    monkeypatch.setattr(night, "programmer_stage", lambda *a, **kw: None)
    monkeypatch.setattr(night, "catalogue_stage", lambda *a, **kw: None)
