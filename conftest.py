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
    """A test that runs the whole night never reaches the programmer's queue, the
    cataloguer's ledger or the carder's stage on disk: the stages read and write the owner's real
    files (data/programmer/queue.json, data/cataloguer/ledger.json). Until 2026-10-27 every test
    run appended a budget error to the real queue. A test of a stage sets its own over this."""
    import qlab.night as night

    monkeypatch.setattr(night, "programmer_stage", lambda *a, **kw: None)
    monkeypatch.setattr(night, "catalogue_stage", lambda *a, **kw: None)
    monkeypatch.setattr(night, "cards_stage", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _no_real_inbox_or_queue(monkeypatch, tmp_path_factory) -> None:
    """The ideas inbox and the implementer's queue are the owner's tracked files; the carder and
    the memo ingestion write them. A test that reaches either finds an empty place in a temporary
    folder (2026-10-27: added with the carder). A test of the carder sets its own over this."""
    import qlab.agents.carder as carder
    import qlab.ideas as ideas

    folder = tmp_path_factory.mktemp("inbox")
    monkeypatch.setattr(ideas, "INBOX", folder / "inbox.yaml")
    monkeypatch.setattr(carder, "QUEUE", folder / "implement_queue.yaml")
    monkeypatch.setattr(carder, "BRIEFS", folder / "briefs")
    monkeypatch.setattr(carder, "DRAFTS", folder / "drafts")
    monkeypatch.setattr(carder, "CARDS", folder / "cards")
