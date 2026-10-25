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
