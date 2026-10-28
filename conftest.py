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


TEST_ACCOUNT = "moroz.denis@gmail.com"


@pytest.fixture(autouse=True)
def _right_account_logged_in(monkeypatch) -> None:
    """The account check (`qlab.budget.agent.check_account`) asks `claude auth status`, which
    would reach the real CLI and the owner's real login. Every test sees the expected account
    logged in, with the per-process cache emptied; a test of the check sets its own status over
    this (`agent._read_status`), or calls `agent._read_status_cli` against a fake binary."""
    import qlab.budget.agent as agent

    monkeypatch.setattr(agent, "_read_status", lambda account: {
        "loggedIn": True, "email": TEST_ACCOUNT, "orgName": "test", "subscriptionType": "pro"})
    monkeypatch.setattr(agent, "_checked", {})


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
    monkeypatch.setattr(night, "anomalies_stage", lambda *a, **kw: None)
    # The overlays stage runs variants on the stand against the real registry and writes spec
    # files and a ledger (2026-10-28: a night test that did not patch it ran two variants of a
    # real base). A test of the stage calls qlab.overlay_stage.overlays_stage directly.
    monkeypatch.setattr(night, "overlays_stage", lambda *a, **kw: None)


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


@pytest.fixture(autouse=True)
def _no_real_implement_ledgers(monkeypatch, tmp_path_factory) -> None:
    """The implement stage's ledger (data/implement/ledger.json), its baseline files and the
    register of agent-written blocks (night/agent_blocks.yaml) are the owner's real files: a
    test that reaches them gets a temporary copy that is empty. A test of the stage sets its own
    over this."""
    import qlab.agent_blocks as agent_blocks
    import qlab.implement_run as implement_run

    folder = tmp_path_factory.mktemp("implement")
    monkeypatch.setattr(implement_run, "LEDGER", folder / "ledger.json")
    monkeypatch.setattr(implement_run, "BASELINE_DIR", folder / "baselines")
    monkeypatch.setattr(agent_blocks, "REGISTRY", folder / "agent_blocks.yaml")


@pytest.fixture(autouse=True)
def _guard_limits_on(monkeypatch, tmp_path_factory) -> None:
    """The owner's switch that turns the guard's ceilings off (night/budget_off.yaml) is a real
    file: while it is in force every test of the guard would see no ceilings. Tests see no
    switch; a test of the switch sets its own over this."""
    import qlab.budget.guard as guard

    monkeypatch.setattr(guard, "LIMITS_OFF_FILE",
                        tmp_path_factory.mktemp("budget") / "budget_off.yaml")


@pytest.fixture(autouse=True)
def _forward_series_in_tmp(monkeypatch, tmp_path_factory) -> None:
    """`evaluate_spec` stores a passed run's forward-test series under data/forward_series (the
    console's chart, docs/FORWARD_CHART.md); a test that runs it writes into a temporary folder
    instead of the owner's real one. A test of the series sets its own over this."""
    monkeypatch.setenv("QLAB_FORWARD_SERIES", str(tmp_path_factory.mktemp("forward_series")))


@pytest.fixture(autouse=True)
def _overlay_files_in_tmp(monkeypatch, tmp_path_factory) -> None:
    """The overlays' stored comparisons and diagnostics (data/overlays, docs/OVERLAYS.md) go to a
    temporary folder in tests. A test of the stage sets its own over this."""
    monkeypatch.setenv("QLAB_OVERLAYS", str(tmp_path_factory.mktemp("overlays")))
