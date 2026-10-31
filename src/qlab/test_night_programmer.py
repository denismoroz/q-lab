"""The programmer's place and size in the night (owner, 2026-10-30: «переноси токены на
программиста. или даже подключай 2 если нужно»): the stage runs before the director and the
researcher, takes two requests a night with the ones that give the generator a new block first,
and spends a fixed share of what the night has left. The agents and the budget are faked."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import yaml

import qlab.agents.programmer as prog
import qlab.budget as budget_module
import qlab.directions as directions_module
import qlab.night as night
from qlab.night import programmer_stage as real_programmer_stage  # the conftest stubs it

DAY = date(2026, 10, 31)
REPO = Path(__file__).resolve().parents[2]


def test_the_programmer_runs_before_the_director_and_the_researcher(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "unwatched_bench", lambda watch: [])
    order: list[str] = []
    for name in ("programmer_stage", "director_stage", "research_stage", "implement_stage",
                 "search_stage", "catalogue_stage", "cards_stage", "generate_stage"):
        monkeypatch.setattr(night, name, lambda *a, _n=name, **kw: order.append(_n))
    night.run(DAY, graveyard=False, paper=False, detectors=False, assess=False, waiting=False,
              research=True, director=True, twins=False, overlays=False, restamp=False,
              anomalies=False, ladder=False)
    assert order == ["programmer_stage", "director_stage", "research_stage", "implement_stage",
                     "search_stage", "catalogue_stage", "cards_stage", "generate_stage"]


def _stage_world(tmp_path, monkeypatch, texts, **config):
    """The real stage over a queue in `tmp_path`, a night budget of one million tokens and a
    `run_request` that records the cap it was given and spends `spend` tokens."""
    queue = [prog.Request(id=f"2026-10-30-{n}", repo="qlab", what=what, why="y", source="owner",
                          created="t") for n, what in enumerate(texts, 1)]
    held = {"queue": queue}
    monkeypatch.setattr(prog, "load_queue", lambda *a, **kw: held["queue"])
    monkeypatch.setattr(prog, "save_queue", lambda q, *a, **kw: held.update(queue=q))
    monkeypatch.setattr(directions_module, "active_directions", lambda *a, **kw: None)
    cfg = tmp_path / "programmer.yaml"
    cfg.write_text(yaml.safe_dump({
        "enabled": True, "per_night": 2, "night_share": 0.5, "max_attempts": 3,
        "repos": {"qlab": {"path": ".", "allowed": ["src/*"], "test_command": ["pytest"]}},
        **config}), encoding="utf-8")
    seen: list[dict] = []

    class _Night:
        @staticmethod
        def open(_session, _reading):
            return _Night()

        def check(self) -> None:
            pass

        def tokens_left(self) -> int:
            return 1_000_000

    def fake_run_request(req, cfg, *, budget, session, checkpoint=None):
        seen.append({"id": req.id, "cap": budget.cap_tokens})
        req.attempts += 1
        req.status, req.tokens = "ready", 100_000
        return req

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    import qlab.registry.db as db_module
    from qlab.registry.models import Base

    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(budget_module, "NightBudget", _Night)
    monkeypatch.setattr(budget_module, "probe", lambda _s: SimpleNamespace(
        report=SimpleNamespace(reading=None)))
    monkeypatch.setattr(prog, "run_request", fake_run_request)
    return cfg, seen, held


def test_two_requests_a_night_new_blocks_first_within_the_share(tmp_path, monkeypatch) -> None:
    cfg, seen, held = _stage_world(tmp_path, monkeypatch, [
        "Add an anomaly flag for the counterfactual",           # 1: not a block
        "Make the eligibility mask use the 24h of volume",      # 2: not a block
        "Блок сигнала: положение ставки среди своей истории",   # 3: a new block
    ])
    state: dict = {"done": {}}
    real_programmer_stage(DAY, state, config=cfg)
    assert [s["id"] for s in seen] == ["2026-10-30-3", "2026-10-30-1"]  # the block first
    # the stage may spend half of the million; the first request gets half of that, the second
    # what the first left of it
    assert [s["cap"] for s in seen] == [250_000, 400_000]
    assert [e["id"] for e in state["programmer"]["entries"]] == ["2026-10-30-3", "2026-10-30-1"]
    assert held["queue"][1].status == "queued"  # the third waits for the next night
    real_programmer_stage(DAY, state, config=cfg)  # the same night again: nothing is repeated
    assert len(seen) == 2


def test_no_new_request_starts_after_the_stop_time(tmp_path, monkeypatch) -> None:
    cfg, seen, _ = _stage_world(tmp_path, monkeypatch, ["Add a flag", "Add another flag"])
    state: dict = {"done": {}}
    real_programmer_stage(DAY, state, config=cfg,
                          stop_at=datetime.now(UTC) - timedelta(minutes=1))
    assert seen == [] and state["stopped_at"]["before"] == "programmer"


def test_a_budget_stop_keeps_the_request_for_the_next_night(tmp_path, monkeypatch) -> None:
    from qlab.budget import NightExhausted

    cfg, seen, held = _stage_world(tmp_path, monkeypatch, ["Блок сигнала: x", "Add a flag"])

    def stopped(req, cfg, *, budget, session, checkpoint=None):
        raise NightExhausted("the night's budget is spent")

    monkeypatch.setattr(prog, "run_request", stopped)
    state: dict = {"done": {}}
    real_programmer_stage(DAY, state, config=cfg)
    assert state["programmer"]["stopped"] and state["programmer"]["entries"] == []
    assert [r.status for r in held["queue"]] == ["queued", "queued"]


def test_the_programmer_config_is_two_levels_two_a_night_with_a_ready_environment() -> None:
    cfg = prog.load_config()
    assert cfg["planner_model"] == "opus" and cfg["model"] == "sonnet"
    assert cfg["per_night"] == 2 and 0 < cfg["night_share"] <= 1
    assert cfg["max_iterations"] == 7  # the writer/reviewer loop (owner, 2026-10-31)
    assert Path(cfg["siblings"]["funding-rate-arbitrage"]).name == "funding-rate-arbitrage"
    assert cfg["repos"]["qlab"]["prepare"] == [["uv", "run", "qlab", "init-db"]]


def test_a_request_whose_share_runs_out_stops_where_it_is_and_the_other_one_goes_on(
        tmp_path, monkeypatch) -> None:
    """The writer/reviewer loop of a request shares the stage's allowance. When its own part is
    spent it stops cleanly (worktree and remarks kept, queued for the next night) and the night's
    other request still gets its part; only the night's own ceiling stops the whole stage."""
    from qlab.budget import CandidateExhausted

    cfg, seen, held = _stage_world(tmp_path, monkeypatch, ["Блок сигнала: x", "Add a flag"])
    checkpoints: list[str] = []

    def run(req, cfg, *, budget, session, checkpoint=None):
        seen.append({"id": req.id, "cap": budget.cap_tokens})
        checkpoint(req)  # the stage saves the queue after every finished iteration
        checkpoints.append(req.id)
        if req.id == "2026-10-30-1":
            req.iterations = 3
            raise CandidateExhausted("used 250,000 of 250,000 tokens")
        req.attempts += 1
        req.status, req.tokens = "ready", 50_000
        return req

    monkeypatch.setattr(prog, "run_request", run)
    state: dict = {"done": {}}
    real_programmer_stage(DAY, state, config=cfg)
    assert [s["id"] for s in seen] == ["2026-10-30-1", "2026-10-30-2"]  # both got a part
    first, second = state["programmer"]["entries"]
    assert first["status"] == "queued" and first["iterations"] == 3
    assert "после 3 итераций" in first["stopped"]
    assert "продолжит в следующую ночь" in first["stopped"]
    assert second["status"] == "ready" and "stopped" not in state["programmer"]
    assert [r.status for r in held["queue"]] == ["queued", "ready"]
    assert checkpoints == ["2026-10-30-1", "2026-10-30-2"]
    # the report says where the loop is and that it goes on first next night
    text = "\n".join(prog.report_lines([first], held["queue"]))
    assert "доля ночного бюджета на этот запрос израсходована" in text


def test_an_unexpected_crash_is_not_a_failed_attempt_and_the_request_waits_for_the_next_night(
        tmp_path, monkeypatch) -> None:
    cfg, seen, held = _stage_world(tmp_path, monkeypatch, ["Add a flag"])

    def crash(req, cfg, *, budget, session, checkpoint=None):
        req.attempts += 1
        raise OSError("disk full")

    monkeypatch.setattr(prog, "run_request", crash)
    state: dict = {"done": {}}
    real_programmer_stage(DAY, state, config=cfg)
    (request,) = held["queue"]
    assert request.status == "queued" and request.attempts == 0 and request.env_stops == 1
    assert "OSError: disk full" in request.problems[0]
