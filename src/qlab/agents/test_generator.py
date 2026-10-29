"""Tests for the idea generator: catalogue parsing, the code-side guards, and
the agent call with a fake `claude` that writes files the way the real one would."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import qlab.agents.generator as gen
import qlab.pipeline.sources as sources
from qlab import fixation
from qlab.budget import ExplicitNoBudget
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.pipeline import stage2
from qlab.registry.models import Base, DataSnapshot

DATA_END = date(2026, 10, 24)

CATALOGUE_TEXT = """# Каталог механизмов (фикстура)

Текст до первого механизма не разбирается.

## Механизмы

### xs-reversal-1w

**Суть:** За неделю выросшие монеты откатываются.
Вторая строка сути.

**Блоки:** universe top_volume; signal reversal; portfolio long_short; sizing equal

**Где проверено:** Binance, 2019-2026.

**Параметры:**

| параметр блока | значения из источника | источник |
|---|---|---|
| signal.lookback_days | 7, 14, 28 | [Author 2020](https://e.org/a), [Other 2021](https://e.org/b) |
| portfolio.top_n | 3, 5 | [Author 2020](https://example.org/a) |
| portfolio.quantiles | 5 | [Author 2020](https://example.org/a) |
| portfolio.fraction | 30% | [Author 2020](https://example.org/a) |
| universe.n | 10, 20 | [Author 2020](https://example.org/a) |
| universe.window_days | 30 (в статье 45 дней) | [Author 2020](https://example.org/a) |
| sizing.gross | 1 | [Author 2020](https://example.org/a) |
| rebalance_days | 7 | [Author 2020](https://example.org/a) |
| sizing.target | 5% | [Author 2020](https://example.org/a) |

**Чего блоки не выражают:** Ничего.

**Источники:** Author 2020; Other 2021.

### tsmom-seasonal

**Суть:** Трендовая стратегия с сезонной поправкой.

**Блоки:** signal momentum; portfolio time_series_sign

**Где проверено:** на акциях.

**Параметры:**

| параметр блока | значения из источника | источник |
|---|---|---|
| signal.lookback_days | 60, 90 | [Third 2019](https://example.org/c) |

**Чего блоки не выражают:** Сезонная поправка по часам суток.

**Источники:** Third 2019.
"""


def _spec_text(*, idea_id: str = "gen-xs-reversal", code_ref: str = gen.CODE_REF,
               lookback: int | float = 7, mechanism: str | None = "xs-reversal-1w",
               signal_source: str | None = None, unexpressed: str = "[]",
               taker_fee: float = 5.0, end: date = DATA_END,
               params_fixed_at: str | None = "named") -> str:
    header = f"# mechanism: {mechanism}\n" if mechanism else "# no header\n"
    if params_fixed_at == "named":  # the day the code names: the start of the last third
        params_fixed_at = fixation.named_date(date(2019, 9, 8), end, 1 / 3).isoformat()
    fixed = (f"params_fixed_at: {params_fixed_at}\n"
             "params_fixed_evidence: \"named by code: the start of the last third of the window\"\n"
             if params_fixed_at else "")
    signal_source = signal_source or (
        "docs/MECHANISMS.md, mechanism xs-reversal-1w, row signal.lookback_days (Author 2020)")
    return f"""{header}idea_id: {idea_id}
title: Откат цены за неделю
code_ref: {code_ref}
params:
  universe:
    rule: top_volume
    n: 10
    window_days: 30
  signal:
    block: reversal
    lookback_days: {lookback}
  portfolio:
    block: long_short
    top_n: 3
  sizing:
    block: equal
    gross: 1
  rebalance_days: 7
  min_history_days: 58
data:
  source: binance
  interval: 1d
  start: 2019-09-08
  end: {end}
costs:
  taker_fee_bps: {taker_fee}
  slippage_bps: 0.9
min_leg_notional: 10.0
simultaneous_legs: 1
noise_universe: panel
unexpressed_mechanisms: {unexpressed}
{fixed}sources:
  universe: "docs/MECHANISMS.md, mechanism xs-reversal-1w, rows universe.n, universe.window_days"
  signal: "{signal_source}"
  portfolio: "docs/MECHANISMS.md, mechanism xs-reversal-1w, row portfolio.top_n (Author 2020)"
  sizing: "docs/MECHANISMS.md, mechanism xs-reversal-1w, row sizing.gross (Author 2020)"
  rebalance_days: "docs/MECHANISMS.md, mechanism xs-reversal-1w, row rebalance_days (Author 2020)"
  min_history_days: "derived: 28 + 30; the longest lookback plus the window (docs/MECHANISMS.md)"
  costs.taker_fee_bps: "night/costs.yaml, venue binance, taker_fee_bps"
  costs.slippage_bps: "night/costs.yaml, venue binance, slippage_bps"
  min_leg_notional: "owner 2026-10-25: test fixture"
"""


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A throwaway project root: a git repo holding the fixture catalogue."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "MECHANISMS.md").write_text(CATALOGUE_TEXT, encoding="utf-8")
    (root / "night").mkdir()  # the cost table, which the specs' costs cite
    (root / "night" / "costs.yaml").write_text(
        (Path(__file__).resolve().parents[3] / "night" / "costs.yaml").read_text(encoding="utf-8"),
        encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=root, check=True)
    monkeypatch.setattr(gen, "QLAB_ROOT", root)
    monkeypatch.setattr("qlab.agents.implementer.QLAB_ROOT", root)
    monkeypatch.setattr(sources, "QLAB_ROOT", root)
    return root


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _check(repo: Path, text: str, *, name: str = "gen-xs-reversal", **kw):
    path = gen.SPEC_DIR / f"{name}.yaml"
    (repo / gen.SPEC_DIR).mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text, encoding="utf-8")
    kw.setdefault("data_end", DATA_END)
    return gen.check_generated(path, mechanisms=gen.catalogue(), reference=None, **kw)


# ---- the catalogue -------------------------------------------------------


def test_catalogue_is_parsed_into_mechanisms_and_parameter_rows(repo) -> None:
    mechanisms = gen.catalogue(Path("docs/MECHANISMS.md"))
    assert [m.id for m in mechanisms] == ["xs-reversal-1w", "tsmom-seasonal"]
    first = mechanisms[0]
    assert first.summary.startswith("За неделю выросшие монеты откатываются.")
    assert "Вторая строка сути." in first.summary
    assert first.blocks.startswith("universe top_volume")
    rows = {r.param: r for r in first.params}
    assert rows["signal.lookback_days"].values == (7.0, 14.0, 28.0)
    assert rows["signal.lookback_days"].values_text == "7, 14, 28"
    assert "Author 2020" in rows["signal.lookback_days"].source
    assert rows["sizing.target"].values == (5.0, 0.05)  # a percentage also reads as a fraction
    assert len(first.params) == 9
    assert first.fully_expressed and first.sources == "Author 2020; Other 2021."
    second = mechanisms[1]
    assert not second.fully_expressed and "Сезонная" in second.unexpressed
    assert [r.param for r in second.params] == ["signal.lookback_days"]


def test_the_real_catalogue_parses_when_it_exists() -> None:
    path = Path(__file__).resolve().parents[3] / "docs" / "MECHANISMS.md"
    if not path.is_file():
        pytest.skip("docs/MECHANISMS.md is written by another task")
    mechanisms = gen.parse_catalogue(path.read_text(encoding="utf-8"))
    assert mechanisms and all(m.id and m.blocks for m in mechanisms)


def test_cell_numbers() -> None:
    assert gen.cell_numbers("7, 14, 28") == (7.0, 14.0, 28.0)
    assert gen.cell_numbers("12-1 и 0.5") == (12.0,)  # values lead the cell; the rest is a note
    assert gen.cell_numbers("нет чисел") == ()


# ---- the guards ----------------------------------------------------------


def test_a_good_spec_passes_every_guard(repo) -> None:
    item, problems = _check(repo, _spec_text())
    assert problems == []
    assert item.mechanism == "xs-reversal-1w" and item.spec.idea_id == "gen-xs-reversal"


def test_a_number_that_is_not_in_the_catalogue_is_rejected(repo) -> None:
    _, problems = _check(repo, _spec_text(lookback=99))
    assert any("99" in p and "does not appear" in p for p in problems)
    assert any("not among the catalogue's values" in p for p in problems)


def test_a_number_in_the_catalogue_but_not_in_its_row_is_rejected(repo) -> None:
    # 5 is in the file (portfolio.top_n) but the row for signal.lookback_days is 7, 14, 28.
    _, problems = _check(repo, _spec_text(lookback=5))
    assert not any("does not appear" in p for p in problems)
    assert any("signal.lookback_days" in p and "not among" in p for p in problems)


def test_a_value_not_in_the_rows_value_list_is_rejected(repo) -> None:
    text = _spec_text().replace("top_n: 3", "top_n: 4")  # the row gives 3, 5
    _, problems = _check(repo, text)
    assert any("portfolio.top_n" in p and "not among the catalogue's values" in p
               and "(3, 5)" in p for p in problems)


def test_a_parameter_without_a_row_is_rejected_even_if_its_number_is_in_the_catalogue(
        repo) -> None:
    # The fixture's sizing.window_days has no row; 30 is in the file (universe.window_days),
    # 0.05 too (sizing.target: 5%), 1 as well (sizing.gross) -- none of that counts.
    sizing = "  sizing:\n    block: vol_target\n    window_days: 30\n    target_daily: 0.05\n" \
             "    leverage_cap: 1\n"
    text = _spec_text().replace("  sizing:\n    block: equal\n    gross: 1\n", sizing)
    _, problems = _check(repo, text)
    for path in ("sizing.window_days", "sizing.target_daily", "sizing.leverage_cap"):
        assert any(path in p and f"the catalogue gives no value for {path} in xs-reversal-1w" in p
                   for p in problems), path
    assert not any("does not appear" in p for p in problems)  # the old guard let these through


def test_a_row_without_values_gives_nothing_to_use(repo) -> None:
    mechanisms = [dataclasses.replace(m, params=tuple(
        dataclasses.replace(r, values=()) if r.param == "rebalance_days" else r
        for r in m.params)) for m in gen.catalogue()]
    path = gen.SPEC_DIR / "gen-xs-reversal.yaml"
    (repo / gen.SPEC_DIR).mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(_spec_text(), encoding="utf-8")
    _, problems = gen.check_generated(path, mechanisms=mechanisms, reference=None,
                                      data_end=DATA_END)
    assert any("no value for rebalance_days" in p for p in problems)


def _without_gross_row() -> list[gen.Mechanism]:
    return [dataclasses.replace(m, params=tuple(r for r in m.params if r.param != "sizing.gross"))
            for m in gen.catalogue()]


def _check_without_gross_row(repo: Path, text: str):
    path = gen.SPEC_DIR / "gen-xs-reversal.yaml"
    (repo / gen.SPEC_DIR).mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text, encoding="utf-8")
    return gen.check_generated(path, mechanisms=_without_gross_row(), reference=None,
                               data_end=DATA_END)


def test_sizing_gross_one_is_the_framework_value_and_nothing_else_is(repo) -> None:
    assert gen.FRAMEWORK_VALUES == {"sizing.gross": (1.0,)}
    # No sizing.gross row at all: 1.0 (and 1) pass, 2.0 does not.
    for one in ("1.0", "1"):
        _, problems = _check_without_gross_row(
            repo, _spec_text().replace("gross: 1\n", f"gross: {one}\n"))
        assert problems == []
    _, problems = _check_without_gross_row(repo, _spec_text().replace("gross: 1\n", "gross: 2.0\n"))
    assert any("sizing.gross" in p and "2.0" in p for p in problems)
    # With the row present, its own values do not widen the framework's: 2.0 is still off.
    _, problems = _check(repo, _spec_text().replace("gross: 1\n", "gross: 2.0\n"))
    assert any("sizing.gross" in p and "not among" in p for p in problems)


def test_a_derived_formula_may_use_only_values_from_the_mechanisms_rows(repo) -> None:
    def with_formula(formula: str, result: int):
        text = _spec_text().replace("derived: 28 + 30;", f"derived: {formula};").replace(
            "min_history_days: 58", f"min_history_days: {result}")
        return _check(repo, text)[1]

    assert with_formula("14 + 30", 44) == []  # 14: signal.lookback_days row, 30: universe row
    # 60 is a value of another mechanism (tsmom-seasonal): it appears in the file, not in this one.
    problems = with_formula("60 + 30", 90)
    assert any("min_history_days" in p and "60" in p and "xs-reversal-1w" in p for p in problems)
    # 45 stands in a note of this mechanism's own row ("в статье 45 дней"): commentary, not a value.
    problems = with_formula("45 + 28", 73)
    assert any("min_history_days" in p and "45" in p for p in problems)
    # A number from nowhere else in the catalogue fails too.
    assert any("31" in p for p in with_formula("28 + 31", 59))


def test_a_quantile_portfolio_with_the_rows_value_is_accepted(repo) -> None:
    text = _spec_text().replace("block: long_short\n    top_n: 3",
                                "block: long_short_quantile\n    quantiles: 5")
    text = text.replace("row portfolio.top_n (Author 2020)",
                        "row portfolio.quantiles (Author 2020)")
    item, problems = _check(repo, text)
    assert problems == [] and item.spec.params["portfolio"]["quantiles"] == 5
    _, problems = _check(repo, text.replace("quantiles: 5", "quantiles: 4"))
    assert any("portfolio.quantiles" in p and "not among" in p for p in problems)
    # top_n has a row in the fixture, so it is allowed here; without a row it would not be.
    assert "long_short_quantile" in gen.INSTRUCTIONS and "portfolio.quantiles" in gen.INSTRUCTIONS


def test_the_fraction_cap_weight_and_carry_blocks_are_checked_by_their_parameter_path(repo) -> None:
    base = _spec_text()
    fraction = base.replace("block: long_short\n    top_n: 3",
                            "block: long_short_fraction\n    fraction: 0.3").replace(
        "row portfolio.top_n (Author 2020)", "row portfolio.fraction (Author 2020)")
    item, problems = _check(repo, fraction)
    assert problems == [] and item.spec.params["portfolio"]["fraction"] == 0.3
    _, problems = _check(repo, fraction.replace("fraction: 0.3", "fraction: 0.2"))
    assert any("portfolio.fraction" in p and "not among" in p for p in problems)
    # cap_weight takes only sizing.gross, the framework's 1.0 (any other value is rejected)
    capped = base.replace("block: equal", "block: cap_weight")
    assert _check(repo, capped)[1] == []
    doubled = _check(repo, capped.replace("gross: 1\n", "gross: 2\n"))[1]
    assert any("sizing.gross" in p for p in doubled)
    # spot_perp_carry takes portfolio.top_n, from the mechanism's own row
    carry = base.replace("block: long_short\n", "block: spot_perp_carry\n")
    assert _check(repo, carry)[1] == []
    four = _check(repo, carry.replace("top_n: 3", "top_n: 4"))[1]
    assert any("portfolio.top_n" in p for p in four)
    for name in ("long_short_fraction", "long_only_fraction", "portfolio.fraction", "cap_weight",
                 "spot_perp_carry", "portfolio.top_n"):
        assert name in gen.INSTRUCTIONS


def test_a_number_not_cited_to_the_catalogue_is_rejected(repo) -> None:
    _, problems = _check(repo, _spec_text(signal_source="https://example.org/paper (Author 2020)"))
    assert any("signal.lookback_days" in p and "does not cite docs/MECHANISMS.md" in p
               for p in problems)


def test_a_derived_number_must_use_catalogue_numbers(repo) -> None:
    text = _spec_text().replace("derived: 28 + 30;", "derived: 28 + 31;").replace(
        "min_history_days: 58", "min_history_days: 59")
    _, problems = _check(repo, text)
    assert any("min_history_days" in p and "31" in p for p in problems)


def test_a_wrong_code_ref_is_rejected(repo) -> None:
    _, problems = _check(repo, _spec_text(code_ref="qlab.strategies.trend:TrendTSMOMEnsemble"))
    assert any("code_ref" in p and gen.CODE_REF in p for p in problems)


def test_header_prefix_name_and_mechanism_are_required(repo) -> None:
    _, problems = _check(repo, _spec_text(mechanism=None))
    assert any("header" in p for p in problems)
    _, problems = _check(repo, _spec_text(mechanism="made-up"))
    assert any("made-up" in p and "not in the catalogue" in p for p in problems)
    _, problems = _check(repo, _spec_text(idea_id="xs-reversal"), name="xs-reversal")
    assert any("does not start with 'gen-'" in p for p in problems)
    _, problems = _check(repo, _spec_text(idea_id="gen-other"), name="gen-xs-reversal")
    assert any("file name" in p for p in problems)


def test_data_window_and_ownership_of_special_fields(repo) -> None:
    _, problems = _check(repo, _spec_text(end=date(2026, 9, 1)))
    assert any("data.end" in p for p in problems)
    text = _spec_text().replace("start: 2019-09-08", "start: 2018-01-01")
    _, problems = _check(repo, text)
    assert any("data.start" in p for p in problems)
    text = _spec_text().replace("source: binance", "source: hyperliquid")
    _, problems = _check(repo, text)
    assert any("data must be binance 1d" in p for p in problems)
    _, problems = _check(repo, _spec_text() + 'exploratory: "I allow it"\n')
    assert any("exploratory" in p for p in problems)
    _, problems = _check(repo, _spec_text() + "required_instruments: [BTC]\n")
    assert any("hand-picked" in p for p in problems)


def test_an_omitted_params_fixed_date_is_a_problem_and_code_names_it(repo) -> None:
    """Owner, 2026-10-29 (В9): the day is named by code, an omitted one is not the whole window."""
    _, problems = _check(repo, _spec_text(params_fixed_at=None))
    assert any("params_fixed_at is missing" in p for p in problems)
    path = repo / gen.SPEC_DIR / "gen-xs-reversal.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_spec_text(params_fixed_at=None), encoding="utf-8")
    note = fixation.normalise_file(path)
    assert note is not None and "named by code" in note
    item, problems = _check(repo, path.read_text(encoding="utf-8"))
    assert problems == []
    assert item.spec.params_fixed_at == fixation.named_date(date(2019, 9, 8), DATA_END, 1 / 3)


def test_an_earlier_day_than_the_named_one_is_a_problem(repo) -> None:
    _, problems = _check(repo, _spec_text(params_fixed_at="2019-09-08"))
    assert any("earlier than the day the code names" in p for p in problems)


def test_a_later_day_needs_a_source(repo) -> None:
    _, problems = _check(repo, _spec_text(params_fixed_at="2026-06-01"))
    assert any("needs sources['params_fixed_at']" in p for p in problems)
    text = _spec_text(params_fixed_at="2026-06-01") + (
        '  params_fixed_at: "docs/MECHANISMS.md, mechanism xs-reversal-1w: Author 2020 sample '
        'ends 2026-05-31"\n')
    _, problems = _check(repo, text)
    assert not any("params_fixed_at" in p for p in problems), problems


@pytest.mark.cost_table
def test_costs_must_be_the_venues_row_of_the_cost_table(repo) -> None:
    """Owner, 2026-10-29 (В7): the agent does not choose costs; a spec with other numbers fails
    the sources check, and the deviation sentence is not the agent's to write."""
    _, problems = _check(repo, _spec_text(taker_fee=1.0))
    assert any("not the table's row of binance" in p for p in problems)
    text = _spec_text(taker_fee=1.0) + (
        '  costs.differs_from_table: "owner 2026-10-29: the agent says so"\n')
    _, problems = _check(repo, text)
    assert any("not the agent's to write" in p for p in problems)


def test_a_mechanism_the_blocks_do_not_fully_express_must_say_so(repo) -> None:
    _, problems = _check(repo, _spec_text(mechanism="tsmom-seasonal"))
    assert any("do not express the whole mechanism" in p for p in problems)
    # Differences from the paper go to differs_from_source, which does not block
    # the stand; unexpressed_mechanisms would make the run not evaluable.
    declared = _spec_text(mechanism="tsmom-seasonal").replace(
        "unexpressed_mechanisms: []", "unexpressed_mechanisms: []\ndiffers_from_source: ["
        "\"seasonal correction\"]")
    _, problems = _check(repo, declared)
    assert not any("do not express" in p for p in problems)
    misplaced = _spec_text(mechanism="tsmom-seasonal", unexpressed='["seasonal correction"]')
    _, problems = _check(repo, misplaced)
    assert any("unexpressed_mechanisms must stay empty" in p for p in problems)


def test_duplicates_of_the_registry_and_of_existing_specs_are_rejected(repo) -> None:
    _, problems = _check(repo, _spec_text(), registry_idea_ids=frozenset({"gen-xs-reversal"}))
    assert any("already in the registry" in p for p in problems)
    key = (gen.CODE_REF, json.dumps(yaml.safe_load(_spec_text())["params"], sort_keys=True,
                                    default=str))
    _, problems = _check(repo, _spec_text(), other_params={key: "gen-older"})
    assert any("same params as the already existing gen-older" in p for p in problems)


# ---- the agent call ------------------------------------------------------


def _fake_claude(tmp_path: Path, monkeypatch, body: str) -> Path:
    """A `claude` that records its arguments outside the repo, runs `body` in
    the repo and prints a result the usage parser understands."""
    reset = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"type": "rate_limit_event", "rate_limit_info": {
            "status": "allowed", "unifiedWindows": {"seven_day": {
                "utilization": 0.2, "resetsAt": reset}}}}) + "\n"
        + json.dumps({"type": "result", "is_error": False, "result": "gen-good: xs-reversal-1w",
                      "total_cost_usd": 0.01, "usage": {"input_tokens": 100, "output_tokens": 10},
                      "modelUsage": {"claude-sonnet": {}}}) + "\n")
    args_file = tmp_path / "args.txt"
    script = tmp_path / "claude"
    script.write_text(f'#!/bin/sh\necho "$@" > "{args_file}"\n{body}\ncat "{out}"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    return args_file


def test_generate_keeps_valid_specs_and_reverts_everything_else(
        repo, session, tmp_path, monkeypatch) -> None:
    (repo / gen.SPEC_DIR).mkdir(parents=True)
    old = repo / gen.SPEC_DIR / "gen-old.yaml"
    old.write_text(_spec_text(idea_id="gen-old", lookback=14), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "o"],
                   cwd=repo, check=True)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "good.yaml").write_text(_spec_text(idea_id="gen-good"), encoding="utf-8")
    (payload / "ref.yaml").write_text(
        _spec_text(idea_id="gen-badref", lookback=14, code_ref="qlab.strategies.x:Y"),
        encoding="utf-8")
    (payload / "extra.yaml").write_text(
        _spec_text(idea_id="gen-extra", lookback=28), encoding="utf-8")
    args_file = _fake_claude(tmp_path, monkeypatch, f"""
cp "{payload}/good.yaml" specs/generated/gen-good.yaml
cp "{payload}/ref.yaml" specs/generated/gen-badref.yaml
cp "{payload}/extra.yaml" specs/generated/gen-extra.yaml
echo tampered > specs/generated/gen-old.yaml
""")
    outcome = gen.generate(budget=ExplicitNoBudget("test"), session=session, data_end=DATA_END,
                           max_specs=1, registry_idea_ids=["something-else"])

    # Sorted: gen-badref fails a guard, gen-extra is the one accepted, gen-good is over max_specs.
    assert [a.spec.idea_id for a in outcome.accepted] == ["gen-extra"]
    rejected = {r.path: r.problems for r in outcome.rejected}
    assert any("code_ref" in p for p in rejected["specs/generated/gen-badref.yaml"])
    assert any("more than 1 specs" in p for ps in rejected.values() for p in ps)
    # An edited existing spec is restored.
    assert old.read_text(encoding="utf-8") == _spec_text(idea_id="gen-old", lookback=14)
    assert any("gen-old.yaml" in r for r in outcome.reverted)
    # The agent runs on sonnet with read-only tools plus Write, through the guard.
    args = args_file.read_text()
    assert "--model sonnet" in args and "Read,Glob,Grep,Write" in args
    assert "--disallowedTools Bash,Edit" in args and "--max-turns" in args
    context = (repo / gen.CONTEXT_DIR / f"generate-{datetime.now(UTC).date()}.md").read_text()
    assert "gen-old" in context and "something-else" in context and str(DATA_END) in context
    spend = session.execute(text("select agent, status from token_spend")).all()
    assert spend == [("generator", "ok; no budget: test")]


def test_generate_without_a_budget_fails_instead_of_spending(repo, session) -> None:
    with pytest.raises(TypeError, match="run_agent needs a CandidateBudget"):
        gen.generate(budget=None, session=session, data_end=DATA_END)  # type: ignore[arg-type]


def test_an_empty_catalogue_does_not_call_the_agent(repo, session, monkeypatch) -> None:
    (repo / "docs" / "MECHANISMS.md").write_text("# nothing yet\n", encoding="utf-8")
    monkeypatch.setenv(CLAUDE_BIN_ENV, "/nonexistent/claude")  # a call would raise
    outcome = gen.generate(budget=ExplicitNoBudget("test"), session=session, data_end=DATA_END)
    assert outcome.accepted == [] and "the agent was not called" in outcome.problems[0]


# ---- the stand -----------------------------------------------------------


def test_the_stand_runs_every_spec_with_noise_and_records_the_idea(
        repo, monkeypatch, tmp_path) -> None:
    import qlab.calibration.shape_aware as shape_aware
    import qlab.registry.db as db_module
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea, SourceType

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    item, problems = _check(repo, _spec_text())
    assert problems == []
    calls = []

    class _Routing:
        route = "needs-forward"
        reason = "waiting"

    class _Candidate:
        trial_id = 77
        routing = _Routing()
        metrics = {"ann_return_net": 0.1, "sharpe_net": 0.8, "other": 1}

    class _Out:
        candidate = _Candidate()

    def fake_eval(spec, *, session, ruleset, n_trials):
        calls.append((spec.idea_id, n_trials))
        return _Out()

    monkeypatch.setattr(shape_aware, "evaluate_spec_with_shape_aware_bar", fake_eval)
    second = []
    monkeypatch.setattr(stage2, "run_stage2", lambda spec, **kw: second.append((spec.idea_id, kw))
                        or stage2.Stage2Result(parent_id=spec.idea_id, idea_id="x-hl",
                                               stage1_route=kw["stage1_route"], status="skipped",
                                               skipped="stub"))
    results = gen.run_generated([item], noise_trials=5)
    assert calls == [("gen-xs-reversal", 5)]
    # The second stage runs in the same call, from the stage 1 route and trial.
    assert [(i, kw["stage1_route"], kw["stage1_trial_id"], kw["noise_trials"]) for i, kw in second
            ] == [("gen-xs-reversal", "needs-forward", 77, 5)]
    assert results[0].stage2["status"] == "skipped"
    assert (results[0].trial_id, results[0].route) == (77, "needs-forward")
    assert results[0].metrics == {"ann_return_net": 0.1, "sharpe_net": 0.8}
    assert "прогон 77" in gen.result_line(results[0])
    with session_scope() as session:
        idea = session.get(Idea, "gen-xs-reversal")
        assert idea.source_type == SourceType.INTERNAL
        assert idea.notes == "Generated from docs/MECHANISMS.md (xs-reversal-1w)"

    def failing_eval(spec, **_kw):
        raise RuntimeError("no data")

    monkeypatch.setattr(shape_aware, "evaluate_spec_with_shape_aware_bar", failing_eval)
    second.clear()
    failed = gen.run_generated([item], noise_trials=5)[0]
    assert failed.error == "RuntimeError: no data" and "прогон упал" in gen.result_line(failed)
    assert second == [] and failed.stage2 is None  # no stage 1 verdict, nothing to confirm
    db_module._engine = None
    db_module._SessionLocal = None


def test_the_second_stage_lines_follow_the_result_line() -> None:
    result = gen.GeneratedResult(idea_id="gen-a", mechanism="m", spec_path="p", trial_id=9,
                                 route="reject")
    assert gen.second_stage_lines(result) == []  # recorded before stage 2 existed
    result.stage2 = {"status": "skipped", "skipped": "stage 1 routed reject: only survivors go on",
                     "stage1_route": "reject"}
    assert gen.second_stage_lines(result)[0].startswith(
        "второй этап (тот же механизм на Hyperliquid) не проводился")


def test_run_second_stage_stores_the_step_as_a_dict_on_the_result(monkeypatch) -> None:
    seen = {}

    def fake(spec, **kw):
        seen.update(kw)
        return stage2.Stage2Result(parent_id=spec.idea_id, idea_id="gen-a-hl", status="evaluated",
                                   route="reject", trial_id=12)

    monkeypatch.setattr(stage2, "run_stage2", fake)
    item = types.SimpleNamespace(spec=types.SimpleNamespace(idea_id="gen-a"))
    result = gen.GeneratedResult(idea_id="gen-a", mechanism="m", spec_path="p", trial_id=9,
                                 route="paper")
    gen.run_second_stage(item, result, noise_trials=7)
    assert (seen["stage1_route"], seen["stage1_trial_id"], seen["noise_trials"]) == ("paper", 9, 7)
    assert "deployable_capital_usd" not in seen  # no common sum any more
    assert result.stage2["trial_id"] == 12 and result.stage2["route"] == "reject"
    # The result survives the night's saved state: plain data in, the same result out.
    assert json.loads(json.dumps(dataclasses.asdict(result)))["stage2"]["idea_id"] == "gen-a-hl"


def test_latest_binance_end_reads_the_registry(session) -> None:
    assert gen.latest_binance_end(session) is None
    for source, end in (("binance", date(2026, 10, 20)), ("binance", date(2026, 10, 24)),
                        ("hyperliquid", date(2026, 10, 25))):
        session.add(DataSnapshot(id=f"{source}{end}", source=source, instruments={},
                                 range_start=date(2019, 9, 8), range_end=end,
                                 fetched_at=datetime.now(UTC), path="x", rows=1))
    session.flush()
    assert gen.latest_binance_end(session) == date(2026, 10, 24)


def test_cell_values_ignore_notes_in_parentheses_and_trailing_text():
    # The real catalogue's cells: a note in parentheses or after the values is
    # commentary; reading its numbers would let a spec use 4 or 26 days.
    from qlab.agents.generator import cell_numbers

    assert cell_numbers("7 (1 неделя), 14 (2 недели), 21 (3 недели), 28 (4 недели) — значимо; "
                        "56 (8 недель), 700 (100 недель)") == (7.0, 14.0, 21.0, 28.0)
    assert cell_numbers("182 (26 недель: монета должна существовать не менее T = 26 weeks)") \
        == (182.0,)
    assert cell_numbers("365 (12 месяцев; месяц = 30.4 дня не применяем — указан год); другие "
                        "сроки таблицы 2 (один, три, шесть)") == (365.0,)
    assert cell_numbers("7 (доходность на следующей неделе, таблица 14, панель B, столбец 1; "
                        "таблица 15), 1 (доходность на следующий день)") == (7.0, 1.0)
    assert cell_numbers("— (источники окна не задают)") == ()
    assert cell_numbers("5%") == (5.0, 0.05)



def test_a_formula_may_set_only_min_history_days(repo) -> None:
    # Catalogue values combined by a formula elsewhere would give numbers no paper used.
    cited = ('rebalance_days: "docs/MECHANISMS.md, mechanism xs-reversal-1w, '
             'row rebalance_days (Author 2020)"')
    text = _spec_text().replace(cited, 'rebalance_days: "derived: 7 + 7; two weeks"')
    text = text.replace("rebalance_days: 7\n", "rebalance_days: 14\n")
    _, problems = _check(repo, text)
    assert any("rebalance_days" in p and "only" in p and "min_history_days" in p for p in problems)




def test_values_of_a_parameter_come_from_all_its_rows(repo) -> None:
    # The real catalogue gives rebalance_days per source (7 from one paper, 14 from
    # another); keyed by param only the last row survived (first real run, 2026-10-25).
    row = "| rebalance_days | 7 | [Author 2020](https://example.org/a) |"
    catalogue = repo / "docs" / "MECHANISMS.md"
    text = catalogue.read_text(encoding="utf-8")
    assert row in text
    catalogue.write_text(text.replace(
        row, "| rebalance_days | 14 | [Other 2021](https://example.org/b) |\n" + row, 1),
        encoding="utf-8")
    spec = _spec_text().replace("rebalance_days: 7\n", "rebalance_days: 14\n")
    _, problems = _check(repo, spec)
    assert not any("rebalance_days" in p for p in problems)
    _, problems = _check(repo, _spec_text())  # 7, from the other row, still allowed
    assert not any("rebalance_days" in p for p in problems)



def test_a_change_outside_its_place_stops_the_run_and_is_not_reverted(
        repo, session, tmp_path, monkeypatch) -> None:
    # 2026-10-25: reverting "outside" changes destroyed another agent's uncommitted
    # docs/VENUES.md; whose change it is cannot be told, so nothing runs instead.
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "good.yaml").write_text(_spec_text(idea_id="gen-good"), encoding="utf-8")
    _fake_claude(tmp_path, monkeypatch, f"""
mkdir -p specs/generated
cp "{payload}/good.yaml" specs/generated/gen-good.yaml
echo stray > STRAY.txt
mkdir -p specs/generated/nested
cp "{payload}/good.yaml" specs/generated/nested/gen-nested.yaml
""")
    outcome = gen.generate(budget=ExplicitNoBudget("test"), session=session, data_end=DATA_END)

    assert outcome.accepted == []
    assert (repo / "STRAY.txt").exists()  # someone else's work would survive
    assert any("changed during the run" in p and "STRAY.txt" in p for p in outcome.problems)
    assert all("outside its place" in r.problems[0] for r in outcome.rejected)


# ---- the research director's directions (night/directions.yaml) -----------------------


def _write_directions(repo: Path, *, pending: bool, pause: str = "xs-reversal-1w") -> None:
    from qlab import directions as dirs

    parsed = dirs.parse_block(
        f"generator:\n  prefer_mechanisms: [tsmom-seasonal]\n  pause_mechanisms:\n"
        f"    - {{id: {pause}, why: \"every idea of it fails the same rule\"}}\n",
        mechanism_ids={"xs-reversal-1w", "tsmom-seasonal"}, strategy_names=set(),
        family_names=set(), day=date(2026, 10, 26), pending_owner_review=pending)
    dirs.write_directions(parsed.directions, repo)


def test_a_spec_of_a_paused_mechanism_is_rejected(repo) -> None:
    _, problems = _check(repo, _spec_text(), paused={"xs-reversal-1w": "fails the same rule"})
    assert any("paused by the research director" in p and "fails the same rule" in p
               for p in problems)
    _, problems = _check(repo, _spec_text(), paused={"another": "x"})
    assert not any("paused" in p for p in problems)
    _, problems = _check(repo, _spec_text())
    assert not any("paused" in p for p in problems)


def test_reviewed_directions_reach_the_prompt_and_the_guard(
        repo, session, tmp_path, monkeypatch) -> None:
    _write_directions(repo, pending=False)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "good.yaml").write_text(_spec_text(idea_id="gen-good"), encoding="utf-8")
    args_file = _fake_claude(tmp_path, monkeypatch,
                             f'mkdir -p specs/generated\ncp "{payload}/good.yaml" '
                             "specs/generated/gen-good.yaml")
    outcome = gen.generate(budget=ExplicitNoBudget("test"), session=session, data_end=DATA_END)
    args = args_file.read_text()
    assert "research director's directions" in args and "tsmom-seasonal" in args
    assert "xs-reversal-1w -- every idea of it fails the same rule" in args
    assert outcome.accepted == []  # the spec is of the paused mechanism
    assert any("paused by the research director" in p
               for r in outcome.rejected for p in r.problems)


def test_directions_waiting_for_the_owner_change_nothing(
        repo, session, tmp_path, monkeypatch) -> None:
    _write_directions(repo, pending=True)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "good.yaml").write_text(_spec_text(idea_id="gen-good"), encoding="utf-8")
    args_file = _fake_claude(tmp_path, monkeypatch,
                             f'mkdir -p specs/generated\ncp "{payload}/good.yaml" '
                             "specs/generated/gen-good.yaml")
    outcome = gen.generate(budget=ExplicitNoBudget("test"), session=session, data_end=DATA_END)
    assert "research director" not in args_file.read_text()
    assert [a.spec.idea_id for a in outcome.accepted] == ["gen-good"]


def test_without_directions_the_generator_is_unchanged(
        repo, session, tmp_path, monkeypatch) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "good.yaml").write_text(_spec_text(idea_id="gen-good"), encoding="utf-8")
    args_file = _fake_claude(tmp_path, monkeypatch,
                             f'mkdir -p specs/generated\ncp "{payload}/good.yaml" '
                             "specs/generated/gen-good.yaml")
    outcome = gen.generate(budget=ExplicitNoBudget("test"), session=session, data_end=DATA_END)
    assert "research director" not in args_file.read_text()
    assert [a.spec.idea_id for a in outcome.accepted] == ["gen-good"]
