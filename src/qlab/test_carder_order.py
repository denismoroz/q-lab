"""The order in which the carder takes ideas (qlab.carder_order): ideas that the newest memos name
as needed first, then the owner's own, then by age; every pick says why. Memos and the inbox are
written to a temporary folder; no agent is involved."""

from __future__ import annotations

from datetime import date, timedelta

from qlab import carder_order, ideas

DAY = date(2026, 11, 3)

BV2_MEMO = """# Bv2 (исследователь)

Старый разговор об idea-0001, которой нет в очереди.

## Гипотезы: польза против затрат

1. **Показать в уликах общий залог** — без испытаний.
2. **Быстрое включение** (idea-0006) — новый код, лежит в ящике.
3. **Список кандидатов строит код** (idea-0007) — новый код.

## Чего фреймворк пока не может сказать

1. Общий залог. См. idea-0009.
"""

FRAB_MEMO = """# FRAB

## Гипотезы по пользе и цене

| № | что | польза | цена |
|---|---|---|---|
| 1 | Сверка (idea-0002) | высокая | ответ владельца |
| 3 | Отрез от первого часа (idea-0022) | высокая | код |
| 10 | Другая площадка (idea-0004) | низкая | загрузчики |
"""


def _memo(root, name, day, text) -> None:
    folder = root / "docs" / "research"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}-{day}.md").write_text(text, encoding="utf-8")


def _inbox(*specs) -> list[dict]:
    inbox: list[dict] = []
    for n, (source, status, attempts) in enumerate(specs, 1):
        entry, _ = ideas.add(inbox, f"idea number {n}, long enough to be an idea", source=source,
                             day=date(2026, 10, 28) + timedelta(days=n - 1))
        entry.update(status=status, attempts=attempts)
    return inbox


def _order(inbox, root, strategies=("bv2", "frab"), max_attempts=3):
    return carder_order.order(inbox, max_attempts=max_attempts, strategies=set(strategies),
                              root=root)


def test_a_memo_ranks_its_hypotheses_by_the_number_of_the_row_or_the_item() -> None:
    found = carder_order.memo_mentions(BV2_MEMO + FRAB_MEMO)
    assert found == {"idea-0001": carder_order.UNRANKED, "idea-0006": 2, "idea-0007": 3,
                     "idea-0009": carder_order.UNRANKED, "idea-0002": 1, "idea-0022": 3,
                     "idea-0004": 10}


def test_ideas_named_as_needed_go_first_then_the_owners_then_by_age(tmp_path) -> None:
    _memo(tmp_path, "bv2", "2026-11-01", BV2_MEMO)
    _memo(tmp_path, "frab", "2026-11-02", FRAB_MEMO)
    inbox = _inbox(
        ("agent:researcher", "new", 2),   # idea-0001: the oldest, nobody names it
        ("agent:researcher", "new", 0),   # idea-0002: named by frab, but not in `new`?  see below
        ("agent:researcher", "new", 0),   # idea-0003: nobody names it
        ("owner", "new", 0),              # idea-0004: the owner's own, and named at rank 10
        ("owner", "new", 0),              # idea-0005: the owner's own, nobody names it
        ("agent:researcher", "new", 0),   # idea-0006: named, rank 2
        ("agent:researcher", "new", 0),   # idea-0007: named, rank 3
    )
    inbox[1]["status"] = "needs-owner"  # idea-0002 waits for the owner: not for the carder
    inbox.append({**inbox[0], "id": "idea-0022", "date": "2026-11-03", "status": "new",
                  "attempts": 0, "text": "the idea the frab memo ranks third"})
    got = [(p.idea["id"], p.tier) for p in _order(inbox, tmp_path)]
    assert got == [("idea-0006", "named"), ("idea-0007", "named"), ("idea-0022", "named"),
                   ("idea-0004", "named"),  # ranked 10 in its memo: still before the unranked
                   ("idea-0001", "named"),  # only mentioned in a memo's text
                   ("idea-0005", "owner"), ("idea-0003", "age")]


def test_every_pick_says_why_in_plain_russian(tmp_path) -> None:
    _memo(tmp_path, "bv2", "2026-11-01", BV2_MEMO)
    inbox = _inbox(("agent:researcher", "new", 0), ("owner", "new", 0),
                   ("agent:researcher", "new", 0), ("agent:researcher", "new", 0),
                   ("agent:researcher", "new", 0), ("agent:researcher", "new", 0))
    picks = {p.idea["id"]: p for p in _order(inbox, tmp_path)}
    assert picks["idea-0006"].why == ("названа как нужная: в записке исследователя bv2 от "
                                      "2026-11-01 (пункт 2 списка гипотез)")
    assert picks["idea-0002"].why == "идея самого владельца"
    assert picks["idea-0003"].why == "по возрасту (с 2026-10-30); никто не назвал её нужной"
    assert picks["idea-0001"].why == ("упомянута: в записке исследователя bv2 от 2026-11-01 "
                                      "(в тексте, вне списка гипотез)")


def test_only_the_newest_memo_of_a_strategy_counts_and_only_its_exact_name(tmp_path) -> None:
    _memo(tmp_path, "bv2", "2026-10-29", "## Гипотезы\n\n1. Старая мысль (idea-0001)\n")
    _memo(tmp_path, "bv2", "2026-11-01", BV2_MEMO)
    _memo(tmp_path, "frab-binance-spot", "2026-11-03", "## Гипотезы\n\n1. Чужая (idea-0003)\n")
    _memo(tmp_path, "frab", "2026-11-02", FRAB_MEMO)
    named = carder_order.named_as_needed({"bv2", "frab", "frab-binance-spot"}, tmp_path)
    assert "idea-0001" in named and all(m.day == "2026-11-01" for m in named["idea-0001"])
    assert [m.who for m in named["idea-0003"]] == ["frab-binance-spot"]  # not frab's
    assert {m.who for m in named["idea-0004"]} == {"frab"}


def test_an_idea_named_by_more_files_goes_before_an_equally_ranked_one(tmp_path) -> None:
    _memo(tmp_path, "bv2", "2026-11-01", "## Гипотезы\n\n2. Раз (idea-0002) и два (idea-0001)\n")
    _memo(tmp_path, "frab", "2026-11-02", "## Гипотезы\n\n| 2 | Два (idea-0002) |\n")
    (tmp_path / "night").mkdir()
    (tmp_path / "night/directions.yaml").write_text("note: idea-0002\n", encoding="utf-8")
    inbox = _inbox(("agent:researcher", "new", 0), ("agent:researcher", "new", 0))
    got = _order(inbox, tmp_path)
    assert [p.idea["id"] for p in got] == ["idea-0002", "idea-0001"]
    assert "в направлениях руководителя (night/directions.yaml)" in got[0].why
    assert [m.who for m in got[0].mentions] == ["bv2", "frab", "directions"]


def test_the_director_memo_counts_like_a_researchers(tmp_path) -> None:
    _memo(tmp_path, "director", "2026-11-03", "Нужна карточка для idea-0002.\n")
    inbox = _inbox(("agent:researcher", "new", 0), ("agent:researcher", "new", 0))
    first = _order(inbox, tmp_path, strategies=())[0]
    assert first.idea["id"] == "idea-0002"
    assert first.why == ("упомянута: в записке руководителя исследований от 2026-11-03 "
                         "(в тексте, вне списка гипотез)")


def test_ideas_without_attempts_left_or_not_new_are_not_ordered(tmp_path) -> None:
    inbox = _inbox(("owner", "new", 3), ("owner", "carded", 0), ("owner", "duplicate", 0),
                   ("owner", "new", 2))
    assert [p.idea["id"] for p in _order(inbox, tmp_path)] == ["idea-0004"]
    assert carder_order.order([], max_attempts=3, strategies=set(), root=tmp_path) == []


def test_without_the_flag_the_order_is_the_old_one_by_age(tmp_path) -> None:
    inbox = _inbox(("agent:researcher", "new", 0), ("owner", "new", 0))
    got = carder_order.age_order(inbox, max_attempts=3)
    assert [p.idea["id"] for p in got] == ["idea-0001", "idea-0002"]
