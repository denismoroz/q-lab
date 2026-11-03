"""Putting the words of one idea together: a family text filled with the spec's numbers, the
sentence about the blocks of a composed strategy, the sentence about the money (docs/PLAIN_PAGE.md).

A family text is written once (`strategies/plain/<family>.yaml`) with `{placeholders}`; the
placeholders are FACTS read from the spec's core parameters (`FACTS`). A sentence whose fact the
spec does not give is dropped, never half-filled: a text the owner reads has no holes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from qlab.plain.specs import SpecView, Unwrapped, candidate_values, core_items
from qlab.plain.texts import Family, Texts, placeholders
from qlab.plain.words import Words, money, num, percent

Fact = Callable[[Mapping[str, Any]], "str | None"]

_WEEKDAYS = ("понедельникам", "вторникам", "средам", "четвергам", "пятницам", "субботам",
             "воскресеньям")


def _join(items: list[str]) -> str:
    if len(items) > 1:
        return ", ".join(items[:-1]) + f" и {items[-1]}"
    return items[0] if items else ""


def _numbers(values: Any) -> str | None:
    if not isinstance(values, list | tuple) or not values:
        return None
    return _join([num(v) for v in values])


def _pct(core: Mapping[str, Any], key: str) -> str | None:
    value = core.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return num(round(value * 100, 6))
    return None


def _coins(core: Mapping[str, Any]) -> str | None:
    coins = core.get("coins")
    if isinstance(coins, list) and coins:
        return ", ".join(str(c) for c in coins)
    columns = core.get("spot_columns")  # the transcription names its coins by their columns
    if isinstance(columns, dict) and columns:
        return ", ".join(str(c) for c in columns)
    universe = core.get("universe")
    if isinstance(universe, dict) and universe.get("n"):
        return (f"до {num(universe['n'])} самых оборотистых монет (набор пересматривается "
                "по ходу истории)")
    return None


def _trend_universe(core: Mapping[str, Any]) -> str | None:
    by_cap, by_volume = _number(core, "top_k_by_market_cap"), _number(core, "top_k_by_volume")
    cap_text = by_volume_text = None
    if by_cap:
        cap_text = f"{by_cap} самых дорогих по рыночной стоимости монет"
        if "stablecoin" in (core.get("exclude_tags") or []):
            cap_text += " без привязанных к доллару"
    if by_volume:
        by_volume_text = f"{by_volume} монет с самым большим оборотом"
    if cap_text and by_volume_text:  # the spec lets code choose between the two rules
        return f"{cap_text} либо {by_volume_text} — какой из двух наборов, выбирает код"
    if cap_text or by_volume_text:
        return cap_text or by_volume_text
    coins = core.get("coins")
    if isinstance(coins, list) and coins:
        return f"{len(coins)} монет из списка"
    if "coins" in core or "lookbacks" in core or "lookbacks_days" in core:
        return "все монеты площадки с достаточной историей"
    return None


def _lookbacks(core: Mapping[str, Any]) -> str | None:
    return _numbers(core.get("lookbacks") or core.get("lookbacks_days"))


def _book_vol(core: Mapping[str, Any]) -> str | None:
    target = _pct(core, "book_vol_target_ann")
    if target is None:
        return None
    return (f"Кроме того, размер всей книги подгоняют так, чтобы она скакала примерно на {target}% "
            "в год: когда она скачет сильнее, её уменьшают, когда спокойнее — увеличивают.")


def _frab_universe(core: Mapping[str, Any]) -> str | None:
    universe = core.get("universe")
    if universe == "all_paired":
        return "все монеты, у которых есть и контракт на Hyperliquid, и пара на Binance"
    if isinstance(universe, list) and universe:
        return ", ".join(str(c) for c in universe)
    columns = core.get("spot_columns")  # the transcription names its coins by their columns
    if isinstance(columns, dict) and columns:
        return ", ".join(str(c) for c in columns)
    return None


def _days_rhythm(days: Any) -> str:
    if days == 1:
        return "каждый день"
    if days == 7:
        return "раз в неделю"
    whole = isinstance(days, int) and days % 10 == 1 and days % 100 != 11
    return f"раз в {num(days)} {'сутки' if whole else 'суток'}"


def _hours_to_days(hours: Any) -> str | None:
    if isinstance(hours, int | float) and hours > 0:
        days = hours / 24
        return f"{num(days)} {'сутки' if days == 1 else 'суток'}"
    return None


def _exit_sentence(core: Mapping[str, Any]) -> str | None:
    if "phase2_exit_threshold" not in core and "exit_threshold_apr" not in core:
        return None
    if "exit_threshold_apr" in core and "phase2_exit_threshold" not in core:
        floor = _pct(core, "exit_threshold_apr")
        return f"Пару закрывает, когда ставка падает ниже {floor}% годовых."
    patience = _number(core, "phase1_negative_patience")
    cap = _hours_to_days(core.get("phase1_breakeven_cap_hours"))
    stop = _pct(core, "neg_stop_threshold_apr")
    floor = _pct(core, "phase2_exit_threshold")
    mult = core.get("phase2_giveback_fee_mult")
    if None in (patience, cap, floor):
        return None
    before = (f"Пока комиссии за открытие не окупились — если ставка ушла в минус дольше чем на "
              f"{patience} часов подряд или комиссии окупаются дольше {cap}")
    stop_hours = _number(core, "neg_stop_patience_hours")
    if stop is not None and stop_hours is not None:
        before += f" (при ставке ниже {stop}% годовых хватает {stop_hours} часов)"
    after = f"Когда окупились — если ставка ушла ниже {floor}% годовых"
    if isinstance(mult, int | float) and mult > 0:
        after += (" и с пика заработанного отдано обратно не меньше, чем стоит весь круг комиссий"
                  if mult == 1 else
                  f" и с пика отдано обратно не меньше {num(mult)} кругов комиссий")
    return f"Пару закрывает, когда ставка перестаёт её оправдывать. {before}. {after}."


def _xsmom_universe(core: Mapping[str, Any]) -> str | None:
    if "lookbacks_days" not in core:
        return None
    universe = core.get("universe")
    if isinstance(universe, list) and universe:
        return f"{len(universe)} монет из списка"
    return "все монеты площадки с достаточной историей"


def _xsmom_rhythm(core: Mapping[str, Any]) -> str | None:
    days = core.get("rebalance_days")
    if not isinstance(days, int | float):
        return None
    out = _days_rhythm(days)
    dow = core.get("anchor_dow")
    if days == 7 and isinstance(dow, int) and 0 <= dow <= 6:
        out += f", по {_WEEKDAYS[dow]},"
    return out


def _xsmom_legs(core: Mapping[str, Any]) -> str | None:
    if "lookbacks_days" not in core:
        return None
    n = core.get("n_positions")
    if isinstance(n, int) and n >= 2:
        return f"по {n // 2} монет с каждой стороны"
    return "примерно по трети монет с каждой стороны"


def _number(core: Mapping[str, Any], *keys: str, scale: float = 1.0) -> str | None:
    """The first of `keys` the core states, as a number; a list of numbers (an open number the
    code chooses, `docs/OWN_NUMBERS.md`) as «a, b или c (выбирает код)»."""
    for key in keys:
        value = core.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return num(round(value * scale, 6))
        if isinstance(value, list) and value and all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in value):
            shown = [num(round(v * scale, 6)) for v in value]
            if len(shown) == 1:
                return shown[0]
            return f"{', '.join(shown[:-1])} или {shown[-1]} (выбирает код)"
    return None


FACTS: dict[str, Fact] = {
    "coins": _coins,
    "short_window": lambda c: _number(c, "short_window_days", "mom_short_days"),
    "long_window": lambda c: _number(c, "long_window_days", "mom_long_days"),
    "sticky_hours": lambda c: _number(c, "sticky_exit_hours"),
    "ratchet_pct": lambda c: _number(c, "ratchet_threshold", scale=100),
    "trend_universe": _trend_universe,
    "lookbacks": _lookbacks,
    "vol_target_pct": lambda c: _number(c, "vol_target_daily", scale=100),
    "book_vol_sentence": _book_vol,
    "frab_universe": _frab_universe,
    "signal_hours": lambda c: _number(c, "signal_window_hours"),
    "entry_pct": lambda c: _number(c, "entry_threshold_apr", scale=100),
    "slots": lambda c: _number(c, "concurrency_cap", "max_slots"),
    "exit_sentence": _exit_sentence,
    "xsmom_universe": _xsmom_universe,
    "xsmom_rhythm": _xsmom_rhythm,
    "xsmom_lookbacks": lambda c: _numbers(c.get("lookbacks_days")),
    "xsmom_legs": _xsmom_legs,
}
"""Placeholder -> how to read it off the core parameters of a spec. None: the spec does not say."""


def fill(sentence: str, core: Mapping[str, Any], defaults: Mapping[str, Any]) -> str | None:
    """A sentence with its placeholders filled; None when one of them cannot be."""
    values: dict[str, str] = {}
    for name in placeholders(sentence):
        found = FACTS[name](core) if name in FACTS else None
        if found is None and name in defaults:
            found = str(defaults[name])
        if found is None:
            return None
        values[name] = found
    return re.sub(r"\{([a-z_]+)\}", lambda m: values[m.group(1)], sentence)


def family_sentences(family: Family, core: Mapping[str, Any]) -> list[str]:
    return [done for s in family.sentences if (done := fill(s, core, family.facts_default))]


def fact_core(view: SpecView) -> dict[str, Any]:
    """The core parameters a family text reads its numbers from: the core's own, plus the
    parameters the spec only gives candidates for (code chooses among them; the text then says
    so instead of naming one value)."""
    core = dict(view.unwrapped.core)
    for key, values in candidate_values(view).items():
        if key not in core:
            core[key] = values if len(values) > 1 else values[0]
    return core


# ---- the money -----------------------------------------------------------------------------


def capital(view: SpecView) -> float | None:
    """The strategy's own money (docs/CAPITAL_BASE.md): the spec's `strategy_capital_usd`, else
    the sum the strategy itself is built on. Never the money on the exchange account."""
    explicit = view.extras.get("strategy_capital_usd")
    if isinstance(explicit, int | float) and not isinstance(explicit, bool):
        return float(explicit)
    core = view.unwrapped.core
    for key in ("capital_usd", "budget_cap_usdc", "book_capital_usd"):
        value = core.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def money_sentence(view: SpecView, family: Family | None) -> str:
    value = capital(view)
    if value is not None:
        return (f"Работает на деньгах стратегии: {money(value)} (от этой суммы считаются все "
                "проценты; свободные деньги на счёте биржи в расчёт не входят).")
    if family is not None and family.money_fallback:
        return family.money_fallback
    return ("Сумма денег этой стратегии в описании не задана: стенд считает всё в долях её "
            "капитала.")


# ---- the blocks of a composed strategy ---------------------------------------------------------


def _gross_text(gross: Any) -> str:
    if isinstance(gross, int | float) and abs(gross - 1) < 1e-9:
        return "Позиции занимают весь капитал, без заёмных денег."
    if isinstance(gross, int | float):
        return f"Позиции занимают {num(gross)} капитала."
    return "Размер позиций задан блоком."


def composed_sentences(unwrapped: Unwrapped, texts: Texts) -> list[str]:
    """What a composed strategy does, read off its blocks: one phrase per block, with the
    numbers of the spec. A block without a phrase reads «блок X с параметрами …»."""
    core = unwrapped.core
    blocks = texts.dictionary.blocks
    out: list[str] = []
    for kind, key in (("universe", "rule"), ("signal", "block"), ("portfolio", "block"),
                      ("sizing", "block")):
        spec = core.get(kind)
        if not isinstance(spec, dict):
            continue
        name = str(spec.get(key))
        template = (blocks.get(kind) or {}).get(name)
        params = {k: v for k, v in spec.items() if k != key}
        values: dict[str, Any] = dict(params)
        values["gross_text"] = _gross_text(params.get("gross"))
        values["gross_text_lower"] = values["gross_text"][:1].lower() + values["gross_text"][1:]
        if "tail_fraction" in params:
            values["tail_pct"] = percent(params["tail_fraction"]).rstrip("%")
        if "fraction" in params:
            values["fraction_pct"] = percent(params["fraction"]).rstrip("%")
        if "target_daily" in params:
            values["target_pct"] = percent(params["target_daily"]).rstrip("%")
        if "classes" in params and isinstance(params["classes"], list):
            values["classes"] = ", ".join(str(c) for c in params["classes"])
        try:
            out.append(str(template).format(**values))
        except (KeyError, IndexError, AttributeError):
            shown = ", ".join(f"{k} {v}" for k, v in params.items())
            out.append(f"{kind}: блок {name}" + (f" с параметрами {shown}." if shown else "."))
    if "rebalance_days" in core and blocks.get("rebalance"):
        out.append(blocks["rebalance"].format(
            rebalance_text=_days_rhythm(core["rebalance_days"])))
    if "min_history_days" in core and blocks.get("history"):
        out.append(blocks["history"].format(min_history_days=num(core["min_history_days"])))
    return out


def core_keys(view: SpecView) -> list[str]:
    """Every parameter path of the core, for the dictionary test."""
    return list(core_items(view.unwrapped))


__all__ = ["FACTS", "Words", "capital", "composed_sentences", "core_keys", "family_sentences",
           "fill", "money_sentence"]
