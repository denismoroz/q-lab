"""«Как выбраны числа»: where the numbers of an idea come from, read off its spec
(docs/PLAIN_PAGE.md, docs/ONE_CHECK.md).

Three cases, never mixed up on the page:

- **by code**: the idea declares the values it leaves open and code chooses among them every
  month from the past only (a `Retune` layer with a grid, or the spec's `choice.axes` /
  `choice.candidates`); the page names the candidates in plain words;
- **fixed**: the numbers were fixed by hand on a date (`params_fixed_at`), with the evidence;
- **nothing to choose**: `choice.nothing_to_choose`, with its reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from qlab.plain.specs import SpecView, core_items
from qlab.plain.words import MISSING, Words

_EVIDENCE_MAX = 260
_BASE_EXPLAINED = "числа основной версии без изменений"
_CANDIDATES_SHOWN = 8


@dataclass(frozen=True)
class Choice:
    kind: str
    """`by_code`, `fixed`, `nothing_to_choose` or `unknown`."""
    text: str
    lines: tuple[str, ...] = ()


def _first_sentence(text: str, limit: int = _EVIDENCE_MAX) -> str:
    flat = " ".join(str(text).split())
    cut = flat.find(". ")
    one = flat if cut < 0 else flat[: cut + 1]
    return one if len(one) <= limit else one[: limit - 1].rstrip() + "…"


def _ordered(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _overlay_lines(candidates: dict[str, dict[str, Any]], words: Words) -> list[str]:
    """One line per overlay named in the candidates: its plain title, how many variants carry it
    and which values its parameters take."""
    layers = words.dictionary.layers.get("overlays", {})
    seen: dict[str, dict[str, list[Any]]] = {}
    count: dict[str, int] = {}
    for overrides in candidates.values():
        for item in (overrides or {}).get("overlays") or ():
            if not isinstance(item, dict):
                continue
            name = str(item.get("name"))
            count[name] = count.get(name, 0) + 1
            for key, value in (item.get("params") or {}).items():
                seen.setdefault(name, {}).setdefault(key, []).append(value)
    lines = []
    for name in count:
        params = seen.get(name, {})
        title = layers.get(name, {}).get("title", f"надстройка {name}")
        parts = []
        for key, values in params.items():
            shown = _ordered(values)
            if len(shown) < 2:
                continue
            path = f"overlay.{name}.{key}"
            if all(isinstance(v, bool) for v in shown):
                parts.append("; ".join(words.name(path, v) for v in shown))
            else:
                what = (words.entry(path) or {}).get("what", f"параметр {key}").split(" (")[0]
                parts.append(f"{what}: {words.choices(path, shown)}")
        lines.append(f"надстройка «{title}», вариантов {count[name]}"
                     + (f" ({'; '.join(parts)})" if parts else ""))
    return lines


def _axis_lines(axis: str, candidates: dict[str, dict[str, Any]], base_name: str,
                words: Words, core: dict[str, Any]) -> list[str]:
    """The lines for one axis of a grid: the values of one number, or one line per variant."""
    if candidates and all("overlays" in (c or {}) for c in candidates.values()):
        return _overlay_lines(candidates, words)
    keys = _ordered([k for c in candidates.values() for k in (c or {})])
    nonempty = {n: c for n, c in candidates.items() if c}
    # many values of one number: «<what>: v1, v2, ...»
    if len(keys) == 1 and len(nonempty) == len(candidates) \
            and not all(isinstance(c[keys[0]], bool) for c in candidates.values()):
        key = keys[0]
        values = _ordered([c[key] for c in candidates.values()])
        what = (words.entry(key) or {}).get("what", f"параметр {key}")
        return [f"{what}: {words.choices(key, values)}"]
    if len(candidates) > _CANDIDATES_SHOWN:
        out = []
        for key in keys:
            values = _ordered([c[key] for c in candidates.values() if key in (c or {})])
            what = (words.entry(key) or {}).get("what", f"параметр {key}")
            out.append(f"{what}: {words.choices(key, values)}")
        return out
    out = []
    for overrides in candidates.values():
        if overrides:
            labels = [words.label(k, v) for k, v in overrides.items()]
            explained = "; ".join(words.describe(k, v) for k, v in overrides.items())
            # a variant is named only when the dictionary has a short name for every change in it
            out.append(f"вариант «{'; '.join(labels)}»: {explained}" if all(labels)
                       else f"вариант: {explained}")
            continue
        contrast = []
        for key in keys:
            others = [c[key] for c in nonempty.values() if key in c]
            if others and all(isinstance(v, bool) for v in others):
                contrast.append(words.describe(key, not others[0]))
            elif key in core:
                contrast.append(words.describe(key, core[key]))
            elif words.default(key) is not MISSING:
                contrast.append(words.describe(key, words.default(key)))
        out.append(f"вариант «{base_name}»: {'; '.join(contrast) if contrast else _BASE_EXPLAINED}")
    return out


def _in_russian(sentence: str, lead: str) -> str:
    """`sentence` after `lead`, marked as English when it has no Russian letter in it: a quote
    stays as written, and the page says in what language it is written."""
    marked = "" if re.search(r"[а-яё]", sentence, re.IGNORECASE) else "по-английски, "
    return f"{lead} ({marked}как записано в описании): «{sentence}»"


def _fixed_evidence(view: SpecView) -> list[str]:
    evidence = str(view.extras.get("params_fixed_evidence") or "")
    if evidence.lower().startswith("named by code"):
        return ["день назвал код: начало последней трети периода данных этой проверки "
                "(docs/FIXATION.md)"]
    if not evidence.strip():
        return []
    return [_in_russian(_first_sentence(evidence), "Основание")]


def _inherited(view: SpecView) -> bool:
    """A second-stage spec carries the numbers the first stage chose (`qlab.pipeline.stage2`)."""
    return str(view.extras.get("params_fixed_evidence") or "").lower().startswith(
        "inherited from stage 1")


def describe(view: SpecView, base_name: str, words: Words) -> Choice:
    layers = words.dictionary.layers.get("retune", {})
    unwrapped = view.unwrapped
    lines: list[str] = []
    retunes = unwrapped.layer("Retune")
    for retune in retunes:
        for axis, candidates in (retune.own.get("grid") or {}).items():
            lines += _axis_lines(axis, {k: dict(v or {}) for k, v in (candidates or {}).items()},
                                 base_name, words, unwrapped.core)
    choice = view.extras.get("choice") if isinstance(view.extras.get("choice"), dict) else {}
    for path, values in (choice.get("axes") or {}).items():
        what = (words.entry(path) or {}).get("what", f"параметр {path}")
        lines.append(f"{what}: {words.choices(path, values)}")
    if choice.get("candidates"):
        lines += _axis_lines("candidates",
                             {k: dict(v or {}) for k, v in choice["candidates"].items()},
                             base_name, words, unwrapped.core)
    open_numbers = {k: v for k, v in core_items(unwrapped).items() if words.open_list(k, v)}
    if open_numbers and not lines:
        # the older kind of idea: the numbers are left open in the params themselves
        # (docs/OWN_NUMBERS.md): chosen once, on the data before the cut, judged after it
        return Choice(
            "by_code",
            "Числа не выбирал человек: перед проверкой код один раз, на дату отреза (начало "
            "последней трети периода данных), выбирает из заранее объявленных значений то, у "
            "которого лучший Шарп после комиссий на днях до отреза (Шарп — средняя доходность, "
            "делённая на размах колебаний), и держит его до конца; оценивают только дни после "
            "отреза.",
            tuple(f"{(words.entry(k) or {}).get('what', f'параметр {k}')}: "
                  f"{words.choices(k, v)}" for k, v in open_numbers.items()))
    if lines:
        first = retunes[0].own if retunes else {}
        refit = layers.get("refit", {}).get(str(first.get("refit_every", "M")), "каждый месяц")
        objective = layers.get("objective", {}).get(
            str(first.get("objective", "sharpe_net")), "по Шарпу после комиссий")
        window = layers.get("window", {}).get(str(first.get("window", "expanding")),
                                              "по всей истории до этого дня")
        return Choice(
            "by_code",
            f"Числа не выбирал человек: {refit} код сам выбирает между заранее объявленными "
            f"вариантами — {objective}, {window}, и ничего не видит вперёд.", tuple(lines))
    if choice.get("nothing_to_choose"):
        extra = ([f"день, с которого идёт проверка вперёд: {view.extras['params_fixed_at']}"]
                 if view.extras.get("params_fixed_at") else [])
        return Choice(
            "nothing_to_choose",
            "Выбирать не из чего: для этой идеи нет списка значений, между которыми код мог бы "
            "выбирать. " + _in_russian(_first_sentence(choice["nothing_to_choose"]), "Причина"),
            tuple(extra + _fixed_evidence(view)))
    fixed_at = view.extras.get("params_fixed_at")
    if fixed_at and _inherited(view):
        return Choice(
            "by_code",
            "Эти числа не выбирал человек и не подгонял под данные этой площадки: они взяты "
            f"из первой проверки (на Binance), где их выбрал код по данным до {fixed_at}; "
            "оценивают только дни после этой даты.", ())
    if fixed_at:
        return Choice(
            "fixed", f"Числа выбраны вручную и зафиксированы {fixed_at}: до этого дня они могли "
            "подгоняться под данные, проверка вперёд идёт после него.",
            tuple(_fixed_evidence(view)))
    return Choice("unknown", "Откуда взяты числа, в описании этой идеи не записано.")


__all__ = ["Choice", "MISSING", "describe"]
