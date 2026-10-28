"""Words for the anomaly work: the console tables, a flag as a plain Russian sentence, the
idea page's block (docs/ANOMALIES.md) and the morning report's section.

Nothing here computes a flag or decides anything; it words what the scans and dossiers stored.
Plain language for the owner (cards/SCHEMA.md, «Правила заполнения»): a term is explained where
it first appears, and a «монето-день» is explained as one coin on one day.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from qlab.anomalies import store
from qlab.anomalies.config import FLAGS
from qlab.anomalies.flags import TITLES_RU, UNDEFINED, FlagResult

SECTION_TITLE = "Аномалии: слишком красивые результаты"

INTRO = (
    "Проверка кодом, без токенов: для каждого прогона идеи, прошедшей правила, стенд повторяется "
    "заново и результат сверяется с несколькими заранее объявленными границами (`night/"
    "anomalies.yaml`) — не слишком ли большая прибыль за один день для такого числа позиций, не "
    "держится ли прибыль на одной-трёх монетах или на нескольких «монето-днях» (монето-день — одна "
    "монета в один день), не взята ли она там, где торговать было нельзя, сходятся ли две биржи на "
    "общем отрезке. Признак ничего не решает: вердикт остаётся за правилами; признаки только "
    "записываются, пока не накопится статистика (`docs/ANOMALIES.md`).")


def _pct(value: Any, digits: int = 1) -> str:
    return "—" if not isinstance(value, int | float) else f"{value * 100:+.{digits}f}%".replace(
        "-", "−")


def _share(value: Any) -> str:
    if not isinstance(value, int | float):
        return "—"
    return f"{value:.1%}" if abs(value) < 0.1 else f"{value:.0%}"


def _money(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.1f} млрд $"
    return f"{value / 1e6:g} млн $" if value >= 1e6 else f"{value / 1e3:g} тыс. $"


def flag_sentence(f: FlagResult) -> str:
    """One flag in plain Russian: what it found (when it fires or is quiet) or why it is not
    defined."""
    d = f.detail
    if f.state == UNDEFINED:
        return f"{TITLES_RU[f.name]}: не определено — {f.reason}"
    head = f"{TITLES_RU[f.name]}: {f.state_ru}"
    if f.name == "implausible_day":
        top = (d.get("top") or [{}])[0]
        return (f"{head}. Самый большой день {top.get('day')}: {_pct(top.get('return'))} при "
                f"{top.get('positions', 0):.0f} позициях (общий объём позиций "
                f"{top.get('gross', 0):.2f} капитала), граница для такого дня "
                f"{_pct(top.get('day_line'))}, то есть в {f.value:.1f} раза выше; таких дней "
                f"{d.get('days_over')} из {d.get('days')}")
    if f.name == "concentrated_instruments":
        best = (d.get("top") or [{}])[0]
        return (f"{head}. Лучшая монета ({best.get('instrument')}) дала "
                f"{_share(d.get('top1_share'))} "
                f"всей прибыли, три лучшие — {_share(d.get('top3_share'))} "
                f"(монет в книге {d.get('instruments')})")
    if f.name == "concentrated_coin_days":
        return (f"{head}. Лучшие {d.get('top_k')} монето-дней из {d.get('held_coin_days')} дали "
                f"{_share(f.value)} всей прибыли (граница {_share(f.line)})")
    if f.name == "tail_move_coin_days":
        top = (d.get("top") or [{}])[0]
        best = (f"; самый крупный — {top.get('instrument')} {top.get('day')}: монета сдвинулась "
                f"на {_pct(top.get('move'), 0)}, оборот за сутки "
                f"{_money(top.get('volume_usd') or 0)}, "
                f"возраст контракта {top.get('age_days')} дн." if top else "")
        return (f"{head}. {d.get('tail_coin_days')} монето-дней, где цена монеты за сутки "
                f"сдвинулась на {d.get('min_move'):.0%} и больше "
                f"({_share(d.get('tail_day_share_of_held'))} всех удерживаемых), дали "
                f"{_share(f.value)} всей прибыли (граница {_share(f.line)}){best}")
    if f.name == "unlisted_on_execution_venue":
        return (f"{head}. С {d.get('from')} (с этого дня известен список биржи исполнения, "
                f"{d.get('venue')}) {_share(f.value)} прибыли получено на монетах, которых там не "
                f"было ({d.get('unlisted_coin_days')} монето-дней из {d.get('held_coin_days')}); "
                f"граница {_share(f.line)}")
    if f.name == "frozen_price":
        longest = (d.get("longest") or [{}])[0]
        tail = (f"; дольше всех — {longest.get('instrument')}, {longest.get('days')} дн. подряд "
                f"с {longest.get('from')}" if longest else "")
        return (f"{head}. {d.get('frozen_coin_days')} удерживаемых монето-дней из "
                f"{d.get('held_coin_days')} ({_share(f.value)}) приходятся на стоянки цены "
                f"{d.get('min_days'):g} дней подряд и дольше{tail}")
    if f.name == "untradeable_coin_days":
        return (f"{head}. {_share(f.value)} прибыли получено в монето-днях с оборотом меньше "
                f"{_money(d.get('min_volume_usd', 0))} за день "
                f"({_share(d.get('low_volume_share'))}) "
                f"или в последние {d.get('delist_days'):g} дней перед снятием монеты с торгов "
                f"({_share(d.get('near_delist_share'))}); граница {_share(f.line)}")
    if f.name == "venue_divergence":
        corr = ("—" if d.get("daily_correlation") is None
                else format(d["daily_correlation"], ".2f"))
        return (f"{head}. На общем отрезке {d.get('days')} дн. ({d.get('start')} — {d.get('end')}) "
                f"первая ступень даёт {_pct(d.get('stage1_ann'), 0)} в год, вторая "
                f"{_pct(d.get('stage2_ann'), 0)}; расхождение {_pct(f.value, 0)} в год при границе "
                f"{_pct(f.line, 0)}; дневные доходности согласуются на "
                f"{corr}")
    if f.name == "stand_vs_paper":
        return (f"{head}. На одном отрезке стенд {_pct(d.get('stand_return'))}, бумажная книга "
                f"{_pct(d.get('paper_return'))}; расхождение {_pct(f.value)} при границе "
                f"{_pct(f.line)}")
    if f.name == "too_good_for_family":
        return (f"{head}. Доходность {_pct(f.value, 0)} в год, граница семейства "
                f"{_pct(f.line, 0)} (середина семейства {_pct(d.get('median'), 0)}, других идей "
                f"{d.get('members')})")
    return head


def scan_lines(scan: Any) -> list[str]:
    lines = [f"прогон {scan.trial_id} ({scan.idea_id}), маршрут {scan.route}; границы версии "
             f"{scan.config_version}; сработало: {', '.join(scan.fired) or 'ничего'}"]
    lines += [f"  - {flag_sentence(FlagResult.from_json(f))}" for f in scan.flags]
    return lines


def render_table(session: Session, rows: list[Any]) -> list[str]:
    """One line per scanned idea: idea, trial, route, the flags that fired, the dossier's
    cause if there is one."""
    if not rows:
        return ["no scanned run in the registry (qlab anomalies scan --passed)"]
    width = max([len(r.idea_id) for r in rows] + [4])
    head = f"{'idea':<{width}} {'trial':>7} {'route':<21} {'fired':>5}  flags / investigation"
    out = [head]
    for r in rows:
        inv = store.latest_investigation(session, trial_id=r.trial_id)
        tail = ", ".join(r.fired) or "-"
        if inv is not None:
            tail += f"  => {inv.cause}"
        out.append(f"{r.idea_id:<{width}} {r.trial_id:>7} {r.route or '-':<21} "
                   f"{len(r.fired):>5}  {tail}")
    return out


def render_summary(summary: Mapping[str, Any]) -> list[str]:
    lines = [f"totals over {summary['trials']} scanned run(s) of {summary['ideas']} idea(s); "
             f"{summary['flagged']} with at least one flag fired:",
             f"  {'flag':<30}{'fires':>7}{'quiet':>7}{'not defined':>13}"]
    for name in FLAGS:
        t = summary["totals"].get(name, {})
        lines.append(f"  {name:<30}{t.get('fires', 0):>7}{t.get('quiet', 0):>7}"
                     f"{t.get('not defined', 0):>13}")
    return lines


def learning_lines(session: Session) -> list[str]:
    data = store.learning(session)
    lines = [f"investigations: {data['investigations']}; causes: {data['causes'] or 'none yet'}",
             f"confirmed (not 'unclear' or 'genuine'): {data['confirmed']}"]
    if data["confirmed"]:
        caught = ", ".join(f"{k} x{v}" for k, v in sorted(data["caught_by"].items())) or "none"
        lines.append(f"  caught by a detector flag: {caught}")
        lines.append(f"  caught by no flag (a check was proposed): {data['caught_by_none']}")
    return lines


# --------------------------------------------------------------------------
# The idea page's block


def _scan_json(scan: Any) -> dict[str, Any]:
    flags = []
    for raw in scan.flags:
        f = FlagResult.from_json(raw)
        flags.append({**f.to_json(), "title": TITLES_RU[f.name], "state_ru": f.state_ru,
                      "sentence": flag_sentence(f)})
    return {"trial_id": scan.trial_id, "route": scan.route, "config_version": scan.config_version,
            "scanned_at": scan.scanned_at.isoformat(), "fired": scan.fired, "flags": flags,
            "facts": scan.facts}


def idea_block(session: Session, idea_id: str) -> dict[str, Any] | None:
    """What the idea page shows under «Аномалии»: the latest scan of the idea's latest scanned
    run and, if the agent looked into it, the dossier. None when the idea was never scanned."""
    from qlab.registry.models import AnomalyScan

    scan = (session.query(AnomalyScan).filter(AnomalyScan.idea_id == idea_id)
            .order_by(AnomalyScan.trial_id.desc(), AnomalyScan.scanned_at.desc()).first())
    if scan is None:
        return None
    inv = store.latest_investigation(session, trial_id=scan.trial_id)
    return {"scan": _scan_json(scan), "investigation": None if inv is None else {
        "id": inv.id, "at": inv.at.isoformat(), "model": inv.model, "cause": inv.cause,
        "cause_ru": store.CAUSES_RU.get(inv.cause, inv.cause), "summary": inv.summary,
        "dossier": inv.dossier, "evidence": inv.evidence, "proposals": inv.proposals,
        "caught_by": inv.caught_by, "new_check": inv.new_check,
        "new_check_status": inv.new_check_status,
        "programmer_requests": inv.programmer_requests, "tokens": inv.tokens}}


# --------------------------------------------------------------------------
# The morning report


def entry_lines(entry: Mapping[str, Any]) -> list[str]:
    """One scanned run in the report."""
    head = f"- **{entry['idea']}** (прогон {entry['trial']}, {entry.get('route')})"
    if entry.get("error"):
        return [f"{head}: **сбой** — {str(entry['error'])[:200]}; повторится в следующую ночь"]
    fired = entry.get("fired") or []
    if not fired:
        return [f"{head}: ни один признак не сработал"]
    out = [f"{head}: сработало {len(fired)} из {len(FLAGS)}"]
    out += [f"    - {s}" for s in entry.get("sentences") or []]
    return out


def investigation_lines(entry: Mapping[str, Any]) -> list[str]:
    head = f"- **{entry['idea']}** (прогон {entry['trial']})"
    if entry.get("stopped"):
        return [f"{head}: разбор остановлен охраной бюджета — {entry['stopped']}; продолжится "
                "в следующую ночь"]
    if entry.get("error"):
        return [f"{head}: разбор не удался — {str(entry['error'])[:200]}"]
    out = [f"{head}: причина — **{store.CAUSES_RU.get(entry['cause'], entry['cause'])}** "
           f"(`{entry['cause']}`). {entry.get('summary', '')}"]
    caught = entry.get("caught_by") or []
    out.append("    - " + (f"поймали признаки: {', '.join(caught)}" if caught else
                           "ни один признак причину не поймал"
                           + (f"; предложена проверка: {entry['new_check']}"
                              if entry.get("new_check") else "")))
    if entry.get("programmer_requests"):
        out.append(f"    - заявки программисту: {', '.join(entry['programmer_requests'])}")
    out.append(f"    - досье: `uv run qlab anomalies show {entry['trial']}`; на странице идеи — "
               "блок «Аномалии». Токенов: "
               f"{entry.get('tokens', 0):,}".replace(",", " "))
    return out


def report_lines(block: Mapping[str, Any] | None, session: Session | None = None) -> list[str]:
    """The stage's section for the morning report. Empty when the stage had nothing to say."""
    if not block:
        return []
    scans = block.get("scans") or []
    investigations = block.get("investigations") or []
    notes = block.get("notes") or []
    pending = store.pending_checks(session) if session is not None else []
    if not (scans or investigations or notes or pending or block.get("deferred")):
        return []
    lines = [INTRO, ""]
    lines += [f"- {n}" for n in notes]
    flagged = [s for s in scans if s.get("fired")]
    lines.append(f"Прогонов проверено: {len(scans)}, с сработавшим признаком: {len(flagged)}; "
                 f"разборов агентом: {len(investigations)}.")
    if block.get("deferred"):
        lines.append(f"Не уместились в ночной предел и ждут следующей ночи: "
                     f"{', '.join(str(t) for t in block['deferred'])}.")
    lines.append("")
    for entry in scans:
        lines += entry_lines(entry)
    if investigations:
        lines += ["", "Разбор агентом-следователем (он ничего не меняет в вердиктах, статусах и "
                  "коде; его причина проверена кодом по журналу его вызовов):"]
        for entry in investigations:
            lines += investigation_lines(entry)
    if pending:
        lines += ["", "Предложенные проверки, которые ждут решения владельца — стоит ли сделать из "
                  "них признак (`uv run qlab anomalies check <номер> adopted|declined`):"]
        lines += [f"- разбор {inv.id}, прогон {inv.trial_id} ({inv.idea_id}): {inv.new_check}"
                  for inv in pending]
    return lines


__all__ = ["INTRO", "SECTION_TITLE", "entry_lines", "flag_sentence", "idea_block",
           "investigation_lines", "learning_lines", "render_summary", "render_table",
           "report_lines", "scan_lines"]
