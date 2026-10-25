"""The night's two publication steps (docs/NIGHT.md, "Публикации как источник идей").

1. `fetch_stage` -- code only, no tokens, before the venue scout: read the new
   tail of the paper APIs, store it, mark candidates by the topic rule.
2. `scout_stage` -- after the venue scout, in the search stage's share of the
   budget (the share CLAUDE.md cuts first): hand at most
   `max_abstracts_per_night` candidate abstracts to the scout, through the
   scout's own call path and the budget guard. A budget stop leaves every
   paper waiting for a later night.

Neither step fails the night: a dead API or a refusal of the budget is written
to the night's state and the report.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import yaml

from qlab.pipeline.sources import QLAB_ROOT
from qlab.publications import store
from qlab.publications.fetch import fetch_arxiv, fetch_crossref, merge
from qlab.publications.filter import Rules, load_rules
from qlab.registry.models import Publication

CONFIG = Path("night/publications.yaml")
_REPLY_LINE = re.compile(r"^[\s*_`>-]*(DRAFT|NO CANDIDATE)\s+(\S+?):\s+(.*?)[\s*_`]*$",
                         re.IGNORECASE)


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _summary_row(row: Publication) -> dict:
    return {"id": row.id, "title": row.title, "link": row.link, "source": row.source,
            "published": row.published.date().isoformat(),
            "why": "; ".join(f"{g}: {', '.join(t)}" for g, t in (row.matched or {}).items())}


def fetch_all(session, cfg: dict, rules: Rules, *, client: httpx.Client, now: datetime,
              days: int | None = None, sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> dict:
    """Fetch the new tail of every enabled source, store it, apply the topic
    rule. `days` reads that far back instead of from the newest stored paper
    (a manual backfill). Returns the numbers the report shows."""
    errors: list[str] = []
    fetched, requests = [], 0
    sources = []
    if (cfg.get("arxiv") or {}).get("enabled"):
        sources.append(("arxiv", lambda since: fetch_arxiv(
            client, cfg["arxiv"], since, sleep=sleep, clock=clock)))
    if (cfg.get("crossref") or {}).get("enabled"):
        sources.append((cfg["crossref"]["source_name"], lambda since: fetch_crossref(
            client, cfg["crossref"], since, contact=cfg.get("contact"), sleep=sleep, clock=clock)))
    for source, fetch in sources:
        since = (now - timedelta(days=days) if days is not None
                 else store.since_for(session, source, now=now,
                                      lookback_days=int(cfg["lookback_days"]),
                                      overlap_days=int(cfg["overlap_days"])))
        report = fetch(since)
        fetched += report.items
        errors += report.errors
        requests += report.requests
    fetched = merge(fetched)
    added = store.add_new(session, fetched, now=now)
    new_candidates = store.apply_filter(session, rules)
    return {
        "fetched": len(fetched), "new": len(added), "requests": requests, "errors": errors,
        "filter_version": rules.version,
        "new_candidates": [_summary_row(r) for r in sorted(
            new_candidates, key=lambda r: (-r.score, -r.published.timestamp()))],
    }


def fetch_stage(day, state: dict, *, config: Path = CONFIG, client: httpx.Client | None = None,
                now: datetime | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
    """Step 1: fetch and filter, once per night (a resumed night does not refetch)."""
    from qlab.registry.db import session_scope

    pubs = state.setdefault("publications", {})
    if "fetch" in pubs or not config.is_file():
        return
    now = now or datetime.now(UTC)
    own_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        cfg = load_config(config)
        rules = load_rules(Path(cfg["filter"]))
        with session_scope() as session:
            pubs["fetch"] = fetch_all(session, cfg, rules, client=client, now=now, sleep=sleep)
    except Exception as exc:  # noqa: BLE001 - the report says so, the night goes on
        pubs["fetch"] = {"fetched": 0, "new": 0, "requests": 0, "new_candidates": [],
                         "errors": [f"{type(exc).__name__}: {exc}"]}
    finally:
        if own_client:
            client.close()
    from qlab.night import save_state

    save_state(day, state)


def _paper(row: Publication, max_chars: int) -> dict:
    return {"id": row.id, "title": row.title, "authors": row.authors, "link": row.link,
            "published": row.published.date().isoformat(), "source": row.source,
            "matched": row.matched, "abstract": row.abstract[:max_chars]}


def parse_reply(reply: str | None) -> dict[str, tuple[str, str]]:
    """The scout's one line per paper: paper id -> (DRAFT | NO CANDIDATE, rest)."""
    found = {}
    for line in (reply or "").splitlines():
        m = _REPLY_LINE.match(line)
        if m:
            found[m.group(2)] = (m.group(1).upper(), m.group(3))
    return found


def match_drafts(rows: list[Publication], drafts: list[Path], reply: dict[str, tuple[str, str]],
                 root: Path | None = None) -> dict[str, str]:
    """Which draft came from which paper: the scout's own reply when it names a
    draft that exists, else the draft that cites the paper's link."""
    root = root or QLAB_ROOT
    written = {str(d) for d in drafts}
    texts = {}
    for d in drafts:
        try:
            texts[str(d)] = (root / d).read_text(encoding="utf-8")
        except OSError:
            texts[str(d)] = ""
    matched: dict[str, str] = {}
    for row in rows:
        said = reply.get(row.id)
        if said and said[0] == "DRAFT" and said[1].strip() in written:
            matched[row.id] = said[1].strip()
            continue
        if said and said[0] == "NO CANDIDATE":
            continue
        for path, text in texts.items():
            if row.link in text and path not in matched.values():
                matched[row.id] = path
                break
    return matched


def _result(paper_id: str, drafts: dict[str, str], reply: dict[str, tuple[str, str]]) -> str:
    if paper_id in drafts:
        return f"черновик {drafts[paper_id]}"
    kind, rest = reply.get(paper_id, ("", ""))
    return f"кандидата нет: {rest}" if kind == "NO CANDIDATE" else "ответа разведчика нет"


def _open_budget(session):
    """One candidate of the search stage's share, from the latest reading of
    the shared windows (a probe call only when the reading is stale)."""
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate

    return open_candidate(session, None, Stage.SEARCH)


def scout_stage(day, state: dict, *, config: Path = CONFIG) -> None:
    """Step 2: a small batch of candidate abstracts to the scout, inside the
    budget guard. Once per night when it ran; a budget stop is tried again."""
    from qlab.agents.scout import scout_publications
    from qlab.budget import BudgetExhausted
    from qlab.night import save_state
    from qlab.registry.db import session_scope

    pubs = state.setdefault("publications", {})
    if not config.is_file() or "fetch" not in pubs:
        return
    done = pubs.get("handoff")
    if done and not done.get("stopped") and not done.get("error"):
        return
    cfg = load_config(config)
    if not cfg.get("scout_handoff", True):
        return
    if (state.get("scout") or {}).get("stopped"):
        pubs["handoff"] = {"stopped": "the night's budget ended in the search stage before this"}
        save_state(day, state)
        return
    with session_scope() as session:
        rows = store.pick_for_scout(
            session, cap=int(cfg["max_abstracts_per_night"]),
            min_abstract_chars=int(cfg["min_abstract_chars"]))
        if not rows:
            pubs["handoff"] = {"ids": [], "note": "no candidate with a readable abstract waiting"}
            save_state(day, state)
            return
        papers = [_paper(r, int(cfg["max_abstract_chars"])) for r in rows]
        try:
            budget = _open_budget(session)
            outcome = scout_publications(day, papers, budget=budget, session=session)
        except BudgetExhausted as stop:
            pubs["handoff"] = {"ids": [r.id for r in rows], "stopped": str(stop)}
            session.commit()
            save_state(day, state)
            return
        except Exception as exc:  # noqa: BLE001 - the report says so, the night goes on
            pubs["handoff"] = {"ids": [r.id for r in rows],
                               "error": f"{type(exc).__name__}: {exc}"}
            session.commit()
            save_state(day, state)
            return
        reply = parse_reply(outcome.reply)
        drafts = match_drafts(rows, outcome.drafts, reply)
        notes = {pid: f"{kind}: {rest}" for pid, (kind, rest) in reply.items()}
        store.mark_handed(session, rows, at=datetime.now(UTC), notes=notes, drafts=drafts)
        pubs["handoff"] = {
            "ids": [r.id for r in rows], "reply": outcome.reply, "problems": outcome.problems,
            "drafts": drafts, "tokens": budget.spent_tokens,
            "papers": [{"id": r.id, "title": r.title, "link": r.link,
                        "result": _result(r.id, drafts, reply)} for r in rows],
            "waiting_after": session.query(Publication).filter(
                Publication.candidate.is_(True), Publication.handed_at.is_(None)).count(),
        }
    save_state(day, state)


def report_lines(state: dict) -> list[str]:
    """The morning report's section (Russian, plain words)."""
    pubs = state.get("publications")
    if not pubs:
        return []
    lines = ["## Новые публикации", ""]
    fetch = pubs.get("fetch")
    if fetch:
        lines.append(
            f"Прочитано статей: {fetch['fetched']}, из них новых — {fetch['new']} "
            f"(запросов к сервисам: {fetch['requests']}). Отбор по теме, версия правил "
            f"{fetch.get('filter_version', '—')}: новых подходящих — "
            f"{len(fetch['new_candidates'])}. "
            "Отбор смотрит только на тему статьи, не на обещанную в ней доходность.")
        lines += [f"- сбой чтения: {e}" for e in fetch.get("errors") or []]
        for c in fetch["new_candidates"]:
            lines.append(f"- [{c['title']}]({c['link']}) ({c['source']}, {c['published']}) — "
                         f"совпало: {c['why']}")
    hand = pubs.get("handoff")
    if hand:
        lines.append("")
        if hand.get("stopped"):
            lines.append(f"Разведчику статьи не передавались: {hand['stopped']}. "
                         "Они ждут следующей ночи.")
        elif hand.get("error"):
            lines.append(f"Разведчик по статьям упал: {hand['error']}. Статьи ждут следующей ночи.")
        elif not hand.get("ids"):
            lines.append(f"Разведчику статьи не передавались: {hand.get('note', '—')}.")
        else:
            drafts = hand.get("drafts") or {}
            lines.append(
                f"Разведчику передано статей: {len(hand['ids'])}, черновиков карточек получилось: "
                f"{len(drafts)}"
                + (f" ({hand['tokens']:,} токенов)" if hand.get("tokens") else "")
                + f". Подходящих статей ждёт очереди: {hand.get('waiting_after', 0)}.")
            lines += [f"- [{p['title']}]({p['link']}) — {p['result']}"
                      for p in hand.get("papers") or []]
            lines += [f"- проверка не пройдена: {p}" for p in hand.get("problems") or []]
    lines.append("")
    return lines


__all__ = ["CONFIG", "fetch_all", "fetch_stage", "match_drafts", "parse_reply", "report_lines",
           "scout_stage"]
