"""The investigator agent: a flagged run -> a dossier, never a verdict (docs/ANOMALIES.md).

Owner, 2026-10-28: «давай введем агента который будет искать такие аномалии и разбирать» and,
on where it lives, «я имел ввиду агента во фреймворк».

The detector (`qlab.anomalies.scan`) writes flags in code. For a flagged run, with passed
ideas on their way to the owner first, this agent takes the run apart with a FIXED read-only
toolkit (`qlab.anomalies.toolkit`, reachable only as `uv run qlab anomaly ...`) and answers
with a cause and evidence. What it may and may not do:

* it **may** read the repository (Read, Glob, Grep) and run the toolkit's commands;
* it **may not** write a file or run any other command: its tools are `Read`, `Glob`, `Grep` and
  `Bash` restricted to the prefix `uv run qlab anomaly`, and `Write`/`Edit` are removed; the
  dossier is not a file it writes but the JSON of its answer, which code checks and stores;
* it **may not** change a verdict, a status, a route or any code. A code fix it proposes goes to
  the programmer's queue as a REQUEST (`qlab.agents.programmer`), built on a branch the owner
  merges by hand (`qlab code merge`), never applied by this agent.

**The model proposes, the code decides** (CLAUDE.md). The cause is kept only with evidence the
code checks: every evidence item names a call of the toolkit (`call N`) and a quote copied
verbatim from that call's output. The calls are logged by the toolkit itself
(`QLAB_ANOMALY_LOG`), not by the agent, so evidence cannot be invented: a quote that is not in
the output of the call it cites is rejected, and a cause with no accepted evidence becomes
`unclear`. The learning loop (`caught_by`, `new_check`) is checked the same way: only flags that
really fired can be named as having caught the cause.

The agent's spending goes through the budget guard (`qlab.budget.run_agent`), stage
«перепроверка» (it protects against false passes), its own per-investigation ceiling, at most
`max_per_night` of them.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from qlab.anomalies import flags as fl
from qlab.anomalies import report, store, toolkit
from qlab.anomalies.config import FLAGS, Config
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.sources import QLAB_ROOT

AGENT = "anomaly-investigator"
TOOLS = "Read,Glob,Grep,Bash"
"""The built-in tools the agent has at all: no Write, no Edit, no web, no sub-agents."""
ALLOWED = "Read,Glob,Grep,Bash(uv run qlab anomaly:*)"
"""`--allowedTools`: Bash only for the toolkit's prefix; any other command is refused in a
non-interactive run."""
DISALLOWED = "Write,Edit,NotebookEdit"
PROPOSAL_KINDS = ("data_filter", "stand_fix", "new_check")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
MAX_QUOTE = 400

INSTRUCTIONS = """You are q-lab's anomaly investigator. q-lab is a research lab that screens
trading strategies; verdicts are made by CODE and you never change one. A run that passed the
rules has results that look too good, and the code detector flagged it. Your job is to find out
WHY, with evidence, and say so in plain Russian for the owner, who is not a trader.

The run: idea {idea_id} ({title}), trial {trial_id}, route {route} (status {status}).
{headline}
{years}

FLAGS that fired (the detector's own numbers, the lines are in night/anomalies.yaml):
{fired}

Flags that did not fire or are not defined:
{quiet}

How to work. You have no shell except the read-only toolkit; run ONLY these (each re-runs the
stored spec on the stored data and prints numbers; none writes anything). Run each as a single
command with no pipe, redirect, `cd` or `&&`:
  uv run qlab anomaly contributions {trial_id} --days 2025-11-02,2026-01-30
      (or --from D --to D, --instrument COIN, --top N) per day and coin: weight, side, price
      move, funding, cost, net contribution, dollar volume
  uv run qlab anomaly coin COIN --trial {trial_id} [--from D --to D]
      a coin's first and last price, volume, whether it stopped trading, whether Hyperliquid
      lists it
  uv run qlab anomaly counterfactual {trial_id} --exclude-volume-below 1000000
  uv run qlab anomaly counterfactual {trial_id} --exclude-days-before-delist 7
  uv run qlab anomaly counterfactual {trial_id} --cap-instrument-share 0.1
  uv run qlab anomaly counterfactual {trial_id} --cap-leg-contribution 0.02
  uv run qlab anomaly counterfactual {trial_id} --only-hl-listed
      the same run with ONE rule, beside the original: yearly return, worst fall, year table.
      By default what a rule bans becomes cash; add --reselect to a ban rule to run the strategy
      again with the banned coin-days untradeable, so it picks other coins ("what if the book
      could only trade these"; slower). --cap-leg-contribution is a stress, not a rule.
  uv run qlab anomaly battery {trial_id} [--reselect]
      the standard cases in one table: volume floors of 1 and 5 million dollars, no position
      near a delisting, one coin's earning clipped to 2% and 1% of capital, only coins the other
      venue lists. A result that moves by a multiple under a small rule is a lottery
  uv run qlab anomaly compare-venues {idea_id}
      the idea's two stages over their common days, month by month, and the coins that earned
  uv run qlab anomaly flags {trial_id}
Every output starts with `[call N]`. You may also Read files of the repository (the strategy's
code under src/qlab/strategies, the harness under src/qlab/harness, docs/) to understand a
mechanism. docs/research/xsmom-anomaly-2026-10-28.md is the example of a good investigation of
this kind (facts and numbers, not a verdict): read its sections 2, 3 and 5 first. Start from the
flags: look at the biggest days first, then at the coins that earned on
them, then ask whether those coins and days could have been traded, then run the counterfactual
that removes them and see what is left. Do not guess a number: if you did not read it in a tool
output, you cannot use it.

Possible causes (pick exactly one):
  data-artifact            the data is wrong or odd: a bad print, a halted or relisted coin, a
                           price that no one could have traded at, a gap filled wrongly
  stand-bug                our own calculation is wrong (weights, returns, funding, costs, a
                           look at the future, a mark at the wrong price)
  unrealistic-tradability  the data is right and the calculation is right, but the money could
                           not have been made: thin markets, a coin the real venue does not list
                           or lists late, an entry at a price the book could not have got
  genuine                  the mechanism is real: the suspicious days are what the strategy is
                           meant to do and survive the counterfactuals
  unclear                  you could not establish the cause within your limits

Your answer is ONE JSON object and nothing else:
{{"cause": "one of the five",
  "summary": "plain Russian, 3 to 6 sentences, the answer first",
  "evidence": [{{"text": "plain Russian: what this call shows", "call": 3,
                "quote": "text copied VERBATIM from that call's output"}}],
  "proposals": [{{"kind": "data_filter | stand_fix | new_check", "text": "plain Russian",
                 "code_request": {{"repo": "qlab", "what": "English, one change",
                                  "why": "English"}}}}],
  "caught_by": ["names of the FIRED flags that pointed at this cause"],
  "new_check": "plain Russian: a check that would have caught it, or null"}}
Rules for the answer, checked by code:
- "call" is the number from the call's output; the quote must be in that output word for word
  (numbers included). Evidence that does not check out is discarded, and a cause with no
  evidence left becomes "unclear". 2 to 8 evidence items.
- "summary", "text" and "new_check" are Russian for a reader who is not a trader (cards/SCHEMA.md,
  plain language): explain a term the first time ("монето-день — одна монета в один день"), do not
  write English trading words in Russian letters, give a number with what it means.
- "code_request" is optional and only for a stand_fix or a new_check that needs code; "repo" is
  "qlab"; at most {max_requests} such proposals. It is a REQUEST to the programmer's queue, built
  on a branch the owner merges by hand; you never change code yourself.
- "caught_by": only flags from the fired list above. Empty if none of them pointed at the cause;
  then say in "new_check" what check would have.
- Never claim a verdict should change. You may say a result is not credible.
"""


@dataclass
class Dossier:
    """The checked answer."""

    cause: str
    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    caught_by: list[str] = field(default_factory=list)
    new_check: str | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# The prompt


def build_prompt(session: Session, scan: Any, cfg: Config) -> str:
    from qlab.registry.models import Idea, Trial
    from qlab.year_view import years_line

    trial = session.get(Trial, scan.trial_id)
    idea = session.get(Idea, scan.idea_id)
    metrics = (trial.metrics or {}) if trial is not None else {}
    # The leading numbers are the run on the strategy's own money (docs/CAPITAL_BASE.md): a
    # percent of the capital it started with, not of a balance grown with set-aside cash. The
    # stand's own are named as such beside them, so the investigator reasons on one base.
    cap_keys = ("cap_ann_return", "cap_total_return", "cap_max_dd", "cap_sharpe", "forward_days")
    stand_keys = ("ann_return_net", "total_return_net", "max_dd", "sharpe_net")
    capital_part = ", ".join(f"{k} {metrics[k]:.4g}" for k in cap_keys
                             if isinstance(metrics.get(k), int | float))
    stand_part = ", ".join(f"{k} {metrics[k]:.4g}" for k in stand_keys
                           if isinstance(metrics.get(k), int | float))
    headline = (f"Numbers on the strategy's own money (the ones to reason on): {capital_part}."
                if capital_part else "This run has no numbers on the strategy's own money.")
    if stand_part:
        headline += (f" As the stand counts them, a share of the book's balance (secondary, "
                     f"not comparable across books): {stand_part}.")
    flags = [fl.FlagResult.from_json(f) for f in scan.flags]
    fired = [f for f in flags if f.fired]
    quiet = [f for f in flags if not f.fired]

    def line(f: fl.FlagResult) -> str:
        why = f" ({f.reason})" if f.reason else ""
        value = "" if f.value is None else f" value {f.value:.4g}, line {f.line:.4g}"
        extra = json.dumps(f.detail, ensure_ascii=False, default=str)[:700] if f.fired else ""
        return f"- {f.name}: {f.state}{value}{why} {extra}".rstrip()

    return INSTRUCTIONS.format(
        idea_id=scan.idea_id, title=idea.title if idea else "", trial_id=scan.trial_id,
        route=scan.route, status=idea.status.value if idea else "?", headline=headline,
        years=years_line(metrics) or "", fired="\n".join(line(f) for f in fired) or "- none",
        quiet="\n".join(line(f) for f in quiet) or "- none",
        max_requests=cfg.investigator.max_programmer_requests)


# --------------------------------------------------------------------------
# Checking the answer


def parse_answer(text: str | None) -> dict | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_calls(path: Path) -> dict[int, str]:
    """The log of one investigation's toolkit calls: call number (1-based, as printed) -> the
    output the call gave."""
    out: dict[int, str] = {}
    if not path.is_file():
        return out
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            out[n] = str(json.loads(raw).get("output", ""))
        except json.JSONDecodeError:
            continue
    return out


def verify(answer: dict, calls: dict[int, str], fired: list[str], cfg: Config,
           repos: set[str]) -> Dossier:
    """Keep the evidence that checks out against the calls log; a cause stands only with some;
    the learning loop names only flags that fired; code requests only for known repositories.
    Pure."""
    from qlab.anomalies.store import CAUSES

    notes: list[str] = []
    accepted, rejected = [], []
    for item in answer.get("evidence") or []:
        if not isinstance(item, dict):
            rejected.append({"raw": str(item)[:200], "why": "not an object"})
            continue
        text = str(item.get("text", "")).strip()
        quote = str(item.get("quote") or "").strip()
        call = item.get("call")
        problems = []
        if not text:
            problems.append("no text")
        elif not _CYRILLIC.search(text):
            problems.append("the text is not Russian")
        if not isinstance(call, int) or isinstance(call, bool) or call not in calls:
            problems.append(f"call {call!r} is not in the log of this investigation's calls")
        elif not quote:
            problems.append("no quote")
        elif _norm(quote) not in _norm(calls[call]):
            problems.append(f"the quote is not verbatim in the output of call {call}")
        entry = {"text": text, "call": call, "quote": quote[:MAX_QUOTE]}
        (rejected.append({**entry, "why": "; ".join(problems)}) if problems
         else accepted.append(entry))

    cause = str(answer.get("cause", "")).strip()
    if cause not in CAUSES:
        notes.append(f"the cause {cause!r} is not one of the five; recorded as unclear")
        cause = "unclear"
    if cause != "unclear" and not accepted:
        notes.append(f"the cause {cause!r} stood without one verified piece of evidence; "
                     "recorded as unclear")
        cause = "unclear"
    summary = str(answer.get("summary", "")).strip()
    if not summary or not _CYRILLIC.search(summary):
        notes.append("the summary is missing or not in Russian")
        summary = ("Агент не дал связного пересказа по-русски; смотрите доказательства и "
                   "журнал его вызовов.")
    caught = [str(f) for f in answer.get("caught_by") or [] if f in FLAGS and f in fired]
    dropped = [str(f) for f in answer.get("caught_by") or [] if str(f) not in caught]
    if dropped:
        notes.append(f"named as having caught the cause, but did not fire: {dropped}")

    proposals = []
    for item in answer.get("proposals") or []:
        if not isinstance(item, dict) or item.get("kind") not in PROPOSAL_KINDS \
                or not str(item.get("text", "")).strip():
            notes.append("a proposal without a known kind or text was dropped")
            continue
        entry = {"kind": item["kind"], "text": str(item["text"]).strip(), "code_request": None}
        request = item.get("code_request")
        if isinstance(request, dict) and str(request.get("what", "")).strip() \
                and str(request.get("why", "")).strip():
            if request.get("repo") in repos and item["kind"] in ("stand_fix", "new_check"):
                entry["code_request"] = {k: str(request[k]).strip()
                                         for k in ("repo", "what", "why")}
            else:
                notes.append("a code request for an unknown repository or a data filter was "
                             "kept as text only")
        proposals.append(entry)
    new_check = answer.get("new_check")
    new_check = str(new_check).strip() if new_check else None
    if new_check and not _CYRILLIC.search(new_check):
        notes.append("the proposed check is not in Russian and was dropped")
        new_check = None
    if cause in ("unclear", "genuine"):
        new_check = None  # nothing to catch: there is nothing confirmed to learn from
    return Dossier(cause=cause, summary=summary, evidence=accepted, rejected=rejected,
                   proposals=proposals, caught_by=caught, new_check=new_check, notes=notes)


# --------------------------------------------------------------------------
# The dossier text


CAUSE_MEANING_RU = {
    "data-artifact": "Данные, на которых считался прогон, содержат ошибку или странность, и "
                     "красивый результат вырос из неё.",
    "stand-bug": "Ошибка в нашем собственном расчёте (стенде): красивое число — следствие "
                 "ошибки, а не рынка.",
    "unrealistic-tradability": "Данные и расчёт верны, но заработать так было бы нельзя: рынок "
                               "слишком тонкий, монеты нет на бирже, где торгуем, или вход был "
                               "бы по цене, которую не получить.",
    "genuine": "Подозрение не подтвердилось: подозрительные дни — это то, что стратегия и должна "
               "делать, и она переживает проверки.",
    "unclear": "Причину установить не удалось в пределах отведённых агенту сил; это не значит, "
               "что всё в порядке.",
}


def render_dossier(scan: Any, dossier: Dossier, *, model: str, tokens: int, calls: int,
                   requests: list[str]) -> str:
    """The Russian text the owner reads. Built by code from the checked parts."""
    lines = [f"# Разбор аномалии: {scan.idea_id}, прогон {scan.trial_id}", "",
             "## Коротко", dossier.summary, "",
             "## Причина",
             f"**{store.CAUSES_RU[dossier.cause]}** (`{dossier.cause}`). "
             f"{CAUSE_MEANING_RU[dossier.cause]}", "",
             "## Какие признаки сработали"]
    lines += [f"- {report.flag_sentence(fl.FlagResult.from_json(f))}"
              for f in scan.flags if f["state"] == fl.FIRES] or ["- ни один"]
    lines += ["", "## Доказательства",
              "Каждое доказательство — цитата из вывода инструмента, который агент действительно "
              "запускал; код сверил её с журналом вызовов дословно.", ""]
    if dossier.evidence:
        lines += ["| Что установлено | Вызов | Цитата из вывода |", "|---|---|---|"]
        lines += [f"| {e['text'].replace('|', '/')} | {e['call']} | "
                  f"`{_norm(e['quote']).replace('|', '/')}` |" for e in dossier.evidence]
    else:
        lines.append("Ни одно доказательство не прошло проверку кодом.")
    if dossier.proposals:
        lines += ["", "## Что предлагается"]
        for p in dossier.proposals:
            kind = {"data_filter": "фильтр данных", "stand_fix": "исправление стенда",
                    "new_check": "новая проверка кодом"}[p["kind"]]
            lines.append(f"- **{kind}:** {p['text']}")
        if requests:
            lines.append(f"- Заявки в очередь программиста (он пишет в отдельной ветке, "
                         f"слияние только по `qlab code merge`): {', '.join(requests)}")
    lines += ["", "## Что извлекли для детектора"]
    if dossier.cause in ("unclear", "genuine"):
        lines.append("Причина не подтверждена как ошибка, поэтому учиться детектору нечему.")
    elif dossier.caught_by:
        lines.append("Причину уловили сработавшие признаки: " + ", ".join(dossier.caught_by) + ".")
    else:
        lines.append("Ни один признак детектора причину не уловил."
                     + (f" Предложена проверка: {dossier.new_check}" if dossier.new_check
                        else " Агент проверки не предложил."))
    if dossier.rejected or dossier.notes:
        lines += ["", "## Что отбросил код"]
        lines += [f"- доказательство отброшено: {r.get('why')}" for r in dossier.rejected]
        lines += [f"- {n}" for n in dossier.notes]
    lines += ["", f"_Модель {model}; вызовов инструментов: {calls}; токенов: {tokens:,}. Агент "
              "ничего не менял в вердиктах, статусах и коде._".replace(",", " ")]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The call


def calls_path(cfg: Config, trial_id: int) -> Path:
    base = cfg.investigator.calls_dir
    base = base if base.is_absolute() else QLAB_ROOT / base
    return base / f"{trial_id}-{uuid.uuid4().hex[:8]}.jsonl"


def programmer_repos() -> set[str]:
    from qlab.agents import programmer as prog

    try:
        return set(prog.load_config().get("repos") or {})
    except (OSError, ValueError):
        return set()


def enqueue_requests(dossier: Dossier, scan: Any, day: date, limit: int, *,
                     queue_path: Path | None = None) -> list[str]:
    """Put the dossier's code requests in the programmer's queue (docs/PROGRAMMER.md). A
    request is a request: nothing is built here, and the owner merges by hand."""
    from qlab.agents import programmer as prog

    queue = prog.load_queue(queue_path or prog.QUEUE)
    ids: list[str] = []
    for p in dossier.proposals:
        request = p.get("code_request")
        if not request or len(ids) >= limit:
            continue
        made = prog.enqueue(queue, repo=request["repo"], what=request["what"],
                            why=f"{request['why']} (anomaly investigation of trial "
                                f"{scan.trial_id}, idea {scan.idea_id})",
                            source=f"anomaly investigator, trial {scan.trial_id}", day=day,
                            strategy=None)
        if made is not None:
            ids.append(made.id)
            p["programmer_request"] = made.id
    if ids:
        prog.save_queue(queue, queue_path or prog.QUEUE)
    return ids


def investigate(session: Session, scan: Any, cfg: Config, *,
                budget: CandidateBudget | ExplicitNoBudget, day: date,
                queue_path: Path | None = None) -> store.NewInvestigation:
    """One investigation: the agent call (through the budget guard), the checks, the dossier,
    the code requests. Writes the dossier row (the caller commits). A budget stop raises
    `BudgetExhausted` before anything is stored."""
    inv = cfg.investigator
    if isinstance(budget, CandidateBudget):
        budget.cap_tokens = min(budget.cap_tokens, inv.max_tokens_per_investigation)
    log = calls_path(cfg, scan.trial_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    before = os.environ.get(toolkit.LOG_ENV)
    os.environ[toolkit.LOG_ENV] = str(log)
    try:
        result = run_agent(
            budget, build_prompt(session, scan, cfg), session=session, agent=AGENT,
            model=inv.model, max_turns=inv.max_turns, cwd=QLAB_ROOT,
            extra_args=("--tools", TOOLS, "--allowedTools", ALLOWED, "--disallowedTools",
                        DISALLOWED))
    finally:
        if before is None:
            os.environ.pop(toolkit.LOG_ENV, None)
        else:
            os.environ[toolkit.LOG_ENV] = before
    calls = read_calls(log)
    answer = parse_answer(result.text)
    fired = list(scan.fired)
    if answer is None:
        dossier = Dossier(
            cause="unclear",
            summary="Агент не вернул ответ в нужном виде, поэтому причина не установлена.",
            notes=["the answer held no JSON"])
    else:
        dossier = verify(answer, calls, fired, cfg, programmer_repos())
    requests = [] if dossier.cause in ("unclear", "genuine") else enqueue_requests(
        dossier, scan, day, inv.max_programmer_requests, queue_path=queue_path)
    tokens = getattr(budget, "spent_tokens", 0)
    text = render_dossier(scan, dossier, model=inv.model, tokens=tokens, calls=len(calls),
                          requests=requests)
    new = store.NewInvestigation(
        idea_id=scan.idea_id, trial_id=scan.trial_id, model=inv.model,
        flags_fingerprint=store.fingerprint(fired, scan.config_version), cause=dossier.cause,
        summary=dossier.summary, dossier=text, evidence=dossier.evidence,
        rejected=dossier.rejected, proposals=dossier.proposals, caught_by=dossier.caught_by,
        new_check=dossier.new_check, programmer_requests=requests, tokens=tokens,
        calls=len(calls))
    store.save_investigation(session, new)
    return new


__all__ = ["AGENT", "ALLOWED", "DISALLOWED", "INSTRUCTIONS", "TOOLS", "Dossier", "build_prompt",
           "calls_path", "enqueue_requests", "investigate", "parse_answer", "read_calls",
           "render_dossier", "verify"]
