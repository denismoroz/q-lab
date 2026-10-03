"""The scout agent: from today's venue events to at most one draft card
(docs/PLAN.md M7; docs/SOURCES_OF_CANDIDATES.md).

Owner, 2026-10-03: «делай все 3 одну за одной» -- sources of new candidates.
The venue watcher (`qlab.venue_watch`) reports facts; the scout reads them
with the cards q-lab already has and either drafts ONE new card in
`cards/drafts/` following `cards/SCHEMA.md` -- in Russian, without jargon,
every claim cited -- or answers "NO CANDIDATE: <reason>". A draft is not a
candidate until the owner moves it to `cards/` and queues it for the
implementer (`night/implement_queue.yaml`): the search stage proposes, a
person decides what enters the pipeline.

Same discipline as the implementer: read anything, write only its own draft,
never run anything; a write elsewhere is reverted. It runs in the search
stage, the one CLAUDE.md cuts first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from qlab.agents.implementer import _changed_paths, _revert
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.sources import QLAB_ROOT

MODEL = "sonnet"  # docs/PLAN.md: Scout -> sonnet
DRAFTS = Path("cards/drafts")
STRUCTURAL = ("listed", "delisted", "relisted", "removed", "changed", "status")

INSTRUCTIONS = """You are q-lab's scout. q-lab is a research lab that finds and filters
trading strategies (it never places orders). Today's venue events are in {events_path}:
listings, delistings, leverage and margin changes, and the day's extremes in funding,
volume and open interest on Hyperliquid (main and the HIP-3 deployment xyz) and Binance.

Your task: decide whether these events point to a trading mechanism worth a card -- a
reason someone pays the strategy (a structural flow, a forced trade, an incentive),
observable with data q-lab has (venue candles, funding, volume, open interest; no order
book, no liquidation feed). Read first:
- cards/SCHEMA.md      the card format and its filling rules (Russian, no jargon,
                       plain_summary required, every claim cited)
- cards/*.yaml         the existing cards: do not repeat one
- docs/SEARCH_CONSTRAINTS.md if present

If one event (or a pattern across events) supports such a mechanism, write ONE card to
{draft_dir}/<idea_id>.yaml following cards/SCHEMA.md, citing {events_path} for the
observed facts and the venue's documentation or research for the mechanism. Never invent
numbers or sources. If nothing qualifies -- most days nothing does -- write no file.

Write nothing anywhere else. Reply with one line: "DRAFT: <path>" or
"NO CANDIDATE: <reason>"."""


@dataclass
class ScoutOutcome:
    reply: str | None
    draft: Path | None = None
    problems: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)


def scout(day: date, *, budget: CandidateBudget | ExplicitNoBudget, session,
          model: str = MODEL, max_turns: int = 40) -> ScoutOutcome:
    import yaml

    events_path = Path("data/events") / f"{day.isoformat()}.json"
    if not (QLAB_ROOT / events_path).is_file():
        return ScoutOutcome(reply=None, problems=[f"no events file {events_path}"])
    events = json.loads((QLAB_ROOT / events_path).read_text())
    if not any(e["kind"] in STRUCTURAL for e in events):
        # The day's extremes alone come every day and almost never make a
        # candidate: no tokens are spent on them.
        return ScoutOutcome(reply="NO CANDIDATE: no listing, delisting or change of terms today")
    (QLAB_ROOT / DRAFTS).mkdir(parents=True, exist_ok=True)
    before = set(_changed_paths())
    prompt = INSTRUCTIONS.format(events_path=events_path, draft_dir=DRAFTS)
    result = run_agent(budget, prompt, session=session, agent="scout", model=model,
                       max_turns=max_turns, cwd=QLAB_ROOT,
                       extra_args=("--allowedTools", "Read,Glob,Grep,Write",
                                   "--disallowedTools", "Bash,Edit"))
    outcome = ScoutOutcome(reply=(result.text or "").strip()[:500])
    written = []
    for path in set(_changed_paths()) - before:
        if path.startswith(f"{DRAFTS}/") and path.endswith(".yaml"):
            written.append(path)
        else:
            _revert(path)
            outcome.reverted.append(path)
    if outcome.reverted:
        outcome.problems.append(f"wrote outside cards/drafts (reverted): {outcome.reverted}")
    if len(written) > 1:
        outcome.problems.append(f"more than one draft: {written}")
    if written:
        outcome.draft = Path(written[0])
        try:
            card = yaml.safe_load((QLAB_ROOT / outcome.draft).read_text(encoding="utf-8"))
            missing = [k for k in ("idea_id", "title", "plain_summary", "sources")
                       if not card.get(k)]
            if missing:
                outcome.problems.append(f"draft lacks {missing}")
        except Exception as exc:  # noqa: BLE001
            outcome.problems.append(f"draft does not load: {exc}")
    return outcome


__all__ = ["MODEL", "ScoutOutcome", "scout"]
