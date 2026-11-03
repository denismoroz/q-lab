"""What the scout may do tonight, decided by code (docs/SCOUT.md).

Owner, 2026-11-03: «нужно разобраться почему он не приносит - а не брать и душить его»,
then «делай все 4». Everything here is a pure function of `night/scout.yaml`, the night's
events and the evidence file (`qlab.event_evidence`): whether the scout is called, which tools
it is handed, how its use of the network is audited, which cited sources are accepted.

The scout agent itself lives in `qlab.agents.scout`; it calls nothing but `qlab.budget.run_agent`.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import yaml

from qlab.pipeline.sources import QLAB_ROOT

CONFIG = Path("night/scout.yaml")
LOOKED_AT = Path("data/scout/looked_at.json")
STRUCTURAL = ("listed", "delisted", "relisted", "removed", "changed", "status")
EVIDENCE_PREFIX = "data/events/evidence/"

# What the scout does when night/scout.yaml is absent (a checkout without it, a test): the
# behaviour of before 2026-11-03 -- structural events only, no network, no evidence.
DEFAULT_CONFIG: dict = {
    "enabled": True, "model": "sonnet", "max_turns": 40,
    "wake": {"on_structural_events": True, "on_evidence": False, "min_events": 30,
             "min_months": 10, "min_net_pct": 1.0, "min_t": 2.5,
             "both_halves_positive": True, "require_tradeable": True, "already_carded": {},
             "rewake_growth": 0.25},
    "budget": {"share_divisor": 1, "max_tokens_per_night": 10**9, "max_calls_per_night": 1},
    "web": {"enabled": False, "search": False, "domains": [], "max_fetches_per_call": 12},
    "papers": {"mode": "queue", "full_text": False, "max_papers_per_night": 3,
               "max_chars_per_night": 60000,
               "max_chars_per_paper": 24000, "min_chars_left": 6000, "rank": {}},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(
            out.get(key), dict) else value
    return out


def load_config(path: Path = CONFIG, root: Path | None = None) -> dict:
    """`night/scout.yaml` over the defaults. A file that does not exist is the defaults."""
    full = path if path.is_absolute() else (root or QLAB_ROOT) / path
    if not full.is_file():
        return DEFAULT_CONFIG
    raw = yaml.safe_load(full.read_text(encoding="utf-8")) or {}
    return _merge(DEFAULT_CONFIG, raw)


# --------------------------------------------------------------------------
# The network: what the scout is handed, and the audit of what it did


def domain_entries(cfg: dict) -> list[dict]:
    return [{"domain": str(d["domain"]).lower(), "paths": list(d.get("paths") or [])}
            for d in (cfg.get("web") or {}).get("domains") or []]


def web_enabled(cfg: dict) -> bool:
    return bool((cfg.get("web") or {}).get("enabled")) and bool(domain_entries(cfg))


def tool_args(cfg: dict) -> tuple[str, ...]:
    """The CLI's permission flags for the scout. Reading and one draft file always; with the
    network switch on, `WebFetch(domain:<d>)` for each declared domain (any other domain is
    refused by the CLI) and WebSearch. With it off the network tools are denied outright."""
    allowed = ["Read", "Glob", "Grep", "Write"]
    denied = ["Bash", "Edit"]
    if web_enabled(cfg):
        allowed += [f"WebFetch(domain:{e['domain']})" for e in domain_entries(cfg)]
        if (cfg.get("web") or {}).get("search"):
            allowed.append("WebSearch")
        else:
            denied.append("WebSearch")
    else:
        denied += ["WebFetch", "WebSearch"]
    return ("--allowedTools", ",".join(allowed), "--disallowedTools", ",".join(denied))


def host_allowed(host: str | None, cfg: dict) -> bool:
    host = (host or "").lower()
    return any(host == e["domain"] or host.endswith("." + e["domain"])
               for e in domain_entries(cfg))


def url_allowed(url: str, cfg: dict) -> bool:
    """The domain is declared and, when the declaration lists paths, the path starts with one."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    for e in domain_entries(cfg):
        if host == e["domain"] or host.endswith("." + e["domain"]):
            return not e["paths"] or any(parsed.path.startswith(p) for p in e["paths"])
    return False


@dataclass
class WebLog:
    """What the scout read from the network in one call, from the tool calls of its stream."""

    fetched: list[dict] = field(default_factory=list)   # {"tool", "url"|"query", "refused"}
    problems: list[str] = field(default_factory=list)

    def urls(self) -> set[str]:
        return {f["url"] for f in self.fetched if f.get("url") and not f.get("refused")}

    def to_dict(self) -> dict:
        return {"fetched": self.fetched, "problems": self.problems}


def audit_web(tool_calls, cfg: dict, *, day: date) -> WebLog:
    """Every WebFetch and WebSearch of a call, in order. A fetch outside the declared domains
    and paths, a search without `allowed_domains` from the list, and more fetches than
    `max_fetches_per_call` are problems; one the CLI refused is only logged."""
    log = WebLog()
    declared = {e["domain"] for e in domain_entries(cfg)}
    fetches = 0
    for call in tool_calls:
        if call.name == "WebFetch":
            url = str(call.input.get("url", ""))
            log.fetched.append({"tool": "WebFetch", "url": url, "day": day.isoformat(),
                                "refused": call.refused})
            fetches += 1
            if not call.refused and not url_allowed(url, cfg):
                log.problems.append(f"fetched {url}, outside night/scout.yaml web.domains")
        elif call.name == "WebSearch":
            query = str(call.input.get("query", ""))
            wanted = [str(d).lower() for d in call.input.get("allowed_domains") or []]
            log.fetched.append({"tool": "WebSearch", "query": query, "allowed_domains": wanted,
                                "day": day.isoformat(), "refused": call.refused})
            if not call.refused and (not wanted or not set(wanted) <= declared):
                log.problems.append(f"searched {query!r} without allowed_domains from "
                                    "night/scout.yaml web.domains")
    cap = int((cfg.get("web") or {}).get("max_fetches_per_call", 12))
    if fetches > cap:
        log.problems.append(f"{fetches} fetches in one call, more than the {cap} allowed")
    return log


def source_problems(card: object, *, evidence_file: str | None, web: WebLog,
                    handed_links: set[str], root: Path | None = None) -> list[str]:
    """The sources check of a draft card (docs/SCOUT.md). Each `sources` entry must name
    something checkable: a file of q-lab that exists (the night's events, the evidence file),
    the link of a paper handed to the scout, or a page the scout fetched in this call (cited
    by its URL, with `retrieved` the date). A URL that was none of these was not read."""
    root = root or QLAB_ROOT
    if not isinstance(card, dict):
        return ["the draft is not a mapping"]
    sources = card.get("sources")
    if not isinstance(sources, list) or not sources:
        return ["sources needs at least one {url, title, published}"]
    problems = []
    fetched = web.urls()
    for i, source in enumerate(sources, 1):
        url = str(source.get("url", "")).strip() if isinstance(source, dict) else ""
        if not url or not source.get("title"):
            problems.append(f"source {i} lacks url or title")
        elif url.startswith("http"):
            if url not in fetched and url not in handed_links:
                problems.append(f"source {i} {url} was not fetched in this call and is not a "
                                "paper handed to the scout")
            elif url in fetched and not source.get("retrieved"):
                problems.append(f"source {i} {url} is a fetched page: add `retrieved` (the date)")
        elif not (root / url).is_file():
            problems.append(f"source {i} {url} is not a file of q-lab")
        elif url.startswith(EVIDENCE_PREFIX) and evidence_file and url != evidence_file:
            problems.append(f"source {i} {url} is not tonight's evidence file {evidence_file}")
    return problems


# --------------------------------------------------------------------------
# Whether the scout is called


@dataclass
class Wake:
    call: bool
    reason: str
    structural: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    held_back: list[dict] = field(default_factory=list)   # passed the numbers, already carded
    seen: list[dict] = field(default_factory=list)        # passed, already looked at

    def to_dict(self) -> dict:
        return {"call": self.call, "reason": self.reason,
                "structural": len(self.structural),
                "signals": [s["id"] for s in self.signals],
                "held_back": [h["id"] for h in self.held_back],
                "seen": [s["id"] for s in self.seen]}


def carded_by(key: str, cfg: dict) -> list[str]:
    """Where a pattern's mechanism is already held (`wake.already_carded`); empty when nowhere."""
    for pattern, holders in ((cfg.get("wake") or {}).get("already_carded") or {}).items():
        if fnmatch.fnmatch(key, pattern):
            return list(holders or [])
    return []


def passes(measure: dict, wake: dict) -> bool:
    """One measure of the evidence file against the numbers of `wake` (night/scout.yaml)."""
    t, mean = measure.get("t"), measure.get("mean_pct")
    if t is None or mean is None:
        return False
    if measure.get("n", 0) < wake["min_events"] or measure.get("months", 0) < wake["min_months"]:
        return False
    if mean < wake["min_net_pct"] or t < wake["min_t"]:
        return False
    if wake.get("both_halves_positive"):
        halves = (measure.get("first_half_pct"), measure.get("second_half_pct"))
        if any(h is None or h <= 0 for h in halves):
            return False
    return True


def evidence_signals(evidence: dict | None, cfg: dict) -> tuple[list[dict], list[dict]]:
    """(signals, held back): the measures that pass, split by whether a card or the catalogue
    already holds the mechanism of their pattern."""
    wake = cfg["wake"]
    signals, held = [], []
    for pattern in (evidence or {}).get("patterns") or []:
        if wake.get("require_tradeable") and not pattern.get("tradeable"):
            continue
        for measure in pattern.get("measures") or []:
            if not passes(measure, wake):
                continue
            row = {"id": f"{pattern['key']}::{measure['id']}", "pattern": pattern["key"],
                   "measure": measure["id"], "position": measure.get("position"),
                   **{k: measure.get(k) for k in ("n", "months", "mean_pct", "median_pct", "t",
                                                  "first_half_pct", "second_half_pct")}}
            holders = carded_by(pattern["key"], cfg)
            (held if holders else signals).append({**row, "carded_by": holders} if holders
                                                  else row)
    return signals, held


def load_looked_at(root: Path | None = None) -> dict:
    path = (root or QLAB_ROOT) / LOOKED_AT
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mark_looked_at(signals: list[dict], day: date, reply: str | None,
                   root: Path | None = None) -> None:
    """Remember that the scout looked at these signals: they wake it again only when their
    sample has grown by `wake.rewake_growth`."""
    if not signals:
        return
    path = (root or QLAB_ROOT) / LOOKED_AT
    held = load_looked_at(root)
    for s in signals:
        held[s["id"]] = {"day": day.isoformat(), "n": s["n"], "reply": (reply or "")[:200]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(held, ensure_ascii=False, indent=1), encoding="utf-8")


def decide(events: list[dict], evidence: dict | None, cfg: dict, *,
           looked_at: dict | None = None) -> Wake:
    """Whether the scout is called tonight. Code's rule, not the agent's: a structural event
    (as before 2026-11-03), or a measured pattern of the evidence file that passes
    `wake`, that no card holds and that the scout has not looked at with this sample."""
    wake = cfg["wake"]
    structural = [e for e in events if e.get("kind") in STRUCTURAL]
    signals, held, seen = [], [], []
    if wake.get("on_evidence"):
        found, held = evidence_signals(evidence, cfg)
        looked = looked_at or {}
        growth = float(wake.get("rewake_growth", 0.25))
        for s in found:
            before = looked.get(s["id"])
            (seen if before and s["n"] < before["n"] * (1 + growth) else signals).append(s)
    if wake.get("on_structural_events") and structural:
        return Wake(True, f"{len(structural)} structural event(s) today", structural, signals,
                    held, seen)
    if signals:
        return Wake(True, f"{len(signals)} measured pattern(s) not carded and not yet looked at",
                    structural, signals, held, seen)
    why = "no listing, delisting or change of terms today"
    if wake.get("on_evidence"):
        why += ("; no measured pattern passes the evidence rules and is new "
                f"({len(held)} held by cards, {len(seen)} already looked at)")
    return Wake(False, why, structural, [], held, seen)


# --------------------------------------------------------------------------
# The scout's own share of the night


def cap_budget(budget, cfg: dict, *, spent_tonight: int) -> None:
    """Lower a candidate budget to what `budget.max_tokens_per_night` leaves after the scout's
    earlier calls tonight (the guard then stops the call when less than the cheapest call is
    left)."""
    cap = int(cfg["budget"]["max_tokens_per_night"]) - spent_tonight
    budget.cap_tokens = max(0, min(budget.cap_tokens, budget.spent_tokens + cap))


__all__ = ["CONFIG", "DEFAULT_CONFIG", "EVIDENCE_PREFIX", "STRUCTURAL", "Wake", "WebLog",
           "audit_web", "cap_budget", "carded_by", "decide", "domain_entries", "evidence_signals",
           "host_allowed", "load_config", "load_looked_at", "mark_looked_at", "passes",
           "source_problems", "tool_args", "url_allowed", "web_enabled"]
