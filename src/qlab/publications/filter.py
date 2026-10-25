"""The topic rule that marks a paper as a candidate (night/publication_filter.yaml).

A pure function of a title and an abstract and the rules file: the same paper
under the same `version` always gets the same answer. The rule is about topic
only and holds no term about claimed returns (a test checks it); `matched` is
what the morning report shows as the reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

FILTER_PATH = Path("night/publication_filter.yaml")
GROUPS = ("strong", "crypto", "mechanism")


@dataclass(frozen=True)
class Verdict:
    candidate: bool
    score: int  # distinct terms matched, all groups
    matched: dict[str, list[str]]  # group -> terms found

    def why(self) -> str:
        return "; ".join(f"{group}: {', '.join(terms)}" for group, terms in self.matched.items())


@dataclass(frozen=True)
class Rules:
    version: str
    terms: dict[str, tuple[str, ...]]
    strong_alone: int
    crypto_min: int
    mechanism_min: int

    def patterns(self) -> dict[str, list[tuple[str, re.Pattern]]]:
        return {group: [(t, _pattern(t)) for t in terms] for group, terms in self.terms.items()}


def _pattern(term: str) -> re.Pattern:
    words = [re.escape(w) for w in re.split(r"[\s-]+", term.strip().lower()) if w]
    return re.compile(r"(?<![a-z0-9])" + r"[\s-]+".join(words) + r"(?:e?s)?(?![a-z0-9])")


def load_rules(path: Path = FILTER_PATH) -> Rules:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = raw["groups"]
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown term groups in {path}: {sorted(unknown)}")
    return Rules(
        version=str(raw["version"]),
        terms={g: tuple(str(t) for t in groups[g]["terms"]) for g in GROUPS},
        strong_alone=int(raw["rule"]["strong_alone"]),
        crypto_min=int(raw["rule"]["crypto_min"]),
        mechanism_min=int(raw["rule"]["mechanism_min"]),
    )


def judge(rules: Rules, title: str, abstract: str) -> Verdict:
    """Candidate = enough `strong` terms, or enough `crypto` terms together with
    enough `mechanism` terms (the thresholds are in the rules file)."""
    text = f"{title}. {abstract}".lower()
    matched: dict[str, list[str]] = {}
    for group, patterns in rules.patterns().items():
        found = [term for term, pattern in patterns if pattern.search(text)]
        # "perpetual future" is also found inside "perpetual futures": keep the longer.
        found = [t for t in found if not any(t != u and t in u for u in found)]
        if found:
            matched[group] = found
    candidate = (len(matched.get("strong", [])) >= rules.strong_alone
                 or (len(matched.get("crypto", [])) >= rules.crypto_min
                     and len(matched.get("mechanism", [])) >= rules.mechanism_min))
    return Verdict(candidate=candidate, score=sum(len(v) for v in matched.values()),
                   matched=matched)


__all__ = ["FILTER_PATH", "Rules", "Verdict", "judge", "load_rules"]
