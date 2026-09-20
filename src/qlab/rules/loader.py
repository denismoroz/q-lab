"""Loading versioned rulesets from the `rules/` directory at the repo root.

Supports `based_on` inheritance: a version that declares `based_on` starts
from the base version's active rules, overrides them by `id`, and drops any
`id` that appears in its own `retired` section from the inherited set.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qlab.rules.schema import RetiredRule, Rule, RuleSet, parse_version

# src/qlab/rules/loader.py -> parents[3] is the project root (q-lab/).
DEFAULT_RULES_DIR = Path(__file__).resolve().parents[3] / "rules"


def _rules_dir(rules_dir: Path | str | None) -> Path:
    return Path(rules_dir) if rules_dir is not None else DEFAULT_RULES_DIR


def _read_ruleset_file(path: Path) -> RuleSet:
    if not path.is_file():
        raise FileNotFoundError(f"no rules file at {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"rules file {path} must contain a YAML mapping at top level")
    return RuleSet.model_validate(raw)


def list_versions(rules_dir: Path | str | None = None) -> list[str]:
    """List rule versions found in `rules_dir`, sorted oldest to newest.

    Sorting is by parsed (date, sequence) — not lexicographic — so
    `2026-09-20.10` correctly sorts after `2026-09-20.9`.
    """
    directory = _rules_dir(rules_dir)
    if not directory.is_dir():
        return []
    versions = []
    for path in directory.glob("*.yaml"):
        # Filenames are expected to be `<version>.yaml`; skip anything that
        # doesn't parse as a version rather than crash the whole listing.
        try:
            parse_version(path.stem)
        except ValueError:
            continue
        versions.append(path.stem)
    versions.sort(key=parse_version)
    return versions


def load(version: str, rules_dir: Path | str | None = None) -> RuleSet:
    """Load a single ruleset version, resolving `based_on` inheritance.

    The returned `RuleSet.version`/`based_on` reflect the requested version's
    own file; `rules` and `retired` are the fully-merged, effective sets.
    """
    directory = _rules_dir(rules_dir)
    path = directory / f"{version}.yaml"
    ruleset = _read_ruleset_file(path)

    if ruleset.version != version:
        raise ValueError(
            f"rules file {path.name} declares version {ruleset.version!r}, "
            f"expected {version!r} (filename must match the version field)"
        )

    if ruleset.based_on is None:
        return ruleset

    base = load(ruleset.based_on, rules_dir=directory)

    merged_rules: dict[str, Rule] = {rule.id: rule for rule in base.rules}
    for rule in ruleset.rules:
        merged_rules[rule.id] = rule

    retired_ids = {retired.id for retired in ruleset.retired}
    for rule_id in retired_ids:
        merged_rules.pop(rule_id, None)

    merged_retired: dict[str, RetiredRule] = {r.id: r for r in base.retired}
    for r in ruleset.retired:
        merged_retired[r.id] = r

    return RuleSet(
        version=ruleset.version,
        based_on=ruleset.based_on,
        rules=list(merged_rules.values()),
        retired=list(merged_retired.values()),
    )


def load_latest(rules_dir: Path | str | None = None) -> RuleSet:
    """Load the newest ruleset version in `rules_dir`."""
    directory = _rules_dir(rules_dir)
    versions = list_versions(directory)
    if not versions:
        raise FileNotFoundError(f"no rules files found in {directory}")
    return load(versions[-1], rules_dir=directory)
