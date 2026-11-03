"""The dead band and the power gate as a rules setting (2026-11-03.1, docs/NOISE_BAR.md)."""

from __future__ import annotations

import pytest

from qlab.rules.loader import DEFAULT_RULES_DIR, load
from qlab.rules.schema import NoiseBar, Rule, RuleSet


def test_the_rules_carry_the_band_and_the_line_is_unchanged():
    ruleset = load("2026-11-03.1", DEFAULT_RULES_DIR)
    bar = ruleset.noise_bar
    assert (bar.rule_id, bar.pass_at, bar.keep_above, bar.no_power_no_pass) == (
        "shape_aware_edge", 0.99, 0.975, True)
    line = next(r for r in ruleset.rules if r.id == "shape_aware_edge")
    assert line.threshold == bar.pass_at == 0.99
    previous = load("2026-10-30.2", DEFAULT_RULES_DIR)
    assert previous.noise_bar is None
    # Nothing else moved: the same rules with the same lines, the same power setting.
    assert ruleset.rules == previous.rules
    assert ruleset.noise_power == previous.noise_power
    assert ruleset.stage2_confirmation == previous.stage2_confirmation


def test_a_later_version_inherits_the_band(tmp_path):
    (tmp_path / "2026-11-03.1.yaml").write_text(
        (DEFAULT_RULES_DIR / "2026-11-03.1.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    for path in DEFAULT_RULES_DIR.glob("*.yaml"):
        target = tmp_path / path.name
        if not target.exists():
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "2026-11-04.1.yaml").write_text(
        'version: "2026-11-04.1"\nbased_on: "2026-11-03.1"\nrules: []\n', encoding="utf-8")
    assert load("2026-11-04.1", tmp_path).noise_bar == load("2026-11-03.1", tmp_path).noise_bar


def test_the_band_cannot_sit_above_the_line():
    with pytest.raises(ValueError, match="keep_above"):
        NoiseBar(rule_id="x", pass_at=0.9, keep_above=0.95, rationale="t")


def test_the_band_must_describe_the_line_of_its_rule():
    rule = Rule(id="x", stage="edge", metric="m", comparator=">=", threshold=0.99, fatal=True)
    bar = NoiseBar(rule_id="x", pass_at=0.99, keep_above=0.975, rationale="t")
    RuleSet(version="2026-01-01.1", rules=[rule], noise_bar=bar).check_noise_bar()
    off = NoiseBar(rule_id="x", pass_at=0.95, keep_above=0.9, rationale="t")
    with pytest.raises(ValueError, match="must be the line of rule"):
        RuleSet(version="2026-01-01.1", rules=[rule], noise_bar=off).check_noise_bar()
    with pytest.raises(ValueError, match="does not have"):
        RuleSet(version="2026-01-01.1", rules=[], noise_bar=bar).check_noise_bar()


def test_a_run_judged_under_the_older_rules_is_measured_again():
    """The band is part of what decides, so `restamp` sees the new version as a change."""
    from qlab.restamp import rules_changed

    assert rules_changed("2026-10-30.2", "2026-11-03.1")
