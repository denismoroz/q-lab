"""q-lab command-line interface (entry point: `qlab`, see pyproject.toml).

Plain, aligned text output only — no extra dependencies (no rich/tabulate).
Every list command ends with a one-line totals summary.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

from qlab.registry.db import get_db_path, session_scope
from qlab.registry.importer import ImportReport, import_graveyard
from qlab.registry.queries import (
    RevivalReport,
    funnel_stats,
    killed_by_retired_rules,
    near_threshold,
    ripe_for_revival,
)
from qlab.rules import load_latest

# src/qlab/cli/__init__.py -> parents[3] is the project root (q-lab/), same
# depth as qlab.rules.loader.DEFAULT_RULES_DIR.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

app = typer.Typer(add_completion=False, help="q-lab registry CLI (research only — never trades).")
graveyard_app = typer.Typer(add_completion=False, help="Graveyard queries.")
app.add_typer(graveyard_app, name="graveyard")


# --------------------------------------------------------------------------
# init-db
# --------------------------------------------------------------------------


@app.command("init-db")
def init_db_cmd() -> None:
    """Run `alembic upgrade head` against the QLAB_DB-configured database."""
    db_path = get_db_path()
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    cfg = AlembicConfig(str(ALEMBIC_INI))
    alembic_command.upgrade(cfg, "head")
    typer.echo(f"database ready: {db_path}")


# --------------------------------------------------------------------------
# import-graveyard
# --------------------------------------------------------------------------


def _print_import_report(report: ImportReport) -> None:
    typer.echo(f"{'drivers inserted':<28}{report.drivers_inserted}")
    typer.echo(f"{'drivers updated':<28}{report.drivers_updated}")
    typer.echo(f"{'ideas inserted':<28}{report.ideas_inserted}")
    typer.echo(f"{'ideas updated':<28}{report.ideas_updated}")
    typer.echo(f"{'status changes':<28}{report.status_changes_count}")
    typer.echo(f"{'verdicts inserted':<28}{report.verdicts_inserted}")
    typer.echo(f"{'  of which measurement':<28}{report.verdicts_measurement}")
    typer.echo(f"{'  of which unknown':<28}{report.verdicts_null_value}")
    typer.echo(f"{'  of which unidentified rule':<28}{report.verdicts_unidentified_rule}")
    typer.echo(f"{'verdicts skipped (dup)':<28}{report.verdicts_skipped_duplicate}")
    typer.echo(f"{'verdicts skipped (bad data)':<28}{report.verdicts_skipped_data_error}")

    if report.status_changes:
        typer.echo("")
        typer.echo("status changes (seed re-import):")
        for idea_id, from_status, to_status in report.status_changes:
            typer.echo(f"  {idea_id}: {from_status} -> {to_status}")

    if report.data_errors:
        typer.echo("")
        typer.echo("data errors:")
        for err in report.data_errors:
            typer.echo(f"  - {err}")

    typer.echo("")
    typer.echo(
        f"total: {report.drivers_inserted + report.drivers_updated} drivers, "
        f"{report.ideas_inserted + report.ideas_updated} ideas, "
        f"{report.verdicts_inserted} verdicts written, "
        f"{report.verdicts_skipped_total} verdicts skipped"
    )


@app.command("import-graveyard")
def import_graveyard_cmd(
    file: Path = typer.Option(  # noqa: B008 - idiomatic typer default, not a mutable-default bug
        Path("seed/graveyard.yaml"), "--file", help="Path to graveyard.yaml"
    ),
) -> None:
    """Import a graveyard.yaml-shaped file into the registry and print the report."""
    with session_scope() as session:
        report = import_graveyard(session, file)
    _print_import_report(report)


# --------------------------------------------------------------------------
# graveyard retired / near / ripe
# --------------------------------------------------------------------------


@graveyard_app.command("retired")
def graveyard_retired_cmd() -> None:
    """Ideas killed by a rule that has since been retired from the current ruleset."""
    ruleset = load_latest()
    with session_scope() as session:
        kills = killed_by_retired_rules(session, ruleset)

    total_verdicts = 0
    for kill in kills:
        typer.echo(f"{kill.idea.id}  ({kill.idea.status.value})  {kill.idea.title}")
        for verdict in kill.verdicts:
            total_verdicts += 1
            typer.echo(
                f"    rule={verdict.rule_id:<28} metric={verdict.metric:<24} "
                f"value={verdict.value!r:<10} threshold={verdict.threshold!r}"
            )

    typer.echo("")
    typer.echo(f"total: {len(kills)} ideas, {total_verdicts} verdicts on retired rules")


@graveyard_app.command("near")
def graveyard_near_cmd(
    margin: float = typer.Option(0.2, "--margin", help="Fallback relative nearness margin"),
) -> None:
    """FAILED verdicts that came close to passing (classify_nearness)."""
    ruleset = load_latest()
    with session_scope() as session:
        report = near_threshold(session, margin, ruleset=ruleset)

    typer.echo(f"near ({len(report.near)}):")
    for miss in report.near:
        v = miss.verdict
        typer.echo(
            f"  {miss.idea.id:<28} rule={v.rule_id:<24} metric={v.metric:<20} "
            f"value={v.value:<10} threshold={v.threshold:<10} "
            f"margin_used={miss.margin_used}"
        )

    if report.undefined_nearness:
        typer.echo("")
        typer.echo(
            f"undefined nearness ({len(report.undefined_nearness)}) — no relative closeness "
            "exists against a zero threshold; judging 'almost passed' here needs an absolute "
            "margin in the metric's own units, and only the ruleset owner can set one "
            "(near_margin_abs), never guessed automatically:"
        )
        for item in report.undefined_nearness:
            v = item.verdict
            typer.echo(
                f"  {item.idea.id:<28} rule={v.rule_id:<24} metric={v.metric:<20} "
                f"value={v.value:<10} threshold={v.threshold:<10} "
                f"raw_distance={item.raw_distance}"
            )

    typer.echo("")
    typer.echo(
        f"total: {len(report.near)} near, {len(report.undefined_nearness)} undefined "
        f"(fallback margin={margin})"
    )


def _print_revival_report(report: RevivalReport, min_new_days: int) -> None:
    # Header first, with both counts, so the headline result (how many are
    # actually ripe) isn't buried under a much longer "cannot auto-revive"
    # tail — on the real graveyard that second group is dozens of lines.
    typer.echo(
        f"ripe for revival: {len(report.ripe)}   "
        f"unknown data range: {len(report.unknown_data_range)}   "
        f"(min_new_days={min_new_days})"
    )
    typer.echo("")

    typer.echo(f"ripe ({len(report.ripe)}):")
    for item in report.ripe:
        v = item.verdict
        typer.echo(
            f"  {item.idea.id:<28} rule={v.rule_id:<24} metric={v.metric:<20} "
            f"data_range_end={v.data_range_end} days_since={item.days_since_rejection}"
        )

    if report.unknown_data_range:
        typer.echo("")
        typer.echo(
            f"cannot auto-revive ({len(report.unknown_data_range)}) — no data_range_end on "
            "the rejecting verdict, so there is no reference point to tell which data is new:"
        )
        for item in report.unknown_data_range:
            v = item.verdict
            typer.echo(f"  {item.idea.id:<28} rule={v.rule_id:<24} metric={v.metric}")


@graveyard_app.command("ripe")
def graveyard_ripe_cmd(
    min_new_days: int = typer.Option(180, "--min-new-days"),
) -> None:
    """Ideas whose rejection is old enough that new data has accumulated since."""
    with session_scope() as session:
        report = ripe_for_revival(session, min_new_days, today=date.today())
    _print_revival_report(report, min_new_days)


# --------------------------------------------------------------------------
# funnel
# --------------------------------------------------------------------------


@app.command("funnel")
def funnel_cmd() -> None:
    """Counts of ideas by status, verdicts by stage, and verdicts by outcome."""
    with session_scope() as session:
        stats = funnel_stats(session)

    typer.echo("ideas by status:")
    for status, count in sorted(stats.ideas_by_status.items()):
        typer.echo(f"  {status:<16}{count}")

    typer.echo("")
    typer.echo("verdicts by stage:")
    for stage, count in sorted(stats.verdicts_by_stage.items()):
        typer.echo(f"  {stage:<16}{count}")

    typer.echo("")
    typer.echo("verdicts by outcome (passed / failed / unknown / measurement):")
    for outcome in ("passed", "failed", "unknown", "measurement"):
        typer.echo(f"  {outcome:<16}{stats.verdicts_by_outcome.get(outcome, 0)}")

    typer.echo("")
    typer.echo("decayed ideas by shutdown cause:")
    if stats.decayed_by_shutdown_cause:
        for cause, count in sorted(stats.decayed_by_shutdown_cause.items()):
            typer.echo(f"  {cause:<16}{count}")
    else:
        typer.echo("  (none)")

    total_ideas = sum(stats.ideas_by_status.values())
    total_verdicts = sum(stats.verdicts_by_outcome.values())
    typer.echo("")
    typer.echo(f"total: {total_ideas} ideas, {total_verdicts} verdicts")


__all__ = ["app"]
