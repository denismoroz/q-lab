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

from qlab.calibration.report import (
    render_report,
    summarize_by_generator,
    summarize_real,
    summarize_series,
    write_report,
)
from qlab.calibration.run import (
    REAL_SPEC_PATHS,
    SERIES,
    load_reference_spec,
    run_noise_series,
    run_real_strategies,
)
from qlab.data.snapshot import DEFAULT_SNAPSHOTS_DIR, build_snapshot, describe_universe
from qlab.pipeline.evaluate import evaluate_spec
from qlab.pipeline.spec import load_spec
from qlab.registry.db import get_db_path, session_scope
from qlab.registry.importer import ImportReport, import_graveyard
from qlab.registry.models import DataSnapshot
from qlab.registry.queries import (
    RevivalReport,
    funnel_stats,
    killed_by_retired_rules,
    near_threshold,
    ripe_for_revival,
)
from qlab.rules import load, load_latest

# src/qlab/cli/__init__.py -> parents[3] is the project root (q-lab/), same
# depth as qlab.rules.loader.DEFAULT_RULES_DIR.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

app = typer.Typer(add_completion=False, help="q-lab registry CLI (research only — never trades).")
graveyard_app = typer.Typer(add_completion=False, help="Graveyard queries.")
app.add_typer(graveyard_app, name="graveyard")
data_app = typer.Typer(add_completion=False, help="Market data snapshots (point-in-time panels).")
app.add_typer(data_app, name="data")


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
# data fetch / data list
# --------------------------------------------------------------------------


@data_app.command("fetch")
def data_fetch_cmd(
    source: str = typer.Option(..., "--source", help="hyperliquid or binance"),
    instruments: str | None = typer.Option(
        None,
        "--instruments",
        help=(
            "Comma-separated instrument tickers, e.g. BTC,ETH,SOL. Omit this to fetch the "
            "source's full discovered universe (survivors + delisted), which is the only "
            "way to get a snapshot that passes the honest_universe rule."
        ),
    ),
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD (UTC)"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD (UTC)"),
    interval: str = typer.Option("1h", "--interval", help="1m/5m/15m/1h/4h/1d"),
    include_spot: bool = typer.Option(
        False,
        "--include-spot",
        help=(
            "Also discover and fetch the source's spot markets, named <COIN>-SPOT "
            "(required by any strategy holding spot against a perp leg -- docs/TASKS.md, "
            "T17). Hyperliquid only; only valid without --instruments."
        ),
    ),
) -> None:
    """Fetch a point-in-time MarketPanel snapshot and register it in data_snapshot.

    Without --instruments, the source's discovered universe (dead coins
    included) is used and the snapshot is marked universe_complete=True.
    With --instruments, the snapshot is a hand-picked list and is marked
    universe_complete=False — a survivorship-biased panel by construction,
    which the registry's honest_universe rule will reject.
    """
    instrument_list = None
    if instruments is not None:
        instrument_list = [i.strip() for i in instruments.split(",") if i.strip()]
    if instrument_list is not None:
        typer.echo(
            "warning: --instruments is a manual, hand-picked list -- this snapshot will be "
            "marked universe_complete=False and will NOT pass the honest_universe rule.",
            err=True,
        )

    try:
        panel = build_snapshot(
            source, instrument_list, start, end, interval, include_spot=include_spot
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    no_funding = panel.meta.get("no_funding_instruments") or []
    typer.echo(f"snapshot_id:       {panel.snapshot_id}")
    typer.echo(f"source:            {source}")
    typer.echo(f"interval:          {interval}")
    typer.echo(f"universe_complete: {panel.meta['universe_complete']}")
    typer.echo(f"instruments:       {', '.join(panel.instruments)}")
    typer.echo(f"no_funding (spot): {', '.join(no_funding) if no_funding else '(none)'}")
    typer.echo(f"rows:              {len(panel.prices.index)}")
    typer.echo(f"range:             {panel.prices.index.min()} -> {panel.prices.index.max()}")
    typer.echo(f"path:              {DEFAULT_SNAPSHOTS_DIR / panel.snapshot_id}")


@data_app.command("universe")
def data_universe_cmd(
    source: str = typer.Option(..., "--source", help="hyperliquid or binance"),
) -> None:
    """Print a source's full point-in-time universe (survivors + delisted),
    if its free API exposes one -- the list `data fetch` uses by default.

    Also lists the source's spot markets (<COIN>-SPOT), if it has any --
    `include_spot=True` is always passed here so this command answers "what
    could a snapshot for this source contain" fully, unlike `data fetch`
    where including spot is an opt-in (`--include-spot`) because it changes
    what gets fetched, not just what gets displayed."""
    try:
        described = describe_universe(source, include_spot=True)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if described is None:
        typer.echo(
            f"{source}: no survivorship-free universe is available from its free API "
            "(delisted instruments aren't listed anywhere)."
        )
        typer.echo(
            "Pass --instruments explicitly to `qlab data fetch`; the resulting snapshot "
            "will be marked universe_complete=False."
        )
        raise typer.Exit(code=1)

    for name, is_delisted in described:
        typer.echo(f"{name:<12} {'DELISTED' if is_delisted else 'listed'}")

    n_delisted = sum(1 for _, is_delisted in described if is_delisted)
    typer.echo("")
    typer.echo(f"total: {len(described)} instruments ({n_delisted} delisted)")


@data_app.command("list")
def data_list_cmd() -> None:
    """List registered data_snapshot rows, newest first."""
    with session_scope() as session:
        rows = session.query(DataSnapshot).order_by(DataSnapshot.fetched_at.desc()).all()
        for row in rows:
            instruments = (
                ",".join(sorted(row.instruments))
                if isinstance(row.instruments, dict)
                else row.instruments
            )
            typer.echo(
                f"{row.id[:16]}  {row.source:<12} {row.range_start} -> {row.range_end}  "
                f"rows={row.rows:<8} {instruments}"
            )
        total = len(rows)

    typer.echo("")
    typer.echo(f"total: {total} snapshots")


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


@app.command("evaluate")
def evaluate_cmd(
    spec_path: Path = typer.Argument(  # noqa: B008 - idiomatic typer default, not a mutable-default bug
        ..., metavar="SPEC", help="Path to a spec YAML file"
    ),
    capital: float = typer.Option(
        1000.0, "--capital", help="Capital currently deployable now (USD)"
    ),
    rules_version: str | None = typer.Option(
        None, "--rules", help="Rules version to evaluate against (default: latest)"
    ),
) -> None:
    """Run the full pipeline: spec -> data -> backtest -> metrics -> verdict.

    Prints the computed metrics, every verdict row with its rule and
    numbers, and the routing decision (`reject` / `needs-more-data` /
    `shelf` / `paper`) with its reason. `qlab evaluate` never returns
    `live` — see `qlab.pipeline.evaluate.decide_route`.
    """
    try:
        spec = load_spec(spec_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: invalid spec: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    ruleset = load(rules_version) if rules_version is not None else load_latest()

    with session_scope() as session:
        try:
            result = evaluate_spec(
                spec, session=session, ruleset=ruleset, deployable_capital_usd=capital
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator, not swallowed
            typer.echo(
                f"error: evaluation failed before a trial could be recorded: {exc}", err=True
            )
            raise typer.Exit(code=1) from exc

    typer.echo(f"idea:         {spec.idea_id}  ({spec.title})")
    typer.echo(f"rules:        {ruleset.version}")
    typer.echo(f"trial_id:     {result.trial_id}")
    typer.echo("")

    if result.error is not None:
        typer.echo(f"run FAILED: {result.error}")
        typer.echo("")
        typer.echo(f"routing: {result.routing.route}  ({result.routing.reason})")
        raise typer.Exit(code=1)

    typer.echo("metrics:")
    for name, value in sorted(result.metrics.items()):
        typer.echo(f"  {name:<24}{value}")
    typer.echo("")

    typer.echo("verdicts:")
    for row in result.rules_result.rows:
        passed = "unknown" if row.passed is None else ("pass" if row.passed else "fail")
        value_str = "?" if row.value is None else f"{row.value:g}"
        typer.echo(
            f"  [{passed:<7}] {row.stage.value:<12} {row.rule_id:<24} "
            f"{row.metric}={value_str} {row.comparator}{row.threshold:g}"
        )
    typer.echo("")

    typer.echo(f"routing: {result.routing.route}  ({result.routing.reason})")
    if result.routing.route == "shelf" and result.routing.required_capital_usd is not None:
        typer.echo(f"required capital: ${result.routing.required_capital_usd:,.2f}")


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


@app.command("calibrate")
def calibrate_cmd(
    rules_version: str | None = typer.Option(
        None, "--rules", help="Rules version to calibrate (default: latest)"
    ),
    n_trials: int = typer.Option(
        200,
        "--n-trials",
        help="Noise trials per series (docs/TASKS.md T16 requires at least 200)",
    ),
    capital: float = typer.Option(
        1000.0, "--capital", help="Capital currently deployable now (USD)"
    ),
) -> None:
    """Calibrate a ruleset against noise (docs/TASKS.md T16): how often does
    the ruleset admit a strategy known to have no edge, and what is the best
    `sharpe_net` noise reaches?

    Runs `n_trials` structurally-matched noise strategies per series
    (dollar-neutral and unconstrained, `qlab.calibration.noise`) through the
    real `evaluate_spec` -- same rules, costs, accrual, universe as any real
    candidate -- plus the three real strategies for comparison, then writes
    `docs/CALIBRATION_<rules-version>.md` in Russian
    (`qlab.calibration.report`). Every run is a recorded `trial`.
    """
    if n_trials < 200:
        typer.echo(
            f"warning: docs/TASKS.md T16 requires at least 200 noise trials per series; "
            f"running {n_trials}",
            err=True,
        )

    ruleset = load(rules_version) if rules_version is not None else load_latest()
    reference = load_reference_spec()

    series_summaries = {}
    generator_breakdowns = {}
    with session_scope() as session:
        for series in SERIES:
            trials = run_noise_series(
                series=series,
                n_trials=n_trials,
                session=session,
                ruleset=ruleset,
                deployable_capital_usd=capital,
                reference=reference,
            )
            series_summaries[series] = summarize_series(trials)
            generator_breakdowns[series] = summarize_by_generator(trials)

        real_evaluations = run_real_strategies(
            session=session, ruleset=ruleset, deployable_capital_usd=capital
        )

    titles = {idea_id: load_spec(path).title for idea_id, path in REAL_SPEC_PATHS.items()}
    real_results = summarize_real(real_evaluations, titles)

    content = render_report(
        ruleset=ruleset,
        series_summaries=series_summaries,
        real_results=real_results,
        reference_idea_id=reference.idea_id,
        generated_on=date.today(),
        generator_breakdowns=generator_breakdowns,
    )
    report_path = write_report(content, ruleset.version)

    typer.echo(f"rules:  {ruleset.version}")
    typer.echo(f"report: {report_path}")
    typer.echo("")
    for series, summary in series_summaries.items():
        typer.echo(
            f"  {series:<16} n={summary.n_trials:<5} admitted={summary.n_admitted:<5} "
            f"rate={summary.admission_rate:.1%}  best_sharpe_net={summary.best_sharpe_net}"
        )


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
