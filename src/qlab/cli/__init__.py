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
from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
from qlab.data.recorder import DEFAULT_STORE_DIR as DEFAULT_RECORDED_DIR
from qlab.data.recorder import record
from qlab.data.snapshot import DEFAULT_SNAPSHOTS_DIR, build_snapshot, describe_universe
from qlab.pipeline.evaluate import evaluate_spec
from qlab.pipeline.spec import load_spec
from qlab.pipeline.variants import split_cadence_variant, timeframe_variants
from qlab.registry.db import get_db_path, session_scope
from qlab.registry.importer import ImportReport, import_graveyard
from qlab.registry.models import DataSnapshot
from qlab.registry.queries import (
    RevivalReport,
    family_report,
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
spec_app = typer.Typer(add_completion=False, help="Declared sets of spec variants (T25).")
app.add_typer(spec_app, name="spec")


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


@graveyard_app.command("reclassified")
def graveyard_reclassified_cmd() -> None:
    """Ideas q-lab rejected and a later run routed elsewhere (docs/TASKS.md
    T26): rejections the instrument itself took back."""
    from qlab.registry.queries import reclassified_rejects

    with session_scope() as session:
        rows = reclassified_rejects(session)
    for r in rows:
        typer.echo(f"{r.idea_id:<36} rejected in trial {r.rejected_trial_id}, "
                   f"later {r.later_route} (trial {r.later_trial_id})")
    typer.echo(f"{len(rows)} rejection(s) taken back")


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


@data_app.command("record")
def data_record_cmd(
    source: list[str] = typer.Option(  # noqa: B008 - idiomatic typer default
        ..., "--source", help="hyperliquid or hyperliquid-<dex>; repeat for several"
    ),
    interval: list[str] = typer.Option(  # noqa: B008 - idiomatic typer default
        ..., "--interval", help="Candle interval to keep, e.g. 1d, 1h; repeat for several"
    ),
    store: Path = typer.Option(  # noqa: B008 - idiomatic typer default
        DEFAULT_RECORDED_DIR, "--store", help="Where the recorded data lives"
    ),
) -> None:
    """Append every closed candle and funding settlement the venue serves now
    to q-lab's own store, before the venue stops serving them (docs/TASKS.md
    T33). Meant to run daily; a missed day is caught up on the next run."""
    failed = False
    for name in source:
        report = record(name, interval, store_dir=store)
        typer.echo(
            f"{name}: listed {report.listed}, delisted {report.delisted}, "
            f"new candles {report.new_candles}, new funding {report.new_funding}, "
            f"failed {len(report.failed)}"
        )
        if report.failed:
            failed = True
            typer.echo(f"  failed: {', '.join(report.failed)}", err=True)
    if failed:
        raise typer.Exit(code=1)


@data_app.command("coin-attributes")
def data_coin_attributes_cmd(
    start: str = typer.Option(..., "--start", help="First date, YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="Last date, YYYY-MM-DD (default: today)"),
) -> None:
    """Fetch CoinMarketCap's weekly historical snapshots (rank, market cap,
    tags as of each Sunday) into data/raw/coinmarketcap/historical
    (docs/COIN_ATTRIBUTES.md). Snapshots already on disk are not refetched."""
    from qlab.data.sources import coinmarketcap

    last = date.fromisoformat(end) if end else date.today()
    new, failed = coinmarketcap.download(date.fromisoformat(start), last)
    typer.echo(f"new snapshots {new}, failed {len(failed)}")
    if failed:
        typer.echo("  failed: " + ", ".join(d.isoformat() for d in failed), err=True)
        raise typer.Exit(code=1)


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
    noise_trials: int | None = typer.Option(
        None,
        "--noise-trials",
        help=(
            "Also run this many structurally matched noise trials and fold the "
            "candidate's noise percentile into its verdict (the shape-aware bar, "
            "docs/TASKS.md T28). Without it, a ruleset with `shape_aware_edge` "
            "reads that rule as unknown and routes needs-more-data."
        ),
    ),
) -> None:
    """Run the full pipeline: spec -> data -> backtest -> metrics -> verdict.

    Prints the computed metrics, every verdict row with its rule and
    numbers, the routing decision (`reject` / `needs-more-data` /
    `not-evaluable` / `shelf` / `paper`) with its reason, and what the run did
    to the idea's status. `qlab evaluate` never returns `live` — see
    `qlab.pipeline.evaluate.decide_route`.
    """
    try:
        spec = load_spec(spec_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: invalid spec: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    ruleset = load(rules_version) if rules_version is not None else load_latest()

    with session_scope() as session:
        try:
            if noise_trials is None:
                result = evaluate_spec(
                    spec, session=session, ruleset=ruleset, deployable_capital_usd=capital
                )
            else:
                result = evaluate_spec_with_shape_aware_bar(
                    spec,
                    session=session,
                    ruleset=ruleset,
                    deployable_capital_usd=capital,
                    n_trials=noise_trials,
                ).candidate
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator, not swallowed
            typer.echo(
                f"error: evaluation failed before a trial could be recorded: {exc}", err=True
            )
            raise typer.Exit(code=1) from exc

    typer.echo(f"idea:         {spec.idea_id}  ({spec.title})")
    typer.echo(f"rules:        {ruleset.version}")
    typer.echo(f"trial_id:     {result.trial_id}")
    typer.echo("")

    def _echo_status() -> None:
        decision = result.status_decision
        if decision is None:
            return
        moved = decision.new_status.value if decision.new_status is not None else "unchanged"
        typer.echo(f"idea status:  {moved}  ({decision.reason})")

    if result.error is not None:
        typer.echo(f"run FAILED: {result.error}")
        typer.echo("")
        typer.echo(f"routing: {result.routing.route}  ({result.routing.reason})")
        _echo_status()
        raise typer.Exit(code=1)

    if result.routing.route == "not-evaluable" and result.metrics is None:
        # Nothing was computed, on purpose (docs/TASKS.md T24): print the
        # reason, not an empty metrics block that would read like a result.
        # An exploratory run did compute, and falls through to print it.
        typer.echo(f"routing: {result.routing.route}  ({result.routing.reason})")
        _echo_status()
        return

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
    _echo_status()


# --------------------------------------------------------------------------
# spec variants and family report (docs/TASKS.md T25)
# --------------------------------------------------------------------------


@spec_app.command("check-sources")
def spec_check_sources_cmd(
    spec_paths: list[Path] = typer.Argument(..., metavar="SPEC..."),  # noqa: B008
) -> None:
    """Does every number and choice in these specs cite a checkable source
    (docs/SOURCES.md)? A spec that fails does not run."""
    from qlab.pipeline.sources import source_problems

    failed = 0
    for path in spec_paths:
        problems = source_problems(load_spec(path))
        typer.echo(f"{path}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        for problem in problems:
            typer.echo(f"  {problem}")
        failed += bool(problems)
    if failed:
        typer.echo(f"{failed} of {len(spec_paths)} spec(s) would not run", err=True)
        raise typer.Exit(code=1)


@spec_app.command("timeframes")
def spec_timeframes_cmd(
    spec_path: Path = typer.Argument(..., metavar="SPEC"),  # noqa: B008
    interval: list[str] = typer.Option(..., "--interval", "-i"),  # noqa: B008
    out_dir: Path = typer.Option(Path("specs"), "--out-dir"),  # noqa: B008
) -> None:
    """Write one variant per interval. The set is declared before any run;
    report it whole with `qlab family`. Refuses intervals the strategy does
    not declare valid."""
    try:
        written = timeframe_variants(spec_path, interval, out_dir)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for path in written:
        typer.echo(str(path))


@spec_app.command("split")
def spec_split_cmd(
    spec_path: Path = typer.Argument(..., metavar="SPEC"),  # noqa: B008
    fast: str = typer.Option(..., "--fast", help="Bar interval the book is checked on, e.g. 1h"),
    entry: str = typer.Option(..., "--entry", help="Grow the book only at these closes, e.g. 1D"),
    exit_: str = typer.Option(..., "--exit", help="Shrink the book at these closes, e.g. 1h"),
    out_dir: Path = typer.Option(Path("specs"), "--out-dir"),  # noqa: B008
) -> None:
    """Write the split-cadence variant: entries and exits on different timeframes."""
    try:
        path = split_cadence_variant(
            spec_path, fast=fast, entry_every=entry, exit_every=exit_, out_dir=out_dir
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(path))


@app.command("family")
def family_cmd(idea_id: str = typer.Argument(..., metavar="IDEA")) -> None:
    """All variants of one idea side by side, with the whole-set summary."""
    with session_scope() as session:
        members = family_report(session, idea_id)
    if not members:
        typer.echo(f"{idea_id}: no routed runs")
        return
    typer.echo(f"{'v':>3} {'bars':>4} {'entry/exit':>11} {'return':>8} {'sharpe':>7} "
               f"{'noise%':>7}  route")
    positive = beats_noise = with_numbers = 0
    for m in members:
        metrics = m.metrics or {}
        ret, sharpe = metrics.get("ann_return_net"), metrics.get("sharpe_net")
        pct = metrics.get("noise_return_percentile")
        cadence = f"{m.entry_every}/{m.exit_every}" if m.entry_every else "-"
        if ret is not None:
            with_numbers += 1
            positive += ret > 0
            beats_noise += pct is not None and pct >= 0.99
        fmt = lambda v, f: f.format(v) if v is not None else "-"  # noqa: E731
        typer.echo(
            f"{m.spec_version:>3} {m.interval or '-':>4} {cadence:>11} "
            f"{fmt(ret, '{:+.2%}'):>8} {fmt(sharpe, '{:.2f}'):>7} {fmt(pct, '{:.3f}'):>7}  "
            f"{m.route}"
        )
    typer.echo(
        f"\nwhole set: {len(members)} variants, {with_numbers} with numbers; "
        f"positive on {positive} of {with_numbers}; beats matched noise on "
        f"{beats_noise} of {with_numbers}"
    )


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


night_app = typer.Typer(help="The nightly run (docs/NIGHT.md).", no_args_is_help=True)
app.add_typer(night_app, name="night")


@night_app.command("run")
def night_run_cmd(
    day: str | None = typer.Option(None, "--date", help="Night to run or resume (YYYY-MM-DD)"),
    graveyard: bool | None = typer.Option(None, "--graveyard/--no-graveyard",
                                          help="Sweep the graveyard by regime (default: Sundays)"),
    paper: bool = typer.Option(True, "--paper/--no-paper",
                               help="Reconcile the production paper books with the stand"),
    implement: bool = typer.Option(True, "--implement/--no-implement",
                                   help="Stage 2: cards in night/implement_queue.yaml (tokens)"),
    search: bool = typer.Option(True, "--search/--no-search",
                                help="Stage 3: venue watcher and the scout's draft (tokens)"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
) -> None:
    """Re-evaluate the watch list on data extended to the last closed day,
    sweep the graveyard by regime on Sundays, reconcile paper, write the
    morning report. Spends no tokens; resumes where a stopped run left off."""
    from datetime import date as _date

    from qlab.night import run

    path = run(_date.fromisoformat(day) if day else None, graveyard=graveyard, paper=paper,
               implement=implement, search=search, noise_trials=noise_trials)
    typer.echo(f"report: {path}")


regimes_app = typer.Typer(help="Market regimes: BTC bull / flat / bear (docs/REGIMES.md).",
                          no_args_is_help=True)
app.add_typer(regimes_app, name="regimes")


@regimes_app.command("build")
def regimes_build_cmd() -> None:
    """Fetch BTC's whole daily history and label every day by the tercile of
    its 30-day return (owner, 2026-10-02: «BTC, терцили, окно 30 дней»)."""
    from qlab.regimes import REGIMES, build

    series = build()
    labels = series.labels.dropna()
    typer.echo(series.source)
    typer.echo(f"bear below {series.bear_below:+.1%}, bull above {series.bull_above:+.1%} "
               "(30-day BTC return)")
    for r in REGIMES:
        typer.echo(f"  {r:<5} {(labels == r).mean():.0%} of {len(labels)} days")
    typer.echo(f"latest: {labels.index[-1]:%Y-%m-%d} {labels.iloc[-1]}")


budget_app = typer.Typer(help="Token budget guard (docs/BUDGET.md).", no_args_is_help=True)
app.add_typer(budget_app, name="budget")


@budget_app.command("status")
def budget_status_cmd() -> None:
    """Tonight's ceiling from the latest recorded reading of the shared
    windows -- no call is made (use `qlab budget probe` for a fresh one)."""
    from datetime import UTC, datetime, timedelta

    from qlab.budget.guard import WEEKLY_CEILING, NightBudget, calibrate, nights_until
    from qlab.budget.usage import UsageReading, Window
    from qlab.registry.models import TokenSpend

    with session_scope() as session:
        last = (session.query(TokenSpend).filter(TokenSpend.seven_day_util.is_not(None))
                .order_by(TokenSpend.at.desc(), TokenSpend.id.desc()).first())
        cal = calibrate(session)
        if last is None:
            typer.echo("no reading recorded yet: run `qlab budget probe`")
            return
        resets = last.seven_day_resets_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        reading = UsageReading("allowed", None, Window(last.seven_day_util, resets))
        night = NightBudget.open(session, reading, now=now)
        week_start = resets - timedelta(days=7)
        rows = session.query(TokenSpend).filter(TokenSpend.at >= week_start).all()
    by_stage: dict[str, int] = {}
    for r in rows:
        total = (r.tokens_in or 0) + (r.tokens_out or 0) + (r.tokens_cache_write or 0) + (
            r.tokens_cache_read or 0)
        by_stage[r.stage] = by_stage.get(r.stage, 0) + total
    typer.echo(f"last reading     {last.at:%Y-%m-%d %H:%M} UTC: weekly {last.seven_day_util:.0%}, "
               f"five-hour {last.five_hour_util or 0:.0%}")
    typer.echo(f"weekly reset     {resets:%Y-%m-%d %H:%M} UTC "
               f"({nights_until(resets, now)} night(s) left)")
    typer.echo(f"tonight's target weekly {night.target_util:.1%} "
               f"(ceiling {WEEKLY_CEILING:.0%} of the week)")
    typer.echo(f"calibration      {cal.util_per_token * 1e8:.2f} pp of the week per million "
               f"tokens ({cal.source}); "
               f"cheapest call {cal.min_call_tokens:,} tokens")
    typer.echo(f"tokens tonight   about {night.tokens_left():,}")
    typer.echo("spent this week  " + (", ".join(f"{k} {v:,}" for k, v in by_stage.items())
                                     or "nothing"))


@budget_app.command("probe")
def budget_probe_cmd() -> None:
    """One minimal call to read the shared windows; recorded in token_spend."""
    from qlab.budget import probe

    with session_scope() as session:
        result = probe(session)
    r = result.report.reading
    if r is None or r.seven_day is None:
        typer.echo("the call reported no window reading", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"status {r.status}; weekly {r.seven_day.utilization:.0%} (resets "
               f"{r.seven_day.resets_at:%Y-%m-%d %H:%M} UTC); five-hour "
               f"{r.five_hour.utilization if r.five_hour else float('nan'):.0%}; "
               f"cost {result.report.usage.total if result.report.usage else 0:,} tokens")


@app.command("allocation")
def allocation_cmd(
    path: Path = typer.Argument(..., metavar="ALLOCATION"),  # noqa: B008
) -> None:
    """Switch capital between strategies by market regime (qlab.allocation,
    docs/REGIMES.md) and compare the declared assignments: whole window,
    declared periods, and per regime. A description: no rule reads it."""
    import pandas as pd
    import yaml

    from qlab.allocation import leg_from_spec, summary, switch
    from qlab.regimes import REGIMES, breakdown, load
    from qlab.strategies.detectors import causal_labels, market_closes

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    with session_scope() as session:
        legs = {name: leg_from_spec(load_spec(Path(spec)), session)
                for name, spec in config["legs"].items()}
    closes = market_closes()
    hindsight = load()
    if closes is None or hindsight is None:
        typer.echo("error: run `qlab regimes build` first", err=True)
        raise typer.Exit(code=1)
    labels = causal_labels(closes)
    start = max(leg.first_active for leg in legs.values()).normalize() + pd.Timedelta(days=1)
    end = min(leg.daily_return.index.max() for leg in legs.values())
    days = pd.date_range(start, end, freq="1D", tz="UTC")
    typer.echo(f"{config['name']}: {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d} ({len(days)} days), "
               "every leg holding positions")
    periods = {"whole": (days[0], days[-1])}
    periods.update({k: (pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
                    for k, (a, b) in (config.get("periods") or {}).items()})
    header = f"{'':<26}" + "".join(f"{p:>34}" for p in periods)
    typer.echo(header)
    from qlab.allocation import learned_plan
    from qlab.regimes import WINDOW_DAYS

    for label, assignment in config["assignments"].items():
        if assignment == "learned":
            # The regime -> leg choice made from the past only, day by day.
            assignment = learned_plan(legs, labels, days, min_days=WINDOW_DAYS)
        returns, held = switch(legs, assignment, labels, days)
        cells = []
        for a, b in periods.values():
            part = returns[(returns.index >= a) & (returns.index <= b)]
            st = summary(part)
            cells.append(f"{st['ann_return']:+7.1%} Sh {st['sharpe']:4.2f} dd {st['max_dd']:6.1%}")
        switches = int((held.fillna("cash") != held.fillna("cash").shift()).iloc[1:].sum())
        typer.echo(f"{label:<26}" + "".join(f"{c:>34}" for c in cells) + f"   switches {switches}")
        by = breakdown(returns, hindsight, 365.0)
        typer.echo(f"{'':<26}" + "  ".join(
            f"{r}: {by.get(f'regime_{r}_return', float('nan')):+.1%}" for r in REGIMES))


@app.command("research")
def research_cmd(
    name: str = typer.Argument(..., metavar="NAME"),
    idea: list[str] = typer.Option(..., "--idea", help="Related idea ids (repeat)"),  # noqa: B008
    spec: list[Path] = typer.Option(..., "--spec", help="The strategy's specs (repeat)"),  # noqa: B008
    doc: list[Path] = typer.Option([], "--doc", help="Documents to read (repeat)"),  # noqa: B008
    max_experiments: int = typer.Option(3, "--max-experiments"),
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
) -> None:
    """The researcher agent studies a strategy across the registry and its
    documents, writes a memo for the owner and declares experiments; code then
    runs the experiments on the stand and appends their results to the memo
    (docs/RESEARCHER.md)."""
    from qlab.agents.researcher import research
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate

    with session_scope() as session:
        budget = open_candidate(session, None, Stage.RECHECK)
        outcome = research(name, idea_ids=idea, specs=spec, docs=doc, budget=budget,
                           session=session, max_experiments=max_experiments)
        spent = budget.spent_tokens
    typer.echo(f"researcher: memo {outcome.memo}; experiments {[str(p) for p in outcome.specs]}; "
               f"{spent:,} tokens")
    for problem in outcome.problems:
        typer.echo(f"  problem: {problem}")
    if not evaluate or outcome.memo is None:
        return
    from qlab.agents.researcher import run_experiments

    typer.echo("\n".join(run_experiments(outcome.memo, outcome.specs, name=name,
                                         noise_trials=noise_trials)))


@app.command("implement")
def implement_cmd(
    card: Path = typer.Argument(..., metavar="CARD"),  # noqa: B008
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
) -> None:
    """Have the implementer agent write the strategy and spec for CARD
    (docs/IMPLEMENTER.md), then run every guard: allowed files only, sources,
    import, the reviewer against the card, and the stand. Through the budget
    guard."""
    from qlab.agents.implementer import implement
    from qlab.agents.reviewer import code_files, run_review
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate

    with session_scope() as session:
        budget = open_candidate(session, None, Stage.IMPLEMENT)
        outcome = implement(card, budget=budget, session=session)
        spent = budget.spent_tokens
    typer.echo(f"implementer: {outcome.idea_id}: {outcome.code_path}, {outcome.spec_path}; "
               f"{spent:,} tokens; reply: {outcome.agent_reply}")
    for problem in outcome.problems:
        typer.echo(f"  problem: {problem}")
    if outcome.not_expressible:
        typer.echo("  the card is not expressible on the stand's data (an answer, not a failure)")
        return
    if not outcome.ok:
        raise typer.Exit(code=1)
    spec = load_spec(outcome.spec_path)
    with session_scope() as session:
        budget = open_candidate(session, spec.idea_id, Stage.IMPLEMENT)
        review = run_review(spec=spec, spec_text=outcome.spec_path.read_text(encoding="utf-8"),
                            sources={str(card): card.read_text(encoding="utf-8")},
                            code=code_files(spec), budget=budget, session=session)
    typer.echo(f"reviewer: {len(review.accepted)} accepted, {len(review.rejected)} rejected")
    for f in review.accepted:
        typer.echo(f"  [{f.kind}] {f.summary}")
    if evaluate:
        from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
        from qlab.registry import repo
        from qlab.registry.models import AssetClass, Idea, Profile, SourceType
        from qlab.rules.loader import load_latest

        with session_scope() as session:
            if session.get(Idea, spec.idea_id) is None:
                repo.upsert_idea(session, id=spec.idea_id, title=spec.title,
                                 source_type=SourceType.INTERNAL,
                                 asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                                 notes=f"Implemented by the implementer agent from {card}.")
            result = evaluate_spec_with_shape_aware_bar(
                spec, session=session, ruleset=load_latest(), deployable_capital_usd=3000.0,
                n_trials=noise_trials).candidate
        typer.echo(f"stand: trial {result.trial_id}: {result.routing.route} "
                   f"({result.routing.reason})")


@app.command("review")
def review_cmd(
    spec_path: Path = typer.Argument(..., metavar="SPEC"),  # noqa: B008
    source: list[Path] = typer.Option(  # noqa: B008
        ..., "--source", help="What the implementation must express: a card, quotes, the "
        "production code (repeat)"),
    blind: bool = typer.Option(False, "--blind", help="Hide the spec's own list of unexpressed "
                               "mechanisms and review answers, to test the reviewer"),
    model: str = typer.Option("opus", "--model"),
) -> None:
    """Have the reviewer agent check the implementation against its source
    (docs/REVIEWER.md). Runs through the budget guard; findings without
    verified evidence are discarded; accepted blocking findings make the spec
    not evaluable until declared or answered."""
    from datetime import UTC, datetime, timedelta

    import yaml

    from qlab.agents.reviewer import code_files, run_review, unanswered_blocking
    from qlab.budget import NightBudget, Stage, probe
    from qlab.budget.usage import UsageReading, Window
    from qlab.pipeline.sources import QLAB_ROOT
    from qlab.registry.models import TokenSpend

    spec = load_spec(spec_path)
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if blind:
        for key in ("unexpressed_mechanisms", "review_answers"):
            raw.pop(key, None)
        spec_text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    else:
        spec_text = spec_path.read_text(encoding="utf-8")
    texts = {}
    for path in source:
        resolved = path if path.is_file() else QLAB_ROOT.parent / path
        texts[str(path)] = resolved.read_text(encoding="utf-8")
    code = code_files(spec)

    with session_scope() as session:
        last = (session.query(TokenSpend).filter(TokenSpend.seven_day_util.is_not(None))
                .order_by(TokenSpend.at.desc(), TokenSpend.id.desc()).first())
        now = datetime.now(UTC)
        # A reading older than the five-hour window's own length describes a
        # window that is gone; take a fresh one first.
        fresh = last is not None and last.at.replace(tzinfo=UTC) > now - timedelta(hours=5)
        if fresh:
            reading = UsageReading("allowed", None,
                                   Window(last.seven_day_util,
                                          last.seven_day_resets_at.replace(tzinfo=UTC)))
        else:
            reading = probe(session).report.reading
        night = NightBudget.open(session, reading, now=now)
        candidate = night.stage(Stage.IMPLEMENT, candidates=1).candidate(spec.idea_id)
        outcome = run_review(spec=spec, spec_text=spec_text, sources=texts, code=code,
                             budget=candidate, session=session, model=model)
        blocking = unanswered_blocking(spec, [f.__dict__ for f in outcome.accepted])
        spent = candidate.spent_tokens

    typer.echo(f"reviewed {spec.idea_id}: code {', '.join(code)}; sources {', '.join(texts)}"
               f"{' (blind)' if blind else ''}; {spent:,} tokens")
    typer.echo(f"accepted {len(outcome.accepted)}:")
    for f in outcome.accepted:
        typer.echo(f"  [{f.kind}] {f.summary}")
        if f.source_quote:
            typer.echo(f"      source: \"{f.source_quote[:160]}\"")
        if f.code_ref:
            typer.echo(f"      code:   {f.code_ref}")
    typer.echo(f"rejected {len(outcome.rejected)} (evidence did not check out):")
    for f, why in outcome.rejected:
        summary = f.summary if hasattr(f, "summary") else str(f)[:80]
        typer.echo(f"  {summary} -- {why}")
    if spec.unexpressed_mechanisms:
        typer.echo("declared in the spec as unexpressed:")
        for m in spec.unexpressed_mechanisms:
            typer.echo(f"  {m}")
    typer.echo(f"blocking until declared or answered: {len(blocking)}")


@app.command("calibrate-planted")
def calibrate_planted_cmd(
    reference: Path = typer.Option(  # noqa: B008
        ..., "--reference", help="Spec whose book shape the planted books copy"
    ),
    books: int = typer.Option(200, "--books", help="Noise books to plant an edge in"),
    alpha: list[float] = typer.Option(  # noqa: B008
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20], "--alpha",
        help="Planted edge, a year (repeat)",
    ),
    window: list[int] = typer.Option(  # noqa: B008
        [180, 365], "--window", help="Also judge on the last N days (repeat); the full "
        "active span is always included",
    ),
    capital: float = typer.Option(3000.0, "--capital", help="Capital deployable now (USD)"),
    rules_version: str | None = typer.Option(None, "--rules"),
) -> None:
    """How many strategies with a KNOWN edge does the ruleset admit, call too
    early, or reject (docs/TASKS.md T26)? Plants `alpha` a year into the
    reference shape's noise books and judges each like a candidate. A
    measurement of the ruleset: nothing is written to the registry."""
    import pandas as pd

    from qlab.calibration.planted import noise_books, planted_outcomes
    from qlab.harness.costs import CostModel
    from qlab.pipeline.evaluate import resolve_panel, resolve_strategy
    from qlab.venues.config import load_venue
    from qlab.venues.derive import derive_venue_metrics

    spec = load_spec(reference)
    ruleset = load(rules_version) if rules_version is not None else load_latest()
    with session_scope() as session:
        panel = resolve_panel(session, spec)
    weights = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    shared = derive_venue_metrics(
        load_venue(spec.data.source),
        snapshot_source=str(panel.meta.get("venue", spec.data.source)),
        venue_id=spec.data.source,
        simultaneous_legs=spec.simultaneous_legs,
    )
    shared["point_in_time_universe"] = 1.0 if panel.meta.get("universe_complete") else 0.0
    shared["accrual_applied"] = 1.0

    active = weights.abs().sum(axis=1) > 0
    if not bool(active.any()):
        typer.echo("error: the reference strategy never holds a position", err=True)
        raise typer.Exit(code=1)
    end = panel.prices.index[-1]
    starts = {"full": active.idxmax()}
    for days in sorted(window, reverse=True):
        starts[f"last {days}d"] = max(active.idxmax(), end - pd.Timedelta(days=days))

    built = noise_books(panel, weights, costs, books)
    typer.echo(f"reference: {spec.idea_id} ({reference}); rules {ruleset.version}; "
               f"noise books {len(built)} of {books} requested")
    results = planted_outcomes(
        panel=panel, reference=weights, books=built, ruleset=ruleset, shared_facts=shared,
        alphas=alpha, window_starts=starts, min_leg_notional=spec.min_leg_notional,
        deployable_capital_usd=capital,
    )
    for label, rows in results.items():
        typer.echo("")
        typer.echo(f"window {label} ({rows[0].window_days:.0f} days)")
        typer.echo(f"  {'alpha':>6} {'mean ret':>9} {'med Sharpe':>10} {'admitted':>9} "
                   f"{'early':>7} {'rejected':>9}  rejected by")
        for r in rows:
            typer.echo(
                f"  {r.alpha:>6.0%} {r.mean_return:>9.1%} {r.median_sharpe:>10.2f} "
                f"{r.admitted / r.n:>9.1%} {r.early / r.n:>7.1%} {r.rejected / r.n:>9.1%}  "
                + ", ".join(f"{k} {v}" for k, v in r.reject_rules.most_common())
            )


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
    reference: str | None = typer.Option(
        None,
        "--reference",
        help=(
            "Spec whose BOOK SHAPE the noise is matched to (default: specs/trend.yaml). "
            "The admission rate belongs to the (ruleset, book shape) pair, not to the "
            "ruleset -- see docs/TASKS.md T27 item 3"
        ),
    ),
) -> None:
    """Calibrate a ruleset against noise (docs/TASKS.md T16): how often does
    the ruleset admit a strategy known to have no edge, and what is the best
    `sharpe_net` noise reaches?

    The answer is NOT a property of the ruleset on its own. Measured on
    `2026-09-21.1`: noise matched to trend's ~150-leg book is admitted 1.0%
    of the time; noise matched to a 20-leg book on the same panel, 26.0%.
    Same rules, same window, a factor of 26. So `--reference` picks the book
    shape, and the report is written per (version, reference) pair.

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
    reference_spec = load_reference_spec(reference) if reference else load_reference_spec()
    # Noise has no fitted parameters: its whole window is a forward test
    # (docs/FIT_VS_FORWARD.md). Inheriting the reference's selection period
    # would route every passing noise book `needs-forward` and read as a 0%
    # false-admission rate.
    reference_spec = reference_spec.model_copy(
        update={
            "params_fixed_at": reference_spec.data.start,
            "params_fixed_evidence": "calibration noise has no fitted parameters",
        }
    )

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
                reference=reference_spec,
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
        reference_idea_id=reference_spec.idea_id,
        generated_on=date.today(),
        generator_breakdowns=generator_breakdowns,
        reference_spec_name=(Path(reference).name if reference else None),
        # The revision note is a correction to the TREND calibration and
        # quotes trend's own numbers; printing it under another book's
        # results would present one book's history as another's.
        include_revision_note=reference is None,
    )
    report_path = write_report(
        content,
        ruleset.version,
        # Keep the historical unsuffixed name for the default reference so
        # every document already citing docs/CALIBRATION_<version>.md keeps
        # resolving; any other book shape gets its own file.
        reference_idea_id=None if reference is None else reference_spec.idea_id,
    )

    typer.echo(f"rules:  {ruleset.version}")
    typer.echo(f"book shape (reference): {reference_spec.idea_id}")
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

    typer.echo("ideas by the route of q-lab's own latest run:")
    if stats.ideas_by_latest_route:
        for route, count in sorted(stats.ideas_by_latest_route.items()):
            typer.echo(f"  {route:<16}{count}")
    else:
        typer.echo("  (no routed runs yet)")

    typer.echo("")
    typer.echo(f"ideas by status (calibration noise set aside: {stats.calibration_ideas}):")
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

    total_ideas = sum(stats.ideas_by_status.values()) + stats.calibration_ideas
    total_verdicts = sum(stats.verdicts_by_outcome.values())
    typer.echo("")
    typer.echo(f"total: {total_ideas} ideas, {total_verdicts} verdicts")


__all__ = ["app"]
