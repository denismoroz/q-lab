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


@app.command("import-cards")
def import_cards_cmd(
    cards_dir: Path = typer.Option(  # noqa: B008 - idiomatic typer default
        Path("cards"), "--dir", help="Directory with cards/*.yaml (drafts/ is never read)"
    ),
) -> None:
    """Import candidate cards into the registry as ideas (T32); safe to repeat."""
    from qlab.registry.card_importer import CardImportError, import_cards

    try:
        with session_scope() as session:
            report = import_cards(session, cards_dir)
    except (CardImportError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{'ideas inserted':<28}{len(report.ideas_inserted)}")
    typer.echo(f"{'ideas already there':<28}{len(report.ideas_existing)}")
    typer.echo(f"{'drivers inserted':<28}{len(report.drivers_inserted)}")
    typer.echo(f"{'drivers attached':<28}{len(report.drivers_attached)}")
    typer.echo(f"{'retired by the card':<28}{len(report.retired)}")
    typer.echo(f"{'graveyard matches checked':<28}{report.matches_checked}")
    for idea_id, reason in report.retired:
        typer.echo(f"  retired {idea_id}: {reason}")


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
    source: str = typer.Option(..., "--source", help="hyperliquid, binance or aster"),
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
    source: str = typer.Option(..., "--source", help="hyperliquid, binance or aster"),
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


@data_app.command("asset-classes")
def data_asset_classes_cmd(
    fetch: bool = typer.Option(
        True, "--fetch/--no-fetch",
        help="Record today's venue annotations first (default); --no-fetch only rebuilds",
    ),
) -> None:
    """Record the xyz deployment's annotations (category and underlying of every instrument)
    and rebuild the point-in-time class table classification/hyperliquid-xyz.yaml by the
    versioned rule classification/hyperliquid-xyz.rule.yaml (docs/ASSET_CLASSES.md)."""
    from datetime import UTC, datetime

    import httpx

    from qlab.data import asset_classes as ac
    from qlab.data.sources import hyperliquid as hl

    rule = ac.load_rule()
    if fetch:
        with httpx.Client() as client:
            names = hl.fetch_meta(client, "xyz")
            annotations = ac.fetch_annotations(client, names)
        path = ac.save_annotation_snapshot(
            ac.DEFAULT_RECORD_DIR, datetime.now(UTC).date(), annotations
        )
        typer.echo(f"recorded {len(annotations)} annotations -> {path}")
    snapshots = ac.load_annotation_snapshots(ac.DEFAULT_RECORD_DIR)
    document = ac.build_table(rule, snapshots, ac.listing_facts(ac.DEFAULT_RECORD_DIR))
    ac.write_table(document)
    counts: dict[str, int] = {}
    for entry in document["instruments"].values():
        key = entry["history"][-1]["class_key"]
        counts[key] = counts.get(key, 0) + 1
    typer.echo(f"rule {rule.version}; {len(document['instruments'])} instruments")
    for key, count in sorted(counts.items()):
        typer.echo(f"  {key}: {count}")
    notes = sorted(
        (name, entry["history"][-1]["note"])
        for name, entry in document["instruments"].items()
        if entry["history"][-1].get("note")
    )
    for name, note in notes:
        typer.echo(f"  note {name}: {note}")


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

    from qlab import capital_text

    head = capital_text.headline(result.metrics)
    typer.echo(f"result ({capital_text.basis_ru(result.metrics)}):")
    typer.echo("  " + capital_text.numbers_ru(result.metrics, noise="present", ex_best=True))
    typer.echo("")

    typer.echo("metrics (every figure the run stored; `cap_*` and `noise_cap_*` are on the "
               "strategy's own money, the rest as the stand counts them, a share of the book's "
               "balance):")
    for name, value in sorted(result.metrics.items()):
        typer.echo(f"  {name:<24}{value}")
    typer.echo("")

    typer.echo("verdicts:")
    for row in result.rules_result.rows:
        passed = ("not defined" if row.not_defined else "unknown" if row.passed is None
                  else ("pass" if row.passed else "fail"))
        value_str = "?" if row.value is None else f"{row.value:g}"
        typer.echo(
            f"  [{passed:<11}] {row.stage.value:<12} {row.rule_id:<24} "
            f"{row.metric}={value_str} {row.comparator}{row.threshold:g}"
        )
    typer.echo("")

    from qlab.year_signs import has_year_view, read_signs, sign_line

    if has_year_view(result.metrics):
        typer.echo("year signs (recorded, they decide nothing):")
        for reading in read_signs(result.metrics, ruleset):
            typer.echo(f"  {sign_line(reading)}")
        typer.echo("")

    if head.on_capital and not any(r.metric.startswith("cap_") for r in ruleset.rules):
        # Rules before 2026-10-29.2 read the stand's share of the balance, so the signs above are
        # the stand's; the same signs on the strategy's own money (docs/CAPITAL_BASE.md).
        from qlab.year_view import capital_view

        typer.echo("the same year signs on the strategy's own money (these rules read the "
                   "stand's figures; the ones above are theirs):")
        capital_metrics_view = capital_view(result.metrics)
        if has_year_view(capital_metrics_view):
            for reading in read_signs(capital_metrics_view, ruleset):
                typer.echo(f"  {sign_line(reading)}")
        typer.echo("")

    from qlab.anomalies.cli import evaluate_lines

    for line in evaluate_lines(result.trial_id, result.routing.route):
        typer.echo(line)

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
    from qlab import capital_text

    with session_scope() as session:
        members = family_report(session, idea_id)
    if not members:
        typer.echo(f"{idea_id}: no routed runs")
        return
    typer.echo("return, Sharpe and noise rank are on the strategy's own money "
               "(a run that has none is shown as the stand counts it)")
    typer.echo(f"{'v':>3} {'bars':>4} {'entry/exit':>11} {'return':>8} {'sharpe':>7} "
               f"{'noise%':>7}  route")
    positive = beats_noise = with_numbers = 0
    for m in members:
        metrics = m.metrics or {}
        # On the strategy's own money when the run has it (docs/CAPITAL_BASE.md).
        h = capital_text.headline(metrics)
        ret, sharpe, pct = h.ann, h.sharpe, h.noise
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
    detectors: bool = typer.Option(True, "--detectors/--no-detectors",
                                   help="Rebuild night/detectors.yaml before the watch list"),
    research: bool | None = typer.Option(
        None, "--research/--no-research",
        help="The researcher on night/research.yaml (tokens; default: its cadence there, nightly)"),
    assess: bool = typer.Option(True, "--assess/--no-assess",
                                help="The assessor's second opinion on changed verdicts and on "
                                "paper/live ideas (tokens)"),
    generate: bool | None = typer.Option(
        None, "--generate/--no-generate",
        help="The idea generator on night/generate.yaml (tokens; default: its weekly day)"),
    decay: bool | None = typer.Option(
        None, "--decay/--no-decay",
        help="The decay detector: realised results against the stand's own history "
        "(default: with --paper)"),
    director: bool | None = typer.Option(
        None, "--director/--no-director",
        help="The research director (tokens; default: every cadence_days nights in "
        "night/director.yaml, not on a night with --no-research; --director forces it)"),
    waiting: bool = typer.Option(
        True, "--waiting/--no-waiting",
        help="Re-run the ideas waiting for data that are not on the watch list (no tokens)"),
    twins: bool | None = typer.Option(
        None, "--twins/--no-twins",
        help="Causal twins and the second stage for ideas waiting for forward data "
        "(no tokens; default: with --waiting)"),
    overlays: bool | None = typer.Option(
        None, "--overlays/--no-overlays",
        help="Every overlay and pair on the strategies that passed, each compared with its base "
        "(no tokens; default: with --twins; docs/OVERLAYS.md)"),
    restamp: bool = typer.Option(
        True, "--restamp/--no-restamp",
        help="Run again the ideas whose latest run was measured under an older stand or rules "
        "version (no tokens; docs/RESTAMP.md)"),
    anomalies: bool = typer.Option(
        True, "--anomalies/--no-anomalies",
        help="The anomaly detector on the passed runs (no tokens), then the investigator on the "
        "flagged ones (tokens, guarded; docs/ANOMALIES.md)"),
    programmer: bool = typer.Option(
        True, "--programmer/--no-programmer",
        help="The programmer agent on its queue (tokens; docs/PROGRAMMER.md)"),
    catalogue: bool = typer.Option(
        True, "--catalogue/--no-catalogue",
        help="The cataloguer agent: one paper into the catalogue (tokens; docs/CATALOGUER.md)"),
    cards: bool = typer.Option(
        True, "--cards/--no-cards",
        help="The carder agent: one inbox idea into a card for the implementer (tokens; "
        "docs/IDEAS.md)"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
    stop_at: str | None = typer.Option(
        None, "--stop-at",
        help="HH:MM on this machine's clock: nothing new starts after it, the rest goes to "
        "the next night (ops/night.sh passes it)"),
) -> None:
    """Re-evaluate the watch list on data extended to the last closed day,
    sweep the graveyard by regime on Sundays, reconcile paper, write the
    morning report. Spends no tokens; resumes where a stopped run left off."""
    from datetime import date as _date

    from qlab.night import run, stop_time

    path = run(_date.fromisoformat(day) if day else None, graveyard=graveyard, paper=paper,
               implement=implement, search=search, detectors=detectors, assess=assess,
               research=research, generate=generate, decay=decay,
               director=director, waiting=waiting, twins=twins, overlays=overlays,
               anomalies=anomalies, restamp=restamp,
               programmer=programmer,
               catalogue=catalogue, cards=cards,
               noise_trials=noise_trials, stop_at=stop_time(stop_at) if stop_at else None)
    typer.echo(f"report: {path}")


twins_app = typer.Typer(help="Causal twins and the second stage (docs/NIGHT.md).",
                        no_args_is_help=True)
app.add_typer(twins_app, name="twins")


@twins_app.command("run")
def twins_run_cmd(
    only: list[str] = typer.Option(  # noqa: B008
        [], "--only", help="Handle these ideas (repeat the option); without it: the night's "
        "own selection, capped"),
    max_new: int | None = typer.Option(
        None, "--max-new", help="At most this many new twins (default: the night's cap, "
        "none with --only)"),
    waiting: bool = typer.Option(
        True, "--waiting/--no-waiting",
        help="With --only: first re-run the ideas' own first stage and their second stage the "
        "way the waiting stage does (in the stage's own plan, whatever the watch list says)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would be done, do nothing"),
    allow_heavy: bool = typer.Option(
        False, "--allow-heavy", help="Run twins whose data takes hours to fetch (hourly candles "
        "of every Binance perpetual)"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
    capital: float = typer.Option(3000.0, "--capital"),
) -> None:
    """The night's twins stage by hand (src/qlab/twin_stage.py): the causal twin of each idea
    that waits for forward data, run on the stand, then its second stage. Trials are written to
    the registry like any run (deflation counts them). Code only: no agent, no tokens. The state
    of a hand run is kept apart in data/twins/runs/, never in a night's."""
    from datetime import date as _date
    from pathlib import Path as _Path

    from qlab import night, twin_stage
    from qlab import waiting as waiting_module

    night.STATE_DIR = _Path("data/twins/runs")  # a hand run never touches a night's state
    day, end = _date.today(), night.last_closed_day()
    state = night.load_state(day)
    state.pop("twins", None)  # this call reports itself; the day's done-markers stay
    state.pop("waiting", None)
    ideas = set(only) or None
    if waiting and ideas and not dry_run:
        waiting_module.waiting_stage(day, end, state, watch=[], noise_trials=noise_trials,
                                     deployable_capital_usd=capital, only=ideas,
                                     force=True)
        for line in waiting_module.report_lines(state.get("waiting") or [],
                                                state.get("waiting_notes")):
            typer.echo(line)
    twin_stage.twins_stage(day, end, state, noise_trials=noise_trials,
                           deployable_capital_usd=capital, only=ideas, max_new=max_new,
                           dry_run=dry_run, allow_heavy=allow_heavy)
    for line in twin_stage.report_lines(state.get("twins")):
        typer.echo(line)
    for name, text in (state.get("stage_errors") or {}).items():
        typer.echo(f"error in {name}: {text}", err=True)


restamp_app = typer.Typer(
    help="Run again the results measured an older way (docs/RESTAMP.md).", no_args_is_help=True)
app.add_typer(restamp_app, name="restamp")


@restamp_app.command("run")
def restamp_run_cmd(
    idea: list[str] = typer.Option(  # noqa: B008
        [], "--idea", help="Recompute these ideas, stale or not (repeat the option); a second "
        "stage stands for its first stage"),
    all_ideas: bool = typer.Option(
        False, "--all", help="Every stale idea, without the night's cap"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would be done, do nothing"),
    max_ideas: int | None = typer.Option(
        None, "--max-ideas", help="At most this many ideas (default: the night's cap, none with "
        "--all or --idea)"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
    capital: float = typer.Option(3000.0, "--capital"),
) -> None:
    """The night's restamp stage by hand (src/qlab/restamp.py): every idea whose latest run was
    made under an older stand version or judged under older rules is run again on the stand (and
    its second stage on Hyperliquid), the same spec on the same days; its forward series is
    rebuilt. Trials are written to the registry like any run (deflation counts them); old trials
    and verdicts stay. Code only: no agent, no tokens. Without --all or --idea the night's cap
    holds. The state of a hand run is kept apart in data/restamp/runs/, never in a night's."""
    from datetime import date as _date
    from pathlib import Path as _Path

    from qlab import night, restamp

    if not (idea or all_ideas or dry_run or max_ideas):
        typer.echo("nothing asked: name ideas with --idea, take every stale one with --all, "
                   "cap with --max-ideas, or look with --dry-run", err=True)
        raise typer.Exit(2)
    night.STATE_DIR = _Path("data/restamp/runs")  # a hand run never touches a night's state
    day, end = _date.today(), night.last_closed_day()
    state = night.load_state(day)
    state.pop("restamp", None)  # this call reports itself; the day's done-markers stay
    restamp.restamp_stage(day, end, state, noise_trials=noise_trials,
                          deployable_capital_usd=capital, only=set(idea) or None,
                          no_cap=all_ideas, dry_run=dry_run, max_ideas=max_ideas)
    for line in restamp.report_lines(state.get("restamp"), even_if_quiet=True) \
            or ["nothing is out of date"]:
        typer.echo(line)
    for name, text in (state.get("stage_errors") or {}).items():
        typer.echo(f"error in {name}: {text}", err=True)


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


@regimes_app.command("detect")
def regimes_detect_cmd(
    name: str = typer.Option("btc-market-logistic", help="Name of the stored predictions."),
    start: str = typer.Option("2020-01-01", help="First day to predict (months whose "
                              "training data lacks a regime are skipped)."),
    market_spec: str = typer.Option(
        "specs/trend-binance-cap.yaml",
        help="A spec whose panel is the whole Binance market (daily, discovered universe): "
             "breadth and mean funding come from it. Empty: BTC only."),
    returns_only: bool = typer.Option(
        False, help="BTC's returns over 7-90 days alone: the set chosen on 2020-04..2023-12 "
                    "and confirmed on 2024-01..2026-09 (scripts/research/detector_features.py)."),
    window: int = typer.Option(30, help="The regime window in days; returns-only features "
                                        "are scaled to it (14: returns over 3-42 days)."),
) -> None:
    """Learn the current regime from the past alone, walking forward month by
    month (docs/TASKS.md T39, docs/REGIME_DETECT.md), store the predictions
    and compare them, and Bv2's own hedge rule, with the testing labels."""
    from qlab import detector_builds as db
    from qlab import regime_detect as rd
    from qlab.regimes import REGIMES, load, market_closes

    regimes, btc = load(), market_closes()
    if regimes is None or btc is None:
        typer.echo("error: run `qlab regimes build` first", err=True)
        raise typer.Exit(code=1)
    try:
        built = db.level1(name, returns_only=returns_only, window=window, start=start,
                          market_spec=market_spec or None)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    predictions = rd.load_predictions(name)
    typer.echo(f"{built.path}: {len(predictions)} days, {built.first:%Y-%m-%d}.."
               f"{built.last:%Y-%m-%d}; {built.note}")
    acc = rd.accuracy(predictions["label"], regimes.labels)
    typer.echo(f"against the 30-day testing labels: accuracy {acc['accuracy']:.0%} of "
               f"{acc['days']:.0f} days (a third is chance)")
    for r in REGIMES:
        typer.echo(f"  {r:<5} recall {acc[f'{r}_recall']:.0%}, precision "
                   f"{acc[f'{r}_precision']:.0%}")
    typer.echo("hedge signals (share of each regime's days hedged; lag into a fall):")
    signals = {
        "Bv2's rule on BTC (14 and 30 days)": rd.bv2_rule(btc).reindex(predictions.index),
        "model: hedge unless bull": predictions["label"] != "bull",
        "model: hedge on bear": predictions["label"] == "bear",
    }
    for label, hedge in signals.items():
        sc = rd.score_hedge(hedge, regimes.labels)
        typer.echo(f"  {label:<36} hedged {sc.hedged_share:.0%}: bear {sc.bear_hedged:.0%}, "
                   f"flat {sc.flat_hedged:.0%}, bull {sc.bull_hedged:.0%}; median lag "
                   f"{sc.median_lag_days:.0f} days over {sc.stretches} falls")


@regimes_app.command("meta")
def regimes_meta_cmd(
    spec: str = typer.Option(..., help="The strategy's ungated daily spec."),
    level1: str = typer.Option("btc-returns-logistic", help="Stored level-1 predictions."),
    name: str = typer.Option(..., help="Name of the stored level-2 decisions."),
) -> None:
    """Level 2 for one strategy (docs/TASKS.md T39, docs/REGIME_DETECT.md):
    from level 1's market probabilities and the strategy's own recent state,
    learn walking forward whether it earns over the next 30 days."""
    from qlab import detector_builds as db

    b = db.meta(name, spec=spec, level1_name=level1)
    typer.echo(f"{b.path}: {b.first:%Y-%m-%d}..{b.last:%Y-%m-%d}; {b.note}")


@regimes_app.command("meta-hedge")
def regimes_meta_hedge_cmd(
    spec: str = typer.Option("specs/bv2-binance.yaml", help="The Bv2 spec (hourly panel)."),
    level1: str = typer.Option("btc-returns-logistic", help="Stored level-1 predictions."),
    name: str = typer.Option("bv2-hedge", help="Name of the stored level-2 decisions."),
    false_alarm_filter: bool = typer.Option(
        False, "--filter", help="Veto the book's own alarms only (meta-labelling proper) "
                                "instead of replacing its rule."),
) -> None:
    """Level 2 for Bv2 (docs/TASKS.md T39, docs/REGIME_DETECT.md): per coin and
    hour, would a hedge opened now pay over the next 14 days? Learned walking
    forward (scripts/research/hedge_horizons.py compares it with the book's rule)."""
    from qlab import detector_builds as db

    b = db.meta_hedge(name, spec=spec, level1_name=level1, false_alarm_filter=false_alarm_filter)
    typer.echo(f"{b.path}: {b.first:%Y-%m-%d}..{b.last:%Y-%m-%d}; {b.note}")


@regimes_app.command("refresh")
def regimes_refresh_cmd() -> None:
    """Rebuild everything in night/detectors.yaml, as the nightly run does:
    on data extended to the last closed day."""
    from qlab import detector_builds as db
    from qlab.night import last_closed_day

    for line in db.build_all(last_closed_day()):
        typer.echo(f"{'ok ' if line['ok'] else 'FAILED'} {line['name']}: {line['note']}")


@app.command("retire")
def retire_cmd(
    ideas: list[str] = typer.Argument(..., metavar="IDEA..."),  # noqa: B008 - typer's own idiom
    reason: str = typer.Option(..., "--reason", help="Why -- the owner's decision, as given."),
) -> None:
    """Withdraw ideas by the owner's decision: status `retired`, with the
    reason in the funnel ledger (`stage_transition`). For an idea that was
    never a candidate -- a control measurement -- or that is superseded;
    `rejected` stays what a run decides, `decayed` what a real shutdown does.
    A retired idea is production-owned: no later run moves it."""
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea, IdeaStatus

    if not reason.strip():
        typer.echo("error: a retirement needs a stated reason", err=True)
        raise typer.Exit(code=1)
    with session_scope() as session:
        missing = [i for i in ideas if session.get(Idea, i) is None]
        if missing:
            typer.echo(f"error: no such idea: {', '.join(missing)}", err=True)
            raise typer.Exit(code=1)
        for idea_id in ideas:
            was = session.get(Idea, idea_id).status
            if was == IdeaStatus.RETIRED:
                typer.echo(f"{idea_id}: already retired")
                continue
            repo.set_status(session, idea_id=idea_id, new_status=IdeaStatus.RETIRED,
                            reason=reason)
            typer.echo(f"{idea_id}: {was.value} -> retired")


@app.command("families")
def families_cmd() -> None:
    """Group variants under their parent strategy (qlab.registry.families):
    assign every idea its parent from the strategy its spec runs, and list
    the families."""
    from qlab.registry.db import session_scope
    from qlab.registry.families import assign_parents
    from qlab.registry.models import Idea

    with session_scope() as session:
        changed = assign_parents(session)
        for idea_id, parent in sorted(changed.items()):
            typer.echo(f"assigned {idea_id} -> {parent}")
        children: dict[str, list[Idea]] = {}
        for idea in session.query(Idea).filter(Idea.parent_id.is_not(None)):
            children.setdefault(idea.parent_id, []).append(idea)
        for parent_id in sorted(children):
            parent = session.get(Idea, parent_id)
            kids = sorted(children[parent_id], key=lambda i: i.id)
            by_status: dict[str, int] = {}
            for kid in kids:
                by_status[kid.status.value] = by_status.get(kid.status.value, 0) + 1
            typer.echo(f"{parent_id} ({parent.status.value}): {len(kids)} variants -- "
                       + ", ".join(f"{n} {s}" for s, n in sorted(by_status.items())))


idea_app = typer.Typer(add_completion=False, help="Ideas in the registry.")
app.add_typer(idea_app, name="idea")


@idea_app.command("set-parent")
def idea_set_parent(
    idea_id: str = typer.Argument(..., help="The variant."),
    parent_id: str = typer.Argument(..., help="The idea it is a variant of."),
) -> None:
    """Make IDEA_ID a variant of PARENT_ID (owner, 2026-10-25: a variant of a
    strategy takes no place of its own in the night's cap). Refuses unknown ids
    and a cycle."""
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea
    from qlab.registry.strategies import would_make_cycle

    with session_scope() as session:
        idea = session.get(Idea, idea_id)
        if idea is None or session.get(Idea, parent_id) is None:
            missing = idea_id if idea is None else parent_id
            typer.echo(f"unknown idea: {missing}", err=True)
            raise typer.Exit(code=1)
        if would_make_cycle(session, idea_id, parent_id):
            typer.echo(f"refused: {parent_id} is {idea_id} or already below it; "
                       "the parent chain would loop", err=True)
            raise typer.Exit(code=1)
        old = idea.parent_id
        idea.parent_id = parent_id
        typer.echo(f"{idea_id}: {old or '(none)'} -> {parent_id}")


@idea_app.command("add")
def idea_add(
    text: str = typer.Argument(..., help="The idea, in your own words: kept verbatim."),
    source: str = typer.Option("owner", "--source", help="owner, or agent:<name>"),
) -> None:
    """Put an idea in the inbox (ideas/inbox.yaml, docs/IDEAS.md). The carder agent turns the
    oldest new idea into a card for the implementer, one a night."""
    from datetime import UTC, datetime

    from qlab import ideas

    try:
        inbox = ideas.load()
        entry, added = ideas.add(inbox, text, source=source, day=datetime.now(UTC).date())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if added:
        ideas.save(inbox)
    typer.echo(f"{entry['id']}: added" if added else
               f"{entry['id']}: these words are already in the inbox ({entry['status']})")


@idea_app.command("list")
def idea_list(status: str | None = typer.Option(
        None, "--status", help="new | carded | duplicate | needs-owner | implemented | rejected")
) -> None:
    """The inbox: every idea, its status and its first words."""
    from qlab import ideas

    if status is not None and status not in ideas.STATUSES:
        typer.echo(f"status must be one of {ideas.STATUSES}", err=True)
        raise typer.Exit(code=1)
    rows = ideas.by_status(ideas.load(), status) if status else ideas.load()
    for line in ideas.report_lines(rows):
        typer.echo(line)
    typer.echo(f"{len(rows)} ideas")


def _inbox_entry(idea_id: str):
    from qlab import ideas

    inbox = ideas.load()
    entry = ideas.find(inbox, idea_id)
    if entry is None:
        typer.echo(f"unknown idea: {idea_id}", err=True)
        raise typer.Exit(code=1)
    return ideas, inbox, entry


@idea_app.command("reject")
def idea_reject(idea_id: str = typer.Argument(...),
                why: str = typer.Option(..., "--why")) -> None:
    """THE OWNER'S COMMAND: close an idea of the inbox (a card already written stays; take it
    out of night/implement_queue.yaml by hand if the implementer must not build it)."""
    ideas, inbox, entry = _inbox_entry(idea_id)
    ideas.reject(inbox, idea_id, why)
    ideas.save(inbox)
    typer.echo(f"{entry['id']}: rejected")


@idea_app.command("retry")
def idea_retry(idea_id: str = typer.Argument(...)) -> None:
    """THE OWNER'S COMMAND: put an idea back to new (after answering its question, or after two
    refused attempts); the carder takes it again with fresh attempts."""
    ideas, inbox, entry = _inbox_entry(idea_id)
    ideas.retry(inbox, idea_id)
    ideas.save(inbox)
    typer.echo(f"{entry['id']}: new again")


@idea_app.command("card")
def idea_card(idea_id: str | None = typer.Argument(
        None, help="An idea of the inbox; none: the one the night would take")) -> None:
    """Run the carder on one idea now (tokens, through the budget guard), as the night would."""
    from datetime import UTC, datetime

    from qlab import ideas
    from qlab.agents import carder
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate
    from qlab.registry.db import session_scope

    config = carder.load_config()
    inbox = ideas.load()
    if carder.sync_implemented(inbox):
        ideas.save(inbox)
    if waiting := carder.pending_cards(inbox):
        typer.echo(f"{waiting[0]['card']} still waits for the implementer: one card of the "
                   "carder at a time", err=True)
        raise typer.Exit(code=1)
    entry = (ideas.find(inbox, idea_id) if idea_id
             else ideas.oldest_new(inbox, max_attempts=int(config.get("max_attempts", 3))))
    if entry is None:
        typer.echo("no idea to card", err=True)
        raise typer.Exit(code=1)
    with session_scope() as session:
        result = carder.run_idea(entry, inbox, config,
                                 budget=open_candidate(session, None, Stage.SEARCH),
                                 session=session, day=datetime.now(UTC).date())
    ideas.save(inbox)
    typer.echo(f"{entry['id']}: {entry['status']} {result.get('card') or ''}; "
               f"{result['tokens']:,} tokens")
    for problem in result["problems"]:
        typer.echo(f"  {problem}")


@idea_app.command("recheck")
def idea_recheck(idea_id: str = typer.Argument(...)) -> None:
    """The draft already written for an idea through the code checks, the reviewer and the queue
    again, without calling the writer (tokens only for the reviewer): after a check that refused
    it was corrected."""
    from datetime import UTC, datetime

    from qlab import ideas
    from qlab.agents import carder
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate
    from qlab.registry.db import session_scope

    inbox, entry = ideas.load(), None
    entry = ideas.find(inbox, idea_id)
    if entry is None:
        typer.echo(f"unknown idea: {idea_id}", err=True)
        raise typer.Exit(code=1)
    with session_scope() as session:
        result = carder.recheck_idea(entry, carder.load_config(),
                                     budget=open_candidate(session, None, Stage.SEARCH),
                                     session=session, day=datetime.now(UTC).date())
    ideas.save(inbox)
    typer.echo(f"{entry['id']}: {entry['status']} {result.get('card') or ''}; "
               f"{result['tokens']:,} tokens")
    for problem in result["problems"]:
        typer.echo(f"  {problem}")


catalogue_app = typer.Typer(add_completion=False, no_args_is_help=True,
                            help="The cataloguer agent's papers (docs/CATALOGUER.md).")
app.add_typer(catalogue_app, name="catalogue")


def _catalogue_paper(paper_id: str):
    from qlab.agents import cataloguer as cat

    ledger = cat.load_ledger()
    paper = next((p for p in ledger if p.id == paper_id), None)
    if paper is None:
        typer.echo(f"unknown paper: {paper_id}", err=True)
        raise typer.Exit(code=1)
    return cat, ledger, paper


@catalogue_app.command("list")
def catalogue_list() -> None:
    """Every paper the cataloguer took and what became of it."""
    from qlab.agents import cataloguer as cat

    for p in cat.load_ledger():
        typer.echo(f"{p.id:28s} {p.status:10s} attempts {p.attempts}  {p.mechanism or ''}  "
                   f"{p.title[:60]}")
        for problem in p.problems + [f"review: {g}" for g in p.review]:
            typer.echo(f"    {problem.splitlines()[0][:160]}")


@catalogue_app.command("add")
def catalogue_add(
    link: str = typer.Argument(..., help="The paper's link (arXiv or a direct PDF)"),
    pdf: str | None = typer.Option(None, "--pdf", help="A local PDF of it, when the link is "
                                   "behind a login"),
    title: str = typer.Option("", "--title"),
) -> None:
    """Queue a paper of the owner's: the night reads it before the feed's candidates."""
    from datetime import UTC, datetime

    from qlab.agents import cataloguer as cat

    ledger = cat.load_ledger()
    paper = cat.add_owner_paper(ledger, link, pdf=pdf, title=title,
                                day=datetime.now(UTC).date())
    cat.save_ledger(ledger)
    typer.echo(f"queued {paper.id}" if paper else "this paper is already in the ledger")


@catalogue_app.command("run")
def catalogue_run(paper_id: str | None = typer.Argument(
        None, help="A paper of the ledger; none: the one the night would take")) -> None:
    """Read one paper now (tokens, through the budget guard), as the night would."""
    from datetime import UTC, datetime

    from qlab.agents import cataloguer as cat
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate
    from qlab.registry.db import session_scope

    config = cat.load_config()
    ledger = cat.load_ledger()
    with session_scope() as session:
        paper = (next((p for p in ledger if p.id == paper_id), None) if paper_id
                 else cat.next_paper(ledger, session,
                                     max_attempts=int(config.get("max_attempts", 3))))
        if paper is None:
            typer.echo("no paper to read", err=True)
            raise typer.Exit(code=1)
        cat.run_paper(paper, config, budget=open_candidate(session, None, Stage.SEARCH),
                      session=session, day=datetime.now(UTC).date())
    cat.save_ledger(ledger)
    typer.echo(f"{paper.id}: {paper.status} {paper.mechanism or paper.reason or ''}; "
               f"{paper.tokens:,} tokens")
    for problem in paper.problems + [f"review: {g}" for g in paper.review]:
        typer.echo(f"  {problem}")


@catalogue_app.command("retry")
def catalogue_retry(paper_id: str = typer.Argument(...)) -> None:
    """Put a blocked or set-aside paper back in the queue (the owner's decision)."""
    cat, ledger, paper = _catalogue_paper(paper_id)
    paper.status, paper.attempts = "queued", 0
    cat.save_ledger(ledger)
    typer.echo(f"{paper.id}: queued again")


@catalogue_app.command("reject")
def catalogue_reject(paper_id: str = typer.Argument(...),
                     reason: str = typer.Option(..., "--reason")) -> None:
    """THE OWNER'S COMMAND: undo a paper -- its catalogue commit is reverted by a new commit."""
    cat, ledger, paper = _catalogue_paper(paper_id)
    cat.reject(paper, reason)
    cat.save_ledger(ledger)
    typer.echo(f"{paper.id}: rejected" + (f", commit {paper.commit[:10]} reverted"
                                          if paper.commit else ""))


code_app = typer.Typer(add_completion=False, no_args_is_help=True,
                       help="The programmer agent's code requests (docs/PROGRAMMER.md).")
app.add_typer(code_app, name="code")


def _code_request(request_id: str):
    from qlab.agents import programmer as prog

    queue = prog.load_queue()
    req = next((r for r in queue if r.id == request_id), None)
    if req is None:
        typer.echo(f"unknown request: {request_id}", err=True)
        raise typer.Exit(code=1)
    return prog, queue, req


@code_app.command("list")
def code_list() -> None:
    """Every code request and where it stands."""
    from qlab.agents import programmer as prog

    for r in prog.load_queue():
        typer.echo(f"{r.id}  {r.repo:5s} {r.status:8s} attempts {r.attempts}  {r.what}"
                   + (f"  [{r.branch}]" if r.branch else ""))
        for problem in r.problems + [f"review: {g}" for g in r.review]:
            typer.echo(f"    {problem.splitlines()[0]}")


@code_app.command("request")
def code_request(
    repo: str = typer.Option(..., "--repo", help="frab or qlab"),
    what: str = typer.Option(..., "--what", help="What to make"),
    why: str = typer.Option(..., "--why", help="The direction it serves"),
    strategy: str | None = typer.Option(None, "--strategy"),
) -> None:
    """Queue a request of the owner's own (the night takes one a night)."""
    from datetime import UTC, datetime

    from qlab.agents import programmer as prog

    config = prog.load_config()
    if repo not in (config.get("repos") or {}):
        typer.echo(f"unknown repo {repo!r}: {sorted(config['repos'])}", err=True)
        raise typer.Exit(code=1)
    queue = prog.load_queue()
    req = prog.enqueue(queue, repo=repo, what=what, why=why, strategy=strategy, source="owner",
                       day=datetime.now(UTC).date())
    prog.save_queue(queue)
    typer.echo(f"queued {req.id}" if req else "the same request is already in the queue")


@code_app.command("run")
def code_run(request_id: str = typer.Argument(..., help="The request to attempt now")) -> None:
    """Attempt one request now (tokens, through the budget guard), as the night would."""
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate
    from qlab.registry.db import session_scope

    prog, queue, req = _code_request(request_id)
    config = prog.load_config()
    with session_scope() as session:
        prog.run_request(req, config, budget=open_candidate(session, None, Stage.IMPLEMENT),
                         session=session)
    prog.save_queue(queue)
    typer.echo(f"{req.id}: {req.status}; {req.tokens:,} tokens; branch {req.branch}")
    for problem in req.problems + [f"review: {g}" for g in req.review]:
        typer.echo(f"  {problem}")


@code_app.command("merge")
def code_merge(request_id: str = typer.Argument(...)) -> None:
    """THE OWNER'S COMMAND: merge a ready request's branch into its base and push."""
    prog, queue, req = _code_request(request_id)
    try:
        head = prog.merge(req, prog.load_config())
    except ValueError as exc:
        typer.echo(f"not merged: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    prog.save_queue(queue)
    typer.echo(f"{req.id}: merged, {req.repo} is at {head[:12]}")


@code_app.command("reject")
def code_reject(request_id: str = typer.Argument(...),
                reason: str = typer.Option(..., "--reason")) -> None:
    """THE OWNER'S COMMAND: drop a request, its branch and worktree."""
    prog, queue, req = _code_request(request_id)
    prog.reject(req, prog.load_config(), reason)
    prog.save_queue(queue)
    typer.echo(f"{req.id}: rejected")


detectors_app = typer.Typer(help="Regime detectors' own trials (docs/REGIME_DETECT.md).",
                            no_args_is_help=True)
app.add_typer(detectors_app, name="detectors")


@detectors_app.command("trials")
def detectors_trials_cmd(family: str = typer.Option("", help="Show this family's rows.")) -> None:
    """How many detector variants were tried per family, how many times the
    holdout was looked at, and which variant was chosen -- what a "best of
    many" detector must be read against."""
    from qlab import detector_trials as dt
    from qlab.registry.db import session_scope
    from qlab.registry.models import DetectorTrial

    with session_scope() as session:
        for s in dt.summary(session):
            typer.echo(f"{s.family}: {s.development_variants} variants on development, "
                       f"{s.holdout_looks} holdout looks; chosen: {', '.join(s.chosen) or '-'}")
        if family:
            for r in (session.query(DetectorTrial).filter_by(family=family)
                      .order_by(DetectorTrial.id)):
                mark = " *chosen*" if r.chosen else ""
                typer.echo(f"  [{r.period}] {r.variant}{mark}: " + ", ".join(
                    f"{k} {v:.3g}" for k, v in r.metrics.items()))


budget_app = typer.Typer(help="Token budget guard (docs/BUDGET.md).", no_args_is_help=True)
app.add_typer(budget_app, name="budget")


def _budget_account() -> str:
    """The account in use, as a line for `qlab budget status`: raises `WrongAccount` when the
    CLI is logged in to another one (or to none)."""
    from qlab.budget.agent import check_account

    found = check_account()
    return f"{found.email} (config directory {found.config_dir})"


@budget_app.command("status")
def budget_status_cmd() -> None:
    """Tonight's ceiling from the latest recorded reading of the shared
    windows -- no call is made (use `qlab budget probe` for a fresh one)."""
    from datetime import UTC, datetime, timedelta

    from qlab.budget.account import load_account
    from qlab.budget.guard import (
        WEEKLY_CEILING,
        NightBudget,
        WrongAccount,
        calibrate,
        last_reading,
        nights_until,
    )
    from qlab.budget.usage import UsageReading, Window
    from qlab.registry.models import TokenSpend

    # The ledger is read for the account the night will use (night/account.yaml); the account
    # the CLI is logged in to is checked first -- `claude auth status` costs no tokens.
    expected = load_account().email
    try:
        typer.echo(f"account          {_budget_account()}")
    except WrongAccount as refusal:
        typer.echo(f"account          WRONG -- {refusal}")
        typer.echo(f"                 (the figures below are the ledger of {expected})")
    with session_scope() as session:
        last = last_reading(session, expected)
        cal = calibrate(session, account=expected)
        if last is None:
            typer.echo(f"no reading of {expected} recorded yet: run `qlab budget probe`")
            return
        resets = last.seven_day_resets_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        reading = UsageReading("allowed", None, Window(last.seven_day_util, resets))
        night = NightBudget.open(session, reading, now=now)
        week_start = resets - timedelta(days=7)
        rows = (session.query(TokenSpend)
                .filter(TokenSpend.at >= week_start, TokenSpend.account == expected).all())
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
    if night.limits_off:
        typer.echo(f"guard limits     OFF -- {night.limits_off}")
    else:
        typer.echo(f"tokens tonight   about {night.tokens_left():,}")
    typer.echo("spent this week  " + (", ".join(f"{k} {v:,}" for k, v in by_stage.items())
                                     or "nothing"))


@budget_app.command("probe")
def budget_probe_cmd() -> None:
    """One minimal call to read the shared windows; recorded in token_spend."""
    from qlab.budget import WrongAccount, probe

    try:
        with session_scope() as session:
            result = probe(session)
    except WrongAccount as refusal:
        typer.echo(str(refusal), err=True)
        raise typer.Exit(code=1) from refusal
    r = result.report.reading
    if r is None or r.seven_day is None:
        typer.echo("the call reported no window reading", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"account {result.account}")
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
    typer.echo("(a model in which the whole capital sits in one leg at a time: each leg's figures "
               "are the stand's share of its own book, not the base of the strategy's own money "
               "of docs/CAPITAL_BASE.md; see its section 8)")
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
    focus: str | None = typer.Option(None, "--focus", help="The owner's question, verbatim"),
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
                           session=session, max_experiments=max_experiments, focus=focus)
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


@app.command("generate")
def generate_cmd(
    catalogue_path: Path = typer.Option(Path("docs/MECHANISMS.md"), "--catalogue",  # noqa: B008
                                        help="The catalogue of published mechanisms"),
    max_specs: int = typer.Option(3, "--max-specs"),
    data_end: str | None = typer.Option(
        None, "--data-end", help="Data end for the specs (default: the latest Binance snapshot)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only parse the catalogue and print its mechanisms (no tokens)"),
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
) -> None:
    """The idea generator writes up to --max-specs new specs under specs/generated/
    from the mechanisms of docs/MECHANISMS.md (no new code); code guards every
    number against the catalogue and the stand runs the specs that pass. A real
    run spends tokens through the budget guard, like `qlab research`."""
    from datetime import date as _date

    from qlab.agents.generator import (
        catalogue,
        generate,
        result_line,
        run_generated,
        second_stage_lines,
    )

    if dry_run:
        mechanisms = catalogue(catalogue_path)
        for m in mechanisms:
            expressed = "fully expressed" if m.fully_expressed else "partly expressed"
            typer.echo(f"{m.id}: {len(m.params)} parameter rows, {expressed}")
            typer.echo(f"    blocks: {m.blocks}")
            for row in m.params:
                typer.echo(f"    {row.param}: {', '.join(f'{v:g}' for v in row.values)}"
                           f"  <- {row.source}")
        typer.echo(f"{len(mechanisms)} mechanisms in {catalogue_path}")
        return

    from qlab.agents.generator import latest_binance_end
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate
    from qlab.registry.models import Idea

    with session_scope() as session:
        end = _date.fromisoformat(data_end) if data_end else latest_binance_end(session)
        if end is None:
            typer.echo("no Binance snapshot in the registry: pass --data-end", err=True)
            raise typer.Exit(1)
        budget = open_candidate(session, None, Stage.SEARCH)
        outcome = generate(budget=budget, session=session, catalogue_path=catalogue_path,
                           max_specs=max_specs, data_end=end,
                           registry_idea_ids=[i for (i,) in session.query(Idea.id)])
        spent = budget.spent_tokens
    typer.echo(f"generator: {len(outcome.accepted)} specs accepted, {len(outcome.rejected)} "
               f"rejected; {spent:,} tokens\nagent: {outcome.reply or '-'}")
    for problem in outcome.problems:
        typer.echo(f"  problem: {problem}")
    for r in outcome.rejected:
        for problem in r.problems:
            typer.echo(f"  rejected {r.path}: {problem}")
    if not evaluate:
        return
    for result in run_generated(outcome.accepted, noise_trials=noise_trials):
        typer.echo(result_line(result))
        for line in second_stage_lines(result):
            typer.echo(f"    {line}")


@app.command("stage2")
def stage2_cmd(
    spec_path: Path = typer.Argument(  # noqa: B008
        ..., metavar="SPEC", help="A stage 1 spec, e.g. specs/generated/gen-....yaml"),
    measure: bool = typer.Option(
        False, "--measure",
        help="Run the second stage although stage 1 did not survive: a measurement of the "
             "mechanism on Hyperliquid, the idea's status is not touched"),
    stage1_route: str | None = typer.Option(
        None, "--stage1-route",
        help="Stage 1's route (default: the idea's latest in the registry)"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
    capital: float = typer.Option(3000.0, "--capital"),
) -> None:
    """The second stage of a generated idea's evaluation (docs/TWO_STAGE.md, T45): the same
    mechanism with the same parameter values built on Hyperliquid and judged there, forward
    from the date the parameters were fixed. Code only: no agent, no tokens. Without --measure
    it is what the night does after a stage 1 survivor, and it moves the idea's status by both
    stages together."""
    from dataclasses import asdict

    from qlab.pipeline.stage2 import report_lines, run_stage2
    from qlab.registry.stages import latest_routed

    try:
        spec = load_spec(spec_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: invalid spec: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    trial_id = None
    with session_scope() as session:
        found = latest_routed(session, spec.idea_id)
        if found is not None:
            trial_id = found[0].id
            stage1_route = stage1_route or found[0].route.value
    if stage1_route is None:
        typer.echo("error: the idea has no stage 1 run in the registry: pass --stage1-route",
                   err=True)
        raise typer.Exit(code=1)
    result = run_stage2(spec, stage1_route=stage1_route, stage1_trial_id=trial_id,
                        noise_trials=noise_trials, deployable_capital_usd=capital,
                        measure=measure)
    for line in report_lines({"route": stage1_route}, asdict(result)):
        typer.echo(line)
    if result.status in ("evaluated", "not-evaluable"):
        typer.echo(f"stage 2 trial {result.trial_id}: {result.route} -- {result.reason}")
    if result.error:
        typer.echo(f"error: {result.error}", err=True)
        raise typer.Exit(code=1)


@app.command("twin")
def twin_cmd(
    spec_path: Path = typer.Argument(  # noqa: B008
        ..., metavar="SPEC", help="A spec whose numbers a person or an agent chose"),
    base_path: Path | None = typer.Option(  # noqa: B008
        None, "--base",
        help="The strategy spec an experiment SPEC modifies: the two become the grid"),
    write: bool = typer.Option(
        False, "--write",
        help="Write <name>-causal.yaml next to SPEC (a different existing file is not "
             "overwritten)"),
) -> None:
    """The causal twin of a spec (docs/RETUNE.md): the same idea with its numbers chosen by
    code every month from the past only, on Binance's history from 2019, so the stand can
    judge it the same night instead of waiting for new data. Prints the twin's YAML, or the
    plain reason why none can be made. Code only: no agent, no tokens, no database."""
    from qlab import twin_stage
    from qlab.twin import causal_twin, needs_listing, twin_yaml, write_twin

    try:
        spec = load_spec(spec_path)
        base = load_spec(base_path) if base_path is not None else None
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: invalid spec: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    config = twin_stage.load_config() or twin_stage.Config(True, 4, 20.0)
    names = None
    if needs_listing(spec, base, coin_params=config.coin_list_params):
        try:  # the names of the coins are the original venue's; Binance's listing maps them
            names = twin_stage._binance_listing()
        except Exception as exc:  # noqa: BLE001 - the network: say so and stop
            typer.echo(f"error: Binance listing not read: {type(exc).__name__}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    result = causal_twin(spec, base=base, spec_path=spec_path, base_path=base_path,
                         venue_names=names, coin_params=config.coin_list_params,
                         seconds_per_instrument=config.seconds_per_instrument)
    if result.spec is None:
        typer.echo(f"no twin for {spec.idea_id}: {result.reason}")
        if base is None:
            typer.echo("(an experiment on another spec needs --base SPEC: the spec it modifies)")
        return
    heavy = twin_stage.heavy_data(result.spec, config.slow_twin_minutes,
                                  config.max_hourly_instruments, config.seconds_per_instrument)
    if result.instruments:
        typer.echo(f"# instrument list: {len(result.instruments)} columns from "
                   f"{result.list_source}; hourly fetch about "
                   f"{(result.fetch_seconds or 0) / 60:.0f} min; "
                   f"{'HEAVY for the night: ' + heavy if heavy else 'within the night bound'}",
                   err=True)
    elif heavy:
        typer.echo(f"# HEAVY for the night: {heavy}", err=True)
    if write:
        try:
            typer.echo(str(write_twin(spec_path, result.spec, spec.idea_id)))
        except FileExistsError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        return
    typer.echo(twin_yaml(result.spec, spec.idea_id), nl=False)


@app.command("director")
def director_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Only print the evidence file the director would read (no tokens)"),
    approve: bool = typer.Option(
        False, "--approve", help="The owner has reviewed night/directions.yaml: clear "
        "`pending_owner_review` so the generator and the researcher obey it (no tokens)"),
    apply: bool = typer.Option(True, "--apply/--no-apply",
                               help="Validate the memo's block into night/directions.yaml"),
    config_path: Path = typer.Option(Path("night/director.yaml"), "--config"),  # noqa: B008
    research_config: Path = typer.Option(Path("night/research.yaml"), "--research-config"),  # noqa: B008
) -> None:
    """The research director watches the directions of the search and steers the idea
    generator and the researcher (docs/DIRECTOR.md). It writes a memo for the owner and a
    block that code validates into night/directions.yaml; the first directions ever wait
    for the owner's review. A real run spends tokens through the budget guard, like
    `qlab research`."""
    from datetime import UTC, datetime

    from qlab import directions as dirs
    from qlab.agents import director as director_module
    from qlab.agents.generator import CATALOGUE, catalogue
    from qlab.night import load_research_plan

    if approve:
        approved = dirs.approve(director_module.QLAB_ROOT)
        if approved is None:
            typer.echo("no readable night/directions.yaml to approve", err=True)
            raise typer.Exit(1)
        typer.echo(f"directions of {approved.date} approved: the generator and the researcher "
                   "obey them from the next run")
        return
    config = dirs.load_config(config_path)
    if config is None:
        typer.echo(f"no {config_path}", err=True)
        raise typer.Exit(1)
    plan = load_research_plan(research_config)
    root = director_module.QLAB_ROOT
    if dry_run:
        with session_scope() as session:
            typer.echo(director_module.build_evidence(session, root=root, config=config,
                                                      plan=plan))
        return

    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate

    with session_scope() as session:
        budget = open_candidate(session, None, Stage.IMPLEMENT)
        outcome = director_module.direct(budget=budget, session=session, config=config, plan=plan)
        spent = budget.spent_tokens
    typer.echo(f"director: memo {outcome.memo}; {spent:,} tokens\nagent: {outcome.reply or '-'}")
    for problem in outcome.problems:
        typer.echo(f"  problem: {problem}")
    if not apply or outcome.memo is None:
        return
    ids, strategies, families = director_module.known_names(
        plan, config, catalogue(root / CATALOGUE))
    result = director_module.apply_memo(outcome.memo, root=root, mechanism_ids=ids,
                                        strategy_names=strategies, family_names=families,
                                        day=datetime.now(UTC).date())
    if result.directions is None:
        typer.echo("directions not written: " + "; ".join(result.problems), err=True)
        raise typer.Exit(1)
    typer.echo(f"directions written to {result.written}"
               + (" -- waiting for the owner's review (`qlab director --approve`)"
                  if result.directions.pending_owner_review else ""))
    for problem in result.problems:
        typer.echo(f"  dropped: {problem}")


research_notes_app = typer.Typer(help="The researcher's memos in the registry.",
                                 no_args_is_help=True)
app.add_typer(research_notes_app, name="research-notes")


@research_notes_app.command("import")
def research_notes_import_cmd(
    memo_dir: Path = typer.Option(Path("docs/research"), "--memo-dir",  # noqa: B008
                                  help="Directory of <strategy>-<date>.md memos"),
    config: Path = typer.Option(Path("night/research.yaml"), "--config",  # noqa: B008
                                help="Maps each strategy name to its idea ids"),
    spec_dir: Path = typer.Option(Path("specs/research"), "--spec-dir",  # noqa: B008
                                  help="Where the researcher's experiment specs live"),
) -> None:
    """Import every memo file into the registry, one note per idea of the
    memo's strategy. Strategies missing from the config are skipped with a
    note. Idempotent: running it again changes nothing."""
    import yaml

    from qlab.registry.research_notes import import_memos

    plan = yaml.safe_load(config.read_text(encoding="utf-8"))
    strategies = {name: e["ideas"] for name, e in (plan.get("strategies") or {}).items()}
    with session_scope() as session:
        report = import_memos(session, strategies=strategies, memo_dir=memo_dir,
                              spec_dir=spec_dir)
    for line in report.imported:
        typer.echo(f"imported {line}")
    for line in report.skipped:
        typer.echo(f"skipped {line}")
    typer.echo(f"total: {len(report.imported)} memos imported, {len(report.skipped)} skipped")


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


def _interactive_night_budget(session):
    """Tonight's budget for an agent called by hand: the latest usage reading
    if it is fresh, a new probe otherwise (docs/BUDGET.md)."""
    from datetime import UTC, datetime, timedelta

    from qlab.budget import NightBudget, probe
    from qlab.budget.guard import last_reading
    from qlab.budget.usage import UsageReading, Window

    last = last_reading(session)  # the current account's rows only
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
    return NightBudget.open(session, reading, now=now)


@app.command("assess")
def assess_cmd(
    idea_id: str = typer.Argument(..., metavar="IDEA"),
    trial_id: int | None = typer.Option(None, "--trial", help="Default: the idea's latest run"),
    model: str = typer.Option("opus", "--model"),
) -> None:
    """Have the assessor agent give a second opinion on a run's verdict
    (docs/ASSESSOR.md). The verdict is never changed; claims whose numbers do
    not match the run are discarded; a dispute is the owner's to settle."""
    from sqlalchemy import select

    from qlab.agents.assessor import build_brief, latest_paper_lines, run_assessment
    from qlab.budget import Stage
    from qlab.registry.models import Spec, Trial

    with session_scope() as session:
        if trial_id is None:
            trial_id = session.execute(
                select(Trial.id).join(Spec, Trial.spec_id == Spec.id)
                .where(Spec.idea_id == idea_id, Trial.metrics.is_not(None))
                .order_by(Trial.id.desc()).limit(1)).scalar()
            if trial_id is None:
                raise typer.BadParameter(f"no run with metrics for {idea_id!r}")
        brief = build_brief(session, trial_id, paper=latest_paper_lines())
        night = _interactive_night_budget(session)
        candidate = night.stage(Stage.RECHECK, candidates=1).candidate(brief.idea_id)
        outcome = run_assessment(brief=brief, budget=candidate, session=session, model=model)
        spent = candidate.spent_tokens

    typer.echo(f"assessed {brief.idea_id}, trial {brief.trial_id}: {outcome.view}"
               f"{' (DISPUTES the verdict)' if outcome.dispute else ''}; {spent:,} tokens")
    typer.echo(outcome.summary)
    typer.echo(f"claims accepted {len(outcome.accepted)}:")
    for c in outcome.accepted:
        evidence = f"{c.fact} = {c.value:g}" if c.fact else f'"{c.quote[:120]}"'
        typer.echo(f"  - {c.text}  [{evidence}]")
    typer.echo(f"claims rejected {len(outcome.rejected)} (evidence did not check out):")
    for c, why in outcome.rejected:
        typer.echo(f"  - {getattr(c, 'text', str(c))[:100]} -- {why}")
    for check in outcome.checks:
        typer.echo(f"  would settle it: {check}")


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
    import yaml

    from qlab.agents.reviewer import code_files, run_review, unanswered_blocking
    from qlab.budget import Stage
    from qlab.pipeline.sources import QLAB_ROOT

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
        night = _interactive_night_budget(session)
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


@app.command("hedge-timing")
def hedge_timing_cmd(
    spec_path: Path = typer.Argument(..., metavar="SPEC"),  # noqa: B008
    start: str = typer.Option(..., "--from", help="Judge from this date (YYYY-MM-DD)"),
    data_start: str | None = typer.Option(
        None, "--data-start", help="Start the book here instead of the spec's data start "
        "(a fresh book: its positions are sized at creation and do not compound)"),
    books: int = typer.Option(200, "--books"),
) -> None:
    """Does the book choose WHEN to hedge better than chance? Its own spot,
    its own hedges shifted in time (docs/HEDGE_TIMING.md). A measurement:
    nothing is written to the registry and nothing is decided."""
    from datetime import date as _date

    import numpy as np
    import pandas as pd

    from qlab.harness.costs import CostModel
    from qlab.hedge_timing import hedge_timing_test
    from qlab.night import last_closed_day, with_end
    from qlab.pipeline.evaluate import resolve_panel, resolve_strategy

    spec = with_end(load_spec(spec_path), last_closed_day())
    if data_start:
        spec = spec.model_copy(update={"data": spec.data.model_copy(
            update={"start": _date.fromisoformat(data_start)})})
    with session_scope() as session:
        panel = resolve_panel(session, spec)
    weights = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    r = hedge_timing_test(panel, weights, costs, pd.Timestamp(start, tz="UTC"), n=books)
    shifted = np.array(r.shifted)
    typer.echo(f"{spec.idea_id} from {start} ({r.days:.0f} days), hedged {r.hedged_share:.0%} "
               f"of the time")
    typer.echo(f"  the book:            {r.own:+.1%} a year")
    typer.echo(f"  hedges shifted:      median {np.median(shifted):+.1%}, 5%..95% "
               f"{np.quantile(shifted, 0.05):+.1%}..{np.quantile(shifted, 0.95):+.1%}; "
               f"the book beats {r.percentile:.0%} of {len(shifted)}")
    typer.echo(f"  spot only:           {r.spot_only:+.1%}")
    typer.echo(f"  spot always hedged:  {r.always_hedged:+.1%}")


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
    from qlab.pipeline.evaluate import proxy_execution_venue, resolve_panel, resolve_strategy
    from qlab.venues.config import load_venue
    from qlab.venues.derive import derive_proxy_venue_metrics, derive_venue_metrics

    spec = load_spec(reference)
    ruleset = load(rules_version) if rules_version is not None else load_latest()
    with session_scope() as session:
        panel = resolve_panel(session, spec)
    weights = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    execution = proxy_execution_venue(spec)
    if execution is not None:
        shared = derive_proxy_venue_metrics(
            load_venue(spec.data.source),
            load_venue(execution),
            snapshot_source=str(panel.meta.get("venue", spec.data.source)),
            data_venue_id=spec.data.source,
            simultaneous_legs=spec.simultaneous_legs,
        )
    else:
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

    built = noise_books(panel, weights, costs, books, held=spec.noise_universe == "held")
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
            "params_fit_from": None,
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



@app.command("pool-allocation")
def pool_allocation_cmd(
    paper_root: Path = typer.Option(  # noqa: B008
        Path("data/paper"), "--paper-root", help="The night's exports of the production host"),
    night_dir: Path = typer.Option(  # noqa: B008
        Path("data/night"), "--night-dir", help="Saved nights (decay results, replay lines)"),
    config: Path = typer.Option(  # noqa: B008
        Path("night/allocation.yaml"), "--config", help="The allocation config"),
) -> None:
    """Print the capital-split PROPOSAL (qlab.pool_allocation, night/allocation.yaml): how the
    deployable capital could be shared between the strategies. Code proposes, the owner
    decides: nothing is moved, changed or written."""
    from datetime import date as _date

    from qlab import pool_allocation as pool

    today = _date.today()
    with session_scope() as session:
        results, rules = pool.night_signals(session, None, night_dir=night_dir, root=paper_root)
        out = pool.run_allocation(session, today, today, config=pool.load_config(config),
                                  root=paper_root, decay_results=results, paper_rules=rules,
                                  night_dir=night_dir)
    typer.echo(f"## {out.get('title') or pool.SECTION_TITLE}\n")
    typer.echo("\n".join(out["lines"]))


forward_series_app = typer.Typer(
    help="Daily chart data of a passed idea's forward test (docs/FORWARD_CHART.md).",
    no_args_is_help=True)
app.add_typer(forward_series_app, name="forward-series")


@forward_series_app.command("rebuild")
def forward_series_rebuild_cmd(
    idea: list[str] = typer.Option(  # noqa: B008
        None, "--idea", help="An idea id (repeat for several); default: every validated idea"),
    force: bool = typer.Option(False, "--force", help="Rebuild series that already exist"),
) -> None:
    """Build the missing daily series of the passed ideas' stages by re-running their stored
    specs on the stored data snapshots (CPU only: no network, no agent, no trial, no verdict).
    The console and the morning report draw the forward test's chart from these files."""
    from qlab import forward_series

    with session_scope() as session:
        done = forward_series.rebuild(session, list(idea) if idea else None, force=force)
    for idea_id, trial_id, outcome in done:
        typer.echo(f"{idea_id:<44} trial {trial_id:<7} {outcome}")
    typer.echo(f"total: {len(done)} runs looked at, series in {forward_series.store_dir()}")


overlays_app = typer.Typer(
    help="Overlays: when and how much a strategy holds (docs/OVERLAYS.md).",
    no_args_is_help=True)
app.add_typer(overlays_app, name="overlays")


@overlays_app.command("run")
def overlays_run_cmd(
    only: list[str] = typer.Option(  # noqa: B008
        [], "--only", help="Take these bases (repeat the option); without it: the night's own "
        "selection, capped"),
    chain: list[str] = typer.Option(  # noqa: B008
        [], "--chain", help="Only the variants made of these overlays, e.g. own_results or "
        "own_results+gradual_entry (repeat the option)"),
    max_variants: int | None = typer.Option(
        None, "--max-variants", help="At most this many new variants (default: the night's cap, "
        "none with --only)"),
    data_end: str = typer.Option(
        "base", "--data-end", help="'base': each base's last run, so no snapshot is fetched "
        "(default); 'night': the last closed day"),
    fixed: bool = typer.Option(
        False, "--fixed", help="Variants with FIXED numbers instead of the monthly causal choice: "
        "one idea per point of the family of each chain named with --chain (every point is a "
        "trial; read the family as a whole, never its best point)"),
    stage2: bool = typer.Option(
        False, "--stage2/--no-stage2", help="Also send a variant that survives stage 1 to the "
        "second stage on Hyperliquid (fetches that venue's candles unless a snapshot exists)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would be done, do nothing"),
    noise_trials: int = typer.Option(200, "--noise-trials"),
    capital: float = typer.Option(3000.0, "--capital"),
) -> None:
    """The night's overlays stage by hand (src/qlab/overlay_stage.py): every overlay and pair on
    the named bases, run on the stand, compared with the base. Trials are written to the registry
    like any run (deflation counts them). Code only: no agent, no tokens. The state of a hand run
    is kept apart in data/overlays/runs/, never in a night's."""
    from datetime import date as _date
    from pathlib import Path as _Path

    from qlab import night, overlay_stage

    night.STATE_DIR = _Path("data/overlays/runs")  # a hand run never touches a night's state
    day, end = _date.today(), night.last_closed_day()
    state = night.load_state(day)
    state.pop("overlays", None)  # this call reports itself; the day's done-markers stay
    overlay_stage.overlays_stage(
        day, end, state, noise_trials=noise_trials, deployable_capital_usd=capital,
        only=set(only) or None, chains_only=set(chain) or None, max_variants=max_variants,
        data_end=data_end, second_stage=stage2, dry_run=dry_run,
        kinds=("fixed",) if fixed else ("causal",))
    for line in overlay_stage.report_lines(state.get("overlays")):
        typer.echo(line)
    for name, text in (state.get("stage_errors") or {}).items():
        typer.echo(f"error in {name}: {text}", err=True)


@overlays_app.command("table")
def overlays_table_cmd(
    idea: str = typer.Argument(..., help="The base idea"),  # noqa: B008
) -> None:
    """A base's overlay variants against it, from the stored comparisons (no run)."""
    from qlab import overlay_compare, overlay_stage

    with session_scope() as session:
        table = overlay_compare.table(session, idea)
    lines = overlay_stage.table_lines(table)
    typer.echo(f"{idea}: {table['family_size']} variants compared, "
               f"{table['family_trials']} trials in the family; {table['counts']}")
    for line in lines:
        typer.echo(line)


@overlays_app.command("diagnostics")
def overlays_diagnostics_cmd(
    idea: list[str] = typer.Argument(  # noqa: B008
        ..., help="Ideas whose forward series to describe"),
    data_end: str = typer.Option("base", "--data-end", help="'base': the idea's last run; "
                                 "'night': the last closed day"),
    save: bool = typer.Option(True, "--save/--no-save", help="Store the result for the console"),
) -> None:
    """The three readings of an idea's forward series (docs/OVERLAYS.md): from every possible
    entry day, does a good period predict the next, and by market state known at the time. They
    describe and decide nothing. Reads the spec file and the snapshot on disk; no network."""
    from qlab import night, overlay_diagnostics, overlay_stage
    from qlab.waiting import spec_files

    files = spec_files(Path("specs"))
    for idea_id in idea:
        paths = files.get(idea_id) or []
        if len(paths) != 1:
            typer.echo(f"{idea_id}: {'no spec file' if not paths else 'several spec files'}")
            continue
        from qlab.pipeline.spec import load_spec

        spec = load_spec(paths[0])
        with session_scope() as session:
            end = (overlay_stage.last_run_end(session, idea_id)
                   if data_end == "base" else None) or night.last_closed_day()
            payload = overlay_diagnostics.compute_for_spec(night.with_end(spec, end), session)
        if save:
            overlay_diagnostics.save(payload)
        typer.echo(f"## {idea_id}")
        for line in overlay_diagnostics.lines_ru(payload):
            typer.echo(line)


@app.command("year-signs")
def year_signs_cmd(
    all_trials: bool = typer.Option(
        False, "--all-trials",
        help="Count every judged trial, not only each idea's latest (every trial counts, as "
             "in deflation)"),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of tables"),
    rules_version: str | None = typer.Option(
        None, "--rules", help="Rules version whose lines the signs use (default: latest)"),
    stand: bool = typer.Option(
        False, "--stand",
        help="Read the signs off the stand's year summaries (a share of the book's balance) "
             "instead of the default, the strategy's own money (docs/CAPITAL_BASE.md; the rules "
             "read it since 2026-10-29.2; runs from before those summaries are left out)"),
) -> None:
    """The four year signs (docs/YEARS.md) over the registry: for each idea's latest judged
    run, whether each of А Б В Г holds, fails or is not defined, then the totals and a split
    by outcome and by stage. They decide nothing (rules 2026-10-28.2); the statistics are
    for choosing, later, which sign should. Reads stored metrics only."""
    import json

    from qlab import year_signs

    ruleset = load(rules_version) if rules_version is not None else load_latest()
    capital = not stand
    with session_scope() as session:
        rows = year_signs.collect(session, ruleset, all_trials=all_trials, on_capital=capital)
    summary = year_signs.summarize(rows)
    if as_json:
        typer.echo(json.dumps({
            "rules_version": ruleset.version, "all_trials": all_trials, "on_capital": capital,
            "signs": {s.letter: {"id": s.rule_id, "title": s.title}
                      for s in year_signs.SIGNS},
            "rows": [r.to_json() for r in rows], "summary": summary},
            ensure_ascii=False, indent=2))
        return
    typer.echo(f"rules: {ruleset.version}; "
               + ("every judged trial" if all_trials else "latest judged trial of each idea")
               + ("; on the strategy's own money" if capital else "; as the stand counts it "
                                                              "(a share of the book's balance)"))
    if not rows:
        typer.echo("no judged trial with a year view in the registry")
        return
    for line in year_signs.render_table(rows):
        typer.echo(line)
    typer.echo("")
    for spec in year_signs.SIGNS:
        typer.echo(f"{spec.letter}: {spec.title}")
    typer.echo("")
    for line in year_signs.render_summary(summary):
        typer.echo(line)


@app.command("protection")
def protection_cmd(
    compute: bool = typer.Option(
        False, "--compute",
        help="Measure again, from the stored spec and snapshot, the runs that carry no "
             "measurement (CPU only, a minute or more each) and store the result in "
             "data/protection/; without it a run with no measurement shows no numbers"),
    all_ideas: bool = typer.Option(
        False, "--all-ideas", help="Include retired, rejected and decayed ideas"),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table"),
    rules_version: str | None = typer.Option(
        None, "--rules", help="Rules version whose lines the readings use (default: latest)"),
) -> None:
    """How well Bv2 protects (docs/PROTECTION.md): for each idea of the Bv2 family, its latest
    routed run's four readings П1 - П4 -- worst fall against its own long side held unhedged,
    loss in the weeks the coins fell, share of a rise it keeps, no loss over the whole part.
    The readings decide nothing (rules 2026-10-29.1); the owner will later choose whether a
    protection sign becomes a rule. Writes nothing to the registry."""
    import json

    from qlab import protection

    ruleset = load(rules_version) if rules_version is not None else load_latest()
    with session_scope() as session:
        rows = protection.collect(session, ruleset, compute=compute, live_only=not all_ideas)
    if as_json:
        typer.echo(json.dumps({
            "rules_version": ruleset.version,
            "readings": {r.letter: {"id": r.rule_id, "title": r.title}
                         for r in protection.READINGS},
            "rows": [r.to_json() for r in rows], "summary": protection.summarize(rows)},
            ensure_ascii=False, indent=2))
        return
    typer.echo(f"rules: {ruleset.version}; latest routed run of each "
               + ("Bv2-family idea" if all_ideas else "live Bv2-family idea"))
    if not rows:
        typer.echo("no run of the Bv2 family in the registry")
        return
    for line in protection.render_table(rows):
        typer.echo(line)
    typer.echo("")
    for spec in protection.READINGS:
        typer.echo(f"{spec.letter}: {spec.title}")
    typer.echo("«да / нет» и число; «—» — не определено; «(нет замера)» — у прогона замера нет, "
               "`--compute` его сделает")


from qlab.anomalies.cli import anomalies_app, anomaly_app  # noqa: E402 - registered last

app.add_typer(anomalies_app, name="anomalies")
app.add_typer(anomaly_app, name="anomaly")

__all__ = ["app"]
