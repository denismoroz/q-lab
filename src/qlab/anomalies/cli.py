"""`qlab anomalies` (the detector's flags, the dossiers) and `qlab anomaly` (the investigator's
read-only toolkit). See docs/ANOMALIES.md.

`qlab anomaly ...` is the ONLY command prefix the investigator agent may run
(`Bash(uv run qlab anomaly:*)`). Everything under it is read-only: it opens a registry session
that is rolled back, never committed, and writes at most the investigation's log of calls
(`QLAB_ANOMALY_LOG`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from sqlalchemy.orm import Session

from qlab.anomalies import toolkit
from qlab.registry.db import get_sessionmaker, session_scope

anomalies_app = typer.Typer(
    add_completion=False, invoke_without_command=True,
    help="The anomaly detector's flags and the investigator's dossiers (docs/ANOMALIES.md).")
anomaly_app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="The investigator's read-only toolkit (docs/ANOMALIES.md): never writes the registry.")


@contextmanager
def read_session() -> Iterator[Session]:
    """A registry session that is rolled back, never committed: the toolkit cannot write."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _emit(command: str, output: toolkit.Output, as_json: bool) -> None:
    """Print a toolkit answer; when an investigation logs its calls, number this one first so
    the dossier can cite it (`call N`)."""
    import os

    path = os.environ.get(toolkit.LOG_ENV)
    if path:
        n = sum(1 for _ in open(path, encoding="utf-8")) + 1 if Path(path).exists() else 1
        header = f"[call {n}] {command}"
        output = toolkit.Output([header, *output.lines], output.data)
        toolkit.log_call(command, output, path)
    typer.echo(json.dumps(output.data, ensure_ascii=False, indent=1, default=str)
               if as_json else output.text)


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


@anomaly_app.command("contributions")
def contributions_cmd(
    trial: int = typer.Argument(..., help="Trial id"),
    days: str = typer.Option(
        "", "--days", help="Comma-separated UTC days, e.g. 2025-11-02,2026-01-30"),
    start: str = typer.Option("", "--from", help="First day"),
    end: str = typer.Option("", "--to", help="Last day"),
    instrument: str = typer.Option("", "--instrument", help="One coin only"),
    top: int = typer.Option(20, "--top", help="Rows, largest absolute contribution first"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Per day and coin: weight, side, price move, funding, cost, net contribution and the day's
    dollar volume, for the largest rows of the selection."""
    from qlab.anomalies.rerun import RunUnavailable

    command = (f"qlab anomaly contributions {trial}" + (f" --days {days}" if days else "")
               + (f" --from {start}" if start else "") + (f" --to {end}" if end else "")
               + (f" --instrument {instrument}" if instrument else "") + f" --top {top}")
    try:
        with read_session() as session:
            out = toolkit.contributions(
                session, trial, days=toolkit.parse_days(days), start=start or None,
                end=end or None, instrument=instrument or None, top=top)
    except RunUnavailable as exc:
        _fail(str(exc))
    _emit(command, out, as_json)


@anomaly_app.command("coin")
def coin_cmd(
    instrument: str = typer.Argument(..., help="Column name as the trial's data has it"),
    trial: int = typer.Option(..., "--trial", help="Trial whose data to read"),
    start: str = typer.Option("", "--from"), end: str = typer.Option("", "--to"),
    rows: int = typer.Option(30, "--rows"), as_json: bool = typer.Option(False, "--json"),
) -> None:
    """One coin: listing and last-price facts, volume, whether Hyperliquid lists it, and its
    daily price, volume, trading status and the weight the book held."""
    from qlab.anomalies.rerun import RunUnavailable

    command = (f"qlab anomaly coin {instrument} --trial {trial}"
               + (f" --from {start}" if start else "") + (f" --to {end}" if end else ""))
    try:
        with read_session() as session:
            out = toolkit.coin(session, trial, instrument, start=start or None, end=end or None,
                               rows=rows)
    except RunUnavailable as exc:
        _fail(str(exc))
    _emit(command, out, as_json)


@anomaly_app.command("counterfactual")
def counterfactual_cmd(
    trial: int = typer.Argument(..., help="Trial id"),
    exclude_volume_below: float = typer.Option(
        None, "--exclude-volume-below",
        help="Coin-days under this many dollars of volume may not be held"),
    exclude_days_before_delist: float = typer.Option(
        None, "--exclude-days-before-delist",
        help="No position in the last N days before a coin's last price"),
    cap_instrument_share: float = typer.Option(
        None, "--cap-instrument-share",
        help="Cap each instrument at this share of the book's gross exposure"),
    cap_leg_contribution: float = typer.Option(
        None, "--cap-leg-contribution",
        help="Clip one instrument's earning on one bar to this share of the capital "
             "(a stress, e.g. 0.02)"),
    only_hl_listed: bool = typer.Option(
        False, "--only-hl-listed", help="Keep only coin-days Hyperliquid lists"),
    reselect: bool = typer.Option(
        False, "--reselect",
        help="For a ban rule: run the strategy again with the banned coin-days untradeable, so "
             "it chooses other coins, instead of leaving the banned positions as cash"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Re-run the trial with exactly ONE declared rule and show the result beside the original:
    yearly return, total, worst fall, the year table."""
    from qlab.anomalies.rerun import Rule, RunUnavailable, load_run

    chosen = [exclude_volume_below is not None, exclude_days_before_delist is not None,
              cap_instrument_share is not None, cap_leg_contribution is not None,
              only_hl_listed]
    if sum(chosen) != 1:
        _fail("give exactly one rule: --exclude-volume-below, --exclude-days-before-delist, "
              "--cap-instrument-share, --cap-leg-contribution or --only-hl-listed")
    if reselect and (cap_instrument_share is not None or cap_leg_contribution is not None):
        _fail("--reselect applies to a ban on coin-days, not to a cap")
    base = f"qlab anomaly counterfactual {trial}"
    try:
        with read_session() as session:
            run = load_run(session, trial)
            if only_hl_listed:
                mask = toolkit.listed_mask(session, run)
                if mask is None:
                    _fail("the registry holds no Hyperliquid snapshot to read listings from")
                rule, command = Rule(only_listed=mask), f"{base} --only-hl-listed"
            elif exclude_volume_below is not None:
                rule = Rule(min_volume_usd=exclude_volume_below)
                command = f"{base} --exclude-volume-below {exclude_volume_below:g}"
            elif exclude_days_before_delist is not None:
                rule = Rule(delist_days=exclude_days_before_delist)
                command = f"{base} --exclude-days-before-delist {exclude_days_before_delist:g}"
            elif cap_leg_contribution is not None:
                rule = Rule(max_leg_contribution=cap_leg_contribution)
                command = f"{base} --cap-leg-contribution {cap_leg_contribution:g}"
            else:
                rule = Rule(max_instrument_share=cap_instrument_share)
                command = f"{base} --cap-instrument-share {cap_instrument_share:g}"
            command += " --reselect" if reselect else ""
            out = toolkit.counterfactual(session, trial, rule, run=run, reselect_book=reselect)
    except RunUnavailable as exc:
        _fail(str(exc))
    _emit(command, out, as_json)


@anomaly_app.command("battery")
def battery_cmd(
    trial: int = typer.Argument(..., help="Trial id"),
    reselect: bool = typer.Option(
        False, "--reselect", help="Run the strategy again where a coin-day is banned (slower)"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """The standard robustness battery on one trial in one table: volume floors of 1 and 5
    million dollars, no position near a delisting, one coin's earning clipped to 2% and 1% of
    capital, and only the other venue's listed coins."""
    from qlab.anomalies.rerun import RunUnavailable

    command = f"qlab anomaly battery {trial}" + (" --reselect" if reselect else "")
    try:
        with read_session() as session:
            out = toolkit.battery(session, trial, reselect_book=reselect)
    except RunUnavailable as exc:
        _fail(str(exc))
    _emit(command, out, as_json)


@anomaly_app.command("compare-venues")
def compare_venues_cmd(
    idea: str = typer.Argument(..., help="Idea id (either stage)"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """An idea's two stages over their common days: totals, correlation, months side by side,
    the best-earning coins of each and whether the other stage's data has them."""
    with read_session() as session:
        out = toolkit.compare_venues(session, idea)
    _emit(f"qlab anomaly compare-venues {idea}", out, as_json)


@anomaly_app.command("flags")
def flags_cmd(trial: int = typer.Argument(..., help="Trial id"),
              as_json: bool = typer.Option(False, "--json")) -> None:
    """The detector's stored flags for the trial, with their numbers and lines."""
    with read_session() as session:
        out = toolkit.stored_flags(session, trial)
    _emit(f"qlab anomaly flags {trial}", out, as_json)


def evaluate_lines(trial_id: int, route: str) -> list[str]:
    """What `qlab evaluate` adds under the verdict for a run that passed (its route is one of
    `scan.routes` of night/anomalies.yaml): the anomaly detector's flags, which the run also
    stores (`anomaly_scan`). They decide nothing. Empty for any other run and when the lines
    file is missing; a scan that fails says so in one line and never fails the evaluation."""
    from qlab.anomalies import report, store
    from qlab.anomalies.config import load_config
    from qlab.anomalies.scan import scan_trial

    try:
        cfg = load_config()
        if cfg is None or not cfg.enabled or route not in cfg.scan.routes:
            return []
        with session_scope() as session:
            result = scan_trial(session, trial_id, cfg)
            store.save_scan(session, result, cfg.version)
        lines = ["anomaly flags (recorded, they decide nothing; docs/ANOMALIES.md):"]
        lines += [f"  {report.flag_sentence(f)}" for f in result.flags]
        return [*lines, ""]
    except Exception as exc:  # noqa: BLE001 - a view must not fail the evaluation
        return [f"anomaly flags: not computed ({type(exc).__name__}: {exc})", ""]


# --------------------------------------------------------------------------
# qlab anomalies


@anomalies_app.callback()
def anomalies_list_cmd(
    ctx: typer.Context,
    all_scans: bool = typer.Option(
        False, "--all-trials", help="Every scanned trial, not only each idea's latest"),
    only_fired: bool = typer.Option(False, "--fired", help="Only runs where a flag fired"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Flagged runs (docs/ANOMALIES.md): for each idea's latest scanned run, which flags fired,
    then totals per flag. Flags decide nothing. Reads the stored scans; `qlab anomalies scan`
    makes them."""
    if ctx.invoked_subcommand is not None:
        return
    from qlab.anomalies import report, store

    with read_session() as session:
        rows = store.latest_scans(session, all_scans=all_scans)
        if only_fired:
            rows = [r for r in rows if r.fired]
        summary = store.summarize(rows)
        if as_json:
            typer.echo(json.dumps({"summary": summary, "rows": [
                {"idea_id": r.idea_id, "trial_id": r.trial_id, "route": r.route,
                 "config_version": r.config_version, "fired": r.fired} for r in rows]},
                ensure_ascii=False, indent=1))
            return
        for line in report.render_table(session, rows):
            typer.echo(line)
        typer.echo("")
        for line in report.render_summary(summary):
            typer.echo(line)


@anomalies_app.command("scan")
def scan_cmd(
    trial: list[int] = typer.Option(  # noqa: B008
        None, "--trial", help="A trial id (repeat for several)"),
    idea: list[str] = typer.Option(  # noqa: B008
        None, "--idea", help="An idea id: its latest routed run"),
    passed: bool = typer.Option(
        False, "--passed",
        help="The latest routed run of every idea whose route is in scan.routes of "
             "night/anomalies.yaml"),
    force: bool = typer.Option(False, "--force", help="Scan again even if this version has a scan"),
) -> None:
    """Run the code detector and store its flags (the `anomaly_scan` table). No agent, no
    tokens; each trial is re-run from its stored spec and snapshot, seconds each."""
    from qlab.anomalies import stage
    from qlab.anomalies.config import load_config

    cfg = load_config()
    if cfg is None:
        _fail("night/anomalies.yaml is missing")
    with session_scope() as session:
        chosen = list(trial or [])
        if idea:
            from qlab.registry.stages import latest_routed

            for idea_id in idea:
                found = latest_routed(session, idea_id)
                if found is None:
                    _fail(f"{idea_id} has no routed run")
                chosen.append(found[0].id)
        if passed:
            chosen += [t for t, _ in stage.scan_candidates(session, cfg, rescan=force)]
        if not chosen:
            _fail("name --trial, --idea or --passed")
        for trial_id in dict.fromkeys(chosen):
            entry = stage.scan_one(session, trial_id, cfg, force=force)
            typer.echo(f"trial {trial_id}: " + entry)


@anomalies_app.command("show")
def show_cmd(trial: int = typer.Argument(..., help="Trial id")) -> None:
    """A trial's flags and its investigation dossier, if there is one."""
    from qlab.anomalies import report, store

    with read_session() as session:
        scan = store.latest_scan(session, trial)
        if scan is None:
            _fail(f"trial {trial} has no stored scan")
        for line in report.scan_lines(scan):
            typer.echo(line)
        inv = store.latest_investigation(session, trial_id=trial)
        if inv is not None:
            typer.echo("")
            typer.echo(inv.dossier)


@anomalies_app.command("learning")
def learning_cmd() -> None:
    """The learning loop: of the confirmed causes, which flags caught them and how many none
    did; and the checks the investigator proposed that await the owner."""
    from qlab.anomalies import report, store

    with read_session() as session:
        for line in report.learning_lines(session):
            typer.echo(line)
        pending = store.pending_checks(session)
        if pending:
            typer.echo("")
            typer.echo("proposed checks awaiting a decision "
                       "(qlab anomalies check ID adopted|declined):")
            for inv in pending:
                typer.echo(f"  {inv.id}  trial {inv.trial_id} ({inv.idea_id}): {inv.new_check}")


@anomalies_app.command("check")
def check_cmd(investigation: int = typer.Argument(..., help="Investigation id"),
              status: str = typer.Argument(..., help="adopted or declined")) -> None:
    """The owner's decision on an investigation's proposed new check."""
    from qlab.anomalies import store

    try:
        with session_scope() as session:
            row = store.set_check_status(session, investigation, status)
            typer.echo(f"investigation {row.id}: proposed check is now {row.new_check_status}")
    except ValueError as exc:
        _fail(str(exc))


@anomalies_app.command("investigate")
def investigate_cmd(
    trial: int = typer.Argument(..., help="Trial id (needs a stored scan with a fired flag)"),
    force: bool = typer.Option(
        False, "--force",
        help="Investigate even if no flag fired or it was investigated under the same flags"),
) -> None:
    """Call the investigator agent on one flagged trial, by hand, through the budget guard
    (stage: recheck). The agent changes no verdict, status or code."""
    from qlab.anomalies import stage
    from qlab.budget import BudgetExhausted

    try:
        text = stage.investigate_by_hand(trial, force=force)
    except BudgetExhausted as stop:
        _fail(f"the budget guard stopped it: {stop}")
    except ValueError as exc:
        _fail(str(exc))
    typer.echo(text)


__all__ = ["anomalies_app", "anomaly_app", "evaluate_lines", "read_session"]
