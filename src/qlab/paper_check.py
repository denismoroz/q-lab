"""The paper books against the stand, over ONE window (owner, 2026-10-25:
«с bv2 не совсем понимаю - на стенде минус на paper в плюсе»).

The idea page showed the stand's forward test (a few weeks, annualized) next
to the paper book's total since it started: different windows, so different
signs. Here both are measured over the stand's forward window:

* the stand's return is the forward total of its latest routed trial,
  `(1 + ann_return_net) ** (forward_days / 365.25) - 1` -- the same figure the
  night report shows;
* the paper return comes from the night's read-only export of the production
  books (`data/paper/<day>/`), from the first row at or after the window's
  start to the last row at or before its end;
* the replay return -- the stand's replay of the paper book
  (`scripts/replay_*_paper.py --out`) -- is read from the replay's own file on
  exactly the rows the paper window used, or left null.

Which paper book belongs to which idea is `night/paper_books.yaml`. Nothing is
invented: a missing file, trial or row gives a report line saying what is
missing, and no row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import func
from sqlalchemy.orm import Session

from qlab.registry.models import Idea, PaperCheck, Spec, Trial, TrialStatus, Verdict

BOOKS = Path("night/paper_books.yaml")
# A paper window starting or ending within this long of the stand's window
# counts as the same window (rows are hourly, the stand's bars hourly or daily).
SAME_WINDOW_TOLERANCE = timedelta(days=1)


@dataclass(frozen=True)
class WindowReturn:
    """A total return over a window, and the rows it actually used."""

    total: float
    first: datetime
    last: datetime
    rows: int


@dataclass(frozen=True)
class StandWindow:
    """The stand's forward test as its latest routed trial recorded it."""

    trial_id: int
    start: date
    end: date
    forward_days: float
    total: float


@dataclass
class PaperCheckResult:
    idea_id: str
    book: str
    label: str
    window_start: date
    window_end: date
    paper_return: float
    stand_return: float | None
    stand_trial_id: int | None
    replay_return: float | None
    stand_window_start: date | None
    stand_window_end: date | None
    note: str


def load_books(path: Path | None = None) -> dict[str, dict]:
    """Idea id -> its paper book's description (night/paper_books.yaml)."""
    return yaml.safe_load((path or BOOKS).read_text(encoding="utf-8"))["books"]


def forward_total(ann_return_net: float, forward_days: float) -> float:
    """The forward test's total over its days, from the run's yearly figure
    (the same conversion the night report uses)."""
    growth = 1 + ann_return_net
    if growth <= 0:
        return -1.0  # a yearly loss of 100% or more is a total loss over any span
    return growth ** (forward_days / 365.25) - 1


def _utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC)


def as_utc(value) -> datetime | None:
    """A config date or timestamp (`realised_from`) as an aware UTC datetime; None stays None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def equity_curve(csv: Path, *, column: str = "equity", strategy_id: int | None = None,
                 sum_coins: bool = False) -> pd.Series:
    """One book's equity by timestamp (UTC index) from an export.

    `sum_coins`: the export has one row per coin and timestamp, and the book's
    equity is their sum; a timestamp where a coin is missing is dropped rather
    than summed short."""
    frame = pd.read_csv(csv)
    if strategy_id is not None and "strategy_id" in frame.columns:
        frame = frame[frame["strategy_id"] == strategy_id]
    if sum_coins:
        grouped = frame.groupby("ts_ms")[column].agg(["sum", "count"])
        grouped = grouped[grouped["count"] == grouped["count"].max()]
        curve = grouped["sum"]
    else:
        curve = frame.drop_duplicates("ts_ms").set_index("ts_ms")[column]
    curve = curve.sort_index()
    curve.index = pd.to_datetime(curve.index, unit="ms", utc=True)
    return curve.astype(float)


def window_return(curve: pd.Series, start: datetime, end: datetime | None = None
                  ) -> WindowReturn | None:
    """Total return from the first row at or after `start` to the last row at
    or before `end` (None: the last row). None when fewer than two rows fit."""
    part = curve[curve.index >= pd.Timestamp(start)]
    if end is not None:
        part = part[part.index <= pd.Timestamp(end)]
    if len(part) < 2 or part.iloc[0] == 0:
        return None
    return WindowReturn(total=float(part.iloc[-1] / part.iloc[0] - 1),
                        first=part.index[0].to_pydatetime(), last=part.index[-1].to_pydatetime(),
                        rows=len(part))


def replay_window_return(csv: Path, first: datetime, last: datetime, *,
                         strategy_id: int | None = None) -> float | None:
    """The replay's return on exactly the rows the paper window used: None
    unless the replay has both end rows."""
    frame = pd.read_csv(csv)
    if strategy_id is not None and "strategy_id" in frame.columns:
        frame = frame[frame["strategy_id"] == strategy_id]
    curve = frame.drop_duplicates("ts_ms").set_index("ts_ms")["replay_equity"].sort_index()
    curve.index = pd.to_datetime(curve.index, unit="ms", utc=True)
    a, b = pd.Timestamp(first), pd.Timestamp(last)
    if a not in curve.index or b not in curve.index or curve[a] == 0 or a == b:
        return None
    return float(curve[b] / curve[a] - 1)


def stand_window(session: Session, idea_id: str, fixed_at: date | None) -> StandWindow | None:
    """The forward test of the idea's latest routed, evaluated trial: its
    window from the verdict rows that trial wrote (their range is the judged
    part), its total from the run's yearly figure and forward days. None when
    the idea has no such trial or it judged no forward test."""
    trial = (session.query(Trial).join(Spec, Spec.id == Trial.spec_id)
             .filter(Spec.idea_id == idea_id, Trial.route.isnot(None),
                     Trial.status == TrialStatus.OK, Trial.metrics.isnot(None))
             .order_by(Trial.id.desc()).first())
    if trial is None:
        return None
    metrics = trial.metrics or {}
    days = metrics.get("forward_days")
    ann = metrics.get("ann_return_net")
    if (metrics.get("judged_on_forward") != 1.0 or not isinstance(days, int | float) or days <= 0
            or not isinstance(ann, int | float)):
        return None
    end = (session.query(func.max(Verdict.data_range_end))
           .filter(Verdict.trial_id == trial.id).scalar())
    start = (session.query(func.min(Verdict.data_range_start))
             .filter(Verdict.trial_id == trial.id, Verdict.data_range_end == end).scalar()
             if end is not None else None)
    start = start or fixed_at
    if start is None:
        return None
    if end is None:
        end = (datetime.combine(start, datetime.min.time(), tzinfo=UTC)
               + timedelta(days=days) - timedelta(milliseconds=1)).date()
    return StandWindow(trial_id=trial.id, start=start, end=end, forward_days=float(days),
                       total=forward_total(ann, days))


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1%}".replace("-", "−")


def _fmt_day(day: date, *, year: bool = False) -> str:
    return day.strftime("%d.%m.%Y" if year else "%d.%m")


def _fmt_span(start: date, end: date) -> str:
    year = start.year != end.year
    return f"{_fmt_day(start, year=year)}–{_fmt_day(end, year=year)}"


def report_line(r: PaperCheckResult) -> str:
    """The check as one plain Russian line for the morning report."""
    differ = r.note.startswith("Отрезки разные")
    replay = ("" if r.replay_return is None
              else f", повтор бумаги на стенде {_fmt_pct(r.replay_return)}")
    if r.stand_return is None:
        return (f"{r.label}: за {_fmt_span(r.window_start, r.window_end)} бумага "
                f"{_fmt_pct(r.paper_return)}{replay}; стенд за этот отрезок не посчитан. "
                f"{r.note}").strip()
    if not differ:
        return (f"{r.label}: за {_fmt_span(r.window_start, r.window_end)} бумага "
                f"{_fmt_pct(r.paper_return)}, стенд {_fmt_pct(r.stand_return)}{replay}")
    return (f"{r.label}: бумага за {_fmt_span(r.window_start, r.window_end)} "
            f"{_fmt_pct(r.paper_return)}{replay}; стенд за "
            f"{_fmt_span(r.stand_window_start, r.stand_window_end)} {_fmt_pct(r.stand_return)} — "
            f"отрезки разные, числа сравнивать нельзя")


def check_book(session: Session, idea_id: str, cfg: dict, folder: Path, day: date,
               ) -> tuple[PaperCheckResult | None, str]:
    """One idea's check: (result, line). No result when the paper side cannot
    be computed; the line then says what is missing."""
    from qlab.pipeline.spec import load_spec

    label = cfg.get("label", idea_id)
    if session.get(Idea, idea_id) is None:
        return None, f"{label}: идеи {idea_id} нет в реестре, сверка пропущена"
    paper_csv = folder / cfg["paper_file"]
    if not paper_csv.is_file():
        return None, f"{label}: нет выгрузки бумажной книги ({cfg['paper_file']}), сверка пропущена"
    try:
        fixed_at = load_spec(cfg["spec"]).params_fixed_at
    except Exception as exc:  # noqa: BLE001 - a broken spec file must not stop the night
        return None, (f"{label}: не удалось прочитать спецификацию {cfg['spec']} "
                      f"({type(exc).__name__})")
    stand = stand_window(session, idea_id, fixed_at)
    start = stand.start if stand is not None else fixed_at
    if start is None:
        return None, f"{label}: у стенда не задана дата начала проверки вперёд, сверка пропущена"
    start_ts = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    end_ts = start_ts + timedelta(days=stand.forward_days) if stand is not None else None
    # A book that replayed history at launch is paper only from `realised_from` on; the
    # replayed part is the stand's own arithmetic and would "agree" with it trivially.
    realised_from = as_utc(cfg.get("realised_from"))
    paper_from = max(start_ts, realised_from) if realised_from is not None else start_ts

    curve = equity_curve(paper_csv, strategy_id=cfg.get("strategy_id"),
                         sum_coins=bool(cfg.get("sum_coins")))
    paper = window_return(curve, paper_from, end_ts) if end_ts is None or paper_from < end_ts \
        else None
    if paper is None:
        if realised_from is not None and end_ts is not None and realised_from >= end_ts:
            return None, (f"{label}: бумага идёт с {_fmt_day(realised_from.date(), year=True)}, "
                          f"а проверка вперёд на стенде кончается "
                          f"{_fmt_day(stand.end, year=True)} — общего отрезка пока нет")
        return None, (f"{label}: в выгрузке бумаги нет двух точек с {_fmt_day(paper_from.date())}, "
                      f"сверка пропущена")

    replay = None
    replay_csv = folder / cfg["replay_file"] if cfg.get("replay_file") else None
    if replay_csv is not None and replay_csv.is_file():
        replay = replay_window_return(replay_csv, paper.first, paper.last,
                                      strategy_id=cfg.get("strategy_id"))

    notes: list[str] = []
    window_start = paper.first.date()
    window_end = (paper.last - timedelta(milliseconds=1)).date()
    if stand is None:
        notes.append("На стенде нет проверки вперёд с записанным итогом.")
    else:
        late = paper.first - start_ts > SAME_WINDOW_TOLERANCE
        early = end_ts - paper.last > SAME_WINDOW_TOLERANCE
        if late or early:
            notes.append(
                f"Отрезки разные: проверка на стенде идёт с {_fmt_day(stand.start, year=True)} "
                f"по {_fmt_day(stand.end, year=True)}, а у бумажной книги есть данные только "
                f"с {_fmt_day(window_start, year=True)} по {_fmt_day(window_end, year=True)}.")
    result = PaperCheckResult(
        idea_id=idea_id, book=cfg["book"], label=label, window_start=window_start,
        window_end=window_end, paper_return=paper.total,
        stand_return=stand.total if stand is not None else None,
        stand_trial_id=stand.trial_id if stand is not None else None, replay_return=replay,
        stand_window_start=stand.start if stand is not None else None,
        stand_window_end=stand.end if stand is not None else None, note=" ".join(notes))
    return result, report_line(result)


def save_check(session: Session, result: PaperCheckResult, day: date) -> PaperCheck:
    """Write the check, replacing the same night's row for the idea."""
    row = (session.query(PaperCheck)
           .filter(PaperCheck.idea_id == result.idea_id, PaperCheck.at == day).one_or_none())
    if row is None:
        row = PaperCheck(idea_id=result.idea_id, at=day)
        session.add(row)
    for name in ("book", "window_start", "window_end", "paper_return", "stand_return",
                 "stand_trial_id", "replay_return", "stand_window_start", "stand_window_end",
                 "note"):
        setattr(row, name, getattr(result, name))
    session.flush()
    return row


def run_paper_checks(session: Session, day: date, folder: Path, books: dict[str, dict] | None = None
                     ) -> list[str]:
    """Every mapped idea's check for the night's export in `folder`; the report
    lines. One idea failing does not stop the others."""
    books = books if books is not None else load_books()
    lines: list[str] = []
    for idea_id, cfg in books.items():
        try:
            result, line = check_book(session, idea_id, cfg, folder, day)
            if result is not None:
                save_check(session, result, day)
        except Exception as exc:  # noqa: BLE001 - recorded in the report, the night goes on
            line = f"{cfg.get('label', idea_id)}: сверка не удалась ({type(exc).__name__}: {exc})"
        lines.append(line)
    return lines


def latest_check(session: Session, idea_id: str) -> PaperCheck | None:
    return (session.query(PaperCheck).filter(PaperCheck.idea_id == idea_id)
            .order_by(PaperCheck.at.desc(), PaperCheck.id.desc()).first())
