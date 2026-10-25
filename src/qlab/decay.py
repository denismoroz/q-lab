"""Decay detector: does a running strategy still behave like itself on the stand?

Owner's goal (docs/PLAN.md, M4-M6): a pool of 5-7 working strategies in which a
decaying one is replaced. The first step is to SEE a strategy stop behaving the
way its own history says is normal.

For every strategy that trades on paper or live money (`night/paper_books.yaml`):

* **Realised** -- the book's equity from the night's read-only export of the
  production host (`data/paper/<day>/`), from the book's first row (or its
  `realised_from`) to the latest row.
* **Expected** -- the same strategy's history on the stand: its spec's strategy
  is run on the CACHED panel, cut at the first realised row (nothing the
  realised period saw reaches it), WITHOUT writing a `trial` (nothing is
  selected here, a recorded strategy is only described -- `qlab.diagnose`).
  Every rolling window of the realised period's length over that history gives
  one return, one deepest fall and one daily spread; together they are the
  distribution of what "normal" looked like.

The output is a set of numbers and where the realised ones fall among the
stand's, and nothing else:

* the realised return and its percentile among same-length stand windows, with
  the stand's median, 10th and 90th percentiles;
* the realised deepest fall against the stand's windows;
* the realised daily spread against the stand's;
* for every band the strategy's preregistration declared
  (`night/decay.yaml`, quoted from `seed/preregistration/*.yaml`), whether the
  realised result is inside it -- a band for a one-year horizon says nothing
  about six weeks, so it is "not applicable" until the realised period is that
  long.

**No verdict is written here.** CLAUDE.md: verdicts are code, thresholds are
declared, never invented. No threshold on the percentile has been declared:
`alert_percentile` in `night/decay.yaml` is null and the owner sets it. While it
is null, nothing is labelled good or bad. Preregistration 2026-09-21 PREDICTION 0
still holds: a paper return over weeks says nothing about profitability, so this
module describes behaviour ("does it look like its own history"), never value.

Measures are taken on a DAILY grid (the last value of each UTC day) on both
sides, so an hourly paper book and a daily stand run are comparable; a drawdown
inside a day is not seen on either side.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from qlab import paper_check, paper_rule
from qlab.paper_check import equity_curve, window_return

CONFIG = Path("night/decay.yaml")
NIGHT_DIR = Path("data/night")

MIN_WINDOWS = 2
"""Fewer same-length windows than this cannot give quantiles. A structural
minimum (two points make a spread), not a judgement about how many are enough;
the report always states how many windows and how many non-overlapping ones the
history holds."""

QUANTILES = (0.1, 0.5, 0.9)
"""The stand's 10th, 50th and 90th percentiles: the "80% band" the owner's
preregistration also speaks in (seed/preregistration/2026-09-21-paper-bv2-trend.yaml)."""


# ---------------------------------------------------------------------------
# Pure measures
# ---------------------------------------------------------------------------


def daily_closes(equity: pd.Series) -> pd.Series:
    """The last value of each UTC day, on a gap-free daily index (a missing day
    holds the previous close). An instant exactly at midnight closes the day
    before it."""
    if equity.empty:
        return equity
    days = (equity.index - pd.Timedelta(milliseconds=1)).floor("D")
    closes = equity.groupby(days).last()
    full = pd.date_range(closes.index[0], closes.index[-1], freq="D", tz="UTC")
    return closes.reindex(full).ffill()


def max_drawdown(values: np.ndarray) -> float:
    """The deepest fall from a running peak, as a negative fraction (0.0 when
    the path never falls)."""
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    return float((values / peak - 1.0).min())


def daily_spread(values: np.ndarray) -> float | None:
    """The standard deviation of the day-to-day returns of a path of daily
    closes; None with fewer than two returns."""
    if len(values) < 3:
        return None
    returns = values[1:] / values[:-1] - 1.0
    return float(np.std(returns, ddof=1))


@dataclass(frozen=True)
class WindowMeasure:
    start: date
    total: float
    max_dd: float
    spread: float | None


def rolling_windows(closes: pd.Series, n_days: int) -> list[WindowMeasure]:
    """Every window of `n_days` days over a daily path: its total return, its
    deepest fall and its daily spread. Empty when the path is not longer than
    one window."""
    values = closes.to_numpy(dtype=float)
    out: list[WindowMeasure] = []
    if n_days < 1:
        return out
    for i in range(0, len(values) - n_days):
        seg = values[i:i + n_days + 1]
        if seg[0] == 0:
            continue
        out.append(WindowMeasure(start=closes.index[i].date(), total=float(seg[-1] / seg[0] - 1),
                                 max_dd=max_drawdown(seg), spread=daily_spread(seg)))
    return out


def percentile_of(value: float, sample: list[float] | np.ndarray) -> float | None:
    """Where `value` falls among `sample`, 0-100: the share of the sample below
    it, ties counting half. None for an empty sample."""
    arr = np.asarray(sample, dtype=float)
    if arr.size == 0:
        return None
    return float(100.0 * ((arr < value).sum() + 0.5 * (arr == value).sum()) / arr.size)


@dataclass(frozen=True)
class Spread:
    """The stand's 10th / 50th / 90th percentile of one measure."""

    p10: float
    p50: float
    p90: float


def spread_of(sample: list[float]) -> Spread | None:
    if not sample:
        return None
    q = np.quantile(np.asarray(sample, dtype=float), QUANTILES)
    return Spread(p10=float(q[0]), p50=float(q[1]), p90=float(q[2]))


@dataclass(frozen=True)
class Measured:
    """One measure: the realised value, the stand's spread of it, and where the
    realised value falls among the stand's windows (0-100)."""

    realised: float | None
    stand: Spread | None
    percentile: float | None


def _measured(realised: float | None, sample: list[float]) -> Measured:
    if realised is None or len(sample) < MIN_WINDOWS:
        return Measured(realised, None, None)
    return Measured(realised, spread_of(sample), percentile_of(realised, sample))


# ---------------------------------------------------------------------------
# Preregistered bands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """A band the preregistration declared, quoted in `night/decay.yaml`.

    `metric` is "return" (total over the realised period) or "max_drawdown";
    `lo` / `hi` are the declared ends (either may be absent); `horizon_days` is
    the horizon the declaration is about: shorter realised periods get
    "not applicable". `kind` is "band" (an expected range) or "stop" (a stop
    condition: outside the range means the stop fired)."""

    id: str
    metric: str
    label: str
    source: str
    kind: str = "band"
    lo: float | None = None
    hi: float | None = None
    horizon_days: float | None = None


@dataclass(frozen=True)
class BandCheck:
    id: str
    label: str
    kind: str
    metric: str
    status: str  # "inside" | "outside" | "not_applicable"
    value: float
    lo: float | None
    hi: float | None
    horizon_days: float | None
    source: str


def check_band(band: Band, *, value: float, realised_days: float) -> BandCheck:
    """Is the realised `value` inside the declared band? Not applicable when the
    realised period is shorter than the band's horizon."""
    if band.horizon_days is not None and realised_days < band.horizon_days:
        status = "not_applicable"
    else:
        below = band.lo is not None and value < band.lo
        above = band.hi is not None and value > band.hi
        status = "outside" if below or above else "inside"
    return BandCheck(id=band.id, label=band.label, kind=band.kind, metric=band.metric,
                     status=status, value=value, lo=band.lo, hi=band.hi,
                     horizon_days=band.horizon_days, source=band.source)


# ---------------------------------------------------------------------------
# The comparison of one book
# ---------------------------------------------------------------------------


@dataclass
class DecayResult:
    idea_id: str
    label: str
    book: str
    live: bool
    first: datetime
    last: datetime
    days: float
    window_days: int
    history_from: date | None
    history_to: date | None
    history_days: int
    windows: int
    independent: int
    ret: Measured
    drawdown: Measured
    spread: Measured
    bands: list[BandCheck] = field(default_factory=list)
    alert: str | None = None  # "below" | "above" | None: only with a declared threshold
    note: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["first"], data["last"] = self.first.isoformat(), self.last.isoformat()
        data["history_from"] = self.history_from.isoformat() if self.history_from else None
        data["history_to"] = self.history_to.isoformat() if self.history_to else None
        data["text"] = result_lines(self)
        return data


def alert_label(percentile: float | None, alert_percentile: float | None) -> str | None:
    """"below" when the realised return sits at or under the owner's declared
    percentile; None when no threshold is declared or no percentile exists --
    nothing is labelled without a declared number."""
    if alert_percentile is None or percentile is None:
        return None
    return "below" if percentile <= alert_percentile else None


def trim_flat_start(net_return: pd.Series) -> pd.Series:
    """Drop the leading periods in which the strategy held nothing (exactly zero
    return): a strategy still warming up has no behaviour to describe."""
    nonzero = np.flatnonzero(net_return.to_numpy() != 0)
    return net_return.iloc[nonzero[0]:] if nonzero.size else net_return.iloc[0:0]


def stand_closes(net_return: pd.Series, until: pd.Timestamp) -> pd.Series:
    """Daily closes of the stand's equity (start 1.0) up to and including the
    day that ends at or before `until`."""
    from qlab.diagnose import equity_from_returns

    equity = equity_from_returns(trim_flat_start(net_return))
    return daily_closes(equity[equity.index <= until])


def compare(*, idea_id: str, label: str, book: str, live: bool, realised: pd.Series,
            stand_net_return: pd.Series, bands: list[Band],
            alert_percentile: float | None, realised_from: datetime | None = None
            ) -> DecayResult | None:
    """The realised equity curve against the stand's net returns (decision-time
    index, the harness's `net_return`). None when the realised curve has fewer
    than two rows from `realised_from`."""
    start = realised_from or realised.index[0].to_pydatetime()
    done = window_return(realised, start)
    if done is None:
        return None
    first, last = pd.Timestamp(done.first), pd.Timestamp(done.last)
    part = realised[(realised.index >= first) & (realised.index <= last)]
    days = (last - first) / pd.Timedelta(days=1)
    n_days = max(int(round(days)), 1)

    closes = daily_closes(part)
    path = np.concatenate([[float(part.iloc[0])], closes.to_numpy(dtype=float)])
    realised_dd = max_drawdown(path)
    realised_spread = daily_spread(closes.to_numpy(dtype=float))

    history = stand_closes(stand_net_return, first)
    windows = rolling_windows(history, n_days)
    history_days = max(len(history) - 1, 0)
    note = ""
    if len(windows) < MIN_WINDOWS:
        note = ("истории стенда до начала этого периода не хватает даже на два таких отрезка: "
                "сравнивать не с чем")
    result = DecayResult(
        idea_id=idea_id, label=label, book=book, live=live, first=done.first, last=done.last,
        days=float(days), window_days=n_days,
        history_from=history.index[0].date() if len(history) else None,
        history_to=history.index[-1].date() if len(history) else None,
        history_days=history_days, windows=len(windows),
        independent=history_days // n_days if n_days else 0,
        ret=_measured(done.total, [w.total for w in windows]),
        drawdown=_measured(realised_dd, [w.max_dd for w in windows]),
        spread=_measured(realised_spread, [w.spread for w in windows if w.spread is not None]),
        note=note)
    values = {"return": done.total, "max_drawdown": realised_dd}
    result.bands = [check_band(b, value=values[b.metric], realised_days=days) for b in bands]
    result.alert = alert_label(result.ret.percentile, alert_percentile)
    return result


# ---------------------------------------------------------------------------
# Configuration and mapping
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> dict:
    """`night/decay.yaml`; `alert_percentile` must be null or a number 0-100."""
    config = yaml.safe_load((path or CONFIG).read_text(encoding="utf-8")) or {}
    alert = config.get("alert_percentile")
    if alert is not None and not (isinstance(alert, int | float) and 0 <= alert <= 100):
        raise ValueError(f"alert_percentile must be null or a number from 0 to 100, got {alert!r}")
    config.setdefault("alert_percentile", None)
    config.setdefault("preregistered", {})
    return config


def bands_for(config: dict, idea_id: str) -> list[Band]:
    return [Band(id=b["id"], metric=b["metric"], label=b["label"], source=b["source"],
                 kind=b.get("kind", "band"), lo=b.get("lo"), hi=b.get("hi"),
                 horizon_days=b.get("horizon_days"))
            for b in (config.get("preregistered") or {}).get(idea_id) or []]


def load_all_books(path: Path | None = None) -> dict[str, dict]:
    """Paper books and live-money books of `night/paper_books.yaml`, idea id ->
    description. Live books are marked `live: True`; they are kept out of
    `paper_check.load_books` on purpose (the paper rule and the paper-vs-stand
    check are about paper)."""
    raw = yaml.safe_load((path or paper_check.BOOKS).read_text(encoding="utf-8"))
    books = {k: dict(v, live=False) for k, v in (raw.get("books") or {}).items()}
    books.update({k: dict(v, live=True) for k, v in (raw.get("live_books") or {}).items()})
    return books


def live_export_sql(strategy_id: int, since_ms: int) -> str:
    """The cheap hourly sample of a live strategy's equity for the production
    host's sqlite (one snapshot per minute is stored; the first minute of each
    hour is kept, from `since_ms` on, so a year of minutes is not read)."""
    return (f"select strategy_id, ts_ms, total_equity from equity_snapshots "
            f"where strategy_id={int(strategy_id)} and ts_ms>={int(since_ms)} "
            f"and ts_ms%3600000<60000 order by ts_ms")


def live_exports(books: dict[str, dict] | None = None) -> list[tuple[str, str]]:
    """(export file name, sql) for every live book that declares `realised_from`."""
    books = books if books is not None else load_all_books()
    out = []
    for cfg in books.values():
        if not cfg.get("live"):
            continue
        since = _as_datetime(cfg.get("realised_from"))
        if since is None:
            continue
        out.append((cfg["paper_file"],
                    live_export_sql(cfg["strategy_id"], int(since.timestamp() * 1000))))
    return out


def _as_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# The stand's history of a strategy (no trial is written)
# ---------------------------------------------------------------------------


class StandHistoryMissing(Exception):
    """The stand has nothing to compare with (panel not cached, too short)."""


def stand_net_return(session: Session, spec_path: str, *, end: date, cut: pd.Timestamp
                     ) -> pd.Series:
    """The spec's strategy run on its cached panel, the panel cut at `cut`
    (the first realised row), as the harness's `net_return`. Never fetches and
    never writes a trial: a recorded strategy is only described, as
    `qlab.diagnose.diagnose_spec` does."""
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest
    from qlab.night import with_end
    from qlab.pipeline.evaluate import (
        _slice_rows,
        complete_book_window,
        resolve_cached_panel,
        resolve_strategy,
    )
    from qlab.pipeline.spec import load_spec

    spec = with_end(load_spec(spec_path), end)
    panel = resolve_cached_panel(session, spec)
    if panel is None:
        raise StandHistoryMissing(f"the panel of {spec_path} is not cached")
    truncated = dict(panel.meta.get("history_truncated") or {})
    if truncated:  # the same honest start the evaluation uses
        panel = _slice_rows(panel, int(panel.prices.index.searchsorted(
            pd.Timestamp(max(truncated.values())))), len(panel.prices.index) - 1)
    index = panel.prices.index
    first, last = 0, len(index) - 1
    if spec.required_instruments is not None:
        coverage = complete_book_window(panel, spec.required_instruments)
        if coverage.window is not None:
            first, last = coverage.window
    last = min(last, int(index.searchsorted(cut, side="right")) - 1)
    if last - first < 3:
        raise StandHistoryMissing(f"{spec_path}: fewer than three bars before {cut:%Y-%m-%d}")
    seen = _slice_rows(panel, 0, last)  # the realised period never reaches the strategy
    weights = resolve_strategy(spec.code_ref).target_weights(seen, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    return run_backtest(seen, weights, costs, seen.funding).net_return.iloc[first:]


# ---------------------------------------------------------------------------
# Russian report text
# ---------------------------------------------------------------------------


def _pct(value: float | None, digits: int = 1, sign: bool = True) -> str:
    if value is None or value != value:
        return "—"
    text = f"{value:+.{digits}%}" if sign else f"{value:.{digits}%}"
    return text.replace("-", "−")


def _day(d: date | datetime) -> str:
    return d.strftime("%d.%m")


def _days_word(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "день"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "дня"
    return "дней"


def _worse_of_100(percentile: float) -> str:
    return f"{percentile:.0f}"


def result_lines(r: DecayResult) -> list[str]:
    """One book's check as plain Russian lines for the morning report and the
    idea page."""
    kind = "живые деньги" if r.live else "бумага"
    n = int(round(r.days))
    head = (f"{r.label} ({kind}, {n} {_days_word(n)} с {_day(r.first)}): "
            f"за это время {_pct(r.ret.realised)}.")
    if r.note:
        return [head, f"    {r.note[0].upper()}{r.note[1:]}."]
    lines = [head]
    s = r.ret.stand
    lines.append(
        f"    Такой же длины отрезков в истории стенда до {_day(r.first)}: {r.windows} "
        f"(с {_day(r.history_from)} по {_day(r.history_to)}; друг на друга не налегают "
        f"примерно {r.independent}). Хуже нынешнего результата были "
        f"{_worse_of_100(r.ret.percentile)} из 100; обычно на стенде от {_pct(s.p10)} до "
        f"{_pct(s.p90)}, посередине {_pct(s.p50)}.")
    d = r.drawdown
    if d.stand is not None:
        lines.append(
            f"    Самое глубокое падение за период: {_pct(d.realised)}; на стенде обычно "
            f"{_pct(d.stand.p50)} (от {_pct(d.stand.p10)} до {_pct(d.stand.p90)}), глубже было "
            f"у {_worse_of_100(d.percentile)} из 100 отрезков.")
    v = r.spread
    if v.realised is not None and v.stand is not None:
        lines.append(
            f"    Дневной разброс (насколько день в среднем отличается от обычного дня): "
            f"{_pct(v.realised, 2, sign=False)}; на стенде обычно "
            f"{_pct(v.stand.p50, 2, sign=False)} (от {_pct(v.stand.p10, 2, sign=False)} до "
            f"{_pct(v.stand.p90, 2, sign=False)}); спокойнее были "
            f"{_worse_of_100(v.percentile)} из 100 отрезков.")
    for b in r.bands:
        lines.append(f"    {band_line(b)}")
    if r.alert == "below":
        lines.append("    Результат ниже порога тревоги, который задал владелец.")
    return lines


def band_line(b: BandCheck) -> str:
    """One preregistered band as a plain Russian line."""
    if b.status == "not_applicable":
        return (f"Записано заранее, до результатов: {b.label}: к этому отрезку не относится "
                f"(заявлено на {b.horizon_days:.0f} {_days_word(int(b.horizon_days))}).")
    if b.kind == "stop":
        verdict = ("условие остановки СРАБОТАЛО" if b.status == "outside"
                   else "условие остановки не сработало")
        return f"Записано заранее, до результатов: {b.label}: сейчас {_pct(b.value)}, {verdict}."
    where = "внутри заявленной полосы" if b.status == "inside" else "ВНЕ заявленной полосы"
    return (f"Записано заранее, до результатов: {b.label}: сейчас {_pct(b.value)}, {where} "
            f"({_pct(b.lo)} … {_pct(b.hi)}).")


SECTION_TITLE = "Ведёт ли себя стратегия как на стенде"

NO_THRESHOLD_TEXT = (
    "Порог тревоги владелец не задал (`night/decay.yaml`, `alert_percentile`), поэтому «хорошо» "
    "и «плохо» здесь не говорится: показано, где результат лежит среди обычных на стенде. "
    "Доходность по неделям о выгоде стратегии не говорит — только о том, похожа ли она на себя.")

THRESHOLD_TEXT = ("Порог тревоги, который задал владелец: результат в нижних {value:g} из 100 "
                  "отрезков стенда.")


# ---------------------------------------------------------------------------
# The night step
# ---------------------------------------------------------------------------


def run_decay(session: Session, day: date, end: date, *, books: dict[str, dict] | None = None,
              config: dict | None = None, root: Path | None = None, stand_fn=None
              ) -> tuple[list[DecayResult], list[str]]:
    """Every mapped book's check against the stand: (results, report lines).
    A book that cannot be computed is a report line; the others go on.
    `stand_fn(session, spec_path, end=, cut=)` returns the stand's net returns
    (default: the spec run on its cached panel)."""
    books = books if books is not None else load_all_books()
    config = config if config is not None else load_config()
    stand_fn = stand_fn or stand_net_return
    results: list[DecayResult] = []
    lines: list[str] = []
    for idea_id, cfg in books.items():
        label = cfg.get("label", idea_id)
        try:
            csv = paper_rule.latest_export(root if root is not None else paper_rule.PAPER_DIR,
                                           cfg["paper_file"])
            if csv is None:
                lines.append(f"{label}: нет выгрузки книги ({cfg['paper_file']}), сверка пропущена")
                continue
            curve = equity_curve(csv, column=cfg.get("equity_column", "equity"),
                                 strategy_id=cfg.get("strategy_id"),
                                 sum_coins=bool(cfg.get("sum_coins")))
            realised_from = _as_datetime(cfg.get("realised_from"))
            sub = (curve if realised_from is None
                   else curve[curve.index >= pd.Timestamp(realised_from)])
            if len(sub) < 2:
                lines.append(f"{label}: в выгрузке меньше двух точек, сверка пропущена")
                continue
            net = stand_fn(session, cfg["spec"], end=end, cut=sub.index[0])
            result = compare(
                idea_id=idea_id, label=label, book=cfg["book"], live=bool(cfg.get("live")),
                realised=curve, stand_net_return=net, bands=bands_for(config, idea_id),
                alert_percentile=config["alert_percentile"], realised_from=realised_from)
            if result is None:
                lines.append(f"{label}: в выгрузке меньше двух точек, сверка пропущена")
                continue
            results.append(result)
            lines += result_lines(result)
        except StandHistoryMissing as exc:
            lines.append(f"{label}: на стенде нет истории для сравнения ({exc})")
        except Exception as exc:  # noqa: BLE001 - a report line, the night goes on
            lines.append(f"{label}: сверка не удалась ({type(exc).__name__}: {exc})")
    footer = (THRESHOLD_TEXT.format(value=config["alert_percentile"])
              if config["alert_percentile"] is not None else NO_THRESHOLD_TEXT)
    return results, [*lines, footer]


def save_results(session: Session, results: list[DecayResult], day: date) -> bool:
    """Write the night's results, replacing the same night's rows. False (and
    nothing written) when the table does not exist yet -- the migration is the
    owner's to apply."""
    from qlab.registry.models import DecayCheck

    if not inspect(session.get_bind()).has_table(DecayCheck.__tablename__):
        return False
    for r in results:
        row = (session.query(DecayCheck)
               .filter(DecayCheck.idea_id == r.idea_id, DecayCheck.at == day).one_or_none())
        if row is None:
            row = DecayCheck(idea_id=r.idea_id, at=day)
            session.add(row)
        row.book = r.book
        row.window_start, row.window_end = r.first.date(), r.last.date()
        row.realised_return = r.ret.realised
        row.return_percentile = r.ret.percentile
        row.detail = r.to_dict()
    session.flush()
    return True


def latest_decay(session: Session, idea_id: str, *, night_dir: Path | None = None
                 ) -> dict | None:
    """The idea's latest check as the dict `DecayResult.to_dict` makes: from the
    registry table when it exists, else from the newest night state that holds
    one. None for an idea with no book or no check yet."""
    from qlab.registry.models import DecayCheck

    if inspect(session.get_bind()).has_table(DecayCheck.__tablename__):
        row = (session.query(DecayCheck).filter(DecayCheck.idea_id == idea_id)
               .order_by(DecayCheck.at.desc(), DecayCheck.id.desc()).first())
        if row is not None:
            return dict(row.detail, at=row.at.isoformat())
    folder = night_dir if night_dir is not None else NIGHT_DIR
    for path in sorted(folder.glob("*.json"), reverse=True):
        try:
            state = json.loads(path.read_text(encoding="utf-8")).get("decay") or {}
        except (OSError, ValueError):
            continue
        for item in state.get("results") or []:
            if item.get("idea_id") == idea_id:
                return dict(item, at=path.stem)
    return None


__all__ = [
    "Band", "BandCheck", "DecayResult", "Measured", "SECTION_TITLE", "Spread",
    "StandHistoryMissing", "WindowMeasure", "alert_label", "band_line", "bands_for",
    "check_band", "compare", "daily_closes", "latest_decay", "live_export_sql", "live_exports",
    "load_all_books", "load_config", "max_drawdown", "percentile_of", "result_lines",
    "rolling_windows", "run_decay", "save_results", "stand_net_return", "trim_flat_start",
]
