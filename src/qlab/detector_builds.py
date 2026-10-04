"""Building the stored regime detectors and level-2 decisions -- one code path
for the CLI and the nightly run (docs/REGIME_DETECT.md, docs/NIGHT.md).

Owner, 2026-10-04: «да, делай оба» -- a strategy that trades on a detector
needs its predictions current every day, not built once by hand.

`night/detectors.yaml` lists what to build, level 1 before level 2 (level 2
reads level 1's file). Every build walks forward from scratch on data up to
the given end: a model for a past month sees only what was known then, so
rebuilding reproduces yesterday's predictions and adds today's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

CONFIG = Path("night/detectors.yaml")


@dataclass
class Built:
    name: str
    path: Path
    first: pd.Timestamp
    last: pd.Timestamp
    note: str = ""


def level1(name: str, *, returns_only: bool, window: int = 30, start: str = "2020-01-01",
           market_spec: str | None = "specs/trend-binance-cap.yaml",
           end: date | None = None) -> Built:
    """The market regime detector (`qlab.regime_detect`)."""
    from qlab import regime_detect as rd
    from qlab.regimes import market_closes

    btc = market_closes()
    if btc is None:
        raise ValueError("no BTC closes: run `qlab regimes build` first")
    if returns_only:
        feats = pd.DataFrame({f"ret_{s}": btc / btc.shift(s) - 1.0
                              for s in rd.return_spans(window)}, index=btc.index)
    else:
        if window != 30:
            raise ValueError("only the returns-only set is scaled to another window")
        market = funding = tradeable = None
        if market_spec:
            from qlab.night import with_end
            from qlab.pipeline.evaluate import resolve_panel
            from qlab.pipeline.spec import load_spec
            from qlab.registry.db import session_scope

            spec = load_spec(Path(market_spec))
            if end is not None:
                spec = with_end(spec, end)
            with session_scope() as session:
                panel = resolve_panel(session, spec)
            day = pd.Timedelta(days=1)  # daily panels are stamped by candle open
            market = panel.prices.set_axis(panel.prices.index + day)
            funding = panel.funding.set_axis(panel.funding.index + day)
            tradeable = panel.tradeable.set_axis(panel.tradeable.index + day)
        feats = rd.features(btc, market, funding, tradeable)
    predictions = rd.walk_forward(btc, feats, pd.Timestamp(start, tz="UTC"), window)
    path = rd.store(name, predictions, {
        "features": list(feats.columns), "window_days": window,
        "model": "StandardScaler + LogisticRegression (scikit-learn defaults)",
        "embargo_days": window, "retrain": "monthly, on labels known at the month's start",
        "market_spec": None if returns_only else market_spec,
        "first": predictions.index[0], "last": predictions.index[-1]})
    return Built(name, path, predictions.index[0], predictions.index[-1],
                 f"features {', '.join(feats.columns)}")


def persistent(name: str, *, source: str, penalty: float) -> Built:
    """A level-1 file's regime with a switching penalty
    (`qlab.regime_detect.persistent`)."""
    from qlab import regime_detect as rd

    lvl = rd.load_predictions(source)
    out = lvl[["p_bull", "p_flat", "p_bear"]].copy()
    out["label"] = rd.persistent(lvl, penalty)
    out["trained_until"] = lvl["trained_until"]
    path = rd.store(name, out, {"from": source, "penalty": penalty,
                                "method": "qlab.regime_detect.persistent"})
    return Built(name, path, out.index[0], out.index[-1], f"penalty {penalty} on {source}")


def meta(name: str, *, spec: str, level1_name: str, end: date | None = None) -> Built:
    """A daily strategy's level 2 (`qlab.meta_label`)."""
    from qlab import meta_label as ml
    from qlab.night import with_end
    from qlab.pipeline.spec import load_spec
    from qlab.regime_detect import load_predictions
    from qlab.regimes import market_closes
    from qlab.registry.db import session_scope

    the_spec = load_spec(Path(spec))
    if end is not None:
        the_spec = with_end(the_spec, end)
    with session_scope() as session:
        returns = ml.strategy_returns_of(the_spec, session)
    first = load_predictions(level1_name)
    feats = ml.features(first, market_closes(), returns)
    decisions = ml.walk_forward(feats, returns, first.index[0])
    path = ml.store(name, decisions, {
        "spec": spec, "level1": level1_name, "features": list(feats.columns),
        "horizon_days": ml.HORIZON_DAYS, "embargo_days": ml.EMBARGO_DAYS,
        "model": "StandardScaler + LogisticRegression (scikit-learn defaults)",
        "first": decisions.index[0], "last": decisions.index[-1]})
    earned = ml.outcome(returns).reindex(decisions.index)
    both = decisions.assign(earned=earned).dropna(subset=["earned"])
    hit = float((both["trade"] == (both["earned"] == 1.0)).mean()) if len(both) else float("nan")
    return Built(name, path, decisions.index[0], decisions.index[-1],
                 f"trades on {decisions['trade'].mean():.0%} of days; right on {hit:.0%}; "
                 f"today: {'trade' if bool(decisions['trade'].iloc[-1]) else 'cash'}")


def meta_hedge(name: str, *, spec: str, level1_name: str, end: date | None = None) -> Built:
    """Bv2's level 2 (`qlab.meta_hedge`)."""
    from qlab import meta_hedge as mh
    from qlab import meta_label as ml
    from qlab.night import with_end
    from qlab.pipeline.evaluate import resolve_panel
    from qlab.pipeline.spec import load_spec
    from qlab.regime_detect import load_predictions
    from qlab.registry.db import session_scope

    the_spec = load_spec(Path(spec))
    if end is not None:
        the_spec = with_end(the_spec, end)
    with session_scope() as session:
        panel = resolve_panel(session, the_spec)
    first = load_predictions(level1_name)
    cost = mh.round_trip_cost(dict(the_spec.params))
    coins = list(the_spec.params["coins"])
    feats = {c: mh.coin_features(panel.prices[c], panel.funding[c], first) for c in coins}
    labels = {c: mh.hedge_pays(panel.prices[c], panel.funding[c], cost) for c in coins}
    start = max(first.index[0], panel.prices.index[0])
    decisions = mh.walk_forward(feats, labels, start)
    path = ml.store(name, decisions, {
        "spec": spec, "level1": level1_name, "horizon_hours": mh.HORIZON_HOURS,
        "round_trip_cost": cost, "model": "StandardScaler + LogisticRegression, pooled",
        "first": decisions.index[0], "last": decisions.index[-1]})
    today = ", ".join(f"{c} {'hedge' if bool(decisions[c].iloc[-1]) else 'no hedge'}"
                      for c in coins)
    return Built(name, path, decisions.index[0], decisions.index[-1], f"latest: {today}")


def build_all(end: date | None = None, config: Path = CONFIG) -> list[dict]:
    """Everything in `night/detectors.yaml`, in order. A failed build is
    reported and the rest go on (a level-2 build whose level 1 failed then
    reads yesterday's level-1 file -- said in its line)."""
    plan = yaml.safe_load(config.read_text(encoding="utf-8"))
    out = []
    if plan.get("regimes", True):
        try:
            from qlab.regimes import build

            series = build()
            out.append({"name": "regimes (BTC labels)", "ok": True,
                        "note": f"closes to {series.labels.index[-1]:%Y-%m-%d}"})
        except Exception as exc:  # noqa: BLE001 - reported; older labels stay
            out.append({"name": "regimes (BTC labels)", "ok": False,
                        "note": f"{type(exc).__name__}: {exc}"})
    steps = [("level1", e) for e in plan.get("level1") or []]
    steps += [("persistent", e) for e in plan.get("persistent") or []]
    steps += [(e["kind"], e) for e in plan.get("level2") or []]
    for kind, e in steps:
        try:
            if kind == "level1":
                b = level1(e["name"], returns_only=bool(e.get("returns_only", False)),
                           window=int(e.get("window", 30)),
                           market_spec=e.get("market_spec", "specs/trend-binance-cap.yaml"),
                           end=end)
            elif kind == "persistent":
                b = persistent(e["name"], source=e["from"], penalty=float(e["penalty"]))
            elif kind == "meta":
                b = meta(e["name"], spec=e["spec"], level1_name=e["level1"], end=end)
            elif kind == "meta-hedge":
                b = meta_hedge(e["name"], spec=e["spec"], level1_name=e["level1"], end=end)
            else:
                raise ValueError(f"unknown build kind {kind!r}")
            out.append({"name": f"{kind} {b.name}", "ok": True,
                        "note": f"{b.first:%Y-%m-%d}..{b.last:%Y-%m-%d}; {b.note}"})
        except Exception as exc:  # noqa: BLE001 - reported; yesterday's file stays
            out.append({"name": f"{kind} {e.get('name')}", "ok": False,
                        "note": f"{type(exc).__name__}: {exc}"})
    return out


__all__ = ["Built", "build_all", "level1", "meta", "meta_hedge", "persistent"]
