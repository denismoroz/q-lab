"""Trade a strategy only in the market regimes where it is supposed to earn
(owner, 2026-10-02: «в стратегию детект режима — можно встроить?»;
docs/REGIMES.md).

`RegimeGate` wraps any strategy: at each bar it reads the regime a strategy
may know at that bar -- `qlab.strategies.detectors.direction_terciles`, BTC's 30-day return
against terciles of the days BEFORE it -- and keeps the wrapped strategy's
weights in the regimes listed in `trade_in`, holding nothing in the others
and while the regime is still unknown.

Which regimes to trade in is the spec's declared choice, taken from the
source's own claim (`regime_claim`), never from looking at how the strategy
did per regime on the data it is tested on.

`detector` (optional) names what reads the regime: `direction_terciles`
(the default above) or `predictions:<name>` -- the walk-forward regime
detector's stored predictions (`qlab.regime_detect`, docs/TASKS.md T39),
stamped by the daily close they were made at; or `meta:<name>` -- a
strategy's own level-2 decisions (`qlab.meta_label`), with `trade_in:
[trade]`.

The regime of bar t uses BTC closes up to the last day closed by t. BTC's
closes come from the stored regime build (`qlab regimes build`), outside the
panel -- like CoinMarketCap snapshots, they are point-in-time by their own
construction and test, not by the panel's look-ahead guard.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies._inner import InnerCapital

_REQUIRED = ("inner_code_ref", "inner_params", "trade_in")


def allowed_regimes(detector: str) -> set[str]:
    """The labels a detector can give and `trade_in` may name."""
    return {"trade"} if detector.startswith("meta:") else {"bull", "flat", "bear"}


def check_trade_in(trade_in: object, detector: str) -> set[str]:
    """`trade_in` as a set, refusing an empty list and labels the detector never gives."""
    wanted = set(trade_in)  # type: ignore[call-overload]
    allowed = allowed_regimes(detector)
    unknown = wanted - allowed
    if unknown or not wanted:
        raise ValueError(f"trade_in must name {sorted(allowed)} for detector {detector!r}; "
                         f"got {wanted}")
    return wanted


def regime_labels(detector: str) -> pd.Series:
    """The labels a detector gives, stamped by the daily close they were made at: BTC's
    trailing-month terciles (`direction_terciles`), the walk-forward detector's stored
    predictions (`predictions:<name>`) or a strategy's own level-2 decisions
    (`meta:<name>`, mapped to `trade` / `cash`)."""
    if detector == "direction_terciles":
        from qlab.strategies.detectors import causal_labels, market_closes

        closes = market_closes()
        if closes is None:
            raise ValueError("RegimeGate needs BTC closes: run `qlab regimes build` first")
        return causal_labels(closes)
    if detector.startswith("predictions:"):
        from qlab.regime_detect import load_predictions

        return load_predictions(detector.split(":", 1)[1])["label"]
    if detector.startswith("meta:"):
        from qlab.meta_label import load_decisions

        trade = load_decisions(detector.split(":", 1)[1])["trade"]
        return trade.map({True: "trade", False: "cash"})
    raise ValueError(f"unknown detector {detector!r} (direction_terciles, "
                     "predictions:<name>, meta:<name>)")


def open_at(index: pd.DatetimeIndex, labels: pd.Series, trade_in: set[str]):
    """Per panel bar, whether the regime known at it is one to trade in (a numpy bool array).
    The regime of bar t is the last label stamped at or before t; unknown until the first."""
    known = labels.dropna()
    at_bar = known.reindex(index, method="ffill")
    return at_bar.isin(trade_in).to_numpy()


class RegimeGate(InnerCapital):
    name = "regime-gate"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        missing = [k for k in _REQUIRED if k not in params]
        if missing:
            raise ValueError(f"RegimeGate needs params {missing}; none has a default")
        detector = str(params.get("detector", "direction_terciles"))
        trade_in = check_trade_in(params["trade_in"], detector)

        from qlab.pipeline.evaluate import resolve_strategy

        labels = regime_labels(detector)
        inner = resolve_strategy(str(params["inner_code_ref"]))
        weights = inner.target_weights(panel, dict(params["inner_params"]))  # type: ignore[arg-type]
        # The regime known at bar t: the last BTC day closed at or before t.
        on = open_at(panel.prices.index, labels, trade_in)
        return weights.mul(on, axis=0)


__all__ = ["RegimeGate", "allowed_regimes", "check_trade_in", "open_at", "regime_labels"]
