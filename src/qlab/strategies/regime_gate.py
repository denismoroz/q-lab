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

_REQUIRED = ("inner_code_ref", "inner_params", "trade_in")


class RegimeGate:
    name = "regime-gate"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        missing = [k for k in _REQUIRED if k not in params]
        if missing:
            raise ValueError(f"RegimeGate needs params {missing}; none has a default")
        trade_in = set(params["trade_in"])  # type: ignore[arg-type]
        detector = str(params.get("detector", "direction_terciles"))
        allowed = {"trade"} if detector.startswith("meta:") else {"bull", "flat", "bear"}
        unknown = trade_in - allowed
        if unknown or not trade_in:
            raise ValueError(f"trade_in must name {sorted(allowed)} for detector {detector!r}; "
                             f"got {trade_in}")

        from qlab.pipeline.evaluate import resolve_strategy
        from qlab.strategies.detectors import causal_labels, market_closes

        if detector == "direction_terciles":
            closes = market_closes()
            if closes is None:
                raise ValueError("RegimeGate needs BTC closes: run `qlab regimes build` first")
            labels = causal_labels(closes)
        elif detector.startswith("predictions:"):
            from qlab.regime_detect import load_predictions

            labels = load_predictions(detector.split(":", 1)[1])["label"]
        elif detector.startswith("meta:"):
            from qlab.meta_label import load_decisions

            trade = load_decisions(detector.split(":", 1)[1])["trade"]
            labels = trade.map({True: "trade", False: "cash"})
        else:
            raise ValueError(f"unknown detector {detector!r} (direction_terciles, "
                             "predictions:<name>, meta:<name>)")
        inner = resolve_strategy(str(params["inner_code_ref"]))
        weights = inner.target_weights(panel, dict(params["inner_params"]))  # type: ignore[arg-type]
        # The regime known at bar t: the last BTC day closed at or before t.
        known = labels.dropna()
        at_bar = known.reindex(panel.prices.index, method="ffill")
        on = at_bar.isin(trade_in).to_numpy()
        return weights.mul(on, axis=0)


__all__ = ["RegimeGate"]
