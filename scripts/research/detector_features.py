"""Which features and which model recognise the current regime (docs/TASKS.md
T39; owner, 2026-10-04: «разберись с моделью — возможно набор фич не тот»).

Development only: walk-forward predictions from 2020-04 to 2023-12, scored
against the testing labels. 2024-01 on is the HOLDOUT and is never predicted
here; the one chosen variant is scored on it once, by `--holdout NAME`.

Usage: uv run python scripts/research/detector_features.py MARKET_DIR [--holdout NAME]
(MARKET_DIR holds mkt_{prices,funding,tradeable,volume}.parquet, the Binance
daily market panel stamped by close time).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qlab import regime_detect as rd
from qlab.regimes import REGIMES, load, market_closes

DEV_START, DEV_END = pd.Timestamp("2020-04-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-12-31", tz="UTC")


def groups(btc: pd.Series, m: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Candidate feature groups, each from data up to its row's close only."""
    prices, funding, alive, volume = m["prices"], m["funding"], m["tradeable"], m["volume"]
    daily = np.log(btc).diff()
    g: dict[str, dict[str, pd.Series]] = {}
    g["ret"] = {f"ret_{w}": btc / btc.shift(w) - 1 for w in (7, 14, 30, 60, 90)}
    g["short_ret"] = {f"ret_{w}": btc / btc.shift(w) - 1 for w in (1, 3, 5, 10)}
    g["vol"] = {f"vol_{w}": daily.rolling(w).std() for w in (7, 30)}
    g["drawdown"] = {"drawdown_90": btc / btc.rolling(90).max() - 1}
    lo, hi = btc.rolling(30).min(), btc.rolling(30).max()
    g["range"] = {"range_pos_30": (btc - lo) / (hi - lo), "from_high_30": btc / hi - 1,
                  "from_low_30": btc / lo - 1}
    ma20, ma50 = btc.rolling(20).mean(), btc.rolling(50).mean()
    g["ma"] = {"ma20": btc / ma20 - 1, "ma50": btc / ma50 - 1,
               "ma20_slope": ma20 / ma20.shift(5) - 1}
    down = daily.where(daily < 0, 0.0)
    g["shape"] = {"downvol_30": down.rolling(30).std(), "skew_30": daily.rolling(30).skew()}
    rets = prices.pct_change(fill_method=None).where(alive)
    breadth = {}
    for w in (7, 30):
        up = (prices / prices.shift(w) - 1) > 0
        counted = alive & prices.shift(w).notna()
        breadth[f"breadth_{w}"] = ((up & counted).sum(axis=1)
                                   / counted.sum(axis=1).replace(0, np.nan))
    g["breadth"] = breadth
    ew = rets.mean(axis=1)
    g["alts"] = {"ew_ret_7": ew.rolling(7).sum(), "ew_ret_30": ew.rolling(30).sum(),
                 "eth_btc_30": (prices["ETH"] / prices["BTC"]).pct_change(30, fill_method=None)}
    fmean = funding.where(alive).mean(axis=1)
    g["funding"] = {"funding_7": fmean.rolling(7).mean()}
    g["btc_funding"] = {"btc_funding_7": funding["BTC"].rolling(7).mean(),
                        "btc_funding_chg": funding["BTC"].rolling(7).mean()
                        - funding["BTC"].rolling(30).mean()}
    g["lowcorr"] = _least_correlated(btc, prices, alive)
    vol_usd = (volume["BTC"] * prices["BTC"])
    g["volume"] = {"btc_volume_7_30": vol_usd.rolling(7).mean() / vol_usd.rolling(30).mean()}
    return {k: pd.DataFrame(v).reindex(btc.index) for k, v in g.items()}


def _least_correlated(btc: pd.Series, prices: pd.DataFrame, alive: pd.DataFrame
                      ) -> dict[str, pd.Series]:
    """Owner, 2026-10-04: «может есть смысл взять самую не коррелированную
    монету из 20ки по капитализации». Each day: among the 20 largest coins by
    market cap known that day (CoinMarketCap weekly snapshots, stablecoins
    out, BTC out), the one whose daily returns correlated least with BTC's
    over the past 90 days; its 7- and 30-day returns."""
    from qlab.data.sources.coinmarketcap import as_of_frame, load_history

    history = load_history()
    cols = list(prices.columns)
    rank = as_of_frame(history, prices.index, cols, "cmcRank")
    stable = as_of_frame(history, prices.index, cols, "is_stablecoin").fillna(False).astype(bool)
    rets = prices.pct_change(fill_method=None).where(alive)
    corr = rets.rolling(90, min_periods=60).corr(rets["BTC"])
    eligible = (rank <= 20) & ~stable & alive & corr.notna()
    eligible["BTC"] = False
    masked = corr.where(eligible)
    has = masked.notna().any(axis=1)
    pick = pd.Series(np.nan, index=masked.index, dtype=object)
    pick[has] = masked[has].idxmin(axis=1)
    r7 = prices / prices.shift(7) - 1
    r30 = prices / prices.shift(30) - 1
    def take(frame: pd.DataFrame) -> pd.Series:
        return pd.Series([frame.at[t, c] if isinstance(c, str) else np.nan
                          for t, c in pick.items()], index=pick.index)
    print("least-correlated picks (share of days):",
          pick.value_counts(normalize=True).head(6).round(2).to_dict(), flush=True)
    return {"lowcorr_ret_7": take(r7), "lowcorr_ret_30": take(r30),
            "lowcorr_corr": corr.where(eligible).min(axis=1)}


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "boosting": lambda: HistGradientBoostingClassifier(random_state=0),
        "ridge_return": lambda: make_pipeline(StandardScaler(), Ridge()),
    }


def walk(btc: pd.Series, feats: pd.DataFrame, model: str, start: pd.Timestamp,
         end: pd.Timestamp) -> pd.Series:
    """Monthly refits on labels known a full embargo before the month
    (`qlab.regime_detect.walk_forward`); `ridge_return` regresses the
    centered return itself and labels it with the terciles known then."""
    make = _models()[model]
    feats = feats.dropna()
    out = []
    for month in pd.date_range(start, end, freq="MS", tz="UTC"):
        nxt = month + pd.offsets.MonthBegin(1)
        cutoff = month - pd.Timedelta(days=rd.EMBARGO_DAYS)
        seen = btc[btc.index <= month]
        around = (seen.shift(-rd.HALF) / seen.shift(rd.HALF) - 1).dropna()
        lo, hi = around.quantile([1 / 3, 2 / 3]).to_list()
        around = around[around.index <= cutoff]
        train = feats.index.intersection(around.index)
        labels = pd.Series(np.where(around >= hi, "bull", np.where(around < lo, "bear", "flat")),
                           index=around.index)
        target = feats[(feats.index >= month) & (feats.index < min(nxt, end))]
        if target.empty or set(labels.loc[train]) != set(REGIMES):
            continue
        if model == "ridge_return":
            fit = make().fit(feats.loc[train], around.loc[train])
            r = fit.predict(target)
            pred = np.where(r >= hi, "bull", np.where(r < lo, "bear", "flat"))
        else:
            pred = make().fit(feats.loc[train], labels.loc[train]).predict(target)
        out.append(pd.Series(pred, index=target.index))
    return pd.concat(out)


def score(pred: pd.Series, labels: pd.Series) -> dict[str, float]:
    a = rd.accuracy(pred, labels)
    return {"acc": a["accuracy"], "bear_rec": a["bear_recall"], "bear_prec": a["bear_precision"],
            "flat_rec": a["flat_recall"]}


def main() -> None:
    market_dir = Path(sys.argv[1])
    btc = market_closes()
    labels = load().labels
    m = {k: pd.read_parquet(market_dir / f"mkt_{k}.parquet")
         for k in ("prices", "funding", "tradeable", "volume")}
    g = groups(btc, m)
    current = ["ret", "vol", "drawdown", "breadth", "funding"]  # the stored detector's set
    everything = list(g)

    def frame(names: list[str]) -> pd.DataFrame:
        return pd.concat([g[n] for n in names], axis=1)

    if "--holdout" in sys.argv:
        name = sys.argv[sys.argv.index("--holdout") + 1]
        names, model = CHOICES[name]
        for label, (n, mdl) in {"chosen": (names, model), "stored": (current, "logistic")}.items():
            pred = walk(btc, frame(n), mdl, DEV_END, HOLDOUT_END)
            print(f"holdout {label:<7} {mdl:<12} {score(pred, labels)}")
        from qlab.strategies.detectors import direction_terciles
        simple = direction_terciles(btc)
        simple = simple[(simple.index >= DEV_END)]
        print(f"holdout simple  trailing-30d {score(simple, labels)}")
        return

    rows = []
    round2 = "--round2" in sys.argv

    def run(tag: str, names: list[str], model: str = "logistic") -> None:
        s = score(walk(btc, frame(names), model, DEV_START, DEV_END), labels)
        rows.append({"variant": tag, "model": model, **s})
        print(f"{tag:<34} {model:<12} " + " ".join(f"{k} {v:.3f}" for k, v in s.items()),
              flush=True)

    if round2:
        for names in (["ret"], ["ret", "ma"], ["ret", "range"], ["ret", "lowcorr"],
                      current + ["lowcorr"], ["lowcorr"]):
            run(" + ".join(names), names)
        return
    run("current set", current)
    for name in everything:
        run(f"only {name}", [name])
    for name in current:
        run(f"current without {name}", [n for n in current if n != name])
    for name in [n for n in everything if n not in current]:
        run(f"current + {name}", current + [name])
    run("everything", everything)
    for model in ("boosting", "ridge_return"):
        run("current set", current, model)
        run("everything", everything, model)
    pd.DataFrame(rows).to_csv(market_dir / "detector_features_dev.csv", index=False)


CHOICES: dict[str, tuple[list[str], str]] = {
    # Chosen 2026-10-04 from the development table (2020-04 .. 2023-12), before
    # the holdout was predicted: the best and the simplest -- BTC's returns over
    # 7-90 days alone (0.607 against 0.565 for the stored set; adding market
    # breadth, funding, the least-correlated top-20 coin or everything lowered it).
    "ret-only": (["ret"], "logistic"),
}
"""Filled in AFTER the development table is read, before --holdout is run."""

if __name__ == "__main__":
    main()
