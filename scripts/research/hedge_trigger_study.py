"""Hedge on/off points of Bv2: the production rule against 169 declared variants per coin set
(docs/research/hedge-trigger-protocol-2026-11-04.md, owner 2026-11-04).

Two phases, so the choice is written before anything about the last third is computed:

    uv run python scripts/research/hedge_trigger_study.py prepare   # data, books, features
    uv run python scripts/research/hedge_trigger_study.py choose    # first 2/3 only -> choice.json
    uv run python scripts/research/hedge_trigger_study.py judge     # last third, book replays

The raw cache is read only (`--raw`, default the main checkout's data/raw). Intermediate tables
go to `--work` (a scratch folder); the choice and the trial table to docs/research.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qlab import hedge_study as hs
from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel
from qlab.harness.run import run_backtest
from qlab.knowable_coins import read_daily_value

START, END = "2019-09-08", "2026-10-30"
CUT = hs.split_date(START, END)
SWITCH_COST = (7.5 + 0.9) / 1e4  # specs/bv2-binance.yaml costs, per switch of the perp leg
COSTS = CostModel(taker_fee_bps=7.5, slippage_bps=0.9)
SPEC = Path("specs/bv2-binance.yaml")
SET_A = ("BTC", "ETH", "SOL", "AVAX")
PER_COIN_CAPITAL = 257.0 / 4
TOP_K, WINDOW_DAYS = 10, 30
STICKY = 12
DOCS = Path("docs/research")
CHOICE = DOCS / "hedge-trigger-study-2026-11-04-choice.json"
TRIALS = DOCS / "hedge-trigger-study-2026-11-04-trials.csv"
MIN_ACTIVITY = 0.05


@dataclass
class CoinData:
    coin: str
    index: pd.DatetimeIndex
    stepped: np.ndarray  # positions on the full index where the coin's book was stepped
    own: np.ndarray  # book's own wish, on stepped bars
    in_pos: np.ndarray  # hedge open after the bar, on stepped bars
    perp: np.ndarray  # on stepped bars
    funding: np.ndarray  # on stepped bars
    mask: np.ndarray  # stepped bar belongs to the set (set b: the day is a top-10 day)
    feats: pd.DataFrame  # full index
    priced: np.ndarray  # full index: the perp has a price
    spot_own: pd.Series  # full index
    events: pd.DataFrame  # production book's events, labelled


def _params() -> dict:
    return dict(yaml.safe_load(SPEC.read_text())["params"])


def _coin_data(coin, rec, panel, spot_own, btc, mask_full) -> CoinData:
    st = rec.stepped[coin].to_numpy()
    at = np.flatnonzero(st)
    perp = panel.prices[coin]
    feats = hs.features(perp, spot_own, panel.funding[coin], panel.volume[coin],
                        None if coin == "BTC" else btc)
    ev = hs.label_events(hs.events(rec, coin), spot_own)
    in_set = pd.Series(mask_full, index=panel.prices.index)
    ev = ev[in_set.reindex(ev["decision"]).to_numpy(dtype=bool)].reset_index(drop=True)
    return CoinData(coin=coin, index=panel.prices.index, stepped=at,
                    own=rec.own[coin].to_numpy()[at], in_pos=rec.in_pos[coin].to_numpy()[at],
                    perp=perp.to_numpy()[at], funding=panel.funding[coin].to_numpy()[at],
                    mask=mask_full[at], feats=feats, priced=perp.notna().to_numpy(),
                    spot_own=spot_own, events=ev)


# ── set (b): the code's top 10 ──────────────────────────────────────────


def _spot_daily_presence(folder: Path) -> pd.DataFrame:
    """Day x coin: the spot traded that day, from the hourly cache (there is no daily spot
    cache). A coin whose cached spot starts after the cache's own start (2019-09-08) has an
    unknown listing before it: it counts as present there, so it competes in the rank, and its
    days without hourly data are dropped later (protocol, section 1)."""
    stem = re.compile(r"^(?P<inst>.+)-SPOT__(?P<s>\d{4}-\d{2}-\d{2})__(?P<e>\d{4}-\d{2}-\d{2})$")
    files: dict[str, list[Path]] = {}
    starts: dict[str, str] = {}
    for path in folder.glob("*-SPOT__*.parquet"):
        m = stem.match(path.stem)
        if not m:
            continue
        files.setdefault(m["inst"], []).append(path)
        starts[m["inst"]] = min(starts.get(m["inst"], "9999"), m["s"])
    cols = {}
    full = pd.date_range(START, END, freq="D")
    for coin, paths in files.items():
        days = set()
        for path in paths:
            frame = pd.read_parquet(path, columns=["price"]).dropna()
            idx = pd.DatetimeIndex(frame.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            days |= set(idx.tz_localize(None).normalize())
        present = pd.Series(full.isin(sorted(days)), index=full)
        if starts[coin] > START:
            present[full < pd.Timestamp(starts[coin])] = True
        cols[coin.upper()] = present.astype(float).where(present)
    return pd.DataFrame(cols)


def top_table(raw: Path) -> pd.DataFrame:
    """Day x coin: in the first 10 by the 30-day mean perp turnover of the previous days, among
    coins whose perp and spot both traded on each of those 30 days."""
    value = read_daily_value(raw / "binance" / "1d")
    spot = _spot_daily_presence(raw / "binance" / "1h")
    full = pd.date_range(value.index.min(), value.index.max(), freq="D")
    value = value.reindex(full)
    spot = spot.reindex(index=full, columns=value.columns)
    trailing = value.rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).mean().shift(1)
    spot_ok = spot.notna().astype(float).rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).sum(
    ).shift(1) >= WINDOW_DAYS
    ranked = trailing.where(spot_ok).rank(axis=1, ascending=False, method="min")
    return (ranked <= TOP_K).fillna(False)


def prepare(raw: Path, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    params = _params()
    t0 = time.time()
    # set (a)
    sp = hs.load_panel(SET_A, START, END, raw)
    rec = hs.run_book(sp.panel, params)
    btc = sp.panel.prices["BTC"]
    set_a = [_coin_data(c, rec, sp.panel, sp.spot_own[c], btc,
                        rec.stepped[c].to_numpy()) for c in SET_A]
    with open(work / "set_a.pkl", "wb") as fh:
        pickle.dump({"coins": set_a, "panel": sp.panel, "spot_own": sp.spot_own}, fh)
    print(f"set a ready {time.time() - t0:.0f}s", flush=True)
    # set (b)
    top = top_table(raw)
    day_coins = top.stack()
    day_coins = day_coins[day_coins]
    wanted = sorted({c for _, c in day_coins.index})
    window = (top.index >= pd.Timestamp(START)) & (top.index <= pd.Timestamp(END))
    total_coin_days = int(top[window].to_numpy().sum())
    have = [c for c in wanted
            if hs.read_history(raw, c, pd.Timestamp(END, tz="UTC")) is not None
            and hs.read_history(raw, f"{c}-SPOT", pd.Timestamp(END, tz="UTC")) is not None]
    big = hs.load_panel(have, START, END, raw)
    print(f"set b panel {len(have)} coins {time.time() - t0:.0f}s", flush=True)
    index = big.panel.prices.index
    days = index.tz_localize(None).floor("D")
    coins_b, covered_days = [], 0
    p_one = {k: v for k, v in params.items() if k != "coins"}
    for c in have:
        sub = _sub_panel(big.panel, c)
        rec_c = hs.run_book(sub, {**p_one, "coins": [c], "capital_usd": PER_COIN_CAPITAL})
        in_top = top[c].reindex(days).fillna(False).to_numpy(dtype=bool) if c in top else \
            np.zeros(len(index), dtype=bool)
        mask = in_top & rec_c.stepped[c].to_numpy()
        stepped_days = set(days[mask])
        covered_days += len(stepped_days)
        if not mask.any():
            continue
        coins_b.append(_coin_data(c, rec_c, sub, big.spot_own[c], btc.reindex(index), mask))
    meta = {"wanted_coins": len(wanted), "loaded_coins": len(have), "used_coins": len(coins_b),
            "top_coin_days": total_coin_days, "covered_coin_days": covered_days,
            "missing_hourly": sorted(set(wanted) - set(have))}
    with open(work / "set_b.pkl", "wb") as fh:
        pickle.dump({"coins": coins_b, "panel": big.panel, "spot_own": big.spot_own,
                     "meta": meta, "top": top}, fh)
    print(json.dumps(meta)[:600], f"{time.time() - t0:.0f}s", flush=True)


def _sub_panel(panel, coin):
    cols = [coin, f"{coin}-SPOT"]
    return MarketPanel(snapshot_id=f"hedge-study-{coin}", prices=panel.prices[cols],
                          funding=panel.funding[cols], tradeable=panel.tradeable[cols],
                          volume=panel.volume[cols], high=panel.high[cols],
                          meta={**panel.meta, "no_funding_instruments": [f"{coin}-SPOT"]})


# ── the vectorised evaluation ───────────────────────────────────────────


def variant_hedged(v: hs.Variant | None, cd: CoinData) -> np.ndarray:
    """Hedge held over (s, s+1] on each stepped bar under the variant (None: production)."""
    start = bool(cd.in_pos[0]) if len(cd.in_pos) else False
    if v is None:
        final = hs.gate_wish(cd.own, None, None, start)
    elif v.kind == "trigger":
        if v.family == "F7" and cd.coin == "BTC":
            final = hs.gate_wish(cd.own, None, None, start)
        else:
            cond = v.condition(cd.feats)
            final = hs.trigger_wish(cond, cd.priced, STICKY)[cd.stepped] > 0.5
    else:
        if v.family == "F7" and cd.coin == "BTC":
            veto = np.zeros(len(cd.own), dtype=bool)
        else:
            veto = v.condition(cd.feats)[cd.stepped]
        final = hs.gate_wish(cd.own, veto if v.kind == "veto_on" else None,
                             veto if v.kind == "veto_off" else None, start)
    out = np.zeros(len(final), dtype=bool)
    out[0] = start
    out[1:] = final[:-1]
    return out


def variant_events(hedged: np.ndarray, cd: CoinData) -> pd.DataFrame:
    at = cd.index[cd.stepped]
    flips = np.flatnonzero(hedged[1:] != hedged[:-1]) + 1
    rows = [{"fill": at[j], "decision": at[j - 1], "kind": "on" if hedged[j] else "off",
             "liquidation": False} for j in flips if cd.mask[j - 1]]
    ev = pd.DataFrame(rows, columns=["fill", "decision", "kind", "liquidation"])
    return hs.label_events(ev, cd.spot_own) if len(ev) else ev


def summarise(coins: list[CoinData], v, part: str) -> dict:
    """Money (% of position per coin-year), on events per coin-year, false/wrong shares, on the
    bars of `part` ("first": before CUT, "last": from CUT)."""
    money = hours = 0.0
    evs = []
    for cd in coins:
        hedged = variant_hedged(v, cd)
        m = hs.hedge_money(cd.perp, cd.funding, hedged, SWITCH_COST)
        at = cd.index[cd.stepped]
        sel = cd.mask & ((at < CUT) if part == "first" else (at >= CUT))
        money += float(m[sel].sum())
        hours += float(sel.sum())
        ev = variant_events(hedged, cd)
        if len(ev):
            ev = ev[(ev["decision"] < CUT) if part == "first" else (ev["decision"] >= CUT)]
            evs.append(ev.assign(coin=cd.coin))
    ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame(
        columns=["fill", "decision", "kind", "liquidation"])
    years = hours / hs.HOURS_PER_YEAR
    out = {"money_pct_coin_year": 100 * money / years if years else float("nan"),
           "on_per_coin_year": float((ev["kind"] == "on").sum() / years) if years else np.nan,
           "coin_years": years}
    for h in hs.HORIZONS_DAYS:
        out[f"false_on_{h}d"], out[f"n_on_{h}d"] = hs.false_share(ev, "on", h)
        out[f"wrong_off_{h}d"], out[f"n_off_{h}d"] = hs.false_share(ev, "off", h)
    return out


def activity(coins: list[CoinData], v: hs.Variant) -> float:
    """Share of the production events of the variant's kind (decided before CUT) whose decision
    bar meets the filter's condition."""
    hit = total = 0
    kind = "on" if v.kind == "veto_on" else "off"
    for cd in coins:
        ev = cd.events
        ev = ev[(ev["kind"] == kind) & (ev["decision"] < CUT) & ~ev["liquidation"].astype(bool)]
        if not len(ev):
            continue
        total += len(ev)
        if v.family == "F7" and cd.coin == "BTC":
            continue
        cond = pd.Series(v.condition(cd.feats), index=cd.index)
        hit += int(cond.reindex(ev["decision"]).sum())
    return hit / total if total else 0.0


def _load(work: Path, name: str) -> dict:
    with open(work / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def choose_phase(work: Path) -> None:
    choice = {"cut": str(CUT.date()), "sets": {}}
    rows = []
    for name in ("set_a", "set_b"):
        data = _load(work, name)
        coins = data["coins"]
        variants = hs.declared_variants()
        prod = summarise(coins, None, "first")
        money, eligible = {}, {}
        for v in variants:
            s = summarise(coins, v, "first")
            act = activity(coins, v) if v.kind != "trigger" else float("nan")
            ok = v.kind == "trigger" or act >= MIN_ACTIVITY
            money[v.name] = s["money_pct_coin_year"]
            eligible[v.name] = ok
            rows.append({"set": name, "family": v.family, "kind": v.kind, "variant": v.name,
                         "activity_first": act, "eligible": ok,
                         **{f"{k}_first": x for k, x in s.items()}})
        series = {k: pd.Series([x], index=[CUT - pd.Timedelta(days=1)]) for k, x in money.items()}

        def pick(names, series=series, eligible=eligible):
            best, value = hs.choose({k: series[k] for k in names}, CUT, eligible)
            return {"variant": best, "money_first": value}

        fams = {}
        for fam in sorted({v.family for v in variants}):
            for group in ("filter", "trigger"):
                names = [v.name for v in variants if v.family == fam
                         and (v.kind == "trigger") == (group == "trigger")]
                fams[f"{fam} {group}"] = pick(names)
        choice["sets"][name] = {
            "production_first": prod,
            "best_filter": pick([v.name for v in variants if v.kind != "trigger"]),
            "best_trigger": pick([v.name for v in variants if v.kind == "trigger"]),
            "per_family": fams,
            "trials": len(variants),
            "meta": data.get("meta", {}),
        }
        print(name, json.dumps(choice["sets"][name]["best_filter"]),
              json.dumps(choice["sets"][name]["best_trigger"]), flush=True)
    CHOICE.write_text(json.dumps(choice, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(work / "trials_first.csv", index=False)


# ── judge: last third, book replays ─────────────────────────────────────


def _full_gates(v: hs.Variant, coins: list[CoinData]):
    gates, wishes = {}, {}
    for cd in coins:
        if v.kind == "trigger":
            if v.family == "F7" and cd.coin == "BTC":
                continue  # the coin keeps the book's own rule
            wishes[cd.coin] = hs.trigger_wish(v.condition(cd.feats), cd.priced, STICKY)
        else:
            if v.family == "F7" and cd.coin == "BTC":
                continue
            cond = v.condition(cd.feats)
            gates[cd.coin] = (cond, None) if v.kind == "veto_on" else (None, cond)
    return gates, wishes


def _book_money(panel, params, capital, gates=None, wishes=None, coins_all=None):
    """Run the book (with an optional gate or external wishes) and the stand on it: cap money
    per part, switches and fees on the last third."""
    if wishes:
        nan = np.full(len(panel.prices.index), np.nan)
        wishes = {c: wishes.get(c, nan) for c in coins_all}
    rec = hs.run_book(panel, params, gates=gates or None, wishes=wishes or None)
    base = (rec.run.book_equity / capital).rename("capital_base")
    res = run_backtest(panel, rec.run.weights, COSTS, panel.funding, capital_base=base,
                       capital_reinvested=False)
    r = res.capital_return
    out = {}
    for part, sel in (("first", r.index < CUT), ("last", r.index >= CUT)):
        n = int(sel.sum())
        out[f"cap_ann_{part}"] = float(r[sel].sum() * hs.HOURS_PER_YEAR / n * 100)
        curve = r[sel].cumsum()
        out[f"cap_max_dd_{part}"] = float((curve - curve.cummax().clip(lower=0)).min() * 100)
        out[f"fees_{part}"] = float((res.cost * base.reindex(res.cost.index).ffill())[sel].sum()
                                    * 100)
    ons = 0
    for c in rec.stepped.columns:
        ev = hs.events(rec, c)
        ons += int(((ev["kind"] == "on") & (ev["decision"] >= CUT)).sum())
    out["switches_on_last"] = ons
    return out, rec


def _book_events_last(rec, coins: list[CoinData]) -> pd.DataFrame:
    evs = []
    for cd in coins:
        if cd.coin not in rec.stepped.columns:
            continue
        ev = hs.label_events(hs.events(rec, cd.coin), cd.spot_own)
        in_set = pd.Series(False, index=cd.index)
        in_set.iloc[cd.stepped[cd.mask]] = True
        ev = ev[in_set.reindex(ev["decision"]).to_numpy(dtype=bool) & (ev["decision"] >= CUT)]
        evs.append(ev.assign(coin=cd.coin))
    return pd.concat(evs, ignore_index=True)


def _per_coin_book_events(v, coins: list[CoinData], panel, params) -> pd.DataFrame:
    """Set (b): each coin's own book with the variant, its events on top-10 days."""
    p_one = {k: x for k, x in params.items() if k != "coins"}
    evs = []
    for cd in coins:
        sub = _sub_panel(panel, cd.coin)
        g, w = _full_gates(v, [cd]) if v is not None else ({}, {})
        rec = hs.run_book(sub, {**p_one, "coins": [cd.coin], "capital_usd": PER_COIN_CAPITAL},
                          gates=g or None, wishes=w or None)
        evs.append(_book_events_last(rec, [cd]))
    return pd.concat(evs, ignore_index=True)


def judge_phase(work: Path) -> None:
    choice = json.loads(CHOICE.read_text(encoding="utf-8"))
    params = _params()
    variants = {v.name: v for v in hs.declared_variants()}
    out = {"cut": choice["cut"], "sets": {}}
    rows = []
    first = pd.read_csv(work / "trials_first.csv")
    for name in ("set_a", "set_b"):
        data = _load(work, name)
        coins, panel = data["coins"], data["panel"]
        res = {"production": {"first": summarise(coins, None, "first"),
                              "last": summarise(coins, None, "last")}}
        # every trial on the last third, for the table (chosen beforehand, see CHOICE)
        for v in variants.values():
            s = summarise(coins, v, "last")
            rows.append({"set": name, "variant": v.name, **{f"{k}_last": x for k, x in s.items()}})
        # base rate: an "on" at a random hour of the set
        rnd = []
        for cd in coins:
            s = cd.spot_own.ffill(limit=24)
            fwd = (s.shift(-7 * 24) / s - 1).to_numpy()[cd.stepped]
            at = cd.index[cd.stepped]
            for part in ("first", "last"):
                sel = cd.mask & ((at < CUT) if part == "first" else (at >= CUT)) & ~np.isnan(fwd)
                rnd.append((part, int((fwd[sel] >= 0).sum()), int(sel.sum())))
        res["random_hour_no_fall_7d"] = {
            part: sum(a for p, a, _ in rnd if p == part) / max(1, sum(n for p, _, n in rnd
                                                                      if p == part))
            for part in ("first", "last")}
        # production book events (truth) and their episode money
        if name == "set_a":
            capital, bparams, coins_all = float(params["capital_usd"]), params, list(SET_A)
        else:
            capital = PER_COIN_CAPITAL * TOP_K
            coins_all = [cd.coin for cd in coins]
            bparams = {**{k: x for k, x in params.items() if k != "coins"},
                       "capital_usd": capital,
                       "universe": {"rule": "top_liquid", "n": TOP_K, "rebalance": "1d",
                                    "by": "perp_dollar_volume", "window_days": WINDOW_DAYS,
                                    "candidates": coins_all}}
        prod_money, prod_rec = _book_money(panel, bparams, capital, coins_all=coins_all)
        if name == "set_a":
            prod_ev = _book_events_last(prod_rec, coins)
        else:
            prod_ev = _per_coin_book_events(None, coins, panel, params)
        res["production"]["book"] = prod_money
        res["production"]["book_false_on_7d"] = hs.false_share(prod_ev, "on")
        res["production"]["book_wrong_off_7d"] = hs.false_share(prod_ev, "off")
        allev = pd.concat([cd.events.assign(coin=cd.coin) for cd in coins], ignore_index=True)
        on = allev[allev["kind"] == "on"].dropna(subset=["money"])
        years = {"first": (CUT - pd.Timestamp(START, tz="UTC")).days / 365,
                 "last": (pd.Timestamp(END, tz="UTC") - CUT).days / 365}
        for part, sel in (("first", on["decision"] < CUT), ("last", on["decision"] >= CUT)):
            m = on[sel]
            res["production"][f"episode_{part}"] = {
                "n": int(len(m)),
                "mean_money_pct": float(100 * m["money"].mean() / capital),
                "share_losing": float((m["money"] < 0).mean()),
                "sum_money_pct_per_year": float(100 * m["money"].sum() / capital / years[part]),
                "false_7d_of_losing": float(((m["ret_7d"] >= 0) & (m["money"] < 0)).sum()
                                            / max(1, (m["money"] < 0).sum())),
            }
        sc = choice["sets"][name]
        picks = {"best_filter": sc["best_filter"]["variant"],
                 "best_trigger": sc["best_trigger"]["variant"]}
        for fam, p in sc["per_family"].items():
            if p["variant"]:
                picks[f"family {fam}"] = p["variant"]
        replay = {}
        done = {}
        for label, vname in picks.items():
            if vname in done:
                replay[label] = done[vname]
                continue
            v = variants[vname]
            g, w = _full_gates(v, coins)
            money, rec = _book_money(panel, bparams, capital, gates=g, wishes=w,
                                     coins_all=coins_all)
            entry = {"variant": vname, "book": money,
                     "model_first": summarise(coins, v, "first"),
                     "model_last": summarise(coins, v, "last")}
            if label in ("best_filter", "best_trigger"):
                ev = (_book_events_last(rec, coins) if name == "set_a"
                      else _per_coin_book_events(v, coins, panel, params))
                entry["book_false_on_7d"] = hs.false_share(ev, "on")
                entry["book_wrong_off_7d"] = hs.false_share(ev, "off")
                entry["bootstrap_on_7d"] = hs.bootstrap_drop(prod_ev, ev, "on")
                entry["bootstrap_off_7d"] = hs.bootstrap_drop(prod_ev, ev, "off")
                for h in hs.HORIZONS_DAYS:
                    entry[f"book_false_on_{h}d"] = hs.false_share(ev, "on", h)
                    res["production"][f"book_false_on_{h}d"] = hs.false_share(prod_ev, "on", h)
            done[vname] = entry
            replay[label] = entry
            print(name, label, vname, json.dumps(money), flush=True)
        res["replay"] = replay
        res["meta"] = data.get("meta", {})
        out["sets"][name] = res
    last = pd.DataFrame(rows)
    table = first.merge(last, on=["set", "variant"], how="left")
    table.to_csv(TRIALS, index=False, float_format="%.5g")
    (work / "judge.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("written", TRIALS, work / "judge.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "choose", "judge"])
    ap.add_argument("--raw", type=Path, default=Path("/Users/d/prj/q-lab/data/raw"))
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    {"prepare": lambda: prepare(a.raw, a.work), "choose": lambda: choose_phase(a.work),
     "judge": lambda: judge_phase(a.work)}[a.phase]()


if __name__ == "__main__":
    main()
