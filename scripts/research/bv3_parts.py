"""Bv3 split into parts (owner 2026-11-05: «нужно распилить её на части и пробовать улучшить
каждую под задачу»). Protocols: docs/research/bv3-parts-{0-labels,1-parts}-protocol-2026-11-05.md.

    uv run python scripts/research/bv3_parts.py --work DIR

`--work`: the folder of `hedge_trigger_study.py prepare`. Writes DIR/bv3-parts.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hedge_followup as hfu  # noqa: E402
import hedge_trigger_study as hts  # noqa: E402

from qlab import bv3_parts as bp  # noqa: E402
from qlab import earning_test  # noqa: E402
from qlab import hedge_followup as hf  # noqa: E402

CUT = hts.CUT
PURGE = pd.Timedelta(days=15)
FEE = hts.SWITCH_COST
DEFS = [(x, n) for x in (0.15, 0.25, 0.35) for n in (7, 14)]
MAIN = (0.25, 7)
PARTS_ORDER = ("A", "B", "C", "D", "E", "F")


def _lag(px, priced, days):
    return hf.lagged(px, priced, days * 24)


def coin_inputs(data: dict, btc_px: np.ndarray) -> list[dict]:
    panel = data["panel"]
    index = panel.prices.index
    out = []
    for cd in data["coins"]:
        px = panel.prices[cd.coin].to_numpy(dtype=float)
        priced = ~np.isnan(px)
        s14, s30 = _lag(px, priced, 14), _lag(px, priced, 30)
        m = np.fmin(np.where(np.isnan(s14), -np.inf, s14), s30)
        prod_phase = ~np.isnan(s30) & (m <= 0)
        ser = pd.Series(px, index=index).ffill()
        close = hf.daily_closes(panel.prices[cd.coin]).dropna()
        f = {"px": px, "priced": priced, "m": m, "prod_phase": prod_phase,
             "r30": _lag(px, priced, 30), "r60": _lag(px, priced, 60), "r90": _lag(px, priced, 90),
             "dd90": (ser / ser.rolling(90 * 24, min_periods=24).max() - 1.0).to_numpy(),
             "dd7": (ser / ser.rolling(7 * 24, min_periods=24).max() - 1.0).to_numpy(),
             "r1": _lag(px, priced, 1), "r3": _lag(px, priced, 3), "r7": _lag(px, priced, 7),
             "inert": bp.daily_to_hours(bp.rolling_inertia(close), index),
             "vr": bp.daily_to_hours(bp.rolling_vr(close), index)}
        if cd.coin == "BTC":
            f["btc_phase"] = prod_phase
        else:
            bp_ = ~np.isnan(btc_px)
            b14, b30 = _lag(btc_px, bp_, 14), _lag(btc_px, bp_, 30)
            bm = np.fmin(np.where(np.isnan(b14), -np.inf, b14), b30)
            f["btc_phase"] = ~np.isnan(b30) & (bm <= 0)
        at = cd.index[cd.stepped]
        falls = {d: bp.zigzag_falls(close, *d) for d in DEFS}
        out.append({"cd": cd, "f": f, "falls": falls,
                    "fall_pos": bp.fall_positions(falls[MAIN], at)})
    return out


# ── variants ────────────────────────────────────────────────────────────


def phase_of(f: dict, a: str) -> np.ndarray:
    pp = f["prod_phase"]
    with np.errstate(invalid="ignore"):
        if a == "prod":
            return pp
        if a in ("r30", "r60", "r90"):
            return np.nan_to_num(f[a], nan=1.0) <= 0
        if a.startswith("dd90"):
            return np.nan_to_num(f["dd90"], nan=0.0) <= -float(a.split()[1])
        if a == "btc":
            return f["btc_phase"]
        if a.startswith("inertia"):
            g = float(a.split()[1])
            return pp & ~(f["inert"] <= g)  # unknown inertia keeps the gate open
        if a == "vr":
            return pp & ~(f["vr"] <= 1.0)
    raise ValueError(a)


def entry_of(f: dict, b: str, phase: np.ndarray):
    with np.errstate(invalid="ignore"):
        if b == "now":
            return None
        if b == "r1":
            return np.nan_to_num(f["r1"], nan=1.0) <= 0
        if b == "r3":
            return np.nan_to_num(f["r3"], nan=1.0) <= 0
        if b.startswith("dd7"):
            return np.nan_to_num(f["dd7"], nan=0.0) <= -float(b.split()[1])
        if b.startswith("delay"):
            return bp.run_length(phase) >= int(b.split()[1])
    raise ValueError(b)


GRID = {
    "A": ["prod", "r30", "r60", "r90", "dd90 0.1", "dd90 0.2", "btc", "inertia 0", "inertia 0.1",
          "vr"],
    "B": ["now", "r1", "r3", "dd7 0.03", "dd7 0.05", "delay 24", "delay 72"],
    "C": ["prod", "rebound 0.05", "rebound 0.1", "rebound 0.15", "r7 up"],
    "D": ["none", "reentry 0.03", "reentry 0.05", "reentry 0.1"],
    "E": ["full", "half", "conf 0.05", "conf 0.1"],
    "F": [f"hold {h} pause {c}" for h in (0, 24, 72, 168) for c in (0, 24)],
}
PROD = {"A": "prod", "B": "now", "C": "prod", "D": "none", "E": "full", "F": "hold 0 pause 0"}


def wish_and_share(f: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray | None]:
    phase = phase_of(f, cfg["A"])
    entry = entry_of(f, cfg["B"], phase)
    c = cfg["C"]
    exit_extra = (np.nan_to_num(f["r7"], nan=-1.0) > 0) if c == "r7 up" else None
    rebound = float(c.split()[1]) if c.startswith("rebound") else None
    d = cfg["D"]
    reentry = float(d.split()[1]) if d.startswith("reentry") else None
    _, h, _, cdown = cfg["F"].split()
    parts = bp.Parts(min_hold=int(h), cooldown=int(cdown), rebound_x=rebound, reentry_y=reentry)
    wish = bp.machine(f["px"], phase, entry, exit_extra, parts)
    e = cfg["E"]
    if e == "full":
        share = None
    elif e == "half":
        share = np.full(len(wish), 0.5)
    else:
        dd = float(e.split()[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(np.isfinite(f["m"]), np.clip(-f["m"] / dd, 0.3, 1.0), 1.0)
    return wish, share


# ── evaluation on the model (one coin's stepped bars) ───────────────────


def evaluate(inputs: list[dict], cfg: dict, regime: np.ndarray, index: pd.DatetimeIndex) -> dict:
    """Every part's metric for one configuration, on the first 2/3 (whole, bull, bear) and the
    last third, pooled over the coins; plus the simplified book per unit of position."""
    subsets = {"first": index < CUT, "last": index >= CUT,
               "bull": (index < CUT - PURGE) & (regime == "bull"),
               "bear": (index < CUT - PURGE) & (regime == "bear")}
    acc = {k: {"tp": 0.0, "fall_h": 0.0, "fp": 0.0, "other_h": 0.0, "loss": [], "false": 0,
               "missed": [], "premature": 0, "falls": 0, "flip_cost": 0.0, "flips": 0,
               "coin_h": 0.0, "ons": 0} for k in subsets}
    n = len(index)
    total = {k: np.zeros(n) for k in ("book", "spot")}
    count = np.zeros(n)
    for item in inputs:
        cd, f = item["cd"], item["f"]
        st = cd.stepped
        wish, share = wish_and_share(f, cfg)
        w = wish[st] > 0.5
        h = np.zeros(len(w), dtype=bool)
        h[0] = bool(cd.in_pos[0]) if len(cd.in_pos) else False
        h[1:] = w[:-1]
        sh = np.ones(len(w)) if share is None else np.r_[share[st][0], share[st][:-1]]
        hv = h * sh
        px = cd.perp
        ret = np.zeros(len(px))
        ret[:-1] = px[1:] / px[:-1] - 1.0
        ret = np.nan_to_num(ret)
        fund = np.zeros(len(px))
        fund[:-1] = np.nan_to_num(cd.funding[1:])
        money = hv * (-ret + fund) - np.abs(np.diff(hv, prepend=0.0)) * FEE
        pos = st[cd.mask]
        total["book"][pos] += (ret + money)[cd.mask]
        total["spot"][pos] += ret[cd.mask]
        count[pos] += 1
        phase_st = phase_of(f, cfg["A"])[st]
        fm = bp.in_falls(len(st), item["fall_pos"])
        for k, sel_full in subsets.items():
            sel = sel_full[st] & cd.mask
            a = acc[k]
            tp, fh, fp, oh = bp.phase_metric(phase_st, fm, sel)
            a["tp"] += tp
            a["fall_h"] += fh
            a["fp"] += fp
            a["other_h"] += oh
            losses, false = bp.entry_metric(h, px, item["fall_pos"], sel)
            a["loss"] += losses
            a["false"] += false
            a["falls"] += len(losses)
            missed, premature = bp.exit_metric(h, px, item["fall_pos"], sel)
            a["missed"] += missed
            a["premature"] += premature
            costs = bp.flip_metric(h, px, sel, FEE)
            a["flip_cost"] += float(np.sum(costs))
            a["flips"] += len(costs)
            a["coin_h"] += float(sel.sum())
            a["ons"] += int((np.r_[False, h[1:] & ~h[:-1]] & sel).sum())
    out = {}
    for k, a in acc.items():
        cy = a["coin_h"] / 8760 if a["coin_h"] else np.nan
        tpr = a["tp"] / a["fall_h"] if a["fall_h"] else np.nan
        fpr = a["fp"] / a["other_h"] if a["other_h"] else np.nan
        out[k] = {"tpr": tpr, "fpr": fpr, "J": tpr - fpr,
                  "phase_outside_share": (a["fp"] / (a["fp"] + a["tp"])
                                          if a["fp"] + a["tp"] else None),
                  "loss_before_on": float(np.mean(a["loss"])) if a["loss"] else None,
                  "false_starts_per_fall": a["false"] / a["falls"] if a["falls"] else None,
                  "missed_rebound": float(np.mean(a["missed"])) if a["missed"] else None,
                  "premature_per_fall": a["premature"] / a["falls"] if a["falls"] else None,
                  "flip_cost_per_coin_year": 100 * a["flip_cost"] / cy if cy else None,
                  "flips_per_coin_year": a["flips"] / cy if cy else None,
                  "ons_per_coin_year": a["ons"] / cy if cy else None,
                  "falls": a["falls"]}
    mean = {k: np.where(count > 0, v / np.maximum(count, 1), 0.0) for k, v in total.items()}
    for part, sel in (("first", index < CUT), ("last", index >= CUT)):
        years = sel.sum() / 8760
        for k in ("book", "spot"):
            out[part][f"{k}_money"] = float(100 * mean[k][sel].sum() / years)
            out[part][f"{k}_dd"] = float(100 * hf.max_drawdown(mean[k][sel]))
        give = out[part]["spot_money"] - out[part]["book_money"]
        gain = out[part]["book_dd"] - out[part]["spot_dd"]
        out[part]["protection_per_cost"] = (gain / give) if give > 0 else (
            float("inf") if gain > 0 else float("nan"))
    return out


# ── choosing a part's setting ───────────────────────────────────────────


def choose(part: str, rows: dict) -> tuple[str, str]:
    """(main pick, worst-regime pick) for a part, by its own metric (protocol)."""
    prod = rows[PROD[part]]
    names = list(rows)

    def val(r, key, sub="first"):
        v = r[sub][key]
        return np.nan if v is None else v

    if part == "A":
        main = max(names, key=lambda k: val(rows[k], "J"))
        worst = max(names, key=lambda k: min(val(rows[k], "J", "bull"), val(rows[k], "J", "bear")))
        return main, worst
    if part in ("B", "C"):
        key, cons = (("loss_before_on", "false_starts_per_fall") if part == "B"
                     else ("missed_rebound", "premature_per_fall"))
        ok = [k for k in names if val(rows[k], cons) <= val(prod, cons) + 1e-12]
        main = min(ok, key=lambda k: val(rows[k], key))
        worst = min(ok, key=lambda k: max(val(rows[k], key, "bull"), val(rows[k], key, "bear")))
        return main, worst
    if part == "D":
        main = min(names, key=lambda k: val(rows[k], "flip_cost_per_coin_year"))
        worst = min(names, key=lambda k: max(val(rows[k], "flip_cost_per_coin_year", "bull"),
                                             val(rows[k], "flip_cost_per_coin_year", "bear")))
        return main, worst
    if part == "E":
        main = max(names, key=lambda k: val(rows[k], "protection_per_cost"))
        return main, main
    ok = [k for k in names if rows[k]["first"]["book_dd"] >= prod["first"]["book_dd"] - 1.0]
    main = min(ok, key=lambda k: val(rows[k], "ons_per_coin_year"))
    return main, main


def book(name: str, data: dict, inputs: list[dict], cfg: dict, btc) -> dict:
    panel = data["panel"]
    coins = [it["cd"].coin for it in inputs]
    extra = {"partial_hedge_share": 0.5} if cfg["E"] != "full" else None
    bparams, capital = hfu._book_params(name, coins, extra)
    wishes, shares = {}, {}
    for it in inputs:
        w, s = wish_and_share(it["f"], cfg)
        wishes[it["cd"].coin] = w
        if s is not None:
            shares[it["cd"].coin] = s
    res, r = hfu.book_run(panel, bparams, capital, wishes=wishes, shares=shares or None,
                          coins_all=coins)
    test = earning_test.metrics_of(r[r.index >= CUT], additive=True)
    res["p_last"] = test.get(earning_test.P_KEY)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    btc_series = hfu._btc(a.work)
    out: dict = {"cut": str(CUT.date()), "grid": GRID}
    for name in ("set_a", "set_b"):
        data = hfu.load(a.work, name)
        index = data["panel"].prices.index
        btc_px = btc_series.reindex(index).to_numpy(dtype=float)
        regime = hf.regime_of_hours(btc_series, index)
        inputs = coin_inputs(data, btc_px)
        res: dict = {"labels": {}}
        for d in DEFS:
            fl = pd.concat([it["falls"][d].assign(coin=it["cd"].coin) for it in inputs])
            if name == "set_b":  # falls whose peak day the coin was in the ten
                keep = []
                for it in inputs:
                    days = set(it["cd"].index[it["cd"].stepped][it["cd"].mask]
                               .tz_convert("UTC").tz_localize(None).floor("D"))
                    keep.append(it["falls"][d]["peak"].isin(days))
                fl = fl[np.concatenate([k.to_numpy() for k in keep])]
            cut = CUT.tz_localize(None)
            for col in ("peak", "trough", "next_peak"):
                fl[col] = pd.to_datetime(fl[col])
            res["labels"][f"X{d[0]} N{d[1]}"] = {
                part: {"n": int(len(g)), "depth": float(g["depth"].mean()) if len(g) else None,
                       "days": float((g["trough"] - g["peak"]).dt.days.mean()) if len(g) else None,
                       "rebound": float(g["rebound"].mean()) if len(g) else None}
                for part, g in (("first", fl[fl["peak"] < cut]), ("last", fl[fl["peak"] >= cut]))}
            if d == MAIN:
                res["main_falls"] = fl.assign(peak=fl["peak"].astype(str),
                                              trough=fl["trough"].astype(str),
                                              next_peak=fl["next_peak"].astype(str)).to_dict(
                    "records")
        print(name, "labels", json.dumps(res["labels"]), flush=True)
        prod_cfg = dict(PROD)
        res["production_model"] = evaluate(inputs, prod_cfg, regime, index)
        res["production_book"] = book(name, data, inputs, prod_cfg, btc_series)
        picks_main, picks_worst = {}, {}
        for part in PARTS_ORDER:
            rows = {}
            for v in GRID[part]:
                cfg = {**PROD, part: v}
                rows[v] = evaluate(inputs, cfg, regime, index)
            main_pick, worst_pick = choose(part, rows)
            picks_main[part], picks_worst[part] = main_pick, worst_pick
            entry = {"rows": rows, "pick": main_pick, "pick_worst": worst_pick}
            if main_pick != PROD[part]:
                entry["book"] = book(name, data, inputs, {**PROD, part: main_pick}, btc_series)
            print(name, part, main_pick, worst_pick, json.dumps(entry.get("book", {})),
                  flush=True)
            res[part] = entry
        for label, picks in (("assembly", picks_main), ("assembly_worst", picks_worst)):
            res[label] = {"cfg": picks, "model": evaluate(inputs, picks, regime, index),
                          "book": book(name, data, inputs, picks, btc_series)}
            print(name, label, picks, json.dumps(res[label]["book"]), flush=True)
        out[name] = res
    out["seconds"] = round(time.time() - t0)
    path = a.work / "bv3-parts.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print("written", path, out["seconds"], "s")


if __name__ == "__main__":
    main()
