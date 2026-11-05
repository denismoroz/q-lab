"""A hedge-share policy learned on money from minutes
(docs/research/ml-15m-{0,1}-*-protocol-2026-11-05.md).

    uv run python scripts/research/ml_money.py prepare --work DIR [--minutes DIR_OF_1M]
    uv run python scripts/research/ml_money.py phase1 --work DIR     # shared scope, 12 configs
    uv run python scripts/research/ml_money.py phase2 --work DIR     # per-coin and fine-tuned, 8
    uv run python scripts/research/ml_money.py book --work DIR       # 6 runs of the real book
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

COINS = ("BTC", "ETH", "SOL", "AVAX")
STEPS = (15, 60)
SEQ = {15: 192, 60: 48}
CUT = pd.Timestamp("2024-06-12", tz="UTC")
START = pd.Timestamp("2022-01-01", tz="UTC")
WARM = pd.Timedelta(days=30)
PENALTIES = ("money", "turnover", "sortino")
BATCH, BATCHES, EPOCHS = 32, 60, 3
FT_BATCHES, FT_LR = 20, 0.3


def _ml(work: Path) -> Path:
    d = work / "ml"
    d.mkdir(exist_ok=True)
    return d


# ── prepare ────────────────────────────────────────────────────────────


def prepare(work: Path, minutes: Path) -> None:
    import hedge_followup as hfu
    import hedge_trigger_study as hts
    import waves as wr

    from qlab import data_passport as dp
    from qlab import hedge_study as hs
    from qlab import ml_money as mm
    from qlab.strategies.live.bv2 import B2Params

    t0 = time.time()
    data = hfu.load(work, "set_a")
    panel = data["panel"]
    index = panel.prices.index
    rec = hs.run_book(panel, hts._params())
    p = B2Params.from_dict(hts._params())
    feats, btc_logc = {}, None
    for c in COINS:
        feats[c] = wr.coin_features(c, minutes, index, btc_logc)
        if c == "BTC":
            btc_logc = feats[c]["logc"]
    hour_ns = index.as_unit("ns").asi8
    for k, c in enumerate(COINS):
        parts = [pd.read_parquet(f, columns=["open_time", "close", "high", "low", "volume",
                                             "taker_buy_base", "trades"])
                 for f in sorted((minutes / "futures" / c).glob("*.parquet"))]
        m = pd.concat(parts).drop_duplicates("open_time").set_index("open_time").sort_index()
        m.index = pd.DatetimeIndex(m.index).as_unit("ns")
        ch = mm.minute_channels(m)
        bad = dp.bad_tick_mask(m["close"])
        ch[bad] = 0.0  # a spike minute says nothing (declared treatment)
        close = m["close"].to_numpy()
        static_h = np.column_stack([feats[c][f] for f in wr.FEATURES]).astype(np.float32)
        fund = panel.funding[c].fillna(0.0).to_numpy()
        fcs = np.r_[0.0, np.cumsum(fund)]
        prod_h = rec.in_pos[c].to_numpy().astype(np.float32)
        out = {"ch": ch, "open_time": m.index.asi8, "scale": p.book_sizes(c)[0] / 257.0,
               "coin": k, "start": int((m.index[0] + WARM).value)}
        for step in STEPS:
            g = mm.decision_grid(m.index, close, step)
            tau = pd.DatetimeIndex(g["tau"])
            h = np.searchsorted(hour_ns, (tau - pd.Timedelta(minutes=60)).floor("h").asi8,
                                side="left")
            h = np.clip(h, 0, len(index) - 1)
            f_now = np.searchsorted(hour_ns, tau.asi8, side="right")
            f_nxt = np.searchsorted(hour_ns, (tau + pd.Timedelta(minutes=step)).asi8,
                                    side="right")
            pos, nxt = g["pos"].to_numpy(), g["nxt"].to_numpy()
            ends_bad = bad[np.clip(pos, 0, len(bad) - 1)] | bad[np.clip(nxt, 0, len(bad) - 1)]
            valid = g["ret"].notna().to_numpy() & ~ends_bad
            out[f"g{step}"] = {
                "valid": valid,
                "tau": tau.asi8, "pos": g["pos"].to_numpy(), "ret": g["ret"].to_numpy(),
                "ok": g["ok"].to_numpy(), "static": static_h[h],
                "summ": mm.window_summary(ch, np.clip(g["pos"].to_numpy(), 0, None)),
                "fund": fcs[f_nxt] - fcs[f_now], "prod": prod_h[h]}
        with open(_ml(work) / f"{c}.pkl", "wb") as fh:
            pickle.dump(out, fh, protocol=5)
        print(c, "prepared", round(time.time() - t0), "s", flush=True)


def _load(work: Path) -> dict:
    out = {}
    for c in COINS:
        with open(_ml(work) / f"{c}.pkl", "rb") as fh:
            out[c] = pickle.load(fh)
    return out


# ── training and walk-forward ──────────────────────────────────────────


GROUPS = {
    "waves": ["dc1", "dc2", "dc4", "since1", "since2", "since4", "overshoot_now", "structure4",
              "slope5", "slope6", "slope7", "slope8", "slope9"],
    "long_returns": ["ret1", "ret7", "ret30", "btc_ret7"],
    "imbalance": ["imb24", "imb168"],
}
DAY = 86_400_000_000_000
EMBARGO = 2 * DAY


class Data:
    """One step size: per coin arrays. `ablate` zeroes a feature group; `reshuffle` replaces
    the training returns by a day-block permutation (the shuffled-target check)."""

    def __init__(self, d: dict, step: int, ablate: str | None = None,
                 shuffle_until: int | None = None, rng=None) -> None:
        import waves as wr

        self.step = step
        self.c = {}
        for c in COINS:
            g = d[c][f"g{step}"]
            onehot = np.zeros((len(g["tau"]), len(COINS)), dtype=np.float32)
            onehot[:, d[c]["coin"]] = 1.0
            static = g["static"].copy()
            ch, summ = d[c]["ch"], g["summ"]
            if ablate in GROUPS:
                cols = [wr.FEATURES.index(f) for f in GROUPS[ablate]]
                static[:, cols] = np.nan  # normalised to 0 below
            if ablate == "imbalance":
                ch = ch.copy()
                ch[:, 3] = 0.0
                summ = summ.copy()
                summ[:, 3] = 0.0
            if ablate == "minutes":
                ch = np.zeros_like(ch)
                summ = np.zeros_like(summ)
            ret0 = np.where(g["valid"], np.nan_to_num(g["ret"]), 0.0)
            self.c[c] = {**g, "ch": ch, "summ": summ, "static": static, "scale": d[c]["scale"],
                         "start": d[c]["start"], "onehot": onehot, "ret0": ret0,
                         "train_ret": ret0}
        self.shuffle_until = shuffle_until
        self.rng = rng

    def reshuffle(self, t_end: int) -> None:
        """Training returns before `t_end` permuted in blocks of one day (per coin)."""
        for c in COINS:
            g = self.c[c]
            r = g["ret0"].copy()
            idx = np.flatnonzero(g["tau"] < t_end)
            day = (g["tau"][idx] // DAY)
            blocks = [idx[day == k] for k in np.unique(day)]
            order = self.rng.permutation(len(blocks))
            flat = np.concatenate([r[blocks[k]] for k in order]) if blocks else np.array([])
            r[idx] = flat[: len(idx)] if len(flat) >= len(idx) else np.r_[
                flat, np.zeros(len(idx) - len(flat))]
            g["train_ret"] = r

    def starts(self, coin: str, lo: int, hi: int, L: int) -> np.ndarray:
        """Sequence starts whose whole L decisions (and their next steps) lie in [lo, hi)."""
        g = self.c[coin]
        usable = (g["tau"] >= max(lo, g["start"])) & (
            g["tau"] + self.step * 60_000_000_000 <= hi)
        idx = np.flatnonzero(usable)
        if len(idx) <= L:
            return np.array([], dtype=int)
        cand = idx[: len(idx) - L]
        last = idx[np.searchsorted(idx, cand) + L - 1]
        span = (L - 1) * self.step * 60_000_000_000
        # contiguous in positions and in time (no missing decision inside a sequence)
        return cand[(last - cand == L - 1) & (g["tau"][last] - g["tau"][cand] == span)]


def _batch(data: Data, coins, starts, L, norm, rng, n):
    import torch

    from qlab import ml_money as mm

    pick = rng.integers(0, len(coins), size=n)
    win, summ, stat, ret, fund, scale, valid = [], [], [], [], [], [], []
    for j in range(n):
        c = coins[pick[j]]
        s0 = rng.choice(starts[c])
        sl = slice(s0, s0 + L)
        g = data.c[c]
        if norm[2]:
            win.append(mm.windows(g["ch"], np.clip(g["pos"][sl], 0, None)))
        summ.append(g["summ"][sl])
        st = (g["static"][sl] - norm[0]) / norm[1]
        stat.append(np.concatenate([np.nan_to_num(st), g["onehot"][sl]], axis=1))
        ret.append(g["train_ret"][sl])
        fund.append(g["fund"][sl])
        scale.append(g["scale"])
        valid.append(g["valid"][sl])
    t = torch.from_numpy
    w = t(np.stack(win)) if norm[2] else None
    return (w, t(np.stack(summ)), t(np.stack(stat).astype(np.float32)),
            t(np.stack(ret).astype(np.float32)), t(np.stack(fund).astype(np.float32)),
            t(np.array(scale, dtype=np.float32)), t(np.stack(valid).astype(np.float32)))


def _money(model, batch, penalty, cost_mult):
    import torch

    from qlab import ml_money as mm

    w, summ, stat, ret, fund, scale, valid = batch
    enc = model.encode(w, summ, stat)
    _, money = mm.unroll(model, enc, ret, fund, scale[:, None], torch.zeros(len(scale)),
                         cost_mult)
    money = money * valid  # a step without a known price says nothing (declared treatment)
    if penalty == "sortino":
        down = torch.clamp(money, max=0.0)
        loss = -money.sum() / valid.sum().clamp(min=1) / (
            torch.sqrt((down ** 2).sum() / valid.sum().clamp(min=1)) + 1e-8)
    else:
        loss = -money.sum() / valid.sum().clamp(min=1)
    return loss, money.sum() / valid.sum().clamp(min=1)


def _fit(model, data, coins, t_end, norm, penalty, batches, lr, rng, curve):
    """Train on decisions before (t_end - embargo), early-stopping on the money of a validation
    block (the last 10% of the training span, after a further embargo); never on later data."""
    import torch

    L = SEQ[data.step]
    first = min(data.c[c]["start"] for c in coins)
    hi = t_end - EMBARGO
    val_lo = hi - int(0.1 * (hi - first))
    tr = {c: data.starts(c, first, val_lo - EMBARGO, L) for c in coins}
    va = {c: data.starts(c, val_lo, hi, L) for c in coins}
    tr_coins = [c for c in coins if len(tr[c])]
    va_coins = [c for c in coins if len(va[c])]
    if not tr_coins:
        return model
    vrng = np.random.default_rng(11)
    vbatch = _batch(data, va_coins, va, L, norm, vrng, 32) if va_coins else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cost_mult = 3.0 if penalty == "turnover" else 1.0
    best_state, best_val = copy.deepcopy(model.state_dict()), -np.inf
    for k in range(batches):
        loss, _ = _money(model, _batch(data, tr_coins, tr, L, norm, rng, BATCH), penalty,
                         cost_mult)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if vbatch is not None and (k + 1) % 20 == 0:
            with torch.no_grad():
                _, vm = _money(model, vbatch, penalty, 1.0)
            curve.append((k + 1, float(loss), float(vm)))
            if float(vm) > best_val:
                best_val, best_state = float(vm), copy.deepcopy(model.state_dict())
    if vbatch is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model, data, coin, sel, norm, prev0):
    import torch

    from qlab import ml_money as mm

    g = data.c[coin]
    idx = np.flatnonzero(sel)
    if not len(idx):
        return np.array([]), prev0
    with torch.no_grad():
        st = np.concatenate([np.nan_to_num((g["static"][idx] - norm[0]) / norm[1]),
                             g["onehot"][idx]], axis=1).astype(np.float32)
        summ = torch.from_numpy(g["summ"][idx])
        encs = []
        for k in range(0, len(idx), 4096):
            sl = slice(k, k + 4096)
            w = (torch.from_numpy(mm.windows(g["ch"], np.clip(g["pos"][idx[sl]], 0, None)))
                 if model.uses_window else None)
            encs.append(model.encode(w, summ[sl], torch.from_numpy(st[sl])))
        enc = torch.cat(encs)
        shares = np.empty(len(idx), dtype=np.float32)
        prev = torch.tensor([prev0])
        for t in range(len(idx)):
            s = model.step(enc[t:t + 1], prev)
            shares[t] = float(s)
            prev = s
    return shares, float(shares[-1])


def walk_forward(cfg: dict, work_s: str) -> dict:
    """One configuration and seed: quarterly expanding-window retrains from 2022-01 (embargo,
    validation block, early stopping), the shares decided in each following quarter, per coin."""
    import torch

    from qlab import ml_money as mm

    torch.set_num_threads(2)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = _load(Path(work_s))
    data = Data(d, cfg["step"], cfg.get("ablate"), rng=np.random.default_rng(seed + 100))
    n_static = len(data.c["BTC"]["static"][0]) + len(COINS)
    lr = 0.003 if cfg["model"] == "linear" else 0.001
    scope = cfg["scope"]
    train_coins = list(cfg.get("train_coins") or COINS)
    shared = mm.make_policy(cfg["model"], n_static)
    per = {c: mm.make_policy(cfg["model"], n_static) for c in COINS} if scope == "per_coin" \
        else {}
    shares = {c: np.full(len(data.c[c]["tau"]), np.nan, dtype=np.float32) for c in COINS}
    prev = dict.fromkeys(COINS, 0.0)
    curves = {}
    quarters = pd.date_range(START, pd.Timestamp("2026-11-01", tz="UTC"), freq="QS")
    t0 = time.time()
    for q0, q1 in zip(quarters[:-1], quarters[1:], strict=True):
        t_end = q0.value
        if cfg.get("shuffle"):
            data.reshuffle(t_end)
        pool = np.concatenate([data.c[c]["static"][(data.c[c]["tau"] < t_end - EMBARGO)
                                                   & (data.c[c]["tau"] >= data.c[c]["start"])]
                               for c in train_coins])
        mu, sd = np.nanmean(pool, 0), np.nanstd(pool, 0) + 1e-6
        mu, sd = np.nan_to_num(mu), np.where(np.isnan(sd), 1.0, sd)
        norm = (mu, sd, cfg["model"] == "tcn")
        curve = []
        if scope in ("shared", "finetune"):
            _fit(shared, data, train_coins, t_end, norm, cfg["penalty"], BATCHES * EPOCHS, lr,
                 rng, curve)
            models = dict.fromkeys(COINS, shared)
            if scope == "finetune":
                models = {}
                for c in COINS:
                    m = copy.deepcopy(shared)
                    models[c] = _fit(m, data, [c], t_end, norm, cfg["penalty"], FT_BATCHES,
                                     lr * FT_LR, rng, [])
        else:
            for c in COINS:
                per[c] = _fit(per[c], data, [c], t_end, norm, cfg["penalty"], BATCHES * EPOCHS,
                              lr, rng, curve)
            models = per
        curves[str(q0.date())] = curve
        for c in COINS:
            tau = data.c[c]["tau"]
            sel = (tau >= q0.value) & (tau < q1.value)
            s, prev[c] = _predict(models[c], data, c, sel, norm, prev[c])
            shares[c][sel] = s
    return {"cfg": cfg, "shares": shares, "curves": curves,
            "params": mm.n_params(shared if scope != "per_coin" else per["BTC"]),
            "seconds": round(time.time() - t0)}


def evaluate(d: dict, step: int, shares: dict) -> dict:
    """The simplified book on the strategy's money: per period, total and per coin."""
    from qlab import hedge_followup as hf
    from qlab import ml_money as mm

    out = {}
    for part, lo, hi in (("first", START.value, CUT.value),
                         ("last", CUT.value + EMBARGO, 1 << 62)):
        tot = []
        per = {}
        for c in COINS:
            g = d[c][f"g{step}"]
            sel = (g["tau"] >= lo) & (g["tau"] < hi) & (g["tau"] >= d[c]["start"])
            s = np.nan_to_num(shares[c][sel])
            prev = np.r_[s[:1], s[:-1]]
            ret = np.where(g["valid"][sel], np.nan_to_num(g["ret"][sel]), 0.0)
            fee = mm.COST * np.abs(s - prev)
            m = d[c]["scale"] * ((1 - s) * ret + s * g["fund"][sel] - fee)
            years = sel.sum() * step / (60 * 24 * 365)
            per[c] = {"money": float(100 * m.sum() / years) if years else None,
                      "changes_per_year": float(np.abs(s - prev).sum() / years) if years else None,
                      "fees": float(100 * (d[c]["scale"] * fee).sum() / years) if years else None,
                      "mean_share": float(s.mean()) if len(s) else None}
            tot.append(pd.Series(m, index=g["tau"][sel]))
        curve = pd.concat(tot, axis=1).fillna(0.0).sum(axis=1).sort_index()
        years = (min(hi, curve.index.max()) - lo) / (365 * 86_400e9)
        out[part] = {"money": float(100 * curve.sum() / years),
                     "dd": float(100 * hf.max_drawdown(curve.to_numpy())),
                     "changes_per_coin_year": float(np.mean([per[c]["changes_per_year"]
                                                             for c in COINS])),
                     "fees": float(sum(per[c]["fees"] for c in COINS)), "per_coin": per}
    return out



SEEDS = (0, 1, 2)


def _name(cfg):
    extra = "".join(f" {k}={cfg[k]}" for k in ("ablate", "shuffle", "train_coins") if cfg.get(k))
    return f"{cfg['step']}m {cfg['model']} {cfg['penalty']} {cfg['scope']}{extra}"


def _short(ev):
    return {p: {k: round(v, 3) for k, v in ev[p].items() if k != "per_coin"} for p in ev}


def run_configs(work: Path, cfgs: list[dict], tag: str) -> dict:
    """Every configuration with three seeds; per seed and for the seed-average share."""
    d = _load(work)
    results = {}
    jobs = [(c, seed) for c in cfgs for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [(c, seed, pool.submit(walk_forward, {**c, "seed": seed}, str(work)))
                for c, seed in jobs]
        got: dict = {}
        for c, seed, fut in futs:
            r = fut.result()
            got.setdefault(_name(c), []).append(r)
            print(_name(c), "seed", seed, json.dumps(_short(evaluate(d, c["step"], r["shares"]))),
                  r["seconds"], "s", flush=True)
    for name, runs in got.items():
        cfg = runs[0]["cfg"]
        seeds = [evaluate(d, cfg["step"], r["shares"]) for r in runs]
        mean_shares = {c: np.nanmean(np.stack([r["shares"][c] for r in runs]), axis=0)
                       for c in COINS}
        ens = evaluate(d, cfg["step"], mean_shares)
        spread = {p: {k: float(np.std([sd[p][k] for sd in seeds])) for k in ("money", "dd")}
                  for p in ("first", "last")}
        mean = {p: {k: float(np.mean([sd[p][k] for sd in seeds])) for k in ("money", "dd")}
                for p in ("first", "last")}
        results[name] = {"cfg": {k: v for k, v in cfg.items() if k != "seed"}, "seeds": seeds,
                         "mean": mean, "spread": spread, "ensemble": ens,
                         "params": runs[0]["params"],
                         "curves": [r["curves"] for r in runs],
                         "seconds": [r["seconds"] for r in runs]}
        with open(_ml(work) / f"shares-{name.replace(' ', '_')}.pkl", "wb") as fh:
            pickle.dump(mean_shares, fh)
        print("==", name, "mean", json.dumps(mean), "spread", json.dumps(spread), flush=True)
    out = {"results": results}
    if tag == "phase1":
        prod = {}
        for st in STEPS:
            prod[st] = evaluate(d, st, {c: d[c][f"g{st}"]["prod"] for c in COINS})
            # no information: a constant share, production's average before 2022
            avg = float(np.mean(np.concatenate([
                d[c][f"g{st}"]["prod"][(d[c][f"g{st}"]["tau"] < START.value)
                                       & (d[c][f"g{st}"]["tau"] >= d[c]["start"])]
                for c in COINS])))
            const = {c: np.full(len(d[c][f"g{st}"]["tau"]), avg, dtype=np.float32)
                     for c in COINS}
            prod[f"constant {st}"] = {"share": avg, **evaluate(d, st, const)}
        out["production"] = prod
    (work / f"ml-{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def _configs_phase1():
    return [{"step": s, "model": m, "penalty": p, "scope": "shared"}
            for s in STEPS for m in ("linear", "tcn") for p in PENALTIES]


def phase2_configs(work: Path) -> list[dict]:
    p1 = json.loads((work / "ml-phase1.json").read_text())["results"]
    cfgs = []
    for s in STEPS:
        for m in ("linear", "tcn"):
            best = max(PENALTIES, key=lambda p, s=s, m=m: p1[f"{s}m {m} {p} shared"]["mean"]
                       ["first"]["money"])
            for scope in ("per_coin", "finetune"):
                cfgs.append({"step": s, "model": m, "penalty": best, "scope": scope})
    return cfgs


def _all_results(work: Path) -> dict:
    out = {}
    for tag in ("phase1", "phase2"):
        f = work / f"ml-{tag}.json"
        if f.exists():
            out.update(json.loads(f.read_text())["results"])
    return out


def chosen(work: Path) -> dict:
    """Per step and scope the best by the first period's seed-mean money (shared scope: the
    penalty per model first); and the overall pick."""
    allr = _all_results(work)
    picks = {}
    for s in STEPS:
        for scope in ("shared", "per_coin", "finetune"):
            cands = [k for k, v in allr.items() if v["cfg"]["step"] == s
                     and v["cfg"]["scope"] == scope and not any(
                         v["cfg"].get(x) for x in ("ablate", "shuffle", "train_coins"))]
            if not cands:
                continue
            picks[f"{s}m {scope}"] = max(cands, key=lambda k: allr[k]["mean"]["first"]["money"])
    best = max(picks.values(), key=lambda k: allr[k]["mean"]["first"]["money"])
    return {"picks": picks, "best": best, "best_cfg": allr[best]["cfg"]}


def sanity(work: Path) -> None:
    ch = chosen(work)
    base = ch["best_cfg"]
    cfgs = [{**base, "shuffle": True}]
    cfgs += [{**base, "ablate": g} for g in ("minutes", "waves", "long_returns", "imbalance")]
    if base["scope"] == "shared":
        cfgs.append({**base, "train_coins": ["BTC", "ETH"]})
    out = run_configs(work, cfgs, "sanity")
    out["chosen"] = ch
    (work / "ml-sanity.json").write_text(json.dumps(out, indent=1, default=str))


def repro(work: Path) -> None:
    d = _load(work)
    ch = chosen(work)
    r = walk_forward({**ch["best_cfg"], "seed": 0}, str(work))
    ev = evaluate(d, ch["best_cfg"]["step"], r["shares"])
    first = _all_results(work)[ch["best"]]["seeds"][0]
    (work / "ml-repro.json").write_text(json.dumps({"rerun": _short(ev),
                                                    "original": _short(first)}, indent=1))
    print("repro", _short(ev), _short(first))


def book(work: Path) -> None:
    import hedge_followup as hfu

    from qlab import earning_test

    ch = chosen(work)
    data = hfu.load(work, "set_a")
    panel = data["panel"]
    index = panel.prices.index
    d = _load(work)
    out = {"chosen": ch}
    bparams, capital = hfu._book_params("set_a", list(COINS), {"partial_hedge_share": 0.5})
    for label, pick in ch["picks"].items():
        s = int(label.split("m")[0])
        with open(_ml(work) / f"shares-{pick.replace(' ', '_')}.pkl", "rb") as fh:
            sh = pickle.load(fh)
        wishes, shares = {}, {}
        for c in COINS:
            g = d[c][f"g{s}"]
            ser = pd.Series(sh[c], index=pd.to_datetime(g["tau"], utc=True))
            # the hourly book: the share decided at the end of each hour bar
            hourly = ser.reindex((index + pd.Timedelta(hours=1)).as_unit("ns")).to_numpy()
            # before the model decides (2022-01) the production rule holds: its own wish
            # (NaN) and the whole spot hedged
            wishes[c] = np.where(np.isnan(hourly), np.nan, (hourly > 0.02).astype(float))
            shares[c] = np.where(np.isnan(hourly), 1.0, np.clip(hourly, 0.02, 1.0))
        res, r = hfu.book_run(panel, bparams, capital, wishes=wishes, shares=shares,
                              coins_all=list(COINS))
        res["p_last"] = earning_test.metrics_of(r[r.index >= CUT], additive=True).get(
            earning_test.P_KEY)
        out[label] = {"config": pick, **res}
        print("book", label, pick, json.dumps(res), flush=True)
    base, r0 = hfu.book_run(panel, *hfu._book_params("set_a", list(COINS)),
                            coins_all=list(COINS))
    base["p_last"] = earning_test.metrics_of(r0[r0.index >= CUT], additive=True).get(
        earning_test.P_KEY)
    out["production"] = base
    (work / "ml-book.json").write_text(json.dumps(out, indent=1, default=str))


def samples(work: Path) -> dict:
    """Effective sample per coin: big falls known by 2022-01 and by the last retrain; legs."""
    import hedge_followup as hfu

    from qlab import hedge_followup as hf
    from qlab import waves as wv

    data = hfu.load(work, "set_a")
    out = {}
    for c in COINS:
        close = hf.daily_closes(data["panel"].prices[c]).dropna()
        f22, _ = wv.confirmed_falls(close, 0.25, 7, pd.Timestamp("2021-12-31"))
        f26, _ = wv.confirmed_falls(close, 0.25, 7, pd.Timestamp("2026-07-31"))
        from qlab.bv3_parts import zigzag_falls

        legs22 = len(zigzag_falls(close[close.index < "2022-01-01"], 0.25, 0)) * 2
        legs26 = len(zigzag_falls(close[close.index < "2026-08-01"], 0.25, 0)) * 2
        out[c] = {"falls_2022": len(f22), "falls_2026": len(f26), "legs_2022": legs22,
                  "legs_2026": legs26}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "phase1", "phase2", "sanity", "book",
                                      "repro", "samples"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    a = ap.parse_args()
    t0 = time.time()
    if a.phase == "prepare":
        prepare(a.work, a.minutes)
    elif a.phase == "phase1":
        run_configs(a.work, _configs_phase1(), "phase1")
    elif a.phase == "phase2":
        run_configs(a.work, phase2_configs(a.work), "phase2")
    elif a.phase == "sanity":
        sanity(a.work)
    elif a.phase == "repro":
        repro(a.work)
    elif a.phase == "book":
        book(a.work)
    else:
        (a.work / "ml-samples.json").write_text(json.dumps(samples(a.work), indent=1))
    print("done", a.phase, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
