"""Forecast pilot 2 on BTC: the fixed nulls, the size baseline, and the fast-input variants
(docs/research/forecast-pilot-2-protocol-2026-11-05.md and its addendum).

    uv run python scripts/research/forecast_pilot2.py --work DIR --variant {a,b,c,d,e}
    uv run python scripts/research/forecast_pilot2.py --work DIR --combine

Needs DIR/ml/BTC.pkl from `ml_money.py prepare`. Writes DIR/forecast2-<variant>.json, and with
--combine DIR/forecast2.json (the first-period choice and the direction rule).
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
import forecast_pilot as fp  # noqa: E402

from qlab import data_passport as dp  # noqa: E402
from qlab import forecast as fc  # noqa: E402
from qlab import forecast_inputs as fi  # noqa: E402
from qlab import ml_money as mm  # noqa: E402

H = (1, 4, 24)
SIDES = ("fall", "rise")
K = 1.0
NULLS = ("other day", "within day", "day block")
VARIANTS = {"a": "minutes", "b": "minute summaries", "c": "15-minute bars",
            "d": "1-hour bars", "e": "dollar bars"}
SEQ_LEN = {"a": mm.WINDOW, "c": 96, "d": 168, "e": 96}
HOUR, DAY = fp.HOUR, fp.DAY


def load_minutes(minutes: Path) -> pd.DataFrame:
    cols = ["open_time", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_base"]
    m = pd.concat([pd.read_parquet(f, columns=cols)
                   for f in sorted((minutes / "futures" / "BTC").glob("*.parquet"))])
    m = m.drop_duplicates("open_time").set_index("open_time").sort_index()
    m.index = pd.DatetimeIndex(m.index).as_unit("ns")
    return m


def dollar_amount(m: pd.DataFrame) -> float:
    """Declared: 2 x the median hourly traded value of the first 90 days of the data."""
    first = m[m.index < m.index[0] + pd.Timedelta(days=90)]
    hourly = first["quote_volume"].groupby(first.index.floor("h")).sum()
    return float(2 * hourly.median())


def build(work: Path, minutes: Path, variant: str) -> dict:
    """The common base (21 wave + 32 standard features), the variant's fast input for the tables
    and its sequence for the TCN (None for b), the targets, the classes and the size baseline."""
    D = fp.prepare_data(work, minutes, "std")
    d, tau, pos, X = D["d"], D["tau"], D["pos"], D["X"]
    base = np.column_stack([X[:, :21], X[:, 26:]])
    base_names = D["names"][:21] + D["names"][26:]
    m = load_minutes(minutes)
    assert len(m) == len(d["ch"]), "minute rows differ from the prepared channels"
    p = np.clip(pos, 0, None)
    logc = np.where(d["g60"]["ok"], np.log(m["close"].to_numpy()[p]), np.nan)
    info: dict = {}
    seq = None
    if variant == "a":
        tab, tab_names = X[:, 21:26], D["names"][21:26]
        seq = mm.windows(d["ch"], p)
    elif variant == "b":
        mc = fi.clean_minutes(m, dp.bad_tick_mask(m["close"]))
        tab, tab_names = fi.minute_summaries(mc)[p], list(fi.SUMMARY_NAMES)
    else:
        mc = fi.clean_minutes(m, dp.bad_tick_mask(m["close"]))
        if variant == "e":
            amount = dollar_amount(m)
            ids, dollar = fi.dollar_bar_ids(mc, amount), True
            info["dollar_amount_usd"] = amount
        else:
            ids, dollar = fi.time_bar_ids(mc, 15 if variant == "c" else 60), False
        ch, end = fi.bars_from_ids(mc, ids, fi.minute_sigma(mc["close"]), dollar)
        info["bars"] = int(len(end))
        if dollar:
            info["median_bar_minutes_by_year"] = {
                str(y): float(np.median(np.diff(end)[pd.DatetimeIndex(end[1:]).year == y])
                              / fi.MINUTE)
                for y in range(2020, 2027) if (pd.DatetimeIndex(end[1:]).year == y).sum() > 10}
        L = SEQ_LEN[variant]
        tab = fi.bar_window_means(ch, end, tau, L)
        tab_names = [f"bar_{c}" for c in ("ret", "range", "dur" if dollar else "volume",
                                          "imbalance", "trades")]
        seq = fi.bar_windows(ch, end, tau, L)
    B, b_names = fi.size_baseline(logc, tau)
    Xv = np.column_stack([base, tab]).astype(np.float64)
    labels = {(h, s): D["labels"][(h, s, K)] for h in H for s in SIDES}
    return {"d": d, "tau": tau, "pos": pos, "dates": D["dates"], "X": Xv,
            "names": base_names + tab_names, "seq": seq, "targets": D["targets"],
            "pasts": D["pasts"], "labels": labels, "B": B, "b_names": b_names, "logc": logc,
            "info": info}


def parts(tau: np.ndarray, h: int) -> dict:
    return {"first": (tau >= fp.START.value) & (tau < fp.CUT),
            "last": tau >= fp.CUT + h * HOUR + DAY}


def run_variant(work: Path, minutes: Path, variant: str, threads: int) -> None:
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    t0 = time.time()
    V = build(work, minutes, variant)
    print("built", variant, V["X"].shape, round(time.time() - t0), "s", flush=True)
    d, tau, X, B = V["d"], V["tau"], V["X"], V["B"]
    targets, labels = V["targets"], V["labels"]
    retrains = pd.date_range(fp.START, pd.Timestamp("2026-11-01", tz="UTC"), freq="6MS")
    preds: dict = {}
    rng = np.random.default_rng(11)

    def gbr():
        return HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                             min_samples_leaf=1000, random_state=0)

    def gbc():
        return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=1000, random_state=0)

    def train_mask(h, t_end, y):
        return (tau + h * HOUR + (h * HOUR + DAY) <= t_end) & ~np.isnan(y) & (
            tau >= d["start"])

    def put(key, sel, values):
        preds.setdefault(key, np.full(len(tau), np.nan))[sel] = values

    trials = 0
    for q0, q1 in zip(retrains[:-1], retrains[1:], strict=True):
        t_end = q0.value
        sel = (tau >= q0.value) & (tau < q1.value)
        for h in H:
            y = targets[h]
            m = train_mask(h, t_end, y)
            mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
            xs = np.nan_to_num((X - mu) / sd)
            put(("ridge", h), sel, Ridge(alpha=10).fit(xs[m], y[m]).predict(xs[sel]))
            put(("gbm", h), sel, gbr().fit(X[m], y[m]).predict(X[sel]))
            for how in NULLS:
                ys = fi.shuffled_labels(y, m, tau, how, rng)
                put(("ridge", h, how), sel, Ridge(alpha=10).fit(xs[m], ys[m]).predict(xs[sel]))
                put(("gbm", h, how), sel, gbr().fit(X[m], ys[m]).predict(X[sel]))
            for s in SIDES:
                lab = labels[(h, s)]
                mk = train_mask(h, t_end, lab)
                put(("logit", h, s), sel, LogisticRegression(C=0.1, max_iter=500).fit(
                    xs[mk], lab[mk]).predict_proba(xs[sel])[:, 1])
                put(("gbc", h, s), sel, gbc().fit(X[mk], lab[mk]).predict_proba(X[sel])[:, 1])
                for how in NULLS:
                    ls = fi.shuffled_labels(lab, mk, tau, how, rng)
                    put(("logit", h, s, how), sel, LogisticRegression(C=0.1, max_iter=500).fit(
                        xs[mk], ls[mk]).predict_proba(xs[sel])[:, 1])
                    put(("gbc", h, s, how), sel,
                        gbc().fit(X[mk], ls[mk]).predict_proba(X[sel])[:, 1])
                # the size baseline: recent volatility, hour of the day, day of the week
                mb = mk & ~np.isnan(B).any(axis=1)
                bmu, bsd = B[mb].mean(0), B[mb].std(0) + 1e-9
                bs = np.nan_to_num((B - bmu) / bsd)
                put(("size logit", h, s), sel, LogisticRegression(C=0.1, max_iter=500).fit(
                    bs[mb], lab[mb]).predict_proba(bs[sel])[:, 1])
                put(("size gbm", h, s), sel,
                    gbc().fit(B[mb], lab[mb]).predict_proba(B[sel])[:, 1])
        print("tabular", variant, q0.date(), round(time.time() - t0), "s", flush=True)
    trials += 3 * (2 + 2 * len(NULLS)) + 6 * (2 + 2 * len(NULLS))  # + size baseline (shared)

    if V["seq"] is not None:
        seqs = V["seq"].astype(np.float32)
        keys_c = [(h, s) for h in H for s in SIDES]
        torch.set_num_threads(threads)
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            srng = np.random.default_rng(seed)
            for kind in ("reg", "cls"):
                n_out = len(H) if kind == "reg" else len(keys_c)
                net = fc.ForecastTCN(X.shape[1], n_out)
                Y = np.column_stack([targets[h] for h in H] if kind == "reg"
                                    else [labels[k] for k in keys_c])
                hz = np.array(list(H) if kind == "reg" else [k[0] for k in keys_c])
                for q0, q1 in zip(retrains[:-1], retrains[1:], strict=True):
                    t_end = q0.value
                    ok = (tau[:, None] + hz[None, :] * HOUR + (hz[None, :] * HOUR + DAY)
                          <= t_end) & ~np.isnan(Y) & (tau[:, None] >= d["start"])
                    rows = np.flatnonzero(ok.any(axis=1))[::4]
                    mu, sd = np.nanmean(X[rows], 0), np.nanstd(X[rows], 0) + 1e-9
                    xs = np.nan_to_num((X - mu) / sd).astype(np.float32)
                    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                    for _ in range(2):
                        order = srng.permutation(rows)
                        for b in range(0, len(order), 256):
                            r = order[b:b + 256]
                            out = net(torch.from_numpy(seqs[r]), torch.from_numpy(xs[r]))
                            yt = torch.from_numpy(np.nan_to_num(Y[r]).astype(np.float32))
                            mt = torch.from_numpy(ok[r].astype(np.float32))
                            if kind == "reg":
                                loss = ((out - yt) ** 2 * mt).sum() / mt.sum().clamp(min=1)
                            else:
                                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                                    out, yt, reduction="none") * mt).sum() / mt.sum().clamp(
                                    min=1)
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                    sel = np.flatnonzero((tau >= q0.value) & (tau < q1.value))
                    with torch.no_grad():
                        o = net(torch.from_numpy(seqs[sel]), torch.from_numpy(xs[sel]))
                        o = (o if kind == "reg" else torch.sigmoid(o)).numpy()
                    for j in range(n_out):
                        put(("tcn", H[j], seed) if kind == "reg"
                            else ("tcn", *keys_c[j], seed), sel, o[:, j])
            print("tcn", variant, "seed", seed, round(time.time() - t0), "s", flush=True)
        trials += 3 + 6

    res = measure(V, preds)
    res["variant"], res["variant_name"] = variant, VARIANTS[variant]
    res["features"] = V["names"]
    res["info"] = V["info"]
    res["trials"] = trials
    res["seconds"] = round(time.time() - t0)
    (work / f"forecast2-{variant}.json").write_text(json.dumps(res, indent=1, default=str))
    print("written", variant, res["seconds"], "s", flush=True)


def measure(V: dict, preds: dict) -> dict:
    tau, dates, targets, labels, pasts = (V["tau"], V["dates"], V["targets"], V["labels"],
                                          V["pasts"])
    has_tcn = ("tcn", 1, 0) in preds
    out: dict = {"regression": {}, "classes": {}, "fall_minus_rise": {}}
    for h in H:
        y = targets[h]
        cands = {"ridge": preds[("ridge", h)], "gbm": preds[("gbm", h)]}
        if has_tcn:
            cands["tcn (3 seeds)"] = np.nanmean(np.stack([preds[("tcn", h, s)]
                                                          for s in (0, 1, 2)]), 0)
        for how in NULLS:
            cands[f"ridge null {how}"] = preds[("ridge", h, how)]
            cands[f"gbm null {how}"] = preds[("gbm", h, how)]
        cands["previous return"] = pasts[h]
        rows = {}
        for name, p in cands.items():
            row = {}
            for part, sel in parts(tau, h).items():
                ok = sel & ~np.isnan(p) & ~np.isnan(y)
                pp, yy, dd = p[ok], y[ok], dates[ok]
                row[part] = {"n": int(ok.sum()),
                             "ic": fp._boot(dd, lambda r, pp=pp, yy=yy: fc.pearson(pp[r], yy[r])),
                             "rank_ic": fp._boot(dd, lambda r, pp=pp, yy=yy: fc.spearman(
                                 pp[r], yy[r])),
                             "hit": fc.hit_rate(pp, yy)}
            rows[name] = row
        out["regression"][h] = rows
    for h in H:
        sc: dict = {}
        for s in SIDES:
            lab = labels[(h, s)]
            cands = {"logit": preds[("logit", h, s)], "gbm": preds[("gbc", h, s)]}
            if has_tcn:
                cands["tcn (3 seeds)"] = np.nanmean(np.stack([preds[("tcn", h, s, k)]
                                                              for k in (0, 1, 2)]), 0)
            for how in NULLS:
                cands[f"logit null {how}"] = preds[("logit", h, s, how)]
                cands[f"gbm null {how}"] = preds[("gbc", h, s, how)]
            cands["size logit"] = preds[("size logit", h, s)]
            cands["size gbm"] = preds[("size gbm", h, s)]
            rows = {}
            for name, p in cands.items():
                row = {}
                for part, sel in parts(tau, h).items():
                    ok = sel & ~np.isnan(p) & ~np.isnan(lab)
                    pp, yy, dd = p[ok], lab[ok], dates[ok]
                    row[part] = {"n": int(ok.sum()), "base_rate": float(yy.mean()),
                                 "auc": fp._boot(dd, lambda r, pp=pp, yy=yy: fc.auc(pp[r],
                                                                                    yy[r]))}
                rows[name] = row
            out["classes"][f"{h}h {s}"] = rows
            sc[s] = cands
        # fall - rise AUC of each model, and of the model minus the size baseline (the size
        # baseline of the same family: logit vs size logit, gbm and tcn vs size gbm)
        fr: dict = {}
        for name in [n for n in sc["fall"] if "null" not in n]:
            row = {}
            for part, sel in parts(tau, h).items():
                pf, pr = sc["fall"][name], sc["rise"][name]
                bname = "size logit" if name == "logit" else "size gbm"
                bf, br = sc["fall"][bname], sc["rise"][bname]
                lf, lr = labels[(h, "fall")], labels[(h, "rise")]
                ok = sel & ~np.isnan(pf) & ~np.isnan(pr) & ~np.isnan(bf) & ~np.isnan(br) & (
                    ~np.isnan(lf))
                a_ = [x[ok] for x in (pf, pr, bf, br, lf, lr)]
                dd = dates[ok]

                def diff(r, a_=a_):
                    return fc.auc(a_[0][r], a_[4][r]) - fc.auc(a_[1][r], a_[5][r])

                def diff_vs_size(r, a_=a_):
                    return (fc.auc(a_[0][r], a_[4][r]) - fc.auc(a_[2][r], a_[4][r])) - (
                        fc.auc(a_[1][r], a_[5][r]) - fc.auc(a_[3][r], a_[5][r]))

                row[part] = {"fall_minus_rise": fp._boot(dd, diff),
                             "vs_size_baseline": fp._boot(dd, diff_vs_size),
                             "baseline": bname}
            fr[name] = row
        out["fall_minus_rise"][h] = fr
    return out


def combine(work: Path) -> None:
    res = {v: json.loads((work / f"forecast2-{v}.json").read_text()) for v in VARIANTS
           if (work / f"forecast2-{v}.json").exists()}
    out: dict = {"variants": list(res), "choice": {}, "direction_rule": {}, "nulls_ok": {}}
    models = ("ridge", "gbm", "tcn (3 seeds)")
    # the first-period choice: the best tabular model's mean first-period IC at 4 h and 1 day
    score = {}
    for v, r in res.items():
        best = max(np.mean([r["regression"][str(h)][mo]["first"]["ic"][0] for h in (4, 24)])
                   for mo in ("ridge", "gbm"))
        score[v] = float(best)
    out["choice"]["tabular_score_first"] = score
    out["choice"]["variant"] = max(score, key=score.get)
    # the direction rule, per variant and horizon: the model chosen on the first period; its
    # interval's lower end above every null's IC and the previous return's, on both periods
    passed = []
    for v, r in res.items():
        for h in (4, 24):
            rows = r["regression"][str(h)]
            cand = [mo for mo in models if mo in rows]
            mo = max(cand, key=lambda x: rows[x]["first"]["ic"][0])
            ok_parts = {}
            for part in ("first", "last"):
                bar = max([rows[n][part]["ic"][0] for n in rows if "null" in n]
                          + [rows["previous return"][part]["ic"][0]])
                ok_parts[part] = {"low": rows[mo][part]["ic"][1], "bar": bar,
                                  "pass": bool(rows[mo][part]["ic"][1] > bar)}
            ok = all(x["pass"] for x in ok_parts.values())
            out["direction_rule"][f"{v} {h}h"] = {"model": mo, **ok_parts, "pass": ok}
            if ok:
                passed.append(f"{v} {h}h")
        # the stop rule on the primary null: every interval contains 0 (IC) / 0.5 (AUC)
        bad = []
        for h, rows in r["regression"].items():
            for n in rows:
                if "null other day" in n:
                    for part in ("first", "last"):
                        _, lo, hi = rows[n][part]["ic"]
                        if not lo <= 0 <= hi:
                            bad.append(f"{n} {h}h {part}")
        for key, rows in r["classes"].items():
            for n in rows:
                if "null other day" in n:
                    for part in ("first", "last"):
                        _, lo, hi = rows[n][part]["auc"]
                        if not lo <= 0.5 <= hi:
                            bad.append(f"{n} {key} {part}")
        out["nulls_ok"][v] = {"ok": not bad, "outside": bad}
    out["passed"] = passed
    out["trials"] = int(sum(r["trials"] for r in res.values()) + 6 * 2 + 3)
    (work / "forecast2.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("choice", "passed", "nulls_ok", "trials")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine:
        combine(a.work)
    else:
        run_variant(a.work, a.minutes, a.variant, a.threads)


if __name__ == "__main__":
    main()
