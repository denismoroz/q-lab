"""Forecast pilot 3: BTC, ETH, SOL, AVAX each judged on its own against a 50-draw null
distribution (docs/research/forecast-pilot-3-protocol-2026-11-06.md).

    uv run python scripts/research/forecast_pilot3.py --work DIR --coin ETH --variant b
    uv run python scripts/research/forecast_pilot3.py --work DIR --pooled
    uv run python scripts/research/forecast_pilot3.py --work DIR --combine

Needs DIR/ml/<COIN>.pkl from `ml_money.py prepare`. Writes DIR/forecast3-<coin>-<variant>.json,
DIR/forecast3-pooled.json and DIR/forecast3.json.
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
import forecast_pilot2 as fp2  # noqa: E402

from qlab import forecast as fc  # noqa: E402
from qlab import forecast_inputs as fi  # noqa: E402

COINS = ("BTC", "ETH", "SOL", "AVAX")
VARIANTS = ("b", "e")
H = (1, 4, 24)
SIDES = ("fall", "rise")
DRAWS = 50
HOUR, DAY = fp.HOUR, fp.DAY
RETRAINS = pd.date_range(fp.START, pd.Timestamp("2026-11-01", tz="UTC"), freq="6MS")
MINUTES = Path("/Users/d/prj/q-lab/data/raw/binance/1m")


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    def gb(cls):
        return cls(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=1000,
                   random_state=0)

    return {"ridge": (lambda: Ridge(alpha=10), True, "reg"),
            "gbm": (lambda: gb(HistGradientBoostingRegressor), False, "reg"),
            "logit": (lambda: LogisticRegression(C=0.1, max_iter=500), True, "cls"),
            "gbc": (lambda: gb(HistGradientBoostingClassifier), False, "cls")}


def _fit_predict(make, scaled, kind, X, m, y, sel):
    if scaled:
        mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
        xt, xp = np.nan_to_num((X[m] - mu) / sd), np.nan_to_num((X[sel] - mu) / sd)
    else:
        xt, xp = X[m], X[sel]
    mod = make().fit(xt, y[m])
    return mod.predict(xp) if kind == "reg" else mod.predict_proba(xp)[:, 1]


def _walk(make, scaled, kind, X, tau, start, h, y, rng=None):
    """Half-yearly walk-forward predictions; with `rng`, trained on the primary null's labels."""
    p = np.full(len(tau), np.nan)
    for q0, q1 in zip(RETRAINS[:-1], RETRAINS[1:], strict=True):
        m = (tau + h * HOUR + (h * HOUR + DAY) <= q0.value) & ~np.isnan(y) & (tau >= start)
        sel = (tau >= q0.value) & (tau < q1.value)
        if not sel.any() or m.sum() < 1000:
            continue
        yy = y if rng is None else fi.shuffled_labels(y, m, tau, "other day", rng)
        p[sel] = _fit_predict(make, scaled, kind, X, m, yy, sel)
    return p


def _score(kind, p, y):
    return fc.pearson(p, y) if kind == "reg" else fc.auc(p, y)


def _point(kind, p, y, mask):
    ok = mask & ~np.isnan(p) & ~np.isnan(y)
    return _score(kind, p[ok], y[ok]) if ok.sum() > 100 else float("nan")


def _boot(kind, p, y, mask, dates):
    ok = mask & ~np.isnan(p) & ~np.isnan(y)
    pp, yy = p[ok], y[ok]
    return fp._boot(dates[ok], lambda r: _score(kind, pp[r], yy[r]))


def _targets(V):
    out = {}
    for h in H:
        out[("reg", h, None)] = V["targets"][h]
        for s in SIDES:
            out[("cls", h, s)] = V["labels"][(h, s)]
    return out


def _tn(t):
    return f"z {t[1]}h" if t[0] == "reg" else f"{t[1]}h {t[2]}"


def _diff_stats(V, pf, pr, bf, br, h, mask):
    lf, lr = V["labels"][(h, "fall")], V["labels"][(h, "rise")]
    ok = mask & ~np.isnan(pf) & ~np.isnan(pr) & ~np.isnan(bf) & ~np.isnan(br) & ~np.isnan(lf)
    a = [x[ok] for x in (pf, pr, bf, br, lf, lr)]

    def dd(r):
        return (fc.auc(a[0][r], a[4][r]) - fc.auc(a[1][r], a[5][r])) - (
            fc.auc(a[2][r], a[4][r]) - fc.auc(a[3][r], a[5][r]))

    return fp._boot(V["dates"][ok], dd)


def run_coin(work: Path, coin: str, variant: str) -> None:
    t0 = time.time()
    V = fp2.build(work, MINUTES, variant, coin)
    tau, X, B, start = V["tau"], V["X"], V["B"], V["d"]["start"]
    print(coin, variant, "built", X.shape, round(time.time() - t0), "s", flush=True)
    models = _models()
    targets = _targets(V)
    preds: dict = {}
    out: dict = {"coin": coin, "variant": variant, "info": V["info"], "real": {}, "null": {},
                 "size": {}, "previous": {}, "fall_minus_rise_vs_size": {}}
    parts = {h: fp2.parts(tau, h) for h in H}
    for t, y in targets.items():
        h = t[1]
        for name, (make, scaled, kind) in models.items():
            if kind != t[0]:
                continue
            p = _walk(make, scaled, kind, X, tau, start, h, y)
            preds[(t, name)] = p
            out["real"][f"{_tn(t)} {name}"] = {
                part: _boot(kind, p, y, mk, V["dates"]) for part, mk in parts[h].items()}
            draws = {part: [] for part in parts[h]}
            for k in range(DRAWS):
                rng = np.random.default_rng(10_000 * (1 + COINS.index(coin)) + 100 * k
                                            + 7 * list(targets).index(t))
                q = _walk(make, scaled, kind, X, tau, start, h, y, rng)
                for part, mk in parts[h].items():
                    draws[part].append(_point(kind, q, y, mk))
            out["null"][f"{_tn(t)} {name}"] = draws
            print(coin, variant, _tn(t), name, round(time.time() - t0), "s", flush=True)
        if t[0] == "reg":
            out["previous"][_tn(t)] = {part: _boot("reg", V["pasts"][h], y, mk, V["dates"])
                                       for part, mk in parts[h].items()}
        else:
            for name in ("logit", "gbc"):
                make, scaled, kind = models[name]
                p = _walk(make, scaled, kind, B, tau, start, h, y)
                preds[(t, "size " + name)] = p
                out["size"][f"{_tn(t)} size {name}"] = {
                    part: _boot(kind, p, y, mk, V["dates"]) for part, mk in parts[h].items()}
    for h in H:
        for fam in ("logit", "gbc"):
            for base in ("size logit", "size gbc"):
                pf, pr = preds[(("cls", h, "fall"), fam)], preds[(("cls", h, "rise"), fam)]
                bf, br = preds[(("cls", h, "fall"), base)], preds[(("cls", h, "rise"), base)]
                out["fall_minus_rise_vs_size"][f"{h}h {fam} vs {base}"] = {
                    part: _diff_stats(V, pf, pr, bf, br, h, mk)
                    for part, mk in parts[h].items()}
    out["seconds"] = round(time.time() - t0)
    (work / f"forecast3-{coin}-{variant}.json").write_text(json.dumps(out, indent=1))
    print("written", coin, variant, out["seconds"], "s", flush=True)


def run_pooled(work: Path) -> None:
    """One model per target on the four coins stacked, with a coin indicator; the variant with
    the higher mean first-period score over the coins."""
    t0 = time.time()
    score = {v: np.mean([_choice_score(json.loads(
        (work / f"forecast3-{c}-{v}.json").read_text())) for c in COINS]) for v in VARIANTS}
    variant = max(score, key=score.get)
    Vs = {c: fp2.build(work, MINUTES, variant, c) for c in COINS}
    rows = []
    for i, c in enumerate(COINS):
        V = Vs[c]
        onehot = np.zeros((len(V["tau"]), len(COINS)))
        onehot[:, i] = 1
        rows.append(np.column_stack([V["X"], onehot]))
    X = np.vstack(rows)
    tau = np.concatenate([Vs[c]["tau"] for c in COINS])
    start_ok = np.concatenate([Vs[c]["tau"] >= Vs[c]["d"]["start"] for c in COINS])
    coin_of = np.concatenate([[i] * len(Vs[c]["tau"]) for i, c in enumerate(COINS)])
    models = _models()
    out: dict = {"variant": variant, "variant_scores": score, "real": {}}
    for t in _targets(Vs["BTC"]):
        h = t[1]
        y = np.concatenate([_targets(Vs[c])[t] for c in COINS])
        for name, (make, scaled, kind) in models.items():
            if kind != t[0]:
                continue
            p = np.full(len(tau), np.nan)
            for q0, q1 in zip(RETRAINS[:-1], RETRAINS[1:], strict=True):
                m = (tau + h * HOUR + (h * HOUR + DAY) <= q0.value) & ~np.isnan(y) & start_ok
                sel = (tau >= q0.value) & (tau < q1.value)
                if sel.any():
                    p[sel] = _fit_predict(make, scaled, kind, X, m, y, sel)
            for i, c in enumerate(COINS):
                rc = coin_of == i
                V = Vs[c]
                out["real"][f"{c} {_tn(t)} {name}"] = {
                    part: _boot(kind, p[rc], y[rc], mk, V["dates"])
                    for part, mk in fp2.parts(V["tau"], h).items()}
            print("pooled", _tn(t), name, round(time.time() - t0), "s", flush=True)
    out["seconds"] = round(time.time() - t0)
    (work / "forecast3-pooled.json").write_text(json.dumps(out, indent=1))
    print("written pooled", out["seconds"], "s", flush=True)


def _choice_score(r: dict) -> float:
    return float(max(np.mean([r["real"][f"z {h}h {mo}"]["first"][0] for h in (4, 24)])
                     for mo in ("ridge", "gbm")))


def _pct(x, draws):
    d = np.asarray([v for v in draws if not np.isnan(v)])
    return float((d < x).mean()), float(np.quantile(d, 0.95))


def combine(work: Path) -> None:
    res = {(c, v): json.loads((work / f"forecast3-{c}-{v}.json").read_text())
           for c in COINS for v in VARIANTS if (work / f"forecast3-{c}-{v}.json").exists()}
    pooled = (json.loads((work / "forecast3-pooled.json").read_text())
              if (work / "forecast3-pooled.json").exists() else None)
    out: dict = {"choice": {}, "table": {}, "passed": [], "pooled": {}}
    for c in COINS:
        have = [v for v in VARIANTS if (c, v) in res]
        if not have:
            continue
        sc = {v: _choice_score(res[(c, v)]) for v in have}
        v = max(sc, key=sc.get)
        out["choice"][c] = {"variant": v, "scores": sc}
        r = res[(c, v)]
        for h in H:
            row: dict = {}
            # direction
            mo = max(("ridge", "gbm"), key=lambda m_: r["real"][f"z {h}h {m_}"]["first"][0])
            d_ = {"model": mo}
            ok = True
            for part in ("first", "last"):
                ic = r["real"][f"z {h}h {mo}"][part]
                pc, q95 = _pct(ic[0], r["null"][f"z {h}h {mo}"][part])
                prev = r["previous"][f"z {h}h"][part][0]
                good = ic[0] > q95 and ic[0] > prev
                ok &= good
                d_[part] = {"ic": ic, "null_pct": pc, "null_q95": q95, "previous": prev,
                            "pass": good}
            d_["pass"] = ok
            row["direction"] = d_
            if ok:
                out["passed"].append(f"{c} direction {h}h")
            # classes
            fam = max(("logit", "gbc"), key=lambda f_: np.mean(
                [r["real"][f"{h}h {s} {f_}"]["first"][0] for s in SIDES]))
            base = max(("size logit", "size gbc"), key=lambda b_: np.mean(
                [r["size"][f"{h}h {s} {b_}"]["first"][0] for s in SIDES]))
            for s in SIDES:
                cs: dict = {"model": fam, "baseline": base}
                ok = True
                for part in ("first", "last"):
                    a_ = r["real"][f"{h}h {s} {fam}"][part]
                    b_ = r["size"][f"{h}h {s} {base}"][part]
                    pc, q95 = _pct(a_[0], r["null"][f"{h}h {s} {fam}"][part])
                    good = a_[0] > q95 and a_[0] > b_[0]
                    cs[part] = {"auc": a_, "size_auc": b_, "null_pct": pc, "null_q95": q95,
                                "above_null_and_size_point": good}
                    ok &= good
                cs["size_beyond_baseline_points"] = ok
                row[s] = cs
            fr = r["fall_minus_rise_vs_size"][f"{h}h {fam} vs {base}"]
            side = all(fr[p][1] > 0 for p in ("first", "last")) or all(
                fr[p][2] < 0 for p in ("first", "last"))
            row["fall_minus_rise_vs_size"] = {**fr, "pass": bool(side)}
            if side:
                out["passed"].append(f"{c} side {h}h")
            out["table"][f"{c} {h}h"] = row
        if pooled:
            for h in H:
                for mo in ("ridge", "gbm"):
                    pr_ = pooled["real"][f"{c} z {h}h {mo}"]
                    nd = res[(c, pooled["variant"])]["null"][f"z {h}h {mo}"] if (
                        c, pooled["variant"]) in res else None
                    row = {}
                    for part in ("first", "last"):
                        pc, q95 = _pct(pr_[part][0], nd[part]) if nd else (None, None)
                        row[part] = {"ic": pr_[part], "null_pct": pc, "null_q95": q95}
                    out["pooled"][f"{c} z {h}h {mo}"] = row
    n_coin = sum(1 for _ in res)
    out["trials"] = {"candidates": n_coin * 18 + len(out["choice"]) * 15
                     + (18 if pooled else 0),
                     "null_draws": n_coin * 18 * DRAWS}
    (work / "forecast3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"choice": out["choice"], "passed": out["passed"],
                      "trials": out["trials"]}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--coin", choices=COINS)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine:
        combine(a.work)
    elif a.pooled:
        run_pooled(a.work)
    else:
        run_coin(a.work, a.coin, a.variant)


if __name__ == "__main__":
    main()
