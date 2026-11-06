"""How the chosen pilot-3 forecast models behave on their own training data, on the boosting's
inner validation hold-out, on the next half-year and on the last third
(docs/research/forecast-size-1000-protocol-2026-11-06.md, part 2). Descriptive.

    uv run python scripts/research/forecast_fit_gap.py --work DIR

Writes DIR/fit-gap.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import forecast_pilot2 as fp2  # noqa: E402
import forecast_pilot3 as f3  # noqa: E402

from qlab import forecast as fc  # noqa: E402
from qlab import forecast_inputs as fi  # noqa: E402

H = (1, 4, 24)


def _inner_split(n: int, y: np.ndarray, classifier: bool) -> np.ndarray:
    """The rows sklearn's HistGradientBoosting (random_state=0, early stopping on, 10%) held out
    for early stopping, as positions in the training rows (its own code path, replayed)."""
    from sklearn.model_selection import train_test_split

    seed = np.random.RandomState(0).randint(np.iinfo(np.uint32).max, dtype="u8")
    _, val = train_test_split(np.arange(n), test_size=0.1, stratify=y if classifier else None,
                              random_state=seed)
    return np.sort(val)


def walk_with_fit(make, scaled, kind, X, tau, start, h, y, rng=None, center=False):
    """`forecast_pilot3._walk` that also reports, per half-year retrain, the fit on its own
    training rows, the boosting's inner hold-out, and the next half-year."""
    p = np.full(len(tau), np.nan)
    rows = []
    for q0, q1 in zip(f3.RETRAINS[:-1], f3.RETRAINS[1:], strict=True):
        m = (tau + h * f3.HOUR + (h * f3.HOUR + f3.DAY) <= q0.value) & ~np.isnan(y) & (
            tau >= start)
        sel = (tau >= q0.value) & (tau < q1.value)
        if not sel.any() or m.sum() < 1000:
            continue
        yy = y if rng is None else fi.shuffled_labels(y, m, tau, "other day", rng)
        if scaled:
            mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
            xt, xp = np.nan_to_num((X[m] - mu) / sd), np.nan_to_num((X[sel] - mu) / sd)
        else:
            xt, xp = X[m], X[sel]
        mod = make().fit(xt, yy[m])

        def pred(x, mod=mod):
            return mod.predict(x) if kind == "reg" else mod.predict_proba(x)[:, 1]

        score = (lambda a, b: fc.pearson(a, b)) if kind == "reg" else (
            lambda a, b: fc.auc(a, b))
        row = {"retrain": str(q0.date()), "train_rows": int(m.sum()),
               "in_sample": score(pred(xt), yy[m])}
        if hasattr(mod, "n_iter_"):
            row["trees_kept_of_200"] = int(mod.n_iter_)
            if getattr(mod, "do_early_stopping_", False):
                val = _inner_split(len(xt), yy[m], kind == "cls")
                row["inner_holdout"] = score(pred(xt[val]), yy[m][val])
                if kind == "reg":
                    # replay check: sklearn's own last validation score is minus half the MSE
                    mse = float(np.mean((yy[m][val] - pred(xt[val])) ** 2))
                    row["holdout_replay_ok"] = bool(np.isclose(-0.5 * mse,
                                                               mod.validation_score_[-1],
                                                               rtol=1e-6))
        out = pred(xp)
        if center:
            out = out - np.mean(yy[m])
        p[sel] = out
        row["next_half_year"] = score(out, y[sel])
        rows.append(row)
    return p, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    rec3 = json.loads((a.work / "forecast3.json").read_text())
    extra3 = json.loads((a.work / "forecast3-extra.json").read_text())
    models = f3._models()
    out: dict = {}
    for coin in f3.COINS:
        V = fp2.build(a.work, f3.MINUTES, rec3["choice"][coin]["variant"], coin)
        targets = list(f3._targets(V))
        ci = f3.COINS.index(coin)
        for h in H:
            parts = fp2.parts(V["tau"], h)
            # z return
            name = rec3["table"][f"{coin} {h}h"]["direction"]["model"]
            make, scaled, kind = models[name]
            y = V["targets"][h]
            entry = {"model": name}
            p, rows = walk_with_fit(make, scaled, kind, V["X"], V["tau"], V["d"]["start"], h, y)
            entry["real"] = {"per_retrain": rows,
                             "first": f3._point("reg", p, y, parts["first"]),
                             "last": f3._point("reg", p, y, parts["last"])}
            rng = np.random.default_rng(10_000 * (1 + ci) + 7 * targets.index(("reg", h, None)))
            q, rows = walk_with_fit(make, scaled, kind, V["X"], V["tau"], V["d"]["start"], h, y,
                                    rng)
            entry["shuffled_draw0"] = {"per_retrain": rows,
                                       "first": f3._point("reg", q, y, parts["first"]),
                                       "last": f3._point("reg", q, y, parts["last"])}
            out[f"{coin} z {h}h"] = entry
            # sign
            name = extra3["sign"][f"{coin} {h}h"]["model"]
            make, scaled, kind = models[name]
            fwd = fc.forward_return(V["logc"], h)
            up = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
            entry = {"model": name}
            p, rows = walk_with_fit(make, scaled, kind, V["X"], V["tau"], V["d"]["start"], h, up,
                                    center=True)
            entry["real"] = {"per_retrain": rows,
                             "first": f3._point("cls", p, up, parts["first"]),
                             "last": f3._point("cls", p, up, parts["last"])}
            rng = np.random.default_rng(50_000 * (1 + ci) + h)
            q, rows = walk_with_fit(make, scaled, kind, V["X"], V["tau"], V["d"]["start"], h, up,
                                    rng, center=True)
            entry["shuffled_draw0"] = {"per_retrain": rows,
                                       "first": f3._point("cls", q, up, parts["first"]),
                                       "last": f3._point("cls", q, up, parts["last"])}
            out[f"{coin} sign {h}h"] = entry
            print(coin, h, round(time.time() - t0), "s", flush=True)
    # the recorded pilot-3 numbers, to check the replay
    check = {}
    for key, e in out.items():
        coin, fam, hh = key.split()
        if fam == "z":
            rec = rec3["table"][f"{coin} {hh}"]["direction"]
            check[key] = [[rec["first"]["ic"][0], rec["last"]["ic"][0]],
                          [e["real"]["first"], e["real"]["last"]]]
        else:
            rec = extra3["sign"][f"{coin} {hh}"]
            check[key] = [[rec["first"]["auc"][0], rec["last"]["auc"][0]],
                          [e["real"]["first"], e["real"]["last"]]]
    res = {"series": out, "replay_check": check, "seconds": round(time.time() - t0)}
    (a.work / "fit-gap.json").write_text(json.dumps(res, indent=1))
    print("written", res["seconds"], "s")


if __name__ == "__main__":
    main()
