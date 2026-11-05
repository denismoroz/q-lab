"""Bv3 as income (owner 2026-11-05: «как сделать то что я хотел в bv3 - не страховка - а
заработок»).

    uv run python scripts/research/bv3_income.py <step> --work DIR

`--work`: the folder of `hedge_trigger_study.py prepare` (set_a.pkl / set_b.pkl). Results go to
`DIR/bv3-income-<step>.json`. Each step's protocol is committed before its run
(docs/research/bv3-income-<n>-*-protocol-2026-11-05.md).
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

from qlab import bv3_income as bi  # noqa: E402
from qlab import earning_test  # noqa: E402
from qlab import hedge_followup as hf  # noqa: E402
from qlab import hedge_study as hs  # noqa: E402
from qlab.harness.run import run_backtest  # noqa: E402
from qlab.strategies.live.bv2 import B2Params  # noqa: E402

CUT = hts.CUT
PURGE = pd.Timedelta(days=15)
FAST = [(w, s) for w in (12, 24) for s in (0, 12)]


def stand(panel, run, capital):
    """The stand's result of a book run on the strategy's money (summed hours)."""
    base = (run.book_equity / capital).rename("capital_base")
    return run_backtest(panel, run.weights, hts.COSTS, panel.funding, capital_base=base,
                        capital_reinvested=False)


def measures(r: pd.Series, regime: np.ndarray | None = None) -> dict:
    """Yearly money, worst drawdown (points of the strategy's money), the earning test on the
    last third; by regime inside the first 2/3 (purged) when `regime` is given."""
    out = {}
    for part, sel in (("first", r.index < CUT), ("last", r.index >= CUT)):
        seg = r[sel]
        out[f"ann_{part}"] = float(100 * seg.sum() * 8760 / len(seg))
        out[f"dd_{part}"] = float(100 * hf.max_drawdown(seg.to_numpy()))
    test = earning_test.metrics_of(r[r.index >= CUT], additive=True)
    out["p_last"] = test.get(earning_test.P_KEY)
    if regime is not None:
        first = (r.index < CUT - PURGE)
        for reg in ("bull", "bear"):
            sel = first & (regime[: len(r)] == reg)
            out[f"ann_first_{reg}"] = float(100 * r[sel].sum() * 8760 / max(1, sel.sum()))
    return out


def targets_of(name: str, coins: list[str]) -> dict[str, float]:
    params = hts._params()
    if name == "set_a":
        p = B2Params.from_dict(params)
        return {c: p.book_sizes(c)[0] for c in coins}
    rest = {k: v for k, v in params.items() if k != "coins"}
    return {c: B2Params.from_dict({**rest, "coins": [c], "capital_usd": hts.PER_COIN_CAPITAL})
            .book_sizes(c)[0] for c in coins}


def fast_run(panel, bparams, a_window, sticky):
    from qlab.strategies.agent.fast_release_gate import FastReleaseGate

    return FastReleaseGate().run(panel, {**bparams, "fast_window_hours": a_window,
                                         "fast_entry": 0, "sticky_exit_hours": sticky})


def step0(work: Path) -> dict:
    out: dict = {"cut": str(CUT.date())}
    for name in ("set_a", "set_b"):
        data = hfu.load(work, name)
        panel = data["panel"]
        coins = [cd.coin for cd in data["coins"]]
        bparams, capital = hfu._book_params(name, coins)
        targets = targets_of(name, coins)
        res = {}
        rec = hs.run_book(panel, bparams)
        st = stand(panel, rec.run, capital)
        parts = bi.decompose(panel, rec.run.weights, rec.run.book_equity, capital, st.cost,
                             targets)
        gap = float((parts.sum(axis=1) - st.capital_return).abs().max())
        res["production"] = {"parts": bi.summarise_parts(parts, CUT),
                             "measures": measures(st.capital_return), "max_gap": gap}
        fast = {}
        for w, s in FAST:
            run = fast_run(panel, bparams, w, s)
            stf = stand(panel, run, capital)
            fast[f"{w}h sticky {s}"] = (run, stf, measures(stf.capital_return))
            print(name, w, s, json.dumps(fast[f"{w}h sticky {s}"][2]), flush=True)
        best = max(fast, key=lambda k: fast[k][2]["ann_first"])
        run, stf, m = fast[best]
        partsf = bi.decompose(panel, run.weights, run.book_equity, capital, stf.cost, targets)
        res["fast_release"] = {"all": {k: v[2] for k, v in fast.items()}, "best": best,
                               "parts": bi.summarise_parts(partsf, CUT), "measures": m}
        out[name] = res
        print(name, json.dumps(res["production"]), flush=True)
    return out


STEPS = {"0": step0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = STEPS[a.step](a.work)
    out["seconds"] = round(time.time() - t0)
    path = a.work / f"bv3-income-{a.step}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print("written", path, out["seconds"], "s")


if __name__ == "__main__":
    main()
