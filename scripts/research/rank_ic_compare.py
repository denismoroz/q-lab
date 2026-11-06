"""Apples-to-apples with the independent rebuild: rank IC (Spearman) and sign hit rate of the
pilot-3 chosen models at 1 h and 4 h, per coin and period, next to the 1-hour reversal baseline.
Descriptive.

    uv run python scripts/research/rank_ic_compare.py --work DIR

Writes DIR/rank-ic.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

COINS = ("BTC", "ETH", "SOL", "AVAX")
H = (1, 4)


def main() -> None:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    rec3 = json.loads((a.work / "forecast3.json").read_text())
    extra3 = json.loads((a.work / "forecast3-extra.json").read_text())
    models = f3._models()
    out: dict = {}
    for coin in COINS:
        V = fp2.build(a.work, f3.MINUTES, rec3["choice"][coin]["variant"], coin)
        tau, X, start, logc = V["tau"], V["X"], V["d"]["start"], V["logc"]
        r1_past = logc - np.r_[np.nan, logc[:-1]]  # the last hour's log return
        for h in H:
            parts = fp2.parts(tau, h)
            fwd = fc.forward_return(logc, h)
            z = V["targets"][h]
            up = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
            zm = rec3["table"][f"{coin} {h}h"]["direction"]["model"]
            sm = extra3["sign"][f"{coin} {h}h"]["model"]
            pz = f3._walk(*models[zm], X, tau, start, h, z)
            ps = f3._walk(*models[sm], X, tau, start, h, up, center=True)
            preds = {f"z model ({zm})": pz, f"sign model ({sm})": ps,
                     "reversal: minus the last hour's return": -r1_past}
            for name, p in preds.items():
                row = {}
                for part, mk in parts.items():
                    ok = mk & ~np.isnan(p) & ~np.isnan(fwd)
                    okz = ok & ~np.isnan(z)
                    row[part] = {"rank_ic_raw": fc.spearman(p[ok], fwd[ok]),
                                 "rank_ic_z": fc.spearman(p[okz], z[okz]),
                                 "pearson_raw": fc.pearson(p[ok], fwd[ok]),
                                 "hit": fc.hit_rate(p[ok], fwd[ok]), "n": int(ok.sum())}
                out[f"{coin} {h}h {name}"] = row
            print(coin, h, flush=True)
    (a.work / "rank-ic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
