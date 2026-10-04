"""Put the detector variants tried on 2026-10-04 on record (docs/REGIME_DETECT.md).

They ran before `detector_trial` existed; every number below is copied from
the run's printed output that day (the scripts named), nothing recomputed or
rounded further. Run once; a second run would count them twice.
"""

from __future__ import annotations

import pandas as pd

from qlab import detector_trials as dt
from qlab.registry.db import session_scope

D = (pd.Timestamp("2020-04-01"), pd.Timestamp("2024-01-01"))
D8 = (pd.Timestamp("2020-08-01"), pd.Timestamp("2024-01-01"))
H = (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-10-01"))
WHOLE = (pd.Timestamp("2020-04-01"), pd.Timestamp("2026-10-01"))
NOTE = "copied from the printed output of 2026-10-04 (before detector_trial existed)"

L1 = "regime-level1"
ROWS = [
    # The first T39 detector, scored on the whole history before any split existed.
    (L1, "11 features, 3-class logistic (btc-market-logistic)", dt.DEVELOPMENT, WHOLE,
     {"acc": 0.58, "bear_recall": 0.55, "bear_precision": 0.60}, "qlab regimes detect", False,
     "scored on 2020-04..2026-09, the later holdout included"),
    # scripts/research/detector_features.py, development
    *[(L1, f"features: {v}, {m}", dt.DEVELOPMENT, D,
       {"acc": a, "bear_recall": br, "bear_precision": bp, "flat_recall": fr},
       "scripts/research/detector_features.py", False, None)
      for v, m, a, br, bp, fr in [
          ("current set", "logistic", .565, .570, .585, .529),
          ("only ret", "logistic", .607, .652, .608, .535),
          ("only short_ret", "logistic", .529, .564, .546, .471),
          ("only vol", "logistic", .315, .166, .302, .755),
          ("only drawdown", "logistic", .467, .464, .609, .245),
          ("only range", "logistic", .579, .622, .659, .497),
          ("only ma", "logistic", .584, .672, .632, .479),
          ("only shape", "logistic", .413, .259, .591, .553),
          ("only breadth", "logistic", .427, .613, .473, .173),
          ("only alts", "logistic", .486, .538, .528, .160),
          ("only funding", "logistic", .374, .829, .390, .170),
          ("only btc_funding", "logistic", .387, .294, .453, .585),
          ("only volume", "logistic", .388, .045, .244, .684),
          ("current without ret", "logistic", .439, .516, .495, .412),
          ("current without vol", "logistic", .547, .581, .563, .489),
          ("current without drawdown", "logistic", .550, .592, .565, .495),
          ("current without breadth", "logistic", .588, .603, .578, .527),
          ("current without funding", "logistic", .593, .585, .635, .612),
          ("current + short_ret", "logistic", .567, .581, .594, .535),
          ("current + range", "logistic", .565, .596, .585, .503),
          ("current + ma", "logistic", .577, .607, .598, .516),
          ("current + shape", "logistic", .556, .583, .564, .511),
          ("current + alts", "logistic", .550, .607, .593, .457),
          ("current + btc_funding", "logistic", .577, .577, .636, .508),
          ("current + volume", "logistic", .571, .583, .583, .535),
          ("everything", "logistic", .531, .600, .598, .412),
          ("current set", "boosting", .494, .503, .533, .449),
          ("everything", "boosting", .547, .568, .646, .500),
          ("current set", "ridge_return", .527, .445, .592, .721),
          ("everything", "ridge_return", .544, .479, .602, .585),
          ("ret + ma", "logistic", .607, .633, .600, .545),
          ("ret + range", "logistic", .597, .633, .602, .537),
          ("ret + lowcorr", "logistic", .558, .622, .605, .543),
          ("current + lowcorr", "logistic", .531, .609, .567, .455),
          ("only lowcorr", "logistic", .309, .162, .240, .378),
      ]],
    (L1, "features: only ret, logistic", dt.HOLDOUT, H,
     {"acc": .6209677419354839, "bear_recall": .6452702702702703,
      "bear_precision": .6063492063492063, "flat_recall": .5886363636363636},
     "scripts/research/detector_features.py", True, None),
    (L1, "features: current set, logistic", dt.HOLDOUT, H,
     {"acc": .6028225806451613, "bear_recall": .5304054054054054, "bear_precision": .628,
      "flat_recall": .6545454545454545}, "scripts/research/detector_features.py", False,
     "the stored detector, scored beside the chosen one"),
    (L1, "simple trailing-30d terciles", dt.HOLDOUT, H,
     {"acc": .5181451612903226, "bear_recall": .5236486486486487,
      "bear_precision": .5308219178082192, "flat_recall": .5431818181818182},
     "scripts/research/detector_features.py", False, "baseline"),
    # scripts/research/detector_binary.py (UP / DOWN pair)
    *[(L1, f"pair: {f}, {m}", dt.DEVELOPMENT, D,
       {"up_auc": ua, "up_precision": up, "up_recall": ur, "down_auc": da,
        "down_precision": dp, "down_recall": dr, "acc": a},
       "scripts/research/detector_binary.py", False, None)
      for f, m, ua, up, ur, da, dp, dr, a in [
          ("returns (5)", "logistic", .813, .85, .56, .791, .62, .56, .595),
          ("returns (5)", "neural net", .813, .77, .53, .685, .53, .50, .542),
          ("returns (5)", "boosting", .780, .74, .56, .784, .62, .54, .555),
          ("returns+ma+range (11)", "logistic", .786, .83, .56, .781, .62, .53, .582),
          ("returns+ma+range (11)", "neural net", .805, .72, .62, .787, .57, .73, .592),
          ("returns+ma+range (11)", "boosting", .776, .73, .53, .792, .59, .54, .539),
          ("old set (11)", "logistic", .797, .84, .56, .778, .61, .52, .570),
          ("old set (11)", "neural net", .797, .74, .56, .700, .55, .52, .553),
          ("old set (11)", "boosting", .762, .69, .48, .752, .60, .46, .517),
      ]],
    (L1, "pair: returns (5), logistic", dt.HOLDOUT, H,
     {"up_auc": .860, "up_precision": .74, "up_recall": .53, "down_auc": .821,
      "down_precision": .70, "down_recall": .51, "acc": .633},
     "scripts/research/detector_binary.py", True, None),
    (L1, "pair: returns+ma+range (11), neural net", dt.HOLDOUT, H,
     {"up_auc": .853, "up_precision": .75, "up_recall": .52, "down_auc": .817,
      "down_precision": .66, "down_recall": .58, "acc": .630},
     "scripts/research/detector_binary.py", False, "the best net, for the record"),
    # scripts/research/detector_timeframes.py (scale-free features, trained on a timeframe)
    *[(L1, f"pair trained on {tf}", p, w,
       {"up_auc": ua, "up_precision": up, "up_recall": ur, "down_auc": da,
        "down_precision": dp, "down_recall": dr, "acc": a},
       "scripts/research/detector_timeframes.py", c, n)
      for tf, p, w, ua, up, ur, da, dp, dr, a, c, n in [
          ("1d", dt.DEVELOPMENT, D, .808, .81, .57, .790, .62, .55, .580, False, None),
          ("4h", dt.DEVELOPMENT, D, .834, .83, .56, .831, .78, .45, .584, False, None),
          ("1h", dt.DEVELOPMENT, D, .834, .80, .58, .823, .76, .46, .583, False, None),
          ("4h", dt.HOLDOUT, H, .845, .64, .63, .811, .71, .55, .634, True, None),
          ("1d", dt.HOLDOUT, H, .854, .64, .61, .821, .67, .56, .623, False, "beside the chosen"),
          ("1h", dt.HOLDOUT, H, .856, .63, .64, .808, .71, .54, .630, False,
           "scored after the choice, on the owner's question"),
      ]],
    # scripts/research/detector_nets.py (trained on 1h, every 4th hour)
    *[(L1, f"1h-trained {m}", dt.DEVELOPMENT, D,
       {"up_auc": ua, "up_precision": up, "up_recall": ur, "down_auc": da,
        "down_precision": dp, "down_recall": dr, "acc": a},
       "scripts/research/detector_nets.py", False, None)
      for m, ua, up, ur, da, dp, dr, a in [
          ("logistic", .832, .80, .58, .822, .76, .45, .583),
          ("net 16", .815, .78, .56, .802, .76, .43, .566),
          ("net 64-32", .830, .81, .57, .807, .74, .43, .576),
          ("net 128-64-32", .809, .78, .56, .796, .74, .45, .571),
          ("net on sequence 64-32", .773, .71, .49, .788, .68, .48, .530),
      ]],
    # scripts/research/detector_seq.py
    (L1, "1h-trained LSTM 32", dt.DEVELOPMENT, D,
     {"up_auc": .793, "up_precision": .71, "up_recall": .40, "down_auc": .792,
      "down_precision": .71, "down_recall": .49, "acc": .504},
     "scripts/research/detector_seq.py", False, None),
    (L1, "1h-trained CNN 2x16", dt.DEVELOPMENT, D,
     {"up_auc": .596, "up_precision": .67, "up_recall": .04, "down_auc": .611,
      "down_precision": .56, "down_recall": .03, "acc": .291},
     "scripts/research/detector_seq.py", False, None),
    # scripts/research/detector_label_shift.py
    *[(L1, f"training label {b} back {f} ahead", dt.DEVELOPMENT, D,
       {"up_auc": ua, "down_auc": da, "acc": a, "bear_notice_days": bn, "bear_hold_days": bh,
        "bull_notice_days": un, "bull_hold_days": uh},
       "scripts/research/detector_label_shift.py", False, None)
      for b, f, ua, da, a, bn, bh, un, uh in [
          (15, 15, .832, .822, .583, 12, 1, 6, 1),
          (20, 10, .838, .833, .585, 6, 6, 6, 4),
          (25, 5, .825, .821, .575, 8, 14, 6, 16),
      ]],
    # scripts/research/detector_windows.py (accuracy on each window's own labels)
    *[(L1, f"window {w}d, returns scaled, 3-class logistic", p, per,
       {"acc": a, "ceiling": cl, "bear_notice_days": bn, "bear_hold_days": bh,
        "bull_notice_days": un, "bull_hold_days": uh},
       "scripts/research/detector_windows.py", c, None)
      for w, p, per, a, cl, bn, bh, un, uh, c in [
          (7, dt.DEVELOPMENT, D, .565, .598, 1, 1, 1, 2, False),
          (14, dt.DEVELOPMENT, D, .617, .631, 2, 2, 4, 2, False),
          (30, dt.DEVELOPMENT, D, .607, .636, 5, 7, 7, 2, False),
          (14, dt.HOLDOUT, H, .592, .608, 1, 2, 3, 2, True),
          (30, dt.HOLDOUT, H, .621, .655, 4, 7, 8, 2, False),
      ]],
]

TREND = "trend-gate-persistence"
ROWS += [
    *[(TREND, f"{kind}, penalty {pen}", dt.DEVELOPMENT, D8,
       {"ann_return": ann, "sharpe": sh, "max_dd": dd, "changes_per_year": ch},
       "scripts/research/persistence.py", False, None)
      for pen, ch, kind, ann, sh, dd in [
          (0.0, 57, "gate", .047, .43, -.216), (0.0, 57, "two levels", .021, .28, -.195),
          (0.5, 28, "gate", .036, .35, -.187), (0.5, 28, "two levels", .018, .24, -.172),
          (1.0, 23, "gate", .047, .43, -.196), (1.0, 23, "two levels", .006, .12, -.185),
          (2.0, 21, "gate", .069, .59, -.189), (2.0, 21, "two levels", .025, .35, -.180),
          (4.0, 17, "gate", .090, .75, -.169), (4.0, 17, "two levels", .018, .25, -.188),
          (8.0, 12, "gate", .113, .91, -.162), (8.0, 12, "two levels", .025, .32, -.153),
      ]],
    (TREND, "gate, penalty 8.0", dt.HOLDOUT, H,
     {"ann_return": .005, "sharpe": .152, "max_dd": -.039, "in_market": .470},
     "scripts/research/persistence.py", True, None),
]

HEDGE = "bv2-hedge-level2"
ROWS += [
    *[(HEDGE, f"horizon {d}d, {'with' if lv else 'without'} level 1", dt.DEVELOPMENT, D8,
       {"hedge_pays": pays, "model_right": m, "book_rule_right": b, "never_right": nv,
        "hedged": hd},
       "scripts/research/hedge_horizons.py", False, None)
      for d, lv, pays, m, b, nv, hd in [
          (7, True, .482, .514, .536, .518, .382), (7, False, .482, .504, .536, .518, .187),
          (14, True, .486, .514, .548, .514, .435), (14, False, .486, .496, .548, .514, .177),
          (30, True, .501, .488, .553, .499, .419), (30, False, .501, .434, .553, .499, .235),
          (60, True, .455, .438, .510, .545, .238), (60, False, .456, .443, .509, .544, .184),
      ]],
]


def main() -> None:
    with session_scope() as session:
        for family, variant, period, window, metrics, script, chosen, note in ROWS:
            dt.record(family, variant, period, window, metrics, script, chosen=chosen,
                      notes=f"{NOTE}; {note}" if note else NOTE, session=session)
    print(f"recorded {len(ROWS)} detector trials")


if __name__ == "__main__":
    main()
