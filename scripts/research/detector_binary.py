"""Two one-question models instead of one three-way model (docs/TASKS.md T39;
owner, 2026-10-04: «я помню, что лучше обучать модель предсказывать 1 вещь.
Я хотел бы, чтобы ты разделил этот шаг на 2 модели — одна предсказывает
подъём, другая спад»; «DNN — вроде как лучше помогают»).

- the UP model answers "is today a bull day (the testing label)?",
- the DOWN model answers "is today a bear day?",
- combined: bull if only UP says yes, bear if only DOWN says yes, otherwise
  the more confident of the two when both say yes, flat when neither does.

Development only (2020-04 .. 2023-12), walk-forward with monthly refits on
labels known `EMBARGO_DAYS` before each month, exactly as
`qlab.regime_detect`. Each model is scored on its own question by ROC AUC
(how well it ranks bull days above the others; 0.5 is chance, threshold-free)
and by precision and recall at P > 0.5; the pair by three-way accuracy. The
holdout (2024-01 on) is scored once for the variant chosen here.

Usage: uv run python scripts/research/detector_binary.py MARKET_DIR [--holdout NAME]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _record import log  # noqa: E402
from detector_features import groups  # noqa: E402

from qlab import regime_detect as rd  # noqa: E402
from qlab.regimes import load, market_closes  # noqa: E402

DEV = (pd.Timestamp("2020-04-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC"))

FEATURE_SETS = {
    "returns (5)": ["ret"],
    "returns+ma+range (11)": ["ret", "ma", "range"],
    "old set (11)": ["ret", "vol", "drawdown", "breadth", "funding"],
}


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # scikit-learn defaults throughout; max_iter only lets the solver converge.
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        # A small neural network: one hidden layer of 100 (the library's
        # default), ReLU, Adam; early stopping on a tenth of the training
        # months so it cannot simply memorise them.
        "neural net": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            max_iter=2000, early_stopping=True, random_state=0)),
        "boosting": lambda: HistGradientBoostingClassifier(random_state=0),
    }


def walk_binary(btc: pd.Series, feats: pd.DataFrame, target: str, model: str,
                window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    """P(today's label is `target`) per day in `window`, walked forward."""
    make = _models()[model]
    feats = feats.dropna()
    out = []
    for month in pd.date_range(window[0], window[1], freq="MS", tz="UTC"):
        nxt = month + pd.offsets.MonthBegin(1)
        labels = rd.known_labels(btc, month)
        labels = labels[labels.index <= month - pd.Timedelta(days=rd.EMBARGO_DAYS)]
        train = feats.index.intersection(labels.index)
        y = (labels.loc[train] == target).astype(int)
        part = feats[(feats.index >= month) & (feats.index < min(nxt, window[1]))]
        if part.empty or y.nunique() < 2:
            continue
        fit = make().fit(feats.loc[train].to_numpy(), y.to_numpy())
        out.append(pd.Series(fit.predict_proba(part.to_numpy())[:, 1], index=part.index))
    return pd.concat(out)


def binary_scores(p: pd.Series, labels: pd.Series, target: str) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    both = pd.concat({"p": p, "l": labels}, axis=1).dropna()
    y = both["l"] == target
    said = both["p"] > 0.5
    return {"auc": float(roc_auc_score(y, both["p"])),
            "precision": float((said & y).sum() / max(said.sum(), 1)),
            "recall": float((said & y).sum() / max(y.sum(), 1)),
            "share_yes": float(said.mean())}


def combine(up: pd.Series, down: pd.Series) -> pd.Series:
    both = pd.concat({"u": up, "d": down}, axis=1).dropna()
    u, d = both["u"] > 0.5, both["d"] > 0.5
    label = np.where(u & ~d, "bull", np.where(d & ~u, "bear",
                     np.where(u & d, np.where(both["u"] >= both["d"], "bull", "bear"), "flat")))
    return pd.Series(label, index=both.index)


def main() -> None:
    market_dir = Path(sys.argv[1])
    btc = market_closes()
    labels = load().labels
    m = {k: pd.read_parquet(market_dir / f"mkt_{k}.parquet")
         for k in ("prices", "funding", "tradeable", "volume")}
    g = groups(btc, m)

    def frame(names: list[str]) -> pd.DataFrame:
        return pd.concat([g[n] for n in names], axis=1)

    def run(fset: str, model: str, window) -> None:
        feats = frame(FEATURE_SETS[fset])
        up = walk_binary(btc, feats, "bull", model, window)
        down = walk_binary(btc, feats, "bear", model, window)
        su, sd = binary_scores(up, labels, "bull"), binary_scores(down, labels, "bear")
        acc = rd.accuracy(combine(up, down), labels)["accuracy"]
        log("regime-level1", f"pair: {fset}, {model}", window,
            {"up_auc": su["auc"], "up_precision": su["precision"], "up_recall": su["recall"],
             "down_auc": sd["auc"], "down_precision": sd["precision"],
             "down_recall": sd["recall"], "acc": acc},
            "scripts/research/detector_binary.py", chosen=(fset, model) == CHOICE)
        print(f"{fset:<22} {model:<10} | UP auc {su['auc']:.3f} prec {su['precision']:.2f} "
              f"rec {su['recall']:.2f} | DOWN auc {sd['auc']:.3f} prec {sd['precision']:.2f} "
              f"rec {sd['recall']:.2f} | both acc {acc:.3f}", flush=True)

    if "--holdout" in sys.argv:
        fset, model = CHOICE
        run(fset, model, HOLDOUT)
        run("returns+ma+range (11)", "neural net", HOLDOUT)  # the best net, for the record
        return
    for fset in FEATURE_SETS:
        for model in _models():
            run(fset, model, DEV)


# Chosen 2026-10-04 from the development table, before the holdout was
# predicted: the best ranking on both questions (UP auc 0.813, DOWN 0.791) and
# the best pair accuracy (0.595); the neural net was no better and unstable on
# DOWN (0.685 on the same features).
CHOICE: tuple[str, str] = ("returns (5)", "logistic")
"""Filled in AFTER the development table is read, before --holdout is run."""

if __name__ == "__main__":
    main()
