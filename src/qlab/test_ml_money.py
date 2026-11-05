"""The money-trained hedge policy reads only the past: channels, the decision grid, the windows and
the summaries at a decision do not change when later minutes do; the money of a step is the
simplified book's."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from qlab import ml_money as mm


def _minutes(n=6000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999,
                         "volume": rng.uniform(1, 10, n), "taker_buy_base": rng.uniform(0, 1, n),
                         "trades": rng.integers(1, 100, n)}, index=idx)


def test_channels_windows_and_summaries_do_not_read_later_minutes() -> None:
    m = _minutes()
    ch = mm.minute_channels(m)
    later = m.copy()
    later.iloc[4000:] *= 3.0
    ch2 = mm.minute_channels(later)
    assert np.array_equal(ch[:4000], ch2[:4000])
    grid = mm.decision_grid(m.index, m["close"].to_numpy(), 15)
    early = grid[grid["pos"] < 3999]
    pos = early["pos"].to_numpy()
    assert np.array_equal(mm.windows(ch, pos), mm.windows(ch2, pos))
    assert np.allclose(mm.window_summary(ch, pos), mm.window_summary(ch2, pos))


def test_the_grid_reads_the_last_minute_before_the_decision_and_the_next_step_return() -> None:
    m = _minutes(300)
    grid = mm.decision_grid(m.index, m["close"].to_numpy(), 15)
    row = grid.iloc[2]
    assert m.index[row["pos"]] == row["tau"] - pd.Timedelta(minutes=1)
    p0 = m["close"].iloc[row["pos"]]
    p1 = m["close"].iloc[row["pos"] + 15]
    assert row["ret"] == pytest.approx(p1 / p0 - 1)


def test_the_unrolled_money_is_the_simplified_book() -> None:
    model = mm.make_policy("linear", 3)
    with torch.no_grad():
        model.lin.weight.zero_()
        model.lin.bias.fill_(100.0)  # always fully hedged
    enc = torch.zeros(1, 3, 3 + mm.CHANNELS)
    ret = torch.tensor([[0.01, -0.02, 0.03]])
    fund = torch.tensor([[0.0, 0.001, 0.0]])
    shares, money = mm.unroll(model, enc, ret, fund, torch.tensor([1.0]), torch.tensor([0.0]))
    assert torch.allclose(shares, torch.ones(1, 3))
    assert money[0, 0].item() == pytest.approx(-mm.COST)  # pay to open, no price risk
    assert money[0, 1].item() == pytest.approx(0.001, abs=1e-9)


def test_the_tcn_encoder_is_causal() -> None:
    torch.manual_seed(0)
    model = mm.make_policy("tcn", 2)
    w = torch.randn(4, mm.CHANNELS, mm.WINDOW)
    s = torch.zeros(4, 2)
    base = model.encode(w, None, s)
    assert base.shape == (4, 8 + 2)
    # the encoder reads the window's last minute through its receptive field of 4 * (1+4+16) + 1
    w2 = w.clone()
    w2[:, :, -1] += 5.0
    assert not torch.allclose(model.encode(w2, None, s), base)


def test_training_sequences_end_before_the_embargo_and_are_contiguous() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
    import ml_money as runner

    n = 400
    tau = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC").as_unit("ns").asi8
    tau = np.delete(tau, [100])  # a missing decision breaks contiguity
    g = {"tau": tau, "pos": np.arange(len(tau)), "ret": np.zeros(len(tau)),
         "valid": np.ones(len(tau), bool), "ok": np.ones(len(tau), bool),
         "static": np.zeros((len(tau), 21), np.float32), "summ": np.zeros((len(tau), 5)),
         "fund": np.zeros(len(tau)), "prod": np.zeros(len(tau))}
    d = {c: {"g60": g, "ch": np.zeros((10, 5), np.float32), "scale": 0.2, "coin": k,
             "start": int(tau[0])} for k, c in enumerate(runner.COINS)}
    data = runner.Data(d, 60)
    hi = int(tau[300])
    starts = data.starts("BTC", int(tau[0]), hi, 48)
    assert len(starts) > 0
    # every sequence's last decision plus its step ends by `hi`, and no sequence spans the hole
    assert (tau[starts + 47] + 60 * 60_000_000_000 <= hi).all()
    assert not ((starts <= 99) & (starts + 47 >= 100)).any()


def test_a_batch_of_sequences_keeps_one_money_per_sequence_and_step() -> None:
    model = mm.make_policy("linear", 3)
    enc = torch.zeros(5, 7, 3 + mm.CHANNELS)
    scale = torch.tensor([1.0, 2.0, 0.0, 1.0, 1.0])
    shares, money = mm.unroll(model, enc, torch.full((5, 7), 0.01), torch.zeros(5, 7), scale,
                              torch.zeros(5))
    assert shares.shape == (5, 7) and money.shape == (5, 7)
    assert torch.all(money[2] == 0)  # a zero scale earns nothing: rows are not mixed
