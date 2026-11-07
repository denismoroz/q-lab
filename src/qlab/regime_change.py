"""Regime and regime change instead of the next price
(docs/research/regime-change-protocol-2026-11-07.md;
owner, 2026-11-07: «нам не нужно предсказывать цену ... нам нужно предсказывать движение цены, смену
режима»). Pure functions; the runner is scripts/research/regime_change.py.

1. Labels (they read the future; the yardstick only): `zigzag`, `r1_state`, `r2_class`, `r3_class`,
   and the labels a past-only learner may use at a retrain time (`r1_known`, `switch_known`).
2. Detectors, past only: rules (`rule_bv2`, `rule_ma_cross`, `rule_drawdown`), a Gaussian HMM with
   filtered states (`GaussianHMM`, `hmm_filtered`), a statistical jump model with online states
   (`jump_fit`, `jump_online`).
3. The yardstick (`yardstick`, `auc`, `block_null_auc`).
4. Controls: the day-permuted history (`permute_days`), placebo relocation of episodes
   (`relocate_runs`), oracle degradation (`delay_wish`, `flip_noise`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365

# ── 1. labels ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Zigzag:
    pivots: list[tuple[int, str]]  # confirmed pivots (index, "peak"/"trough"), in order
    trend: str | None  # the direction of the current, unconfirmed leg
    extreme: int  # the index of the current leg's running extreme


def zigzag(v: np.ndarray, x: float) -> Zigzag:
    """The zig-zag of `v` with threshold `x` (the algorithm of bv3_parts.zigzag_falls): a pivot is
    confirmed once the price has turned by `x` from it. NaNs must be filled before."""
    pivots: list[tuple[int, str]] = []
    trend = None
    hi = lo = 0
    for i in range(1, len(v)):
        if v[i] > v[hi]:
            hi = i
        if v[i] < v[lo]:
            lo = i
        if trend in (None, "up") and v[i] <= v[hi] * (1 - x):
            pivots.append((hi, "peak"))
            trend, lo = "down", i
        elif trend in (None, "down") and v[i] >= v[lo] * (1 + x):
            pivots.append((lo, "trough"))
            trend, hi = "up", i
        if trend == "down" and v[i] < v[lo]:
            lo = i
        if trend == "up" and v[i] > v[hi]:
            hi = i
    extreme = lo if trend == "down" else hi
    return Zigzag(pivots, trend, extreme)


def r1_state(v: np.ndarray, x: float, fill_tail: bool = False) -> tuple[np.ndarray, Zigzag]:
    """R1: 1.0 in a down leg [peak, trough), 0.0 in an up leg [trough, peak); NaN before the first
    pivot and from the current leg's running extreme on (unknown), unless `fill_tail` (the
    oracle): then the tail keeps the current leg's direction."""
    z = zigzag(v, x)
    out = np.full(len(v), np.nan)
    if not z.pivots:
        return out, z
    for (p, kind), (q, _) in zip(z.pivots[:-1], z.pivots[1:], strict=True):
        out[p:q] = 1.0 if kind == "peak" else 0.0
    last, kind = z.pivots[-1]
    state = 1.0 if kind == "peak" else 0.0
    stop = len(v) if fill_tail else z.extreme
    out[last:stop] = state
    return out, z


def r1_legs(v: np.ndarray, x: float) -> list[tuple[int, int]]:
    """Confirmed down legs (peak index, trough index)."""
    z = zigzag(v, x)
    legs = []
    for (p, kind), (q, _) in zip(z.pivots[:-1], z.pivots[1:], strict=True):
        if kind == "peak":
            legs.append((p, q))
    return legs


def r1_known(v: np.ndarray, x: float) -> np.ndarray:
    """The R1 state a learner standing at the last element of `v` knows: NaN where not yet known
    (before the first pivot and from the running extreme on)."""
    return r1_state(v, x, fill_tail=False)[0]


def switch_known(v: np.ndarray, x: float, k: int, kinds: tuple[str, ...] = ("peak", "trough")
                 ) -> np.ndarray:
    """1.0 if a pivot of the given kinds lies in (t, t+k] (in grid steps), 0.0 if none can, NaN
    where a learner standing at the last element cannot yet know (t + k at or after the current
    leg's running extreme: the next pivot can only be there or later)."""
    z = zigzag(v, x)
    n = len(v)
    hit = np.zeros(n + k + 1)
    for p, kind in z.pivots:
        if kind in kinds:
            lo = max(p - k, 0)
            hit[lo:p] = 1.0  # t in [p-k, p) has p in (t, t+k]
    out = hit[:n].copy()
    first = z.pivots[0][0] if z.pivots else n
    out[: max(first - k, 0)] = np.nan  # before the first pivot the zig-zag has no history
    t = np.arange(n)
    out[t + k >= z.extreme] = np.where(out[t + k >= z.extreme] == 1.0, 1.0, np.nan)
    return out


def r2_class(logc_hourly: np.ndarray, at: np.ndarray, days: int = 30, trend_cut: float = 0.30,
             chop_cut: float = 0.15) -> np.ndarray:
    """R2 at hourly positions `at`: +2 trend up, -2 trend down, 0 chop, +-1 middle (sign of the
    move); NaN when the 30 days run past the data. Path sampled every 24 hours from t."""
    n = len(logc_hourly)
    out = np.full(len(at), np.nan)
    for j, h in enumerate(at):
        end = h + days * 24
        if end >= n:
            continue
        pts = logc_hourly[h:end + 1:24]
        if np.isnan(pts).any():
            continue
        net = pts[-1] - pts[0]
        path = np.abs(np.diff(pts)).sum()
        er = abs(net) / path if path > 0 else 0.0
        sgn = 1.0 if net > 0 else -1.0
        out[j] = 2 * sgn if er >= trend_cut else (0.0 if er < chop_cut else sgn)
    return out


def r3_class(logc_hourly: np.ndarray, index: pd.DatetimeIndex, at: np.ndarray, days: int = 30,
             min_past: int = 365) -> np.ndarray:
    """R3 at hourly positions `at`: forward 30-day volatility against the terciles of the coin's
    own PAST trailing 30-day volatility (daily samples of all previous days): 0 calm, 1 normal,
    2 turbulent; NaN without enough past or future."""
    r = pd.Series(np.diff(logc_hourly, prepend=np.nan))
    w = days * 24
    trail = r.rolling(w, min_periods=int(w * 0.9)).std().to_numpy()
    fwd = r[::-1].rolling(w, min_periods=int(w * 0.9)).std().to_numpy()[::-1]
    fwd = np.concatenate([fwd[1:], [np.nan]])  # returns (h, h+w]
    day = index.floor("D").asi8
    ends = np.flatnonzero(index.hour == 23)
    samp_day = day[ends]
    samp_val = trail[ends]
    out = np.full(len(at), np.nan)
    for j, h in enumerate(at):
        k = np.searchsorted(samp_day, day[h], side="left")  # days strictly before h's day
        past = samp_val[:k]
        past = past[~np.isnan(past)]
        if len(past) < min_past or np.isnan(fwd[h]):
            continue
        lo, hi = np.quantile(past, [1 / 3, 2 / 3])
        out[j] = 0.0 if fwd[h] < lo else (2.0 if fwd[h] > hi else 1.0)
    return out


# ── 2. detectors ─────────────────────────────────────────────────────────


def rule_bv2(logc: np.ndarray, steps_per_day: int) -> np.ndarray:
    """Bv2's own wish without the sticky exit: down unless both the 14- and 30-day returns > 0."""
    s = pd.Series(logc)
    r14 = (s - s.shift(14 * steps_per_day)).to_numpy()
    r30 = (s - s.shift(30 * steps_per_day)).to_numpy()
    out = np.where((r14 > 0) & (r30 > 0), 0.0, 1.0)
    out[np.isnan(r30)] = np.nan
    return out


def rule_ma_cross(logc: np.ndarray, steps_per_day: int, fast: int = 50, slow: int = 200
                  ) -> np.ndarray:
    p = pd.Series(np.exp(logc))
    f = p.rolling(fast * steps_per_day, min_periods=fast * steps_per_day).mean()
    s = p.rolling(slow * steps_per_day, min_periods=slow * steps_per_day).mean()
    out = (f < s).astype(float).to_numpy().copy()
    out[s.isna().to_numpy()] = np.nan
    return out


def rule_drawdown(v: np.ndarray, x: float) -> np.ndarray:
    """The online zig-zag: down once the price fell `x` from its high since the last up switch,
    up once it rose `x` from its low since the last down switch. Starts up."""
    out = np.zeros(len(v))
    down = False
    hi = lo = v[0]
    for i, p in enumerate(v):
        if np.isnan(p):
            out[i] = 1.0 if down else 0.0
            continue
        if down:
            lo = min(lo, p)
            if p >= lo * (1 + x):
                down, hi = False, p
        else:
            hi = max(hi, p)
            if p <= hi * (1 - x):
                down, lo = True, p
        out[i] = 1.0 if down else 0.0
    return out


# Gaussian HMM, diagonal covariances, EM with scaling (filtered states only at inference)


@dataclass
class GaussianHMM:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray  # K x D
    var: np.ndarray  # K x D
    loglik: float = float("nan")

    def emission(self, X: np.ndarray) -> np.ndarray:
        """log b_t(k), T x K."""
        d = X[:, None, :] - self.mu[None, :, :]
        return -0.5 * (np.log(2 * np.pi * self.var)[None] + d * d / self.var[None]).sum(-1)


def _forward(logb: np.ndarray, pi: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, K = logb.shape
    m = logb.max(1, keepdims=True)
    b = np.exp(logb - m)
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    At = A
    for t in range(1, T):
        a = (alpha[t - 1] @ At) * b[t]
        s = a.sum()
        c[t] = s
        alpha[t] = a / s
    return alpha, np.log(c) + m[:, 0]


def _backward(logb: np.ndarray, A: np.ndarray, c_log: np.ndarray) -> np.ndarray:
    T, K = logb.shape
    m = logb.max(1, keepdims=True)
    b = np.exp(logb - m)
    c = np.exp(c_log - m[:, 0])
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def hmm_fit(X: np.ndarray, K: int, rng: np.random.Generator, starts: int = 5, iters: int = 200,
            tol: float = 1e-4, init: GaussianHMM | None = None, var_floor: float = 1e-3
            ) -> GaussianHMM:
    """EM on one sequence; the best of `starts` random starts (plus `init` when given)."""
    T, D = X.shape
    best = None
    inits = []
    if init is not None:
        inits.append(GaussianHMM(init.pi.copy(), init.A.copy(), init.mu.copy(), init.var.copy()))
    while len(inits) < starts + (init is not None):
        idx = rng.choice(T, size=K, replace=False)
        A = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(A, 0.95)
        inits.append(GaussianHMM(np.full(K, 1.0 / K), A, X[idx].copy(),
                                 np.tile(X.var(0), (K, 1))))
    for m in inits:
        prev = -np.inf
        for _ in range(iters):
            logb = m.emission(X)
            alpha, c_log = _forward(logb, m.pi, m.A)
            ll = float(c_log.sum())
            beta = _backward(logb, m.A, c_log)
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            mb = np.exp(logb - logb.max(1, keepdims=True))
            cc = np.exp(c_log - logb.max(1))
            xi = (alpha[:-1, :, None] * m.A[None] * (mb[1:] * beta[1:])[:, None, :]
                  / cc[1:, None, None]).sum(0)
            m.pi = gamma[0] + 1e-6
            m.pi /= m.pi.sum()
            m.A = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            m.mu = (gamma.T @ X) / w[:, None]
            m.var = np.maximum((gamma.T @ (X * X)) / w[:, None] - m.mu ** 2, var_floor)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        m.loglik = ll
        if best is None or m.loglik > best.loglik:
            best = m
    return best


def hmm_filtered(m: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """P(state | data up to t), T x K (the forward pass only)."""
    return _forward(m.emission(X), m.pi, m.A)[0]


# statistical jump model (Nystrup et al.; Aydinhan, Kolm, Mulvey, Shu): squared distance + lambda
# per switch


def _jump_path(loss: np.ndarray, lam: float) -> np.ndarray:
    T, K = loss.shape
    V = np.empty((T, K))
    back = np.empty((T, K), dtype=int)
    V[0] = loss[0]
    pen = lam * (1 - np.eye(K))
    for t in range(1, T):
        cand = V[t - 1][:, None] + pen  # from j to k
        back[t] = cand.argmin(0)
        V[t] = loss[t] + cand.min(0)
    s = np.empty(T, dtype=int)
    s[-1] = int(V[-1].argmin())
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s


def _loss(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = X[:, None, :] - C[None]
    return 0.5 * (d * d).sum(-1)


def jump_fit(X: np.ndarray, K: int, lam: float, rng: np.random.Generator, starts: int = 10,
             iters: int = 30) -> np.ndarray:
    """Centres (K x D) of the jump model on X: alternate the best path given the centres and the
    centres given the path; the best objective of `starts` k-means++ starts."""
    T = len(X)
    best, best_obj = None, np.inf
    for _ in range(starts):
        C = [X[rng.integers(T)]]
        for _k in range(1, K):
            d = _loss(X, np.array(C)).min(1)
            C.append(X[rng.choice(T, p=d / d.sum())])
        C = np.array(C)
        s_prev = None
        for _ in range(iters):
            s = _jump_path(_loss(X, C), lam)
            if s_prev is not None and np.array_equal(s, s_prev):
                break
            s_prev = s
            for k in range(K):
                if (s == k).any():
                    C[k] = X[s == k].mean(0)
        obj = _loss(X, C)[np.arange(T), s].sum() + lam * (np.diff(s) != 0).sum()
        if obj < best_obj:
            best, best_obj = C.copy(), obj
    return best


def jump_online(X: np.ndarray, C: np.ndarray, lam: float, V0: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """The online state at each t: the last state of the best path over data up to t (the
    forward pass). Returns (states, the last value vector, to continue from)."""
    loss = _loss(X, C)
    T, K = loss.shape
    pen = lam * (1 - np.eye(K))
    s = np.empty(T, dtype=int)
    V = loss[0].copy() if V0 is None else loss[0] + (V0[:, None] + pen).min(0)
    s[0] = int(V.argmin())
    for t in range(1, T):
        V = loss[t] + (V[:, None] + pen).min(0)
        V = V - V.min()
        s[t] = int(V.argmin())
    return s, V


# ── 3. yardstick ─────────────────────────────────────────────────────────


def yardstick(logc: np.ndarray, truth: np.ndarray, det: np.ndarray, legs: list[tuple[int, int]],
              mask: np.ndarray, step_hours: float) -> dict:
    """Detection measures of one coin on the grid points in `mask` (see the protocol, section 4).
    Returns sums so coins can be pooled (`pool`)."""
    ok = mask & ~np.isnan(truth) & ~np.isnan(det)
    t1, d1 = truth == 1.0, det == 1.0
    out = {"n": int(ok.sum()), "n_down": int((ok & t1).sum()), "n_up": int((ok & ~t1).sum()),
           "tp": int((ok & t1 & d1).sum()), "tn": int((ok & ~t1 & ~d1).sum()),
           "years": float(ok.sum() * step_hours / HOURS_PER_YEAR)}
    dl = np.zeros(len(logc))
    dl[:-1] = np.diff(logc)
    dl = np.nan_to_num(dl)
    out["fall_total"] = float(-(dl[ok & t1]).sum())
    out["fall_caught"] = float(-(dl[ok & t1 & d1]).sum())
    out["rise_total"] = float(dl[ok & ~t1].sum())
    out["rise_lost"] = float(dl[ok & ~t1 & d1].sum())
    # legs whose peak is in the mask
    delays, lost, detected = [], [], 0
    for p, q in legs:
        if not ok[p]:
            continue
        seg = np.flatnonzero(d1[p:q + 1])
        if len(seg):
            j = p + seg[0]
            detected += 1
            delays.append((j - p) * step_hours / 24)
            depth = logc[p] - logc[q]
            lost.append(float((logc[p] - logc[j]) / depth) if depth > 0 else 0.0)
        else:
            lost.append(1.0)
    out["legs"] = len(lost)
    out["detected"] = detected
    out["delays"] = delays
    out["lost"] = lost
    # false switches: an up->down switch inside the mask whose whole down run lies in up truth
    false = 0
    switches = 0
    i = 1
    n = len(det)
    while i < n:
        if d1[i] and not d1[i - 1] and ok[i]:
            switches += 1
            j = i
            while j < n and d1[j]:
                j += 1
            seg = truth[i:j]
            if len(seg) and np.all(seg == 0.0):
                false += 1
            i = j
        else:
            i += 1
    out["switches"] = switches
    out["false"] = false
    return out


def pool(parts: list[dict]) -> dict:
    """Pooled measures over coins."""
    s = {k: sum(p[k] for p in parts) for k in ("n", "n_down", "n_up", "tp", "tn", "years",
                                              "fall_total", "fall_caught", "rise_total",
                                              "rise_lost", "legs", "detected", "switches",
                                              "false")}
    delays = [d for p in parts for d in p["delays"]]
    lost = [x for p in parts for x in p["lost"]]
    tpr = s["tp"] / s["n_down"] if s["n_down"] else np.nan
    tnr = s["tn"] / s["n_up"] if s["n_up"] else np.nan
    return {"ba": float((tpr + tnr) / 2), "tpr": float(tpr), "tnr": float(tnr),
            "wrong_share": float(1 - (s["tp"] + s["tn"]) / s["n"]) if s["n"] else np.nan,
            "legs": s["legs"], "detected_share": s["detected"] / s["legs"] if s["legs"] else np.nan,
            "delay_median_days": float(np.median(delays)) if delays else np.nan,
            "delay_mean_days": float(np.mean(delays)) if delays else np.nan,
            "fall_lost_mean": float(np.mean(lost)) if lost else np.nan,
            "false_per_coin_year": s["false"] / s["years"] if s["years"] else np.nan,
            "switches_per_coin_year": s["switches"] / s["years"] if s["years"] else np.nan,
            "fall_caught_share": s["fall_caught"] / s["fall_total"] if s["fall_total"] else np.nan,
            "rise_lost_share": s["rise_lost"] / s["rise_total"] if s["rise_total"] else np.nan,
            "coin_years": s["years"]}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    ok = ~np.isnan(score) & ~np.isnan(label)
    s, y = score[ok], label[ok]
    n1 = int((y == 1).sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_permute(label: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(label)
    starts = np.arange(0, n, block)
    order = rng.permutation(len(starts))
    return np.concatenate([label[starts[k]:starts[k] + block] for k in order])[:n]


def block_null_auc(scores: list[np.ndarray], labels: list[np.ndarray], block: int, draws: int,
                   seed: int) -> np.ndarray:
    """Pooled AUC under labels permuted in blocks, per coin (the scores stay)."""
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    s_all = np.concatenate(scores)
    for d in range(draws):
        y = np.concatenate([block_permute(lab, block, rng) for lab in labels])
        out[d] = auc(s_all, y)
    return out


# ── 4. controls ──────────────────────────────────────────────────────────


def permute_days(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The coin's hourly frame with whole UTC days in a random order (index unchanged). The price
    is rebuilt from the permuted hourly log returns; every other column moves with its day.
    Expects columns `logc` (log close) and others; only full days are permuted."""
    n = len(frame)
    days = n // 24
    lr = np.diff(frame["logc"].to_numpy(), prepend=np.nan)
    lr = np.nan_to_num(lr)
    order = rng.permutation(days)
    take = (order[:, None] * 24 + np.arange(24)[None]).ravel()
    take = np.concatenate([take, np.arange(days * 24, n)])
    out = frame.iloc[take].copy()
    out.index = frame.index
    lr_p = lr[take]
    lr_p[0] = 0.0
    out["logc"] = frame["logc"].iloc[0] + np.cumsum(lr_p)
    return out


def relocate_runs(det: np.ndarray, rng: np.random.Generator, valid: np.ndarray) -> np.ndarray:
    """A placebo: the down runs of `det` (same count, same lengths) placed at random, without
    overlaps, inside the `valid` stretch (contiguous)."""
    d1 = np.nan_to_num(det) == 1.0
    runs = []
    i = 0
    while i < len(d1):
        if d1[i]:
            j = i
            while j < len(d1) and d1[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    idx = np.flatnonzero(valid)
    lo, hi = idx[0], idx[-1] + 1
    span = hi - lo
    free = span - sum(runs)
    out = np.where(np.isnan(det), np.nan, 0.0)
    if free < 0:
        return det.copy()
    order = rng.permutation(len(runs))
    gaps = np.sort(rng.integers(0, free + 1, size=len(runs)))
    pos = lo
    prev_gap = 0
    for g, k in zip(gaps, order, strict=True):
        pos += g - prev_gap
        prev_gap = g
        out[pos:pos + runs[k]] = 1.0
        pos += runs[k]
    return out


def delay_wish(w: np.ndarray, hours: int) -> np.ndarray:
    if hours == 0:
        return w.copy()
    out = np.full(len(w), np.nan)
    out[hours:] = w[:-hours]
    return out


def flip_noise(w: np.ndarray, per_year: float, rng: np.random.Generator, min_days: int = 3,
               max_days: int = 10) -> np.ndarray:
    """`per_year` random episodes per year (hourly array), each flipping the wish for 3-10 days."""
    out = w.copy()
    valid = np.flatnonzero(~np.isnan(w))
    if per_year <= 0 or not len(valid):
        return out
    lo, hi = valid[0], valid[-1] + 1
    years = (hi - lo) / HOURS_PER_YEAR
    n_ep = rng.poisson(per_year * years)
    for _ in range(n_ep):
        s = int(rng.integers(lo, hi))
        length = int(rng.integers(min_days * 24, max_days * 24 + 1))
        seg = out[s:min(s + length, hi)]
        out[s:min(s + length, hi)] = np.where(np.isnan(seg), np.nan, 1.0 - w[s:min(s + length, hi)])
    return out


__all__ = [
    "GaussianHMM", "Zigzag", "auc", "block_null_auc", "block_permute", "delay_wish", "flip_noise",
    "hmm_filtered", "hmm_fit", "jump_fit", "jump_online", "permute_days", "pool", "r1_known",
    "r1_legs", "r1_state", "r2_class", "r3_class", "relocate_runs", "rule_bv2", "rule_drawdown",
    "rule_ma_cross", "switch_known", "yardstick", "zigzag",
]
