"""Almost passed — the basis for graveyard near-threshold queries.

See docs/REGISTRY.md query 2: `near_threshold(margin)` — failures where
`|value - threshold| <= margin * |threshold|`.

That relative formula breaks down at `threshold == 0`: "25% closer to zero"
is not a meaningful distance. The only meaningful distance at a zero
threshold is an absolute one, and its size depends on the metric (0.1 is a
lot for a Sharpe ratio, 0.1 is 10 percentage points for an annual return) —
so it must be supplied explicitly per rule (`Rule.near_margin_abs`), never
guessed. `classify_nearness()` returns `UNDEFINED`, not `False`, when that
number hasn't been supplied: "not near" and "we don't know" are different
answers, and collapsing them lost real information (see the "almost passed
at Sharpe -0.22" incident this module's history refers to).
"""

from __future__ import annotations

from enum import Enum

from qlab.rules.schema import Comparator


def _as_comparator(comparator: Comparator | str) -> Comparator:
    return comparator if isinstance(comparator, Comparator) else Comparator(comparator)


def nearness(value: float, threshold: float) -> float:
    """How close `value` is to `threshold`, as a fraction of `threshold`.

    `|value - threshold| / |threshold|`, except when `threshold == 0`, where
    dividing by zero doesn't make sense — nearness then falls back to the
    plain absolute difference `|value - threshold|`. That fallback number is
    in the metric's own units, not a fraction, and must not be compared
    against a fractional margin — see `classify_nearness()`.
    """
    diff = abs(value - threshold)
    if threshold == 0:
        return diff
    return diff / abs(threshold)


class NearnessVerdict(Enum):
    """Three-valued answer to "did this failure come close to passing?".

    `NOT_NEAR` also covers values that actually passed `comparator` — a pass
    is not a near-miss, it's a pass. `UNDEFINED` means the question can't be
    answered at all with what was supplied: either `value` is missing, or
    the threshold is zero and no `margin_abs` was given (relative nearness
    is meaningless against a zero threshold, and no absolute margin was
    configured to use instead).
    """

    NEAR = "near"
    NOT_NEAR = "not_near"
    UNDEFINED = "undefined"


def classify_nearness(
    value: float | None,
    threshold: float,
    comparator: Comparator | str,
    margin: float | None = None,
    margin_abs: float | None = None,
) -> NearnessVerdict:
    """Classify how close a rule's `value` came to passing `comparator`.

    - `value is None` -> `UNDEFINED` (nothing to measure).
    - `value` already satisfies `comparator` -> `NOT_NEAR` (it's a pass, not
      a near-miss).
    - `threshold != 0` -> relative nearness (`nearness(value, threshold)`)
      compared against `margin`, the fraction of `|threshold|`. If `margin`
      wasn't supplied, `UNDEFINED` (no basis to compare against).
    - `threshold == 0` -> `margin` (a fraction of zero) is meaningless here.
      Absolute nearness (`|value - threshold|`) is compared against
      `margin_abs` instead. If `margin_abs` wasn't supplied, `UNDEFINED` —
      this is the honest answer when nobody has decided what "almost
      profitable" means for this metric, rather than silently reusing
      `margin` against the wrong units.
    """
    if value is None:
        return NearnessVerdict.UNDEFINED

    cmp = _as_comparator(comparator)
    if cmp.compare(value, threshold):
        return NearnessVerdict.NOT_NEAR

    if threshold == 0:
        if margin_abs is None:
            return NearnessVerdict.UNDEFINED
        is_close = abs(value - threshold) <= margin_abs
        return NearnessVerdict.NEAR if is_close else NearnessVerdict.NOT_NEAR

    if margin is None:
        return NearnessVerdict.UNDEFINED
    is_close = nearness(value, threshold) <= margin
    return NearnessVerdict.NEAR if is_close else NearnessVerdict.NOT_NEAR


def is_near(
    value: float | None,
    threshold: float,
    comparator: Comparator | str,
    margin: float | None = None,
    margin_abs: float | None = None,
) -> bool:
    """Thin boolean wrapper: `True` only when `classify_nearness()` is `NEAR`.

    WARNING: this collapses `NOT_NEAR` and `UNDEFINED` into the same
    `False`. "This clearly wasn't close" and "we can't tell, no margin was
    configured" look identical through this function. That's fine for a
    quick yes/no filter, but wrong for anything shown to a person — a
    `near_threshold()`-style query must call `classify_nearness()` directly
    so the undefined cases can be listed separately instead of silently
    disappearing into "not near".
    """
    verdict = classify_nearness(value, threshold, comparator, margin=margin, margin_abs=margin_abs)
    return verdict is NearnessVerdict.NEAR
