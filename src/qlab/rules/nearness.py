"""Almost passed — the basis for graveyard near-threshold queries.

See docs/REGISTRY.md query 2: `near_threshold(margin)` — failures where
`|value - threshold| <= margin * |threshold|`.
"""

from __future__ import annotations

from qlab.rules.schema import Comparator


def nearness(value: float, threshold: float) -> float:
    """How close `value` is to `threshold`, as a fraction of `threshold`.

    `|value - threshold| / |threshold|`, except when `threshold == 0`, where
    dividing by zero doesn't make sense — nearness then falls back to the
    plain absolute difference `|value - threshold|`.
    """
    diff = abs(value - threshold)
    if threshold == 0:
        return diff
    return diff / abs(threshold)


def is_near(
    value: float | None,
    threshold: float,
    comparator: Comparator | str,
    margin: float,
) -> bool:
    """Was `value` a *failing* verdict that came close to passing?

    A value that already satisfies `comparator` isn't a "near miss" — it's a
    pass — so this returns `False` for it regardless of `margin`. A missing
    value (`None`) is unknown, not near, so it also returns `False`.
    """
    if value is None:
        return False
    cmp = comparator if isinstance(comparator, Comparator) else Comparator(comparator)
    if cmp.compare(value, threshold):
        return False
    return nearness(value, threshold) <= margin
