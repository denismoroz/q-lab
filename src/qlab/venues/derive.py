"""Pure functions: venue facts + a spec's leg count -> the three preflight
metrics `qlab.rules` needs and no backtest can produce
(`venue_supported`, `data_forward_available`, `atomic_execution`) — see
`docs/TASKS.md` task T13 and `rules/2026-09-20.1.yaml`'s `preflight` stage
for why these exist and why guessing them is not an option.

Every function here returns `bool | None`. `None` means "unknown" and must
travel unchanged into `Evaluation.metrics` as an ABSENT key (never `0.0`,
never a guessed default) — `qlab.rules.engine.evaluate` already treats a
missing metric as "verdict unknown," which is exactly the honest
`needs-more-data` outcome this task exists to preserve. `derive_venue_metrics`
does the "drop unknowns" step once, in one place, so `qlab.pipeline.evaluate`
does not have to repeat that filtering logic for each of the three metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

from qlab.venues.config import VenueConfig

#: How many instrument names a failure explanation spells out before it
#: says "and N more".
EXPLAINED_INSTRUMENTS = 8


def venue_supported(venue: VenueConfig | None) -> bool | None:
    """SCREENING.md §2: «площадка: есть ли инфраструктура (сейчас только
    HL)».

    This is just `venue.execution_adapter`, passed through as-is: `True`
    only if the config says so explicitly, `False` if it explicitly says
    there is no adapter, and `None` (unknown) if there is no venue config at
    all or the field was left unset — never guessed as `False` "to be
    safe," since an unconfigured venue is a missing fact, not a known "no."
    """
    if venue is None:
        return None
    return venue.execution_adapter


def data_forward_available(
    venue: VenueConfig | None, *, snapshot_source: str, venue_id: str
) -> bool | None:
    """SCREENING.md §2: «данные: бесплатные и доступные forward, а не
    только исторически».

    Two things must both be true, and both must be KNOWN:

      1. the venue's own data is free and available going forward
         (`venue.free_public_data`);
      2. the snapshot this evaluation actually ran on was sourced from THIS
         venue (`snapshot_source == venue_id`) — a strategy backtested on
         venue A's data cannot honestly inherit venue B's data-availability
         fact just because it is meant to eventually run on B.

    Returns `None` when `venue` is missing or `free_public_data` was left
    unset (fact 1 unknown). Returns `False` outright when the snapshot's
    source does not match the venue, even if `free_public_data` is `True`
    for that venue in the abstract — a known mismatch is a known "no," not
    an unknown.
    """
    if venue is None or venue.free_public_data is None:
        return None
    if snapshot_source != venue_id:
        return False
    return venue.free_public_data


def atomic_execution(venue: VenueConfig | None, *, simultaneous_legs: int) -> bool | None:
    """SCREENING.md §2: «атомарность: не остаётся ли неприкрытая нога при
    сбое» — note this asks whether an uncovered leg REMAINS after a
    failure, not whether fills happen simultaneously. That distinction is
    load-bearing: taken as "simultaneous fills," this rule would reject
    FRAB outright (Hyperliquid fills spot and perp as two separate,
    sequential orders — see `venues/hyperliquid.yaml`'s
    `supports_atomic_multileg` citation), even though FRAB is the strategy
    actually running and earning today. Read as "no uncovered leg persists
    after a failure," it does not: the live engine actively unwinds
    whichever leg opened before marking the entry FAILED (see
    `venues/hyperliquid.yaml`'s `uncovered_leg_recovery` citation).

    `simultaneous_legs` (`qlab.pipeline.spec.StrategySpec.simultaneous_legs`)
    says how many legs this strategy needs open TOGETHER for one entry to
    make economic sense:

      - `simultaneous_legs <= 1`: trivially satisfied — there is no
        "uncovered partner leg" a single-leg entry could ever strand, so
        this returns `True` without even looking at the venue.
      - `simultaneous_legs > 1`: satisfied if the venue can fill every leg
        as one all-or-nothing unit (`supports_atomic_multileg is True`), OR
        the venue names a real recovery mechanism that prevents an
        uncovered leg from persisting after a failure
        (`uncovered_leg_recovery` is a non-blank string). Neither ->
        `False`. If `supports_atomic_multileg` is unknown (`None`) and no
        recovery mechanism is named either, the honest answer is `None`
        (unknown), not `False` — see the docstring note on fail-open vs.
        fail-closed below.
      - No venue config at all, and `simultaneous_legs > 1`: `None`
        (unknown) — never guessed.

    Fail-closed note: `supports_atomic_multileg is False` together with an
    unset `uncovered_leg_recovery` returns `False` (known: neither atomic
    nor recovered), matching this rule's `fatal: true` in
    `rules/2026-09-20.1.yaml` — a strategy must not slip through preflight
    just because nobody got around to documenting a recovery mechanism that
    might not exist.
    """
    if simultaneous_legs <= 1:
        return True
    if venue is None:
        return None

    atomic = venue.supports_atomic_multileg
    recovery = venue.uncovered_leg_recovery
    has_recovery = recovery is not None and recovery.strip() != ""

    if atomic is True or has_recovery:
        return True
    if atomic is False:
        return False
    # atomic is None (unknown) and no recovery is named: unknown, not False.
    return None


def derive_venue_metrics(
    venue: VenueConfig | None,
    *,
    snapshot_source: str,
    venue_id: str,
    simultaneous_legs: int,
) -> dict[str, float]:
    """Bundle the three preflight metrics for `qlab.pipeline.evaluate`,
    dropping whichever came back `None`.

    `qlab.rules.engine.evaluate` reads a metric via `metrics.get(rule.metric)`
    and treats a missing key exactly like an explicit `None` value (both
    become an "unknown" verdict row) — so dropping `None`s here, rather
    than writing them into the dict, keeps `Evaluation.metrics` (persisted
    verbatim to `trial.metrics` by `qlab.pipeline.evaluate.evaluate_spec`)
    free of null placeholders for facts nobody actually knows, without
    changing the rules engine's behaviour either way.
    """
    raw: dict[str, bool | None] = {
        "venue_supported": venue_supported(venue),
        "data_forward_available": data_forward_available(
            venue, snapshot_source=snapshot_source, venue_id=venue_id
        ),
        "atomic_execution": atomic_execution(venue, simultaneous_legs=simultaneous_legs),
    }
    return {name: float(value) for name, value in raw.items() if value is not None}


def explain_venue_failures(
    venue: VenueConfig | None,
    *,
    venue_id: str,
    instruments: Sequence[str] = (),
) -> dict[str, str]:
    """Plain-language reason for every venue metric that came out FALSE, keyed
    by metric name (`venue_supported`, ...). A metric that passed or is
    unknown has no entry: this only explains a known "no".

    Why it exists: the rule row says `venue_supported = 0` and nothing else, so
    a reader could not tell whether the venue is closed to the owner (a legal
    fact, `docs/SEARCH_CONSTRAINTS.md` point 2) or merely has no order adapter
    of ours (an infrastructure fact). `venue_supported` is only the second:
    `venue.execution_adapter`. Access is a separate fact, recorded in
    `docs/VENUES.md`, and this metric never reads it.

    The venue checked is the spec's data source: a spec has no execution-venue
    field, so the pipeline asks whether orders could be placed where the
    candles come from. It does not ask whether the traded instruments exist
    on Hyperliquid (that check belongs to the second evaluation stage, T45);
    the text says so, and names the instruments the book actually held.
    """
    reasons: dict[str, str] = {}
    if venue is not None and venue.execution_adapter is False:
        names = list(instruments)
        shown = ", ".join(names[:EXPLAINED_INSTRUMENTS])
        more = len(names) - EXPLAINED_INSTRUMENTS
        held = (
            f"; the book held {len(names)} instrument(s): {shown}"
            + (f" and {more} more" if more > 0 else "")
            if names
            else ""
        )
        reasons["venue_supported"] = (
            f"venue {venue_id} has no order-placing adapter of ours "
            f"(venues/{venue_id}.yaml execution_adapter: false); this is a fact about "
            "our infrastructure, not about access (docs/VENUES.md) and not about "
            "the strategy; whether the instruments exist on Hyperliquid, where "
            f"orders could be placed, is not checked here{held}"
        )
    return reasons


__all__ = [
    "atomic_execution",
    "data_forward_available",
    "derive_venue_metrics",
    "explain_venue_failures",
    "venue_supported",
]
