"""Overlay variants of a strategy spec (docs/OVERLAYS.md).

Owner, 2026-10-28: «это все идеи для платформы. было бы здорово проверять разные варианты
возможно их можно смиксовать», and, correcting a single hand-picked trial: «ну ты вот взял
проверил один вариант с потолка и зарубил его». So no overlay is judged at one value: a variant
is an overlay (or a pair) with its whole FAMILY of settings, and the settings are chosen by code
every month from the past only (`qlab.strategies.retune.Retune`, docs/RETUNE.md), exactly as a
causal twin chooses its numbers (`qlab.twin`). Every day after the first choice is then a forward
test.

**What is generated** for a base spec: every single overlay and every pair, no triples. A pair is
an unordered combination applied in a fixed order -- book-level overlays first (they decide how
big the book is), position-level ones last (they decide how the book gets there) -- so that
(A, B) and (B, A) are not two variants. Another order, or a longer chain, is possible by writing a
spec by hand with the `Overlaid` wrapper, which does not limit the chain; the night builds none.
Triples are left out because the count grows too fast: with n
overlays there are n singles, n(n-1)/2 pairs and n(n-1)(n-2)/6 triples, each a stand run and a
trial that deflation counts, and a family tested ever more widely finds ever more luck
(docs/T27_DEFLATION.md).

Each variant is an idea of its own, `<base>-ov-<slug>`, registered as a variant of the base
(`parent_id`), with its own spec file. A variant of kind `causal` (the default) is a `Retune` of
`Overlaid` over the product of the overlays' families (the coarse families for a pair, so that
two grids multiplied stay affordable); a variant of kind `fixed` is one point of the family,
`Overlaid` with exactly those numbers -- every point is a trial, and the family is read as a
whole, never by its best member.

**Which part of the history is judged.** The base's own forward test is kept: a variant of a
base whose numbers were fixed on a date (`params_fixed_at`) is judged from that date at the
earliest, and a causal variant from its own first choice, which needs the overlay's longest
look-back plus two whole months (`qlab.twin.first_choice`). The comparison with the base is made
on the part both are judged on (`qlab.overlay_compare`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import yaml

from qlab.pipeline.spec import StrategySpec
from qlab.strategies.overlays import get, names
from qlab.twin import PERIODS, RETUNE_DOC, first_choice, warmup_days

OVERLAID = "qlab.strategies.overlays:Overlaid"
RETUNE = "qlab.strategies.retune:Retune"
OVERLAYS_DOC = "docs/OVERLAYS.md"
OVERLAYS_CODE = "src/qlab/strategies/overlays/core.py"
RETUNE_CODE = "src/qlab/strategies/retune.py"
INFIX = "-ov-"
CANONICAL_ORDER = ("market_gate", "meta_gate", "vol_target", "own_results", "gradual_entry",
                   "gradual_exit")
"""The order overlays are chained in a generated variant: book-level first, position-level last."""
KIND_CAUSAL, KIND_FIXED = "causal", "fixed"
TITLE_RU = {
    "market_gate": "не торговать в одном из рыночных состояний",
    "meta_gate": "торговать только в дни, когда решение второго уровня говорит «заработает»",
    "vol_target": "уменьшать размер, когда собственные результаты стали неспокойнее обычного",
    "own_results": "размер по собственным недавним результатам",
    "gradual_entry": "постепенный вход в позицию",
    "gradual_exit": "постепенный выход из позиции",
}
_SPEC_LEVEL = ("data", "costs", "regime_claim", "min_leg_notional", "judged_from",
               "instruments_chosen_at", "code_ref", "params_fixed_at", "execution_venue")


def is_variant_id(idea_id: str) -> bool:
    return INFIX in idea_id


def base_of(idea_id: str) -> str | None:
    """The base idea of an overlay variant id, None for any other id."""
    return idea_id.split(INFIX, 1)[0] if INFIX in idea_id else None


def available_overlays(extra_grids: Mapping[str, Mapping] | None = None) -> list[str]:
    """The overlays a generated variant can use, in the canonical order: those that have a
    family (`meta_gate` has one only where `extra_grids` supplies its stored decisions)."""
    extra = extra_grids or {}
    return [n for n in CANONICAL_ORDER if n in names() and (get(n).grid() or extra.get(n))]


def chains(overlay_names: Sequence[str]) -> list[tuple[str, ...]]:
    """Every single overlay and every pair, in the canonical order. No triples."""
    ordered = [n for n in CANONICAL_ORDER if n in overlay_names]
    ordered += [n for n in overlay_names if n not in ordered]
    return [(n,) for n in ordered] + list(itertools.combinations(ordered, 2))


def variant_id(base_id: str, chain: Sequence[str], point: str | None = None) -> str:
    """`<base>-ov-<short>[-<short>][-<point label>]`."""
    slug = "-".join(get(n).short for n in chain)
    return f"{base_id}{INFIX}{slug}" + (f"-{point}" if point else "")


def family(chain: Sequence[str], extra_grids: Mapping[str, Mapping] | None = None
           ) -> dict[str, list[dict]]:
    """The candidates of a chain: label -> the chain as `[{name, params}]`. A single overlay
    offers its whole family, a pair its coarse families multiplied."""
    extra = extra_grids or {}
    per_overlay = []
    for name in chain:
        overlay = get(name)
        grid = (extra.get(name) if name in extra else None) or (
            overlay.grid() if len(chain) == 1 else overlay.coarse_grid())
        if not grid:
            raise ValueError(f"overlay {name!r} has no family to choose from")
        per_overlay.append([(name, label, dict(point)) for label, point in grid.items()])
    out: dict[str, list[dict]] = {}
    for combo in itertools.product(*per_overlay):
        label = "+".join(f"{name}:{point_label}" for name, point_label, _ in combo)
        out[label] = [{"name": name, "params": params} for name, _, params in combo]
    return out


def family_warmup_days(candidates: Mapping[str, list[dict]]) -> float:
    """The longest warm-up any candidate of the family needs."""
    from qlab.strategies.overlays import chain_warmup_days

    return max((chain_warmup_days(chain) for chain in candidates.values()), default=0.0)


# --------------------------------------------------------------------------
# The first causal choice


def first_trading_day(base: StrategySpec) -> date:
    """The first day the base can hold a position: the date of its own first causal choice,
    or the start of its data plus the longest look-back its parameters name."""
    if base.selects_causally and base.params_fixed_at is not None:
        return base.params_fixed_at
    warm, _ = warmup_days([base.params], base.data.interval)
    return base.data.start + timedelta(days=math.ceil(warm))


def forward_start(base: StrategySpec, warm_days: float, refit_every: str = "M"
                  ) -> tuple[date, tuple[date, date, date]]:
    """The first day a causal variant is judged from, and the dates it comes from.

    The overlay can act from `first_trading_day + warm_days`; a candidate is eligible only
    after it has traded through one whole re-tuning period, so the first choice is made at the
    second re-tuning after that (`qlab.twin.first_choice`). A base whose numbers were fixed on a
    later date keeps that date: the days before it are the base's own selection period."""
    dates = first_choice(first_trading_day(base), warm_days, refit_every)
    choice = dates[2]
    if base.params_fixed_at is not None and base.params_fixed_at > choice:
        choice = base.params_fixed_at
    return choice, dates


# --------------------------------------------------------------------------
# The spec


def _carry_sources(base: StrategySpec, base_cite: str) -> dict[str, str]:
    """The base's citations at the places its parameters now live, one level down
    (`base_params.inner_params.<path>`); the spec-level ones stay where they are."""
    top = set(base.params)
    out: dict[str, str] = {}
    for key, citation in base.sources.items():
        head = key.split(".")[0]
        if head in top:
            out[f"base_params.inner_params.{key}"] = citation
        elif head not in ("code_ref",) and head in _SPEC_LEVEL:
            out[key] = citation
    out["base_params.inner_code_ref"] = f"{base_cite}: the base's code_ref, unchanged"
    return out


def _costs_cite(base: StrategySpec, name: str) -> str:
    return (base.sources.get(f"costs.{name}") or base.sources.get("costs")
            or f"{RETUNE_DOC}: scores use the spec's costs")


def _fixed_sources(base: StrategySpec, base_cite: str) -> dict[str, str]:
    """As `_carry_sources`, for the wrapper without the causal chooser: the base's parameters
    are `inner_params`."""
    top = set(base.params)
    out: dict[str, str] = {}
    for key, citation in base.sources.items():
        head = key.split(".")[0]
        if head in top:
            out[f"inner_params.{key}"] = citation
        elif head in _SPEC_LEVEL and head != "code_ref":
            out[key] = citation
    out["inner_code_ref"] = f"{base_cite}: the base's code_ref, unchanged"
    return out


def build_variant(base: StrategySpec, chain: Sequence[str], *, base_cite: str,
                  kind: str = KIND_CAUSAL, point: tuple[str, list[dict]] | None = None,
                  extra_grids: Mapping[str, Mapping] | None = None, refit_every: str = "M",
                  end: date | None = None) -> StrategySpec:
    """The overlay variant of `base` for `chain`.

    `base_cite` is the base's spec file as a path the sources check can open (it is cited as
    the place the base's code_ref and numbers come from). A `fixed` variant needs `point`:
    (its label, the chain as `[{name, params}]`). `end` extends the data window (never
    shortens it) so that a snapshot already on disk answers the request."""
    if kind not in (KIND_CAUSAL, KIND_FIXED):
        raise ValueError(f"kind is {KIND_CAUSAL!r} or {KIND_FIXED!r}, got {kind!r}")
    if len(chain) not in (1, 2):
        raise ValueError("a variant is a single overlay or a pair; no triples")
    if refit_every not in PERIODS:
        raise ValueError(f"refit_every must be one of {sorted(PERIODS)}, got {refit_every!r}")
    costs = {"taker_fee_bps": base.costs.taker_fee_bps, "slippage_bps": base.costs.slippage_bps}
    shown = " + ".join(TITLE_RU[n] for n in chain)
    data = base.data.model_copy(update={"end": max(base.data.end, end)} if end else {})
    update: dict = {
        "idea_id": variant_id(base.idea_id, chain, point[0] if point else None),
        "title": f"{base.title}: надстройки — {shown}",
        "data": data.model_dump(mode="python"),
    }
    note = (f"Overlay variant of {base.idea_id} (qlab.overlay_variants, {OVERLAYS_DOC}): the same "
            f"strategy with {', '.join(chain)} applied to its target weights; the strategy's own "
            "numbers are untouched.")
    if kind == KIND_CAUSAL:
        candidates = family(chain, extra_grids)
        warm = family_warmup_days(candidates)
        fixed_at, dates = forward_start(base, warm, refit_every)
        if dates[2] >= data.end:
            raise ValueError(f"the first causal choice ({dates[2]}) falls on or after the end "
                             f"of the data ({data.end}): no day would be a forward test")
        period = PERIODS[refit_every][0]
        base_params = {"inner_code_ref": base.code_ref, "inner_params": base.params,
                       "overlays": [], "costs": costs}
        update.update(
            code_ref=RETUNE,
            params={
                "inner_code_ref": OVERLAID,
                "base_params": base_params,
                "grid": {"overlay": {label: {"overlays": ov} for label, ov in candidates.items()}},
                "refit_every": refit_every,
                "window": "expanding",
                "objective": "sharpe_net",
                "score_costs": dict(costs),
            },
            selects_causally=True, params_fit_from=None, params_fixed_at=fixed_at,
            params_fixed_evidence=(
                f"The wrapper's first causal choice (qlab.strategies.retune, {RETUNE_DOC}): the "
                f"base can first trade on {first_trading_day(base)}; the longest warm-up of the "
                f"overlay family ({', '.join(chain)}) is "
                f"{math.ceil(warm)} days, so the overlay can first act on {dates[0]}; a "
                f"candidate is eligible only after one whole {period} re-tuning period, so the "
                f"first re-tuning after {dates[0]} is {dates[1]} and the first choice is made at "
                f"the one after it, {dates[2]}"
                + (f"; the base's own numbers were fixed on {base.params_fixed_at}, and the "
                   "variant is not judged before it" if fixed_at != dates[2] else "")
                + f". The family is declared whole in {OVERLAYS_DOC} before any run: "
                f"{len(candidates)} candidates, none chosen by looking at the data it is judged "
                "on. What this does not prove: the family itself (which overlays and which "
                "settings are offered) was written by people who had seen the market."),
            sources={
                **_carry_sources(base, base_cite),
                "inner_code_ref": f"{OVERLAYS_CODE} (the overlay wrapper)",
                "base_params.overlays": f"{OVERLAYS_DOC}: the chain is empty here, every "
                                        "candidate of the grid replaces it",
                "base_params.costs.taker_fee_bps": _costs_cite(base, "taker_fee_bps"),
                "base_params.costs.slippage_bps": _costs_cite(base, "slippage_bps"),
                "grid.overlay": f"{OVERLAYS_DOC}: the overlay families (tables of the settings)",
                "refit_every": f"{RETUNE_DOC}: re-tuned every {period}; the owner 2026-10-02 "
                               "named «раз в неделю/день или месяц»",
                "window": f"{RETUNE_DOC} (expanding window: no number introduced)",
                "objective": f"{RETUNE_DOC} (Sharpe after costs)",
                "score_costs.taker_fee_bps": _costs_cite(base, "taker_fee_bps"),
                "score_costs.slippage_bps": _costs_cite(base, "slippage_bps"),
                "code_ref": f"{RETUNE_CODE} (existing wrapper)",
            },
            differs_from_source=[*base.differs_from_source, note + " The numbers of the overlays "
                                 "are chosen by code every month from the past only."],
        )
    else:
        if point is None:
            raise ValueError("a fixed variant needs the point of the family it fixes")
        update.update(
            code_ref=OVERLAID,
            params={"inner_code_ref": base.code_ref, "inner_params": base.params,
                    "overlays": point[1], "costs": costs},
            params_fixed_evidence=None if base.params_fixed_at is None else (
                (base.params_fixed_evidence or "")
                + f" The overlay numbers ({point[0]}) are one point of the family declared whole "
                f"in {OVERLAYS_DOC}; none was chosen on this data, every point of the family is "
                "run as a trial, and the family is read as a whole."),
            sources={
                **_fixed_sources(base, base_cite),
                "overlays": f"{OVERLAYS_DOC}: one point of the overlay family (tables of settings)",
                "costs.taker_fee_bps": _costs_cite(base, "taker_fee_bps"),
                "costs.slippage_bps": _costs_cite(base, "slippage_bps"),
            },
            differs_from_source=[*base.differs_from_source, note],
        )
    raw = {**base.model_dump(mode="python"), **update}
    return StrategySpec.model_validate(raw)


# --------------------------------------------------------------------------
# The file


def variant_yaml(spec: StrategySpec, base_id: str) -> str:
    """The variant as a spec file's text (no None fields; `params` keeps its nulls)."""
    raw = spec.model_dump(mode="json")
    raw = {k: v for k, v in raw.items() if v is not None
           and not (k in ("differs_from_source", "sources", "regime_claim", "review_answers")
                    and not v)}
    raw["data"] = {k: v for k, v in raw["data"].items() if v is not None}
    header = (f"# Overlay variant of {base_id} (qlab overlays, src/qlab/overlay_variants.py):\n"
              "# the same strategy with a chain of overlays on its weights (docs/OVERLAYS.md).\n"
              "# Written by code, not by hand.\n")
    return header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)


def variant_path(spec_dir: Path, base_id: str, idea_id: str) -> Path:
    return spec_dir / "overlays" / base_id / f"{idea_id}.yaml"


def write_variant(spec_dir: Path, base_id: str, spec: StrategySpec) -> Path:
    """Write the variant's spec file. An identical file is left as it is; a different one with
    the same name is replaced (the family or the base changed: the ledger sees the new text)."""
    target = variant_path(spec_dir, base_id, spec.idea_id)
    text = variant_yaml(spec, base_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target


__all__ = ["CANONICAL_ORDER", "INFIX", "KIND_CAUSAL", "KIND_FIXED", "OVERLAID", "TITLE_RU",
           "available_overlays", "base_of", "build_variant", "chains", "family",
           "family_warmup_days", "first_trading_day", "forward_start", "is_variant_id",
           "variant_id", "variant_path", "variant_yaml", "write_variant"]
