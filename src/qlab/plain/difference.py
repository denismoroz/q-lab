"""How an idea differs from its base, computed by code from the two specs (docs/PLAIN_PAGE.md).

The base is the family's reference spec (`base_specs` of the family text: the production book's
spec for Bv2, FRAB, Trend, ...). Nobody writes the difference by hand: the parameters that differ
are turned into phrases through the dictionary (`strategies/plain/params.yaml`), and so are the
layers the idea adds (overlays, a market-state gate, an entry rhythm) and the data and costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qlab.plain.specs import SpecView, candidate_values, core_items
from qlab.plain.words import MISSING, Words, num, same, venue_name

SKIPPED_PARAMS = frozenset({"mode"})
"""Parameters that are the same everywhere by rule (`mode` is always paper) and say nothing."""


@dataclass(frozen=True)
class Item:
    """One line of «Чем этот вариант отличается»."""

    text: str
    kind: str
    """`param`, `layer`, `data` or `cost`."""
    tag: str | None = None
    """The shortest name of the change, for the plain title; None if there is none."""


def percent_text(bps: float) -> str:
    """Basis points as the percent the owner reads: 5.25 -> 0.0525%."""
    return f"{num(round(bps / 100, 6))}%"


def costs_text(costs: dict[str, Any]) -> str | None:
    fee, slip = costs.get("taker_fee_bps"), costs.get("slippage_bps")
    if fee is None or slip is None:
        return None
    return (f"комиссия за сделку {percent_text(fee)}, проскальзывание цены {percent_text(slip)}")


def overlay_names(view: SpecView) -> list[str]:
    """The overlays a spec adds: the fixed chain of an Overlaid layer, and every overlay named by
    a candidate of a Retune grid's `overlay` axis, in order of first appearance."""
    out: list[str] = []

    def take(chain: Any) -> None:
        for item in chain if isinstance(chain, list) else ():
            name = item.get("name") if isinstance(item, dict) else None
            if name and name not in out:
                out.append(str(name))

    unwrapped = view.unwrapped
    for layer in unwrapped.layer("Overlaid"):
        take(layer.own.get("overlays"))
    for layer in unwrapped.layer("Retune"):
        for candidates in (layer.own.get("grid") or {}).values():
            for overrides in (candidates or {}).values():
                if isinstance(overrides, dict):
                    take(overrides.get("overlays"))
    return out


def gate_states(view: SpecView) -> list[str] | None:
    """The market states a fixed gate trades in; None if the spec has no gate."""
    gates = view.unwrapped.layer("RegimeGate")
    if not gates:
        return None
    trade_in = gates[0].own.get("trade_in")
    return list(trade_in) if isinstance(trade_in, list) else []


def _gate_chosen_by_code(view: SpecView) -> bool:
    """A gate whose states are one of the candidates code chooses among."""
    for layer in view.unwrapped.layer("Retune"):
        for candidates in (layer.own.get("grid") or {}).values():
            for overrides in (candidates or {}).values():
                if isinstance(overrides, dict) and "trade_in" in overrides:
                    return True
    return False


def layer_items(view: SpecView, base: SpecView, words: Words) -> list[Item]:
    layers = words.dictionary.layers
    out: list[Item] = []
    mine, theirs = overlay_names(view), overlay_names(base)
    added = [n for n in mine if n not in theirs]
    if added:
        titles = [layers.get("overlays", {}).get(n, {}).get("title", f"надстройка {n}")
                  for n in added]
        out.append(Item(
            "к решениям стратегии добавлена надстройка (небольшая деталь между стратегией и "
            f"биржей, которая меняет, когда и сколько держать): {'; '.join(titles)}",
            "layer", "; ".join(titles)))
    states, base_states = gate_states(view), gate_states(base)
    if states is not None and base_states is None:
        if states:
            out.append(Item(
                f"торгует только когда рынок в состоянии: {words.states(states)}; "
                f"{layers.get('regime', {}).get('detector_default', '')}", "layer",
                "не торгует в части состояний рынка"))
        elif _gate_chosen_by_code(view):
            out.append(Item(
                "торгует не при любом состоянии рынка: в каких состояниях торговать, выбирает код; "
                f"{layers.get('regime', {}).get('detector_default', '')}", "layer",
                "с выбором состояний рынка"))
    elif states is not None and base_states is not None and not same(states, base_states):
        out.append(Item(f"состояния рынка, в которых торгует: было {words.states(base_states)}, "
                        f"стало {words.states(states)}", "layer"))
    for layer in view.unwrapped.layer("SplitCadence"):
        base_cadence = base.unwrapped.layer("SplitCadence")
        for key in ("entry_every", "exit_every"):
            old = base_cadence[0].own.get(key, MISSING) if base_cadence else MISSING
            new = layer.own.get(key)
            if new is not None and not same(old, new):
                out.append(Item(words.change(key, old, new), "layer"))
    return out


def _strategy_coins(view: SpecView) -> set[str]:
    """The coins the strategy's own parameters name (`coins`, or FRAB's `universe` list)."""
    core = view.unwrapped.core
    for key in ("coins", "universe"):
        if isinstance(core.get(key), list):
            return {str(c) for c in core[key]}
    return set()


def data_items(view: SpecView, base: SpecView, words: Words) -> list[Item]:
    out: list[Item] = []
    mine, theirs = view.data, base.data
    source_differs = mine.get("source") != theirs.get("source")
    if source_differs:
        out.append(Item(
            f"данные: {venue_name(mine.get('source'))} вместо {venue_name(theirs.get('source'))}",
            "data"))
    if mine.get("interval") != theirs.get("interval"):
        out.append(Item(f"свечи: {mine.get('interval')} вместо {theirs.get('interval')} "
                        "(1h — часовые, 1d — суточные)", "data"))
    for key, phrase in (("include_spot", "в данных есть спотовые рынки (покупка самой монеты)"),
                        ("spot_source", "монету покупают на другой площадке: ")):
        if mine.get(key) != theirs.get(key) and mine.get(key):
            out.append(Item(phrase + (venue_name(mine[key]) if key == "spot_source" else ""),
                            "data"))
    old_list, new_list = theirs.get("instruments"), mine.get("instruments")
    if isinstance(new_list, list) and new_list != old_list:
        coins = sorted({str(c).removesuffix("-SPOT") for c in new_list})
        # the list that only repeats the coins the strategy itself names says nothing new
        if coins != sorted(_strategy_coins(view)):
            out.append(Item(f"монеты названы списком: {', '.join(coins)}", "data"))
    elif old_list and not new_list:
        out.append(Item("монеты не названы списком: берутся все, что есть на площадке", "data"))
    if mine.get("min_daily_volume_usd") != theirs.get("min_daily_volume_usd") \
            and mine.get("min_daily_volume_usd"):
        out.append(Item("в отбор берутся только монеты с дневным оборотом не меньше "
                        f"${num(mine['min_daily_volume_usd'])}", "data"))
    if not same(view.costs, base.costs):
        new_costs, old_costs = costs_text(view.costs), costs_text(base.costs)
        if new_costs and old_costs:
            out.append(Item(f"издержки: {new_costs} (у основной версии — {old_costs})", "cost"))
    old_leg, new_leg = base.extras.get("min_leg_notional"), view.extras.get("min_leg_notional")
    if new_leg is not None and old_leg is not None and not same(new_leg, old_leg):
        out.append(Item(f"наименьшая заявка: ${num(new_leg)} вместо ${num(old_leg)}", "cost"))
    return out


def _open_item(words: Words, key: str, values: list[Any], old: Any) -> Item:
    what = (words.entry(key) or {}).get("what", f"параметр {key}")
    return Item(f"{what}: код выбирает из значений {words.choices(key, values)} "
                f"(у основной версии — {words.value(key, old)})", "param")


def param_items(view: SpecView, base: SpecView, words: Words,
                varied: frozenset[str] = frozenset()) -> list[Item]:
    """The core parameters that differ. A parameter the idea only gives candidates for (`varied`)
    is described by `grid_items`, not here."""
    mine, theirs = core_items(view.unwrapped), core_items(base.unwrapped)
    out: list[Item] = []
    keys = [k for k in mine if k not in SKIPPED_PARAMS and k not in varied] \
        + [k for k in theirs if k not in mine and k not in SKIPPED_PARAMS and k not in varied]
    for key in keys:
        new, old = mine.get(key, MISSING), theirs.get(key, MISSING)
        # a parameter the spec does not state has the value the code gives it, where the
        # dictionary knows that value (parameters added to the code later); where it does not,
        # nothing is known about the idea's value and nothing is said
        new_eff = words.default(key) if new is MISSING else new
        old_eff = words.default(key) if old is MISSING else old
        if new_eff is MISSING:
            continue
        if old_eff is not MISSING and same(new_eff, old_eff):
            continue
        if words.open_list(key, new_eff) and not words.open_list(key, old_eff):
            out.append(_open_item(words, key, new_eff, old_eff))
            continue
        out.append(Item(words.change(key, old_eff, new_eff), "param", words.tag(key, new_eff)))
    return out


def grid_items(view: SpecView, base: SpecView, words: Words) -> list[Item]:
    """What the candidates code chooses among change against the base. A variant that is one named
    set of numbers against the base (a researcher's experiment) IS the idea, so its numbers are
    its difference."""
    base_core = core_items(base.unwrapped)
    out: list[Item] = []
    for key, values in candidate_values(view).items():
        old = base_core.get(key, MISSING)
        old_eff = words.default(key) if old is MISSING else old
        differing = [v for v in values if old_eff is MISSING or not same(v, old_eff)]
        if not differing:
            continue
        if len(differing) == 1 and len(values) <= 2:
            out.append(Item(words.change(key, old_eff, differing[0]), "param",
                            words.tag(key, differing[0])))
        else:
            out.append(_open_item(words, key, values, old_eff))
    return out


def difference(view: SpecView, base: SpecView, words: Words) -> list[Item]:
    """Every difference between the spec and its base, in the order: parameters, candidates,
    layers, data. Empty when the two are the same strategy on the same data. Two specs whose core
    strategies are different implementations are not compared parameter by parameter (their keys
    do not match): only the layers and the data are."""
    out: list[Item] = []
    if view.unwrapped.core_name == base.unwrapped.core_name:
        out += param_items(view, base, words, varied=frozenset(candidate_values(view)))
        out += grid_items(view, base, words)
    out += layer_items(view, base, words)
    out += data_items(view, base, words)
    return out
