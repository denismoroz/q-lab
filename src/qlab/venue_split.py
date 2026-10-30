"""A spec judged for Binance itself (docs/VENUE_SPLIT.md, owner 2026-10-30).

Until 2026-10-30 a run on Binance's history was a PROXY for Hyperliquid: its spec said
`execution_venue: hyperliquid` and carried Hyperliquid's costs and minimum leg, and the second stage
repeated the idea on Hyperliquid. The owner: «нужно разделять на HL and Binance» and «да, делай
раздельно Binance и HL». Now a run is judged for the venue it is made on: `execution_venue` is
omitted (the venue is `data.source`), the costs are the venue's row of `night/costs.yaml`, the
minimum leg is the venue's (`venues/binance.yaml`, per contract where Binance states it), and the
facts about execution come from the same venue file.

This module turns a spec made the old way into the Binance-judged one, both as a model
(`binance_version`, used for the twins the lab builds from now on) and as the text of a spec file
(`binance_text`, used once for the specs already on disk, so that their comments survive). Pure
functions; they read the cost table and the venue file and compute nothing about a strategy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import yaml

from qlab import costs as cost_table

BINANCE = "binance"
DIVIDED_ON = "2026-10-30"
MIN_LEG_CITATION = (
    "venues/binance.yaml min_leg_notional_usd: 5.0, Binance USDT-M futures' MIN_NOTIONAL of the "
    "ordinary contract (exchangeInfo read 2026-10-30, 516 of 522 contracts); BTC (50) and five "
    "contracts (20) differ and are read per contract (min_leg_notional_by_instrument, "
    "seed/venue_facts/2026-10-30-binance_um_min_notional.csv)")
NOTE = (
    "Venue split ({day}, docs/VENUE_SPLIT.md): this spec is judged for Binance itself. Its costs "
    "are the binance row of night/costs.yaml{blend}, its minimum leg is Binance's "
    "(venues/binance.yaml, per contract where Binance states it), and the facts about "
    "execution (venue_supported, atomic_execution) are Binance's: a run that passes every rule "
    "routes needs-infrastructure because the lab has no Binance order adapter. Text above that "
    "speaks of the original's or Hyperliquid's costs and minimum leg describes the spec as it was "
    "before that day.")


@dataclass(frozen=True)
class BinanceNumbers:
    taker_fee_bps: float
    slippage_bps: float
    min_leg_notional: float
    taker_source: str
    slippage_source: str
    two_leg: bool


def numbers(simultaneous_legs: int) -> BinanceNumbers:
    """The Binance numbers a spec with this many legs at once takes."""
    row = cost_table.venue_costs(BINANCE)
    if row is None:
        raise ValueError("night/costs.yaml has no row for binance")
    two = simultaneous_legs >= 2 and row.two_leg_taker_fee_bps is not None
    return BinanceNumbers(
        taker_fee_bps=row.two_leg_taker_fee_bps if two else row.taker_fee_bps,
        slippage_bps=row.slippage_bps, min_leg_notional=5.0,
        taker_source=row.sources["two_leg_taker_fee_bps" if two else "taker_fee_bps"],
        slippage_source=row.sources["slippage_bps"], two_leg=two)


def note(n: BinanceNumbers) -> str:
    blend = " (the two-leg mean of the perpetual's and the spot's fee)" if n.two_leg else ""
    return NOTE.format(day=DIVIDED_ON, blend=blend)


def new_sources(old: dict[str, str], n: BinanceNumbers, *, has_score_costs: bool = False
                ) -> dict[str, str]:
    """The `sources` of the Binance-judged spec: every citation of a cost number (the spec's own,
    the retune's scoring costs, an overlay wrapper's inner costs) replaced by Binance's, the
    minimum leg's too, and the sentence that said the costs differ from the table dropped."""
    out = {k: v for k, v in old.items() if k != cost_table.DEVIATION_KEY}
    texts = {"taker_fee_bps": n.taker_source, "slippage_bps": n.slippage_source}
    for key in list(out):
        for name, text in texts.items():
            if key.endswith(f"costs.{name}"):
                out[key] = text
    for name, text in texts.items():
        out[f"costs.{name}"] = text
        if has_score_costs:
            out.setdefault(f"score_costs.{name}", text)
    out["min_leg_notional"] = MIN_LEG_CITATION
    return out


def _with_costs(value: Any, n: BinanceNumbers) -> Any:
    """`value` with every `costs` / `score_costs` mapping inside it (a pair of fee and slippage)
    set to Binance's numbers."""
    if isinstance(value, dict):
        out = {k: _with_costs(v, n) for k, v in value.items()}
        for key in ("costs", "score_costs"):
            inner = out.get(key)
            if isinstance(inner, dict) and {"taker_fee_bps", "slippage_bps"} <= set(inner):
                out[key] = {**inner, "taker_fee_bps": n.taker_fee_bps,
                            "slippage_bps": n.slippage_bps}
        return out
    return value


def binance_version(spec):
    """`spec` as the Binance verdict of its idea: Binance's costs (also the retune's scoring costs),
    Binance's minimum leg, no `execution_venue`."""
    from qlab.pipeline.spec import SpecCosts

    n = numbers(int(spec.simultaneous_legs or 1))
    params: Any = _with_costs(spec.params, n)
    scored = "score_costs" in json.dumps(params)
    return spec.model_copy(update={
        "params": params,
        "costs": SpecCosts(taker_fee_bps=n.taker_fee_bps, slippage_bps=n.slippage_bps),
        "min_leg_notional": n.min_leg_notional,
        "execution_venue": None,
        "sources": new_sources(dict(spec.sources), n, has_score_costs=scored),
        "differs_from_source": [*spec.differs_from_source, note(n)],
    })


# --------------------------------------------------------------------------
# The text of a spec file


_TOP_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:", re.M)


def _block_span(text: str, key: str) -> tuple[int, int] | None:
    """(start, end) of the top-level block `key:` (its line and everything indented or blank or
    commented under it, up to the next top-level key or the end)."""
    match = re.search(rf"^{re.escape(key)}:", text, re.M)
    if match is None:
        return None
    nxt = _TOP_KEY.search(text, match.end())
    end = nxt.start() if nxt else len(text)
    # comment lines right above the next key belong to that key, not to this block
    while end > match.end():
        line_start = text.rfind("\n", 0, end - 1) + 1
        line = text[line_start:end]
        if line.startswith("#") or not line.strip():
            end = line_start
        else:
            break
    return match.start(), max(end, match.end())


def _replace_block(text: str, key: str, value: Any, *, before: str | None = None) -> str:
    dumped = yaml.safe_dump({key: value}, allow_unicode=True, width=100, sort_keys=False)
    span = _block_span(text, key)
    if span is not None:
        return text[:span[0]] + dumped + text[span[1]:]
    anchor = _block_span(text, before) if before else None
    if anchor is None:
        return text.rstrip("\n") + "\n" + dumped
    return text[:anchor[0]] + dumped + text[anchor[0]:]


def binance_text(text: str) -> str:
    """The text of a spec file rewritten to its Binance-judged version; comments outside the
    rewritten blocks (`costs`, `sources`, `differs_from_source`, `score_costs`) are kept."""
    data = yaml.safe_load(text)
    n = numbers(int(data.get("simultaneous_legs") or 1))
    cost_block = (f"costs:\n  taker_fee_bps: {n.taker_fee_bps:g}\n"
                  f"  slippage_bps: {n.slippage_bps:g}\n")
    span = _block_span(text, "costs")
    if span is None:
        raise ValueError("the spec has no costs block")
    text = text[:span[0]] + cost_block + text[span[1]:]
    text = re.sub(r"(?m)^min_leg_notional:.*\n", f"min_leg_notional: {n.min_leg_notional:g}\n",
                  text)
    text = re.sub(r"(?m)^execution_venue:.*\n", "", text)
    scored = bool(re.search(r"(?m)^ *score_costs:", text))
    text = re.sub(
        r"(?m)^( +)(score_costs|costs):\n\1  taker_fee_bps:[^\n]*\n\1  slippage_bps:[^\n]*\n",
        lambda m: (f"{m.group(1)}{m.group(2)}:\n{m.group(1)}  taker_fee_bps: {n.taker_fee_bps:g}\n"
                   f"{m.group(1)}  slippage_bps: {n.slippage_bps:g}\n"), text)
    old_sources = {str(k): str(v) for k, v in (data.get("sources") or {}).items()}
    text = _replace_block(text, "sources", new_sources(old_sources, n, has_score_costs=scored))
    notes = [*(data.get("differs_from_source") or []), note(n)]
    text = _replace_block(text, "differs_from_source", notes, before="noise_universe")
    return text


__all__ = ["BINANCE", "BinanceNumbers", "MIN_LEG_CITATION", "binance_text", "binance_version",
           "new_sources", "note", "numbers"]
