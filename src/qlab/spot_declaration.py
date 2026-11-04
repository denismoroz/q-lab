"""A spec that reads the coin's own spot closes says so (`data.spot_prices: own`).

The stand has two conventions for a spot column's price (`qlab.data.snapshot.SPOT_PRICES`,
docs/SPOT_PRICES.md): ``perp`` (the baseline, the spot leg is marked at its perp's price) and
``own`` (the spot market's own closes). An idea whose rule reads the spot's closes -- a fast
return of the coin's spot, a leftover valued at the spot -- cannot be judged on the first, and
it CAN be judged on the second. Three ideas were once refused for that reason ("implementation
does not express: ... the stand marks a spot column at its perp's price"): the agent that wrote
the spec did not know `own` existed and declared the gap instead of asking for it.

This check is code, run on every spec an agent writes (implementer, researcher):

- an `unexpressed_mechanisms` line that names the perp-marked spot column while the spec marks
  its spot at the perp is a problem: the stand can express it, declare `spot_prices: own`;
- the same line while the spec already declares `own` is a problem too: the line is no longer
  true and would keep a judgeable idea "not evaluable".

The wording it recognises is the one the stand's own refusal uses ("data.spot_prices") and the
plain description of the perp mark ("marks a spot column at its perp"); a line about anything else
(a lot table, a rounding) is not touched.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import yaml

_MARKERS = ("spot_prices", "spot column at its perp", "marks a spot column at the perp")


def names_the_perp_mark(line: str) -> bool:
    """Whether an `unexpressed_mechanisms` line is about the spot column priced at the perp."""
    low = line.lower()
    return any(marker in low for marker in _MARKERS)


def spot_mark_lines(unexpressed: Iterable[str]) -> list[str]:
    """The lines of `unexpressed_mechanisms` that are about the spot column's price."""
    return [line for line in unexpressed if names_the_perp_mark(line)]


def spot_declaration_problems(spec) -> list[str]:
    """Problems with how `spec` declares its spot prices (module docstring); empty is fine."""
    lines = spot_mark_lines(spec.unexpressed_mechanisms)
    if not lines:
        return []
    if spec.data.spot_prices == "own":
        return ["unexpressed_mechanisms still says the spot column is priced at the perp, but "
                "the spec declares data.spot_prices: own; remove the line"]
    return ["unexpressed_mechanisms declares the spot-perp price gap as not expressible, but the "
            "stand expresses it: declare data.spot_prices: own (the spot market's own closes; "
            "needs include_spot: true or '-SPOT' instruments) and remove the line"]


OWN_NOTE = ("  # The idea reads the coin's own spot closes: the spot column is the spot\n"
            "  # market's own close, not the perp's price (docs/SPOT_PRICES.md).")


def declare_own_in_spec_text(text: str) -> tuple[str, list[str]]:
    """The spec file's text with `data.spot_prices: own` declared and the perp-mark lines of
    `unexpressed_mechanisms` removed, and the lines removed. Comments and the rest of the file
    are kept as they are. The result is checked: it must load to the same mapping as the input
    except for those two changes, or the call raises ValueError and nothing is written."""
    before = yaml.safe_load(text)
    if not isinstance(before, dict) or not isinstance(before.get("data"), dict):
        raise ValueError("the spec has no data block")
    lines = text.splitlines()
    out: list[str] = []
    removed: list[str] = []
    in_unexpressed = in_data = False
    note_lines = {line.strip() for line in OWN_NOTE.splitlines()}
    for line in lines:
        if re.match(r"^data:\s*$", line):
            out.append(line)
            out.extend(OWN_NOTE.splitlines())
            out.append("  spot_prices: own")
            in_data, in_unexpressed = True, False
            continue
        if in_data and (re.match(r"^\s+spot_prices:", line) or line.strip() in note_lines):
            continue  # replaced by the lines above
        if re.match(r"^unexpressed_mechanisms:\s*$", line):
            in_unexpressed, in_data = True, False
            out.append(line)
            continue
        if in_unexpressed:
            item = re.match(r"^\s+-\s+(.*)$", line)
            if item:
                value = yaml.safe_load(item.group(1))
                if isinstance(value, str) and names_the_perp_mark(value):
                    removed.append(value)
                    continue
            elif line.strip() and not line.lstrip().startswith("#"):
                in_unexpressed = False
        elif line and not line[0].isspace() and not line.startswith("#"):
            in_data = False
        out.append(line)
    new = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    after = yaml.safe_load(new)
    # an empty list is written as `unexpressed_mechanisms:` followed by nothing: make it explicit
    if after.get("unexpressed_mechanisms") is None and "unexpressed_mechanisms" in after:
        new = re.sub(r"^unexpressed_mechanisms:[ \t]*$", "unexpressed_mechanisms: []", new,
                     flags=re.M)
        after = yaml.safe_load(new)
    expected = dict(before)
    expected["data"] = {**before["data"], "spot_prices": "own"}
    if "unexpressed_mechanisms" in before:
        expected["unexpressed_mechanisms"] = [
            v for v in before["unexpressed_mechanisms"] if not names_the_perp_mark(v)]
    if after != expected:
        raise ValueError("the edit would change more than the spot declaration; edit by hand")
    return new, removed


__all__ = ["declare_own_in_spec_text", "names_the_perp_mark", "spot_declaration_problems",
           "spot_mark_lines"]
