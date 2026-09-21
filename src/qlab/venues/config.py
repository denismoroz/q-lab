"""Per-venue capability config: `venues/<id>.yaml` -> `VenueConfig`.

Each file under the repo-root `venues/` directory records facts about OUR
infrastructure on one venue (do we have working execution there today, is
its market data free and available forward, can it fill several legs
atomically, ...) — never facts about the market, which is what
`qlab.harness` computes instead. Every field is optional and defaults to
`None` ("unknown"), by design: this schema has no fabricated defaults for
the same reason `qlab.pipeline.spec.StrategySpec` has none for costs or
`min_leg_notional` (see that module's docstring) — a venue fact nobody has
actually recorded must read as "we don't know," not as a guessed value that
could silently let an unsupported venue pass preflight.

Every value that DOES appear in a `venues/*.yaml` file must be backed by a
cited source, recorded as a YAML comment next to the field (see
`venues/hyperliquid.yaml` for the format) — this module does not enforce
that mechanically (a citation is prose, not a machine-checkable
constraint), but every field's docstring below names what the fact means so
a reviewer can tell whether a citation actually supports it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

# `venues/` at the repo root, a sibling of `src/`. This file lives at
# `src/qlab/venues/config.py`, so `parents[3]` is the repo root — the same
# depth `qlab.pipeline.evaluate._code_sha` uses from
# `src/qlab/pipeline/evaluate.py` to find the repo root for `git rev-parse`.
DEFAULT_VENUES_DIR = Path(__file__).resolve().parents[3] / "venues"


class VenueConfig(BaseModel):
    """One venue's recorded capability facts.

    `id`/`title` are required (a config file with neither is not naming
    anything); every capability field is optional and `None` by default.
    `extra="forbid"` so a typo'd field name fails loudly at load time
    instead of silently being ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str

    #: Do we have working order-placing execution infrastructure on this
    #: venue TODAY? Feeds `venue_supported` directly
    #: (`qlab.venues.derive.venue_supported`).
    execution_adapter: bool | None = None

    #: The venue's own minimum tradeable leg notional, in USD. Not consumed
    #: by any derivation in this task (only `execution_adapter`,
    #: `free_public_data`, `supports_atomic_multileg`, and
    #: `uncovered_leg_recovery` are), but still a fact about the venue that
    #: belongs in its config rather than being hardcoded into a spec or
    #: rule. Distinct from `qlab.pipeline.spec.StrategySpec.min_leg_notional`
    #: and from `rules/2026-09-20.1.yaml`'s `capital_fit` rule, both of
    #: which encode q-lab/frab's own OPERATING floor (with a slippage
    #: buffer), not the exchange's raw minimum.
    min_leg_notional_usd: float | None = None

    #: Is market and funding data available for this venue without API
    #: keys, both historically and going forward (not merely a historical
    #: dump)? Feeds `data_forward_available`
    #: (`qlab.venues.derive.data_forward_available`), together with the
    #: snapshot's actual source.
    free_public_data: bool | None = None

    #: Can several legs be filled as one all-or-nothing unit on this venue?
    #: Feeds `atomic_execution`
    #: (`qlab.venues.derive.atomic_execution`) for any spec that needs more
    #: than one leg open at once (`simultaneous_legs > 1`).
    supports_atomic_multileg: bool | None = None

    #: Free text naming the mechanism that stops an uncovered leg from
    #: persisting after a multi-leg entry partially fails, when legs cannot
    #: fill atomically — e.g. an active unwind/rollback path, a
    #: hedge-of-last-resort, a same-block cancel-all. `None` means "nothing
    #: prevents it" (or "not investigated"), not "atomic" — see
    #: `qlab.venues.derive.atomic_execution` for exactly how this and
    #: `supports_atomic_multileg` combine.
    uncovered_leg_recovery: str | None = None

    @field_validator("id", "title")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def load_venue(venue_id: str, *, venues_dir: Path | None = None) -> VenueConfig | None:
    """Load `<venues_dir>/<venue_id>.yaml`, or `None` if no such file exists.

    Returning `None` for a missing venue file — rather than raising — is
    deliberate (docs/TASKS.md T13: "A venue with no config file ... must
    leave the metric absent"). `qlab.pipeline.evaluate` must be able to
    evaluate a spec that names an unconfigured venue and get an honest
    `needs-more-data` verdict, not a crashed pipeline:
    `qlab.venues.derive`'s functions all treat a `None` venue exactly like
    one where every field happens to be unset.

    `venues_dir` defaults to the repo-root `venues/` directory
    (`DEFAULT_VENUES_DIR`); tests pass an isolated `tmp_path` instead so
    they never depend on — or are broken by edits to — the real venue
    configs.
    """
    directory = venues_dir if venues_dir is not None else DEFAULT_VENUES_DIR
    path = directory / f"{venue_id}.yaml"
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"venue file {path} must contain a YAML mapping at top level")
    return VenueConfig.model_validate(raw)


__all__ = ["DEFAULT_VENUES_DIR", "VenueConfig", "load_venue"]
