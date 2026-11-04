"""One idea, one check (docs/ONE_CHECK.md).

Owner, 2026-10-31, on the causal twin that doubled every idea: «не уверен что нам нужна вот такая
дубликация. поидее просто для стратегии достаточна проверка. так мы гоняем все 2х хотя все равно
судим по двойнику»; on the proposal to change it: «да, делай после пайплайна», and on 2026-11-01
«да, делай по порядку».

Until now an idea whose numbers a person or an agent chose was run on the stand as it was, and then
again as its causal twin (`<idea>-causal`, `qlab.twin`), whose numbers code chooses every month from
the past only: two ideas, two pages, two to three times the runs, and the verdict came from the
twin. Now an idea is ONE idea with ONE check:

1. it declares up front the values of the numbers it leaves open (`StrategySpec.choice`: the lab's
   lists of `night/own_numbers.yaml` or a list a source gives), or says why there is nothing to
   choose (`choice.nothing_to_choose`);
2. code chooses among the values from the past only (`qlab.strategies.retune.Retune`, expanding
   window, after-cost Sharpe on the strategy's own money) -- since rules 2026-11-04.1 ONCE, at the
   stand's cut, on the first 2/3 of the window, held on the last third (owner 2026-11-04: «все
   стратегии должны проходить проверку вперед по 2/3 и 1/3 правилу»; until then every month) --
   and the verdict comes at once on the whole history of the venue (Binance, since 2019):
   `check_spec` builds that form with the SAME machinery that built a
   twin (`qlab.twin.causal_twin`), but under the idea's own id, so there is no second idea and no
   second page;
3. the Binance verdict is the idea's own run, the Hyperliquid verdict is its second stage
   (`<idea>-hl`, `qlab.pipeline.stage2`): both as `docs/VENUE_SPLIT.md` says.

Every value of every list is a trial of its own (`record_candidates`), so deflation counts every
number that was tried, as it did for the lab's own numbers (`qlab.own_numbers`, rule 4).

Three things are NOT checked this way, and say so:

- books on paper or live (`night/paper_books.yaml`): their numbers are fixed on production, and
  their check is real trading against the stand;
- an idea with nothing to choose from: it is judged with the numbers it gives, on the last third
  of its window like every spec (rules 2026-11-04.1), and `choice.nothing_to_choose` is the reason
  the page shows;
- an idea that existed before this rule has no `choice` and is judged as it was (its twin, if it
  has one, is an ordinary idea).

The module reads specs and the lists and builds specs; the only things it writes are the trials of
`record_candidates` and, for a check it cannot make, the entry in the ledger of refusals
(`data/twins/ledger.json`, the file the ladder already reads its reasons from). No agent is called,
no token is spent.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from qlab import knowable_coins, own_numbers, twin
from qlab.pipeline.spec import Choice, StrategySpec

STORE = Path("data/one_check")
LISTING_FILE = "binance_listing.json"
LISTING_MAX_AGE_DAYS = 30
"""Binance's list of perpetuals changes slowly, and the check only needs to know which of a
Hyperliquid spec's coins exist there: a month-old list is fine and saves a network call on every
night's re-run of the idea."""

NO_DECLARATION = (
    "the spec declares no list of values for the numbers it leaves open (`choice.axes` or "
    "`choice.candidates`) and does not say why there is nothing to choose "
    "(`choice.nothing_to_choose`); an idea has one check, and code needs the values to choose "
    "among (docs/ONE_CHECK.md)")
LEGACY_OPEN = (
    "open numbers are declared in `choice.axes` now, not as a list inside `params`: the first "
    "value stays in `params`, the whole list goes to `choice.axes`, and the source of the list "
    "to `choice.source` (docs/ONE_CHECK.md)")


class OneCheckRefused(ValueError):
    """The idea cannot have its check: the message is the reason, in the code's words."""


@dataclass(frozen=True)
class Prepared:
    """An idea ready for the stand: `spec` is what the stand runs (the idea's check, or the spec as
    it is when there is nothing to choose)."""

    spec: StrategySpec
    declared: str
    """`lists` (the check form of the idea) or `nothing` (the idea as it is, with its reason)."""
    candidates: tuple[str, ...] = ()
    candidate_trials: tuple[int, ...] = ()
    first_choice: date | None = None
    reason: str | None = None
    """For `nothing`: the idea's own words for why there is nothing to choose."""
    heavy: str | None = None
    """Why the check's data is too heavy to fetch in a night (`qlab.twin_stage.heavy_data`)."""


# --------------------------------------------------------------------------
# What the idea declares


def declares_lists(spec: StrategySpec) -> bool:
    """Whether the spec declares values for code to choose among (its check is the causal form)."""
    return spec.choice is not None and spec.choice.lists


def nothing_to_choose(spec: StrategySpec) -> str | None:
    """The idea's own reason for having no list, or None."""
    return None if spec.choice is None else spec.choice.nothing_to_choose


def _count(choice: Choice) -> int:
    if choice.candidates:
        return len(choice.candidates)
    return math.prod(len(values) for values in choice.axes.values())


def _cited_files(source: str) -> list[str]:
    from qlab.pipeline.sources import _PATH_TOKEN

    return _PATH_TOKEN.findall(source)


def declaration_problems(spec: StrategySpec, config_path: Path = own_numbers.CONFIG, *,
                         require: bool = True) -> list[str]:
    """What is wrong with the way the spec declares its check; empty means it may run.

    With `require` (a NEW idea: the researcher's, the generator's, the implementer's) a spec that
    declares neither a list nor why there is none is refused. A declared list must stay within
    `max_candidates`, the spec's `params` must carry the first value of every list, and an axis is
    either the WHOLE list `night/own_numbers.yaml` declares for that parameter (never a subset the
    agent picked) or sourced from another file the citation names."""
    choice = spec.choice
    if choice is None:
        if own_numbers.open_params(spec, config_path):
            return [LEGACY_OPEN]
        return [NO_DECLARATION] if require else []
    out: list[str] = []
    if spec.selects_causally or spec.params_fit_from is not None:
        out.append("selects_causally and params_fit_from are code's to set for a declared list")
    if not choice.lists:
        return out
    config = own_numbers.load_config(config_path)
    cap = int(config["max_candidates"])
    if _count(choice) > cap:
        out.append(f"{_count(choice)} variants in the declared lists; at most {cap} "
                   f"({config_path}, max_candidates)")
    lists = config.get("lists") or {}
    cited = _cited_files(choice.source)
    others = [f for f in cited if f != str(config_path)]
    for path, values in choice.axes.items():
        declared = lists.get(path)
        whole = declared is not None and sorted(float(v) for v in declared) == sorted(
            float(v) for v in values)
        if whole and str(config_path) in choice.source:
            continue
        if not others:
            hint = (f"; the whole declared list is {declared}" if declared is not None
                    else f"; {config_path} declares no list for it")
            out.append(f"{path} = {values}: neither the whole list of {config_path} nor sourced "
                       f"from another file the citation names{hint}")
    grid, why = twin.card_grid(spec, axes=dict(choice.axes) or None,
                          candidates=dict(choice.candidates) or None, cite=choice.source,
                          max_candidates=max(cap, _count(choice)), declared_by="choice")
    if grid is None:
        out.append(why or "the declared lists cannot be turned into a grid")
    return out


# --------------------------------------------------------------------------
# Binance's listing, kept for a month


def binance_listing(*, store: Path | None = None, today: date | None = None,
                    fetch=None) -> list[str]:
    """Binance's perpetuals and their spot columns, as `twin_stage._binance_listing` reads them,
    kept in `data/one_check/` for `LISTING_MAX_AGE_DAYS`. A listing that cannot be read falls back
    to the kept one, however old; no kept one and no network is an error."""
    from qlab import twin_stage

    path = (store or STORE) / LISTING_FILE
    today = today or datetime.now(UTC).date()
    kept: dict = {}
    try:
        kept = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        kept = {}
    names = kept.get("names") if isinstance(kept, dict) else None
    day = kept.get("day") if isinstance(kept, dict) else None
    if names and day and (today - date.fromisoformat(day)).days <= LISTING_MAX_AGE_DAYS:
        return list(names)
    try:
        fresh = list((fetch or twin_stage._binance_listing)())
    except Exception:  # noqa: BLE001 - an old listing is better than a night without the check
        if names:
            return list(names)
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"day": today.isoformat(), "names": fresh}), encoding="utf-8")
    return fresh


def peers_of(spec: StrategySpec, spec_dir: Path = Path("specs")) -> list[tuple[StrategySpec, str]]:
    """The other specs of the lab (with where they live): they may date a hand-picked coin list,
    the way the base of an experiment dated a twin's (`qlab.twin.chosen_list`, `peers`)."""
    from qlab.pipeline.spec import load_spec

    out: list[tuple[StrategySpec, str]] = []
    for path in sorted(spec_dir.rglob("*.yaml")):
        if path.stem.endswith(twin.TWIN_SUFFIX):
            continue
        try:
            other = load_spec(path)
        except Exception:  # noqa: BLE001 - an unloadable spec is not a peer
            continue
        if other.idea_id != spec.idea_id:
            out.append((other, path.as_posix()))
    return out


# --------------------------------------------------------------------------
# The check


def check_spec(spec: StrategySpec, *, config_path: Path = own_numbers.CONFIG,
               venue_names: Collection[str] | None = None,
               coin_params: Collection[str] = twin.COIN_LIST_PARAMS,
               seconds_per_instrument: float = twin.SECONDS_PER_INSTRUMENT,
               peers: list[tuple[StrategySpec, str]] | None = None,
               spec_path: Path | str | None = None,
               honest_from: tuple[date, str] | None = None,
               execution_venue: str | None = None,
               volume_of: Callable[[str], pd.DataFrame | None] | None = None,
               knowable_config: knowable_coins.KnowableConfig | None = None) -> twin.TwinResult:
    """The form in which the idea's one check runs: the same idea (same id), the numbers chosen by
    code ONCE among the values it declares, at the stand's cut, on the first 2/3 of the window,
    and held on the last third (rules 2026-11-04.1; `refit_every: once`, `choose_at: split`), on
    Binance's history. A result with a `reason` instead of a spec says why the check cannot be
    made.

    Pure but for two reads: Binance's listing when the spec names coins of another venue
    (`binance_listing`, kept a month) and the lab's specs when it names a hand-picked coin list
    (`peers_of`). `honest_from` is a card's statement of the first day its data honestly exists
    (`qlab.twin.causal_twin`). `volume_of` gives a venue's daily traded value for the
    knowable-coins check (`qlab.knowable_coins`: a hand-picked list that was knowable on every
    past day is judged from the start of the data); by default the raw cache is read, and
    without it the list is dated by hand."""
    if not declares_lists(spec):
        return twin.TwinResult(None, nothing_to_choose(spec) or NO_DECLARATION)
    config = own_numbers.load_config(config_path)
    grid, why = twin.choice_grid(spec, max_candidates=int(config["max_candidates"]))
    if grid is None:
        return twin.TwinResult(None, why or NO_DECLARATION)
    names = venue_names
    if names is None and twin.needs_listing(spec, None, coin_params=coin_params):
        names = binance_listing()
    if peers is None:
        peers = peers_of(spec) if spec.data.interval != "1d" else []
    return twin.causal_twin(
        spec, spec_path=spec_path, venue_names=names, coin_params=coin_params,
        seconds_per_instrument=seconds_per_instrument, peers=peers, grid=grid, keep_id=True,
        config_path=config_path, honest_from=honest_from, execution_venue=execution_venue,
        volume_of=volume_of or knowable_coins.raw_volume, knowable_config=knowable_config,
        choose_once=True)


def rerun_form(spec: StrategySpec) -> StrategySpec:
    """The spec a night re-run evaluates for an idea that declares lists (`qlab.night.rerun_spec`):
    its check. Raises `OneCheckRefused` with the reason when the check cannot be made."""
    result = check_spec(spec)
    if result.spec is None:
        raise OneCheckRefused(result.reason or NO_DECLARATION)
    if result.spec.coins_knowable is not None:
        try:
            knowable_coins.remember(spec.idea_id, result.spec.coins_knowable)
        except OSError:  # the page line is a convenience; a night without it still judges
            pass
    return result.spec


# --------------------------------------------------------------------------
# Trials of every value


def candidate_specs(checked: StrategySpec) -> dict[str, StrategySpec]:
    """Every value of the lists as a plain spec of the idea, on the check's data and costs: what
    one trial of that value is. {name: spec}."""
    from qlab.strategies.retune import expand_grid

    params = checked.params
    inner = str(params["inner_code_ref"])
    out = {}
    for name, cand in expand_grid(dict(params["base_params"]), dict(params["grid"])).items():
        out[name] = checked.model_copy(update={
            "code_ref": inner, "params": cand, "selects_causally": False,
            "params_fixed_at": None, "params_fixed_evidence": None})
    return out


def record_candidates(checked: StrategySpec, session, ruleset) -> list[int]:
    """Every value of the lists as a trial of its own under the idea (no noise books, the idea's
    status untouched), so that the ledger counts every number that was tried. Returns the trial
    ids."""
    from qlab.pipeline.evaluate import evaluate_spec

    ids = []
    for candidate in candidate_specs(checked).values():
        out = evaluate_spec(candidate, session=session, ruleset=ruleset,
                            update_idea_status=False, check_sources=False)
        ids.append(out.trial_id)
    return ids


def ensure_idea(session, spec: StrategySpec, notes: str) -> None:
    """The idea in the registry, if it is not there yet (the trials need the idea to exist)."""
    from qlab.registry import repo
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType

    if session.get(Idea, spec.idea_id) is None:
        repo.upsert_idea(session, id=spec.idea_id, title=spec.title,
                         source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                         profile=Profile.OTHER, notes=notes)
        session.flush()


def note_refusal(idea_id: str, reason: str, spec_path: str, *, ledger_path: Path | None = None
                 ) -> None:
    """Write why an idea has no check into the ledger the ladder reads (data/twins/ledger.json)."""
    from qlab import twin_stage

    path = ledger_path or twin_stage.LEDGER
    ledger = twin_stage.load_ledger(path)
    ledger[idea_id] = {"reason": reason, "fingerprint": "one-check",
                       "day": datetime.now(UTC).date().isoformat(), "spec": spec_path}
    twin_stage.save_ledger(ledger, path)


def prepare(spec: StrategySpec, session, ruleset, *, record: bool = True,
            spec_path: Path | str | None = None, config_path: Path = own_numbers.CONFIG,
            allow_heavy: bool = False) -> Prepared:
    """An idea ready for the stand. Raises `OneCheckRefused` for a spec that declares nothing (a
    new idea must) or whose check cannot be made.

    - `choice.nothing_to_choose`: the spec as it is.
    - a declared list: the check form; with `record`, every value is a trial first; with a check
      whose data is too heavy for a night (`heavy`), nothing is run or recorded here, unless
      `allow_heavy` (a hand run)."""
    problems = declaration_problems(spec, config_path)
    if problems:
        raise OneCheckRefused("; ".join(problems))
    if not declares_lists(spec):
        return Prepared(spec, "nothing", reason=nothing_to_choose(spec))
    result = check_spec(spec, config_path=config_path, spec_path=spec_path)
    if result.spec is None:
        raise OneCheckRefused(result.reason or NO_DECLARATION)
    heavy = None if allow_heavy else _heavy(result.spec)
    trials: list[int] = []
    if record and heavy is None:
        trials = record_candidates(result.spec, session, ruleset)
    return Prepared(result.spec, "lists", candidates=tuple(result.candidates),
                    candidate_trials=tuple(trials), first_choice=result.spec.params_fixed_at,
                    heavy=heavy)


def _heavy(checked: StrategySpec) -> str | None:
    from qlab import twin_stage

    cfg = twin_stage.load_config() or twin_stage.Config(
        enabled=True, max_new_twins_per_night=0, slow_twin_minutes=20.0)
    return twin_stage.heavy_data(checked, cfg.slow_twin_minutes, cfg.max_hourly_instruments,
                                 cfg.seconds_per_instrument)


__all__ = ["LEGACY_OPEN", "NO_DECLARATION", "OneCheckRefused", "Prepared", "binance_listing",
           "candidate_specs", "check_spec", "declaration_problems", "declares_lists",
           "ensure_idea", "nothing_to_choose", "note_refusal", "peers_of", "prepare",
           "record_candidates", "rerun_form"]
