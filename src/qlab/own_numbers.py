"""Numbers from q-lab's own data when no paper gives them (docs/OWN_NUMBERS.md).

Owner, 2026-10-26, on the rule and its four limits: «Разрешить с этими четырьмя
ограничениями ... откроются механизмы про фандинг. - еще я бы довавил правило - придумать
свой механизм основываясь на идее».

A generated spec may leave a parameter open by giving, instead of one number, the WHOLE list
`night/own_numbers.yaml` declares for that parameter, cited to that file. Code then:

1. checks the list is exactly the declared one (no subset the agent picked) and that the
   product of the open lists stays within `max_candidates`;
2. turns the spec into a `qlab.strategies.retune:Retune` spec that chooses ONE value, by the
   Sharpe it has realised, at the cut (`refit_every: once`) and holds it to the end -- the
   choice sees only data before the cut;
3. marks the result `selects_causally` from the cut: the stand judges only the days from it,
   which the choice never saw;
4. writes every candidate as a trial of its own (`record_candidates`), so deflation counts
   every value that was tried.

The cut is where the owner's split puts the forward part (`forward_share`, the last third of
the window, rules/2026-10-24.1.yaml), or the spec's own `params_fixed_at` when its sources
were published later.
"""

from __future__ import annotations

import itertools
from datetime import date, timedelta
from pathlib import Path

import yaml

from qlab.pipeline.sources import QLAB_ROOT, covering_key
from qlab.pipeline.spec import StrategySpec

CONFIG = Path("night/own_numbers.yaml")
RETUNE = "qlab.strategies.retune:Retune"


def load_config(path: Path = CONFIG) -> dict:
    """The declared lists, validated: positive numbers, a cap on candidates, a share."""
    path = path if path.is_absolute() else QLAB_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    lists = config.get("lists") or {}
    for name, values in lists.items():
        if not values or len(values) < 2 or any(
                not isinstance(v, int | float) or isinstance(v, bool) or v <= 0 for v in values):
            raise ValueError(f"{path}: list {name} needs at least two positive numbers")
    cap = config.get("max_candidates")
    if not isinstance(cap, int) or cap < 2:
        raise ValueError(f"{path}: max_candidates must be an integer of at least 2")
    share = config.get("forward_share")
    if not isinstance(share, int | float) or not 0 < share < 1:
        raise ValueError(f"{path}: forward_share must be between 0 and 1")
    return config


def _leaves(prefix: str, value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(f"{prefix}.{key}" if prefix else str(key), item)
    else:
        yield prefix, value


def open_params(spec: StrategySpec, config_path: Path = CONFIG) -> dict[str, list[float]]:
    """The parameters the spec leaves open: a list value whose citation names the config."""
    out = {}
    for path, value in _leaves("", spec.params):
        if not isinstance(value, list):
            continue
        key = covering_key(path, spec.sources)
        if key is not None and str(config_path) in spec.sources.get(key, ""):
            out[path] = [float(v) for v in value]
    return out


def problems(spec: StrategySpec, config: dict, config_path: Path = CONFIG) -> list[str]:
    """What is wrong with the spec's open parameters (empty when it has none)."""
    found = open_params(spec, config_path)
    lists = config.get("lists") or {}
    out = []
    for path, values in found.items():
        declared = lists.get(path)
        if declared is None:
            out.append(f"{path}: {config_path} declares no list for it")
        elif sorted(values) != sorted(float(v) for v in declared):
            out.append(f"{path} = {values}: must be the whole declared list "
                       f"{declared} of {config_path}, not a choice from it")
    count = 1
    for values in found.values():
        count *= len(values)
    if found and count > int(config["max_candidates"]):
        out.append(f"{count} candidates from the open parameters; at most "
                   f"{config['max_candidates']} ({config_path})")
    if spec.selects_causally or spec.params_fit_from is not None:
        out.append("selects_causally and params_fit_from are code's to set for open parameters")
    return out


def cut_date(spec: StrategySpec, config: dict) -> date:
    """Where the value is chosen: the first day of the last `forward_share` of the window, or
    the spec's `params_fixed_at` if later (its sources could not have seen the data before)."""
    start, end = spec.data.start, spec.data.end
    days = (end - start).days
    cut = start + timedelta(days=int(round(days * (1 - float(config["forward_share"])))))
    if spec.params_fixed_at is not None and spec.params_fixed_at > cut:
        cut = spec.params_fixed_at
    return cut


def _label(path: str, value: float) -> str:
    return f"{path.split('.')[-1]}={value:g}"


def candidates(spec: StrategySpec, config_path: Path = CONFIG) -> dict[str, dict]:
    """{name: params} for every combination of the open values, each a plain spec's params."""
    import json

    found = open_params(spec, config_path)
    out = {}
    for combo in itertools.product(*[[(p, v) for v in vals] for p, vals in found.items()]):
        params = json.loads(json.dumps(spec.params))
        for path, value in combo:
            *parents, leaf = path.split(".")
            node = params
            for name in parents:
                node = node[name]
            node[leaf] = int(value) if value == int(value) else value
        out["/".join(_label(p, v) for p, v in combo)] = params
    return out


def to_retune(spec: StrategySpec, config: dict, config_path: Path = CONFIG) -> StrategySpec:
    """The spec as a one-choice Retune spec: the value chosen at the cut, judged after it."""
    found = open_params(spec, config_path)
    if not found:
        raise ValueError(f"{spec.idea_id}: no open parameters to choose")
    cut = cut_date(spec, config)
    base = next(iter(candidates(spec, config_path).values()))
    grid = {path: {_label(path, v): {path: int(v) if v == int(v) else v} for v in values}
            for path, values in found.items()}
    cited = ", ".join(sorted(found))
    spec_level = ("costs.", "min_leg_notional", "regime_claim.", "strategy_capital_usd")
    sources = {f"base_params.{k}": v for k, v in spec.sources.items()
               if not k.startswith(spec_level)}
    sources.update({k: v for k, v in spec.sources.items() if k.startswith(spec_level)})
    own = (f"{config_path} (version {config['version']}): the declared lists for {cited}; "
           "owner 2026-10-26: own numbers chosen on data before the cut")
    sources.update({
        "inner_code_ref": f"the generated spec's own code_ref; {own}",
        "grid": own,
        "refit_every": f"{own}; one choice at the cut (qlab.strategies.retune, once)",
        "choose_at": f"{own}; the cut, forward_share of {config_path}",
        "window": f"{own}; the choice sees every realised day before the cut",
        "objective": f"{own}; the realised Sharpe, as every retune of the lab (docs/RETUNE.md)",
    })
    for name in ("taker_fee_bps", "slippage_bps"):
        sources[f"score_costs.{name}"] = spec.sources.get(f"costs.{name}") or spec.sources.get(
            "costs", "")
    params = {
        "inner_code_ref": spec.code_ref,
        "base_params": base,
        "grid": grid,
        "refit_every": "once",
        "choose_at": cut.isoformat(),
        "window": "expanding",
        "objective": "sharpe_net",
        "score_costs": {"taker_fee_bps": spec.costs.taker_fee_bps,
                        "slippage_bps": spec.costs.slippage_bps},
    }
    return spec.model_copy(update={
        "code_ref": RETUNE, "params": params, "sources": sources, "selects_causally": True,
        "params_fixed_at": cut,
        "params_fixed_evidence": (
            f"the open parameters ({cited}) are chosen once, at {cut}, by their realised Sharpe "
            f"on the data before it ({config_path}, owner 2026-10-26); every day from {cut} on "
            "trades that choice without having seen the day"
            + (f"; the sources' own date was {spec.params_fixed_at}"
               if spec.params_fixed_at else "")),
    })


def prepare(spec: StrategySpec, session, ruleset, *,
            config_path: Path = CONFIG) -> tuple[StrategySpec, dict]:
    """A spec with open numbers made ready for the stand: every candidate written as a trial,
    then the one-choice spec to evaluate. Returns it and what to report (the candidate trials,
    the cut). A spec without open numbers comes back as it is."""
    if not open_params(spec, config_path):
        return spec, {}
    config = load_config(config_path)
    found = problems(spec, config, config_path)
    if found:
        raise ValueError("; ".join(found))
    trials = record_candidates(spec, session, ruleset,
                               config_path=config_path)
    retune = to_retune(spec, config, config_path)
    return retune, {"candidate_trials": trials, "cut": retune.params_fixed_at.isoformat()}


def chosen_params(retune_spec: StrategySpec, session) -> dict | None:
    """The candidate the one-choice spec chose at its cut (its params), or None if it chose
    none. Read from the stand's own run of it (the wrapper's cache makes this cheap)."""
    from qlab.pipeline.evaluate import cut_truncated_history, resolve_panel
    from qlab.strategies.retune import Retune, expand_grid

    panel = cut_truncated_history(resolve_panel(session, retune_spec))
    run = Retune().run(panel, retune_spec.params)
    names = run.choices.dropna()
    if names.empty:
        return None
    return expand_grid(retune_spec.params["base_params"], retune_spec.params["grid"])[
        str(names.iloc[-1])]


def fixed_at_choice(spec: StrategySpec, params: dict, cut: date) -> StrategySpec:
    """The generated spec with the chosen values in place of its open lists: what the second
    stage carries to another venue (the SAME values, docs/TWO_STAGE.md), forward from the cut."""
    return spec.model_copy(update={
        "params": params, "params_fixed_at": cut,
        "params_fixed_evidence": f"the values chosen at {cut} on the first venue's data before "
                                 f"it ({CONFIG}, owner 2026-10-26)"})


def record_candidates(spec: StrategySpec, session, ruleset, *,
                      config_path: Path = CONFIG) -> list[int]:
    """Every candidate as its own trial under the spec's idea (no noise, the idea's status
    untouched): the ledger then counts every value tried. Returns the trial ids."""
    from qlab.pipeline.evaluate import evaluate_spec

    ids = []
    for params in candidates(spec, config_path).values():
        candidate = spec.model_copy(update={"params": params, "params_fixed_at": None,
                                            "params_fixed_evidence": None})
        out = evaluate_spec(candidate, session=session, ruleset=ruleset,
                            update_idea_status=False, check_sources=False)
        ids.append(out.trial_id)
    return ids


__all__ = ["CONFIG", "candidates", "chosen_params", "cut_date", "fixed_at_choice",
           "load_config", "open_params", "prepare", "problems", "record_candidates", "to_retune"]
