"""The night's «Надстройки» stage: every overlay and every pair on a strategy that passed
(docs/OVERLAYS.md, docs/NIGHT.md).

Owner, 2026-10-28: «это все идеи для платформы. было бы здорово проверять разные варианты
возможно их можно смиксовать». A strategy that passed stage 1 is the BASE. For it the stage
generates the overlay variants (`qlab.overlay_variants`: every single overlay and every pair, no
triples), writes each as a spec file under `specs/overlays/<base>/`, registers it as a variant of
the base, runs it on the stand like any idea (200 random books, every trial written, deflation
counts it in the base's family), sends a survivor to the second stage on Hyperliquid, and compares
it with the base on the days both are judged on (`qlab.overlay_compare`). The base's forward series
is described too (`qlab.overlay_diagnostics`). Code only: no agent, no tokens, outside the budget
guard.

**Which bases.** The ideas listed in `night/overlays.yaml` (`bases`), then, when `auto` is on,
every idea that passed stage 1 (`qlab.forward_series.passed_ideas`: status `validated` or a
latest run that passed every rule), minus variants, second stages and ideas with no single spec
file. Order: the strategies of `priority` first (the config puts trend and XSMOM first and Bv2 as
a control), the listed ones before the discovered ones.

**Bounded.** At most `max_bases_per_night` bases and `max_variants_per_night` NEW variants a
night; what does not fit waits and the report names it. The night's stop time is checked before
each variant (`qlab.night.past_stop`): a started variant always finishes. **Idempotent.** A
variant is run once; the ledger (`data/overlays/ledger.json`) keeps a fingerprint of its spec
(without the data end, which moves every night) and the variant is run again only when the spec
changes -- the base's numbers, the overlay family or the code that writes the spec.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from qlab.pipeline.spec import StrategySpec

CONFIG = Path("night/overlays.yaml")
LEDGER = Path("data/overlays/ledger.json")
SPEC_DIR = Path("specs")
SECTION_TITLE = "Надстройки: когда стратегия входит и сколько держит"
LEDGER_VERSION = 1
"""Bumped when the rules that build a variant change, so an old fingerprint is not kept."""
DATA_END_NIGHT, DATA_END_BASE = "night", "base"


# --------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class Config:
    enabled: bool
    max_bases_per_night: int
    max_variants_per_night: int
    slow_variant_minutes: float
    priority: tuple[str, ...]
    bases: tuple[str, ...]
    auto: bool
    second_stage: bool
    meta_decisions: dict[str, tuple[str, ...]]
    """Base idea -> the stored level-2 decision sets (`data/regimes/meta-<name>`) built for
    that very strategy; the `meta_gate` overlay is offered only for such a base."""


def load_config(path: Path = CONFIG) -> Config | None:
    """`night/overlays.yaml`; None when the file is missing (the stage is then off)."""
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Config(
            enabled=bool(raw["enabled"]),
            max_bases_per_night=int(raw["max_bases_per_night"]),
            max_variants_per_night=int(raw["max_variants_per_night"]),
            slow_variant_minutes=float(raw["slow_variant_minutes"]),
            priority=tuple(raw.get("priority") or ()),
            bases=tuple(raw.get("bases") or ()),
            auto=bool(raw.get("auto", True)),
            second_stage=bool(raw.get("second_stage", True)),
            meta_decisions={k: tuple(v) for k, v in (raw.get("meta_decisions") or {}).items()})
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None


# --------------------------------------------------------------------------
# The ledger (a pure file, no database)


def load_ledger(path: Path = LEDGER) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_ledger(ledger: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)


def fingerprint(spec: StrategySpec) -> str:
    """What a finished variant depends on: its whole spec except the data end, which moves every
    night without changing the variant, and the version of the rules that build variants."""
    raw = spec.model_dump(mode="json")
    raw["data"].pop("end", None)
    text = json.dumps(raw, sort_keys=True, default=str, ensure_ascii=False)
    return f"v{LEDGER_VERSION}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


# --------------------------------------------------------------------------
# Bases


@dataclass
class Base:
    idea_id: str
    path: Path
    spec: StrategySpec
    strategy: str | None
    listed: bool


def find_bases(session, specs: dict[str, list[Path]], config: Config, *,
               only: Collection[str] | None = None) -> tuple[list[Base], dict[str, str]]:
    """(the bases in the order they are taken, ideas named in `only` that cannot be one with the
    reason). Without `only`: the listed ones and, with `auto`, every idea that passed stage 1."""
    from qlab.forward_series import passed_ideas
    from qlab.overlay_variants import is_variant_id
    from qlab.pipeline.spec import load_spec
    from qlab.registry.families import strategy_key
    from qlab.registry.models import Idea
    from qlab.registry.stages import is_stage2_child

    if only is not None:
        wanted = list(only)
    else:
        wanted = list(config.bases) + (
            [i for i in passed_ideas(session) if i not in config.bases] if config.auto else [])
    found: list[Base] = []
    refused: dict[str, str] = {}
    for idea_id in dict.fromkeys(wanted):
        idea = session.get(Idea, idea_id)
        if idea is None:
            refused[idea_id] = "the idea is not in the registry"
        elif is_variant_id(idea_id) or is_stage2_child(idea):
            refused[idea_id] = "a variant or a second stage is not a base"
        elif len(specs.get(idea_id) or []) != 1:
            refused[idea_id] = "no spec file" if not specs.get(idea_id) else "several spec files"
        else:
            path = specs[idea_id][0]
            try:
                spec = load_spec(path)
            except Exception as exc:  # noqa: BLE001 - a broken file fails in its own run
                refused[idea_id] = f"the spec does not load: {exc}"
                continue
            found.append(Base(idea_id, path, spec, strategy_key(spec.code_ref, spec.params),
                              idea_id in config.bases))

    def rank(base: Base) -> tuple:
        order = config.priority.index(base.strategy) if base.strategy in config.priority \
            else len(config.priority)
        listing = config.bases.index(base.idea_id) if base.listed else len(config.bases)
        return order, listing, base.idea_id

    return sorted(found, key=rank), refused


def last_run_end(session, idea_id: str) -> date | None:
    """The last day of the data the idea's latest run saw."""
    from qlab.registry.models import Spec, Trial

    row = (session.query(Spec.data_requirements).join(Trial, Trial.spec_id == Spec.id)
           .filter(Spec.idea_id == idea_id, Trial.route.isnot(None))
           .order_by(Trial.id.desc()).first())
    end = (row[0] or {}).get("end") if row else None
    return None if end is None else date.fromisoformat(str(end)[:10])


def ensure_overlay_idea(spec: StrategySpec, base_id: str) -> None:
    """The variant as an idea of the registry: a variant of its base (one branch, not one more
    place among the strategy's variants), counted in the base's family -- it inherits the base's
    driver, so `qlab.registry.deflation.family_trials` (scope `driver`) counts its trials."""
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea
    from qlab.twin_stage import ensure_variant_idea

    ensure_variant_idea(
        spec, base_id, f"Overlay variant of {base_id} (qlab.overlay_stage, docs/OVERLAYS.md): the "
                       "same strategy with overlays on its weights, judged against it.")
    with session_scope() as session:
        idea, node = session.get(Idea, spec.idea_id), session.get(Idea, base_id)
        seen = 0
        while node is not None and node.driver_id is None and node.parent_id and seen < 8:
            node, seen = session.get(Idea, node.parent_id), seen + 1
        if idea is not None and idea.driver_id is None and node is not None:
            idea.driver_id = node.driver_id


# --------------------------------------------------------------------------
# The stage


def overlays_stage(day: date, end: date, state: dict, *, noise_trials: int = 200,
                   deployable_capital_usd: float = 3000.0, stop_at: datetime | None = None,
                   only: Collection[str] | None = None, chains_only: Collection[str] | None = None,
                   max_variants: int | None = None, config_path: Path = CONFIG,
                   spec_dir: Path = SPEC_DIR, ledger_path: Path = LEDGER,
                   data_end: str = DATA_END_NIGHT, kinds: Collection[str] = ("causal",),
                   second_stage: bool | None = None, dry_run: bool = False,
                   store: Path | None = None, label: str = "overlays") -> None:
    """Build, run, judge and compare the overlay variants of the strategies that passed; write
    what happened to `state["overlays"]` for the morning report. See the module docstring.

    `only` names the bases (then the nightly caps do not hold the bases back, unless
    `max_variants` is given); `chains_only` limits the variants to those whose overlays are
    named like `own_results` or `own_results+gradual_entry`; `data_end` is `night` (the last
    closed day) or `base` (each base's last run, so no snapshot is fetched); `dry_run` writes
    and runs nothing."""
    from qlab import night
    from qlab.registry.db import session_scope
    from qlab.waiting import spec_files

    config = load_config(config_path)
    if config is None or not config.enabled:
        return
    block = state.setdefault("overlays", {"entries": [], "new": 0})
    block["deferred"], block["bases"], block["notes"] = [], [], []
    ledger = load_ledger(ledger_path)
    cap = max_variants if max_variants is not None else (
        None if only else config.max_variants_per_night)
    run = _Run(day=day, end=end, state=state, block=block, config=config,
               noise_trials=noise_trials, deployable_capital_usd=deployable_capital_usd,
               stop_at=stop_at, label=label, ledger=ledger, ledger_path=ledger_path,
               spec_dir=spec_dir, data_end=data_end, kinds=tuple(kinds), dry_run=dry_run,
               store=store, forced=bool(only), cap=cap, chains_only=set(chains_only or ()),
               second_stage=config.second_stage if second_stage is None else second_stage)
    with session_scope() as session:
        bases, refused = find_bases(session, spec_files(spec_dir), config, only=only)
    if only:
        block["refused"] = refused
    run.base_limit = len(bases) if only else config.max_bases_per_night
    for base in bases:
        try:
            if run.handle(base) == "stop":
                break
        except Exception as exc:  # noqa: BLE001 - one base's failure must not stop the others
            block["entries"].append({"base": base.idea_id, "variant": None, "action": "error",
                                     "error": f"{type(exc).__name__}: {exc}"})
    if not dry_run:
        night.save_state(day, state)


@dataclass
class _Run:
    day: date
    end: date
    state: dict
    block: dict
    config: Config
    noise_trials: int
    deployable_capital_usd: float
    stop_at: datetime | None
    label: str
    ledger: dict
    ledger_path: Path
    spec_dir: Path
    data_end: str
    kinds: tuple[str, ...]
    dry_run: bool
    store: Path | None
    forced: bool
    cap: int | None
    chains_only: set[str]
    second_stage: bool
    stage2_config: object = None
    base_limit: int = 10**9
    """How many bases may get new variants tonight; a base whose variants are all done does not
    count, so the same two bases cannot hold the others back for ever."""

    def full(self) -> bool:
        return self.cap is not None and self.block["new"] >= self.cap

    def wanted(self, chain: tuple[str, ...]) -> bool:
        return not self.chains_only or "+".join(chain) in self.chains_only

    def handle(self, base: Base) -> str | None:
        """All the variants of one base; "stop" when the night's stop time has passed."""
        from qlab import night
        from qlab.overlay_variants import (
            KIND_CAUSAL,
            available_overlays,
            build_variant,
            chains,
        )
        from qlab.registry.db import session_scope

        with session_scope() as session:
            end = (last_run_end(session, base.idea_id) if self.data_end == DATA_END_BASE
                   else None) or self.end
        extra = {"meta_gate": get_meta_grid(self.config, base.idea_id)}
        planned = [(chain, kind) for chain in chains(available_overlays(extra))
                   for kind in self.kinds if kind == KIND_CAUSAL and self.wanted(chain)]
        entry_base = {"base": base.idea_id, "strategy": base.strategy, "end": end.isoformat(),
                      "variants": []}
        active = False
        for chain, kind in planned:
            try:
                spec = build_variant(base.spec, chain, base_cite=base.path.as_posix(), kind=kind,
                                     extra_grids=extra, end=end)
            except ValueError as exc:
                self.block["entries"].append({
                    "base": base.idea_id, "variant": None, "chain": list(chain),
                    "action": "none", "reason": str(exc)})
                active = True
                continue
            stamp = fingerprint(spec)
            known = self.ledger.get(spec.idea_id) or {}
            if known.get("fingerprint") == stamp and not self.forced:
                entry_base["variants"].append(spec.idea_id)
                if not self.dry_run:
                    self.catch_up(base, spec, entry_base)
                    if known.get("stage2_pending") and self.second_stage and not self.full():
                        self.catch_up_stage2(base, spec, known, end)
                continue  # done once; nothing changed since
            if self.full() or (not active and len(self.block["bases"]) >= self.base_limit):
                self.block["deferred"].append(spec.idea_id)
                continue
            if night.past_stop(self.stop_at, self.state, f"{self.label}:{spec.idea_id}"):
                self.finish_base(base, entry_base, end, active)
                return "stop"
            if not active:
                active = True
                self.block["bases"].append(entry_base)
            entry = {"base": base.idea_id, "variant": spec.idea_id, "chain": list(chain),
                     "kind": kind, "fingerprint": stamp}
            if self.dry_run:
                entry["action"] = "would-run"
                self.block["entries"].append(entry)
                continue
            self.run_variant(base, spec, entry, end)
            self.block["entries"].append(entry)
            if entry["action"] != "error":
                self.block["new"] += 1
                entry_base["variants"].append(spec.idea_id)
                self.ledger[spec.idea_id] = {
                    "fingerprint": stamp, "day": self.day.isoformat(), "base": base.idea_id,
                    "route": entry.get("route"), "trial": entry.get("trial"),
                    "stage2_pending": bool(entry.get("stage2_pending"))}
                save_ledger(self.ledger, self.ledger_path)
                self.state["done"][f"overlays:{spec.idea_id}"] = datetime.now(UTC).isoformat()
            night.save_state(self.day, self.state)
        self.finish_base(base, entry_base, end, active)
        return None

    def finish_base(self, base: Base, entry_base: dict, end: date, active: bool) -> None:
        """A base that got new work tonight is described (its forward series) and kept in the
        report; one whose variants were all done already leaves no trace."""
        if active and not self.dry_run and entry_base not in self.block["bases"]:
            self.block["bases"].append(entry_base)
        if active and not self.dry_run:
            self.describe_base(base, entry_base, end)

    def catch_up(self, base: Base, spec: StrategySpec, entry_base: dict) -> None:
        """A variant that was run but never compared (the comparison failed that night, or the
        stored file is gone) is compared now: no stand run, no new trial."""
        from qlab import overlay_compare
        from qlab.registry.db import session_scope

        if overlay_compare.load_record(spec.idea_id, self.store) is not None:
            return
        try:
            with session_scope() as session:
                record = overlay_compare.compare_specs(
                    base.spec.model_copy(update={"data": spec.data}), spec, session)
            record["trial_id"] = (self.ledger.get(spec.idea_id) or {}).get("trial")
            overlay_compare.save_record(record, self.store)
            self.block["notes"].append(
                f"сравнение с базой для `{spec.idea_id}` досчитано позже: прогон уже был")
        except Exception as exc:  # noqa: BLE001 - tried again tomorrow
            self.block["notes"].append(
                f"сравнение с базой для `{spec.idea_id}` не вышло ({type(exc).__name__}: "
                f"{str(exc)[:120]}); повторится в следующую ночь")

    def catch_up_stage2(self, base: Base, spec: StrategySpec, known: dict, end: date) -> None:
        """The second stage a survivor owes (its first run was made without it): done now, one
        stand run, no new first-stage trial."""
        from qlab import night, twin_stage

        entry = {"base": base.idea_id, "variant": spec.idea_id, "action": "stage2-late",
                 "chain": [], "route": known.get("route"), "trial": known.get("trial")}
        try:
            twin_stage.second_stage(spec, entry, end=end, noise_trials=self.noise_trials,
                                    deployable_capital_usd=self.deployable_capital_usd,
                                    config=self.stage2())
        except Exception as exc:  # noqa: BLE001 - tried again tomorrow
            entry.update(action="error", error=f"{type(exc).__name__}: {exc}")
            self.block["entries"].append(entry)
            return
        self.block["entries"].append(entry)
        self.block["new"] += 1
        self.ledger[spec.idea_id] = {**known, "stage2_pending": False}
        save_ledger(self.ledger, self.ledger_path)
        night.save_state(self.day, self.state)

    def describe_base(self, base: Base, entry_base: dict, end: date) -> None:
        """The three readings of the base's forward series, stored and kept in the report."""
        from qlab import night, overlay_diagnostics
        from qlab.registry.db import session_scope

        try:
            with session_scope() as session:
                payload = overlay_diagnostics.compute_for_spec(
                    night.with_end(base.spec, end), session)
            overlay_diagnostics.save(payload, self.store)
            entry_base["diagnostics"] = overlay_diagnostics.lines_ru(payload)
        except Exception as exc:  # noqa: BLE001 - a description must not stop the comparison
            entry_base["diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    def run_variant(self, base: Base, spec: StrategySpec, entry: dict, end: date) -> None:
        from qlab import overlay_compare, twin_stage
        from qlab.overlay_variants import write_variant
        from qlab.registry.db import session_scope

        path = write_variant(self.spec_dir, base.idea_id, spec)
        entry["spec"] = path.as_posix()
        started = time.monotonic()
        ok = twin_stage.stage1_on_stand(
            spec, entry, ensure_idea=lambda: ensure_overlay_idea(spec, base.idea_id),
            end=end, noise_trials=self.noise_trials,
            deployable_capital_usd=self.deployable_capital_usd,
            slow_minutes=self.config.slow_variant_minutes, kind="overlays")
        if not ok:
            return
        entry["action"] = "run"
        try:
            with session_scope() as session:
                shared = base.spec.model_copy(update={"data": spec.data})
                record = overlay_compare.compare_specs(shared, spec, session)
            record["trial_id"] = entry.get("trial")
            overlay_compare.save_record(record, self.store)
            entry["comparison"] = {k: record[k] for k in (
                "start", "end", "days", "base", "variant", "delta_sharpe", "delta_sharpe_ci",
                "delta_sharpe_p", "delta_ann_return", "delta_max_dd")}
        except Exception as exc:  # noqa: BLE001 - the run stands; the comparison is retried
            entry["comparison_error"] = f"{type(exc).__name__}: {exc}"
        if self.second_stage:
            twin_stage.second_stage(spec, entry, end=end, noise_trials=self.noise_trials,
                                    deployable_capital_usd=self.deployable_capital_usd,
                                    config=self.stage2())
        else:
            # A survivor whose second stage was not asked for (a hand run that must not fetch
            # Hyperliquid's candles) keeps it owed: a later night does it.
            from qlab.pipeline import stage2

            config = self.stage2()
            entry["stage2_pending"] = config is not None and stage2.survives(
                entry.get("route"), config)
        entry["total_seconds"] = round(time.monotonic() - started, 1)

    def stage2(self):
        if self.stage2_config is None:
            from qlab.pipeline import stage2

            try:
                self.stage2_config = stage2.load_config()
            except (OSError, ValueError) as exc:
                self.block["notes"].append(f"night/stage2.yaml не прочитан: {exc}")
                self.stage2_config = False
        return self.stage2_config or None


def get_meta_grid(config: Config, base_id: str) -> dict:
    from qlab.strategies.overlays import get

    names_ = config.meta_decisions.get(base_id) or ()
    return get("meta_gate").grid_for(list(names_)) if names_ else {}


# --------------------------------------------------------------------------
# The morning report


def pct(value: object, digits: int = 1) -> str:
    return f"{value:+.{digits}%}" if isinstance(value, int | float) and value == value else "—"


def num(value: object, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, int | float) and value == value else "—"


def short_name(base_id: str, variant_id: str) -> str:
    return variant_id[len(base_id):].lstrip("-").removeprefix("ov-")


def table_lines(table: dict) -> list[str]:
    """A base's variants against it, as a Markdown table (the verdict at the family's current
    size, `qlab.overlay_compare.verdict`)."""
    rows = [r for r in table["variants"] if r["comparison"]]
    if not rows:
        return []
    out = ["| Надстройка | Дней | В год: база → с надстройкой | Худшее падение: база → с "
           "надстройкой | Шарп: база → с надстройкой | Разница Шарпа (95% интервал) | "
           "p с поправкой | Вердикт |",
           "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        c = row["comparison"]
        b, v = c["base"], c["variant"]
        lo, hi = c["delta_sharpe_ci"]
        out.append(
            f"| `{short_name(table['base_id'], row['idea_id'])}` | {c['days']} "
            f"| {pct(b['ann_return'])} → {pct(v['ann_return'])} "
            f"| {pct(b['max_dd'])} → {pct(v['max_dd'])} "
            f"| {num(b['sharpe'])} → {num(v['sharpe'])} "
            f"| {c['delta_sharpe']:+.2f} ({lo:+.2f} … {hi:+.2f}) "
            f"| {num(c['corrected_p'], 3)} | **{c['verdict_ru']}** |")
    return out


def entry_lines(base_entry: dict, table: dict | None) -> list[str]:
    head = (f"**{base_entry['base']}** ({base_entry.get('strategy') or '—'}), данные по "
            f"{base_entry['end']}")
    out = [head]
    for line in base_entry.get("diagnostics") or []:
        out.append(f"    {line}")
    if base_entry.get("diagnostics_error"):
        out.append(f"    Описание базы не построено: {base_entry['diagnostics_error'][:160]}")
    if table is not None:
        counts = table["counts"]
        out.append(f"    Вариантов сравнено: {table['family_size']}; помогла "
                   f"{counts['helped']}, не помогла {counts['not-helped']}, неотличимо от "
                   f"случайности {counts['indistinguishable']}. Испытаний в семействе базы: "
                   f"{table['family_trials']} (дефляция считает их все).")
        out += [f"    {ln}" for ln in table_lines(table)]
    return out


def tables_for(block: dict, store: Path | None = None) -> dict[str, dict]:
    """The comparison table of each base the block names, read from the registry and the
    stored comparisons; empty when the registry cannot be read."""
    from qlab import overlay_compare
    from qlab.registry.db import session_scope

    try:
        with session_scope() as session:
            return {b["base"]: overlay_compare.table(session, b["base"], store)
                    for b in block.get("bases") or []}
    except Exception:  # noqa: BLE001 - a report must still be written
        return {}


def report_lines(block: dict | None, tables: dict[str, dict] | None = None) -> list[str]:
    """The section's body for the morning report. `tables` maps a base id to
    `qlab.overlay_compare.table` (the report writer reads the registry for it)."""
    if not block:
        return []
    entries = block.get("entries") or []
    if not (entries or block.get("bases") or block.get("notes") or block.get("deferred")
            or block.get("refused")):
        return []
    tables = tables if tables is not None else tables_for(block)
    lines = [
        "Надстройка меняет, когда и сколько стратегия держит: не входить в одном из состояний "
        "рынка, уменьшать размер, когда результаты стали неспокойными или плохими, входить и "
        "выходить частями (`docs/OVERLAYS.md`). Для каждой стратегии, прошедшей первый этап, "
        "пробуются все надстройки поодиночке и все пары; числа надстройки раз в месяц выбирает "
        "код только по прошлому. Сравнение с самой стратегией идёт на одних и тех же днях; "
        "поправка учитывает, что вариантов много. Токены не тратятся.", ""]
    lines += [f"- {n}" for n in block.get("notes") or []]
    ran = [e for e in entries if e.get("action") == "run"]
    late = [e for e in entries if e.get("action") == "stage2-late"]
    lines.append(f"Вариантов прогнано за ночь: {len(ran)}"
                 + (f", вторых этапов, долгов прошлых ночей: {len(late)}" if late else "")
                 + f", сбоев: {sum(e.get('action') == 'error' for e in entries)}.")
    if block.get("deferred"):
        lines.append("Не уместились в ночной предел и ждут следующей ночи: "
                     f"{', '.join(block['deferred'])}.")
    for idea, why in (block.get("refused") or {}).items():
        lines.append(f"- {idea}: не подходит как база — {why}")
    lines.append("")
    for base_entry in block.get("bases") or []:
        lines += entry_lines(base_entry, tables.get(base_entry["base"]))
        for e in entries:
            if e.get("base") != base_entry["base"]:
                continue
            if e.get("action") == "error":
                lines.append(f"    - `{e.get('variant')}`: **сбой** — {str(e.get('error'))[:200]};"
                             " повторится в следующую ночь")
            elif e.get("action") == "none":
                lines.append(f"    - {'+'.join(e.get('chain') or [])}: вариант не построен — "
                             f"{e.get('reason')}")
            elif e.get("comparison_error"):
                lines.append(f"    - `{e.get('variant')}`: прогон есть, сравнение не вышло — "
                             f"{str(e['comparison_error'])[:200]}")
            elif e.get("slow"):
                lines.append(f"    - `{e.get('variant')}`: **дольше нормы** "
                             f"({e.get('seconds')} с, см. `night/overlays.yaml`)")
            if e.get("stage2") is not None and e.get("action") != "error":
                from qlab.pipeline.stage2 import report_lines as stage2_lines

                lines += [f"    - `{e.get('variant')}`: {ln}" for ln in
                          stage2_lines({"route": e.get("route")}, e["stage2"])]
        lines.append("")
    for e in entries:
        if not e.get("variant") and e.get("action") == "error":
            lines.append(f"- база `{e['base']}`: **сбой** — {str(e.get('error'))[:200]}")
    return lines


__all__ = ["CONFIG", "Base", "Config", "SECTION_TITLE", "entry_lines", "ensure_overlay_idea",
           "find_bases", "fingerprint", "last_run_end", "load_config", "load_ledger",
           "overlays_stage", "report_lines", "save_ledger", "table_lines", "tables_for"]
