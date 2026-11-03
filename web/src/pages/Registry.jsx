import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchIdeas } from '../api.js'
import { Absent, IdeaLink, Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import {
  ASSET_CLASS,
  FAMILY,
  IDEA_STATUS,
  PROFILE,
  SHUTDOWN_CAUSE,
  SOURCE_TYPE,
  STATUS_KEYS,
  STATUS_STYLE,
  BENCH_STYLE,
  TERMINAL_STATUSES,
  label,
} from '../labels.js'

// q-lab's own random books: measuring instruments, not ideas (they exist as
// rows so their runs have somewhere to hang).
const isNoise = (idea) => idea.id.startsWith('noise-')

/** Families: variants grouped under the strategy they vary (idea.parent_id,
 * assigned by code -- qlab.registry.families). Thirty bench ideas were three
 * strategies and their variants; the list shows the three. A variant of a variant (a causal
 * twin of an experiment, its second stage) belongs to the same top strategy: until
 * 2026-10-28 such ideas were grouped under a variant that is never opened, and so were not
 * shown at all. */
function buildFamilies(ideas) {
  const byId = new Map(ideas.map((i) => [i.id, i]))
  const rootOf = (idea) => {
    const seen = new Set()
    let current = idea
    while (current.parent_id && byId.has(current.parent_id) && !seen.has(current.id)) {
      seen.add(current.id)
      current = byId.get(current.parent_id)
    }
    return current
  }
  const children = new Map()
  const roots = []
  for (const idea of ideas) {
    const root = rootOf(idea)
    if (root.id === idea.id) {
      roots.push(idea)
    } else {
      if (!children.has(root.id)) children.set(root.id, [])
      children.get(root.id).push(idea)
    }
  }
  return roots.map((parent) => ({ parent, variants: children.get(parent.id) ?? [] }))
}

function plural(n, one, few, many) {
  const d = n % 10
  const h = n % 100
  if (d === 1 && h !== 11) return one
  if (d >= 2 && d <= 4 && (h < 12 || h > 14)) return few
  return many
}

// The words of an idea's standing: for the bench, the API's split of it (qlab.bench_label).
const statusWords = (idea) => idea.bench?.label_ru ?? label(IDEA_STATUS, idea.status)
const statusKey = (idea) => idea.status_key ?? idea.status

function variantSummary(variants) {
  const counts = new Map()
  for (const v of variants) {
    const words = statusWords(v)
    counts.set(words, (counts.get(words) ?? 0) + 1)
  }
  return [...counts.entries()].map(([words, n]) => `${n} — ${words}`).join(', ')
}

// A status that changed in the last 24 hours (owner, 2026-11-03: «сменилось недавно»): the mark,
// and old -> new on hover. The new bench words are the idea's current ones.
function ChangedMark({ idea }) {
  const change = idea.changed_24h
  if (!change) return null
  const old = change.from ? label(IDEA_STATUS, change.from) : 'новая идея'
  const now = change.to === idea.status ? statusWords(idea) : label(IDEA_STATUS, change.to)
  return (
    <span
      title={`было: ${old} → стало: ${now}`}
      className="ml-2 inline-block rounded bg-fuchsia-50 px-1.5 py-0.5 align-middle text-[10px] font-medium text-fuchsia-800 ring-1 ring-inset ring-fuchsia-300"
    >
      сменилось
    </span>
  )
}

function IdeaRow({ idea, variants = [], open = false, onToggle, nested = false, context = false }) {
  return (
    <tr className={`align-top ${nested ? 'bg-slate-50/60' : ''} ${context ? 'opacity-50' : ''}`}>
      <td className={`py-2 pr-3 ${nested ? 'border-l-2 border-indigo-200 pl-5' : ''}`}>
        <span title={idea.title}>
          <IdeaLink id={idea.id}>{idea.plain_title || idea.title}</IdeaLink>
        </span>
        <ChangedMark idea={idea} />
        <div className="font-mono text-xs text-slate-400">{idea.id}</div>
        {variants.length > 0 && (
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={open}
            className="mt-1 rounded text-left text-xs text-indigo-700 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
          >
            {open ? '▾' : '▸'} {variants.length}{' '}
            {plural(variants.length, 'вариант', 'варианта', 'вариантов')}: {variantSummary(variants)}
          </button>
        )}
      </td>
      <td className="py-2 pr-3">
        <StatusBadge status={idea.status} bench={idea.bench} />
        {idea.shutdown_cause && (
          <div className="mt-1 text-xs text-stone-600">
            {label(SHUTDOWN_CAUSE, idea.shutdown_cause)}
          </div>
        )}
      </td>
      <td className="py-2 pr-3">
        {idea.driver_id ? (
          <>
            <div className="text-slate-800">{idea.driver_title}</div>
            <div className="font-mono text-xs text-slate-400">{idea.driver_id}</div>
          </>
        ) : (
          <Absent>драйвер не назначен</Absent>
        )}
      </td>
      <td className="py-2 pr-3">
        <div>{label(PROFILE, idea.profile)}</div>
        <div className="text-xs text-slate-400">{label(ASSET_CLASS, idea.asset_class)}</div>
      </td>
      <td className="py-2 pr-3">
        <div>{label(SOURCE_TYPE, idea.source_type)}</div>
        {idea.source_url && (
          <a
            className="break-all text-xs text-sky-700 underline-offset-2 hover:underline"
            href={idea.source_url}
            target="_blank"
            rel="noreferrer"
          >
            {idea.source_url}
          </a>
        )}
      </td>
    </tr>
  )
}

const chip =
  'rounded-full px-2.5 py-1 text-xs ring-1 ring-inset focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400'
const chipOff = 'bg-white text-slate-600 ring-slate-300 hover:bg-slate-50'
const chipAll = 'bg-indigo-600 text-white ring-indigo-600'

function ChipRow({ title, children }) {
  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label={title}>
      <span className="w-20 shrink-0 text-xs uppercase tracking-wide text-slate-400">{title}</span>
      {children}
    </div>
  )
}

function toggled(set, key) {
  const next = new Set(set)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  return next
}

// Owner, 2026-10-28: «в реестре было бы здорово увидеть фильтр по статусам». The chosen
// statuses live in the address (?status=bench-too-early,validated), so a filtered view can be
// reopened. «Скамейка» is three labels and an unnamed rest (2026-11-03: «разведи на три надписи»);
// the bare ?status=bench of older links means all four.
function StatusFilter({ ideas, everyone, chosen, onChange }) {
  const counts = new Map()
  for (const i of ideas) counts.set(statusKey(i), (counts.get(statusKey(i)) ?? 0) + 1)
  // the words come from every idea, so a chosen status that no idea matches any more stays a button
  const words = new Map(everyone.map((i) => [statusKey(i), statusWords(i)]))
  const present = STATUS_KEYS.filter((s) => counts.has(s) || chosen.has(s))
  return (
    <ChipRow title="Статус">
      <button
        type="button"
        onClick={() => onChange(new Set())}
        aria-pressed={chosen.size === 0}
        className={`${chip} ${chosen.size === 0 ? chipAll : chipOff}`}
      >
        все · {ideas.length}
      </button>
      {present.map((s) => {
        const on = chosen.has(s)
        const style = BENCH_STYLE[s] ?? STATUS_STYLE[s]
        return (
          <button
            key={s}
            type="button"
            onClick={() => onChange(toggled(chosen, s))}
            aria-pressed={on}
            className={`${chip} ${on ? `${style} font-semibold ring-2` : chipOff}`}
          >
            {words.get(s) ?? s} · {counts.get(s) ?? 0}
          </button>
        )
      })}
    </ChipRow>
  )
}

// Owner, 2026-11-03: «почему я не видел этих стратегий в фильтре?» -> «добавь фильтр по
// семейству». A family is the root of an idea's parent chain (qlab.registry.strategies); the
// choice lives in the address (?family=bv2,trend) and combines with the status filter.
function FamilyFilter({ ideas, chosen, onChange }) {
  const counts = new Map()
  for (const i of ideas) counts.set(i.family, (counts.get(i.family) ?? 0) + 1)
  const present = Object.keys(FAMILY).filter((f) => counts.has(f) || chosen.has(f))
  return (
    <ChipRow title="Семейство">
      <button
        type="button"
        onClick={() => onChange(new Set())}
        aria-pressed={chosen.size === 0}
        className={`${chip} ${chosen.size === 0 ? chipAll : chipOff}`}
      >
        все · {ideas.length}
      </button>
      {present.map((f) => {
        const on = chosen.has(f)
        return (
          <button
            key={f}
            type="button"
            onClick={() => onChange(toggled(chosen, f))}
            aria-pressed={on}
            className={`${chip} ${on ? 'bg-indigo-50 font-semibold text-indigo-900 ring-2 ring-indigo-400' : chipOff}`}
          >
            {label(FAMILY, f)} · {counts.get(f) ?? 0}
          </button>
        )
      })}
    </ChipRow>
  )
}

function ChangedFilter({ ideas, on, onChange }) {
  const n = ideas.filter((i) => i.changed_24h).length
  return (
    <ChipRow title="Сменилось">
      <button
        type="button"
        onClick={() => onChange(!on)}
        aria-pressed={on}
        className={`${chip} ${on ? 'bg-fuchsia-50 font-semibold text-fuchsia-900 ring-2 ring-fuchsia-400' : chipOff}`}
      >
        изменились за сутки · {n}
      </button>
    </ChipRow>
  )
}

/** The filters as the address holds them: `?status=`, `?family=`, `?changed=1`. */
function readFilters(params) {
  const list = (name) => (params.get(name) ?? '').split(',').filter(Boolean)
  const status = new Set()
  for (const s of list('status')) {
    if (s === 'bench') STATUS_KEYS.filter((k) => k.startsWith('bench-')).forEach((k) => status.add(k))
    else status.add(s)
  }
  return { status, family: new Set(list('family')), changed: params.get('changed') === '1' }
}

function writeFilters({ status, family, changed }) {
  const out = {}
  const statuses = STATUS_KEYS.filter((s) => status.has(s)).join(',')
  const families = Object.keys(FAMILY).filter((f) => family.has(f)).join(',')
  if (statuses) out.status = statuses
  if (families) out.family = families
  if (changed) out.changed = '1'
  return out
}

const matcher = ({ status, family, changed }) => (idea) =>
  (status.size === 0 || status.has(statusKey(idea))) &&
  (family.size === 0 || family.has(idea.family)) &&
  (!changed || Boolean(idea.changed_24h))

// With a filter on, a family shows only its matching variants, opened; a parent that does
// not match itself stays as a faded row, so its matching variants keep their context.
function filterFamilies(families, matches, active) {
  if (!active) return families
  return families
    .map(({ parent, variants }) => ({
      parent,
      variants: variants.filter(matches),
      context: !matches(parent),
    }))
    .filter(({ variants, context }) => !context || variants.length > 0)
}

function IdeaTable({ families, filtered = false }) {
  const [open, setOpen] = useState(() => new Set())
  const toggle = (id) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2 pr-3 font-medium">Идея</th>
            <th className="py-2 pr-3 font-medium">Статус</th>
            <th className="py-2 pr-3 font-medium">Драйвер</th>
            <th className="py-2 pr-3 font-medium">Профиль</th>
            <th className="py-2 pr-3 font-medium">Источник</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {families.length === 0 && (
            <tr>
              <td colSpan={5} className="py-3 text-sm text-slate-500">
                нет идей с выбранными условиями
              </td>
            </tr>
          )}
          {families.flatMap(({ parent, variants, context = false }) => [
            <IdeaRow
              key={parent.id}
              idea={parent}
              variants={variants}
              context={context}
              open={filtered || open.has(parent.id)}
              onToggle={() => toggle(parent.id)}
            />,
            ...(filtered || open.has(parent.id)
              ? variants.map((v) => <IdeaRow key={v.id} idea={v} nested />)
              : []),
          ])}
        </tbody>
      </table>
    </div>
  )
}

export default function Registry() {
  const query = useQuery({ queryKey: ['ideas'], queryFn: fetchIdeas })
  const [showNoise, setShowNoise] = useState(false)
  const [params, setParams] = useSearchParams()
  const filters = readFilters(params)
  const setFilters = (patch) =>
    setParams(writeFilters({ ...filters, ...patch }), { replace: true })
  const filtered = filters.status.size > 0 || filters.family.size > 0 || filters.changed
  const matches = matcher(filters)

  if (!query.data) return <QueryState query={query} />

  const ideas = query.data.items.filter((i) => !isNoise(i))
  const noise = query.data.items.filter(isNoise)
  const families = buildFamilies(ideas)
  // A family stays in the registry while any member is still in work.
  const alive = ({ parent, variants }) =>
    [parent, ...variants].some((i) => !TERMINAL_STATUSES.has(i.status))
  const working = families.filter(alive)
  const buried = families.filter((f) => !alive(f))
  const variantCount = working.reduce((n, f) => n + f.variants.length, 0)
  // Each row of chips counts what the OTHER filters leave, so a number is what a click would show.
  const without = (drop) => ideas.filter(matcher({ ...filters, ...drop }))

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <StatusFilter
          ideas={without({ status: new Set() })}
          everyone={ideas}
          chosen={filters.status}
          onChange={(status) => setFilters({ status })}
        />
        <FamilyFilter
          ideas={without({ family: new Set() })}
          chosen={filters.family}
          onChange={(family) => setFilters({ family })}
        />
        <ChangedFilter
          ideas={without({ changed: false })}
          on={filters.changed}
          onChange={(changed) => setFilters({ changed })}
        />
      </div>

      <Panel
        title="Реестр"
        subtitle={`Стратегий в работе: ${working.length}; их вариантов: ${variantCount}. Варианты одной стратегии собраны под ней.`}
      >
        <IdeaTable families={filterFamilies(working, matches, filtered)} filtered={filtered} />
      </Panel>

      <Panel
        title="Кладбище"
        subtitle="Отвергнутые, выдохшиеся и выведенные. Хранятся полностью: отказ — такой же результат, как проход."
      >
        <IdeaTable families={filterFamilies(buried, matches, filtered)} filtered={filtered} />
      </Panel>

      {noise.length > 0 && (
        <Panel
          title="Случайные книги"
          subtitle="Измерительный прибор q-lab, а не идеи: с ними сравнивается каждая стратегия."
        >
          <button
            type="button"
            onClick={() => setShowNoise((v) => !v)}
            aria-expanded={showNoise}
            className="rounded text-xs text-indigo-700 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
          >
            {showNoise ? '▾ скрыть' : `▸ показать ${noise.length}`}
          </button>
          {showNoise && (
            <div className="mt-3">
              <IdeaTable families={noise.map((parent) => ({ parent, variants: [] }))} />
            </div>
          )}
        </Panel>
      )}
    </div>
  )
}
