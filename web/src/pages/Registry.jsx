import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchIdeas } from '../api.js'
import { Absent, IdeaLink, Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import {
  ASSET_CLASS,
  IDEA_STATUS,
  PROFILE,
  SHUTDOWN_CAUSE,
  SOURCE_TYPE,
  STATUS_STYLE,
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

function variantSummary(variants) {
  const counts = new Map()
  for (const v of variants) counts.set(v.status, (counts.get(v.status) ?? 0) + 1)
  return [...counts.entries()].map(([status, n]) => `${n} — ${label(IDEA_STATUS, status)}`).join(', ')
}

function IdeaRow({ idea, variants = [], open = false, onToggle, nested = false, context = false }) {
  return (
    <tr className={`align-top ${nested ? 'bg-slate-50/60' : ''} ${context ? 'opacity-50' : ''}`}>
      <td className={`py-2 pr-3 ${nested ? 'border-l-2 border-indigo-200 pl-5' : ''}`}>
        <IdeaLink id={idea.id}>{idea.title}</IdeaLink>
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
        <StatusBadge status={idea.status} />
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

// Owner, 2026-10-28: «в реестре было бы здорово увидеть фильтр по статусам». The chosen
// statuses live in the address (?status=bench,validated), so a filtered view can be reopened.
function StatusFilter({ ideas, chosen, onChange }) {
  const counts = new Map()
  for (const i of ideas) counts.set(i.status, (counts.get(i.status) ?? 0) + 1)
  const present = Object.keys(IDEA_STATUS).filter((s) => counts.has(s))
  const toggle = (s) => {
    const next = new Set(chosen)
    if (next.has(s)) next.delete(s)
    else next.add(s)
    onChange(next)
  }
  const chip = 'rounded-full px-2.5 py-1 text-xs ring-1 ring-inset focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400'
  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Фильтр по статусам">
      <button
        type="button"
        onClick={() => onChange(new Set())}
        aria-pressed={chosen.size === 0}
        className={`${chip} ${chosen.size === 0 ? 'bg-indigo-600 text-white ring-indigo-600' : 'bg-white text-slate-600 ring-slate-300 hover:bg-slate-50'}`}
      >
        все · {ideas.length}
      </button>
      {present.map((s) => {
        const on = chosen.has(s)
        return (
          <button
            key={s}
            type="button"
            onClick={() => toggle(s)}
            aria-pressed={on}
            className={`${chip} ${on ? `${STATUS_STYLE[s]} font-semibold ring-2` : 'bg-white text-slate-600 ring-slate-300 hover:bg-slate-50'}`}
          >
            {label(IDEA_STATUS, s)} · {counts.get(s)}
          </button>
        )
      })}
    </div>
  )
}

// With a filter on, a family shows only its matching variants, opened; a parent that does
// not match itself stays as a faded row, so its matching variants keep their context.
function filterFamilies(families, chosen) {
  if (chosen.size === 0) return families
  return families
    .map(({ parent, variants }) => ({
      parent,
      variants: variants.filter((v) => chosen.has(v.status)),
      context: !chosen.has(parent.status),
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
                нет идей с выбранными статусами
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
  const chosen = new Set((params.get('status') ?? '').split(',').filter(Boolean))
  const setChosen = (next) => {
    const value = Object.keys(IDEA_STATUS).filter((s) => next.has(s)).join(',')
    setParams(value ? { status: value } : {}, { replace: true })
  }
  const filtered = chosen.size > 0

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

  return (
    <div className="space-y-6">
      <StatusFilter ideas={ideas} chosen={chosen} onChange={setChosen} />

      <Panel
        title="Реестр"
        subtitle={`Стратегий в работе: ${working.length}; их вариантов: ${variantCount}. Варианты одной стратегии собраны под ней.`}
      >
        <IdeaTable families={filterFamilies(working, chosen)} filtered={filtered} />
      </Panel>

      <Panel
        title="Кладбище"
        subtitle="Отвергнутые, выдохшиеся и выведенные. Хранятся полностью: отказ — такой же результат, как проход."
      >
        <IdeaTable families={filterFamilies(buried, chosen)} filtered={filtered} />
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
