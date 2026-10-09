import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchIdeas } from '../api.js'
import { Absent, IdeaLink, Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import {
  ASSET_CLASS,
  IDEA_STATUS,
  PROFILE,
  SHUTDOWN_CAUSE,
  SOURCE_TYPE,
  TERMINAL_STATUSES,
  label,
} from '../labels.js'

// q-lab's own random books: measuring instruments, not ideas (they exist as
// rows so their runs have somewhere to hang).
const isNoise = (idea) => idea.id.startsWith('noise-')

/** Families: variants grouped under the strategy they vary (idea.parent_id,
 * assigned by code -- qlab.registry.families). Thirty bench ideas were three
 * strategies and their variants; the list shows the three. */
function buildFamilies(ideas) {
  const byId = new Map(ideas.map((i) => [i.id, i]))
  const children = new Map()
  const roots = []
  for (const idea of ideas) {
    if (idea.parent_id && byId.has(idea.parent_id)) {
      if (!children.has(idea.parent_id)) children.set(idea.parent_id, [])
      children.get(idea.parent_id).push(idea)
    } else {
      roots.push(idea)
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

function IdeaRow({ idea, variants = [], open = false, onToggle, nested = false }) {
  return (
    <tr className={`align-top ${nested ? 'bg-slate-50/60' : ''}`}>
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

function IdeaTable({ families }) {
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
          {families.flatMap(({ parent, variants }) => [
            <IdeaRow
              key={parent.id}
              idea={parent}
              variants={variants}
              open={open.has(parent.id)}
              onToggle={() => toggle(parent.id)}
            />,
            ...(open.has(parent.id)
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
      <Panel
        title="Реестр"
        subtitle={`Стратегий в работе: ${working.length}; их вариантов: ${variantCount}. Варианты одной стратегии собраны под ней.`}
      >
        <IdeaTable families={working} />
      </Panel>

      <Panel
        title="Кладбище"
        subtitle="Отвергнутые, выдохшиеся и выведенные. Хранятся полностью: отказ — такой же результат, как проход."
      >
        <IdeaTable families={buried} />
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
