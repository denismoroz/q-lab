import { useQuery } from '@tanstack/react-query'
import { fetchIdeas } from '../api.js'
import { Absent, IdeaLink, Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import {
  ASSET_CLASS,
  PROFILE,
  SHUTDOWN_CAUSE,
  SOURCE_TYPE,
  TERMINAL_STATUSES,
  label,
} from '../labels.js'

function IdeaTable({ ideas }) {
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
          {ideas.map((idea) => (
            <tr key={idea.id} className="align-top">
              <td className="py-2 pr-3">
                <IdeaLink id={idea.id}>{idea.title}</IdeaLink>
                <div className="font-mono text-xs text-slate-400">{idea.id}</div>
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
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Registry() {
  const query = useQuery({ queryKey: ['ideas'], queryFn: fetchIdeas })

  return (
    <QueryState query={query}>
      {query.data && (
        <div className="space-y-6">
          <Panel
            title="Реестр"
            subtitle={`Идеи в работе: ${
              query.data.items.filter((i) => !TERMINAL_STATUSES.has(i.status)).length
            }`}
          >
            <IdeaTable ideas={query.data.items.filter((i) => !TERMINAL_STATUSES.has(i.status))} />
          </Panel>

          <Panel
            title="Кладбище"
            subtitle="Отвергнутые, выдохшиеся и выведенные. Хранятся полностью: отказ — такой же результат, как проход."
          >
            <IdeaTable ideas={query.data.items.filter((i) => TERMINAL_STATUSES.has(i.status))} />
          </Panel>
        </div>
      )}
    </QueryState>
  )
}
