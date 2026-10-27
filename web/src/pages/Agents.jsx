import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { fetchAgents } from '../api.js'
import { Panel, QueryState } from '../components/ui.jsx'

// What each agent did and what came of it, night by night (qlab.agent_reports).
// Every row is copied from the night's saved state or the agent's own ledger:
// nothing is judged here.

const STATUS = {
  ran: { text: 'работал', style: 'bg-emerald-50 text-emerald-900 ring-emerald-300' },
  stopped: { text: 'остановлен бюджетом или сроком', style: 'bg-amber-50 text-amber-900 ring-amber-400' },
  skipped: { text: 'не запускался', style: 'bg-slate-100 text-slate-600 ring-slate-300' },
  failed: { text: 'сбой', style: 'bg-rose-50 text-rose-900 ring-rose-300' },
}

const tokens = (n) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(2)} млн` : n >= 1e3 ? `${Math.round(n / 1e3)} тыс.` : `${n}`

function Badge({ status }) {
  const s = STATUS[status] ?? STATUS.failed
  return (
    <span className={`inline-block whitespace-nowrap rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${s.style}`}>
      {s.text}
    </span>
  )
}

function RunRow({ run }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="py-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="w-24 shrink-0 tabular-nums text-xs text-slate-500">{run.day}</span>
        <Badge status={run.status} />
        <span className="tabular-nums text-xs text-slate-500">{tokens(run.tokens)} токенов</span>
      </div>
      <p className="mt-1 text-sm text-slate-700">
        <span className="text-slate-500">Что делал: </span>
        {run.did}
      </p>
      <p className="text-sm font-medium text-slate-900">
        <span className="font-normal text-slate-500">Результат: </span>
        {run.result}
      </p>
      {run.details.length > 0 && (
        <>
          <button
            type="button"
            onClick={() => setOpen(!open)}
            className="mt-1 text-xs text-sky-700 hover:underline"
          >
            {open ? 'скрыть подробности' : `подробности (${run.details.length})`}
          </button>
          {open && (
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs text-slate-600">
              {run.details.map((d, i) => (
                <li key={i} className="break-words">{d}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </li>
  )
}

function AgentCard({ agent }) {
  const [all, setAll] = useState(false)
  const runs = all ? agent.runs : agent.runs.slice(0, 3)
  const last = agent.runs[0]
  return (
    <Panel
      title={`${agent.name} · ${agent.model}`}
      subtitle={agent.role}
    >
      <div className="mb-2 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500">
        <span>запусков: <b className="text-slate-800">{agent.runs.length}</b></span>
        <span>токенов за период: <b className="text-slate-800">{tokens(agent.tokens)}</b></span>
        <span>последний: <b className="text-slate-800">{last ? last.day : 'не запускался'}</b></span>
      </div>
      {agent.runs.length === 0 ? (
        <p className="text-sm text-slate-500">За этот период агент не запускался.</p>
      ) : (
        <ul className="divide-y divide-slate-100">
          {runs.map((run, i) => (
            <RunRow key={`${run.day}-${i}`} run={run} />
          ))}
        </ul>
      )}
      {agent.runs.length > 3 && (
        <button
          type="button"
          onClick={() => setAll(!all)}
          className="mt-2 text-xs text-sky-700 hover:underline"
        >
          {all ? 'только последние три' : `все запуски (${agent.runs.length})`}
        </button>
      )}
    </Panel>
  )
}

export default function Agents() {
  const [days, setDays] = useState(14)
  const query = useQuery({ queryKey: ['agents', days], queryFn: () => fetchAgents(days) })
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold">Агенты: что сделано и каков результат</h1>
          <p className="text-xs text-slate-500">
            Из сохранённого состояния каждой ночи и журналов агентов; вердикты выносит стенд, а не агент.
          </p>
        </div>
        <label className="text-sm text-slate-600">
          за последние{' '}
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            {[7, 14, 30, 60].map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>{' '}
          ночей
        </label>
      </div>
      <QueryState query={query}>
        <div className="grid gap-4 lg:grid-cols-2">
          {query.data?.agents.map((agent) => (
            <AgentCard key={agent.key} agent={agent} />
          ))}
        </div>
      </QueryState>
    </div>
  )
}
