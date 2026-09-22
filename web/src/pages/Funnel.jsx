import { useQuery } from '@tanstack/react-query'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchFunnel } from '../api.js'
import { Panel, QueryState } from '../components/ui.jsx'
import { VerdictLegend } from '../components/Verdict.jsx'
import {
  IDEA_STATUS,
  SHUTDOWN_CAUSE,
  VERDICT_OUTCOME,
  VERDICT_STAGE,
  label,
} from '../labels.js'

// The counts come from `queries.funnel_stats` exactly as it computes them.
// No aggregation is added here — the lifecycle ORDER below is presentation
// (it is the order docs/REGISTRY.md lists the statuses in), not arithmetic.
const LIFECYCLE = [
  'candidate',
  'speccing',
  'implemented',
  'validated',
  'bench',
  'paper',
  'live',
  'rejected',
  'decayed',
  'retired',
]

const STATUS_FILL = {
  candidate: '#94a3b8',
  speccing: '#94a3b8',
  implemented: '#38bdf8',
  validated: '#0ea5e9',
  bench: '#6366f1',
  paper: '#10b981',
  live: '#047857',
  rejected: '#a8a29e',
  decayed: '#a8a29e',
  retired: '#a8a29e',
}

const OUTCOME_ORDER = ['passed', 'failed', 'unknown', 'measurement']
const OUTCOME_STYLE = {
  passed: 'bg-emerald-50 text-emerald-900 ring-emerald-300',
  failed: 'bg-rose-50 text-rose-900 ring-rose-300',
  unknown: 'bg-slate-100 text-slate-700 ring-slate-300',
  measurement: 'bg-amber-50 text-amber-900 ring-amber-400',
}

function CountList({ entries, empty = 'пусто' }) {
  if (!entries.length) return <p className="text-sm text-slate-500">{empty}</p>
  return (
    <dl className="divide-y divide-slate-100">
      {entries.map(([key, text, count]) => (
        <div key={key} className="flex items-baseline justify-between gap-4 py-1.5">
          <dt className="text-sm text-slate-700">{text}</dt>
          <dd className="tabular-nums text-sm font-medium text-slate-900">{count}</dd>
        </div>
      ))}
    </dl>
  )
}

export default function Funnel() {
  const query = useQuery({ queryKey: ['funnel'], queryFn: fetchFunnel })

  return (
    <QueryState query={query}>
      {query.data && <FunnelBody stats={query.data} />}
    </QueryState>
  )
}

function FunnelBody({ stats }) {
  const chartData = LIFECYCLE.filter((status) => stats.ideas_by_status[status] != null).map(
    (status) => ({
      status,
      name: label(IDEA_STATUS, status),
      count: stats.ideas_by_status[status],
    }),
  )

  const statusEntries = chartData.map((row) => [row.status, row.name, row.count])
  const stageEntries = Object.entries(stats.verdicts_by_stage)
    .sort((a, b) => b[1] - a[1])
    .map(([stage, count]) => [stage, label(VERDICT_STAGE, stage), count])
  const causeEntries = Object.entries(stats.decayed_by_shutdown_cause).map(([cause, count]) => [
    cause,
    label(SHUTDOWN_CAUSE, cause),
    count,
  ])

  return (
    <div className="space-y-6">
      <Panel
        title="Идеи по статусам"
        subtitle="Текущий срез реестра. Порядок — жизненный цикл из docs/REGISTRY.md."
      >
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="name"
                angle={-30}
                textAnchor="end"
                interval={0}
                height={60}
                tick={{ fontSize: 12 }}
              />
              <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value) => [value, 'идей']} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {chartData.map((row) => (
                  <Cell key={row.status} fill={STATUS_FILL[row.status] ?? '#94a3b8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-3">
        <Panel title="Идеи по статусам — счётчики">
          <CountList entries={statusEntries} />
        </Panel>

        <Panel title="Вердикты по стадиям">
          <CountList entries={stageEntries} />
        </Panel>

        <Panel title="Выдохшиеся — по причине остановки">
          <CountList entries={causeEntries} empty="выдохшихся нет" />
        </Panel>
      </div>

      <Panel
        title="Вердикты по исходу"
        subtitle="Четыре числа, а не два: «неизвестно» и «измерение» — не провал и не проход."
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {OUTCOME_ORDER.map((outcome) => (
            <div
              key={outcome}
              className={`rounded border p-3 ring-1 ring-inset ${OUTCOME_STYLE[outcome]}`}
            >
              <div className="text-xs font-medium uppercase tracking-wide">
                {label(VERDICT_OUTCOME, outcome)}
              </div>
              <div className="mt-1 tabular-nums text-2xl font-semibold">
                {stats.verdicts_by_outcome[outcome] ?? 0}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4">
          <VerdictLegend />
        </div>
      </Panel>
    </div>
  )
}
