import { Fragment, useState } from 'react'
import { Absent, DateText, IdeaLink, Json, Num } from './ui.jsx'
import { ROW_SOURCE, TRIAL_STATUS, label } from '../labels.js'

// The metrics shown as columns. Anything else a trial recorded is still
// there — the row expands to the stored JSON — but these four are what the
// ledger is read for. A metric a trial did not record shows as absent; the
// UI never computes or substitutes one.
const METRIC_COLUMNS = [
  ['sharpe_net', 'Sharpe net'],
  ['ann_return_net', 'Доходность net'],
  ['max_dd', 'Max DD'],
  ['turnover', 'Оборот'],
]

function MetricCell({ metrics, key_ }) {
  if (!metrics || metrics[key_] == null) return <Absent>нет</Absent>
  return <Num value={metrics[key_]} digits={key_ === 'turnover' ? 2 : 4} />
}

export default function TrialTable({ trials, showIdea = true }) {
  const [expanded, setExpanded] = useState(null)

  if (!trials.length) return <p className="text-sm text-slate-500">Прогонов нет.</p>

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2 pr-3 font-medium">#</th>
            <th className="py-2 pr-3 font-medium">Начат</th>
            {showIdea && <th className="py-2 pr-3 font-medium">Идея</th>}
            <th className="py-2 pr-3 font-medium">Маршрут</th>
            {METRIC_COLUMNS.map(([key_, title]) => (
              <th key={key_} className="py-2 pr-3 text-right font-medium">
                {title}
              </th>
            ))}
            <th className="py-2 pr-3 font-medium">Статус</th>
            <th className="py-2 font-medium" />
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {trials.map((trial) => (
            <Fragment key={trial.id}>
              <tr
                className={`align-top ${trial.kept ? '' : 'bg-slate-50 text-slate-500'}`}
              >
                <td className="py-2 pr-3 tabular-nums text-slate-400">{trial.id}</td>
                <td className="py-2 pr-3">
                  <DateText value={trial.started_at} withTime />
                </td>
                {showIdea && (
                  <td className="py-2 pr-3">
                    {trial.idea_id ? (
                      <IdeaLink id={trial.idea_id}>{trial.idea_title ?? trial.idea_id}</IdeaLink>
                    ) : (
                      <Absent>идея не найдена</Absent>
                    )}
                  </td>
                )}
                <td className="py-2 pr-3 text-xs">
                  <div className="font-mono text-slate-600">{trial.code_ref ?? '—'}</div>
                  <div className="text-slate-400">
                    spec {trial.spec_id}
                    {trial.spec_version != null && ` v${trial.spec_version}`} · код{' '}
                    {trial.code_sha?.slice(0, 7)}
                  </div>
                  <div className="text-slate-400">
                    {trial.snapshot_id ? (
                      <>снимок {trial.snapshot_id.slice(0, 8)}</>
                    ) : (
                      <Absent>без снимка (импорт)</Absent>
                    )}
                  </div>
                </td>
                {METRIC_COLUMNS.map(([key_]) => (
                  <td key={key_} className="py-2 pr-3 text-right">
                    <MetricCell metrics={trial.metrics} key_={key_} />
                  </td>
                ))}
                <td className="py-2 pr-3 text-xs">
                  <div
                    className={
                      trial.status === 'ok' ? 'text-slate-700' : 'font-medium text-rose-700'
                    }
                  >
                    {label(TRIAL_STATUS, trial.status)}
                  </div>
                  <div className="text-slate-400">{label(ROW_SOURCE, trial.source)}</div>
                  {/* A discarded run stays in the ledger: deflation counts
                      every trial, not just the reported ones. */}
                  {!trial.kept && <div className="text-slate-500">выброшен</div>}
                </td>
                <td className="py-2">
                  <button
                    type="button"
                    className="rounded border border-slate-300 px-2 py-0.5 text-xs"
                    onClick={() => setExpanded(expanded === trial.id ? null : trial.id)}
                  >
                    {expanded === trial.id ? 'скрыть' : 'детали'}
                  </button>
                </td>
              </tr>
              {expanded === trial.id && (
                <tr>
                  <td colSpan={showIdea ? 10 : 9} className="pb-4 pr-3">
                    <div className="grid gap-3 md:grid-cols-2">
                      <div>
                        <div className="mb-1 text-xs font-medium text-slate-500">Параметры</div>
                        <Json value={trial.params} />
                      </div>
                      <div>
                        <div className="mb-1 text-xs font-medium text-slate-500">
                          Метрики{' '}
                          {trial.metrics == null && (
                            <Absent>метрик нет — прогон не дал их</Absent>
                          )}
                        </div>
                        {trial.metrics != null && <Json value={trial.metrics} />}
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}
