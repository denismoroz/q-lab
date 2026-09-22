import { VERDICT_STAGE, ROW_SOURCE, label } from '../labels.js'
import { Absent, DateText, Num } from './ui.jsx'

// docs/REGISTRY.md defines exactly three legitimate verdict rows, and the
// whole point of the registry is that they cannot be confused with one
// another. Rendering a measurement as though a rule had been applied would
// recreate the fabricated threshold the schema exists to prevent, so the
// three kinds get different colours, different badges, different wording
// for the middle column, and different wording for the outcome.
//
// `kind` is decided on the server from the stored columns, and `passed` is
// displayed exactly as stored — the frontend never recomputes a verdict,
// a threshold or an aggregate.

export const KIND_META = {
  decision: {
    title: 'решение',
    explanation: 'правило применено к посчитанной метрике',
    frame: 'border-l-4 border-l-sky-600 bg-white',
    badge: 'bg-sky-100 text-sky-900 ring-sky-400',
  },
  unknown: {
    title: 'неизвестно',
    explanation: 'правило есть, метрика не посчитана — это не проход и не провал',
    frame: 'border-l-4 border-l-slate-400 border-dashed bg-slate-50',
    badge: 'bg-slate-200 text-slate-700 ring-slate-400',
  },
  measurement: {
    title: 'измерение',
    explanation: 'число без критерия: порога не было, правило не применялось',
    frame: 'border-l-4 border-l-amber-500 bg-amber-50',
    badge: 'bg-amber-100 text-amber-900 ring-amber-500',
  },
}

export function VerdictKindBadge({ kind }) {
  const meta = KIND_META[kind] ?? KIND_META.unknown
  return (
    <span
      className={`inline-block whitespace-nowrap rounded px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ring-1 ring-inset ${meta.badge}`}
      title={meta.explanation}
    >
      {meta.title}
    </span>
  )
}

/** The legend. Three kinds that look alike would be worse than no screen at
 * all, so the vocabulary is stated on every screen that shows verdicts. */
export function VerdictLegend() {
  return (
    <div className="grid gap-2 md:grid-cols-3">
      {Object.entries(KIND_META).map(([kind, meta]) => (
        <div key={kind} className={`rounded border border-slate-200 p-3 ${meta.frame}`}>
          <VerdictKindBadge kind={kind} />
          <p className="mt-2 text-xs text-slate-600">{meta.explanation}</p>
        </div>
      ))}
    </div>
  )
}

/** The middle column: what was compared, if anything was. */
function Comparison({ verdict }) {
  const { kind, metric, value, comparator, threshold } = verdict

  if (kind === 'decision') {
    return (
      <div className="font-mono text-sm text-slate-900">
        <span className="text-slate-500">{metric}</span>{' '}
        <Num value={value} />{' '}
        <span className="font-semibold text-slate-700">{comparator}</span>{' '}
        <Num value={threshold} />
      </div>
    )
  }

  if (kind === 'unknown') {
    return (
      <div className="text-sm">
        <div className="font-mono text-slate-500">
          {metric} — <span className="italic">метрика не посчитана</span>
        </div>
        <div className="mt-0.5 font-mono text-xs text-slate-400">
          правило ждёт: {comparator} <Num value={threshold} />
        </div>
      </div>
    )
  }

  // measurement: a number and nothing else. There is deliberately no
  // comparator or threshold column here to fill.
  return (
    <div className="text-sm">
      <div className="font-mono text-slate-900">
        <span className="text-slate-500">{metric}</span> = <Num value={value} />
      </div>
      <div className="mt-0.5 text-xs font-medium text-amber-800">
        порога не было — критерий не формализован
      </div>
    </div>
  )
}

/** The outcome column. Only a decision can carry one, and it is read from
 * the stored `passed`, never derived here. */
function Outcome({ verdict }) {
  if (verdict.kind === 'decision') {
    return verdict.passed ? (
      <span className="whitespace-nowrap rounded bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-900 ring-1 ring-inset ring-emerald-400">
        прошло
      </span>
    ) : (
      <span className="whitespace-nowrap rounded bg-rose-100 px-2 py-0.5 text-xs font-semibold text-rose-900 ring-1 ring-inset ring-rose-400">
        провалено
      </span>
    )
  }
  if (verdict.kind === 'unknown') {
    return <span className="text-xs italic text-slate-500">решения нет</span>
  }
  return <span className="text-xs italic text-amber-800">правило не применялось</span>
}

export function VerdictRow({ verdict }) {
  const meta = KIND_META[verdict.kind] ?? KIND_META.unknown
  return (
    <li className={`rounded border border-slate-200 p-3 ${meta.frame}`}>
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
        <div className="flex w-28 shrink-0 flex-col gap-1">
          <VerdictKindBadge kind={verdict.kind} />
          <span className="text-xs text-slate-500">{label(VERDICT_STAGE, verdict.stage)}</span>
        </div>

        <div className="min-w-[16rem] flex-1">
          <Comparison verdict={verdict} />
          <div className="mt-1 text-xs text-slate-500">
            правило <span className="font-mono">{verdict.rule_id}</span> · версия{' '}
            <span className="font-mono">{verdict.rules_version}</span> ·{' '}
            {label(ROW_SOURCE, verdict.source)}
          </div>
        </div>

        <div className="w-32 shrink-0">
          <Outcome verdict={verdict} />
        </div>

        <div className="w-44 shrink-0 text-xs text-slate-500">
          <div>
            вынесен <DateText value={verdict.decided_at} />
          </div>
          <div>
            данные:{' '}
            {verdict.data_range_start || verdict.data_range_end ? (
              <>
                <DateText value={verdict.data_range_start} />…
                <DateText value={verdict.data_range_end} />
              </>
            ) : (
              <Absent>диапазон не записан</Absent>
            )}
          </div>
        </div>
      </div>

      {verdict.note && (
        <p className="mt-2 break-words border-t border-slate-200 pt-2 text-xs text-slate-500">
          {verdict.note}
        </p>
      )}
    </li>
  )
}

export function VerdictList({ items }) {
  if (!items.length) {
    return <p className="text-sm text-slate-500">Вердиктов нет.</p>
  }
  return (
    <ul className="space-y-2">
      {items.map((verdict) => (
        <VerdictRow key={verdict.id} verdict={verdict} />
      ))}
    </ul>
  )
}
