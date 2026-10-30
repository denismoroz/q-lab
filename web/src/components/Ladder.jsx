import { Link } from 'react-router-dom'
import YearView from './YearView.jsx'
import Protection from './Protection.jsx'
import ForwardCharts from './ForwardCharts.jsx'
import { Panel } from './ui.jsx'
import { basisText, standLine } from '../headline.js'

// The ladder of an idea (docs/LADDER.md; owner, 2026-10-29: «поидее все стратегии должны быть
// оценены по одному принципу»): the same seven steps for every idea. Every word here is copied
// from the API's `ladder` field (qlab.ladder reads the registry and the night's ledgers); the page
// computes nothing. A step not run says why, from a closed list, never silence.
//
// Venue split (docs/VENUE_SPLIT.md; owner, 2026-10-30: «да, делай раздельно Binance и HL»): Binance
// and Hyperliquid are two venues, each with its own verdict, its own check of the strategy's money
// and its own fact about execution. They are shown side by side before the seven steps; passing on
// one venue and not on the other is a result and is named.

const OUTCOME_STYLE = {
  passed: 'border-emerald-300 bg-emerald-50 text-emerald-900',
  failed: 'border-stone-400 bg-stone-100 text-stone-800',
  undecided: 'border-sky-300 bg-sky-50 text-sky-900',
  not_run: 'border-slate-200 bg-slate-50 text-slate-600',
}

const pct = (value) => (typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : null)

function NumberRows({ rows }) {
  if (rows.length === 0) return null
  return (
    <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-sm">
      {rows.map(([title, value]) => (
        <div key={title} className="contents">
          <dt className="text-slate-500">{title}</dt>
          <dd className="tabular-nums text-slate-900">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

// The numbers a stage card used to show, on the strategy's own money (trial.headline).
function RunNumbers({ trial }) {
  if (!trial) return null
  const m = trial.metrics ?? {}
  const h = trial.headline
  const lead = h?.sharpe ?? m.sharpe_net
  const rows = [
    ['доходность в год после комиссий', pct(h?.ann_return ?? m.ann_return_net)],
    ['Шарп после комиссий', typeof lead === 'number' ? lead.toFixed(2) : null],
    ['худшая просадка', pct(h?.max_dd ?? m.max_dd)],
    ['обгоняет шум той же формы', pct(h?.noise_percentile ?? m.noise_return_percentile)],
    ['проверка вперёд, дней', typeof m.forward_days === 'number' ? Math.round(m.forward_days) : null],
  ].filter(([, v]) => v !== null)
  return (
    <>
      <NumberRows rows={rows} />
      {h && (
        <p className="mt-1 text-xs text-slate-500" data-testid="ladder-basis">
          {basisText(h)}.
          {standLine(h) && <span data-testid="ladder-stand"> {standLine(h)}.</span>}
        </p>
      )}
      <YearView view={trial.year_view} compact />
      <Protection view={trial.protection} compact />
    </>
  )
}

function RunLine({ run, ideaId }) {
  if (!run) return null
  return (
    <p className="mt-2 break-words text-xs text-slate-500" data-testid="ladder-run">
      данные: {run.venue ?? '—'}
      {run.data_start && run.data_end ? `, ${run.data_start} — ${run.data_end}` : ''} · прогон #
      {run.trial_id} · правила {run.rules_version ?? '—'} · стенд {run.stand_version ?? '—'}
      {run.idea_id && run.idea_id !== ideaId && (
        <>
          {' · '}
          <Link className="text-sky-700 underline-offset-2 hover:underline" to={`/ideas/${run.idea_id}`}>
            {run.idea_id}
          </Link>
        </>
      )}
    </p>
  )
}

function VenueCard({ view }) {
  const money = view.money
  return (
    <div
      className="rounded border border-slate-200 p-3"
      data-testid={`venue-${view.venue}`}
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-sm font-semibold text-slate-900">{view.title}</h3>
        <span
          className={`rounded border px-2 py-0.5 text-xs font-medium ${OUTCOME_STYLE[view.outcome] ?? ''}`}
          data-testid={`venue-${view.venue}-outcome`}
        >
          {view.outcome_ru}
        </span>
        {view.outcome === 'not_run' && view.reason_ru && (
          <span className="text-sm italic text-slate-600" data-testid={`venue-${view.venue}-reason`}>
            — {view.reason_ru}
          </span>
        )}
      </div>
      <p className="mt-1 break-words text-sm text-slate-700">{view.detail}</p>
      <p
        className={`mt-2 text-xs ${view.adapter === false ? 'text-amber-800' : 'text-slate-600'}`}
        data-testid={`venue-${view.venue}-execution`}
      >
        исполнение: {view.execution_ru}
      </p>
      {money && (
        <p className="mt-1 break-words text-xs text-slate-700" data-testid={`venue-${view.venue}-money`}>
          деньги стратегии: <span className="font-medium">{money.outcome_ru}</span>
          {money.reason_ru && <span className="italic"> ({money.reason_ru})</span>}
          {money.detail ? ` — ${money.detail}` : ''}
        </p>
      )}
    </div>
  )
}

// Why the two venues differ (docs/VENUE_SPLIT.md; owner, 2026-10-30: «если проходит одна но не
// проходит другая нужно чтобы была причина почему не проходит»). Written by qlab.venue_gap from
// measurements only; a cause it could not measure is listed as «причина не измерена» with why.
function VenueGap({ gap }) {
  if (!gap) return null
  return (
    <div
      className="mt-3 rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950"
      data-testid="venue-gap"
    >
      <p className="font-medium" data-testid="venue-gap-title">
        {gap.kind === 'uneven'
          ? 'Обе площадки прошли правила, но не на равных'
          : 'Почему площадки расходятся'}
      </p>
      {gap.rule_ru && (
        <p className="mt-1 break-words" data-testid="venue-gap-rule">
          {gap.rule_ru}
        </p>
      )}
      <ul className="mt-1 space-y-1">
        {gap.causes.map((cause, i) => (
          <li
            key={`${cause.kind}-${i}`}
            className={`break-words ${cause.measured ? '' : 'italic text-amber-800'}`}
            data-testid={`venue-gap-cause-${cause.kind}`}
          >
            {cause.text_ru}
          </li>
        ))}
      </ul>
    </div>
  )
}

function Venues({ ladder }) {
  if (!ladder.venues?.length) return null
  return (
    <div className="mb-4" data-testid="ladder-venues">
      <p className="mb-2 text-sm font-medium text-slate-900" data-testid="ladder-summary">
        По площадкам: {ladder.summary_ru}
      </p>
      <div className="grid gap-3 md:grid-cols-2">
        {ladder.venues.map((view) => (
          <VenueCard key={view.venue} view={view} />
        ))}
      </div>
      <VenueGap gap={ladder.venue_gap} />
    </div>
  )
}

function MoneyLines({ money }) {
  if (!money?.length) return null
  return (
    <ul className="mt-2 space-y-0.5 text-sm text-slate-700">
      {money.map((m) => (
        <li key={m.venue} data-testid={`money-${m.venue}`}>
          <span className="font-medium">{m.venue === 'binance' ? 'Binance' : 'Hyperliquid'}:</span>{' '}
          {m.outcome_ru}
          {m.reason_ru && <span className="italic text-slate-500"> ({m.reason_ru})</span>}
          {m.detail ? ` — ${m.detail}` : ''}
        </li>
      ))}
    </ul>
  )
}

function Readings({ readings }) {
  if (!readings?.length) return null
  return (
    <ul className="mt-2 space-y-0.5 text-sm text-slate-700">
      {readings.map((r) => (
        <li key={r.name} data-testid={`reading-${r.name}`}>
          <span className={r.recorded ? 'text-emerald-700' : 'text-slate-400'}>
            {r.recorded ? '● ' : '○ '}
          </span>
          {r.text}
          {r.reason_ru && <span className="italic text-slate-500"> ({r.reason_ru})</span>}
        </li>
      ))}
    </ul>
  )
}

function StepRow({ step, ideaId }) {
  const notRun = step.outcome === 'not_run'
  // The causal twin's chart: the twin's forward test is the evidence behind step 2 of an
  // original that runs on Hyperliquid; a passed twin has one stored series.
  const twinChart =
    step.number === 2 && step.other_idea?.endsWith('-causal') && step.run?.route === 'paper'
  return (
    <li className="rounded border border-slate-200 p-3" data-testid={`step-${step.number}`}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-xs font-semibold tabular-nums text-slate-400">{step.number}</span>
        <h3 className="text-sm font-semibold text-slate-900">{step.title}</h3>
        <span
          className={`rounded border px-2 py-0.5 text-xs font-medium ${OUTCOME_STYLE[step.outcome] ?? ''}`}
          data-testid={`step-${step.number}-outcome`}
        >
          {step.outcome_ru}
        </span>
        {notRun && step.reason_ru && (
          <span className="text-sm italic text-slate-600" data-testid={`step-${step.number}-reason`}>
            — {step.reason_ru}
          </span>
        )}
      </div>
      {step.number === 4 ? (
        <MoneyLines money={step.money} />
      ) : (
        step.detail && <p className="mt-1 break-words text-sm text-slate-700">{step.detail}</p>
      )}
      {step.execution_ru && (
        <p className="mt-1 text-xs text-slate-500" data-testid={`step-${step.number}-execution`}>
          исполнение: {step.execution_ru}
        </p>
      )}
      {step.number === 5 ? (
        <Readings readings={step.readings} />
      ) : (
        <RunNumbers trial={step.trial} />
      )}
      <RunLine run={step.run} ideaId={ideaId} />
      {twinChart && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-sky-700">
            график проверки вперёд двойника
          </summary>
          <div className="mt-2">
            <ForwardCharts ideaId={step.other_idea} />
          </div>
        </details>
      )}
    </li>
  )
}

export default function Ladder({ ladder }) {
  if (!ladder) return null
  return (
    <Panel
      title="Лестница проверки"
      subtitle="Одни и те же семь ступеней для каждой идеи; Binance и Hyperliquid — две площадки, у каждой свой вердикт, свои монеты, издержки и минимальная ножка. У ступени, которую не проводили, названа причина. «Не решён» — ступень проведена, но данных пока мало для ответа. Решает код по правилам; показания пятой ступени ничего не решают."
    >
      <Venues ladder={ladder} />
      <ol className="space-y-3">
        {ladder.steps.map((step) => (
          <StepRow key={step.number} step={step} ideaId={ladder.idea_id} />
        ))}
      </ol>
    </Panel>
  )
}
