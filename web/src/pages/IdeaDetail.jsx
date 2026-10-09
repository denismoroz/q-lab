import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { fetchIdea, fetchIdeaPage } from '../api.js'
import { Absent, Json, Pager, Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import { VerdictLegend, VerdictList } from '../components/Verdict.jsx'
import TrialTable from '../components/TrialTable.jsx'
import {
  ASSET_CLASS,
  PROFILE,
  SHUTDOWN_CAUSE,
  SOURCE_TYPE,
  TRIAL_ROUTE,
  routeHint,
  label,
} from '../labels.js'

const PAGE_SIZE = 50

function usePagedSection(ideaId, resource) {
  const [offset, setOffset] = useState(0)
  const query = useQuery({
    queryKey: ['idea', ideaId, resource, offset],
    queryFn: () => fetchIdeaPage(ideaId, resource, { limit: PAGE_SIZE, offset }),
    placeholderData: keepPreviousData,
  })
  return { query, offset, setOffset }
}

function Field({ title, children }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{title}</dt>
      <dd className="mt-0.5 text-sm text-slate-800">{children}</dd>
    </div>
  )
}

function Section({ title, subtitle, section, children }) {
  const { query, offset, setOffset } = section
  return (
    <Panel title={title} subtitle={subtitle}>
      <QueryState query={query}>
        {query.data && (
          <>
            <Pager
              total={query.data.total}
              limit={query.data.limit}
              offset={query.data.offset}
              onChange={setOffset}
            />
            {children(query.data.items)}
            {query.data.total > offset + PAGE_SIZE && (
              <Pager
                total={query.data.total}
                limit={query.data.limit}
                offset={query.data.offset}
                onChange={setOffset}
              />
            )}
          </>
        )}
      </QueryState>
    </Panel>
  )
}

export default function IdeaDetail() {
  const { ideaId } = useParams()
  const ideaQuery = useQuery({ queryKey: ['idea', ideaId], queryFn: () => fetchIdea(ideaId) })
  const specs = usePagedSection(ideaId, 'specs')
  const trials = usePagedSection(ideaId, 'trials')
  const verdicts = usePagedSection(ideaId, 'verdicts')

  return (
    <div className="space-y-6">
      <Link className="text-sm text-sky-700 underline-offset-2 hover:underline" to="/ideas">
        ← к реестру
      </Link>

      <QueryState query={ideaQuery}>
        {ideaQuery.data && (
          <>
            <Outcome outcome={ideaQuery.data.latest_outcome} />
            <Header idea={ideaQuery.data} />
          </>
        )}
      </QueryState>

      <Section
        title="Спецификации"
        subtitle="Формализации идеи: параметры, требования к данным, модель костов."
        section={specs}
      >
        {(items) => (
          <div className="space-y-3">
            {items.length === 0 && <p className="text-sm text-slate-500">Спецификаций нет.</p>}
            {items.map((spec) => (
              <div key={spec.id} className="rounded border border-slate-200 p-3">
                <div className="flex flex-wrap items-baseline gap-x-4 text-sm">
                  <span className="font-medium">версия {spec.version}</span>
                  <span className="font-mono text-xs text-slate-500">{spec.code_ref}</span>
                  <span className="text-xs text-slate-500">ребаланс: {spec.rebalance}</span>
                  <span className="text-xs text-slate-400">{spec.created_at?.slice(0, 10)}</span>
                </div>
                {/* Collapsed by default: a calibration idea carries hundreds
                    of spec versions, and expanding every one of them buries
                    the trials and verdicts further down the page. */}
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-sky-700">
                    параметры, данные, косты
                  </summary>
                  <div className="mt-2 grid gap-3 md:grid-cols-3">
                    <div>
                      <div className="mb-1 text-xs font-medium text-slate-500">Параметры</div>
                      <Json value={spec.params} />
                    </div>
                    <div>
                      <div className="mb-1 text-xs font-medium text-slate-500">Данные</div>
                      <Json value={spec.data_requirements} />
                    </div>
                    <div>
                      <div className="mb-1 text-xs font-medium text-slate-500">Косты</div>
                      <Json value={spec.costs_model} />
                    </div>
                  </div>
                </details>
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section
        title="Прогоны"
        subtitle="Все прогоны идеи, новые сверху — включая выброшенные."
        section={trials}
      >
        {(items) => <TrialTable trials={items} showIdea={false} />}
      </Section>

      <Panel
        title="Вердикты"
        subtitle="Три вида строки различаются: решение, неизвестно, измерение. Проход/провал берётся из реестра, а не пересчитывается."
      >
        <VerdictLegend />
      </Panel>

      <Section title="Вердикты — строки" section={verdicts}>
        {(items) => <VerdictList items={items} />}
      </Section>
    </div>
  )
}

const OUTCOME_STYLE = {
  paper: 'border-emerald-300 bg-emerald-50',
  shelf: 'border-indigo-300 bg-indigo-50',
  'needs-infrastructure': 'border-indigo-300 bg-indigo-50',
  'needs-forward': 'border-sky-300 bg-sky-50',
  reject: 'border-stone-300 bg-stone-50',
  'not-evaluable': 'border-amber-300 bg-amber-50',
  'needs-more-data': 'border-amber-300 bg-amber-50',
  error: 'border-rose-300 bg-rose-50',
}

function pct(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : null
}

// The one question the idea page must answer first (owner, 2026-10-01):
// what did q-lab itself conclude, and why. Statuses can come from imports
// or from people; this block is only q-lab's own latest decision.
function Outcome({ outcome }) {
  const trial = outcome?.trial
  if (!trial) {
    return (
      <Panel title="Что говорит q-lab">
        <p className="text-sm text-slate-600">
          q-lab ещё не выносил по этой идее решения с записанным исходом.
          {outcome?.unrouted_trials
            ? ` Есть ${outcome.unrouted_trials} прогонов до 2026-10-01 — тогда исход печатался и не сохранялся.`
            : ''}
        </p>
      </Panel>
    )
  }
  const m = trial.metrics ?? {}
  const sharpe = (v) => (typeof v === 'number' ? v.toFixed(2) : null)
  const days = (v) => (typeof v === 'number' ? `${Math.round(v)} дн.` : null)
  // docs/FIT_VS_FORWARD.md: the plain metrics are the JUDGED part -- the
  // forward test when there is one, else the selection period; `fit_*` are
  // the selection period's when both exist.
  const split = typeof m.judged_on_forward === 'number'
  const judged = split && m.judged_on_forward === 1 ? 'проверка вперёд' : 'период подбора'
  const numbers = [
    ['доходность в год после комиссий', pct(m.ann_return_net)],
    ['Шарп после комиссий', sharpe(m.sharpe_net)],
    ['худшая просадка', pct(m.max_dd)],
    ['обгоняет шум той же формы', pct(m.noise_return_percentile)],
    ['книга была полной', pct(m.book_coverage)],
  ].filter(([, v]) => v !== null)
  const fitNumbers = [
    ['доходность в год после комиссий', pct(m.fit_ann_return_net)],
    ['Шарп после комиссий', sharpe(m.fit_sharpe_net)],
    ['худшая просадка', pct(m.fit_max_dd)],
  ].filter(([, v]) => v !== null)
  const periods = split
    ? [
        ['период подбора', days(m.selection_days)],
        ['проверка вперёд', days(m.forward_days)],
        ['нужно для вывода', days(m.forward_days_needed)],
      ].filter(([, v]) => v !== null)
    : []
  return (
    <section className={`rounded border p-4 ${OUTCOME_STYLE[trial.route] ?? 'border-slate-200'}`}>
      <div className="text-xs uppercase tracking-wide text-slate-500">Что говорит q-lab</div>
      <div className="mt-1 text-lg font-semibold text-slate-900">{label(TRIAL_ROUTE, trial.route)}</div>
      <p className="text-sm text-slate-700">{routeHint(trial)}</p>
      <p className="mt-2 text-xs text-slate-500">
        <span>Как записано в реестре: </span>
        <span className="font-mono">{trial.route_reason}</span>
      </p>
      {periods.length > 0 && <NumberList title="Данные" numbers={periods} />}
      {numbers.length > 0 && (
        <NumberList title={split ? `${judged} — по ней решение` : null} numbers={numbers} />
      )}
      {fitNumbers.length > 0 && (
        <NumberList title="период подбора — только для сведения" numbers={fitNumbers} />
      )}
      <RegimeTable m={m} />
      <p className="mt-3 text-xs text-slate-500">
        прогон #{trial.id}, {trial.started_at?.slice(0, 10)}
        {outcome.unrouted_trials
          ? ` · ещё ${outcome.unrouted_trials} прогонов до 2026-10-01 без записанного исхода; их числа ниже — история, а не вывод`
          : ''}
      </p>
    </section>
  )
}

// docs/REGIMES.md: BTC's 30-day return, split in thirds over 2019-2026 --
// a description of where the strategy earns and loses, read by no rule.
const REGIME_ROWS = [
  ['bull', 'рост'],
  ['flat', 'боковик'],
  ['bear', 'падение'],
]

function RegimeTable({ m }) {
  if (typeof m.regime_bull_share !== 'number') return null
  const cell = (v, f) => (typeof v === 'number' ? f(v) : '—')
  const p = (v) => `${(v * 100).toFixed(1)}%`
  return (
    <div className="mt-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        по режимам рынка (BTC за 30 дней: падение ниже {cell(m.regime_bear_below, p)}, рост выше{' '}
        {cell(m.regime_bull_above, p)}) — для сведения
      </div>
      <table className="mt-1 w-full text-sm tabular-nums">
        <thead className="text-xs text-slate-500">
          <tr>
            <th className="text-left font-normal">режим</th>
            <th className="text-right font-normal">доля дней</th>
            <th className="text-right font-normal">стратегия</th>
            <th className="text-right font-normal">в год</th>
            <th className="text-right font-normal">Шарп</th>
            <th className="text-right font-normal">просадка</th>
            <th className="text-right font-normal">BTC</th>
            <th className="text-right font-normal">обгоняет шум</th>
            <th className="text-right font-normal">заявка сбылась</th>
          </tr>
        </thead>
        <tbody>
          {REGIME_ROWS.map(([key, name]) => (
            <tr key={key}>
              <td>{name}</td>
              <td className="text-right">{cell(m[`regime_${key}_share`], p)}</td>
              <td className="text-right">{cell(m[`regime_${key}_return`], p)}</td>
              <td className="text-right">{cell(m[`regime_${key}_ann_return`], p)}</td>
              <td className="text-right">{cell(m[`regime_${key}_sharpe`], (v) => v.toFixed(2))}</td>
              <td className="text-right">{cell(m[`regime_${key}_max_dd`], p)}</td>
              <td className="text-right">{cell(m[`regime_${key}_btc_return`], p)}</td>
              <td className="text-right">{cell(m[`regime_${key}_noise_percentile`], p)}</td>
              <td className="text-right">
                {cell(m[`regime_${key}_claim_met`], (v) => (v === 1 ? 'да' : 'нет'))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function NumberList({ title, numbers }) {
  return (
    <div className="mt-3">
      {title && <div className="text-xs uppercase tracking-wide text-slate-500">{title}</div>}
      <dl className="mt-1 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
        {numbers.map(([name, value]) => (
          <div key={name} className="flex justify-between gap-4">
            <dt className="text-slate-500">{name}</dt>
            <dd className="tabular-nums font-medium">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function Header({ idea }) {
  const counts = idea.counts
  return (
    <div className="space-y-4">
      <Panel>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold">{idea.title}</h1>
            <p className="font-mono text-xs text-slate-400">{idea.id}</p>
          </div>
          <div className="text-right">
            <StatusBadge status={idea.status} />
            {idea.shutdown_cause && (
              <div className="mt-1 text-xs text-stone-600">
                причина остановки: {label(SHUTDOWN_CAUSE, idea.shutdown_cause)}
              </div>
            )}
          </div>
        </div>

        <dl className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field title="профиль">{label(PROFILE, idea.profile)}</Field>
          <Field title="класс актива">{label(ASSET_CLASS, idea.asset_class)}</Field>
          <Field title="источник">
            {label(SOURCE_TYPE, idea.source_type)}
            {idea.source_url && (
              <a
                className="ml-2 break-all text-xs text-sky-700 underline-offset-2 hover:underline"
                href={idea.source_url}
                target="_blank"
                rel="noreferrer"
              >
                {idea.source_url}
              </a>
            )}
          </Field>
          <Field title="драйвер">
            {idea.driver ? idea.driver.title : <Absent>драйвер не назначен</Absent>}
          </Field>
        </dl>

        <dl className="mt-4 grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Field title="спеков">{counts.specs}</Field>
          <Field title="прогонов">{counts.trials}</Field>
          <Field title="вердиктов">{counts.verdicts}</Field>
          <Field title="решений">{counts.verdicts_decision}</Field>
          <Field title="неизвестно">{counts.verdicts_unknown}</Field>
          <Field title="измерений">{counts.verdicts_measurement}</Field>
        </dl>
      </Panel>

      {idea.driver && (
        <Panel title="Драйвер" subtitle={idea.driver.id}>
          <dl className="grid gap-4 md:grid-cols-2">
            <Field title="механизм">{idea.driver.description}</Field>
            <Field title="что убьёт эдж">{idea.driver.kill_condition}</Field>
            <Field title="как наблюдать">{idea.driver.observable}</Field>
          </dl>
        </Panel>
      )}

      <Panel title="Что утверждает источник и что мы записали">
        <dl className="grid gap-4">
          <Field title="заявленный эдж">
            {idea.claimed_edge ?? <Absent>не записан</Absent>}
          </Field>
          <Field title="заметки">
            {idea.notes ? (
              <p className="whitespace-pre-wrap text-sm leading-relaxed">{idea.notes}</p>
            ) : (
              <Absent>нет</Absent>
            )}
          </Field>
        </dl>
      </Panel>
    </div>
  )
}
