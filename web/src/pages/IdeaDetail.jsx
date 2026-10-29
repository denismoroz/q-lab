import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { fetchIdea, fetchIdeaPage } from '../api.js'
import { Absent, Json, Pager, Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import { VerdictLegend, VerdictList } from '../components/Verdict.jsx'
import TrialTable from '../components/TrialTable.jsx'
import YearView from '../components/YearView.jsx'
import Protection from '../components/Protection.jsx'
import ForwardCharts from '../components/ForwardCharts.jsx'
import OverlayPanels from '../components/OverlayPanels.jsx'
import Anomalies from '../components/Anomalies.jsx'
import {
  ASSET_CLASS,
  PROFILE,
  SHUTDOWN_CAUSE,
  SOURCE_TYPE,
  RESEARCH_NOTES,
  TRIAL_ROUTE,
  routeHint,
  label,
} from '../labels.js'
import { daysText, shortForwardReturn } from '../forwardReturn.js'
import { basisText, powerOf, regimePrefix, standLine } from '../headline.js'

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
            <TwoStages stages={ideaQuery.data.two_stage} />
            <ForwardCharts ideaId={ideaQuery.data.id} />
            <OverlayPanels ideaId={ideaQuery.data.id} />
            <CausalTwin twin={ideaQuery.data.causal_twin} />
            <Assessment
              assessment={ideaQuery.data.assessment}
              latestTrialId={ideaQuery.data.latest_outcome?.trial?.id}
            />
            <Anomalies block={ideaQuery.data.anomalies} />
            <PaperCheck check={ideaQuery.data.paper_check} />
            <PaperRule rule={ideaQuery.data.paper_rule} />
            <Decay decay={ideaQuery.data.decay} />
            <HowItWorks how={ideaQuery.data.how_it_works} ideaId={ideaQuery.data.id} />
            <ResearchNotes notes={ideaQuery.data.research_notes} />
            <Header idea={ideaQuery.data} />
            <StrategyLine strategy={ideaQuery.data.strategy} />
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
  const h = trial.headline
  const sharpe = (v) => (typeof v === 'number' ? v.toFixed(2) : null)
  const days = (v) => (typeof v === 'number' ? `${Math.round(v)} дн.` : null)
  // docs/FIT_VS_FORWARD.md: the plain metrics are the JUDGED part -- the
  // forward test when there is one, else the selection period; `fit_*` are
  // the selection period's when both exist.
  const split = typeof m.judged_on_forward === 'number'
  const judged = split && m.judged_on_forward === 1 ? 'проверка вперёд' : 'период подбора'
  // A forward test shorter than 90 days is shown as its total over the period;
  // the yearly figure of so short a stretch is a small note, not the headline.
  const short = shortForwardReturn(m)
  const numbers = [
    short
      ? [`доходность за ${daysText(short.days)} после комиссий`, pct(short.total)]
      : ['доходность в год после комиссий', pct(h?.ann_return ?? m.ann_return_net)],
    ['Шарп после комиссий', sharpe(h?.sharpe ?? m.sharpe_net)],
    ['худшая просадка', pct(h?.max_dd ?? m.max_dd)],
    ['обгоняет шум той же формы', pct(h?.noise_percentile ?? m.noise_return_percentile)],
    ['хорошая стратегия (+7% в год) обогнала бы шум', pct(powerOf(m, h))],
    ['книга была полной', pct(m.book_coverage)],
  ].filter(([, v]) => v !== null)
  const fitNumbers = [
    ['доходность в год после комиссий', pct(h?.fit ? h.fit.ann_return : m.fit_ann_return_net)],
    ['Шарп после комиссий', sharpe(h?.fit ? h.fit.sharpe : m.fit_sharpe_net)],
    ['худшая просадка', pct(h?.fit ? h.fit.max_dd : m.fit_max_dd)],
  ].filter(([, v]) => v !== null)
  const periods = split
    ? [
        ['окно подбора параметров', days(m.fit_window_days)],
        ['период подбора в этих данных', days(m.selection_days)],
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
      {h && (
        <p className="mt-1 text-xs text-slate-500" data-testid="headline-basis">
          Числа выше — {basisText(h)}.
          {standLine(h) && <span data-testid="headline-stand"> {standLine(h)}.</span>}
        </p>
      )}
      {short && (
        <p className="mt-1 text-xs text-slate-400">
          в пересчёте на год: {pct(short.yearly)} — на коротком отрезке ненадёжно
        </p>
      )}
      {fitNumbers.length > 0 && (
        <NumberList title="период подбора — только для сведения" numbers={fitNumbers} />
      )}
      <RegimeTable m={m} />
      <YearView view={trial.year_view} />
      <Protection view={trial.protection} />
      <p className="mt-3 text-xs text-slate-500">
        прогон #{trial.id}, {trial.started_at?.slice(0, 10)}
        {outcome.unrouted_trials
          ? ` · ещё ${outcome.unrouted_trials} прогонов до 2026-10-01 без записанного исхода; их числа ниже — история, а не вывод`
          : ''}
      </p>
    </section>
  )
}

// docs/TWO_STAGE.md (T45): an idea the generator wrote is judged twice -- on
// Binance's long history, then as the same mechanism on Hyperliquid -- and needs
// both. The two runs side by side; the sentence below them is a reading of the
// stored routes, not a new verdict.
const STAGE_VENUE = { binance: 'Binance', hyperliquid: 'Hyperliquid' }
const STAGE_PASSED = ['paper', 'shelf', 'needs-infrastructure']

function twoStageSentence(stages) {
  const r1 = stages.stage1?.trial?.route
  const r2 = stages.stage2?.trial?.route
  if (!stages.stage2) {
    return 'Второй этап ещё не проводился: его запускает ночной конвейер для тех, кто прошёл первый.'
  }
  if (r2 === 'reject') {
    const end = stages.stage2.data_range?.end
    return `Отвергнута на втором этапе${end ? `: на данных Hyperliquid по ${end}` : ''}. Воскресить можно только на данных, которых тогда не было.`
  }
  if (r1 && STAGE_PASSED.includes(r1) && ['paper', 'shelf'].includes(r2)) {
    return 'Прошла оба этапа. Дальше решает владелец.'
  }
  return 'Оба этапа вместе пока не решают: идея ждёт. Пройти надо обоим.'
}

function StageColumn({ title, hint, stage, ideaId }) {
  const trial = stage?.trial
  const m = trial?.metrics ?? {}
  const h = trial?.headline
  const lead = h?.sharpe ?? m.sharpe_net
  const sharpe = typeof lead === 'number' ? lead.toFixed(2) : null
  const rows = [
    ['доходность в год после комиссий', pct(h?.ann_return ?? m.ann_return_net)],
    ['Шарп после комиссий', sharpe],
    ['худшая просадка', pct(h?.max_dd ?? m.max_dd)],
    ['обгоняет шум той же формы', pct(h?.noise_percentile ?? m.noise_return_percentile)],
    ['проверка вперёд, дней', typeof m.forward_days === 'number' ? Math.round(m.forward_days) : null],
  ].filter(([, v]) => v !== null)
  return (
    <div className="rounded border border-slate-200 p-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">{title}</div>
      <p className="text-xs text-slate-500">{hint}</p>
      {!trial ? (
        <p className="mt-2 text-sm text-slate-500">прогона ещё нет</p>
      ) : (
        <>
          <div className="mt-1 text-base font-semibold text-slate-900">
            {label(TRIAL_ROUTE, trial.route)}
          </div>
          {rows.length > 0 && <NumberList title={null} numbers={rows} />}
          {h && (
            <p className="mt-1 text-xs text-slate-500" data-testid="stage-basis">
              {basisText(h)}.
              {standLine(h) && <span data-testid="stage-stand"> {standLine(h)}.</span>}
            </p>
          )}
          <YearView view={trial.year_view} compact />
          <Protection view={trial.protection} compact />
          <p className="mt-2 break-words font-mono text-xs text-slate-500">{trial.route_reason}</p>
          <p className="mt-2 text-xs text-slate-500">
            данные: {STAGE_VENUE[stage.venue] ?? stage.venue}
            {stage.data_range ? `, ${stage.data_range.start} — ${stage.data_range.end}` : ''} · прогон #
            {trial.id} ·{' '}
            <Link className="text-sky-700 underline-offset-2 hover:underline" to={`/ideas/${ideaId}`}>
              {ideaId}
            </Link>
          </p>
        </>
      )}
    </div>
  )
}

function TwoStages({ stages }) {
  if (!stages) return null
  return (
    <Panel
      title="Два этапа проверки"
      subtitle="Идею проверяют дважды: на длинной истории Binance и тем же механизмом, с теми же числами, на Hyperliquid, где идёт бумажная и боевая торговля. Нужно пройти оба."
    >
      <div className="grid gap-3 md:grid-cols-2">
        <StageColumn
          title="Этап 1 — длинная история Binance"
          hint="спека как записана; контракты Binance с 2019 года, в том числе снятые с торгов"
          stage={stages.stage1}
          ideaId={stages.parent_id}
        />
        <StageColumn
          title="Этап 2 — Hyperliquid"
          hint="те же блоки и числа, монеты и комиссии Hyperliquid; проверка вперёд на данных после даты фиксации параметров; код без агента"
          stage={stages.stage2}
          ideaId={stages.child_id}
        />
      </div>
      <p className="mt-3 text-sm font-medium text-slate-800">{twoStageSentence(stages)}</p>
    </Panel>
  )
}

// 2026-10-28, owner: «почему не пересчинаны по принципу HL/binance?». The Binance-then-
// Hyperliquid answer of an idea lives under its causal twin (qlab.twin_stage); the original's
// page shows it here instead of only on the twin's own page.
function CausalTwin({ twin }) {
  if (!twin) return null
  return (
    <section className="space-y-3 rounded border border-sky-200 bg-sky-50/40 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        Та же идея, проверенная на Binance и Hyperliquid
      </div>
      <p className="text-sm text-slate-700">
        Выше — прогон самой идеи, как она записана: числа выбраны один раз, и проверка вперёд у
        неё — только время после этого выбора. Ответ по схеме «сначала длинная
        история Binance, потом Hyperliquid» записан у её двойника: те же правила, но числа код
        выбирает каждый месяц только по прошлому, поэтому всё время после первого выбора — это
        проверка вперёд за годы.{' '}
        <Link className="text-sky-700 underline-offset-2 hover:underline" to={`/ideas/${twin.id}`}>
          {twin.id}
        </Link>{' '}
        · <StatusBadge status={twin.status} />
      </p>
      {twin.two_stage ? (
        <TwoStages stages={twin.two_stage} />
      ) : (
        <p className="text-sm text-slate-500">
          Второго этапа у двойника пока нет: его запускает ночной конвейер для тех, кто прошёл первый.
        </p>
      )}
      <ForwardCharts ideaId={twin.id} />
    </section>
  )
}

// 2026-10-25: the paper book and the stand's test of the same configuration,
// over ONE window (qlab.paper_check). Two numbers on different windows had the
// stand at minus and the paper book at plus.
function dayText(iso) {
  return iso ? `${iso.slice(8, 10)}.${iso.slice(5, 7)}.${iso.slice(0, 4)}` : '—'
}

function PaperCheck({ check }) {
  if (!check) return null
  const differ = check.note.startsWith('Отрезки разные')
  const rows = [
    [`бумажная книга за ${dayText(check.window_start)}–${dayText(check.window_end)}`, pct(check.paper_return)],
    [
      check.stand_window_start
        ? `стенд за ${dayText(check.stand_window_start)}–${dayText(check.stand_window_end)}`
        : 'стенд',
      pct(check.stand_return) ?? 'не посчитан',
    ],
    ['та же бумажная книга, пересчитанная на стенде', pct(check.replay_return)],
  ].filter(([, v]) => v !== null)
  return (
    <section className={`rounded border p-4 ${differ ? 'border-amber-300 bg-amber-50' : 'border-slate-200'}`}>
      <div className="text-xs uppercase tracking-wide text-slate-500">Бумага и стенд за один отрезок</div>
      <p className="mt-1 text-sm text-slate-700">
        Итоги за сам отрезок, без пересчёта на год. Бумажная книга — та, что торгует на бумаге на
        боевом сервере тем же набором настроек.
      </p>
      <NumberList numbers={rows} />
      {check.note && <p className="mt-2 text-sm font-medium text-amber-800">{check.note}</p>}
      <p className="mt-3 text-xs text-slate-500">
        Почему две книги с одними настройками могут разойтись: книга на стенде могла начать работу
        раньше и уже держать позиции в первый день отрезка, а бумажная начала с пустого счёта; к
        тому же стенд может считать по дням, а бумажная книга шагает по часам. Поэтому сравнивать
        можно только числа за один и тот же отрезок.
      </p>
      <p className="mt-2 text-xs text-slate-400">
        сверка за {dayText(check.at)}, книга «{check.book}»
        {check.stand_trial_id ? `, прогон стенда #${check.stand_trial_id}` : ''}
      </p>
    </section>
  )
}

// 2026-10-25, owner: «как минимум 4-6 недель в paper». How long the idea's paper
// book has run, and whether the stand's replay of it matched (qlab.paper_rule).
// Nothing here changes a status: paper does not judge profitability, the stand does.
function PaperRule({ rule }) {
  if (!rule) return null
  const replay = {
    match: 'повтор бумаги на стенде сошёлся',
    mismatch: 'повтор бумаги на стенде не сошёлся',
    unknown: 'повтор бумаги на стенде не посчитан',
  }[rule.replay]
  const tone = rule.ready ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200'
  return (
    <section className={`rounded border p-4 ${tone}`}>
      <div className="text-xs uppercase tracking-wide text-slate-500">
        Правило владельца: бумага не меньше {rule.min_weeks}–{rule.max_weeks} недель
      </div>
      <p className="mt-1 text-sm font-medium text-slate-800">
        На бумаге {rule.whole_weeks} {weeksWord(rule.whole_weeks)} из минимум {rule.min_weeks}–
        {rule.max_weeks} (с {dayText(rule.first)})
      </p>
      <p className="mt-1 text-sm text-slate-700">{replay}</p>
      {rule.ready ? (
        <p className="mt-2 text-sm font-medium text-emerald-800">
          По правилу владельца книга готова к решению (бумага не судит о доходности — судит стенд).
        </p>
      ) : rule.weeks >= rule.min_weeks ? (
        <p className="mt-2 text-sm text-amber-800">
          Срок вышел, но книга не готова к решению, пока повтор на стенде не сойдётся.
        </p>
      ) : (
        <p className="mt-2 text-sm text-slate-600">До решения ещё рано.</p>
      )}
      <p className="mt-3 text-xs text-slate-500">
        Бумажная торговля нужна не для оценки доходности, а чтобы проверить, что стратегия на
        живых данных ведёт себя так, как её повторяет стенд. Книга «{rule.book}», последняя
        строка выгрузки — {dayText(rule.last)}. Статус идеи это правило не меняет.
      </p>
    </section>
  )
}

// qlab.decay: does the running strategy still behave like its own history on the
// stand? The realised result against same-length windows of the stand's history.
// It shows where the result falls and labels nothing good or bad: the alert
// threshold is the owner's to set (night/decay.yaml), and while it is null no
// label appears. A weeks-long return says nothing about profitability.
function signedPct(value, digits = 1) {
  if (typeof value !== 'number') return '—'
  const text = `${value >= 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`
  return text.replace('-', '−')
}

function plainPct(value, digits = 2) {
  return typeof value === 'number' ? `${(value * 100).toFixed(digits)}%` : '—'
}

function Decay({ decay }) {
  if (!decay) return null
  const days = Math.round(decay.days)
  const rows = [
    {
      name: 'Доходность за период',
      m: decay.ret,
      fmt: signedPct,
      place: (p) => `хуже были ${Math.round(p)} из 100`,
    },
    {
      name: 'Самое глубокое падение',
      m: decay.drawdown,
      fmt: signedPct,
      place: (p) => `глубже было у ${Math.round(p)} из 100`,
    },
    {
      name: 'Дневной разброс',
      m: decay.spread,
      fmt: plainPct,
      place: (p) => `спокойнее были ${Math.round(p)} из 100`,
    },
  ]
  const tone = decay.alert ? 'border-amber-300 bg-amber-50' : 'border-slate-200'
  return (
    <section className={`rounded border p-4 ${tone}`}>
      <div className="text-xs uppercase tracking-wide text-slate-500">
        Ведёт ли себя стратегия как на стенде
      </div>
      <p className="mt-1 text-sm text-slate-700">
        {decay.live ? 'Живые деньги' : 'Бумажная книга'} «{decay.book}»: {days} {daysWord(days)} с{' '}
        {dayText(decay.first)}. Её результат поставлен рядом с такими же по длине отрезками из
        истории этой же стратегии на стенде до {dayText(decay.first)}.
      </p>
      {decay.note ? (
        <p className="mt-2 text-sm font-medium text-amber-800">{decay.note}</p>
      ) : (
        <>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-1 pr-4 font-normal">что смотрим</th>
                  <th className="py-1 pr-4 font-normal">сейчас</th>
                  <th className="py-1 pr-4 font-normal">на стенде обычно (от … до, посередине)</th>
                  <th className="py-1 font-normal">где лежит</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ name, m, fmt, place }) => (
                  <tr key={name} className="border-t border-slate-100 align-top">
                    <td className="py-1 pr-4 text-slate-600">{name}</td>
                    <td className="py-1 pr-4 tabular-nums font-medium">{fmt(m.realised)}</td>
                    <td className="py-1 pr-4 tabular-nums">
                      {m.stand
                        ? `${fmt(m.stand.p10)} … ${fmt(m.stand.p90)}, ${fmt(m.stand.p50)}`
                        : '—'}
                    </td>
                    <td className="py-1 tabular-nums">
                      {typeof m.percentile === 'number' ? place(m.percentile) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Отрезков такой длины в истории стенда: {decay.windows} (с {dayText(decay.history_from)}{' '}
            по {dayText(decay.history_to)}); друг на друга не налегают примерно {decay.independent}.
            Чем их меньше, тем грубее оценка.
          </p>
        </>
      )}
      {decay.bands?.map((b) => (
        <p key={b.id} className="mt-2 text-sm text-slate-700">
          Записано заранее, до результатов: {b.label} —{' '}
          {b.status === 'not_applicable'
            ? `к этому отрезку не относится (заявлено на ${Math.round(b.horizon_days)} ${daysWord(Math.round(b.horizon_days))})`
            : b.kind === 'stop'
              ? `сейчас ${signedPct(b.value)}, условие остановки ${b.status === 'outside' ? 'СРАБОТАЛО' : 'не сработало'}`
              : `сейчас ${signedPct(b.value)}, ${b.status === 'inside' ? 'внутри' : 'ВНЕ'} заявленной полосы`}
          .
        </p>
      ))}
      {decay.alert === 'below' && (
        <p className="mt-2 text-sm font-medium text-amber-800">
          Результат ниже порога тревоги, который задал владелец.
        </p>
      )}
      <p className="mt-3 text-xs text-slate-500">
        Порог тревоги задаёт владелец (`night/decay.yaml`, `alert_percentile`); пока он не задан,
        «хорошо» и «плохо» здесь не говорится: показано только, где результат лежит среди обычных
        на стенде. Доходность за недели о выгоде стратегии не говорит — только о том, похожа ли
        она на себя. Проверка за {dayText(decay.at)}.
      </p>
    </section>
  )
}

function daysWord(n) {
  if (n % 10 === 1 && n % 100 !== 11) return 'день'
  if ([2, 3, 4].includes(n % 10) && ![12, 13, 14].includes(n % 100)) return 'дня'
  return 'дней'
}

function weeksWord(n) {
  if (n % 10 === 1 && n % 100 !== 11) return 'неделя'
  if ([2, 3, 4].includes(n % 10) && ![12, 13, 14].includes(n % 100)) return 'недели'
  return 'недель'
}

// From the idea's card (cards/<idea_id>.yaml): what the strategy does, in
// plain words, and the exact rule for entering and leaving.
function HowItWorks({ how, ideaId }) {
  if (!how) return null
  const rows = [
    ['когда входит', how.signal?.entry],
    ['когда выходит', how.signal?.exit],
    ['сколько', how.sizing?.rule],
  ].filter(([, v]) => v)
  return (
    <section className="rounded border border-slate-200 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">Как работает стратегия</div>
      <p className="mt-2 text-sm text-slate-700">{how.plain_summary}</p>
      <dl className="mt-3 space-y-2 text-sm">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
            <dd className="text-slate-700">{v}</dd>
          </div>
        ))}
      </dl>
      <Link className="mt-3 inline-block text-xs text-sky-700 hover:underline" to={`/cards/${ideaId}`}>
        вся карточка: источники, защита, как умрёт →
      </Link>
    </section>
  )
}

// docs/ASSESSOR.md: the assessor agent's second opinion. The verdict stays
// the code's; every number in a claim was checked against the run by code.
const ASSESSMENT_VIEW = {
  agree: 'согласен с вердиктом',
  too_harsh: 'считает вердикт слишком строгим',
  too_lenient: 'считает вердикт слишком мягким',
  unsure: 'не уверен',
}

function Assessment({ assessment, latestTrialId }) {
  if (!assessment) return null
  const a = assessment
  const stale = latestTrialId != null && a.trial_id !== latestTrialId
  return (
    <section
      className={`rounded border p-4 ${a.disputes ? 'border-amber-400 bg-amber-50' : 'border-slate-200'}`}
    >
      <div className="text-xs uppercase tracking-wide text-slate-500">Второе мнение агента</div>
      <div className="mt-1 text-lg font-semibold text-slate-900">
        {ASSESSMENT_VIEW[a.view] ?? a.view}
      </div>
      {a.disputes && (
        <p className="text-sm font-medium text-amber-800">
          Агент спорит с вердиктом. Вердикт не меняется — решать вам.
        </p>
      )}
      <p className="mt-2 text-sm text-slate-700">{a.summary}</p>
      {a.claims.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
          {a.claims.map((c, i) => (
            <li key={i}>
              {c.text}{' '}
              <span className="font-mono text-xs text-slate-400">
                [{c.fact ? `${c.fact} = ${Number(c.value).toPrecision(3)}` : `«${c.quote}»`}]
              </span>
            </li>
          ))}
        </ul>
      )}
      {a.checks.length > 0 && (
        <>
          <div className="mt-3 text-xs uppercase tracking-wide text-slate-500">что решило бы вопрос</div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-slate-700">
            {a.checks.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </>
      )}
      <p className="mt-3 text-xs text-slate-500">
        прогон #{a.trial_id}
        {stale ? ' (не последний — мнение о прошлом прогоне)' : ''}, {a.at.slice(0, 10)}, модель{' '}
        {a.model}. Числа в утверждениях сверены кодом с прогоном
        {a.rejected_claims ? `; не сошлись и отброшены: ${a.rejected_claims}` : ''}.
      </p>
    </section>
  )
}

// docs/RESEARCHER.md: the researcher agent's memos about this strategy and the
// checks the stand ran on its proposals. The memo is shown as plain wrapped
// text, folded away until asked for.
function ExperimentRow({ e }) {
  const L = RESEARCH_NOTES
  const route = e.route === 'error' ? L.error : e.route === 'not-run' ? L.notRun : label(TRIAL_ROUTE, e.route)
  return (
    <li>
      {e.idea_id && e.trial_id != null ? (
        <Link className="font-mono text-sky-700 hover:underline" to={`/ideas/${e.idea_id}`}>
          {e.idea_id}
        </Link>
      ) : (
        <span className="font-mono text-slate-700">{e.idea_id ?? e.spec}</span>
      )}
      {' — '}
      {route}
      {e.trial_id != null && <span className="text-xs text-slate-400">, {L.run} #{e.trial_id}</span>}
    </li>
  )
}

function ResearchNote({ note }) {
  const L = RESEARCH_NOTES
  const [open, setOpen] = useState(false)
  return (
    <div className="border-t border-slate-200 pt-3 first:border-t-0 first:pt-0">
      <div className="text-sm font-medium text-slate-900">{note.at.slice(0, 10)}</div>
      <div className="mt-2 text-xs uppercase tracking-wide text-slate-500">{L.experiments}</div>
      {note.experiments.length > 0 ? (
        <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-slate-700">
          {note.experiments.map((e, i) => (
            <ExperimentRow key={i} e={e} />
          ))}
        </ul>
      ) : (
        <p className="mt-1 text-sm text-slate-500">{L.noExperiments}</p>
      )}
      {note.problems.length > 0 && (
        <>
          <div className="mt-3 text-xs uppercase tracking-wide text-slate-500">{L.problems}</div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-amber-800">
            {note.problems.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </>
      )}
      <button
        type="button"
        className="mt-3 text-xs text-sky-700 hover:underline"
        onClick={() => setOpen(!open)}
      >
        {open ? L.hideMemo : L.showMemo}
      </button>
      {open && (
        <pre className="mt-2 whitespace-pre-wrap break-words rounded bg-slate-50 p-3 text-xs text-slate-800">
          {note.memo}
        </pre>
      )}
    </div>
  )
}

function ResearchNotes({ notes }) {
  if (!notes || notes.length === 0) return null
  return (
    <section className="space-y-3 rounded border border-slate-200 p-4">
      <div>
        <div className="text-xs uppercase tracking-wide text-slate-500">{RESEARCH_NOTES.title}</div>
        <p className="mt-1 text-sm text-slate-600">{RESEARCH_NOTES.subtitle}</p>
      </div>
      {notes.map((n) => (
        <ResearchNote key={n.memo_path} note={n} />
      ))}
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
  // The split on the strategy's own money when the run has it (docs/CAPITAL_BASE.md).
  const pre = regimePrefix(m)
  const cell = (v, f) => (typeof v === 'number' ? f(v) : '—')
  const p = (v) => `${(v * 100).toFixed(1)}%`
  return (
    <div className="mt-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        по режимам рынка (BTC за 30 дней: падение ниже {cell(m.regime_bear_below, p)}, рост выше{' '}
        {cell(m.regime_bull_above, p)}) — {pre === 'cap_regime_' ? 'на деньгах стратегии, ' : 'по счёту стенда, '}для сведения
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
              <td className="text-right">{cell(m[`${pre}${key}_share`], p)}</td>
              <td className="text-right">{cell(m[`${pre}${key}_return`], p)}</td>
              <td className="text-right">{cell(m[`${pre}${key}_ann_return`], p)}</td>
              <td className="text-right">{cell(m[`${pre}${key}_sharpe`], (v) => v.toFixed(2))}</td>
              <td className="text-right">{cell(m[`${pre}${key}_max_dd`], p)}</td>
              <td className="text-right">{cell(m[`${pre}${key}_btc_return`], p)}</td>
              <td className="text-right">{cell(m[`${pre}${key}_noise_percentile`], p)}</td>
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

// Variants of one strategy take no place of their own (owner, 2026-10-25):
// a variant points at its strategy, a strategy lists its active variants.
function StrategyLine({ strategy }) {
  if (!strategy) return null
  if (strategy.is_variant) {
    return (
      <p className="text-sm text-slate-700">
        вариант стратегии{' '}
        <Link className="font-mono text-sky-700 underline-offset-2 hover:underline" to={`/ideas/${strategy.root}`}>
          {strategy.root}
        </Link>
      </p>
    )
  }
  if (strategy.active_variants.length === 0) return null
  return (
    <p className="text-sm text-slate-700">
      активные варианты этой стратегии ({strategy.active_variants.length}):{' '}
      {strategy.active_variants.map((id, i) => (
        <span key={id}>
          {i > 0 && ', '}
          <Link className="font-mono text-sky-700 underline-offset-2 hover:underline" to={`/ideas/${id}`}>
            {id}
          </Link>
        </span>
      ))}
    </p>
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
