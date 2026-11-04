import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchForward } from '../api.js'
import { Panel } from './ui.jsx'
import { SHORT_FORWARD_DAYS, daysText } from '../forwardReturn.js'

// docs/FORWARD_CHART.md: the strategy's money over the whole data period, one chart per venue,
// for any outcome. The numbers come from the API (qlab.forward_series: the stored daily money of
// the run the verdict rests on, split into the parts the rules cut); nothing is decided or
// recomputed here, only worded. The percent is of the strategy's own capital, never of the
// account balance.

const STRATEGY = '#10b981'
const HOLD = '#94a3b8'
const SELECTION_FILL = '#f59e0b'
const FORWARD_FILL = '#3b82f6'

const signed = (v, digits = 1) =>
  typeof v === 'number' ? `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(digits)}%` : '—'

const PART_LABEL = {
  before: 'до подбора',
  warmup: 'разогрев',
  selection: 'подбор',
  forward: 'проверка вперёд',
}

const STAGE_TITLE = {
  1: 'Этап 1 — длинная история',
  2: 'Этап 2 — Hyperliquid',
}

export function stageTitle(stage, twoStage) {
  const venue = stage.venue ? `, ${stage.venue}` : ''
  if (!twoStage) return `Площадка${venue || ' не названа'}`
  return stage.stage === 1 ? `${STAGE_TITLE[1]}${venue}` : STAGE_TITLE[2]
}

// What a stage draws: the whole window when the file has it, else the forward part alone (a file
// stored before the whole window was kept).
export function chartOf(series) {
  const whole = series?.whole
  if (whole && whole.points?.length) {
    return { ...whole, wholeWindow: true, base: series.base, reinvested: series.reinvested,
      stand_total_return: series.stand_total_return }
  }
  return {
    start: series.start,
    end: series.end,
    days: series.days,
    points: series.points,
    total_return: series.total_return,
    ann_return: series.ann_return,
    max_dd: series.max_dd,
    hold_total: series.hold_total,
    hold_ann: series.hold_ann,
    hold_coins: series.hold_coins,
    parts: [],
    split_date: null,
    wholeWindow: false,
    base: series.base,
    reinvested: series.reinvested,
    stand_total_return: series.stand_total_return,
  }
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="rounded border border-slate-200 bg-white p-2 text-xs shadow">
      <div className="font-medium text-slate-900">{p.date}</div>
      <div>стратегия: {signed(p.strategy / 100)}</div>
      {typeof p.hold === 'number' && <div>просто держать: {signed(p.hold / 100)}</div>}
    </div>
  )
}

function PartLine({ label, part }) {
  if (!part) return null
  const short = part.days < SHORT_FORWARD_DAYS
  return (
    <p>
      <span className="font-medium">{label}</span> · {part.start} — {part.end} ·{' '}
      {daysText(part.days)} · итог <span className="font-semibold">{signed(part.total_return)}</span>
      {short ? (
        <span className="text-slate-500"> (в год не считаем: короче {SHORT_FORWARD_DAYS} дней)</span>
      ) : (
        <>
          , в год <span className="font-semibold">{signed(part.ann_return)}</span>
        </>
      )}
      {typeof part.max_dd === 'number' && <>, худшее падение {signed(part.max_dd)}</>}
      {typeof part.hold_total === 'number' && (
        <span className="text-slate-500"> · просто держать: {signed(part.hold_total)}</span>
      )}
    </p>
  )
}

function Caption({ stage, title, series, chart }) {
  const short = chart.days < SHORT_FORWARD_DAYS
  const names = (chart.hold_coins ?? []).map((c) => c.replace('-SPOT', ''))
  // A wide book holds hundreds of coins: name the first few, count the rest.
  const coins = names.length > 8 ? `${names.slice(0, 8).join(', ')} и ещё ${names.length - 8}` : names.join(', ')
  const byPart = Object.fromEntries((chart.parts ?? []).map((p) => [p.part, p]))
  return (
    <div className="mt-1 space-y-1 text-sm text-slate-700">
      <p>
        <span className="font-medium">{title}</span> ·{' '}
        {chart.wholeWindow ? 'весь период данных' : 'только проверка вперёд'}: {chart.start} —{' '}
        {chart.end} · {daysText(chart.days)} · итог{' '}
        <span className="font-semibold">{signed(chart.total_return)}</span>{' '}
        {series.base === 'strategy_capital' ? 'от денег стратегии' : 'по счёту стенда (от баланса книги)'}
        {short ? (
          <span className="text-slate-500">
            {' '}
            (в год не считаем: отрезок короче {SHORT_FORWARD_DAYS} дней)
          </span>
        ) : (
          <>
            , в год <span className="font-semibold">{signed(chart.ann_return)}</span>
          </>
        )}
        {typeof chart.max_dd === 'number' && <>, худшее падение {signed(chart.max_dd)}</>}
      </p>
      {chart.wholeWindow && (
        <div className="space-y-0.5 border-l-2 border-slate-200 pl-3">
          <PartLine label="Разогрев" part={byPart.warmup} />
          <PartLine label="До подбора" part={byPart.before} />
          <PartLine label="Подбор" part={byPart.selection} />
          <PartLine label="Проверка вперёд" part={byPart.forward} />
          {byPart.selection?.chosen_in_sample && (
            <p className="text-xs text-amber-700" data-testid="chosen-in-sample-note">
              На подборе показаны числа, которые код выбрал, глядя именно на этот отрезок, — они
              приукрашены; честная часть — проверка вперёд. Выбранное значение:{' '}
              <span className="font-mono">{byPart.selection.chosen}</span>.
            </p>
          )}
          {(chart.parts ?? []).length === 0 && (
            <p className="text-slate-500">
              В правилах этого прогона деления на подбор и проверку вперёд не было.
            </p>
          )}
        </div>
      )}
      {series.base === 'strategy_capital' && series.reinvested === false && (
        <p className="text-xs text-slate-500">
          Эта стратегия не вкладывает прибыль обратно в позиции, поэтому дни складываются, а не
          склеиваются, и проценты считаются от денег, с которых она начала
          (docs/CAPITAL_BASE.md).
          {typeof series.stand_total_return === 'number' &&
            ` Как считал стенд раньше, от баланса книги со всеми отложенными деньгами (проверка вперёд): ${signed(series.stand_total_return)}.`}
        </p>
      )}
      {typeof chart.hold_total === 'number' && (
        <p className="text-slate-600">
          Если просто держать те же монеты{coins ? ` (${coins})` : ''} те же дни:{' '}
          {signed(chart.hold_total)}
          {!short && typeof chart.hold_ann === 'number' ? `, в год ${signed(chart.hold_ann)}` : ''}.
        </p>
      )}
      {series.matches_trial === false && (
        <p className="text-amber-700">
          Ряд пересчитан заново и не сходится с итогом, записанным в прогоне #{stage.trial_id}
          (код стратегии с тех пор менялся): цифры графика надёжны хуже обычного.
        </p>
      )}
      <p className="text-xs text-slate-400">
        прогон #{stage.trial_id} · {stage.idea_id}
        {series.built_by === 'rebuild' ? ' · ряд построен повторным расчётом по записанным данным' : ''}
      </p>
    </div>
  )
}

function StageChart({ stage, twoStage, showHold }) {
  const title = stageTitle(stage, twoStage)
  const series = stage.series
  if (!series) {
    return (
      <div className="rounded border border-dashed border-slate-300 p-3" data-testid="forward-missing">
        <div className="text-sm font-medium text-slate-800">{title}</div>
        <p className="mt-1 text-sm text-slate-500">{stage.message}</p>
      </div>
    )
  }
  const chart = chartOf(series)
  const data = chart.points.map(([date, strategy, hold]) => ({
    date,
    strategy: strategy * 100,
    hold: typeof hold === 'number' ? hold * 100 : null,
  }))
  const byPart = Object.fromEntries((chart.parts ?? []).map((p) => [p.part, p]))
  return (
    <div className="rounded border border-slate-200 p-3" data-testid="forward-chart">
      {stage.window === 'forward_only' && (
        <p className="mb-2 text-xs text-amber-700" data-testid="forward-only-note">
          {stage.message}
        </p>
      )}
      <div style={{ width: '100%', height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={48} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} width={48} />
            <Tooltip content={<ChartTooltip />} />
            {byPart.selection && (
              <ReferenceArea
                x1={byPart.selection.start}
                x2={byPart.selection.end}
                fill={SELECTION_FILL}
                fillOpacity={0.12}
                label={{ value: 'подбор', position: 'insideTopLeft', fontSize: 11, fill: '#b45309' }}
              />
            )}
            {byPart.forward && (
              <ReferenceArea
                x1={byPart.forward.start}
                x2={byPart.forward.end}
                fill={FORWARD_FILL}
                fillOpacity={0.1}
                label={{ value: 'проверка вперёд', position: 'insideTopRight', fontSize: 11, fill: '#1d4ed8' }}
              />
            )}
            {chart.split_date && (
              <ReferenceLine
                x={chart.split_date}
                stroke="#1d4ed8"
                strokeDasharray="5 3"
                label={{ value: `с ${chart.split_date}`, position: 'insideBottomLeft', fontSize: 11, fill: '#1d4ed8' }}
              />
            )}
            <ReferenceLine y={0} stroke="#64748b" />
            {showHold && (
              <Line
                type="monotone"
                dataKey="hold"
                stroke={HOLD}
                strokeDasharray="4 3"
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            )}
            <Line
              type="monotone"
              dataKey="strategy"
              stroke={STRATEGY}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Caption stage={stage} title={title} series={series} chart={chart} />
    </div>
  )
}

export default function ForwardCharts({ ideaId }) {
  const query = useQuery({ queryKey: ['idea', ideaId, 'forward'], queryFn: () => fetchForward(ideaId) })
  // The hold line often swings far wider than the strategy and flattens it: it can be hidden.
  const [showHold, setShowHold] = useState(true)
  const view = query.data
  if (!view) return null
  // An idea the owner put away has no stages: the panel says why instead of staying silent.
  if (!view.stages?.length) {
    return view.reason ? (
      <Panel title="Деньги стратегии за весь период данных">
        <p className="text-sm text-slate-500">{view.reason}</p>
      </Panel>
    ) : null
  }
  return (
    <Panel
      title="Деньги стратегии за весь период данных"
      subtitle="Деньги стратегии день за днём, процент от денег стратегии, не от остатка на счёте; для любого исхода, и для отказа тоже. Зелёная линия — стратегия после комиссий, серая пунктирная — просто держать те же монеты."
    >
      <label className="mb-2 flex items-center gap-2 text-xs text-slate-600">
        <input type="checkbox" checked={showHold} onChange={(e) => setShowHold(e.target.checked)} />
        показывать серую линию «просто держать монеты»
      </label>
      <p className="mb-3 text-xs text-slate-500" data-testid="parts-legend">
        <span className="font-medium text-amber-700">Подбор</span> — первая часть данных (две
        трети): на ней выбирались числа стратегии, её результат ничего не доказывает.{' '}
        <span className="font-medium text-blue-700">Проверка вперёд</span> — последняя часть
        (треть): числа уже не менялись, и правила читают её.
      </p>
      <div className="space-y-3">
        {view.stages.map((stage) => (
          <StageChart key={stage.stage} stage={stage} twoStage={view.two_stage} showHold={showHold} />
        ))}
        {!view.two_stage && view.second_stage_message && (
          <p className="text-sm text-slate-500" data-testid="second-stage-note">
            {view.second_stage_message}
          </p>
        )}
      </div>
    </Panel>
  )
}
