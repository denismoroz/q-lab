import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchForward } from '../api.js'
import { Panel } from './ui.jsx'
import { SHORT_FORWARD_DAYS, daysText } from '../forwardReturn.js'

// docs/FORWARD_CHART.md: how a passed idea did on its forward test, one chart per stage.
// The numbers come from the API (qlab.forward_series: the stored daily money of the run the
// verdict rests on); nothing is decided or recomputed here, only worded. The percent is of
// the strategy's own capital, never of the account balance.

const STRATEGY = '#10b981'
const HOLD = '#94a3b8'

const signed = (v, digits = 1) =>
  typeof v === 'number' ? `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(digits)}%` : '—'

const STAGE_TITLE = {
  1: 'Этап 1 — длинная история',
  2: 'Этап 2 — Hyperliquid',
}

function stageTitle(stage, twoStage) {
  const venue = stage.venue ? `, ${stage.venue}` : ''
  if (!twoStage) return `Проверка вперёд${venue}`
  return stage.stage === 1 ? `${STAGE_TITLE[1]}${venue}` : STAGE_TITLE[2]
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

function Caption({ stage, title, series }) {
  const short = series.days < SHORT_FORWARD_DAYS
  const coins = (series.hold_coins ?? []).map((c) => c.replace('-SPOT', '')).join(', ')
  return (
    <div className="mt-1 space-y-1 text-sm text-slate-700">
      <p>
        <span className="font-medium">{title}</span> · {series.start} — {series.end} ·{' '}
        {daysText(series.days)} · итог{' '}
        <span className="font-semibold">{signed(series.total_return)}</span> от денег стратегии
        {short ? (
          <span className="text-slate-500">
            {' '}
            (в год не считаем: отрезок короче {SHORT_FORWARD_DAYS} дней)
          </span>
        ) : (
          <>
            , в год <span className="font-semibold">{signed(series.ann_return)}</span>
          </>
        )}
        {typeof series.max_dd === 'number' && <>, худшее падение {signed(series.max_dd)}</>}
      </p>
      {series.base === 'strategy_capital' && series.reinvested === false && (
        <p className="text-xs text-slate-500">
          Эта стратегия не вкладывает прибыль обратно в позиции, поэтому дни складываются, а не
          склеиваются, и проценты считаются от денег, с которых она начала
          (docs/CAPITAL_BASE.md).
          {typeof series.stand_total_return === 'number' &&
            ` Как считал стенд раньше, от баланса книги со всеми отложенными деньгами: ${signed(series.stand_total_return)}.`}
        </p>
      )}
      {typeof series.hold_total === 'number' && (
        <p className="text-slate-600">
          Если просто держать те же монеты{coins ? ` (${coins})` : ''} те же дни: {signed(series.hold_total)}
          {!short && typeof series.hold_ann === 'number' ? `, в год ${signed(series.hold_ann)}` : ''}.
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
  const data = series.points.map(([date, strategy, hold]) => ({
    date,
    strategy: strategy * 100,
    hold: typeof hold === 'number' ? hold * 100 : null,
  }))
  return (
    <div className="rounded border border-slate-200 p-3" data-testid="forward-chart">
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={48} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} width={48} />
            <Tooltip content={<ChartTooltip />} />
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
      <Caption stage={stage} title={title} series={series} />
    </div>
  )
}

export default function ForwardCharts({ ideaId }) {
  const query = useQuery({ queryKey: ['idea', ideaId, 'forward'], queryFn: () => fetchForward(ideaId) })
  // The hold line often swings far wider than the strategy and flattens it: it can be hidden.
  const [showHold, setShowHold] = useState(true)
  const view = query.data
  // Nothing is drawn for an idea that did not pass; the page says so elsewhere.
  if (!view?.passed) return null
  return (
    <Panel
      title="Как отработала проверка вперёд"
      subtitle="Деньги стратегии день за днём на данных, которых при подборе параметров не было: процент от денег стратегии, не от остатка на счёте. Зелёная линия — стратегия после комиссий, серая пунктирная — просто держать те же монеты."
    >
      <label className="mb-2 flex items-center gap-2 text-xs text-slate-600">
        <input type="checkbox" checked={showHold} onChange={(e) => setShowHold(e.target.checked)} />
        показывать серую линию «просто держать монеты»
      </label>
      <div className="space-y-3">
        {view.stages.map((stage) => (
          <StageChart key={stage.stage} stage={stage} twoStage={view.two_stage} showHold={showHold} />
        ))}
        {!view.two_stage && (
          <p className="text-sm text-slate-500">
            Второго этапа у этой идеи нет, поэтому график один.
          </p>
        )}
      </div>
    </Panel>
  )
}
