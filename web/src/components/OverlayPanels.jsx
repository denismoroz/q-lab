import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchOverlays } from '../api.js'
import { IdeaLink, Panel } from './ui.jsx'

// docs/OVERLAYS.md: two panels next to the forward charts. «Надстройки» sets every overlay
// variant of a strategy against the strategy itself on the same days; «Когда входить» describes
// the strategy's forward series (from every entry day, whether a good period predicts the next,
// by market state). The numbers come from the API (qlab.overlay_compare, qlab.overlay_diagnostics);
// nothing is decided or recomputed here, only worded. Percent is of the strategy's own capital.

const pct = (v, digits = 1) =>
  typeof v === 'number' ? `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(digits)}%` : '—'
const num = (v, digits = 2) => (typeof v === 'number' ? v.toFixed(digits) : '—')

const VERDICT_STYLE = {
  helped: 'bg-emerald-50 text-emerald-800 ring-emerald-300',
  'not-helped': 'bg-rose-50 text-rose-800 ring-rose-300',
  indistinguishable: 'bg-slate-100 text-slate-700 ring-slate-300',
}

function Verdict({ comparison }) {
  return (
    <span
      className={`inline-block whitespace-nowrap rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
        VERDICT_STYLE[comparison.verdict]
      }`}
    >
      {comparison.verdict_ru}
    </span>
  )
}

function VariantRow({ row, current }) {
  const c = row.comparison
  const cell = 'px-2 py-1.5 align-top tabular-nums'
  return (
    <tr className={`border-t border-slate-100 ${current ? 'bg-sky-50' : ''}`}>
      <td className="px-2 py-1.5 align-top">
        <IdeaLink id={row.idea_id}>{row.name}</IdeaLink>
        {row.route_ru && <div className="text-xs text-slate-500">стенд: {row.route_ru}</div>}
      </td>
      {c ? (
        <>
          <td className={cell}>{c.days}</td>
          <td className={cell}>
            {pct(c.base.ann_return)} → {pct(c.variant.ann_return)}
          </td>
          <td className={cell}>
            {pct(c.base.max_dd)} → {pct(c.variant.max_dd)}
          </td>
          <td className={cell}>
            {num(c.base.sharpe)} → {num(c.variant.sharpe)}
          </td>
          <td className={cell}>
            {c.delta_sharpe >= 0 ? '+' : '−'}
            {Math.abs(c.delta_sharpe).toFixed(2)}{' '}
            <span className="text-xs text-slate-500">
              ({num(c.delta_sharpe_ci[0])} … {num(c.delta_sharpe_ci[1])})
            </span>
          </td>
          <td className={cell}>{num(c.corrected_p, 3)}</td>
          <td className="px-2 py-1.5 align-top">
            <Verdict comparison={c} />
          </td>
        </>
      ) : (
        <td colSpan={7} className="px-2 py-1.5 text-sm italic text-slate-500">
          прогнана на стенде, с базой ещё не сравнена
        </td>
      )}
    </tr>
  )
}

function OverlayTable({ view }) {
  const { table, role, idea_id: ideaId, base_id: baseId } = view
  return (
    <Panel
      title="Надстройки"
      subtitle="Надстройка меняет, когда и сколько стратегия держит: не входить в одном из состояний рынка, уменьшать размер при плохих или неспокойных результатах, входить и выходить частями. Каждый вариант — надстройка или пара надстроек, числа которых раз в месяц выбирает код только по прошлому; сравнение с самой стратегией — на одних и тех же днях."
    >
      {role === 'variant' && (
        <p className="mb-2 text-sm text-slate-700">
          Это вариант стратегии{' '}
          <Link className="text-sky-700 hover:underline" to={`/ideas/${baseId}`}>
            {baseId}
          </Link>
          ; ниже он и его соседи.
        </p>
      )}
      <p className="mb-2 text-sm text-slate-700">
        Сравнено вариантов: {table.family_size}; помогла {table.counts.helped}, не помогла{' '}
        {table.counts['not-helped']}, неотличимо от случайности {table.counts.indistinguishable}.
        Испытаний в семействе стратегии: {table.family_trials} (дефляция считает их все). Вариантов
        много, поэтому каждое p умножено на их число: один из двадцати случайных успехов иначе
        проходил бы за открытие.
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm" data-testid="overlay-table">
          <thead className="text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-2 py-1">Надстройка</th>
              <th className="px-2 py-1">Дней</th>
              <th className="px-2 py-1">В год: база → с надстройкой</th>
              <th className="px-2 py-1">Худшее падение</th>
              <th className="px-2 py-1">Шарп</th>
              <th className="px-2 py-1">Разница Шарпа (95%)</th>
              <th className="px-2 py-1">p с поправкой</th>
              <th className="px-2 py-1">Вердикт</th>
            </tr>
          </thead>
          <tbody>
            {table.variants.map((row) => (
              <VariantRow key={row.idea_id} row={row} current={row.idea_id === ideaId} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        «помогла» — разница Шарпа положительна, а p с поправкой меньше 5%; «не помогла» — достоверно
        хуже; всё остальное, а также отрезок короче года — «неотличимо от случайности».
      </p>
    </Panel>
  )
}

function Cell({ children }) {
  return <td className="px-2 py-1 tabular-nums">{children}</td>
}

function Diagnostics({ d }) {
  const states = d.market_state
  const names = { bull: 'рост', flat: 'боковик', bear: 'падение', high: 'высокая изменчивость', low: 'низкая изменчивость', unknown: 'состояние ещё не известно' }
  const periods = { week: 'недели', month: 'месяцы', quarter: 'кварталы' }
  return (
    <Panel
      title="Когда входить: описание проверки вперёд"
      subtitle={`Отрезок ${d.window.start} — ${d.window.end}, ${d.window.days} дн. Это описание: оно ничего не решает, и ни одна надстройка из-за него не пропускается.`}
    >
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Войти в любой день и держать
      </h3>
      <div className="overflow-x-auto">
        <table className="mb-3 min-w-full text-left text-sm" data-testid="entry-days">
          <thead className="text-xs text-slate-500">
            <tr>
              <th className="px-2 py-1">Срок</th>
              <th className="px-2 py-1">Входов</th>
              <th className="px-2 py-1">Худший</th>
              <th className="px-2 py-1">Средний</th>
              <th className="px-2 py-1">Лучший</th>
              <th className="px-2 py-1">В плюсе</th>
              <th className="px-2 py-1">Просто держать монеты: худший … лучший</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(d.entry_days).map(([days, row]) => (
              <tr key={days} className="border-t border-slate-100">
                <Cell>{days} дн.</Cell>
                {row.strategy ? (
                  <>
                    <Cell>{row.strategy.starts}</Cell>
                    <Cell>{pct(row.strategy.worst)}</Cell>
                    <Cell>{pct(row.strategy.median)}</Cell>
                    <Cell>{pct(row.strategy.best)}</Cell>
                    <Cell>{(row.strategy.share_positive * 100).toFixed(0)}%</Cell>
                    <Cell>
                      {row.hold ? `${pct(row.hold.worst)} … ${pct(row.hold.best)}` : '—'}
                    </Cell>
                  </>
                ) : (
                  <td colSpan={6} className="px-2 py-1 italic text-slate-500">
                    отрезок короче срока
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Предсказывает ли хороший период следующий
      </h3>
      <ul className="mb-3 space-y-0.5 text-sm text-slate-700" data-testid="persistence">
        {Object.entries(d.persistence).map(([key, row]) => (
          <li key={key}>
            {periods[key]}:{' '}
            {row.correlation === null
              ? `не определено, пар периодов ${row.pairs} (нужно хотя бы 8)`
              : `корреляция ${row.correlation >= 0 ? '+' : '−'}${Math.abs(row.correlation).toFixed(2)} при ${row.pairs} парах, случайный разброс ±${row.noise_band.toFixed(2)} — ${
                  row.distinguishable ? 'связь заметна' : 'связи нет'
                }`}
          </li>
        ))}
      </ul>
      {states && (
        <>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            По состоянию рынка, известному на начало дня
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm" data-testid="market-state">
              <thead className="text-xs text-slate-500">
                <tr>
                  <th className="px-2 py-1">Состояние</th>
                  <th className="px-2 py-1">Дней</th>
                  <th className="px-2 py-1">В год</th>
                  <th className="px-2 py-1">Дней в плюсе</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['По направлению BTC за 30 дней', states.direction],
                  ['По изменчивости BTC', states.volatility],
                ].map(([title, group]) => (
                  <FragmentRows key={title} title={title} group={group} names={names} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  )
}

function FragmentRows({ title, group, names }) {
  return (
    <>
      <tr>
        <td colSpan={4} className="px-2 pt-2 text-xs font-medium text-slate-600">
          {title}
        </td>
      </tr>
      {Object.entries(group)
        .filter(([, v]) => v.days > 0)
        .map(([key, v]) => (
          <tr key={key} className="border-t border-slate-100">
            <Cell>{names[key]}</Cell>
            <Cell>{v.days}</Cell>
            <Cell>{pct(v.ann_return)}</Cell>
            <Cell>{typeof v.share_positive === 'number' ? `${(v.share_positive * 100).toFixed(0)}%` : '—'}</Cell>
          </tr>
        ))}
    </>
  )
}

export default function OverlayPanels({ ideaId }) {
  const query = useQuery({ queryKey: ['idea', ideaId, 'overlays'], queryFn: () => fetchOverlays(ideaId) })
  const view = query.data
  if (!view) return null
  const any = view.table.variants.length > 0
  if (!any && !view.diagnostics) return null
  return (
    <>
      {any && <OverlayTable view={view} />}
      {view.diagnostics && <Diagnostics d={view.diagnostics} />}
    </>
  )
}
