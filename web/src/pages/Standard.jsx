import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchStandard } from '../api.js'
import { Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import { FAMILY, label } from '../labels.js'

// Owner, 2026-11-04: «гдето они есть гдето нет - разберись пожалуйста с это кашей ... прогони все
// стратегии под одну гребенку». The table: ideas by the fourteen things every strategy page has
// (qlab.idea_standard, docs/IDEA_STANDARD.md). Green: it is there; red: it is missing, the reason
// is in the hover; grey: not needed for this idea, the reason is in the hover.

const CELL = {
  есть: 'bg-emerald-200 text-emerald-900',
  нет: 'bg-rose-300 text-rose-950',
  'не нужно': 'bg-slate-100 text-slate-400',
}
const MARK = { есть: '✓', нет: '✕', 'не нужно': '·' }

const chip =
  'rounded-full px-2.5 py-1 text-xs ring-1 ring-inset focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400'
const chipOff = 'bg-white text-slate-600 ring-slate-300 hover:bg-slate-50'
const chipOn = 'bg-indigo-50 font-semibold text-indigo-900 ring-2 ring-indigo-400'
const chipAll = 'bg-indigo-600 text-white ring-indigo-600'

function readFamilies(params) {
  return new Set((params.get('family') ?? '').split(',').filter(Boolean))
}

function FamilyFilter({ rows, chosen, onChange }) {
  const counts = new Map()
  for (const r of rows) counts.set(r.family, (counts.get(r.family) ?? 0) + 1)
  const present = Object.keys(FAMILY).filter((f) => counts.has(f) || chosen.has(f))
  const toggle = (f) => {
    const next = new Set(chosen)
    if (next.has(f)) next.delete(f)
    else next.add(f)
    onChange(next)
  }
  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Семейство">
      <span className="w-20 shrink-0 text-xs uppercase tracking-wide text-slate-400">Семейство</span>
      <button
        type="button"
        onClick={() => onChange(new Set())}
        aria-pressed={chosen.size === 0}
        className={`${chip} ${chosen.size === 0 ? chipAll : chipOff}`}
      >
        все · {rows.length}
      </button>
      {present.map((f) => (
        <button
          key={f}
          type="button"
          onClick={() => toggle(f)}
          aria-pressed={chosen.has(f)}
          className={`${chip} ${chosen.has(f) ? chipOn : chipOff}`}
        >
          {label(FAMILY, f)} · {counts.get(f) ?? 0}
        </button>
      ))}
    </div>
  )
}

function Cell({ item, name }) {
  const words = item.text ? `${item.text}` : item.state
  return (
    <td
      title={`${name}: ${item.state}${item.text ? ` — ${item.text}` : ''}`}
      className={`h-7 w-12 min-w-12 cursor-help border border-white text-center text-xs ${CELL[item.state]}`}
      data-state={item.state}
      aria-label={words}
    >
      {MARK[item.state]}
    </td>
  )
}

function TotalsRow({ rows, items }) {
  const count = (n, state) => rows.filter((r) => r.items[n - 1].state === state).length
  return (
    <tr className="bg-slate-50 font-semibold" data-testid="standard-totals">
      <td className="sticky left-0 bg-slate-50 px-2 py-1.5 text-xs text-slate-700">
        Нет на странице, идей
      </td>
      <td />
      {items.map((d) => {
        const missing = count(d.number, 'нет')
        return (
          <td
            key={d.number}
            title={`${d.title}: нет у ${missing}, не нужно у ${count(d.number, 'не нужно')}`}
            className={`border border-white text-center text-xs ${
              missing ? 'text-rose-800' : 'text-emerald-800'
            }`}
          >
            {missing}
          </td>
        )
      })}
      <td />
    </tr>
  )
}

export default function Standard() {
  const [params, setParams] = useSearchParams()
  const query = useQuery({ queryKey: ['standard'], queryFn: fetchStandard })
  const chosen = readFamilies(params)
  const setFamilies = (next) => {
    const out = {}
    if (next.size) out.family = [...next].join(',')
    setParams(out, { replace: true })
  }

  return (
    <div className="space-y-4">
      <Panel
        title="Стандарт страницы стратегии"
        subtitle="У каждой стратегии должны быть одни и те же четырнадцать вещей; код проверяет, что есть, а чего нет, и почему."
      >
        <p className="text-sm text-slate-700">
          Зелёное — есть, красное — не хватает (причина — в подсказке при наведении), серое — этой
          идее не нужно. Строка — идея, столбец — пункт; название пункта — в подсказке над номером.
          Убранные идеи и книги для калибровки шума не проверяются; второй этап идеи (Hyperliquid)
          проверяется вместе с самой идеей.
        </p>
      </Panel>
      <QueryState query={query}>
        {query.data && <Table data={query.data} chosen={chosen} onFamilies={setFamilies} />}
      </QueryState>
    </div>
  )
}

function Table({ data, chosen, onFamilies }) {
  const rows = data.rows.filter((r) => chosen.size === 0 || chosen.has(r.family))
  return (
    <div className="space-y-3">
      <FamilyFilter rows={data.rows} chosen={chosen} onChange={onFamilies} />
      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
        <table className="w-full border-collapse text-sm" data-testid="standard-table">
          <thead>
            <tr className="bg-slate-100 text-xs text-slate-600">
              <th className="sticky left-0 bg-slate-100 px-2 py-2 text-left">Идея</th>
              <th className="px-2 py-2 text-left">Статус</th>
              {data.items.map((d) => (
                <th key={d.number} title={d.title} className="min-w-12 px-1 py-2 text-center">
                  <div className="font-semibold">{d.number}</div>
                  <div className="font-normal">{d.short}</div>
                </th>
              ))}
              <th className="px-2 py-2 text-right">Есть</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="hover:bg-slate-50">
                <td className="sticky left-0 max-w-xs truncate bg-white px-2 py-1" title={r.title}>
                  <Link
                    className="text-sky-700 underline-offset-2 hover:underline"
                    to={`/ideas/${r.id}`}
                  >
                    {r.id}
                  </Link>
                </td>
                <td className="px-2 py-1">
                  <StatusBadge status={r.status} />
                </td>
                {r.items.map((item, k) => (
                  <Cell key={item.number} item={item} name={data.items[k].title} />
                ))}
                <td className="px-2 py-1 text-right text-xs tabular-nums text-slate-600">
                  {r.have} из {data.items.length}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={data.items.length + 3} className="p-4 text-sm text-slate-500">
                  В этом семействе идей нет.
                </td>
              </tr>
            )}
          </tbody>
          <tfoot>
            <TotalsRow rows={rows} items={data.items} />
          </tfoot>
        </table>
      </div>
      <Commonest totals={data.totals} rows={rows} />
    </div>
  )
}

// What is lacking most, in words: the commonest reason per item, over all ideas.
function Commonest({ totals }) {
  const lacking = totals.filter((t) => t.missing > 0)
  if (lacking.length === 0) return null
  return (
    <Panel title="Чего не хватает чаще всего" subtitle="По всем проверяемым идеям, не по выбранному семейству.">
      <ul className="space-y-1 text-sm">
        {lacking
          .slice()
          .sort((a, b) => b.missing - a.missing)
          .map((t) => (
            <li key={t.number}>
              <span className="font-medium">{t.number}. {t.short}</span> — нет у {t.missing}
              {t.common_reasons[0] ? `, чаще всего: ${t.common_reasons[0].reason}` : ''}
            </li>
          ))}
      </ul>
    </Panel>
  )
}
