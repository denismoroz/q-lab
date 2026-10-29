import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

// docs/YEARS.md: the judged part of a run, year by year. A description, read by
// no rule: the numbers come from the API's `year_view` (regrouped from the trial's
// stored metrics), nothing is computed here beyond turning fractions into percent.

const POSITIVE = '#10b981'
const NEGATIVE = '#e11d48'
const SHORT = 0.45 // opacity of a year too short to be counted (under 120 days)

const percent = (v, digits = 1) => (typeof v === 'number' ? `${(v * 100).toFixed(digits)}%` : '—')

function yearLabel(y) {
  return y.partial ? `${y.year} (${y.days} дн.)` : String(y.year)
}

function YearTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const y = payload[0].payload
  return (
    <div className="rounded border border-slate-200 bg-white p-2 text-xs shadow">
      <div className="font-medium text-slate-900">{yearLabel(y)}</div>
      <div>доходность за год: {percent(y.return)}</div>
      <div>худшее падение внутри года: {percent(y.max_dd)}</div>
      <div>Шарп года: {typeof y.sharpe === 'number' ? y.sharpe.toFixed(2) : 'мало дней'}</div>
      {!y.counted && <div className="text-slate-500">меньше 120 дней — в сводках не участвует</div>}
    </div>
  )
}

// The four signs А Б В Г (docs/YEARS.md, rules 2026-10-28.2): each says whether the
// result holds against its line, or that it is not defined for this run and why. They
// are recorded to collect statistics and decide nothing.
const SIGN_STATE = { holds: 'держится', fails: 'не держится', 'not defined': 'не определено' }

// docs/CAPITAL_BASE.md: the years are read on the strategy's own money (owner, 2026-10-29: the
// leading figure everywhere is the strategy's own money's), a share of the capital it started with,
// summed when it does not reinvest its profit. The stand's column is a share of a book whose
// balance has grown with money set aside (Bv2), compounded; it stays as a small table beside them.
function StandYears({ view }) {
  const capital = view.capital
  if (!capital?.years?.length) return null
  const stand = new Map(view.years.map((y) => [y.year, y]))
  const differs = capital.years.some((y) => {
    const s = stand.get(y.year)
    return !s || Math.abs(s.return - y.return) > 0.0005
  })
  if (!differs) return null
  return (
    <div className="mt-2" data-testid="year-view-stand">
      <div className="text-xs text-slate-500">как считает стенд (от баланса книги):</div>
      <table className="mt-1 text-xs tabular-nums text-slate-500">
        <tbody>
          {capital.years.map((y) => (
            <tr key={y.year}>
              <td className="pr-4">{yearLabel(y)}</td>
              <td className="pr-4">на деньгах стратегии {percent(y.return)}</td>
              <td>у стенда {percent(stand.get(y.year)?.return)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function YearView({ view, compact = false }) {
  if (!view?.years?.length) return null
  // The strategy's own money's years lead; a run that has none shows the stand's, and says so.
  const onCapital = Boolean(view.capital?.years?.length)
  const main = onCapital ? view.capital : view
  const summary = main.summary ?? {}
  const rows = [
    summary.years_counted != null
      ? ['прибыльных лет', `${summary.years_positive} из ${summary.years_counted}`]
      : null,
    typeof summary.ann_return_net_ex_best_year === 'number'
      ? [
          `доходность в год без лучшего года${main.best_year ? ` (${main.best_year})` : ''}`,
          percent(summary.ann_return_net_ex_best_year),
        ]
      : null,
    typeof summary.best_year_share_of_profit === 'number'
      ? ['доля лучшего года во всей прибыли', percent(summary.best_year_share_of_profit, 0)]
      : null,
  ].filter(Boolean)
  const signs = main.signs ?? []
  const data = main.years.map((y) => ({ ...y, label: yearLabel(y), pct: y.return * 100 }))
  const basis = onCapital
    ? view.capital.reinvested === false
      ? 'на деньгах стратегии, прибыль не вкладывается обратно, годы складываются'
      : 'на деньгах стратегии, со сложным процентом'
    : 'по счёту стенда: у прогона нет лет на деньгах стратегии'

  return (
    <div className="mt-3" data-testid="year-view">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        по календарным годам — {basis}; для сведения, на вердикт не влияет
      </div>
      <div style={{ width: '100%', height: compact ? 130 : 170 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 11 }} interval={0} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} width={48} />
            <Tooltip content={<YearTooltip />} cursor={{ fill: 'rgba(148,163,184,0.15)' }} />
            <ReferenceLine y={0} stroke="#64748b" />
            <Bar dataKey="pct" isAnimationActive={false}>
              {data.map((y) => (
                <Cell
                  key={y.year}
                  fill={y.return >= 0 ? POSITIVE : NEGATIVE}
                  fillOpacity={y.counted ? 1 : SHORT}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {rows.length > 0 && (
        <dl className="mt-1 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          {rows.map(([name, value]) => (
            <div key={name} className="flex justify-between gap-4">
              <dt className="text-slate-500">{name}</dt>
              <dd className="tabular-nums font-medium">{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {signs.length > 0 && (
        <ul className="mt-1 text-xs text-slate-500" data-testid="year-signs">
          {signs.map((s) => (
            <li key={s.id}>
              {s.letter}: {s.title} — {SIGN_STATE[s.state] ?? s.state}
              {s.holds == null ? (s.reason ? ` (${s.reason})` : '') : ` (${percent(s.value)})`}
            </li>
          ))}
        </ul>
      )}
      <StandYears view={view} />
      <p className="mt-1 text-xs text-slate-400">
        Столбец — доходность за календарный год после комиссий (неполный год — за его дни, бледный:
        меньше 120 дней, в сводках не участвует). «Без лучшего года» — доходность в год, если
        выбросить все дни самого удачного года.
      </p>
    </div>
  )
}
