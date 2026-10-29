// docs/PROTECTION.md: how well a Bv2-family run protects. Four readings, each a number
// against a line declared up front in the ruleset (rules 2026-10-29.1), shown as "holds /
// does not hold / not defined". They are recorded to collect statistics and decide nothing;
// the numbers come from the API's `protection` (regrouped from the trial's stored metrics),
// nothing is computed here beyond turning fractions into text.

const STATE = {
  holds: 'держится',
  fails: 'не держится',
  'not defined': 'не определено',
  number: 'число',
}
const TONE = {
  holds: 'text-emerald-700',
  fails: 'text-rose-700',
  'not defined': 'text-slate-400',
  number: 'text-slate-700',
}

const percent = (v, digits = 1) =>
  typeof v === 'number' ? `${(v * 100).toFixed(digits).replace('-', '−')}%` : '—'
const ratio = (v) => (typeof v === 'number' ? v.toFixed(2).replace('-', '−') : '—')

function reading(r) {
  if (r.value == null) return r.reason ? ` (${r.reason})` : ''
  const shown = r.id === 'protection_no_loss' ? percent(r.value) : ratio(r.value)
  return r.comparator ? ` (${shown}; граница ${r.comparator} ${r.threshold})` : ` (${shown})`
}

export default function Protection({ view, compact = false }) {
  if (!view?.readings?.length) return null
  const n = view.numbers ?? {}
  const rows = [
    ['полных недель в отрезке', typeof n.prot_weeks === 'number' ? Math.round(n.prot_weeks) : null],
    [
      'недель падения / роста монет',
      typeof n.prot_fall_weeks === 'number'
        ? `${Math.round(n.prot_fall_weeks)} / ${Math.round(n.prot_rise_weeks)}`
        : null,
    ],
    ['худшее падение стратегии', typeof n.prot_worst_fall === 'number' ? percent(n.prot_worst_fall) : null],
    [
      'худшее падение тех же монет без защиты',
      typeof n.prot_long_worst_fall === 'number' ? percent(n.prot_long_worst_fall) : null,
    ],
    ['итог стратегии за отрезок', typeof n.prot_total_return === 'number' ? percent(n.prot_total_return) : null],
    [
      'итог тех же монет без защиты',
      typeof n.prot_long_total_return === 'number' ? percent(n.prot_long_total_return) : null,
    ],
  ].filter(([, v]) => v !== null)
  return (
    <div className="mt-3" data-testid="protection">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        защита: насколько стратегия страхует монеты — для сведения, на вердикт не влияет
      </div>
      <ul className="mt-1 text-sm" data-testid="protection-readings">
        {view.readings.map((r) => (
          <li key={r.id}>
            <span className="font-medium">{r.letter}</span>: {r.title} —{' '}
            <span className={TONE[r.state] ?? ''}>{STATE[r.state] ?? r.state}</span>
            <span className="text-xs text-slate-500">{reading(r)}</span>
          </li>
        ))}
      </ul>
      {!compact && rows.length > 0 && (
        <dl className="mt-1 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          {rows.map(([name, value]) => (
            <div key={name} className="flex justify-between gap-4">
              <dt className="text-slate-500">{name}</dt>
              <dd className="tabular-nums font-medium">{value}</dd>
            </div>
          ))}
        </dl>
      )}
      <p className="mt-1 text-xs text-slate-400">
        «Те же монеты без защиты» — длинная часть самой книги (её монеты в тех же долларах, без
        шортов, комиссий и фандинга). Деньги стратегии — как в расчёте доходности; недели —
        целые календарные, не меньше {view.min_weeks ?? 8} для любого показания.
      </p>
    </div>
  )
}
