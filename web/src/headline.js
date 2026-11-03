// The numbers the console leads with are the run on the strategy's own money
// (docs/CAPITAL_BASE.md; owner, 2026-10-29: «было бы здорово пофиксить это везде»). The API
// gives them as `trial.headline` (qlab.api.serializers.headline_json: copied from the stored
// metrics, nothing is computed here). The stand's share of the book's balance stays as one small
// line, only where it says something else. Pure functions, no imports.

const fraction = (v, digits = 2) => (typeof v === 'number' ? `${(v * 100).toFixed(digits)}%` : null)

// «от денег стратегии, прибыль не вкладывается обратно, считается суммой» / «по счёту стенда …».
export function basisText(headline) {
  if (!headline) return null
  if (headline.base_refused) {
    return 'основа процентов не определена (в книге пополнения): чисел на деньгах стратегии нет'
  }
  if (!headline.on_capital) return 'по счёту стенда: у прогона нет чисел на деньгах стратегии'
  return headline.reinvested === false
    ? 'от денег стратегии, прибыль не вкладывается обратно, считается суммой'
    : 'от денег стратегии, прибыль вкладывается, считается со сложным процентом'
}

// The stand's own figures as one small line, or null when they say the same thing.
export function standLine(headline) {
  if (!headline?.on_capital || !headline.differs || !headline.stand) return null
  const s = headline.stand
  const parts = []
  if (typeof s.ann_return === 'number') parts.push(`в год ${fraction(s.ann_return)}`)
  if (typeof s.sharpe === 'number') parts.push(`Шарп ${s.sharpe.toFixed(2)}`)
  if (typeof s.max_dd === 'number') parts.push(`просадка ${fraction(s.max_dd)}`)
  if (typeof s.noise_percentile === 'number') parts.push(`обгоняет шум ${fraction(s.noise_percentile)}`)
  return parts.length ? `Как считает стенд (от баланса книги): ${parts.join(', ')}` : null
}

// Keys of the regime split: the strategy's own money's when the run has it.
export function regimePrefix(metrics) {
  return typeof metrics?.cap_regime_bull_share === 'number' ? 'cap_regime_' : 'regime_'
}

// The noise bar's power on the base the rules read.
export function powerOf(metrics, headline) {
  const m = metrics ?? {}
  return headline?.on_capital && typeof m.noise_cap_power === 'number' ? m.noise_cap_power : m.noise_power
}

// «Заработок отличим от нуля» (docs/NOISE_BAR.md, stand 2026-11-03.1): the share of resampled
// average days at or below zero (p) and the 95% interval of the yearly return, on the strategy's
// own money. Copied from the stored metrics; it decides nothing. Two rows beside the noise bar, or
// an empty list for a run made before the reading existed.
export function earningRows(metrics) {
  const m = metrics ?? {}
  if (typeof m.cap_mean_positive_p !== 'number') return []
  const signed = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)}%`
  const p = m.cap_mean_positive_p
  const rows = [
    [
      'заработок отличим от нуля: доля пересчётов со средним днём не выше нуля (p ≤ 0.05 — отличим)',
      p <= 0 ? 'меньше 1 из 2000' : p.toFixed(3),
    ],
  ]
  if (typeof m.cap_ann_ci_low === 'number' && typeof m.cap_ann_ci_high === 'number') {
    rows.push([
      '95% интервал дохода в год на деньгах стратегии',
      `от ${signed(m.cap_ann_ci_low)} до ${signed(m.cap_ann_ci_high)}`,
    ])
  }
  return rows
}
