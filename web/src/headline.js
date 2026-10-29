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
