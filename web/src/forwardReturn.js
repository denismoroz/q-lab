// Short forward tests are shown as their total over the period, not as a yearly
// figure (owner, 2026-10-25: «минус 34% в год» was 21 days annualized). Same
// conversion as the night report (qlab.night._report_row, qlab.paper_check).
// Pure functions, no imports: tested from src/qlab/test_paper_check.py.

export const SHORT_FORWARD_DAYS = 90

// null unless the run's own figures are a forward test shorter than
// SHORT_FORWARD_DAYS; else the total over its days and the yearly figure.
// On the strategy's own money (docs/CAPITAL_BASE.md) when the run has it: the yearly figure is
// `cap_ann_return`, and the total is linear in the days for a strategy that does not reinvest
// its profit (`cap_reinvested` 0), compounded otherwise -- the same as
// qlab.paper_check.forward_total_capital.
export function shortForwardReturn(metrics) {
  const m = metrics ?? {}
  const days = m.forward_days
  const onCapital = typeof m.cap_ann_return === 'number' && typeof m.cap_reinvested === 'number'
  const ann = onCapital ? m.cap_ann_return : m.ann_return_net
  if (m.judged_on_forward !== 1 || typeof days !== 'number' || typeof ann !== 'number') return null
  if (!(days > 0) || days >= SHORT_FORWARD_DAYS) return null
  if (onCapital && m.cap_reinvested === 0) {
    return { days: Math.round(days), total: (ann * days) / 365.25, yearly: ann }
  }
  const growth = 1 + ann
  const total = growth <= 0 ? -1 : Math.pow(growth, days / 365.25) - 1
  return { days: Math.round(days), total, yearly: ann }
}

// «1 день», «2 дня», «5 дней», «21 день».
export function daysText(n) {
  const last = n % 10
  const tens = n % 100
  if (last === 1 && tens !== 11) return `${n} день`
  if (last >= 2 && last <= 4 && (tens < 10 || tens >= 20)) return `${n} дня`
  return `${n} дней`
}
