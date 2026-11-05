import { Panel } from './ui.jsx'

// docs/COVERAGE.md; owner, 2026-11-05: «да, сделай замер обоими способами». A book whose coins
// appeared at different times counted two ways side by side: as the stand judges it (from the day
// every coin has history) and with each coin entering as it appears. A reading that decides
// nothing; every word and number here comes from the API's `coverage_compare`
// (qlab.coverage_compare), the page computes nothing.

const COLUMNS = [
  ['window', 'Окно'],
  ['cut', 'Последняя треть с'],
  ['ann_return', 'Доход в год'],
  ['max_dd', 'Худшее падение'],
  ['earning', 'Заработок'],
  ['incomplete', 'Книга неполная'],
]

export default function CoverageCompare({ block }) {
  if (!block?.rows?.length) return null
  return (
    <Panel title={block.title} subtitle={block.note}>
      <div data-testid="coverage-compare">
        <p className="text-sm text-slate-800" data-testid="coverage-compare-differs">
          {block.differs}
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm tabular-nums">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="pr-4 pb-1 font-normal">Способ</th>
                {COLUMNS.map(([key, title]) => (
                  <th key={key} className="pr-4 pb-1 font-normal">
                    {title}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row) => (
                <tr key={row.id} className="border-t border-slate-100 align-top" data-testid={`coverage-row-${row.id}`}>
                  <td className="py-1 pr-4 font-medium text-slate-800">{row.title}</td>
                  {row.error ? (
                    <td className="py-1 text-rose-700" colSpan={COLUMNS.length}>
                      не посчитано: {row.error}
                    </td>
                  ) : (
                    COLUMNS.map(([key]) => (
                      <td key={key} className="py-1 pr-4 text-slate-700">
                        {row[key] ?? '—'}
                      </td>
                    ))
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Доход и падение — на деньгах стратегии, за последнюю треть окна. Шум не считался: его место
          стенд определяет на своём запуске.
          {block.data_end ? ` Данные до ${block.data_end}.` : ''}
        </p>
        {block.knowable && (
          <p className="mt-1 text-xs text-slate-500" data-testid="coverage-compare-knowable">
            {block.knowable}
          </p>
        )}
      </div>
    </Panel>
  )
}
