import { useState } from 'react'

// «Что есть на странице: N из 14» (qlab.idea_standard, docs/IDEA_STANDARD.md). The list is checked
// by code, not by a person; the panel names only what is missing, the full list is one click away.

export const STATE_STYLE = {
  есть: 'bg-emerald-100 text-emerald-900',
  нет: 'bg-rose-100 text-rose-900',
  'не нужно': 'bg-slate-100 text-slate-600',
}

const MARK = { есть: '✓', нет: '✕', 'не нужно': '–' }

function ItemRow({ item }) {
  const words = item.reason ?? item.detail
  return (
    <li className="flex gap-2 py-0.5 text-sm">
      <span
        className={`mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded text-xs ${STATE_STYLE[item.state]}`}
        title={item.state}
      >
        {MARK[item.state]}
      </span>
      <span className="text-slate-800">
        <span className="font-medium">{item.number}. {item.short}.</span>{' '}
        <span className={item.state === 'нет' ? 'text-rose-800' : 'text-slate-600'}>
          {words ?? (item.state === 'есть' ? item.title : '')}
        </span>
      </span>
    </li>
  )
}

export default function StandardPanel({ standard }) {
  const [open, setOpen] = useState(false)
  if (!standard) return null
  const missing = standard.items.filter((i) => i.state === 'нет')
  const tone = missing.length === 0 ? 'border-emerald-300 bg-emerald-50' : 'border-amber-300 bg-amber-50'
  return (
    <section className={`rounded border p-3 ${tone}`} data-testid="standard-panel">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <h2 className="text-sm font-semibold text-slate-900">
          Что есть на странице: {standard.have} из {standard.total}
        </h2>
        <span className="text-xs text-slate-500">
          список проверяет код, одинаковый для всех стратегий
        </span>
      </div>
      {missing.length === 0 ? (
        <p className="mt-1 text-sm text-emerald-900">Всё, что положено, на странице есть.</p>
      ) : (
        <>
          <p className="mt-1 text-xs text-slate-600">Не хватает:</p>
          <ul className="mt-1" data-testid="standard-missing">
            {missing.map((i) => (
              <ItemRow key={i.number} item={i} />
            ))}
          </ul>
        </>
      )}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="mt-2 text-xs text-sky-700 underline-offset-2 hover:underline"
      >
        {open ? 'скрыть полный список' : 'показать полный список'}
      </button>
      {open && (
        <ul className="mt-2 border-t border-amber-200 pt-2" data-testid="standard-full">
          {standard.items.map((i) => (
            <ItemRow key={i.number} item={i} />
          ))}
        </ul>
      )}
    </section>
  )
}
