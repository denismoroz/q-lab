import { Link } from 'react-router-dom'
import { STATUS_STYLE, IDEA_STATUS, label } from '../labels.js'

// Shared primitives. Deliberately plain: Tailwind defaults plus what
// legibility needs, nothing more.

export function Panel({ title, subtitle, children, className = '' }) {
  return (
    <section className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      {title && (
        <header className="border-b border-slate-200 px-4 py-3">
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

/** An absent value, stated rather than left blank.
 *
 * A blank cell reads as "nothing to see"; an absent number in this project
 * is a fact about the source or about what was never computed, and the UI
 * is required to say so out loud. */
export function Absent({ children = 'нет в источнике' }) {
  return <span className="italic text-slate-400">{children}</span>
}

export function NotComputed() {
  return <span className="italic text-slate-500">не посчитана</span>
}

export function QueryState({ query, children }) {
  if (query.isPending) {
    return <p className="p-4 text-sm text-slate-500">Загрузка…</p>
  }
  if (query.isError) {
    return (
      <p className="m-4 rounded border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800">
        Ошибка загрузки: {String(query.error?.message ?? query.error)}
      </p>
    )
  }
  return children
}

export function StatusBadge({ status }) {
  const style = STATUS_STYLE[status] ?? 'bg-slate-100 text-slate-700 ring-slate-300'
  return (
    <span
      className={`inline-block whitespace-nowrap rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${style}`}
    >
      {label(IDEA_STATUS, status)}
    </span>
  )
}

export function IdeaLink({ id, children }) {
  return (
    <Link className="text-sky-700 underline-offset-2 hover:underline" to={`/ideas/${id}`}>
      {children ?? id}
    </Link>
  )
}

/** Server-side pager. The UI never holds a full table in memory: the trial
 * ledger alone is thousands of rows. */
export function Pager({ total, limit, offset, onChange }) {
  const from = total === 0 ? 0 : offset + 1
  const to = Math.min(offset + limit, total)
  const canPrev = offset > 0
  const canNext = offset + limit < total
  const button = 'rounded border border-slate-300 px-2 py-1 text-xs disabled:opacity-40'
  return (
    <div className="flex items-center gap-3 py-2 text-xs text-slate-600">
      <span>
        {from}–{to} из {total}
      </span>
      <button
        type="button"
        className={button}
        disabled={!canPrev}
        onClick={() => onChange(Math.max(0, offset - limit))}
      >
        назад
      </button>
      <button
        type="button"
        className={button}
        disabled={!canNext}
        onClick={() => onChange(offset + limit)}
      >
        вперёд
      </button>
    </div>
  )
}

export function Num({ value, digits = 4 }) {
  if (value == null) return <Absent>нет</Absent>
  return <span className="tabular-nums">{Number(value).toFixed(digits)}</span>
}

export function DateText({ value, withTime = false }) {
  if (!value) return <Absent>нет</Absent>
  return (
    <span className="tabular-nums">{withTime ? value.replace('T', ' ').slice(0, 19) : value.slice(0, 10)}</span>
  )
}

export function Json({ value }) {
  return (
    <pre className="max-h-72 overflow-auto rounded bg-slate-50 p-2 text-xs text-slate-700">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}
