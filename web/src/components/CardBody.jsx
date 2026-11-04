import { Absent, Panel } from './ui.jsx'
import { CARD_FIELD, CARD_PART, label } from '../labels.js'

// The card of an idea (cards/<idea_id>.yaml, schema cards/SCHEMA.md). Since 2026-11-04 it is a
// section of the idea page (the registry is the one place for an idea that exists); the inbox page
// shows the same body for a card whose idea is not registered yet.

// The six parts, in the order cards/SCHEMA.md defines them.
const PARTS = ['driver', 'signal', 'sizing', 'protection', 'execution', 'death']

/** A card value exactly as the YAML holds it.
 *
 * `null` is never rendered as a blank: cards/SCHEMA.md is explicit that an
 * absent field must say so ("нет в источнике"), because a blank cell and a
 * source that is silent on the question look identical otherwise. */
function Value({ value }) {
  if (value == null) return <Absent />
  if (typeof value === 'boolean') {
    return <span className={value ? 'font-medium text-emerald-800' : 'text-slate-700'}>{value ? 'да' : 'нет'}</span>
  }
  if (Array.isArray(value)) {
    if (!value.length) return <Absent>пусто</Absent>
    return (
      <ul className="list-disc space-y-0.5 pl-5">
        {value.map((item, index) => (
          <li key={index}>{typeof item === 'object' ? JSON.stringify(item) : String(item)}</li>
        ))}
      </ul>
    )
  }
  if (typeof value === 'object') {
    return <span className="font-mono text-xs">{JSON.stringify(value)}</span>
  }
  return <span className="whitespace-pre-wrap">{String(value)}</span>
}

function Fields({ fields }) {
  const entries = Object.entries(fields)
  if (!entries.length) return <Absent>раздела нет в карточке</Absent>
  return (
    <dl className="space-y-2">
      {entries.map(([key, value]) => (
        <div key={key} className="grid gap-1 sm:grid-cols-[11rem,1fr] sm:gap-4">
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            {label(CARD_FIELD, key)}
          </dt>
          <dd className="text-sm text-slate-800">
            <Value value={value} />
          </dd>
        </div>
      ))}
    </dl>
  )
}

/** Every quote, verbatim, with its source link.
 *
 * cards/SCHEMA.md requires each claim to stand on a quote; a card that
 * shows the claim but hides the quote turns a sourced reading back into an
 * assertion. Quotes are never translated (CLAUDE.md). */
function Quotes({ quotes }) {
  if (!quotes.length) {
    return (
      <p className="mt-3 text-xs">
        <Absent>цитаты в карточке нет</Absent>
      </p>
    )
  }
  return (
    <div className="mt-3 space-y-2">
      {quotes.map((quote) => (
        <blockquote
          key={quote.key}
          className="border-l-2 border-slate-300 bg-slate-50 py-2 pl-3 pr-2"
        >
          <p className="whitespace-pre-wrap text-sm italic text-slate-700">
            {quote.text ?? <Absent>текст цитаты отсутствует</Absent>}
          </p>
          {quote.source ? (
            <a
              className="mt-1 block break-all text-xs text-sky-700 underline-offset-2 hover:underline"
              href={quote.source}
              target="_blank"
              rel="noreferrer"
            >
              {quote.source}
            </a>
          ) : (
            <p className="mt-1 text-xs">
              <Absent>ссылка на источник не указана</Absent>
            </p>
          )}
        </blockquote>
      ))}
    </div>
  )
}

/** The protection row.
 *
 * An empty one is a finding, not missing data (cards/README.md: "пустая
 * строка «защита» — это сама по себе находка"), so it is rendered as a
 * stated absence in its own frame rather than as blank fields that read
 * like the card was never filled in. */
function Protection({ part, isEmpty }) {
  return (
    <section
      className={`rounded-lg border p-4 ${
        isEmpty ? 'border-rose-400 bg-rose-50' : 'border-emerald-300 bg-emerald-50/40'
      }`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">{CARD_PART.protection}</h3>
        {isEmpty ? (
          <span className="rounded bg-rose-600 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-white">
            хвосты не закрыты ничем
          </span>
        ) : (
          <span className="rounded bg-emerald-700 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-white">
            защита описана
          </span>
        )}
      </div>

      {isEmpty && (
        <p className="mt-2 text-sm text-rose-900">
          Ни один механизм закрытия хвостов в источнике не назван. Это находка разбора, а не
          пропуск в карточке: большинство описанных стратегий про хвосты молчит.
        </p>
      )}

      <div className="mt-3">
        <Fields fields={part.fields} />
      </div>
      <Quotes quotes={part.quotes} />
    </section>
  )
}

/** The card's content: the plain summary, sources, the six parts, feasibility and prior. */
export default function CardBody({ card }) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-xs uppercase tracking-wide text-slate-500">Простыми словами</h3>
        {card.plain_summary ? (
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-800">
            {card.plain_summary}
          </p>
        ) : (
          <Absent>
            пересказа нет: карточка написана до правила «без жаргона» (cards/SCHEMA.md, 2026-10-01)
          </Absent>
        )}
      </div>

      <div>
        <h3 className="text-xs uppercase tracking-wide text-slate-500">Источники</h3>
        <ul className="mt-1 space-y-1">
          {card.sources.map((source, index) => (
            <li key={index} className="text-sm">
              <a
                className="break-all text-sky-700 underline-offset-2 hover:underline"
                href={source.url}
                target="_blank"
                rel="noreferrer"
              >
                {source.title ?? source.url}
              </a>
              <span className="ml-2 text-xs text-slate-400">
                {source.published ?? 'дата публикации не указана'}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {PARTS.map((name) => {
        const part = card.parts[name]
        if (name === 'protection') {
          return <Protection key={name} part={part} isEmpty={card.protection_is_empty} />
        }
        return (
          <Panel key={name} title={CARD_PART[name]}>
            {part.present ? <Fields fields={part.fields} /> : <Absent>раздела нет в карточке</Absent>}
            <Quotes quotes={part.quotes} />
          </Panel>
        )
      })}

      <Panel title="Проверяемость" subtitle="Видно до расчётов, жива ли идея вообще.">
        {card.feasibility ? <Fields fields={card.feasibility} /> : <Absent>раздела нет</Absent>}
      </Panel>

      <Panel title="Связь с уже известным" subtitle="Совпадение по драйверу — не фильтр.">
        {card.prior ? <Fields fields={card.prior} /> : <Absent>раздела нет</Absent>}
      </Panel>

      <p className="text-xs text-slate-500">
        найдено: {card.found_at ?? <Absent>дата не указана</Absent>} · файл{' '}
        <span className="font-mono">{card.path}</span>
      </p>
    </div>
  )
}
