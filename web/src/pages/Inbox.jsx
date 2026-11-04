import { Link, Navigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchCard, fetchInbox } from '../api.js'
import CardBody from '../components/CardBody.jsx'
import { Absent, Panel, QueryState } from '../components/ui.jsx'

// Owner, 2026-11-04: «у нас есть карточки кандидатов и реестр - и там тоже есть статус кандидат -
// очень похоже на дупликацию». The registry is the one place for an idea that exists; this page holds
// only what is NOT in it yet, by the stage it is at (src/qlab/api/inbox.py). An idea that is
// registered is a link to its page, where its card is a section.

const STAGE_STYLE = {
  waiting: 'bg-slate-100 text-slate-700 ring-slate-300',
  writing: 'bg-amber-50 text-amber-900 ring-amber-300',
  owner: 'bg-rose-50 text-rose-900 ring-rose-300',
  duplicate: 'bg-stone-100 text-stone-700 ring-stone-300',
  card_ready: 'bg-sky-50 text-sky-900 ring-sky-300',
  registered: 'bg-emerald-50 text-emerald-900 ring-emerald-300',
  closed: 'bg-stone-100 text-stone-600 ring-stone-300',
}

const STAGE_HINT = {
  waiting: 'Идея записана, карточник ещё не брался за неё.',
  writing: 'Карточник пробовал; код не принял черновик. Причины — под идеей.',
  owner: 'Без вашего ответа карточку не написать.',
  duplicate: 'Это уже проверяли или похоронили; новой проверки не будет.',
  card_ready: 'Карточка написана, в реестр идея ещё не внесена: её реализует программист.',
  registered: 'Идея в реестре. Карточка — раздел её страницы.',
  closed: 'Закрыта вами.',
}

function Item({ item }) {
  return (
    <li className="py-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        {item.id && <span className="font-mono text-xs text-slate-400">{item.id}</span>}
        <span className="text-xs text-slate-400">{item.date}</span>
        {item.source && <span className="text-xs text-slate-400">{item.source}</span>}
        <span
          className={`rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${STAGE_STYLE[item.stage]}`}
        >
          {item.stage_ru}
        </span>
      </div>
      <p className="mt-1 line-clamp-3 max-w-4xl whitespace-pre-wrap text-sm text-slate-800">
        {item.text}
      </p>

      {item.question && (
        <p className="mt-2 max-w-4xl rounded border border-rose-200 bg-rose-50 p-2 text-sm text-rose-900">
          <span className="font-medium">Вопрос: </span>
          {item.question}
        </p>
      )}
      {item.repeats && (
        <p className="mt-2 max-w-4xl text-sm text-slate-700">
          Повторяет: <span className="font-mono text-xs">{item.repeats}</span>
          {item.reason ? `. ${item.reason}` : ''}
        </p>
      )}
      {item.stage === 'closed' && item.reason && (
        <p className="mt-2 text-sm text-slate-600">{item.reason}</p>
      )}
      {item.problems.length > 0 && (
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs text-amber-900">
          {item.problems.map((problem, index) => (
            <li key={index}>{problem}</li>
          ))}
        </ul>
      )}
      {item.page && (
        <p className="mt-2 text-sm">
          <Link className="text-sky-700 underline-offset-2 hover:underline" to={item.page}>
            {item.stage === 'registered'
              ? 'страница идеи в реестре →'
              : `карточка: ${item.card_title ?? item.idea_id} →`}
          </Link>
        </p>
      )}
    </li>
  )
}

function InboxBody({ inbox }) {
  const stages = Object.keys(inbox.counts).filter((s) => inbox.counts[s] > 0)
  return (
    <div className="space-y-6">
      <Panel
        title="Входящие"
        subtitle="Идеи, которых ещё нет в реестре, и на какой они ступени. Реестр — единственное место для идей, которые уже есть."
      >
        <ul className="flex flex-wrap gap-2">
          {Object.keys(inbox.counts).map((s) => (
            <li
              key={s}
              className={`rounded px-2.5 py-1 text-xs ring-1 ring-inset ${STAGE_STYLE[s]}`}
            >
              {inbox.stage_ru[s]} · {inbox.counts[s]}
            </li>
          ))}
        </ul>
      </Panel>

      {stages.length === 0 && (
        <Panel>
          <Absent>во входящих ничего нет</Absent>
        </Panel>
      )}
      {stages.map((s) => (
        <Panel key={s} title={`${inbox.stage_ru[s]} · ${inbox.counts[s]}`} subtitle={STAGE_HINT[s]}>
          <ul className="divide-y divide-slate-100">
            {inbox.items
              .filter((item) => item.stage === s)
              .map((item) => (
                <Item key={item.id ?? item.idea_id} item={item} />
              ))}
          </ul>
        </Panel>
      ))}
    </div>
  )
}

export function Inbox() {
  const query = useQuery({ queryKey: ['inbox'], queryFn: fetchInbox })
  return <QueryState query={query}>{query.data && <InboxBody inbox={query.data} />}</QueryState>
}

/** A card whose idea is not in the registry yet (`/inbox/<idea_id>`). Once the idea is
 * registered the card is a section of its page, so the address leads there instead. */
export function InboxCard() {
  const { ideaId } = useParams()
  const query = useQuery({ queryKey: ['card', ideaId], queryFn: () => fetchCard(ideaId) })
  if (query.data?.registered) return <Navigate to={query.data.page} replace />
  return (
    <div className="space-y-4">
      <Link className="text-sm text-sky-700 underline-offset-2 hover:underline" to="/inbox">
        ← ко входящим
      </Link>
      <QueryState query={query}>
        {query.data && (
          <Panel title={query.data.title} subtitle={`${query.data.idea_id} · в реестр ещё не внесена`}>
            <CardBody card={query.data} />
          </Panel>
        )}
      </QueryState>
    </div>
  )
}

/** The old `/cards/<idea_id>` address: the idea page when the idea is registered, else the inbox. */
export function CardRedirect() {
  const { ideaId } = useParams()
  const query = useQuery({ queryKey: ['card', ideaId], queryFn: () => fetchCard(ideaId) })
  if (query.data) return <Navigate to={query.data.page} replace />
  if (query.isError) return <Navigate to="/inbox" replace />
  return <p className="p-4 text-sm text-slate-500">Загрузка…</p>
}
