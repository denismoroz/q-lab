import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchAgentFunnel } from '../api.js'
import { IdeaLink, Panel, QueryState } from './ui.jsx'
import {
  AGENT_NAME,
  AGENT_PERIOD,
  AGENT_SOURCE,
  AGENT_SOURCE_HINT,
  AGENT_STAGE,
  NO_SURVIVORS,
  TRIAL_ROUTE,
  label,
} from '../labels.js'

// qlab.funnel counts everything below from the registry and the token ledger;
// this component only lays the numbers out and explains each step.

const IDEA_SOURCES = ['generator', 'researcher', 'implementer']

const number = (value) => value.toLocaleString('ru-RU')

function Step({ index, title, value, explain, children }) {
  return (
    <li className="rounded border border-slate-200 p-3">
      <div className="flex items-baseline justify-between gap-4">
        <div className="text-sm font-medium text-slate-800">
          {index}. {title}
        </div>
        <div className="tabular-nums text-xl font-semibold text-slate-900">{value}</div>
      </div>
      <p className="mt-1 text-xs text-slate-500">{explain}</p>
      {children}
    </li>
  )
}

function Rows({ rows, empty }) {
  if (!rows.length) return <p className="mt-2 text-xs text-slate-400">{empty}</p>
  return (
    <dl className="mt-2 divide-y divide-slate-100">
      {rows.map(([key, text, count]) => (
        <div key={key} className="flex items-baseline justify-between gap-4 py-1">
          <dt className="text-xs text-slate-700">{text}</dt>
          <dd className="tabular-nums text-xs font-medium text-slate-900">{count}</dd>
        </div>
      ))}
    </dl>
  )
}

function Period({ period }) {
  const sources = Object.fromEntries(period.by_source.map((row) => [row.source, row]))
  const routeRows = Object.entries(period.routes)
    .sort((a, b) => b[1] - a[1])
    .map(([route, count]) => [route, label(TRIAL_ROUTE, route), count])
  const agentRows = Object.entries(period.tokens_by_agent)
    .sort((a, b) => b[1] - a[1])
    .map(([agent, tokens]) => [agent, label(AGENT_NAME, agent), number(tokens)])
  const stageRows = Object.entries(period.tokens_by_stage)
    .sort((a, b) => b[1] - a[1])
    .map(([stage, tokens]) => [stage, label(AGENT_STAGE, stage), number(tokens)])
  const survivors = period.survivors

  return (
    <div className="space-y-4">
      <ol className="grid gap-3 lg:grid-cols-2">
        <Step
          index={1}
          title="Агенты придумали идей"
          value={period.created}
          explain="Идеи, которые агент сам занёс в реестр. Идеи владельца, импорт старого фреймворка и калибровочный шум не считаются."
        >
          <Rows
            rows={IDEA_SOURCES.map((source) => [
              source,
              <span key={source}>
                {AGENT_SOURCE[source]}
                <span className="block text-slate-400">{AGENT_SOURCE_HINT[source]}</span>
              </span>,
              sources[source]?.created ?? 0,
            ])}
            empty=""
          />
          <p className="mt-2 text-xs text-slate-600">
            Разведчик: черновиков карточек {period.scout_drafts} (вызовов {period.scout_calls}).{' '}
            <span className="text-slate-400">{AGENT_SOURCE_HINT.scout}.</span>
          </p>
        </Step>

        <Step
          index={2}
          title="Прогнано на стенде"
          value={`${period.evaluated} из ${period.created}`}
          explain="У идеи есть хотя бы один прогон с записанным исходом: стенд её действительно проверил. Остальные до стенда ещё не дошли."
        >
          <Rows rows={routeRows} empty="ни одной идеи ещё не прогнали" />
        </Step>

        <Step
          index={3}
          title="Дошли до владельца"
          value={survivors}
          explain="Выжившие: идея прошла все правила стенда и ждёт решения владельца (статус «прошла, ждёт решения» или дальше). В бумагу идеи переводит только человек."
        >
          {survivors > 0 ? (
            <ul className="mt-2 space-y-0.5 text-xs">
              {period.survivor_ids.map((id) => (
                <li key={id}>
                  <IdeaLink id={id} />
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-xs text-slate-400">{NO_SURVIVORS}</p>
          )}
        </Step>

        <Step
          index={4}
          title="Цена одного выжившего"
          value={
            period.tokens_per_survivor == null
              ? NO_SURVIVORS
              : `${number(period.tokens_per_survivor)} токенов`
          }
          explain="Все токены агентов за период, делённые на число выживших. Токены — единица подписки: считаются вход, выход и обращения к кэшу, как их считает бюджетный гард."
        >
          <p className="mt-2 text-xs text-slate-600">
            Потрачено за период: {number(period.tokens_total)} токенов
            {period.probe_tokens > 0 &&
              ` (и ещё ${number(period.probe_tokens)} на замеры окон подписки — они в цену не входят)`}
          </p>
        </Step>
      </ol>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Токены по агентам
          </h3>
          <Rows rows={agentRows} empty="токены не тратились" />
        </div>
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Токены по стадиям ночи
          </h3>
          <Rows rows={stageRows} empty="токены не тратились" />
        </div>
      </div>
    </div>
  )
}

export default function AgentFunnel() {
  const query = useQuery({ queryKey: ['agent-funnel'], queryFn: fetchAgentFunnel })
  const [chosen, setChosen] = useState(0)

  return (
    <Panel
      title="Находят ли агенты стратегии сами"
      subtitle="Цель владельца — система агентов, которая сама находит прибыльные стратегии. Здесь видно, сколько идей агенты придумали, как далеко они дошли и сколько токенов стоит один выживший. Числа считаются по реестру, не оцениваются."
    >
      <QueryState query={query}>
        {query.data && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-1">
              {query.data.periods.map((period, i) => (
                <button
                  key={period.label}
                  type="button"
                  onClick={() => setChosen(i)}
                  className={`rounded px-3 py-1.5 text-sm ${
                    i === chosen
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-700 ring-1 ring-slate-200 hover:bg-slate-100'
                  }`}
                >
                  {label(AGENT_PERIOD, period.label)}
                </button>
              ))}
            </div>
            <Period period={query.data.periods[chosen]} />
          </div>
        )}
      </QueryState>
    </Panel>
  )
}
