import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { fetchIdea, fetchIdeaPage } from '../api.js'
import { Absent, Json, Pager, Panel, QueryState, StatusBadge } from '../components/ui.jsx'
import { VerdictLegend, VerdictList } from '../components/Verdict.jsx'
import TrialTable from '../components/TrialTable.jsx'
import {
  ASSET_CLASS,
  PROFILE,
  SHUTDOWN_CAUSE,
  SOURCE_TYPE,
  label,
} from '../labels.js'

const PAGE_SIZE = 50

function usePagedSection(ideaId, resource) {
  const [offset, setOffset] = useState(0)
  const query = useQuery({
    queryKey: ['idea', ideaId, resource, offset],
    queryFn: () => fetchIdeaPage(ideaId, resource, { limit: PAGE_SIZE, offset }),
    placeholderData: keepPreviousData,
  })
  return { query, offset, setOffset }
}

function Field({ title, children }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{title}</dt>
      <dd className="mt-0.5 text-sm text-slate-800">{children}</dd>
    </div>
  )
}

function Section({ title, subtitle, section, children }) {
  const { query, offset, setOffset } = section
  return (
    <Panel title={title} subtitle={subtitle}>
      <QueryState query={query}>
        {query.data && (
          <>
            <Pager
              total={query.data.total}
              limit={query.data.limit}
              offset={query.data.offset}
              onChange={setOffset}
            />
            {children(query.data.items)}
            {query.data.total > offset + PAGE_SIZE && (
              <Pager
                total={query.data.total}
                limit={query.data.limit}
                offset={query.data.offset}
                onChange={setOffset}
              />
            )}
          </>
        )}
      </QueryState>
    </Panel>
  )
}

export default function IdeaDetail() {
  const { ideaId } = useParams()
  const ideaQuery = useQuery({ queryKey: ['idea', ideaId], queryFn: () => fetchIdea(ideaId) })
  const specs = usePagedSection(ideaId, 'specs')
  const trials = usePagedSection(ideaId, 'trials')
  const verdicts = usePagedSection(ideaId, 'verdicts')

  return (
    <div className="space-y-6">
      <Link className="text-sm text-sky-700 underline-offset-2 hover:underline" to="/ideas">
        ← к реестру
      </Link>

      <QueryState query={ideaQuery}>{ideaQuery.data && <Header idea={ideaQuery.data} />}</QueryState>

      <Section
        title="Спецификации"
        subtitle="Формализации идеи: параметры, требования к данным, модель костов."
        section={specs}
      >
        {(items) => (
          <div className="space-y-3">
            {items.length === 0 && <p className="text-sm text-slate-500">Спецификаций нет.</p>}
            {items.map((spec) => (
              <div key={spec.id} className="rounded border border-slate-200 p-3">
                <div className="flex flex-wrap items-baseline gap-x-4 text-sm">
                  <span className="font-medium">версия {spec.version}</span>
                  <span className="font-mono text-xs text-slate-500">{spec.code_ref}</span>
                  <span className="text-xs text-slate-500">ребаланс: {spec.rebalance}</span>
                  <span className="text-xs text-slate-400">{spec.created_at?.slice(0, 10)}</span>
                </div>
                {/* Collapsed by default: a calibration idea carries hundreds
                    of spec versions, and expanding every one of them buries
                    the trials and verdicts further down the page. */}
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-sky-700">
                    параметры, данные, косты
                  </summary>
                  <div className="mt-2 grid gap-3 md:grid-cols-3">
                    <div>
                      <div className="mb-1 text-xs font-medium text-slate-500">Параметры</div>
                      <Json value={spec.params} />
                    </div>
                    <div>
                      <div className="mb-1 text-xs font-medium text-slate-500">Данные</div>
                      <Json value={spec.data_requirements} />
                    </div>
                    <div>
                      <div className="mb-1 text-xs font-medium text-slate-500">Косты</div>
                      <Json value={spec.costs_model} />
                    </div>
                  </div>
                </details>
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section
        title="Прогоны"
        subtitle="Все прогоны идеи, новые сверху — включая выброшенные."
        section={trials}
      >
        {(items) => <TrialTable trials={items} showIdea={false} />}
      </Section>

      <Panel
        title="Вердикты"
        subtitle="Три вида строки различаются: решение, неизвестно, измерение. Проход/провал берётся из реестра, а не пересчитывается."
      >
        <VerdictLegend />
      </Panel>

      <Section title="Вердикты — строки" section={verdicts}>
        {(items) => <VerdictList items={items} />}
      </Section>
    </div>
  )
}

function Header({ idea }) {
  const counts = idea.counts
  return (
    <div className="space-y-4">
      <Panel>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold">{idea.title}</h1>
            <p className="font-mono text-xs text-slate-400">{idea.id}</p>
          </div>
          <div className="text-right">
            <StatusBadge status={idea.status} />
            {idea.shutdown_cause && (
              <div className="mt-1 text-xs text-stone-600">
                причина остановки: {label(SHUTDOWN_CAUSE, idea.shutdown_cause)}
              </div>
            )}
          </div>
        </div>

        <dl className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field title="профиль">{label(PROFILE, idea.profile)}</Field>
          <Field title="класс актива">{label(ASSET_CLASS, idea.asset_class)}</Field>
          <Field title="источник">
            {label(SOURCE_TYPE, idea.source_type)}
            {idea.source_url && (
              <a
                className="ml-2 break-all text-xs text-sky-700 underline-offset-2 hover:underline"
                href={idea.source_url}
                target="_blank"
                rel="noreferrer"
              >
                {idea.source_url}
              </a>
            )}
          </Field>
          <Field title="драйвер">
            {idea.driver ? idea.driver.title : <Absent>драйвер не назначен</Absent>}
          </Field>
        </dl>

        <dl className="mt-4 grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Field title="спеков">{counts.specs}</Field>
          <Field title="прогонов">{counts.trials}</Field>
          <Field title="вердиктов">{counts.verdicts}</Field>
          <Field title="решений">{counts.verdicts_decision}</Field>
          <Field title="неизвестно">{counts.verdicts_unknown}</Field>
          <Field title="измерений">{counts.verdicts_measurement}</Field>
        </dl>
      </Panel>

      {idea.driver && (
        <Panel title="Драйвер" subtitle={idea.driver.id}>
          <dl className="grid gap-4 md:grid-cols-2">
            <Field title="механизм">{idea.driver.description}</Field>
            <Field title="что убьёт эдж">{idea.driver.kill_condition}</Field>
            <Field title="как наблюдать">{idea.driver.observable}</Field>
          </dl>
        </Panel>
      )}

      <Panel title="Что утверждает источник и что мы записали">
        <dl className="grid gap-4">
          <Field title="заявленный эдж">
            {idea.claimed_edge ?? <Absent>не записан</Absent>}
          </Field>
          <Field title="заметки">
            {idea.notes ? (
              <p className="whitespace-pre-wrap text-sm leading-relaxed">{idea.notes}</p>
            ) : (
              <Absent>нет</Absent>
            )}
          </Field>
        </dl>
      </Panel>
    </div>
  )
}
