import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import {
  fetchPipeline,
  fetchPipelineRuns,
  fetchPipelineStep,
  previewPipeline,
  restartStep,
  runPipeline,
} from '../api.js'
import { Panel, QueryState } from '../components/ui.jsx'

// The night as steps (docs/PIPELINE.md): the dependency graph with each step coloured by what
// happened to it, the table of what takes time, one step's history, and the two buttons that start
// a night. Everything shown is copied from the journal and the declared table (night/pipeline.yaml);
// nothing is judged here. The buttons are the console's only writes, and they only START a run.

const STATE = {
  ok: { text: 'прошёл', box: 'border-emerald-400 bg-emerald-50', dot: 'bg-emerald-500', ink: 'text-emerald-900' },
  failed: { text: 'упал', box: 'border-rose-400 bg-rose-50', dot: 'bg-rose-500', ink: 'text-rose-900' },
  running: { text: 'идёт', box: 'border-sky-400 bg-sky-50 animate-pulse', dot: 'bg-sky-500', ink: 'text-sky-900' },
  blocked: { text: 'заблокирован', box: 'border-amber-400 bg-amber-50', dot: 'bg-amber-400', ink: 'text-amber-900' },
  interrupted: { text: 'прерван', box: 'border-orange-400 bg-orange-50', dot: 'bg-orange-500', ink: 'text-orange-900' },
  off: { text: 'выключен', box: 'border-slate-300 bg-slate-100', dot: 'bg-slate-400', ink: 'text-slate-500' },
  skipped: { text: 'не нужен сегодня', box: 'border-slate-300 bg-slate-100', dot: 'bg-slate-400', ink: 'text-slate-600' },
  cut: { text: 'не начат: срок', box: 'border-slate-300 bg-slate-100', dot: 'bg-slate-400', ink: 'text-slate-600' },
  pending: { text: 'ждёт', box: 'border-slate-300 bg-white', dot: 'bg-slate-300', ink: 'text-slate-600' },
}
const KIND = { data: 'данные', code: 'код', agent: 'агент' }
const EDGE = {
  data: { text: 'данные: упавший родитель блокирует шаг', stroke: '#334155', dash: '' },
  fresh: { text: 'свежесть: ждёт, но не блокируется', stroke: '#94a3b8', dash: '5 4' },
  exclusive: { text: 'не вместе: оба пишут испытания одних идей', stroke: '#c2410c', dash: '2 3' },
  budget: { text: 'порядок трат токенов, данных не передаёт', stroke: '#cbd5e1', dash: '1 4' },
}
const TRIGGER = { schedule: 'по расписанию', button: 'кнопкой', hand: 'вручную' }

const BOX_W = 176
const BOX_H = 64
const GAP_X = 52
const GAP_Y = 12
const HEAD = 30

function duration(seconds) {
  if (seconds == null) return '—'
  if (seconds < 1) return '<1 с'
  if (seconds < 90) return `${Math.round(seconds)} с`
  if (seconds < 5400) return `${Math.round(seconds / 60)} мин`
  const total = Math.round(seconds / 60)
  return `${Math.floor(total / 60)} ч ${String(total % 60).padStart(2, '0')} мин`
}

function when(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function tokens(n) {
  if (!n) return '—'
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)} млн` : n >= 1e3 ? `${Math.round(n / 1e3)} тыс.` : `${n}`
}

function Dot({ state }) {
  const s = STATE[state] ?? STATE.pending
  return <span className={`inline-block h-2.5 w-2.5 shrink-0 rounded-full ${s.dot}`} />
}

function StateBadge({ state }) {
  const s = STATE[state] ?? STATE.pending
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded border px-2 py-0.5 text-xs font-medium ${s.box} ${s.ink}`}>
      <Dot state={state} />
      {s.text}
    </span>
  )
}

function layout(steps, waveKey) {
  const waves = [...new Set(steps.map((s) => s[waveKey]))].sort((a, b) => a - b)
  const rows = {}
  const pos = {}
  for (const step of steps) {
    const col = waves.indexOf(step[waveKey])
    const row = (rows[col] = (rows[col] ?? -1) + 1)
    pos[step.id] = { x: col * (BOX_W + GAP_X), y: HEAD + row * (BOX_H + GAP_Y), col }
  }
  const height = HEAD + (Math.max(...Object.values(rows)) + 1) * (BOX_H + GAP_Y)
  return { pos, width: waves.length * (BOX_W + GAP_X) - GAP_X, height, waves }
}

function Graph({ steps, states, selected, onSelect, running, free }) {
  const waveKey = free ? 'wave_without_budget' : 'wave'
  const { pos, width, height, waves } = useMemo(() => layout(steps, waveKey), [steps, waveKey])
  const edges = useMemo(() => {
    const out = []
    for (const step of steps) {
      if (step.id === 'report') continue // reads everything: drawn as a note, not 24 lines
      for (const e of step.depends_on) {
        if (free && e.kind === 'budget') continue
        out.push({ from: e.step, to: step.id, kind: e.kind })
      }
    }
    return out
  }, [steps, free])
  const touching = (e) => selected && (e.from === selected || e.to === selected)
  return (
    <div className="overflow-x-auto pb-2">
      <div className="relative" style={{ width, height }}>
        <svg className="absolute inset-0" width={width} height={height}>
          {waves.map((w, col) => (
            <text key={w} x={col * (BOX_W + GAP_X) + BOX_W / 2} y={16} textAnchor="middle" fontSize="11" fill="#64748b">
              волна {col + 1}
            </text>
          ))}
          {edges.map((e, i) => {
            const a = pos[e.from]
            const b = pos[e.to]
            if (!a || !b || b.x <= a.x) return null
            const x1 = a.x + BOX_W
            const y1 = a.y + BOX_H / 2
            const x2 = b.x
            const y2 = b.y + BOX_H / 2
            const dx = Math.max(24, (x2 - x1) / 2)
            const style = EDGE[e.kind]
            const hot = touching(e)
            return (
              <path
                key={i}
                d={`M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`}
                fill="none"
                stroke={hot ? '#0369a1' : style.stroke}
                strokeWidth={hot ? 2.2 : e.kind === 'data' ? 1.6 : 1.1}
                strokeDasharray={style.dash}
                opacity={selected && !hot ? 0.25 : 1}
              />
            )
          })}
        </svg>
        {steps.map((step) => {
          const st = states[step.id]
          const state = st?.state ?? 'pending'
          const s = STATE[state] ?? STATE.pending
          const p = pos[step.id]
          return (
            <button
              key={step.id}
              type="button"
              onClick={() => onSelect(step.id === selected ? null : step.id)}
              className={`absolute rounded-md border px-2 py-1 text-left shadow-sm hover:shadow ${s.box} ${
                step.id === selected ? 'ring-2 ring-sky-600' : ''
              }`}
              style={{ left: p.x, top: p.y, width: BOX_W, height: BOX_H }}
              title={`${step.title}: ${s.text}`}
            >
              <div className="flex items-start justify-between gap-1">
                <span className={`line-clamp-2 text-xs font-semibold leading-tight ${s.ink}`}>{step.title}</span>
                <span className="shrink-0 text-[10px] text-slate-500">
                  {KIND[step.kind]}
                  {step.tokens ? ' · токены' : ''}
                </span>
              </div>
              <div className="mt-1 flex items-center justify-between text-[11px] text-slate-600">
                <span className="inline-flex items-center gap-1">
                  <Dot state={state} />
                  {s.text}
                  {st?.items_failed > 0 && <b className="text-rose-700"> · {st.items_failed} с ошибкой</b>}
                </span>
                <span className="tabular-nums">{duration(st?.duration_s)}</span>
              </div>
              <div className="text-[10px] text-slate-500">
                {running && state === 'running' ? 'с ' : ''}
                {when(st?.finished_at ?? st?.started_at)}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600">
      {['ok', 'failed', 'running', 'blocked', 'off'].map((k) => (
        <span key={k} className="inline-flex items-center gap-1.5">
          <Dot state={k} />
          {{ off: 'серый: выключен, не нужен сегодня, ещё ждёт' }[k] ?? STATE[k].text}
        </span>
      ))}
      <span className="text-slate-300">|</span>
      {Object.entries(EDGE).map(([k, e]) => (
        <span key={k} className="inline-flex items-center gap-1.5">
          <svg width="26" height="8">
            <line x1="0" y1="4" x2="26" y2="4" stroke={e.stroke} strokeWidth="1.6" strokeDasharray={e.dash} />
          </svg>
          {e.text}
        </span>
      ))}
    </div>
  )
}

function TimeTable({ steps, states, onSelect }) {
  const rows = steps
    .map((s) => {
      const st = states[s.id]
      const measured = s.stats?.runs ? s.stats : s.stats_approximate
      return {
        step: s,
        now: st && ['ok', 'failed', 'running'].includes(st.state) ? st.duration_s : null,
        median: measured?.median_s ?? measured?.avg_s ?? null,
        max: measured?.max_s ?? null,
        runs: measured?.runs ?? 0,
        approx: !s.stats?.runs && !!s.stats_approximate?.runs,
      }
    })
    .sort((a, b) => (b.now ?? b.median ?? -1) - (a.now ?? a.median ?? -1))
  const total = rows.reduce((sum, r) => sum + (r.now ?? 0), 0)
  const top = Math.max(1, ...rows.map((r) => r.now ?? r.median ?? 0))
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs text-slate-500">
          <th className="py-1 pr-3 font-medium">Шаг</th>
          <th className="py-1 pr-3 font-medium">Состояние</th>
          <th className="py-1 pr-3 text-right font-medium">Последняя ночь</th>
          <th className="py-1 pr-3 font-medium" />
          <th className="py-1 pr-3 text-right font-medium">Доля</th>
          <th className="py-1 pr-3 text-right font-medium">Обычно (медиана)</th>
          <th className="py-1 pr-3 text-right font-medium">Самый долгий</th>
          <th className="py-1 text-right font-medium">Запусков</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.step.id} className="cursor-pointer border-t border-slate-100 hover:bg-slate-50" onClick={() => onSelect(r.step.id)}>
            <td className="py-1 pr-3">{r.step.title}</td>
            <td className="py-1 pr-3"><StateBadge state={states[r.step.id]?.state ?? 'pending'} /></td>
            <td className="py-1 pr-3 text-right tabular-nums">{duration(r.now)}</td>
            <td className="w-40 py-1 pr-3">
              <div className="h-2 rounded bg-slate-100">
                <div className="h-2 rounded bg-sky-500" style={{ width: `${Math.round(((r.now ?? 0) / top) * 100)}%` }} />
              </div>
            </td>
            <td className="py-1 pr-3 text-right tabular-nums text-slate-600">
              {total > 0 && r.now ? `${Math.round((r.now / total) * 100)}%` : '—'}
            </td>
            <td className="py-1 pr-3 text-right tabular-nums">
              {duration(r.median)}{r.approx && r.median != null ? ' ≈' : ''}
            </td>
            <td className="py-1 pr-3 text-right tabular-nums">{duration(r.max)}{r.approx && r.max != null ? ' ≈' : ''}</td>
            <td className="py-1 text-right tabular-nums text-slate-600">{r.runs || '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function StepDetail({ stepId, steps, onSelect, onRestart, busy }) {
  const query = useQuery({
    queryKey: ['pipeline-step', stepId],
    queryFn: () => fetchPipelineStep(stepId),
    refetchInterval: busy ? 3000 : 20000,
  })
  const titles = Object.fromEntries(steps.map((s) => [s.id, s.title]))
  return (
    <QueryState query={query}>
      {query.data && (() => {
        const d = query.data
        const st = d.state
        const nstate = d.neighbour_states ?? {}
        const link = (id) => (
          <button type="button" className="text-sky-700 hover:underline" onClick={() => onSelect(id)}>
            {titles[id] ?? id}
          </button>
        )
        const stats = d.stats
        const approx = d.stats_approximate
        return (
          <div className="space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold text-slate-900">{d.title}</h3>
                <p className="text-sm text-slate-700">{d.description}</p>
                <p className="mt-1 text-xs text-slate-500">
                  {KIND[d.kind]}{d.tokens ? ' · тратит токены' : ' · токены не тратит'}
                  {d.item_kind ? ` · из чего состоит: ${d.item_kind}` : ''}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <StateBadge state={st?.state ?? 'pending'} />
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onRestart(d)}
                  className="rounded bg-slate-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
                >
                  Перезапустить шаг…
                </button>
              </div>
            </div>

            {st?.error && (
              <pre className="whitespace-pre-wrap break-words rounded border border-rose-200 bg-rose-50 p-2 text-xs text-rose-900">{st.error}</pre>
            )}
            {st?.state === 'blocked' && (
              <p className="rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
                Не запускался: зависит от «{titles[st.blocked_by] ?? st.blocked_by}». {st.note}
              </p>
            )}
            {st?.note && st.state !== 'blocked' && <p className="text-xs text-slate-500">{st.note}</p>}

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase text-slate-500">Зависит от</h4>
                {d.depends_on.length === 0 && <p className="text-xs text-slate-500">Ни от чего: может идти с самого начала.</p>}
                <ul className="space-y-1.5 text-sm">
                  {d.depends_on.filter((e) => d.id !== 'report').map((e) => (
                    <li key={e.step}>
                      <span className="inline-flex items-center gap-1.5"><Dot state={nstate[e.step] ?? 'pending'} />{link(e.step)}</span>
                      <span className="ml-1.5 rounded bg-slate-100 px-1 text-[10px] text-slate-600">{e.kind}</span>
                      <div className="text-xs text-slate-500">{e.why}</div>
                    </li>
                  ))}
                  {d.id === 'report' && <li className="text-xs text-slate-500">От всех шагов, но только по свежести: отчёт появляется, даже если шаги упали.</li>}
                </ul>
              </div>
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase text-slate-500">От него зависят</h4>
                {d.children.length === 0 && <p className="text-xs text-slate-500">Никто.</p>}
                <ul className="space-y-1.5 text-sm">
                  {d.children.filter((e) => e.step !== 'report').map((e) => (
                    <li key={e.step}>
                      <span className="inline-flex items-center gap-1.5"><Dot state={nstate[e.step] ?? 'pending'} />{link(e.step)}</span>
                      <span className="ml-1.5 rounded bg-slate-100 px-1 text-[10px] text-slate-600">{e.kind}</span>
                      <div className="text-xs text-slate-500">{e.why}</div>
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-xs text-slate-500">
                  Перезапуск запустит: {d.restart_set.map((id) => titles[id]).join(' → ')}.
                </p>
              </div>
            </div>

            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase text-slate-500">Что может идти одновременно с ним</h4>
              <p className="text-sm text-slate-700">
                {d.can_run_with.length ? d.can_run_with.map((id) => titles[id]).join(', ') : 'Ничего: он связан со всеми.'}
              </p>
              {d.can_run_with_without_budget.length > d.can_run_with.length && (
                <p className="text-xs text-slate-500">
                  Если не держать порядок трат токенов: ещё {d.can_run_with_without_budget.filter((id) => !d.can_run_with.includes(id)).map((id) => titles[id]).join(', ')}.
                </p>
              )}
            </div>

            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase text-slate-500">Статистика</h4>
              {stats && stats.runs > 0 ? (
                <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm md:grid-cols-4">
                  <Stat k="запусков" v={`${stats.runs} (прошло ${stats.ok}, упало ${stats.failed})`} />
                  <Stat k="доля успешных" v={stats.success_share == null ? '—' : `${Math.round(stats.success_share * 100)}%`} />
                  <Stat k="среднее / медиана" v={`${duration(stats.avg_s)} / ${duration(stats.median_s)}`} />
                  <Stat k="самый долгий" v={duration(stats.max_s)} />
                  <Stat k="токенов в среднем" v={tokens(stats.tokens_avg)} />
                  <Stat k="блокировок" v={stats.blocked} />
                </dl>
              ) : (
                <p className="text-sm text-slate-500">Замеренных запусков ещё нет.</p>
              )}
              {approx && approx.runs > 0 && (
                <p className="mt-1 text-xs text-slate-500">
                  Приблизительно, по записям прошлых ночей (первая и последняя отметка шага): {approx.runs} ночей, медиана {duration(approx.median_s)}, самая долгая {duration(approx.max_s)}.
                </p>
              )}
              {stats?.last_error && (
                <p className="mt-1 text-xs text-rose-800">Последняя ошибка ({when(stats.last_error_at)}): {stats.last_error}</p>
              )}
            </div>

            {d.items.length > 0 && (
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase text-slate-500">
                  Из чего состоит последний запуск ({d.items.length})
                </h4>
                <table className="w-full text-xs">
                  <tbody>
                    {d.items.map((it, i) => (
                      <tr key={i} className="border-t border-slate-100 align-top">
                        <td className="py-1 pr-3 font-mono break-all">{it.name}</td>
                        <td className="py-1 pr-3 text-right tabular-nums">{duration(it.duration_s)}</td>
                        <td className={`py-1 pr-3 ${it.outcome === 'failed' ? 'text-rose-800' : 'text-emerald-800'}`}>
                          {it.outcome === 'failed' ? 'ошибка' : 'сделано'}
                        </td>
                        <td className="py-1 text-slate-500 break-words">{it.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase text-slate-500">История запусков</h4>
              {d.history.length === 0 && <p className="text-sm text-slate-500">Записей нет.</p>}
              <table className="w-full text-xs">
                <tbody>
                  {d.history.map((h) => (
                    <tr key={h.step_run_id} className="border-t border-slate-100 align-top">
                      <td className="py-1 pr-3 tabular-nums">{h.night_date}</td>
                      <td className="py-1 pr-3 text-slate-500">{TRIGGER[h.trigger] ?? h.trigger}</td>
                      <td className="py-1 pr-3"><StateBadge state={h.state} /></td>
                      <td className="py-1 pr-3 text-right tabular-nums">{h.approximate ? '≈ ' : ''}{duration(h.duration_s)}</td>
                      <td className="py-1 pr-3 text-right tabular-nums text-slate-500">{h.items_done ? `${h.items_done} шт.` : ''}</td>
                      <td className="py-1 pr-3 text-right tabular-nums text-slate-500">{h.tokens ? tokens(h.tokens) : ''}</td>
                      <td className="py-1 break-words text-rose-800">{h.error ?? (h.blocked_by ? `заблокирован шагом «${titles[h.blocked_by] ?? h.blocked_by}»` : '')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })()}
    </QueryState>
  )
}

function Stat({ k, v }) {
  return (
    <div>
      <dt className="text-xs text-slate-500">{k}</dt>
      <dd className="tabular-nums text-slate-900">{v}</dd>
    </div>
  )
}

// The confirmation of a start: what will run, what spends tokens, what is blocked. The numbers come
// from the API's preview, the same function the runner uses, so what is promised is what happens.
function PreviewBody({ preview, titles }) {
  if (!preview) return <p className="text-sm text-slate-500">Считаю, что запустится…</p>
  const name = (id) => titles[id] ?? id
  return (
    <div className="space-y-2 text-sm">
      <p>
        Запустится шагов: <b>{preview.run.length}</b>
        {preview.outside.length > 0 && <>, остальные {preview.outside.length} не трогаются</>}.
      </p>
      {preview.token_steps.length > 0 ? (
        <p className="rounded border border-amber-300 bg-amber-50 p-2 text-amber-900">
          <b>Тратят токены подписки:</b> {preview.token_steps.map(name).join(', ')}.
        </p>
      ) : (
        <p className="text-slate-600">Токены не тратятся.</p>
      )}
      {preview.off.length > 0 && <p className="text-slate-600">Выключены: {preview.off.map(name).join(', ')}.</p>}
      {preview.blocked.length > 0 && (
        <div className="rounded border border-slate-200 bg-slate-50 p-2">
          <p className="font-medium text-slate-800">Не запустятся, потому что зависят от выключенного:</p>
          <ul className="list-disc pl-5 text-slate-600">
            {preview.blocked.map((b) => (
              <li key={b.step}>{name(b.step)} — {b.reason}</li>
            ))}
          </ul>
        </div>
      )}
      {preview.steps.some((s) => s.note) && (
        <p className="text-xs text-slate-500">
          По расписанию работы не будет сегодня у: {preview.steps.filter((s) => s.note).map((s) => name(s.step)).join(', ')} (они запустятся и сразу скажут, что делать нечего).
        </p>
      )}
      {preview.reads_after?.length > 0 && (
        <p className="text-xs text-slate-500">
          Эти шаги читают результат, но не зависят от него, поэтому не перезапускаются: {preview.reads_after.map(name).join(', ')}.
        </p>
      )}
      {preview.refusal && <p className="rounded border border-rose-300 bg-rose-50 p-2 text-rose-900">{preview.refusal}</p>}
    </div>
  )
}

function Dialog({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 z-20 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4" onClick={onClose}>
      <div className="mt-10 w-full max-w-2xl rounded-lg bg-white p-5 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <h3 className="mb-3 text-base font-semibold text-slate-900">{title}</h3>
        {children}
      </div>
    </div>
  )
}

function RunDialog({ steps, onClose, onDone }) {
  const [off, setOff] = useState([])
  const titles = Object.fromEntries(steps.map((s) => [s.id, s.title]))
  const preview = useQuery({
    queryKey: ['pipeline-preview', 'run', off],
    queryFn: () => previewPipeline({ off }),
    placeholderData: (previous) => previous,
  })
  const start = useMutation({ mutationFn: () => runPipeline({ off }), onSuccess: onDone })
  const toggle = (id) => setOff((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]))
  const agents = steps.filter((s) => s.tokens).map((s) => s.id)
  return (
    <Dialog title="Запустить весь пайплайн" onClose={onClose}>
      <p className="mb-2 text-xs text-slate-500">
        Снимите галочку у шага, который не нужен в этот раз. Шаги, зависящие от выключенного, не запустятся, если он сегодня уже не выполнялся.
      </p>
      <div className="mb-2 flex gap-3 text-xs">
        <button type="button" className="text-sky-700 hover:underline" onClick={() => setOff(agents)}>выключить все шаги с токенами</button>
        <button type="button" className="text-sky-700 hover:underline" onClick={() => setOff([])}>включить все</button>
      </div>
      <div className="mb-3 grid max-h-64 grid-cols-1 gap-x-4 overflow-y-auto rounded border border-slate-200 p-2 sm:grid-cols-2">
        {steps.map((s) => (
          <label key={s.id} className="flex items-center gap-2 py-0.5 text-sm">
            <input type="checkbox" checked={!off.includes(s.id)} onChange={() => toggle(s.id)} />
            <span className={off.includes(s.id) ? 'text-slate-400 line-through' : ''}>{s.title}</span>
            {s.tokens && <span className="rounded bg-amber-100 px-1 text-[10px] text-amber-900">токены</span>}
          </label>
        ))}
      </div>
      <PreviewBody preview={preview.data} titles={titles} />
      {start.isError && <p className="mt-2 rounded border border-rose-300 bg-rose-50 p-2 text-sm text-rose-900">{String(start.error.message)}</p>}
      <div className="mt-4 flex justify-end gap-2">
        <button type="button" className="rounded border border-slate-300 px-3 py-1.5 text-sm" onClick={onClose}>Отмена</button>
        <button
          type="button"
          disabled={start.isPending || !!preview.data?.refusal}
          className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
          onClick={() => start.mutate()}
        >
          {start.isPending ? 'Запускаю…' : preview.data?.token_steps?.length ? 'Запустить и потратить токены' : 'Запустить'}
        </button>
      </div>
    </Dialog>
  )
}

function RestartDialog({ detail, steps, onClose, onDone }) {
  const titles = Object.fromEntries(steps.map((s) => [s.id, s.title]))
  const preview = useQuery({
    queryKey: ['pipeline-preview', 'restart', detail.id],
    queryFn: () => previewPipeline({ restart: detail.id }),
  })
  const start = useMutation({ mutationFn: () => restartStep({ step: detail.id }), onSuccess: onDone })
  return (
    <Dialog title={`Перезапустить «${detail.title}»`} onClose={onClose}>
      <p className="mb-2 text-xs text-slate-500">
        Шаг и всё, что зависит от его результата, начнут заново: отметки «сделано» этих шагов в состоянии ночи стираются.
      </p>
      <PreviewBody preview={preview.data} titles={titles} />
      {start.isError && <p className="mt-2 rounded border border-rose-300 bg-rose-50 p-2 text-sm text-rose-900">{String(start.error.message)}</p>}
      <div className="mt-4 flex justify-end gap-2">
        <button type="button" className="rounded border border-slate-300 px-3 py-1.5 text-sm" onClick={onClose}>Отмена</button>
        <button
          type="button"
          disabled={start.isPending || !preview.data || !!preview.data?.refusal}
          className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
          onClick={() => start.mutate()}
        >
          {start.isPending ? 'Запускаю…' : preview.data?.token_steps?.length ? 'Перезапустить и потратить токены' : 'Перезапустить'}
        </button>
      </div>
    </Dialog>
  )
}

function Runs() {
  const query = useQuery({ queryKey: ['pipeline-runs'], queryFn: fetchPipelineRuns, refetchInterval: 15000 })
  return (
    <QueryState query={query}>
      {query.data && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-slate-500">
              <th className="py-1 pr-3 font-medium">Ночь</th>
              <th className="py-1 pr-3 font-medium">Запуск</th>
              <th className="py-1 pr-3 font-medium">Начат</th>
              <th className="py-1 pr-3 font-medium">Длился</th>
              <th className="py-1 pr-3 font-medium">Итог</th>
              <th className="py-1 font-medium">Выключено / состав</th>
            </tr>
          </thead>
          <tbody>
            {query.data.runs.map((r) => (
              <tr key={r.id} className="border-t border-slate-100 align-top">
                <td className="py-1 pr-3 tabular-nums">{r.night_date}{r.approximate ? ' ≈' : ''}</td>
                <td className="py-1 pr-3">{TRIGGER[r.trigger] ?? r.trigger}{r.restart ? ` · перезапуск ${r.restart}` : ''}</td>
                <td className="py-1 pr-3 tabular-nums">{when(r.started_at)}</td>
                <td className="py-1 pr-3 tabular-nums">
                  {duration(r.finished_at ? (new Date(r.finished_at) - new Date(r.started_at)) / 1000 : null)}
                </td>
                <td className="py-1 pr-3"><StateBadge state={r.state === 'ok' ? 'ok' : r.state === 'failed' ? 'failed' : r.state === 'running' ? 'running' : 'interrupted'} /></td>
                <td className="py-1 text-slate-500">
                  {r.steps_off.length ? `выключено: ${r.steps_off.join(', ')}` : ''}
                  {r.only ? ` · только: ${r.only.join(', ')}` : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </QueryState>
  )
}

export default function Pipeline() {
  const client = useQueryClient()
  const [selected, setSelected] = useState(null)
  const [free, setFree] = useState(false)
  const [dialog, setDialog] = useState(null) // 'run' | {restart: detail}
  const [notice, setNotice] = useState(null)
  const [, setTick] = useState(0) // a re-render each second while a run goes, for its elapsed time
  const query = useQuery({
    queryKey: ['pipeline'],
    queryFn: fetchPipeline,
    refetchInterval: (q) => (q.state.data?.running ? 3000 : 15000),
  })
  const running = !!query.data?.running
  useEffect(() => {
    if (!running) return undefined
    const t = setInterval(() => setTick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [running])
  const refresh = () => {
    client.invalidateQueries({ queryKey: ['pipeline'] })
    client.invalidateQueries({ queryKey: ['pipeline-step'] })
    client.invalidateQueries({ queryKey: ['pipeline-runs'] })
  }
  const started = (result) => {
    setDialog(null)
    setNotice(result?.pending ? result.note : 'Запуск начат на отдельном процессе: он не остановится, если закрыть консоль или перезапустить API.')
    refresh()
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Пайплайн ночи</h1>
        <p className="text-sm text-slate-600">
          Вся ночь по шагам: что от чего зависит, что прошло, что упало, что ждёт. Упавший шаг блокирует только тех, кто берёт его результат; остальные идут.
        </p>
      </div>

      <QueryState query={query}>
        {query.data && (() => {
          const { steps, states, run, holder } = query.data
          const byId = Object.fromEntries(steps.map((s) => [s.id, s]))
          const failed = steps.filter((s) => states[s.id]?.state === 'failed')
          const blocked = steps.filter((s) => states[s.id]?.state === 'blocked')
          const now = steps.find((s) => states[s.id]?.state === 'running')
          const elapsed = run && running ? (Date.now() - new Date(run.started_at)) / 1000 : null
          return (
            <>
              {!query.data.journal_ready && (
                <p className="rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
                  Таблицы журнала в реестре ещё не созданы (миграция не применена): виден только список шагов.
                </p>
              )}
              <Panel>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="space-y-1 text-sm">
                    {run ? (
                      <>
                        <div className="flex flex-wrap items-center gap-2">
                          <StateBadge state={run.state === 'ok' ? 'ok' : run.state === 'failed' ? 'failed' : run.state === 'running' ? 'running' : 'interrupted'} />
                          <span>
                            Ночь {run.night_date}, {TRIGGER[run.trigger] ?? run.trigger}
                            {run.restart ? `, перезапуск «${byId[run.restart]?.title ?? run.restart}»` : ''}
                            {run.approximate ? ' (восстановлено, времена приблизительные)' : ''}
                          </span>
                        </div>
                        <div className="text-xs text-slate-500">
                          начат {when(run.started_at)}
                          {run.finished_at ? `, закончен ${when(run.finished_at)}` : ''}
                          {elapsed != null ? `, идёт ${duration(elapsed)}` : ''}
                          {now ? `, сейчас: «${now.title}»` : ''}
                        </div>
                        {(failed.length > 0 || blocked.length > 0) && (
                          <div className="text-xs">
                            {failed.length > 0 && <span className="text-rose-800">Упало: {failed.map((s) => s.title).join(', ')}. </span>}
                            {blocked.length > 0 && <span className="text-amber-800">Заблокировано: {blocked.map((s) => s.title).join(', ')}.</span>}
                          </div>
                        )}
                        {run.steps_off.length > 0 && <div className="text-xs text-slate-500">Выключено в этом запуске: {run.steps_off.map((id) => byId[id]?.title ?? id).join(', ')}.</div>}
                      </>
                    ) : (
                      <span className="text-slate-600">Запусков в журнале ещё нет.</span>
                    )}
                  </div>
                  <div className="text-right">
                    <button
                      type="button"
                      disabled={running}
                      onClick={() => setDialog('run')}
                      className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
                    >
                      Запустить весь пайплайн…
                    </button>
                    {running && (
                      <div className="mt-1 max-w-xs text-xs text-slate-500">
                        Идёт запуск (процесс {holder?.pid ?? '?'}); второй сейчас не запустить.
                      </div>
                    )}
                  </div>
                </div>
                {notice && <p className="mt-3 rounded border border-sky-200 bg-sky-50 p-2 text-xs text-sky-900">{notice}</p>}
              </Panel>

              <Panel
                title="Дерево шагов"
                subtitle="Колонка — волна: шаги одной волны ничем не связаны друг с другом и могли бы идти одновременно. Нажмите на шаг: описание, зависимости, статистика, история."
              >
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <Legend />
                  <label className="flex items-center gap-2 text-xs text-slate-700">
                    <input type="checkbox" checked={free} onChange={(e) => setFree(e.target.checked)} />
                    не держать порядок трат токенов (что было бы без него)
                  </label>
                </div>
                <Graph steps={steps} states={states} selected={selected} onSelect={setSelected} running={running} free={free} />
                <p className="mt-1 text-xs text-slate-500">
                  «Утренний отчёт» читает все шаги и не показан линиями: он ждёт всех, но упавший шаг его не блокирует.
                </p>
              </Panel>

              {selected && (
                <Panel>
                  <StepDetail
                    stepId={selected}
                    steps={steps}
                    onSelect={setSelected}
                    busy={running}
                    onRestart={(detail) => setDialog({ restart: detail })}
                  />
                </Panel>
              )}

              <Panel title="Что занимает время" subtitle="Шаги по убыванию длительности в последней ночи. «≈» — по записям прошлых ночей, без точного замера.">
                <TimeTable steps={steps} states={states} onSelect={setSelected} />
              </Panel>

              <Panel title="Запуски">
                <Runs />
              </Panel>

              {dialog === 'run' && <RunDialog steps={steps} onClose={() => setDialog(null)} onDone={started} />}
              {dialog?.restart && (
                <RestartDialog detail={dialog.restart} steps={steps} onClose={() => setDialog(null)} onDone={started} />
              )}
            </>
          )
        })()}
      </QueryState>
    </div>
  )
}
