import { useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { fetchTrials } from '../api.js'
import { Pager, Panel, QueryState } from '../components/ui.jsx'
import TrialTable from '../components/TrialTable.jsx'

const PAGE_SIZE = 50

export default function Trials() {
  const [offset, setOffset] = useState(0)
  // Random books are q-lab's measuring instrument: nine in ten runs are
  // theirs, and by default the list shows the strategies' own.
  const [noise, setNoise] = useState(false)
  // Paginated on the server: the ledger holds thousands of rows and is
  // never loaded whole.
  const query = useQuery({
    queryKey: ['trials', offset, noise],
    queryFn: () => fetchTrials({ limit: PAGE_SIZE, offset, noise }),
    placeholderData: keepPreviousData,
  })

  return (
    <Panel
      title="Прогоны"
      subtitle="Журнал, новые сверху. Пишется каждый прогон, включая выброшенные — по ним считается дефляция."
    >
      <QueryState query={query}>
        {query.data && (
          <>
            <label className="mb-3 flex items-center gap-2 text-xs text-slate-600">
              <input
                type="checkbox"
                checked={noise}
                onChange={(e) => {
                  setNoise(e.target.checked)
                  setOffset(0)
                }}
              />
              показывать прогоны случайных книг ({query.data.noise_trials.toLocaleString('ru-RU')}{' '}
              — с ними сравнивается каждая стратегия; книга, не подошедшая по форме, записана
              как «не оценима», а не как ошибка)
            </label>
            <Pager
              total={query.data.total}
              limit={query.data.limit}
              offset={query.data.offset}
              onChange={setOffset}
            />
            <TrialTable trials={query.data.items} />
            <Pager
              total={query.data.total}
              limit={query.data.limit}
              offset={query.data.offset}
              onChange={setOffset}
            />
          </>
        )}
      </QueryState>
    </Panel>
  )
}
