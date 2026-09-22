import { useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { fetchTrials } from '../api.js'
import { Pager, Panel, QueryState } from '../components/ui.jsx'
import TrialTable from '../components/TrialTable.jsx'

const PAGE_SIZE = 50

export default function Trials() {
  const [offset, setOffset] = useState(0)
  // Paginated on the server: the ledger holds thousands of rows and is
  // never loaded whole.
  const query = useQuery({
    queryKey: ['trials', offset],
    queryFn: () => fetchTrials({ limit: PAGE_SIZE, offset }),
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
