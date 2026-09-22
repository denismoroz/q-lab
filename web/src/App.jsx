import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import Funnel from './pages/Funnel.jsx'
import Registry from './pages/Registry.jsx'
import IdeaDetail from './pages/IdeaDetail.jsx'
import Cards from './pages/Cards.jsx'
import Trials from './pages/Trials.jsx'

const NAV = [
  { to: '/funnel', text: 'Воронка' },
  { to: '/ideas', text: 'Реестр и кладбище' },
  { to: '/cards', text: 'Карточки кандидатов' },
  { to: '/trials', text: 'Прогоны' },
]

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-baseline gap-x-6 gap-y-2 px-4 py-3">
          <span className="text-base font-semibold">q-lab</span>
          <span className="text-xs text-slate-500">
            только чтение — консоль ничего не меняет в реестре
          </span>
          <nav className="flex flex-wrap gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded px-3 py-1.5 text-sm ${
                    isActive
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-700 hover:bg-slate-100'
                  }`
                }
              >
                {item.text}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/funnel" replace />} />
          <Route path="/funnel" element={<Funnel />} />
          <Route path="/ideas" element={<Registry />} />
          <Route path="/ideas/:ideaId" element={<IdeaDetail />} />
          <Route path="/cards" element={<Cards />} />
          <Route path="/cards/:ideaId" element={<Cards />} />
          <Route path="/trials" element={<Trials />} />
          <Route path="*" element={<p className="text-sm">Страница не найдена.</p>} />
        </Routes>
      </main>
    </div>
  )
}
