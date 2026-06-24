import { useEffect, useState } from 'react'
import { api } from './api/client'
import { useAsync } from './hooks/useAsync'
import { Header, SECTIONS, type Section } from './components/layout/Header'
import { DashboardView } from './views/DashboardView'
import { HistoryView } from './views/HistoryView'
import { PredictionsView } from './views/PredictionsView'
import { ModelsView } from './views/ModelsView'
import { InsightsView } from './views/InsightsView'

const TICKER = 'NVDA'

function App() {
  const [section, setSection] = useState<Section>(SECTIONS[0])
  const [dark, setDark] = useState(true)
  const info = useAsync(() => api.stockInfo(TICKER), [])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  return (
    <div className="min-h-full bg-surface text-gray-200">
      <Header
        info={info.data}
        section={section}
        onSectionChange={setSection}
        dark={dark}
        onToggleDark={() => setDark((d) => !d)}
      />

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        {section === 'Dashboard' && <DashboardView ticker={TICKER} info={info.data} />}
        {section === 'History' && <HistoryView ticker={TICKER} />}
        {section === 'Predictions' && <PredictionsView />}
        {section === 'Models' && <ModelsView ticker={TICKER} />}
        {section === 'Insights' && <InsightsView />}
      </main>

      <footer className="px-4 py-6 text-center text-xs text-gray-600 sm:px-6">
        Research &amp; engineering portfolio project — not investment advice.
      </footer>
    </div>
  )
}

export default App
