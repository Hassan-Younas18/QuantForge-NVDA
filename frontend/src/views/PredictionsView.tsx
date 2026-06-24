import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import { Card } from '../components/common/Card'
import { ErrorBanner } from '../components/common/ErrorBanner'
import { LoadingSpinner } from '../components/common/LoadingSpinner'
import { ChangeBadge } from '../components/common/ChangeBadge'
import { ForecastChart } from '../components/forecast/ForecastChart'
import { ForecastTable } from '../components/forecast/ForecastTable'
import type { ForecastResponse } from '../types/api'

const HORIZONS: { key: string; label: string; days: number }[] = [
  { key: 'day_1', label: '1 Day', days: 1 },
  { key: 'day_7', label: '7 Days', days: 7 },
  { key: 'day_30', label: '30 Days', days: 30 },
]

export function PredictionsView() {
  const [activeHorizon, setActiveHorizon] = useState('day_1')
  const [data, setData] = useState<ForecastResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = (refresh: boolean) => {
    const setBusy = refresh ? setRefreshing : setLoading
    setBusy(true)
    setError(null)
    api
      .forecast(refresh)
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Something went wrong.'))
      .finally(() => setBusy(false))
  }

  useEffect(() => load(false), [])

  if (loading) return <LoadingSpinner label="Loading forecast..." />
  if (error) return <ErrorBanner message={error} onRetry={() => load(false)} />
  if (!data) return null

  const { model_name, last_close, path, summary } = data
  const activeSummary = summary[activeHorizon]
  const activeDays = HORIZONS.find((h) => h.key === activeHorizon)!.days
  const visiblePath = path.slice(0, activeDays)

  return (
    <div className="space-y-4">
      <Card
        title={`Forecast — ${model_name.toUpperCase()}`}
        action={
          <button
            onClick={() => load(true)}
            disabled={refreshing}
            className="rounded-md border border-surface-border px-3 py-1 text-xs text-gray-300 hover:bg-surface disabled:opacity-50"
          >
            {refreshing ? 'Refreshing...' : 'Refresh from latest data'}
          </button>
        }
      >
        <div className="mb-4 flex gap-2">
          {HORIZONS.map((h) => (
            <button
              key={h.key}
              onClick={() => setActiveHorizon(h.key)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                activeHorizon === h.key
                  ? 'bg-nvda-green text-black'
                  : 'border border-surface-border text-gray-400 hover:bg-surface-raised'
              }`}
            >
              {h.label}
            </button>
          ))}
        </div>

        {activeSummary && (
          <div className="mb-4 flex flex-wrap items-baseline gap-4">
            <div>
              <div className="text-xs text-gray-500">Predicted close on {activeSummary.date}</div>
              <div className="text-2xl font-bold text-gray-100">
                ${activeSummary.pred_close.toFixed(2)}
              </div>
            </div>
            <ChangeBadge value={activeSummary.pct_change} />
            <div className="text-xs text-gray-500">
              90% CI: ${activeSummary.lower.toFixed(2)} – ${activeSummary.upper.toFixed(2)}
            </div>
          </div>
        )}

        <ForecastChart path={visiblePath} lastClose={last_close} />
      </Card>

      <Card title="Day-by-day path">
        <ForecastTable path={visiblePath} lastClose={last_close} />
      </Card>

      <p className="text-xs text-gray-500">
        Recursive forecasts compound error with horizon — treat the 7/30-day paths as
        scenario illustrations, not tradeable signals. Intervals reflect MC-Dropout model
        uncertainty, not full market risk.
      </p>
    </div>
  )
}
