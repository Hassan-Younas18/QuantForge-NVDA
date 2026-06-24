import { useState } from 'react'
import { useStockData } from '../hooks/useStockData'
import { Card } from '../components/common/Card'
import { ErrorBanner } from '../components/common/ErrorBanner'
import { LoadingSpinner } from '../components/common/LoadingSpinner'
import { PriceVolumeChart } from '../components/charts/PriceVolumeChart'
import { IndicatorToggle } from '../components/charts/IndicatorToggle'
import { OscillatorChart } from '../components/charts/OscillatorChart'

const RANGES = [
  { label: '1Y', years: 1 },
  { label: '2Y', years: 2 },
  { label: '5Y', years: 5 },
  { label: '10Y', years: 10 },
]

export function HistoryView({ ticker }: { ticker: string }) {
  const [years, setYears] = useState(2)
  const [overlays, setOverlays] = useState<string[]>(['SMA_50', 'SMA_200'])
  const stock = useStockData(ticker, years)

  return (
    <div className="space-y-4">
      <Card
        title="Historical price & volume"
        action={
          <div className="flex gap-1">
            {RANGES.map((r) => (
              <button
                key={r.label}
                onClick={() => setYears(r.years)}
                className={`rounded-md px-2 py-1 text-xs font-medium transition ${
                  years === r.years
                    ? 'bg-nvda-green text-black'
                    : 'text-gray-400 hover:bg-surface-raised'
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
        }
      >
        <div className="mb-3">
          <IndicatorToggle active={overlays} onChange={setOverlays} />
        </div>
        {stock.loading ? (
          <LoadingSpinner label="Loading historical data..." />
        ) : stock.error ? (
          <ErrorBanner message={stock.error} onRetry={stock.refetch} />
        ) : (
          <>
            <PriceVolumeChart data={stock.data} overlays={overlays} />
            <p className="mt-2 text-center text-xs text-gray-500">
              Drag the handles below the chart to zoom into a date range.
            </p>
          </>
        )}
      </Card>

      {!stock.loading && !stock.error && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card title="RSI (14)">
            <OscillatorChart data={stock.data} mode="RSI" />
          </Card>
          <Card title="MACD">
            <OscillatorChart data={stock.data} mode="MACD" />
          </Card>
        </div>
      )}
    </div>
  )
}
