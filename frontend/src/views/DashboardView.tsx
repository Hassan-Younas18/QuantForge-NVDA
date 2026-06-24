import { api } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import { useStockData } from '../hooks/useStockData'
import { Card } from '../components/common/Card'
import { ErrorBanner } from '../components/common/ErrorBanner'
import { LoadingSpinner } from '../components/common/LoadingSpinner'
import { StatsGrid } from '../components/dashboard/StatsGrid'
import { PriceVolumeChart } from '../components/charts/PriceVolumeChart'
import type { StockInfo } from '../types/api'

export function DashboardView({ ticker, info }: { ticker: string; info: StockInfo | null }) {
  const eda = useAsync(() => api.eda(), [ticker])
  const stock = useStockData(ticker, 1)

  if (!info) return <LoadingSpinner label="Loading dashboard..." />

  return (
    <div className="space-y-4">
      <StatsGrid info={info} eda={eda.data} />

      <Card title="Price — last 12 months">
        {stock.loading ? (
          <LoadingSpinner />
        ) : stock.error ? (
          <ErrorBanner message={stock.error} onRetry={stock.refetch} />
        ) : (
          <PriceVolumeChart data={stock.data} overlays={['SMA_50']} height={320} />
        )}
      </Card>
    </div>
  )
}
