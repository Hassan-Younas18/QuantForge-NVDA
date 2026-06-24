import { useState } from 'react'
import { api } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import { Card } from '../components/common/Card'
import { ErrorBanner } from '../components/common/ErrorBanner'
import { LoadingSpinner } from '../components/common/LoadingSpinner'
import { ModelComparisonChart } from '../components/models/ModelComparisonChart'
import { MetricsTable } from '../components/models/MetricsTable'
import { FeatureImportanceChart } from '../components/models/FeatureImportanceChart'
import { TrainTrigger } from '../components/models/TrainTrigger'
import type { ModelMetric } from '../types/api'

const METRIC_TABS: { key: keyof ModelMetric; label: string }[] = [
  { key: 'rmse', label: 'RMSE' },
  { key: 'mae', label: 'MAE' },
  { key: 'mape', label: 'MAPE' },
  { key: 'dir_acc', label: 'Directional Accuracy' },
]

export function ModelsView({ ticker }: { ticker: string }) {
  const [metricKey, setMetricKey] = useState<keyof ModelMetric>('rmse')
  const comparison = useAsync(() => api.modelComparison(), [])
  const importance = useAsync(() => api.modelImportance(), [])

  return (
    <div className="space-y-4">
      <Card title="Train a new model bake-off">
        <TrainTrigger
          ticker={ticker}
          onComplete={() => {
            comparison.refetch()
            importance.refetch()
          }}
        />
        <p className="mt-2 text-xs text-gray-500">
          Trains LSTM, GRU, BiLSTM, CNN-LSTM and Transformer on {ticker}, selects the winner by
          validation RMSE. Takes a few minutes on CPU.
        </p>
      </Card>

      {comparison.loading ? (
        <LoadingSpinner label="Loading model comparison..." />
      ) : comparison.error ? (
        <ErrorBanner message={comparison.error} onRetry={comparison.refetch} />
      ) : comparison.data ? (
        <>
          <Card
            title="Model comparison"
            action={
              <div className="flex gap-1">
                {METRIC_TABS.map((t) => (
                  <button
                    key={t.key}
                    onClick={() => setMetricKey(t.key)}
                    className={`rounded-md px-2 py-1 text-xs font-medium transition ${
                      metricKey === t.key
                        ? 'bg-nvda-green text-black'
                        : 'text-gray-400 hover:bg-surface'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            }
          >
            <ModelComparisonChart
              metrics={comparison.data.metrics}
              metricKey={metricKey}
              bestModel={comparison.data.best_model}
            />
          </Card>

          <Card title="Detailed metrics">
            <MetricsTable metrics={comparison.data.metrics} bestModel={comparison.data.best_model} />
          </Card>
        </>
      ) : null}

      {importance.data && (
        <Card title="Top feature importance (permutation)">
          <FeatureImportanceChart importance={importance.data} />
        </Card>
      )}
    </div>
  )
}
