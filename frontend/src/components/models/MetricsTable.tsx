import type { ModelMetric } from '../../types/api'

export function MetricsTable({
  metrics,
  bestModel,
}: {
  metrics: Record<string, ModelMetric>
  bestModel: string
}) {
  const rows = Object.entries(metrics).sort(([, a], [, b]) => a.rmse - b.rmse)

  return (
    <div className="overflow-x-auto rounded-lg border border-surface-border">
      <table className="w-full text-left text-sm">
        <thead className="bg-surface-raised text-xs text-gray-500">
          <tr>
            <th className="px-3 py-2">Model</th>
            <th className="px-3 py-2">RMSE</th>
            <th className="px-3 py-2">MAE</th>
            <th className="px-3 py-2">MAPE</th>
            <th className="px-3 py-2">R²</th>
            <th className="px-3 py-2">Dir. Acc.</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-surface-border">
          {rows.map(([name, m]) => (
            <tr key={name} className={name === bestModel ? 'bg-nvda-green/10' : undefined}>
              <td className="px-3 py-2 font-medium text-gray-200">
                {name === bestModel && '🏆 '}
                {name}
              </td>
              <td className="px-3 py-2 font-mono text-gray-300">{m.rmse.toFixed(4)}</td>
              <td className="px-3 py-2 font-mono text-gray-300">{m.mae.toFixed(4)}</td>
              <td className="px-3 py-2 font-mono text-gray-300">{m.mape.toFixed(2)}%</td>
              <td className="px-3 py-2 font-mono text-gray-300">{m.r2.toFixed(4)}</td>
              <td className="px-3 py-2 font-mono text-gray-300">
                {m.dir_acc != null ? `${m.dir_acc.toFixed(1)}%` : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
