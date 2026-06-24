import type { ForecastPoint } from '../../types/api'
import { ChangeBadge } from '../common/ChangeBadge'

export function ForecastTable({
  path,
  lastClose,
}: {
  path: ForecastPoint[]
  lastClose: number
}) {
  return (
    <div className="max-h-64 overflow-y-auto rounded-lg border border-surface-border">
      <table className="w-full text-left text-sm">
        <thead className="sticky top-0 bg-surface-raised text-xs text-gray-500">
          <tr>
            <th className="px-3 py-2">Date</th>
            <th className="px-3 py-2">Predicted Close</th>
            <th className="px-3 py-2">90% CI</th>
            <th className="px-3 py-2">vs Today</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-surface-border">
          {path.map((p) => (
            <tr key={p.date}>
              <td className="px-3 py-2 text-gray-300">{p.date}</td>
              <td className="px-3 py-2 font-mono text-gray-100">${p.pred_close.toFixed(2)}</td>
              <td className="px-3 py-2 font-mono text-xs text-gray-500">
                ${p.lower.toFixed(2)} – ${p.upper.toFixed(2)}
              </td>
              <td className="px-3 py-2">
                <ChangeBadge value={((p.pred_close - lastClose) / lastClose) * 100} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
