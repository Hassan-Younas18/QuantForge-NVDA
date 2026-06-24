import type { InsightsResponse } from '../../types/api'
import { Card } from '../common/Card'

const CONFIDENCE_STYLES: Record<string, string> = {
  moderate: 'bg-nvda-green/15 text-nvda-green',
  low: 'bg-amber-500/15 text-amber-400',
  'very low': 'bg-red-500/15 text-red-400',
  unknown: 'bg-gray-500/15 text-gray-400',
}

export function InsightsPanel({ insights }: { insights: InsightsResponse }) {
  const badgeClass = CONFIDENCE_STYLES[insights.confidence] ?? CONFIDENCE_STYLES.unknown

  return (
    <div className="space-y-4">
      <Card title="Why this model was selected">
        <p className="text-sm leading-relaxed text-gray-300">{insights.why_selected}</p>
      </Card>

      <Card title="Recent trend">
        <p className="text-sm leading-relaxed text-gray-300">{insights.trend_summary}</p>
      </Card>

      <Card title="Confidence level">
        <span className={`inline-block rounded-md px-3 py-1 text-sm font-semibold capitalize ${badgeClass}`}>
          {insights.confidence}
        </span>
        {insights.confidence_note && (
          <p className="mt-2 text-xs text-gray-500">{insights.confidence_note}</p>
        )}
      </Card>
    </div>
  )
}
