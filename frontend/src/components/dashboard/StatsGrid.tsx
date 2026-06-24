import type { EdaResponse, StockInfo } from '../../types/api'
import { Card } from '../common/Card'

function fmtUsd(n: number | null | undefined, digits = 2) {
  if (n == null) return '—'
  return `$${n.toFixed(digits)}`
}

function fmtBig(n: number | null | undefined) {
  if (n == null) return '—'
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
  return `$${n.toFixed(0)}`
}

export function StatsGrid({ info, eda }: { info: StockInfo; eda: EdaResponse | null }) {
  const stats = [
    { label: 'Market Cap', value: fmtBig(info.market_cap) },
    { label: '52-Week Range', value: `${fmtUsd(info.fifty_two_week_low)} – ${fmtUsd(info.fifty_two_week_high)}` },
    { label: 'Previous Close', value: fmtUsd(info.previous_close) },
    { label: 'Volume', value: info.volume ? info.volume.toLocaleString() : '—' },
    { label: 'Sector', value: info.sector ?? '—' },
    { label: 'Industry', value: info.industry ?? '—' },
    { label: 'Annualised Volatility', value: eda ? `${eda.annual_vol_pct.toFixed(1)}%` : '—' },
    { label: 'CAGR (analysed window)', value: eda ? `${eda.cagr_pct.toFixed(1)}%` : '—' },
  ]

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {stats.map((s) => (
        <Card key={s.label} className="!p-3">
          <div className="text-xs text-gray-500">{s.label}</div>
          <div className="mt-1 text-sm font-semibold text-gray-100">{s.value}</div>
        </Card>
      ))}
    </div>
  )
}
