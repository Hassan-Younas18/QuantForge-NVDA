import {
  Area,
  Bar,
  Brush,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { MergedBar } from '../../hooks/useStockData'

const OVERLAY_COLORS: Record<string, string> = {
  SMA_10: '#60a5fa',
  SMA_20: '#f59e0b',
  SMA_50: '#a78bfa',
  SMA_200: '#f472b6',
  EMA_10: '#34d399',
  EMA_20: '#22d3ee',
  EMA_50: '#fb923c',
  BB_upper: '#94a3b8',
  BB_lower: '#94a3b8',
}

export function PriceVolumeChart({
  data,
  overlays,
  showVolume = true,
  height = 380,
}: {
  data: MergedBar[]
  overlays: string[]
  showVolume?: boolean
  height?: number
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="Date" tick={{ fontSize: 11, fill: '#8b949e' }} minTickGap={40} />
        <YAxis
          yAxisId="price"
          domain={['auto', 'auto']}
          tick={{ fontSize: 11, fill: '#8b949e' }}
          width={56}
        />
        {showVolume && (
          <YAxis yAxisId="volume" orientation="right" hide tick={{ fontSize: 11 }} />
        )}
        <Tooltip
          contentStyle={{ background: '#151b23', border: '1px solid #21262d', fontSize: 12 }}
          labelStyle={{ color: '#8b949e' }}
        />
        {showVolume && (
          <Bar yAxisId="volume" dataKey="Volume" fill="#21262d" barSize={2} isAnimationActive={false} />
        )}
        <Area
          yAxisId="price"
          dataKey="Close"
          stroke="#76b900"
          fill="#76b900"
          fillOpacity={0.08}
          strokeWidth={1.5}
          isAnimationActive={false}
          dot={false}
        />
        {overlays.map((key) => (
          <Line
            key={key}
            yAxisId="price"
            dataKey={key}
            stroke={OVERLAY_COLORS[key] ?? '#e2e8f0'}
            strokeWidth={1.25}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        ))}
        <Brush
          dataKey="Date"
          height={24}
          stroke="#76b900"
          fill="#151b23"
          travellerWidth={8}
          tickFormatter={() => ''}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
