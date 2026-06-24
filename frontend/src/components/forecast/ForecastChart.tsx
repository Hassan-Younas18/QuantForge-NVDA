import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ForecastPoint } from '../../types/api'

export function ForecastChart({
  path,
  lastClose,
}: {
  path: ForecastPoint[]
  lastClose: number
}) {
  const data = [
    { date: 'Today', pred_close: lastClose, lower: lastClose, upper: lastClose },
    ...path,
  ]

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#8b949e' }} minTickGap={30} />
        <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11, fill: '#8b949e' }} width={56} />
        <Tooltip
          contentStyle={{ background: '#151b23', border: '1px solid #21262d', fontSize: 12 }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Area dataKey="upper" stroke="none" fill="#76b900" fillOpacity={0.08} isAnimationActive={false} />
        <Area dataKey="lower" stroke="none" fill="#0d1117" fillOpacity={1} isAnimationActive={false} />
        <Line
          dataKey="pred_close"
          stroke="#76b900"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="upper"
          stroke="#30363d"
          strokeDasharray="3 3"
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="lower"
          stroke="#30363d"
          strokeDasharray="3 3"
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
