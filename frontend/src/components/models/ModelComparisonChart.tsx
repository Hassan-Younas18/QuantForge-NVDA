import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ModelMetric } from '../../types/api'

export function ModelComparisonChart({
  metrics,
  metricKey,
  bestModel,
}: {
  metrics: Record<string, ModelMetric>
  metricKey: keyof ModelMetric
  bestModel: string
}) {
  const data = Object.entries(metrics)
    .map(([name, m]) => ({ name, value: m[metricKey] ?? 0 }))
    .sort((a, b) => a.value - b.value)

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#8b949e' }} />
        <YAxis tick={{ fontSize: 11, fill: '#8b949e' }} width={48} />
        <Tooltip
          contentStyle={{ background: '#151b23', border: '1px solid #21262d', fontSize: 12 }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell
              key={d.name}
              fill={
                d.name === bestModel ? '#76b900' : d.name === 'naive_rw' ? '#6b7280' : '#3b82f6'
              }
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
