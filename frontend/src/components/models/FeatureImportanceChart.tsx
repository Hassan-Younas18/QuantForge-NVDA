import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export function FeatureImportanceChart({ importance }: { importance: Record<string, number> }) {
  const data = Object.entries(importance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([name, value]) => ({ name, value }))

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis type="number" tick={{ fontSize: 11, fill: '#8b949e' }} />
        <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 11, fill: '#8b949e' }} />
        <Tooltip
          contentStyle={{ background: '#151b23', border: '1px solid #21262d', fontSize: 12 }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Bar dataKey="value" fill="#76b900" radius={[0, 4, 4, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}
