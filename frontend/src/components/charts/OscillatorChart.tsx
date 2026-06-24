import {
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  ComposedChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { MergedBar } from '../../hooks/useStockData'

export function OscillatorChart({ data, mode }: { data: MergedBar[]; mode: 'RSI' | 'MACD' }) {
  return (
    <ResponsiveContainer width="100%" height={140}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="Date" tick={{ fontSize: 10, fill: '#8b949e' }} minTickGap={60} />
        <YAxis tick={{ fontSize: 10, fill: '#8b949e' }} width={36} />
        <Tooltip
          contentStyle={{ background: '#151b23', border: '1px solid #21262d', fontSize: 12 }}
          labelStyle={{ color: '#8b949e' }}
        />
        {mode === 'RSI' ? (
          <>
            <ReferenceLine y={70} stroke="#f87171" strokeDasharray="3 3" />
            <ReferenceLine y={30} stroke="#34d399" strokeDasharray="3 3" />
            <Line dataKey="RSI_14" stroke="#60a5fa" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </>
        ) : (
          <>
            <ReferenceLine y={0} stroke="#30363d" />
            <Line dataKey="MACD" stroke="#60a5fa" dot={false} strokeWidth={1.25} isAnimationActive={false} />
            <Line dataKey="MACD_signal" stroke="#f59e0b" dot={false} strokeWidth={1.25} isAnimationActive={false} />
            <Line dataKey="MACD_hist" stroke="#76b900" dot={false} strokeWidth={1} isAnimationActive={false} />
          </>
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
