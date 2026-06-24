export function ChangeBadge({ value, suffix = '%' }: { value: number; suffix?: string }) {
  const positive = value >= 0
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-sm font-medium ${
        positive ? 'bg-nvda-green/15 text-nvda-green' : 'bg-red-500/15 text-red-400'
      }`}
    >
      {positive ? '▲' : '▼'} {Math.abs(value).toFixed(2)}
      {suffix}
    </span>
  )
}
