const AVAILABLE = [
  { key: 'SMA_20', label: 'SMA 20' },
  { key: 'SMA_50', label: 'SMA 50' },
  { key: 'SMA_200', label: 'SMA 200' },
  { key: 'EMA_20', label: 'EMA 20' },
  { key: 'EMA_50', label: 'EMA 50' },
  { key: 'BB_upper', label: 'Bollinger' },
]

export function IndicatorToggle({
  active,
  onChange,
}: {
  active: string[]
  onChange: (next: string[]) => void
}) {
  const toggle = (key: string) => {
    if (key === 'BB_upper') {
      const on = active.includes('BB_upper')
      onChange(
        on
          ? active.filter((k) => k !== 'BB_upper' && k !== 'BB_lower')
          : [...active, 'BB_upper', 'BB_lower'],
      )
      return
    }
    onChange(active.includes(key) ? active.filter((k) => k !== key) : [...active, key])
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {AVAILABLE.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => toggle(key)}
          className={`rounded-md border px-2 py-1 text-xs font-medium transition ${
            active.includes(key)
              ? 'border-nvda-green bg-nvda-green/15 text-nvda-green'
              : 'border-surface-border text-gray-400 hover:bg-surface-raised'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
