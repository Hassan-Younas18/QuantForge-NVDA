import type { StockInfo } from '../../types/api'
import { ChangeBadge } from '../common/ChangeBadge'
import { ThemeToggle } from './ThemeToggle'

export const SECTIONS = ['Dashboard', 'History', 'Predictions', 'Models', 'Insights'] as const
export type Section = (typeof SECTIONS)[number]

export function Header({
  info,
  section,
  onSectionChange,
  dark,
  onToggleDark,
}: {
  info: StockInfo | null
  section: Section
  onSectionChange: (s: Section) => void
  dark: boolean
  onToggleDark: () => void
}) {
  return (
    <header className="sticky top-0 z-10 border-b border-surface-border bg-surface/95 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-nvda-green text-sm font-bold text-black">
            N
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold text-gray-100">
                {info?.short_name ?? 'NVIDIA Corporation'}
              </h1>
              <span className="rounded bg-surface-border px-1.5 py-0.5 text-xs text-gray-400">
                {info?.ticker ?? 'NVDA'}
              </span>
            </div>
            {info && (
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span className="font-mono text-gray-200">${info.last_close.toFixed(2)}</span>
                <ChangeBadge value={info.change_pct} />
              </div>
            )}
          </div>
        </div>

        <nav className="flex flex-1 justify-center gap-1 overflow-x-auto sm:flex-none">
          {SECTIONS.map((s) => (
            <button
              key={s}
              onClick={() => onSectionChange(s)}
              className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition ${
                section === s
                  ? 'bg-nvda-green text-black'
                  : 'text-gray-400 hover:bg-surface-raised hover:text-gray-200'
              }`}
            >
              {s}
            </button>
          ))}
        </nav>

        <ThemeToggle dark={dark} onToggle={onToggleDark} />
      </div>
    </header>
  )
}
