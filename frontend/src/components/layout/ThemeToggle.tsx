export function ThemeToggle({
  dark,
  onToggle,
}: {
  dark: boolean
  onToggle: () => void
}) {
  return (
    <button
      onClick={onToggle}
      aria-label="Toggle dark mode"
      className="rounded-md border border-surface-border p-2 text-gray-300 transition hover:bg-surface-raised"
    >
      {dark ? '☀️' : '🌙'}
    </button>
  )
}
