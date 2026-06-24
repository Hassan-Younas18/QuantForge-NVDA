export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-red-900/50 bg-red-950/40 px-4 py-3 text-sm text-red-300">
      <span>{message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="shrink-0 rounded-md border border-red-800 px-2 py-1 text-xs hover:bg-red-900/40"
        >
          Retry
        </button>
      )}
    </div>
  )
}
