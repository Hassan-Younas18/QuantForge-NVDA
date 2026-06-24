import { useTrainJob } from '../../hooks/useTrainJob'

const MODEL_OPTIONS = ['lstm', 'gru', 'bilstm', 'cnn_lstm', 'transformer']

export function TrainTrigger({ ticker, onComplete }: { ticker: string; onComplete: () => void }) {
  const { start, status, starting, isRunning, error } = useTrainJob(onComplete)

  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        onClick={() =>
          start({ ticker, years: 10, target: 'log_return', models: MODEL_OPTIONS, epochs: 80 })
        }
        disabled={starting || isRunning}
        className="rounded-md bg-nvda-green px-4 py-2 text-sm font-semibold text-black transition hover:bg-nvda-green-dim disabled:opacity-50"
      >
        {isRunning ? 'Training...' : 'Run new training (bake-off)'}
      </button>

      {isRunning && (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <span className="h-3 w-3 animate-spin rounded-full border-2 border-gray-600 border-t-nvda-green" />
          {status?.message}
        </div>
      )}
      {status?.status === 'completed' && (
        <span className="text-sm text-nvda-green">{status.message}</span>
      )}
      {(status?.status === 'failed' || error) && (
        <span className="text-sm text-red-400">{status?.error ?? error}</span>
      )}
    </div>
  )
}
