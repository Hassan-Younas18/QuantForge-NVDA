import { useCallback, useRef, useState } from 'react'
import { api } from '../api/client'
import type { TrainRequest, TrainStatusResponse } from '../types/api'

const POLL_MS = 3000

export function useTrainJob(onComplete?: () => void) {
  const [status, setStatus] = useState<TrainStatusResponse | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  const start = useCallback(
    async (req: TrainRequest) => {
      setStarting(true)
      setError(null)
      try {
        const { job_id } = await api.startTraining(req)
        intervalRef.current = setInterval(async () => {
          try {
            const s = await api.trainingStatus(job_id)
            setStatus(s)
            if (s.status === 'completed' || s.status === 'failed') {
              stopPolling()
              if (s.status === 'completed') onComplete?.()
            }
          } catch {
            stopPolling()
          }
        }, POLL_MS)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to start training.')
      } finally {
        setStarting(false)
      }
    },
    [onComplete, stopPolling],
  )

  const isRunning = status?.status === 'pending' || status?.status === 'running'

  return { start, status, starting, isRunning, error }
}
