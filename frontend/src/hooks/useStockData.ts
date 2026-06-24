import { useMemo } from 'react'
import { api } from '../api/client'
import { useAsync } from './useAsync'
import type { IndicatorBar, OhlcvBar } from '../types/api'

export interface MergedBar extends OhlcvBar, Omit<IndicatorBar, 'Date' | 'Close'> {}

/** History + indicators merged on Date, so one chart can show price, volume and overlays. */
export function useStockData(ticker: string, years: number) {
  const history = useAsync<OhlcvBar[]>(() => api.stockHistory(ticker, years), [ticker, years])
  const indicators = useAsync<IndicatorBar[]>(
    () => api.stockIndicators(ticker, years),
    [ticker, years],
  )

  const merged = useMemo<MergedBar[]>(() => {
    if (!history.data) return []
    const byDate = new Map(indicators.data?.map((row) => [row.Date, row]) ?? [])
    return history.data.map((bar) => ({ ...bar, ...(byDate.get(bar.Date) ?? {}) }))
  }, [history.data, indicators.data])

  return {
    data: merged,
    loading: history.loading || indicators.loading,
    error: history.error ?? indicators.error,
    refetch: () => {
      history.refetch()
      indicators.refetch()
    },
  }
}
