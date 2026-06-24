const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail ?? `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),

  stockInfo: (ticker = 'NVDA') =>
    request<import('../types/api').StockInfo>(`/api/stock/info?ticker=${ticker}`),

  stockHistory: (ticker = 'NVDA', years = 10) =>
    request<import('../types/api').OhlcvBar[]>(
      `/api/stock/history?ticker=${ticker}&years=${years}`,
    ),

  stockIndicators: (ticker = 'NVDA', years = 10) =>
    request<import('../types/api').IndicatorBar[]>(
      `/api/stock/indicators?ticker=${ticker}&years=${years}`,
    ),

  modelComparison: () =>
    request<import('../types/api').ModelComparisonResponse>('/api/models/comparison'),

  modelImportance: () => request<Record<string, number>>('/api/models/importance'),

  forecast: (refresh = false) =>
    request<import('../types/api').ForecastResponse>(`/api/forecast?refresh=${refresh}`),

  insights: () => request<import('../types/api').InsightsResponse>('/api/insights'),

  eda: () => request<import('../types/api').EdaResponse>('/api/eda'),

  startTraining: (body: import('../types/api').TrainRequest) =>
    request<import('../types/api').TrainJobResponse>('/api/train', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  trainingStatus: (jobId: string) =>
    request<import('../types/api').TrainStatusResponse>(`/api/train/status/${jobId}`),
}
